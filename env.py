import math

import gymnasium as gym
import numpy as np

from beatmap import FOLLOW_AREA, PLAYFIELD, SLIDER, SPINNER, TAIL_LENIENCY, random_beatmap

MAX_CURSOR_SPEED = 4.8  # osu px per ms
K = 3  # upcoming objects in observation
MISS_WINDOW = 400  # lazer OsuHitWindows.MISS_WINDOW: presses further than this from an object are ignored
REWARD = {300: 1.0, 100: 0.4, 50: 0.1, 0: -1.0}
NESTED_REWARD = {True: 0.1, False: -0.3}  # slider head / tick / repeat / tail
SPIN_REWARD = 0.5  # spread over the spins required to clear a spinner
CLICK_PENALTY = -0.05
SHAPING = 1.0
CURSOR_MARGIN = 100  # the real cursor can leave the playfield, and slider paths sometimes do
LOOKAHEAD_MS = (50, 150)  # extra slider ball positions in observation (besides next step)
CURSOR_MAX = PLAYFIELD + CURSOR_MARGIN


class OsuEnv(gym.Env):
    def __init__(self, beatmap=None, segment=40, hz=60):
        self.fixed = beatmap
        self.segment = segment
        self.dt = 1000 / hz
        self.max_speed = MAX_CURSOR_SPEED * self.dt  # osu px per step
        n_obs = 6 + 6 * K + 3 + 2 * (1 + len(LOOKAHEAD_MS)) + 5
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (4,), np.float32)  # dx, dy, key1, key2 (>0 = held)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        bm = self.fixed or random_beatmap(self.np_random, self.segment)
        self.bm, self.objs, self.kinds, self.ends = bm, bm.objects, bm.kinds, bm.ends
        self.r, self.preempt, self.w = bm.radius, bm.preempt, bm.windows
        self.ox, self.oy, self.ot = (bm.objects[:, k].tolist() for k in range(3))  # plain floats for the per-step loops
        self.kl = bm.kinds.tolist()
        n = len(self.objs)
        self.clock = max(0.0, self.objs[0, 2] - self.preempt)
        self.t = self.prev_t = float(round(self.clock))  # replay frames store whole ms, so the game sees these times
        self.started = np.zeros(n, bool)  # circle / slider head judged, spinner begun
        self.done = np.zeros(n, bool)  # fully judged (lazer AllJudged)
        self.next = 0  # earliest object not started
        self.first_open = 0  # earliest object not fully judged
        self.active = []  # sliders in progress
        self.pre_tracking = {}  # slider index -> tracking state before its head is judged
        self.spinning = []  # spinners in progress
        self.cursor = self.np_random.uniform(0, PLAYFIELD)
        self.prev_cursor = self.last_update_cursor = self.cursor.copy()
        self.keys = self.prev_keys = (False, False)
        self.judgements = []  # (object index, time, 300/100/50/0) per circle / whole slider / spinner
        self.nested_log = []  # (time, kind, hit) per slider head / tick / repeat / tail and spinner tick
        self.extra_frames = []  # (time, x, y, keys) in-between updates the replay must contain, see _update_sliders
        self.combo = self.max_combo = 0
        return self._obs(), {}

    @property
    def held(self):
        return any(self.keys)

    def step(self, action):
        a = np.minimum(np.maximum(action, -1.0), 1.0)  # np.clip, minus its per-call overhead
        prev_d = self._target_dist()
        self.prev_cursor = self.cursor
        self.cursor = np.minimum(np.maximum(self.cursor + a[:2] * self.max_speed, -CURSOR_MARGIN), CURSOR_MAX)
        keys = (bool(a[2] > 0), bool(a[3] > 0))
        presses = [k for k in (0, 1) if keys[k] and not self.keys[k]]
        self.prev_keys, self.keys = self.keys, keys
        self.clock += self.dt
        self.prev_t, self.t = self.t, float(round(self.clock))
        reward = SHAPING * (prev_d - self._target_dist()) / 512
        # A client updates gameplay many times between replay frames, with the previous frame's keys and an
        # interpolated cursor (lazer FramedReplayInputHandler). Updates that can change a judgement are pinned
        # into the replay as extra frames on whole ms, so every client (and this simulator) agrees on them.
        frame = self.t, self.cursor, self.keys
        for tau, pin in self._between_times():
            self.t, self.cursor, self.keys = frame
            cursor = self._lerp_cursor(tau)
            self.t, self.cursor, self.keys = tau, cursor, self.prev_keys
            if pin:
                self.extra_frames.append((tau, *cursor, self.keys))
            reward += self._update()
        self.t, self.cursor, self.keys = frame
        self._start_spinners()
        for k in presses:  # input is handled before the frame's update
            reward += self._press(k)
        reward += self._update()
        return self._obs(), reward, self.first_open >= len(self.objs), False, {}

    def _between_times(self):
        """[(time, pinned)] in (prev frame, this frame) where some judgement could change, rounded up to whole ms.
        An event in this frame's last ms can't be pinned; it is left to this frame's update, which is what a
        client sees unless it happens to update inside that sub-ms gap."""
        times = set()
        for s in self.active:
            times |= {time for time, _ in s["pending"]} | {s["sl"].end + TAIL_LENIENCY, s["sl"].end}
        for s in self.spinning:
            times |= {s["sp"].start, s["sp"].end, self.t - 1}  # t - 1: spin under the previous frame's keys
        i = self.next
        while i < len(self.ot) and self.ot[i] + self.w[2] < self.t:
            if not self.started[i]:
                late = math.floor(self.ot[i] + self.w[2]) + 1  # first ms past the 50 window: late miss
                times |= {late, late + 1}  # + 1: a timed-out slider head's nested objects are judged on the next update
            i += 1
        return [(c, True) for c in sorted({float(np.ceil(t)) for t in times if self.prev_t < t and np.ceil(t) < self.t})]

    def _update(self):
        """One gameplay update at self.t: spinner starts, late misses, sliders, spinners."""
        self._start_spinners()
        self._track_unstarted()
        reward = 0.0
        i = self.next
        while i < len(self.ot) and self.t > self.ot[i] + self.w[2]:
            if not self.started[i] and self.kl[i] != SPINNER:
                reward += self._hit(i, None, None)
            i += 1
        reward += self._update_sliders() + self._update_spinners()
        while self.next < len(self.objs) and self.started[self.next]:
            self.next += 1
        while self.first_open < len(self.objs) and self.done[self.first_open]:
            self.first_open += 1
        self.last_update_cursor = self.cursor
        return reward

    def _press(self, key):
        """lazer LegacyHitPolicy (classic note lock): the earliest object under the cursor takes the click."""
        for j in range(self.next, len(self.objs)):
            x, y, t = self.ox[j], self.oy[j], self.ot[j]
            if t - self.preempt > self.t:
                break
            if self.started[j] or self.kl[j] == SPINNER or math.hypot(self.cursor[0] - x, self.cursor[1] - y) > self.r:
                continue
            if j > 0 and self.bm.stack[j - 1] > 0 and not self.done[j - 1]:
                continue  # ClickAction.Ignore: the click passes to the next object under the cursor
            err = self.t - t
            if abs(err) >= MISS_WINDOW:
                return CLICK_PENALTY  # shake (LegacyHitPolicy: hittable only strictly inside the miss window)
            if any(not self.done[i] and self.ends[i] + 3 < t for i in range(self.first_open, j)):
                return CLICK_PENALTY  # shake: an earlier object must be judged first
            return self._hit(j, err, key)
        return CLICK_PENALTY

    def _hit(self, i, err, key):
        """Judge a circle / slider head; err=None means it timed out."""
        self.started[i] = True
        e = np.inf if err is None else abs(err)
        w300, w100, w50 = self.w
        if self.kinds[i] == SLIDER:  # classic mod: slider head is plain hit/miss within the 50 window
            return self._start_slider(i, e <= w50, key if e <= w50 else None, timed_out=err is None)
        self.done[i] = True
        return self._judge(i, 300 if e <= w300 else 100 if e <= w100 else 50 if e <= w50 else 0)

    def _judge(self, i, score):
        self.judgements.append((i, self.t, score))
        self._combo(score > 0)
        return REWARD[score]

    def _combo(self, hit):
        self.combo = self.combo + 1 if hit else 0
        self.max_combo = max(self.max_combo, self.combo)

    def _nested(self, s, time, kind, hit):
        self.nested_log.append((time, kind, hit))
        s["hits"] += hit
        if kind != "tail":  # classic tail is a small tick: no combo
            self._combo(hit)
        return NESTED_REWARD[hit]

    def _in_follow(self, sl, expanded):
        return np.hypot(*(self.cursor - sl.ball(self.t))) <= self.r * (FOLLOW_AREA if expanded else 1)

    def _start_slider(self, i, hit, key, timed_out):
        sl = self.bm.sliders[i]
        s = dict(idx=i, sl=sl, pending=list(sl.nested), hits=0, total=1 + len(sl.nested), tracking=self.pre_tracking.pop(i, False), hit_key=key,
                 accept_any=None, since=self.t, timed_out=timed_out)
        reward = self._nested(s, self.objs[i, 2], "head", hit)
        # lazer SliderInputManager.PostProcessHeadJudgement: a late head hit settles nested objects already passed
        if hit and self._in_follow(sl, True):
            passed = [n for n in s["pending"] if n[0] <= self.t]
            ok = all(np.hypot(*(self.cursor - sl.ball(n[0]))) <= self.r * FOLLOW_AREA for n in passed)
            for n in passed:
                reward += self._nested(s, *n, ok)
            s["pending"] = s["pending"][len(passed):]
            s["tracking"] = ok or self._in_follow(sl, False)
        self.active.append(s)
        return reward

    def _lerp_cursor(self, tau):
        return self.prev_cursor + (self.cursor - self.prev_cursor) * (tau - self.prev_t) / (self.t - self.prev_t)

    def _track(self, s, tau, keys, cursor):
        """lazer SliderInputManager.updateTracking at time tau."""
        sl, hk = s["sl"], s["hit_key"]
        # only the key that hit the head tracks, until the other key has been seen released
        if hk is None:
            s["accept_any"] = None
        elif s["accept_any"] is None and not self.prev_keys[1 - hk]:
            s["accept_any"] = tau
        key_ok = keys[hk] if hk is not None and (s["accept_any"] is None or tau <= s["accept_any"]) else any(keys)
        radius = self.r * (FOLLOW_AREA if s["tracking"] else 1)
        s["tracking"] = key_ok and np.hypot(*(cursor - sl.ball(tau))) <= radius

    def _track_unstarted(self):
        """lazer SliderInputManager tracks from the moment a slider is alive, before its head is judged (any key counts)."""
        i = self.next
        while i < len(self.ot) and self.ot[i] - self.preempt <= self.t:
            if self.kl[i] == SLIDER and not self.started[i]:
                radius = self.r * (FOLLOW_AREA if self.pre_tracking.get(i) else 1)
                self.pre_tracking[i] = any(self.keys) and np.hypot(*(self.cursor - self.bm.sliders[i].ball(self.t))) <= radius
            i += 1

    def _update_sliders(self):
        reward = 0.0
        for s in list(self.active):
            sl = s["sl"]
            self._track(s, self.t, self.keys, self.cursor)
            # lazer judges ticks / repeats / tail before the head within an update (DrawableSlider child order),
            # so after a head timed out in this update they wait for the next one
            while s["pending"] and not (s["hit_key"] is None and s["since"] == self.t and s["timed_out"]):
                time, kind = s["pending"][0]
                off = self.t - time
                if kind == "tail":
                    if off < TAIL_LENIENCY or (off < 0 and not s["tracking"]):
                        break
                elif off < 0:
                    break
                s["pending"].pop(0)
                reward += self._nested(s, time, kind, s["tracking"])
            if not s["pending"] and self.t >= sl.end:  # lazer DrawableSlider judges itself only once its end time passes
                frac = s["hits"] / s["total"]
                reward += self._judge(s["idx"], 300 if frac == 1 else 0 if frac == 0 else 100 if frac >= 0.5 else 50)
                self.done[s["idx"]] = True
                self.active.remove(s)
        return reward

    def _start_spinners(self):
        i = self.next
        while i < len(self.ot) and self.ot[i] <= self.t:
            if self.kl[i] == SPINNER and not self.started[i]:
                self.started[i] = True
                self.spinning.append(dict(idx=i, sp=self.bm.spinners[i], acc=0.0, acc_at_last=0.0, completed=0, cur_max=0.0, ticks=0))
            i += 1

    def _update_spinners(self):
        reward = 0.0
        for s in list(self.spinning):
            sp, centre = s["sp"], self.objs[s["idx"], :2]
            before = self._spin_progress(s)
            # lazer SpinnerRotationTracker: rotation since the previous update counts while a key is held
            if sp.start <= self.t < sp.end and self.held:
                a0, a1 = (-np.degrees(np.arctan2(*(c - centre))) for c in (self.last_update_cursor, self.cursor))
                self._report_spin(s, (a1 - a0 + 180) % 360 - 180)
            # lazer DrawableSpinner.updateBonusScore: one tick per completed full spin
            spins = int(self._total_rotation(s) / 360)
            while s["ticks"] < spins:
                if s["ticks"] < sp.n_ticks:
                    self.nested_log.append((sp.start, "spin", True))
                s["ticks"] += 1
            reward += SPIN_REWARD * (self._spin_progress(s) - before)
            if self.t >= sp.end:
                for _ in range(max(0, sp.n_ticks - s["ticks"])):
                    self.nested_log.append((sp.start, "spin", False))
                p = self._spin_progress(s)
                reward += self._judge(s["idx"], 300 if p >= 1 else 100 if p > 0.9 else 50 if p > 0.75 else 0)
                self.done[s["idx"]] = True
                self.spinning.remove(s)
        return reward

    @staticmethod
    def _report_spin(s, delta):
        """lazer SpinnerSpinHistory.ReportDelta: direction changes don't add up."""
        if delta == 0:
            return
        s["acc"] += delta
        cur = s["acc"] - s["acc_at_last"]
        s["cur_max"] = max(s["cur_max"], abs(cur))
        while s["cur_max"] >= 360:
            s["completed"] += 1
            s["acc_at_last"] += np.sign(cur) * 360
            s["cur_max"] = abs(s["acc"] - s["acc_at_last"])

    @staticmethod
    def _total_rotation(s):
        return 360 * s["completed"] + s["cur_max"]

    def _spin_progress(self, s):
        req = s["sp"].spins_required
        return 1.0 if req == 0 else min(1.0, self._total_rotation(s) / 360 / req)

    def _visible(self, i):
        return i < len(self.ot) and self.ot[i] - self.t <= self.preempt

    def _target_dist(self):
        c = self.cursor
        if self.active:
            b = self.active[0]["sl"].ball(self.t)
            return math.hypot(b[0] - c[0], b[1] - c[1])
        if self.spinning or not self._visible(self.next) or self.kl[self.next] == SPINNER:
            return 0.0
        return math.hypot(self.ox[self.next] - c[0], self.oy[self.next] - c[1])

    def _obs(self):
        o = [*(self.cursor / PLAYFIELD), *map(float, self.keys), self.r / 64, self.w[2] / 200]
        upcoming = []
        for i in range(self.next, len(self.objs)):
            if len(upcoming) == K:
                break
            if not self.started[i]:
                upcoming.append(i)
        for i in upcoming:
            if self._visible(i):
                x, y, t, kind = self.ox[i], self.oy[i], self.ot[i], self.kl[i]
                o += [(x - self.cursor[0]) / 256, (y - self.cursor[1]) / 256, (t - self.t) / 200, 1.0,
                      float(kind == SLIDER), float(kind == SPINNER)]
            else:
                o += [0.0] * 6
        o += [0.0] * 6 * (K - len(upcoming))
        if self.active:
            s = self.active[0]
            sl = s["sl"]
            o += [1.0, float(s["tracking"]), (sl.end - self.t) / 200]
            for la in (self.dt, *LOOKAHEAD_MS):
                o += list((sl.ball(self.t + la) - self.cursor) / 256)
        else:
            o += [0.0] * (3 + 2 * (1 + len(LOOKAHEAD_MS)))
        if self.spinning:
            s = self.spinning[0]
            o += [1.0, self._spin_progress(s), (s["sp"].end - self.t) / 1000, *((self.cursor - self.objs[s["idx"], :2]) / 256)]
        else:
            o += [0.0] * 5
        return np.array(o, np.float32)
