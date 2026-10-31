import math
from bisect import bisect_right
from dataclasses import dataclass, field, replace

import numpy as np

PLAYFIELD = np.array([512.0, 384.0])
CENTRE = PLAYFIELD / 2
FOLLOW_AREA = 2.4  # lazer DrawableSliderBall.FOLLOW_AREA
TAIL_LENIENCY = -36  # lazer SliderEventGenerator.TAIL_LENIENCY
STACK_DISTANCE = 3  # lazer OsuBeatmapProcessor.STACK_DISTANCE
CIRCLE, SLIDER, SPINNER = 0, 1, 2


@dataclass
class Slider:
    start: float
    span_duration: float
    spans: int
    path: np.ndarray  # (M, 2) absolute polyline, already cut to the expected length
    cum: np.ndarray  # (M,) cumulative length along path
    nested: list  # [(time, kind)] kind: "tick" / "repeat" / "tail", time-ordered

    def __post_init__(self):
        # plain-float copies: ball() runs every step and numpy's per-call overhead dominates on tiny inputs
        self.end = self.start + self.spans * self.span_duration
        self._cum, self._xs, self._ys = self.cum.tolist(), self.path[:, 0].tolist(), self.path[:, 1].tolist()

    def ball(self, t):
        """Slider ball position at time t (lazer CurvePositionAt with time progress)."""
        if self.end <= self.start:
            return self.path[0].copy()
        p = min(max((t - self.start) / (self.end - self.start), 0.0), 1.0) * self.spans
        span = min(int(p), self.spans - 1)
        f = p - span
        cum = self._cum
        d = (1 - f if span % 2 else f) * cum[-1]
        # same arithmetic as np.interp
        j = bisect_right(cum, d) - 1
        if j < 0:
            return np.array([self._xs[0], self._ys[0]])
        if j >= len(cum) - 1:
            return np.array([self._xs[-1], self._ys[-1]])
        u = d - cum[j]
        w = cum[j + 1] - cum[j]
        return np.array([(self._xs[j + 1] - self._xs[j]) / w * u + self._xs[j], (self._ys[j + 1] - self._ys[j]) / w * u + self._ys[j]])


@dataclass
class Spinner:
    start: float
    end: float
    spins_required: int
    n_ticks: int  # required + 2 + max bonus spins; one tick per completed full spin


@dataclass
class Beatmap:
    cs: float
    ar: float
    od: float
    objects: np.ndarray  # (N, 3): stacked x, y, start time (spinner: its centre)
    kinds: np.ndarray  # CIRCLE / SLIDER / SPINNER
    ends: np.ndarray  # end time
    stack: np.ndarray  # stack height
    sliders: dict = field(default_factory=dict)  # object index -> Slider
    spinners: dict = field(default_factory=dict)  # object index -> Spinner
    title: str = "random"

    @property
    def radius(self):
        return radius(self.cs)

    @property
    def preempt(self):
        return preempt(self.ar)

    @property
    def windows(self):  # 300 / 100 / 50 hit windows (± ms), rounded like lazer's OsuHitWindows
        od = self.od
        return tuple(float(np.floor(w) - 0.5) for w in (80 - 6 * od, 140 - 8 * od, 200 - 10 * od))


def radius(cs):
    return (54.4 - 4.48 * cs) * 1.00041  # lazer broken_gamefield_rounding_allowance


def preempt(ar):
    return 1200 + 600 * (5 - ar) / 5 if ar < 5 else 1200 - 750 * (ar - 5) / 5


def difficulty_range(d, lo, mid, hi):
    return mid + (hi - mid) * (d - 5) / 5 if d > 5 else mid - (mid - lo) * (5 - d) / 5 if d < 5 else mid


# ---------- slider paths (lazer SliderPath + osu.Framework PathApproximator) ----------

def _bezier(pts):
    n = max(2, int(np.sum(np.hypot(*np.diff(pts, axis=0).T)) / 2))
    t = np.linspace(0, 1, n)[:, None, None]
    p = np.broadcast_to(pts, (n, *pts.shape)).copy()
    while p.shape[1] > 1:  # de Casteljau
        p = (1 - t) * p[:, :-1] + t * p[:, 1:]
    return p[:, 0]


def _arc(a, b, c):
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) < 1e-3:
        return None
    sa, sb, sc = a @ a, b @ b, c @ c
    centre = np.array([sa * (b[1] - c[1]) + sb * (c[1] - a[1]) + sc * (a[1] - b[1]),
                       sa * (c[0] - b[0]) + sb * (a[0] - c[0]) + sc * (b[0] - a[0])]) / d
    r = np.hypot(*(a - centre))
    t0, t1, t2 = (np.arctan2(p[1] - centre[1], p[0] - centre[0]) for p in (a, b, c))
    ccw = (t2 - t0) % (2 * np.pi)
    # go counter-clockwise a -> c if b lies on that side, otherwise clockwise
    rng = ccw if (t1 - t0) % (2 * np.pi) <= ccw else ccw - 2 * np.pi
    n = 2 if 2 * r <= 0.1 else max(2, int(np.ceil(abs(rng) / (2 * np.arccos(1 - 0.1 / r)))))
    if n >= 1000:  # lazer falls back to bezier for huge arcs
        return None
    ts = t0 + np.linspace(0, 1, n) * rng
    return centre + r * np.column_stack([np.cos(ts), np.sin(ts)])


def _catmull(pts):
    """PathApproximator.CatmullToPiecewiseLinear + lazer OptimiseCatmull. Returns (points, extra length)."""
    t = np.repeat(np.arange(51) / 50, 2)[1:-1, None]  # (c / 50, (c + 1) / 50) for c in 0..49
    out = []
    for i in range(len(pts)):
        v1 = pts[i - 1] if i > 0 else pts[i]
        v2 = pts[i]
        v3 = pts[i + 1] if i < len(pts) - 1 else 2 * v2 - v1
        v4 = pts[i + 2] if i < len(pts) - 2 else 2 * v3 - v2
        out.append(0.5 * (2 * v2 + (-v1 + v3) * t + (2 * v1 - 5 * v2 + 4 * v3 - v4) * t * t + (-v1 + 3 * v2 - 3 * v3 + v4) * t ** 3))
    out = np.concatenate(out)
    step = np.hypot(*np.diff(out, axis=0).T).tolist()  # step[i - 1] = |out[i] - out[i - 1]|
    pl = out.tolist()
    opt, extra, last, removed = [], 0.0, None, 0.0
    for i, p in enumerate(pl):
        if last is None:
            opt.append(p)
            last = p
            continue
        dist = math.hypot(p[0] - last[0], p[1] - last[1])
        removed += step[i - 1]
        if dist > 6 or (i + 1) % 100 == 0 or i == len(pl) - 1:
            opt.append(p)
            extra += removed - dist
            last, removed = None, 0.0
    return np.array(opt), extra


def build_path(kind, pts, expected=None):
    """Legacy control points (absolute, head first) -> polyline cut/extended to `expected` length."""
    pts = np.asarray(pts, float)
    extra = 0.0
    if kind == "P" and len(pts) == 3:
        a, b, c = pts
        if abs((b[1] - a[1]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[1] - a[1])) < 1e-3:
            kind = "L"  # legacy decoder: collinear perfect curve becomes linear
        else:
            arc = _arc(a, b, c)
            kind = "B" if arc is None else "P"
    elif kind == "P":
        kind = "B"  # legacy decoder: perfect curve needs exactly 3 points
    if kind == "P":
        segs = [arc]
    elif kind == "B":
        segs, start = [], 0  # repeated consecutive points split bezier segments (red anchors)
        for i in range(1, len(pts)):
            if np.array_equal(pts[i], pts[i - 1]) and i < len(pts) - 1:
                segs.append(pts[start:i])
                start = i
        segs.append(pts[start:])
        segs = [_bezier(s) if len(s) > 2 else s for s in segs]
    elif kind == "C":
        path, extra = _catmull(pts)
        segs = [path]
    else:
        segs = [pts]
    path = segs[0]
    for s in segs[1:]:
        path = np.concatenate([path, s[1:] if np.array_equal(path[-1], s[0]) else s])
    cum = extra + np.concatenate([[0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
    cum[0] = 0
    if expected is not None and expected > 0 and len(path) >= 2 and cum[-1] != expected:
        if np.array_equal(path[-1], path[-2]) and expected > cum[-1]:
            return path, cum
        if cum[-1] > expected:
            k = int(np.searchsorted(cum, expected))  # first index with cum >= expected
            path, cum = path[:k + 1].copy(), cum[:k + 1].copy()
        k = len(path) - 1
        seg = path[k] - path[k - 1]
        n = np.hypot(*seg)
        path[k] = path[k - 1] + (seg / n if n > 0 else 0) * (expected - cum[k - 1])
        cum[k] = expected
    return path, cum


def make_slider(start, kind, pts, length, spans, beat_length, bpm_mult, slider_mult, tick_rate, tick_mult=1.0):
    path, cum = build_path(kind, pts, length)
    velocity = 100 * slider_mult / (beat_length * bpm_mult)
    tick_dist = velocity * beat_length / tick_rate * tick_mult
    length = cum[-1]
    span_dur = length / velocity
    nested = []
    tick_d = np.clip(tick_dist, 0, length) if length > 0 else 0
    for span in range(spans):
        s0 = start + span * span_dur
        ticks, d = [], tick_d
        while tick_d > 0 and d <= length and d < length - velocity * 10:
            prog = d / length
            ticks.append((s0 + (1 - prog if span % 2 else prog) * span_dur, "tick"))
            d += tick_d
        nested += sorted(ticks)
        if span < spans - 1:
            nested.append((s0 + span_dur, "repeat"))
    nested.append((start + spans * span_dur, "tail"))
    return Slider(start, span_dur, spans, path, cum, nested)


# ---------- stacking (lazer OsuBeatmapProcessor.ApplyStacking) ----------

def stack_heights(kinds, starts, ends, pos, end_pos, threshold, version):
    n = len(kinds)
    h = [0] * n
    d = lambda a, b: np.hypot(a[0] - b[0], a[1] - b[1])
    if version < 6:
        for i in range(n):
            if h[i] != 0 and kinds[i] != SLIDER:
                continue
            start_t, slider_stack = ends[i], 0
            for j in range(i + 1, n):
                if starts[j] - threshold > start_t:
                    break
                if d(pos[j], pos[i]) < STACK_DISTANCE:
                    h[i] += 1
                    start_t = starts[j]
                elif d(pos[j], end_pos[i]) < STACK_DISTANCE:
                    slider_stack += 1
                    h[j] -= slider_stack
                    start_t = starts[j]
        return h
    extended_start = 0
    for i in range(n - 1, 0, -1):
        if h[i] != 0 or kinds[i] == SPINNER:
            continue
        oi, m = i, i
        if kinds[i] == CIRCLE:
            while (m := m - 1) >= 0:
                if kinds[m] == SPINNER:
                    continue
                if int(starts[oi]) - int(ends[m]) > threshold:
                    break
                if m < extended_start:
                    h[m] = 0
                    extended_start = m
                if kinds[m] == SLIDER and d(end_pos[m], pos[oi]) < STACK_DISTANCE:
                    offset = h[oi] - h[m] + 1
                    for j in range(m + 1, i + 1):
                        if d(end_pos[m], pos[j]) < STACK_DISTANCE:
                            h[j] -= offset
                    break
                if d(pos[m], pos[oi]) < STACK_DISTANCE:
                    h[m] = h[oi] + 1
                    oi = m
        elif kinds[i] == SLIDER:
            while (m := m - 1) >= 0:
                if kinds[m] == SPINNER:
                    continue
                if starts[oi] - starts[m] > threshold:
                    break
                if d(end_pos[m], pos[oi]) < STACK_DISTANCE:
                    h[m] = h[oi] + 1
                    oi = m
    return h


# ---------- .osu parsing ----------

def parse_osu(path):
    with open(path, encoding="utf-8-sig") as f:
        return parse_lines(f.read().splitlines())


def parse_lines(lines):
    section, kv, timing, raw = None, {}, [], []
    version = int(lines[0].strip().rsplit("v", 1)[-1]) if lines and "file format" in lines[0] else 14
    offset = 24 if version < 5 else 0  # lazer LegacyBeatmapDecoder.EARLY_VERSION_TIMING_OFFSET
    for line in lines:
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("["):
            section = line[1:-1]
        elif section in ("General", "Difficulty", "Metadata"):
            k, _, v = line.partition(":")
            kv[k.strip()] = v.strip()
        elif section == "TimingPoints":
            s = line.split(",")
            uninherited = len(s) < 7 or s[6].strip() == "1"
            timing.append((float(s[0]) + offset, float(s[1]), uninherited))
        elif section == "HitObjects":
            raw.append(line.split(","))

    timing.sort(key=lambda p: p[0])  # stable: same-time points keep file order
    sm, tick_rate = float(kv.get("SliderMultiplier", 1.4)), float(kv.get("SliderTickRate", 1))
    od = float(kv["OverallDifficulty"])
    cs, ar = float(kv["CircleSize"]), float(kv.get("ApproachRate", od))

    def timing_at(t):
        beat, mult = next((b for _, b, u in timing if u), 500.0), 1.0
        for pt, b, u in timing:
            if pt > t:
                break
            if u:
                beat, mult = b, 1.0
            elif b < 0:
                mult = np.clip(-b, 10, 1000) / 100  # lazer GetPrecisionAdjustedBeatLength
        return beat, mult

    min_rps, max_rps = difficulty_range(od, 90, 150, 225) / 60, difficulty_range(od, 250, 380, 430) / 60
    kinds, pos, starts, ends, end_pos, sliders, spinners = [], [], [], [], [], {}, {}
    for s in raw:
        x, y, typ = int(float(s[0])), int(float(s[1])), int(s[3])
        t = float(s[2]) + offset
        i = len(kinds)
        if typ & 8:
            end = max(t, float(s[5]) + offset)
            secs = (end - t) / 1000
            req = int(min_rps * secs + 0.0001)
            bonus = max(0, int(max_rps * secs + 0.0001) - req - 2)
            spinners[i] = Spinner(t, end, req, req + 2 + bonus)
            kinds.append(SPINNER)
            ends.append(end)
            end_pos.append((x, y))
        elif typ & 2:
            kind, *pts = s[5].split("|")
            pts = [(x, y)] + [tuple(int(float(v)) for v in p.split(":")) for p in pts]
            length = float(s[7]) if len(s) > 7 and float(s[7]) > 0 else None
            beat, mult = timing_at(t)
            # lazer OsuBeatmapConverter: before v8, tick spacing ignores slider velocity
            sl = make_slider(t, kind, pts, length, int(s[6]), beat, mult, sm, tick_rate, mult if version < 8 else 1.0)
            sliders[i] = sl
            kinds.append(SLIDER)
            ends.append(sl.end)
            end_pos.append(tuple(sl.ball(sl.end)))
        else:
            kinds.append(CIRCLE)
            ends.append(t)
            end_pos.append((x, y))
        pos.append((x, y))
        starts.append(t)

    r = radius(cs)
    h = stack_heights(kinds, starts, ends, pos, end_pos, int(preempt(ar)) * float(kv.get("StackLeniency", 0.7)), version)
    objs = np.array([(x, y, t) for (x, y), t in zip(pos, starts)], float).reshape(-1, 3)
    for i, sh in enumerate(h):
        if sh and kinds[i] != SPINNER:
            off = sh * (r / 64) * -6.4  # lazer OsuHitObject.StackOffset
            objs[i, :2] += off
            if i in sliders:
                sliders[i] = replace(sliders[i], path=sliders[i].path + off)
    return Beatmap(cs, ar, od, objs, np.array(kinds), np.array(ends), np.array(h), sliders, spinners,
                   f"{kv.get('Title', '?')} [{kv.get('Version', '?')}]")


# ---------- random training / fuzzing maps, written as real .osu text ----------

def random_osu(rng, n=40):
    beat = 60000 / rng.uniform(120, 240)
    cs, ar, od = round(rng.uniform(3, 5.5), 1), round(rng.uniform(7, 10), 1), round(rng.uniform(5, 9), 1)
    sm, tick_rate = float(rng.choice([1.4, 1.8, 2.2])), int(rng.choice([1, 2]))
    timing, objs = [f"0,{beat!r},4,1,0,100,1,0"], []
    pos, t = np.round(rng.uniform(0, PLAYFIELD)), 1000

    def jump(p, gap):
        ang = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(20, min(450, 2.5 * gap))
        return np.round(np.clip(p + dist * np.array([np.cos(ang), np.sin(ang)]), 0, PLAYFIELD))

    while len(objs) < n:
        x, y = int(pos[0]), int(pos[1])
        r = rng.random()
        end_pos, end_t = pos, t
        if r < 0.04:  # spinner
            end_t = t + int(rng.uniform(800, 3000))
            objs.append(f"256,192,{t},8,0,{end_t},0:0:0:0:")
            end_pos = CENTRE
        elif r < 0.10:  # stack of circles, maybe under the previous slider's end
            k, gap = int(rng.integers(2, 6)), beat / 4
            for j in range(k):
                objs.append(f"{x},{y},{int(t + j * gap)},1,0,0:0:0:0:")
            end_t = int(t + (k - 1) * gap)
        elif r < 0.16:  # stream
            k, gap = int(rng.integers(3, 9)), beat / 4
            ang = rng.uniform(0, 2 * np.pi)
            for j in range(k):
                p = np.round(np.clip(pos + j * 25 * np.array([np.cos(ang), np.sin(ang)]), 0, PLAYFIELD))
                objs.append(f"{int(p[0])},{int(p[1])},{int(t + j * gap)},1,0,0:0:0:0:")
                end_pos = p
            end_t = int(t + (k - 1) * gap)
        elif r < 0.56:  # slider
            seg = rng.uniform(40, 260)
            ang = rng.uniform(0, 2 * np.pi)
            d = np.array([np.cos(ang), np.sin(ang)])
            tip = np.clip(pos + d * seg, 0, PLAYFIELD)
            bend = np.array([-d[1], d[0]]) * seg * rng.uniform(-0.5, 0.5)
            kind = str(rng.choice(["L", "P", "B", "B", "C"]))
            if kind == "L":
                ctrl = [tip]
            elif kind == "P":
                ctrl = [(pos + tip) / 2 + bend, tip]
            else:
                ctrl = [pos + (tip - pos) / 3 + bend, pos + 2 * (tip - pos) / 3 - bend, tip]
                if kind == "B" and rng.random() < 0.3:  # red anchor: duplicated point splits segments
                    ctrl = [ctrl[0], ctrl[1], ctrl[1], tip]
            ctrl = [np.round(np.clip(c, 0, PLAYFIELD)) for c in ctrl]
            pts = [pos] + ctrl
            natural = build_path(kind, pts)[1][-1]
            if natural >= 20:
                length = round(natural * rng.uniform(0.85, 1.05), 2)
                sv = round(rng.uniform(0.6, 1.6), 2)
                spans = int(rng.choice([1, 1, 1, 2, 3]))
                timing.append(f"{t},{-100 / sv!r},4,1,0,100,0,0")
                ctrl_str = "|".join(f"{int(c[0])}:{int(c[1])}" for c in ctrl)
                objs.append(f"{x},{y},{t},2,0,{kind}|{ctrl_str},{spans},{length}")
                sl = make_slider(t, kind, pts, length, spans, beat, np.clip(100 / sv, 10, 1000) / 100, sm, tick_rate)
                end_pos, end_t = sl.ball(sl.end), sl.end
            else:
                objs.append(f"{x},{y},{t},1,0,0:0:0:0:")
        else:
            objs.append(f"{x},{y},{t},1,0,0:0:0:0:")
        gap = float(rng.choice([beat, beat / 2], p=[0.4, 0.6]))
        if r < 0.04:
            gap = beat
        pos = end_pos if rng.random() < 0.05 else jump(end_pos, gap)  # sometimes stack on the slider end
        t = int(end_t + gap)

    return ["osu file format v14", "", "[General]", "StackLeniency: 0.7", "Mode: 0", "", "[Metadata]", "Title:random", "Version:random",
            "", "[Difficulty]", "HPDrainRate:5", f"CircleSize:{cs}", f"OverallDifficulty:{od}", f"ApproachRate:{ar}",
            f"SliderMultiplier:{sm}", f"SliderTickRate:{tick_rate}", "", "[TimingPoints]", *timing, "", "[HitObjects]", *objs]


def random_beatmap(rng, n=40):
    return parse_lines(random_osu(rng, n))
