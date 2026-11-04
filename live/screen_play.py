"""Plays a map in a real osu!lazer window through the virtual tablet (live/tablet.py), with nothing read from the game:
the moves come from the simulator ahead of time (live.prepare), and the game clock is found by watching the screen,
from the moment each hit circle starts fading in (its start time minus the preempt).

    python -m live.screen_play maps/test/renatus_hard.osu --level expert

Start it, then start the same map in lazer (background dim 100%, no mods). It waits for the dark playfield.
"""
import argparse
import ctypes
import json
import threading
import time

import mss
import numpy as np

from beatmap import SPINNER, parse_osu
from live.prepare import LEVELS, prepare
from live.tablet import M1, M2, LostFocus, Tablet, lazer_window

user32 = ctypes.WinDLL("user32")
WATCH_BEFORE, WATCH_AFTER = 0.25, 0.15  # seconds of samples around each fade-in
DARK = 12  # mean brightness (0-255) of a gameplay playfield with a fully dimmed background


class Playfield:
    """osu! pixels -> screen pixels, as lazer's OsuPlayfieldAdjustmentContainer lays the playfield out."""

    def __init__(self):
        hwnd = lazer_window()
        r, p = (ctypes.c_long * 4)(), (ctypes.c_long * 2)()
        user32.GetClientRect(hwnd, r)
        user32.ClientToScreen(hwnd, p)
        w, h = r[2], r[3]
        box_h = min(0.8 * h, 0.8 * w * 3 / 4)  # 80% of the window, fitted to 4:3
        self.scale = box_h * 4 / 3 / 512
        self.x0 = p[0] + (w - box_h * 4 / 3) / 2
        self.y0 = p[1] + (h - box_h) / 2 + 8 * self.scale  # AlignWithStoryboard shifts it 8 osu! pixels down

    def __call__(self, x, y):
        return self.x0 + x * self.scale, self.y0 + y * self.scale


def seg_dist(p, a, b):
    ab = b - a
    u = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
    return np.hypot(*(a + u * ab - p))


def fade_ins(bm):
    """[(appear_s, fade_s, x, y)] for hit circles and slider heads whose patch nothing else crosses while they fade
    in: no other object, approach circle or follow point line."""
    preempt, r = bm.preempt, bm.radius
    fade = 400 * min(1, preempt / 450)
    shapes = [np.array([[ox, oy]]) if i not in bm.sliders else np.vstack([[ox, oy], bm.sliders[i].path[::4]])
              for i, (ox, oy, _) in enumerate(bm.objects)]
    playable = [j for j in range(len(bm.objects)) if bm.kinds[j] != SPINNER]
    out = []
    for i in playable:
        x, y, t = bm.objects[i]
        p = np.array([x, y])
        a, b = t - preempt - 50, t - preempt + fade + 100

        def clear(j):
            tj = bm.objects[j][2]
            if tj - preempt > b or bm.ends[j] + 300 < a:
                return True  # not on screen meanwhile
            if bm.kinds[j] == SPINNER or np.min(np.hypot(*(shapes[j] - p).T)) < 2.2 * r:
                return False
            # j's approach circle shrinks from 4r to r around its head; does its ring pass over the patch?
            lo, hi = max(a, tj - preempt), min(b, tj)
            if lo < hi:
                ring = [r * (1 + 3 * (tj - s) / preempt) for s in (lo, hi)]
                if ring[1] - 0.6 * r < np.hypot(*(bm.objects[j][:2] - p)) < ring[0] + 0.6 * r:
                    return False
            return True

        follow = all(seg_dist(p, shapes[j][-1], shapes[k][0]) > r
                     for j, k in zip(playable, playable[1:])
                     if j != i and k != i and bm.objects[j][2] - preempt < b and bm.objects[k][2] > a)
        if follow and all(clear(j) for j in range(len(bm.objects)) if j != i):
            out.append(((t - preempt) / 1000, fade / 1000, x, y))
    return out


class Clock(threading.Thread):
    """Finds wall clock - game clock (seconds) from the fade-ins it sees: one robust line through all of them, so
    the noise of single fade-ins averages out instead of moving the clock around."""

    def __init__(self, events, field, radius, cursor):
        super().__init__(daemon=True)
        self.events, self.field, self.radius, self.cursor = events, field, radius, cursor
        self.fits = []  # (appear game time, wall - game, amp, rms)
        self.model = None  # (intercept, slope) of wall - game against game time

    @property
    def offset(self):
        return None if self.model is None else self.model[0]

    def at(self, game):
        icept, slope = self.model
        return icept + slope * game

    def refit(self):
        g, o = (np.array(c) for c in zip(*[(f[0], f[1]) for f in self.fits]))
        keep = np.ones(len(g), bool)
        for _ in range(3):
            if keep.sum() >= 20 and np.ptp(g[keep]) > 20:
                slope, icept = np.polyfit(g[keep], o[keep], 1)
                if abs(slope) > 1e-3:  # more than 1 ms per second is not clock drift
                    slope, icept = 0.0, float(np.median(o[keep]))
            else:
                slope, icept = 0.0, float(np.median(o[keep]))
            resid = o - (icept + slope * g)
            mad = max(0.003, 1.4826 * float(np.median(np.abs(resid[keep]))))
            keep = np.abs(resid) < 3 * mad
        self.model = (float(icept), float(slope))

    def grab(self, sct, box):
        v = float(np.asarray(sct.grab(box))[:, :, :3].mean())
        return v, time.perf_counter()  # GDI capture waits for the next composition: the image is from about now

    def run(self):
        f = self.field
        with mss.mss() as sct:
            x0, y0 = f(0, 0)
            whole = {"left": int(x0), "top": int(y0), "width": int(512 * f.scale), "height": int(384 * f.scale)}
            dark_since = None
            while dark_since is None or time.perf_counter() - dark_since < 0.3:  # gameplay has started
                dark = self.grab(sct, whole)[0] < DARK + 3
                dark_since = (dark_since or time.perf_counter()) if dark else None
                time.sleep(0.02)
            half = max(4, int(0.3 * self.radius * f.scale))
            for appear, fade, x, y in self.events:
                cx, cy = f(x, y)
                box = {"left": int(cx) - half, "top": int(cy) - half, "width": 2 * half, "height": 2 * half}
                if self.model is not None:
                    start = self.at(appear) + appear - WATCH_BEFORE
                    if time.perf_counter() > start:
                        continue
                    time.sleep(max(0.0, start - time.perf_counter()))
                    end = start + WATCH_BEFORE + fade + WATCH_AFTER
                elif self.grab(sct, box)[0] > DARK:
                    continue  # already showing: this one cannot be the first fade-in we see
                else:
                    end = None
                samples = []
                while True:
                    b, now = self.grab(sct, box)
                    px, py = self.cursor()
                    samples.append((now, b, np.hypot(px - cx, py - cy) < half + 40 * f.scale))
                    if end is None and b > 2 * DARK:
                        end = now + fade + 0.05  # the first fade-in, watched from the dark screen on
                    if end is not None and now > end:
                        break
                if self.model is None:
                    samples = [s for s in samples if s[0] > end - fade - 0.05 - WATCH_BEFORE]
                got = fit(samples, fade)
                if got is not None:
                    self.fits.append((appear, got[0] - appear, *got[1:]))
                    self.refit()


def fit(samples, fade):
    """(wall time a fade-in started, its brightness rise, fit rms): the best fit of flat, then a linear rise over
    `fade` seconds, then flat; None when the patch does not look like a clean fade-in."""
    t, b, near = (np.array(c) for c in zip(*samples))
    t, b = t[~near], b[~near]
    if len(t) < 20:
        return None
    t0s = np.arange(t[0] - fade, t[-1], 0.0005)
    ramp = np.clip((t[None, :] - t0s[:, None]) / fade, 0, 1)  # (candidates, samples)
    n, sr, srr = len(t), ramp.sum(1), (ramp * ramp).sum(1)
    sb, srb = b.sum(), ramp @ b
    det = n * srr - sr * sr
    det[det < 1e-9] = np.inf
    amp = (n * srb - sr * sb) / det  # least squares of b = base + amp * ramp, per candidate
    base = (sb - amp * sr) / n
    err = ((base[:, None] + amp[:, None] * ramp - b) ** 2).mean(1)
    k = int(np.argmin(err))
    if amp[k] < 15 or np.sqrt(err[k]) > 0.15 * amp[k]:  # no clean fade-in (something else moved through the patch)
        return None
    return float(t0s[k]), float(amp[k]), float(np.sqrt(err[k]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("map")
    p.add_argument("--level", default="expert", choices=LEVELS)
    p.add_argument("--lag-ms", type=float, default=None,
                   help="send every move this much later (negative: earlier); default: minus the measured loop latency")
    p.add_argument("--seconds", type=float, default=1e9, help="stop after this much of the map")
    p.add_argument("--log", help="write every fade-in fit here (json)")
    args = p.parse_args()

    osu = open(args.map, "rb").read()
    play = prepare(osu, level=args.level)
    print(f"simulated in lazer: {play['submission']['accuracy']:.2%}, {len(play['frames'])} moves")
    if args.lag_ms is None:
        from live.latency import measure  # live.latency imports this module
        print("measuring the loop latency (lazer must show its cursor, e.g. song select) ...")
        # the fade-ins are seen late by the same display and capture path, and the moves arrive late by the input path
        args.lag_ms = -float(np.median(measure(20)))
    bm = parse_osu(args.map)
    events = fade_ins(bm)
    field = Playfield()
    tablet = Tablet()
    pos = [0.0, 0.0]
    clock = Clock(events, field, bm.radius, lambda: pos)
    clock.start()
    print(f"lag {args.lag_ms:.1f} ms; {len(events)} clean fade-ins to time the game by; start the map in lazer now")
    while clock.model is None:
        time.sleep(0.005)
    print("game clock found")
    lag = args.lag_ms / 1000
    try:
        for t, x, y, keys in play["frames"]:
            if t > args.seconds * 1000:
                break
            due = clock.at(t / 1000) + t / 1000 + lag
            wait = due - time.perf_counter()
            if wait < -0.05:
                continue  # already past
            if wait > 0:
                time.sleep(max(0.0, wait - 0.001))
                while time.perf_counter() < due:
                    pass
            pos[:] = field(x, y)
            tablet.send(*pos, (M1 if keys & 1 else 0) | (M2 if keys & 2 else 0))
    except LostFocus as e:
        print(e)
    finally:
        tablet.close()
    g, o = (np.array(c) for c in zip(*[(f[0], f[1]) for f in clock.fits]))
    resid = (o - np.array([clock.at(x) for x in g])) * 1000
    icept, slope = clock.model
    print(f"timed by {len(g)} fade-ins: drift {slope * 1e6:.0f} ppm, residuals median |{np.median(np.abs(resid)):.1f}| ms, "
          f"{(np.abs(resid) > 15).sum()} beyond 15 ms")
    if args.log:
        with open(args.log, "w", encoding="utf-8") as fh:
            json.dump({"lag_ms": args.lag_ms, "model": clock.model,
                       "fits": [dict(zip(("appear", "offset", "amp", "rms"), f)) for f in clock.fits]}, fh)


if __name__ == "__main__":
    main()
