"""Plays a map in a real osu!lazer window through the virtual tablet (live/tablet.py), with nothing read from the game:
the moves come from the simulator ahead of time (live.prepare), and the game clock is found by watching the screen,
from the moment each hit circle starts fading in (its start time minus the preempt).

    python -m live.screen_play maps/test/renatus_hard.osu --level expert

Start it, then start the same map in lazer (background dim 100%, no mods). It waits for the dark playfield.
"""
import argparse
import ctypes
import statistics
import threading
import time

import mss
import numpy as np

from beatmap import SPINNER, parse_osu
from live.prepare import LEVELS, prepare
from live.tablet import M1, M2, Tablet

user32 = ctypes.WinDLL("user32")
WATCH_BEFORE, WATCH_AFTER = 0.25, 0.15  # seconds of samples around each fade-in
DARK = 12  # mean brightness (0-255) of a gameplay playfield with a fully dimmed background


def lazer_window():
    """The visible window whose title starts with "osu!" (lazer adds the map's name while playing)."""
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def visit(hwnd, _):
        title = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(ctypes.c_void_p(hwnd), title, 256)
        if user32.IsWindowVisible(ctypes.c_void_p(hwnd)) and title.value.startswith("osu!"):
            found.append(hwnd)
        return True

    user32.EnumWindows(visit, None)
    if not found:
        raise SystemExit("no lazer window (title starting with osu!)")
    return ctypes.c_void_p(found[0])


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


def fade_ins(bm):
    """[(appear_s, fade_s, x, y)] for hit circles and slider heads no other object overlaps while they fade in."""
    preempt = bm.preempt
    fade = 400 * min(1, preempt / 450)
    shapes = [np.array([[ox, oy]]) if i not in bm.sliders else np.vstack([[ox, oy], bm.sliders[i].path[::4]])
              for i, (ox, oy, _) in enumerate(bm.objects)]
    out = []
    for i, (x, y, t) in enumerate(bm.objects):
        if bm.kinds[i] == SPINNER:
            continue
        a, b = t - preempt - 50, t - preempt + fade + 100
        clear = all(bm.objects[j][2] - preempt > b or bm.ends[j] + 300 < a
                    or bm.kinds[j] != SPINNER and np.min(np.hypot(*(shapes[j] - (x, y)).T)) > 2.2 * bm.radius
                    for j in range(len(bm.objects)) if j != i)
        if clear:
            out.append(((t - preempt) / 1000, fade / 1000, x, y))
    return out


class Clock(threading.Thread):
    """Finds offset = wall clock - game clock (seconds) from the fade-ins it sees; None until the first one."""

    def __init__(self, events, field, radius, cursor):
        super().__init__(daemon=True)
        self.events, self.field, self.radius, self.cursor = events, field, radius, cursor
        self.offset, self.samples = None, []

    def grab(self, sct, box):
        return float(np.asarray(sct.grab(box))[:, :, :3].mean())

    def run(self):
        f = self.field
        with mss.mss() as sct:
            x0, y0 = f(0, 0)
            whole = {"left": int(x0), "top": int(y0), "width": int(512 * f.scale), "height": int(384 * f.scale)}
            dark_since = None
            while dark_since is None or time.perf_counter() - dark_since < 0.3:  # gameplay has started
                dark = self.grab(sct, whole) < DARK + 3
                dark_since = (dark_since or time.perf_counter()) if dark else None
                time.sleep(0.02)
            half = max(4, int(0.3 * self.radius * f.scale))
            for appear, fade, x, y in self.events:
                cx, cy = f(x, y)
                box = {"left": int(cx) - half, "top": int(cy) - half, "width": 2 * half, "height": 2 * half}
                if self.offset is not None:
                    start = self.offset + appear - WATCH_BEFORE
                    if time.perf_counter() > start:
                        continue
                    time.sleep(max(0.0, start - time.perf_counter()))
                    end = start + WATCH_BEFORE + fade + WATCH_AFTER
                elif self.grab(sct, box) > DARK:
                    continue  # already showing: this one cannot be the first fade-in we see
                else:
                    end = None
                samples = []
                while True:
                    now = time.perf_counter()
                    b = self.grab(sct, box)
                    px, py = self.cursor()
                    samples.append((now, b, np.hypot(px - cx, py - cy) < half + 40 * f.scale))
                    if end is None and b > 2 * DARK:
                        end = now + fade + 0.05  # the first fade-in, watched from the dark screen on
                    if end is not None and now > end:
                        break
                if end is not None and self.offset is None:
                    samples = [s for s in samples if s[0] > end - fade - 0.05 - WATCH_BEFORE]
                t0 = fit(samples, fade)
                if t0 is not None:
                    self.samples.append(t0 - appear)
                    self.offset = statistics.median(self.samples[-15:])


def fit(samples, fade):
    """Wall time a fade-in started: the best fit of flat, then a linear rise over `fade` seconds, then flat."""
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
    return float(t0s[k])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("map")
    p.add_argument("--level", default="expert", choices=LEVELS)
    p.add_argument("--lag-ms", type=float, default=0.0, help="send every move this much later (negative: earlier)")
    p.add_argument("--seconds", type=float, default=1e9, help="stop after this much of the map")
    args = p.parse_args()

    osu = open(args.map, "rb").read()
    play = prepare(osu, level=args.level)
    print(f"simulated in lazer: {play['submission']['accuracy']:.2%}, {len(play['frames'])} moves")
    bm = parse_osu(args.map)
    events = fade_ins(bm)
    field = Playfield()
    tablet = Tablet()
    pos = [0.0, 0.0]
    clock = Clock(events, field, bm.radius, lambda: pos)
    clock.start()
    print(f"{len(events)} clean fade-ins to time the game by; start the map in lazer now")
    while clock.offset is None:
        time.sleep(0.005)
    print(f"game clock found ({len(clock.samples)} fade-in)")
    lag = args.lag_ms / 1000
    try:
        for t, x, y, keys in play["frames"]:
            if t > args.seconds * 1000:
                break
            due = clock.offset + t / 1000 + lag
            wait = due - time.perf_counter()
            if wait < -0.05:
                continue  # already past
            if wait > 0:
                time.sleep(max(0.0, wait - 0.001))
                while time.perf_counter() < due:
                    pass
            pos[:] = field(x, y)
            tablet.send(*pos, (M1 if keys & 1 else 0) | (M2 if keys & 2 else 0))
    finally:
        tablet.close()
    s = np.array(clock.samples) * 1000
    print(f"timed by {len(s)} fade-ins, offset spread {s.std():.1f} ms (max-min {s.max() - s.min():.1f} ms)")
    print("each one minus the first (ms):", " ".join(f"{v:.0f}" for v in s - s[0]))


if __name__ == "__main__":
    main()
