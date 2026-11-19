"""Makes the state model's play look like a person's: it aims at the circle's centre and hits on the beat every time,
which on stream looks like Autoplay. This bends its prepared moves (live.prepare frames) instead of retraining it:

- timing: every object is hit a little early or late (a normal spread, in ms: unstable rate = 10 x sigma), more so
  where objects come quickly;
- aim: every object is aimed at a little off its centre (a normal spread, as a part of the circle's radius), more
  so after a fast jump, so small circles and wide jumps are missed now and then;
- slips: now and then (more often after a fast jump) an object is aimed at well outside its circle: a miss;
- jitter: the hand is never quite still: a fast tremor (around 10 Hz, a pixel or so) and a slow wobble (under 2 Hz,
  a few pixels) on top of the whole path, faded out around spinners.

The bend is smooth: between two objects the shift slides from one object's to the next, so the cursor never jumps
and the order of the moves never changes. Spinners are left as they are.

    python -m live.humanize maps/test/renatus_hard.osu --feel good     # scores each feel in headless lazer
"""
import argparse

import numpy as np

from beatmap import SPINNER, parse_osu

# feel -> (timing sigma in ms, aim sigma as a part of the circle's radius, chance of a slip per object, tremor and
# wobble in osu! px, RMS per axis); None: the model's own play
FEELS = {
    "full": None,
    "pro": (12.0, 0.15, 0.0015, 0.5, 1.2),
    "good": (18.0, 0.25, 0.004, 0.8, 2.0),
    "casual": (28.0, 0.36, 0.010, 1.2, 3.0),
}
SLIP = 1.6  # a slip lands this many radii from the centre
DEFAULT_FEEL = "good"
DENSE_MS = 150.0  # objects closer than this get a wider timing spread (up to x1.6)
FAST = 1.0  # osu! px per ms: jumps faster than this get a wider aim spread (up to x2)
TREMOR_HZ, TREMOR_WIDTH = 10.0, 2.0  # the fast tremor's band (Hz, a Gaussian around it)
WOBBLE_HZ = 1.2  # the slow wobble's band: a Gaussian from 0 Hz with this width
SPIN_FADE = 150.0  # ms over which the jitter fades out before a spinner and back in after it


def offsets(bm, feel, rng):
    """(anchor times ms, timing shift ms, aim shift (n, 2) osu! px) per object, or None for "full"."""
    spread = FEELS[feel]
    if spread is None:
        return None
    sigma_t, sigma_a, slip = spread[:3]
    n = len(bm.objects)
    times, dt, aim = [], [], []
    prev_t, prev_xy, prev_shift = None, None, 0.0
    for i in range(n):
        x, y, t = bm.objects[i]
        if bm.kinds[i] == SPINNER:
            # held still from its start to its end, so the spin is not bent
            times += [t, bm.ends[i]]
            dt += [0.0, 0.0]
            aim += [(0.0, 0.0), (0.0, 0.0)]
            prev_t, prev_xy, prev_shift = bm.ends[i], None, 0.0
            continue
        gap = 1000.0 if prev_t is None else max(1.0, t - prev_t)
        dense = float(np.clip(DENSE_MS / gap, 1.0, 1.6))
        shift = rng.normal(0.0, sigma_t * dense)
        if prev_t is not None:  # keeps every move in its order: the shift changes by less than the gap
            shift = float(np.clip(shift, prev_shift - 0.4 * gap, prev_shift + 0.4 * gap))
        speed = 0.0 if prev_xy is None else float(np.hypot(x - prev_xy[0], y - prev_xy[1])) / gap
        fast = float(np.clip(speed / FAST, 0.4, 2.0))
        off = rng.normal(0.0, sigma_a * bm.radius * fast, 2)
        if rng.random() < slip * fast:
            angle = rng.uniform(0, 2 * np.pi)
            off = SLIP * bm.radius * np.array([np.cos(angle), np.sin(angle)])
        times.append(t)
        dt.append(shift)
        aim.append(tuple(off))
        prev_t, prev_xy, prev_shift = t, (x, y), shift
    order = np.argsort(times, kind="stable")
    return np.asarray(times, float)[order], np.asarray(dt, float)[order], np.asarray(aim, float)[order]


def jitter(t, bm, tremor, wobble, rng, step=2.0):
    """(x, y) osu! px of hand jitter at times t (ms): band-limited noise, `tremor` px RMS around TREMOR_HZ plus
    `wobble` px RMS under WOBBLE_HZ, per axis; none during spinners."""
    t = np.asarray(t, float)
    if not len(t) or (tremor <= 0 and wobble <= 0):
        return np.zeros(len(t)), np.zeros(len(t))
    grid = np.arange(t.min() - step, t.max() + 2 * step, step)
    f = np.fft.rfftfreq(len(grid), step / 1000)
    out = []
    for _ in range(2):
        axis = np.zeros(len(grid))
        for amp, band in ((tremor, np.exp(-0.5 * ((f - TREMOR_HZ) / TREMOR_WIDTH) ** 2)),
                          (wobble, np.exp(-0.5 * (f / WOBBLE_HZ) ** 2) * (f > 0))):
            noise = np.fft.irfft(np.fft.rfft(rng.normal(size=len(grid))) * band, len(grid))
            rms = float(np.sqrt(np.mean(noise ** 2)))
            if amp > 0 and rms > 0:
                axis += noise * (amp / rms)
        out.append(np.interp(t, grid, axis))
    fade = np.ones(len(t))
    for i, kind in enumerate(bm.kinds):
        if kind == SPINNER:
            start, end = bm.objects[i][2], bm.ends[i]
            away = np.maximum(np.maximum(start - t, t - end), 0.0)
            fade = np.minimum(fade, np.clip(away / SPIN_FADE, 0.0, 1.0))
    return out[0] * fade, out[1] * fade


def humanize(frames, bm, feel=DEFAULT_FEEL, seed=None):
    """frames [(t ms, x, y, buttons)] -> the same moves bent by `feel` (FEELS), sorted by time."""
    if feel not in FEELS:
        raise ValueError(f"feel must be one of {list(FEELS)}")
    rng = np.random.default_rng(seed)
    got = offsets(bm, feel, rng)
    if got is None or not len(got[0]):
        return [tuple(f) for f in frames]
    at, dt, aim = got
    t = np.array([f[0] for f in frames], float)
    shift = np.interp(t, at, dt)
    jx, jy = jitter(t, bm, *FEELS[feel][3:5], rng)  # drawn after the offsets: those stay what they were
    ox, oy = np.interp(t, at, aim[:, 0]) + jx, np.interp(t, at, aim[:, 1]) + jy
    out = [(float(t[k] + shift[k]), float(f[1] + ox[k]), float(f[2] + oy[k]), f[3]) for k, f in enumerate(frames)]
    out.sort(key=lambda f: f[0])
    return out


def score(osu_path, frames, mods=()):
    """lazer's own judgement of these moves (headless, as live.prepare scores): its submission dict."""
    import os
    import tempfile

    from live.prepare import lazer_score
    from replay import write_osr
    with tempfile.TemporaryDirectory() as tmp:
        osr = os.path.join(tmp, "play.osr")
        write_osr(osr, osu_path, [(t, x, y, (bool(b & 1), bool(b & 2))) for t, x, y, b in frames], {}, 0)
        return lazer_score(osu_path, osr, list(mods), os.path.join(tmp, "score.json"))["submission"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("map")
    p.add_argument("--feel", nargs="*", default=list(FEELS), choices=list(FEELS))
    p.add_argument("--seeds", type=int, default=3, help="plays per feel (each a different spread)")
    a = p.parse_args()
    from live.prepare import prepare
    play = prepare(open(a.map, "rb").read(), level="expert")
    bm = parse_osu(a.map)
    print(f"{bm.title}: OD {bm.od}, CS {bm.cs}, {len(bm.objects)} objects")
    for feel in a.feel:
        accs, misses = [], []
        for seed in range(a.seeds if FEELS[feel] else 1):
            s = score(a.map, humanize(play["frames"], bm, feel, seed))
            accs.append(s["accuracy"])
            misses.append(s["statistics"].get("miss", 0))
        print(f"{feel:7s} accuracy {np.mean(accs):.2%} ({min(accs):.2%}-{max(accs):.2%}), misses {misses}", flush=True)


if __name__ == "__main__":
    main()
