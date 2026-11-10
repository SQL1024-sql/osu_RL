"""How the vision student does on real lazer frames (a live.record recording) against the frames it was trained on,
both judged by the teacher's moves in the same play (the teacher drove the cursor through live.screen_play).

    python -m live.vision_eval runs/live_rec/renatus_hard-*.npz maps/test/renatus_hard.osu \
        --ckpt results/experiments/020-student/seed0/round6.pt

1. Time: the recording's game times can be a few ms off what the screen showed (capture and display delay), so the
   shift that best lines the real frames up with rendered ones is searched first and applied.
2. For every recorded step, the student gets the 4 frames it would have seen (real: the latest recorded frame at or
   before each frame time; rendered: student.build, exactly as in training), with the cursor dot drawn where the
   teacher's cursor was, and the same aux input.
3. Its move and keys are compared with the teacher's: move error and direction agreement while the teacher moves,
   click and hold agreement. Writes a picture of a few real / rendered frame pairs next to the recording.
"""
import argparse
import glob
import hashlib
import os

import cv2
import numpy as np
import torch

import student as S
from env import MAX_CURSOR_SPEED
from gpu_render import draw_dots
from live.prepare import prepare
from render import H, W

LAT = 10.0  # ms the student is told its newest frame is old (training draws 0-30)


def teacher_track(osu, level):
    """The teacher's play as the cursor and keys at any game time: (times ms, positions (n, 2), keys bits)."""
    fr = prepare(osu, level=level)["frames"]
    t = np.array([f[0] for f in fr], float)
    pos = np.array([(f[1], f[2]) for f in fr], float)
    keys = np.array([f[3] for f in fr], int)
    return t, pos, keys


def at(t_track, values, t):
    """values of the latest track entry at or before each time t."""
    k = np.clip(np.searchsorted(t_track, t, side="right") - 1, 0, len(t_track) - 1)
    return values[k]


def real_frames(rec_t, rec_frames, times):
    """(n, 4) frame times -> (n, 4, H, W) float: the latest recorded frame at or before each time (black before any)."""
    k = np.searchsorted(rec_t, times, side="right") - 1
    out = rec_frames[np.clip(k, 0, len(rec_t) - 1)].astype(np.float32) / 255
    out[k < 0] = 0
    return out


def corr(a, b):
    a = a.reshape(len(a), -1) - a.reshape(len(a), -1).mean(1, keepdims=True)
    b = b.reshape(len(b), -1) - b.reshape(len(b), -1).mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rec", help="a live.record .npz (a glob takes the newest match)")
    p.add_argument("map")
    p.add_argument("--ckpt", default="results/experiments/020-student/seed0/round6.pt")
    p.add_argument("--level", default="expert", help="the level the teacher played at (live.prepare.LEVELS)")
    p.add_argument("--shift-ms", type=float, help="skip the search and use this time shift")
    p.add_argument("--from-ms", type=float, default=-1e9, help="only judge steps from this game time on (a held-out part)")
    p.add_argument("--click-th", type=float, default=S.CLICK_TH, help="click logit a peak must reach to count as a click")
    args = p.parse_args()
    S.CLICK_TH = args.click_th
    path = sorted(glob.glob(args.rec), key=os.path.getmtime)[-1]
    rec = np.load(path)
    rec_frames, rec_t = rec["frames"], rec["game_ms"].astype(float)
    osu = open(args.map, "rb").read()
    assert str(rec["md5"]) == hashlib.md5(osu).hexdigest(), "the recording is of another map"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    maps = S.Maps(dev)
    mid = maps.ids([args.map])[0]
    tt, tpos, tkeys = teacher_track(osu, args.level)
    labelled = os.path.exists(path[:-4] + ".labels.npz")
    if labelled:
        from live.dagger_label import load_labels
        lt, lcur, ly, ldown = load_labels(path)
        tt, tpos = lt, lcur.astype(float)  # the cursor the labels were asked at
        print("labels: live.dagger_label's (the student's teacher)")
    print(f"{path}: {len(rec_t)} frames over {rec_t[0] / 1000:.1f}-{rec_t[-1] / 1000:.1f} s; teacher {len(tt)} moves")

    def rendered(times, dots):
        """(n, 4) frame times, (n, 4, 3) dots -> (n, 4, H, W) float, as in training (no augmentation)."""
        n = len(times)
        ids = torch.full((n * S.N_FRAMES,), int(mid), dtype=torch.long, device=dev)
        tm = torch.tensor(times.reshape(-1), dtype=torch.float32, device=dev)
        d = torch.tensor(dots.reshape(-1, 3), dtype=torch.float32, device=dev)
        img, _ = S.build(maps, ids, tm, d, torch.zeros(n, S.AUX, device=dev))
        return img.float().cpu().numpy()

    # 1. time shift: real frame at recorded time t shows the game at t + shift
    sel = np.linspace(0, len(rec_t) - 1, min(400, len(rec_t))).astype(int)
    sel = sel[rec_frames[sel].mean((1, 2)) > 3]  # skip black frames (nothing on screen)
    if args.shift_ms is None:
        shifts = np.arange(-80, 81, 4.0)
        scores = []
        for s in shifts:
            times = np.repeat((rec_t[sel] + s)[:, None], S.N_FRAMES, 1)
            dots = np.zeros((len(sel), S.N_FRAMES, 3))
            dots[..., :2] = -1e4  # no cursor: the real frame's cursor is lazer's own
            syn = rendered(times, dots)[:, 0]
            blur = lambda x: np.stack([cv2.GaussianBlur(f, (0, 0), 1.5) for f in x])
            scores.append(float(np.mean(corr(blur(rec_frames[sel].astype(np.float32) / 255), blur(syn)))))
        best = int(np.argmax(scores))
        shift = float(shifts[best])
        print(f"time shift {shift:+.0f} ms (real vs rendered frame correlation {scores[best]:.3f}; "
              f"at 0 ms {scores[len(shifts) // 2]:.3f})")
    else:
        shift = args.shift_ms

    # 2. student inputs at every step of the teacher's play that the recording covers
    dt = S.DT
    lo, hi = max(rec_t[0] + shift, tt[0], args.from_ms) + 40, min(rec_t[-1] + shift, tt[-1]) - 20
    steps = np.arange(lo, hi, dt)
    ftimes = steps[:, None] - LAT - np.array(S.FRAME_STEPS)[None, :] * dt  # (n, 4) game times shown
    cur = at(tt, tpos, steps)
    dots = np.concatenate([at(tt, tpos, ftimes.reshape(-1)).reshape(-1, S.N_FRAMES, 2),
                           np.full((len(steps), S.N_FRAMES, 1), 2.2)], 2)
    aux = np.stack([cur[:, 0] / 512, cur[:, 1] / 384, np.full(len(steps), LAT / S.MAX_LAT)], 1)
    # teacher labels: the move to the next step, a key going down within 2 steps, any key held
    nxt = at(tt, tpos, steps + dt)
    move = np.clip((nxt - cur) / (MAX_CURSOR_SPEED * dt), -1, 1)
    k0, k1, k2 = at(tt, tkeys, steps), at(tt, tkeys, steps + dt), at(tt, tkeys, steps + 2 * dt)
    click = ((k1 & ~k0) | (k2 & ~k1)) != 0
    hold = k0 != 0
    down = (k1 & ~k0) != 0
    if labelled:  # live.dagger_label's labels: the student's own teacher, asked in each state of the play
        ls = np.clip(np.searchsorted(lt, steps, side="right") - 1, 0, len(lt) - 1)
        move, click, hold = ly[ls, :2], ly[ls, 2] > 0.5, ly[ls, 3] > 0.5
        d = np.zeros(len(lt), bool)
        d[ldown] = True
        down = d[ls] & np.r_[True, ls[1:] != ls[:-1]]  # one event per labelled step

    model = S.Student().to(dev).to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev))
    model.eval()

    # the times the recorded frames actually showed: the screen refreshes slower than the student's 120 Hz steps,
    # so neighbouring frames are often the same picture; "rendered at rec times" separates that from how frames look
    shown = rec_t + shift
    k = np.clip(np.searchsorted(shown, ftimes, side="right") - 1, 0, len(shown) - 1)
    snapped = shown[k]

    def run(kind):
        outs_m, outs_k = [], []
        for b in range(0, len(steps), 512):
            sl = slice(b, b + 512)
            if kind == "real":
                img = real_frames(rec_t + shift, rec_frames, ftimes[sl])
                img = torch.tensor(img, device=dev)
                d = torch.tensor(dots[sl].reshape(-1, 3), dtype=torch.float32, device=dev)
                img = draw_dots(img.view(-1, H, W), d).view(-1, S.N_FRAMES, H, W)
            else:
                img = torch.tensor(rendered(ftimes[sl] if kind == "rendered" else snapped[sl], dots[sl]), device=dev)
            with torch.no_grad(), S.autocast(dev):
                m, k = model(img, torch.tensor(aux[sl], dtype=torch.float32, device=dev))
            outs_m.append(m.float().cpu().numpy())
            outs_k.append(k.float().cpu().numpy())
        return np.concatenate(outs_m), np.concatenate(outs_k)

    moving = np.hypot(*move.T) > 0.05
    # clicks as events: the teacher's key-downs, the student's firing rule (a peak of the click logit above the
    # threshold, fired the step after it); matched within TOL steps either way (a step or two early or late is normal)
    TOL = 3
    t_ev = np.nonzero(down)[0]
    print(f"{len(steps)} steps ({moving.mean():.0%} with the teacher moving, {len(t_ev)} teacher clicks, {hold.mean():.0%} held)")
    # clicks against the map itself, whatever the teacher would do: objects clicked within their 50 window, and clicks
    # with no object due within it (stray)
    from beatmap import SPINNER, parse_osu
    bm = parse_osu(args.map)
    w50 = bm.windows[2]
    obj_t = np.array([t for (_, _, t), kd in zip(bm.objects, bm.kinds) if kd != SPINNER and steps[0] <= t <= steps[-1]])
    print(f"{'frames':>22} | {'move err':>8} | {'direction':>9} | {'clicks found':>12} | {'extra clicks':>12} | {'hold acc':>8}"
          f" | {'objects clicked':>15} | {'stray clicks':>12}")
    for kind, name in (("rendered", "rendered (120 Hz)"), ("snapped", "rendered at rec times"), ("real", "real lazer")):
        m, k = run(kind)
        err = np.hypot(*(m - move).T).mean()
        cos = (m * move).sum(1)[moving] / (np.hypot(*m[moving].T) * np.hypot(*move[moving].T) + 1e-9)
        c = k[:, 0]
        s_ev = np.nonzero((c[:-2] < c[1:-1]) & (c[1:-1] > c[2:]) & (c[1:-1] > S.CLICK_TH))[0] + 2
        near = lambda a, b: np.array([np.any(np.abs(b - x) <= TOL) for x in a], bool) if len(a) else np.zeros(0, bool)
        found, extra = near(t_ev, s_ev), ~near(s_ev, t_ev)
        ph = k[:, 1] > 0
        ev_t = steps[np.clip(s_ev, 0, len(steps) - 1)]
        clicked = np.array([np.any(np.abs(ev_t - t) <= w50) for t in obj_t]) if len(obj_t) else np.zeros(0)
        stray = int(sum(not np.any(np.abs(obj_t - e) <= w50) for e in ev_t))
        print(f"{name:>22} | {err:8.3f} | {np.mean(cos > 0.7):9.1%} | {found.mean():12.1%} | "
              f"{extra.sum():5d} of {len(s_ev):<4d} | {(ph == hold).mean():8.1%} | {clicked.mean():8.1%} of {len(obj_t):<4d}"
              f" | {stray:5d} of {len(ev_t):<4d}")

    # 3. a few real / rendered pairs, newest frame of each, cursor dot drawn
    pick = np.linspace(0, len(steps) - 1, 6).astype(int)[1:-1]
    real = real_frames(rec_t + shift, rec_frames, ftimes[pick])[:, 0]
    syn = rendered(ftimes[pick], dots[pick])[:, 0]
    rows = [np.hstack([r, np.full((H, 2), 1.0), s]) for r, s in zip(real, syn)]
    img = (np.vstack(rows) * 255).astype(np.uint8)
    out = path[:-4] + "-real_vs_rendered.png"
    cv2.imwrite(out, cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3), interpolation=cv2.INTER_NEAREST))
    print(f"real (left) vs rendered (right) -> {out}")


if __name__ == "__main__":
    main()
