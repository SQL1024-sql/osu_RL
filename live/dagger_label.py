"""Teacher labels for a real recording, as student.rollouts makes them in the simulator: the play's own cursor and keys
are replayed step by step through the simulator (OsuEnv, room rules, 120 Hz), and at every step the teacher the
student learnt from (student.TEACHER) says what it would do in that state. For a play the student drove
(live.vision_play --record), those are the corrections DAgger needs: what to do where its own mistakes took it.

    python -m live.dagger_label runs/live_play/*-rec.npz runs/live_rec/1_train_*.npz

The trajectory is the student's logged cursor and keys when the recording names its play log (vision_play), else the
prepared teacher play the cursor followed (screen_play). Writes <recording>.labels.npz next to each: t (ms), cursor,
action (the teacher's, (n, 4)), keys (held in the game before each step), like a student.rollouts play, plus the
simulator's lazer accuracy of the replayed play.
"""
import argparse
import glob
import hashlib
import os

import numpy as np
import torch

import vision.student as S
from state.env import MAX_CURSOR_SPEED, OsuEnv, k_from_obs
from live.prepare import prepare


MAP_GLOBS = ["maps/test/*.osu", "runs/maps/live_set/*.osu", "runs/maps/pool/*/*.osu",
             "runs/maps/tier*/*/*.osu"]  # live.improve's difficulty tiers (live.make_pool --out runs/maps/tierN)


def map_index(globs):
    """MD5 -> .osu path for every map the globs find."""
    out = {}
    for g in globs:
        for f in glob.glob(g):
            out[hashlib.md5(open(f, "rb").read()).hexdigest()] = f
    return out


def trajectory(rec, map_path):
    """(game ms, cursor (n, 2), keys (n, 2) bool) the play followed."""
    if "play" in rec.files:  # a vision_play recording: the student's own log
        d = np.load(str(rec["play"]))
        a, cols = d["log"], list(d["cols"])
        b = a[:, cols.index("buttons")].astype(int)
        ok = np.isfinite(a[:, cols.index("game")])
        pos, b = a[ok][:, [cols.index("x"), cols.index("y")]], b[ok]
        if "press" in d.files and str(d["press"]) == "first":
            # the tablet pressed where the cursor was before the step's move (live.tablet.Tablet press="first")
            down = np.nonzero(b[1:] & ~b[:-1])[0] + 1
            pos = pos.copy()
            pos[down] = pos[down - 1]
        return a[ok, cols.index("game")], pos, np.stack([b & 1 != 0, b & 2 != 0], 1)
    fr = prepare(open(map_path, "rb").read(), level="expert")["frames"]  # what screen_play sent
    # the replay's extra frames (in-between updates, the previous step's keys) can share a step's ms: keep the
    # step's own frame, which comes first
    t = np.array([f[0] for f in fr], float)
    _, first = np.unique(t, return_index=True)
    fr = [fr[i] for i in first]
    bits = np.array([f[3] for f in fr])
    return (t[first], np.array([(f[1], f[2]) for f in fr], float),
            np.stack([bits & 5 != 0, bits & 10 != 0], 1))  # replay.KEY_BITS: K1 = M1|K1, K2 = M2|K2


def load_labels(rec_path, clicks="teacher"):
    """A recording's labels as student.Tracks makes them: (t ms, cursor (n, 2), labels (n, 4): move x, move y, click,
    hold, click events (step indices)). The move is the teacher's. Clicks "teacher": a teacher key going down that is
    up in the game, on this step or the next, and hold: any teacher key held. Clicks "map" (student.map_keys, Exp 025):
    from the map and where the cursor was: each object clicked at the step nearest its start with the cursor on it,
    held through sliders and spinners. In states the teacher did not drive itself, its keys are a poor guide: it
    wants a key down whenever the game has it up (Exp 019-023)."""
    L = np.load(rec_path[:-4] + ".labels.npz")
    act, held, t, cur = L["action"], L["keys"], L["t"].astype(np.float64), L["cursor"].astype(np.float32)
    if clicks == "map":
        path = str(L["map"])
        mk = S.map_keys(S.scene(path), t, cur, path)
        y = np.c_[np.clip(act[:, :2], -1, 1), mk].astype(np.float32)
        down = mk[:, 0] > 0.5
        return t, cur, y, np.nonzero(down & ~np.r_[False, down[:-1]])[0]
    down = ((act[:, 2:] > 0) & (held < 0.5)).any(1)
    click = down | np.r_[down[1:], False]
    y = np.c_[np.clip(act[:, :2], -1, 1), click, (act[:, 2:] > 0).any(1)].astype(np.float32)
    return t, cur, y, np.nonzero(down)[0]


def label(rec_path, index, teacher, k):
    rec = np.load(rec_path)
    md5 = str(rec["md5"])
    assert md5 in index, f"{rec_path}: its map ({md5}) is not among the maps"
    path = index[md5]
    gt, gc, gk = trajectory(rec, path)
    at = lambda t: int(np.clip(np.searchsorted(gt, t, side="right") - 1, 0, len(gt) - 1))
    env = OsuEnv(S.load(path), hz=S.HZ, k=k, classic=False)
    obs, _ = env.reset(seed=0)
    env.cursor = np.array(gc[at(env.t)], float)
    speed = MAX_CURSOR_SPEED * env.dt
    ts, cur, act, held = [], [], [], []
    done = False
    while not done and round(env.clock + env.dt) <= gt[-1]:
        a_t = teacher.predict(obs, deterministic=True)[0]
        ts.append(env.t)
        cur.append(env.cursor.copy())
        act.append(a_t)
        held.append(np.array(env.keys, np.float32))
        # where the play was at the next step. Not at the simulator's rounded ms: a vision_play row 0.1 ms after it
        # (rows come every 8.33 ms, at any phase) would be skipped, and with it often the step a key went down
        j = at(env.clock + env.dt)
        move = np.clip((gc[j] - env.cursor) / speed, -1, 1)
        obs, _, done, _, _ = env.step(np.array([*move, *np.where(gk[j], 1.0, -1.0)], np.float32))
    acc = env.lazer_accuracy()
    out = rec_path[:-4] + ".labels.npz"
    np.savez_compressed(out, t=np.array(ts), cursor=np.array(cur, np.float32), action=np.array(act, np.float32),
                        keys=np.array(held, np.float32), map=path, teacher=S.TEACHER, sim_lazer_acc=acc)
    drove = "student" if "play" in rec.files else "teacher"
    print(f"{os.path.basename(rec_path)}: {len(ts)} steps of {os.path.basename(path)}, {drove} driving, "
          f"replayed in the simulator: lazer acc {acc:.2%} -> {out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("recs", nargs="+", help="recordings (globs work)")
    p.add_argument("--maps", nargs="*", default=MAP_GLOBS)
    args = p.parse_args()
    index = map_index(args.maps)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = S.teacher("cpu")  # one observation at a time: the CPU is faster
    k = k_from_obs(teacher.observation_space.shape[0])
    for g in args.recs:
        for f in sorted(glob.glob(g)):
            if not f.endswith(".labels.npz"):
                label(f, index, teacher, k)


if __name__ == "__main__":
    main()
