"""Exp 019: a student that plays from the screen alone, taught by the RL teacher (DAgger, "learning by cheating").

The teacher (Exp 018 seed 1) reads the beatmap; the student sees only the last 4 frames lazer would show (render.py),
plus where its own cursor is and how old the frames are. Every step of every rollout is labelled with what the
teacher would do there, whoever was driving.

    python student.py --rounds 4 --maps 300 --updates 20000      # in WSL, on the GPU
    python student.py --smoke                                    # a few minutes, checks the whole loop
"""
import argparse
import glob
import json
import os
import time
from multiprocessing import get_context

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from beatmap import parse_osu
from env import OsuEnv, k_from_obs
from render import H, PX, W, X0, Y0, Renderer

HZ = 120
DT = 1000 / HZ
N_FRAMES = 4
MAX_LAT = 30.0  # ms
TEACHER = "results/experiments/018-room/seed1/model.zip"
SEL_SEED = 20_000  # the HF test sample of eval_hf.py / select_ckpt.py


# ---------------------------------------------------------------- what the student sees

def observe(renderer, ts, cursors, i, lat, lag=0, radius=2.2, rng=None):
    """(N_FRAMES, H, W) float frames ending `lat` ms before step i, with the cursor dot where the cursor was then
    (`lag` steps later still, like a cursor sprite drawn a frame behind), and the aux input."""
    out = np.empty((N_FRAMES, H, W), np.float32)
    for j in range(N_FRAMES):
        t = ts[i] - lat - j * DT
        img = renderer.frame(t).astype(np.float32) / 255
        k = max(0, int(np.searchsorted(ts, t - lag * DT, "right")) - 1)
        c = ((cursors[k] - (X0, Y0)) * PX * 16).astype(np.int32)
        cv2.circle(img, (int(c[0]), int(c[1])), int(radius * 16), 1.0, -1, cv2.LINE_AA, 4)
        out[j] = img
    if rng is not None:
        out = augment(out, rng)
    aux = np.array([cursors[i][0] / 512, cursors[i][1] / 384, lat / MAX_LAT], np.float32)
    return out, aux


def augment(x, rng):
    x = x * rng.uniform(0.75, 1.25)
    if rng.random() < 0.3:
        s = rng.uniform(0.3, 0.8)
        x = np.stack([cv2.GaussianBlur(f, (0, 0), s) for f in x])
    if rng.random() < 0.5:  # glows and other things the renderer does not draw
        yy, xx = np.mgrid[:H, :W]
        for _ in range(rng.integers(1, 4)):
            cy, cx, s, a = rng.uniform(0, H), rng.uniform(0, W), rng.uniform(2, 6), rng.uniform(0.05, 0.35)
            x = x + a * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))[None]
    x = x + rng.normal(0, rng.uniform(0.0, 0.03), x.shape)
    return np.clip(x, 0, 1).astype(np.float32)


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(N_FRAMES, 32, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),  # (128, 8, 10): keeps where things are
        )
        self.head = nn.Sequential(nn.Linear(128 * 8 * 10 + 3, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 4))

    def forward(self, frames, aux):
        z = self.conv(frames).flatten(1)
        out = self.head(torch.cat([z, aux], 1))
        return torch.tanh(out[:, :2]), out[:, 2:]  # move in [-1, 1], key logits


def to_action(move, keys):
    return np.concatenate([move, np.where(keys > 0, 1.0, -1.0)]).astype(np.float32)


# ---------------------------------------------------------------- rollouts (worker processes, CPU)

_teacher = None


def teacher():
    global _teacher
    if _teacher is None:
        from stable_baselines3 import PPO
        torch.set_num_threads(1)
        _teacher = PPO.load(TEACHER, device="cpu")
    return _teacher


def rollout(job):
    """job = (map path, student state_dict or None, share of steps the student drives, latency ms, seed).
    -> per step: time, cursor, the teacher's action; and the lazer accuracy of the play."""
    path, state, drive, lat, seed = job
    rng = np.random.default_rng(seed)
    t_model = teacher()
    bm = parse_osu(path)
    env = OsuEnv(bm, hz=HZ, k=k_from_obs(t_model.observation_space.shape[0]), classic=False)
    obs, _ = env.reset(seed=seed)
    student = None
    if state is not None and drive > 0:
        student = Student()
        student.load_state_dict(state)
        student.eval()
        renderer = Renderer(bm)
    n_max = int((bm.ends.max() + 5000) / DT) + 10
    ts, cursors, actions = np.zeros(n_max), np.zeros((n_max, 2), np.float32), np.zeros((n_max, 4), np.float32)
    n, done = 0, False
    while not done:
        a_t = t_model.predict(obs, deterministic=True)[0]
        ts[n], cursors[n], actions[n] = env.t, env.cursor, a_t
        a = a_t
        if student is not None and rng.random() < drive:
            f, aux = observe(renderer, ts[:n + 1], cursors[:n + 1], n, lat)
            with torch.no_grad():
                m, k = student(torch.from_numpy(f)[None], torch.from_numpy(aux)[None])
            a = to_action(m[0].numpy(), k[0].numpy())
        n += 1
        obs, _, done, _, _ = env.step(a)
    return dict(map=path, t=ts[:n], cursor=cursors[:n], action=actions[:n], lazer_acc=env.lazer_accuracy())


def run_jobs(jobs, procs):
    with get_context("fork").Pool(procs, initializer=torch.set_num_threads, initargs=(1,)) as pool:
        return pool.map(rollout, jobs, chunksize=1)


# ---------------------------------------------------------------- training (main process, GPU)

class Samples(torch.utils.data.IterableDataset):
    def __init__(self, plays, seed):
        self.plays, self.seed = plays, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, info.id if info else 0])
        renderers = {}
        while True:
            p = self.plays[rng.integers(len(self.plays))]
            if p["map"] not in renderers:
                if len(renderers) > 64:
                    renderers.clear()
                renderers[p["map"]] = Renderer(parse_osu(p["map"]))
            for _ in range(32):  # several samples per map: parsing is the slow part
                i = int(rng.integers(N_FRAMES, len(p["t"])))
                f, aux = observe(renderers[p["map"]], p["t"], p["cursor"], i, lat=rng.uniform(0, MAX_LAT),
                                 lag=int(rng.integers(0, 3)), radius=rng.uniform(1.5, 3.0), rng=rng)
                a = p["action"][i]
                yield f, aux, np.clip(a[:2], -1, 1), (a[2:] > 0).astype(np.float32)


def train(model, plays, updates, device, seed, workers):
    loader = torch.utils.data.DataLoader(Samples(plays, seed), batch_size=256, num_workers=workers, persistent_workers=False)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    t0, it = time.time(), iter(loader)
    for step in range(updates):
        f, aux, move, keys = (x.to(device) for x in next(it))
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):  # 2x faster on the 4060
            m, k = model(f.to(memory_format=torch.channels_last), aux)
        loss = F.mse_loss(m.float(), move) + F.binary_cross_entropy_with_logits(k.float(), keys)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"  update {step}/{updates} loss {loss.item():.4f} ({(time.time() - t0) / max(step, 1):.3f} s/update)", flush=True)
    model.eval()


def evaluate(model, maps, lat, procs, seed=0):
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    plays = run_jobs([(m, state, 1.0, lat, seed + n) for n, m in enumerate(maps)], procs)
    return float(np.mean([p["lazer_acc"] for p in plays])), plays


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--maps", type=int, default=300, help="training maps rolled out per round")
    p.add_argument("--updates", type=int, default=20000, help="gradient updates per round")
    p.add_argument("--procs", type=int, default=12)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/experiments/019-student")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.rounds, args.maps, args.updates = 2, 4, 200
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    pool = sorted(glob.glob("maps/hf/train/*.osu"))
    val = sorted(glob.glob("maps/hf/val/*.osu"))
    val = [str(f) for f in np.random.default_rng(1).choice(val, 3 if args.smoke else 20, replace=False)]
    drive = [0.0, 0.5, 0.75, 1.0]
    model, plays, best = Student().to(device).to(memory_format=torch.channels_last), [], (-1.0, None)
    os.makedirs(args.out, exist_ok=True)
    log = []
    for r in range(args.rounds):
        maps = [str(f) for f in rng.choice(pool, args.maps, replace=False)]
        state = None if r == 0 else {k: v.cpu() for k, v in model.state_dict().items()}
        t0 = time.time()
        new = run_jobs([(m, state, drive[min(r, 3)], float(rng.uniform(0, MAX_LAT)), int(rng.integers(1 << 30))) for m in maps], args.procs)
        plays += new
        print(f"round {r}: {len(new)} rollouts ({drive[min(r, 3)]:.0%} student) in {time.time() - t0:.0f} s, "
              f"{sum(len(x['t']) for x in plays)} labelled steps", flush=True)
        train(model, plays, args.updates, device, args.seed * 100 + r, args.workers)
        acc, _ = evaluate(model, val, 0.0, args.procs)
        print(f"round {r}: validation lazer acc {acc:.2%}", flush=True)
        log.append(dict(round=r, val_lazer_acc=acc, steps=sum(len(x["t"]) for x in plays)))
        torch.save(model.state_dict(), os.path.join(args.out, f"round{r}.pt"))
        if acc > best[0]:
            best = (acc, r)
    model.load_state_dict(torch.load(os.path.join(args.out, f"round{best[1]}.pt")))
    real = sorted(glob.glob("maps/test/*.osu"))
    hf = sorted(glob.glob("maps/hf/test/*.osu"))
    hf = [str(f) for f in np.random.default_rng(SEL_SEED).choice(hf, min(60, len(hf)), replace=False)]
    if args.smoke:
        real, hf = real[:2], hf[:2]
    summary = dict(best_round=best[1], rounds=log)
    for name, maps, lat in [("real", real, 0.0), ("hf", hf, 0.0), ("real_lat20", real, 20.0)]:
        acc, ps = evaluate(model, maps, lat, args.procs)
        summary[name] = dict(lazer_acc=acc, per_map={os.path.basename(x["map"]): x["lazer_acc"] for x in ps})
        print(f"{name}: lazer acc {acc:.2%}", flush=True)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
