"""Exp 019: a student that plays from the screen alone, taught by the RL teacher (DAgger, "learning by cheating").

The teacher (Exp 018 seed 1) reads the beatmap; the student sees only the last 4 frames lazer would show (gpu_render.py),
plus where its own cursor is and how old the frames are. Every step of every rollout is labelled with what the
teacher would do there, whoever was driving.

Everything that can runs on the GPU: frames, the cursor dot, augmentation, and the teacher's and student's inference
for all environments at once. Worker processes only step the simulator.

    python student.py --rounds 4 --maps 300 --updates 20000      # Linux / WSL with CUDA
    python student.py --smoke                                    # a few minutes, checks the whole loop
"""
import argparse
import glob
import json
import os
import queue
import threading
import time
from multiprocessing import Pipe, get_context

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from beatmap import parse_osu
from env import OsuEnv, k_from_obs
from gpu_render import Scene, pack, render_fast
from render import H, PX, W, X0, Y0

HZ = 120
DT = 1000 / HZ
N_FRAMES = 4
MAX_LAT = 30.0  # ms
TEACHER = "results/experiments/018-room/seed1/model.zip"
SEL_SEED = 20_000  # the HF test sample of eval_hf.py / select_ckpt.py
SLOTS = 8  # environments per worker process

_scenes = {}


def scene(path):
    if path not in _scenes:
        _scenes[path] = Scene(parse_osu(path))
    return _scenes[path]


# ---------------------------------------------------------------- what the student sees

def prepare(items):
    """CPU half of the student's input. items: [(map, times, cursors, i, lat, lag, radius)], times/cursors per step of
    a play up to at least step i. -> packed render input, cursor dots, aux."""
    frames, dots, aux = [], [], []
    for path, ts, cur, i, lat, lag, rad in items:
        sc = scene(path)
        for j in range(N_FRAMES):
            t = ts[i] - lat - j * DT
            k = max(0, int(np.searchsorted(ts[:i + 1], t - lag * DT, "right")) - 1)
            frames.append((sc, t))
            dots.append((cur[k][0], cur[k][1], rad))
        aux.append((cur[i][0] / 512, cur[i][1] / 384, lat / MAX_LAT))
    return pack(frames, "cpu"), np.array(dots, np.float32), np.array(aux, np.float32)


def build(prepared, device, rng=None):
    """GPU half: render, draw the cursor dot, augment. -> frames (B, N_FRAMES, H, W), aux (B, 3)."""
    p, dots, aux = prepared
    p = {k: v.to(device, non_blocking=True) for k, v in p.items()}
    img = render_fast(p).view(-1, N_FRAMES, H, W)
    d = torch.from_numpy(dots).to(device).view(-1, N_FRAMES, 3, 1, 1)
    gx = X0 + (torch.arange(W, device=device) + 0.5) / PX
    gy = (Y0 + (torch.arange(H, device=device) + 0.5) / PX).view(H, 1)
    dist = torch.sqrt((gx - d[:, :, 0]) ** 2 + (gy - d[:, :, 1]) ** 2) * PX  # frame pixels
    img = torch.maximum(img, (d[:, :, 2] - dist + 0.5).clamp(0, 1))
    if rng is not None:
        img = augment(img, rng)
    return img, torch.from_numpy(aux).to(device)


def augment(x, rng):
    """Brightness, blur, glows and noise the renderer does not have, per sample."""
    b, dev = x.shape[0], x.device
    x = x * torch.from_numpy(rng.uniform(0.75, 1.25, b).astype(np.float32)).to(dev).view(b, 1, 1, 1)
    blur = np.nonzero(rng.random(b) < 0.3)[0]
    if len(blur):
        sig = torch.from_numpy(rng.uniform(0.3, 0.8, len(blur)).astype(np.float32)).to(dev)
        taps = torch.arange(-2, 3, device=dev, dtype=torch.float32)
        k = torch.exp(-taps ** 2 / (2 * sig.view(-1, 1) ** 2))
        k = (k / k.sum(1, keepdim=True)).repeat_interleave(N_FRAMES, 0)  # (n * frames, 5)
        y = x[blur].reshape(1, -1, H, W)
        y = F.conv2d(F.pad(y, (2, 2, 0, 0), mode="replicate"), k.view(-1, 1, 1, 5), groups=y.shape[1])
        y = F.conv2d(F.pad(y, (0, 0, 2, 2), mode="replicate"), k.view(-1, 1, 5, 1), groups=y.shape[1])
        x[blur] = y.view(-1, N_FRAMES, H, W)
    glow = rng.uniform(size=(b, 3, 4)) * [H, W, 4, 0.30] + [0, 0, 2, 0.05]  # centre y, x, size, strength
    glow[:, :, 3] *= (np.arange(3) < rng.integers(1, 4, (b, 1))) & (rng.random((b, 1)) < 0.5)
    g = torch.from_numpy(glow.astype(np.float32)).to(dev)
    yy = torch.arange(H, device=dev, dtype=torch.float32).view(1, 1, H, 1)
    xx = torch.arange(W, device=dev, dtype=torch.float32).view(1, 1, 1, W)
    field = (g[..., 3, None, None] * torch.exp(-((yy - g[..., 0, None, None]) ** 2 + (xx - g[..., 1, None, None]) ** 2)
                                               / (2 * g[..., 2, None, None] ** 2))).sum(1)
    x = x + field.unsqueeze(1)
    noise = torch.from_numpy(rng.uniform(0, 0.03, b).astype(np.float32)).to(dev).view(b, 1, 1, 1)
    return (x + noise * torch.randn_like(x)).clamp(0, 1)


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
        z = self.conv(frames.to(memory_format=torch.channels_last)).flatten(1)
        out = self.head(torch.cat([z, aux], 1))
        return torch.tanh(out[:, :2]), out[:, 2:]  # move in [-1, 1], key logits


def autocast(device):
    return torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda")  # 2x faster on the 4060


# ---------------------------------------------------------------- rollouts: workers step envs, the GPU decides

def env_worker(conn, jobs, k):
    """Steps up to SLOTS environments, one job (id, map, seed) at a time each. Every tick it sends their states and
    the jobs that just finished, and gets an action per slot back."""
    torch.set_num_threads(1)
    pending, slots, finished = list(jobs), {}, []

    def start(slot):
        if pending:
            jid, path, seed = pending.pop(0)
            env = OsuEnv(parse_osu(path), hz=HZ, k=k, classic=False)
            obs, _ = env.reset(seed=seed)
            slots[slot] = (jid, env, obs)

    for s in range(SLOTS):
        start(s)
    while True:
        conn.send(([(s, jid, obs, env.t, env.cursor.copy()) for s, (jid, env, obs) in slots.items()], finished))
        if not slots:
            return
        finished = []
        for s, a in conn.recv().items():
            jid, env, _ = slots.pop(s)
            obs, _, done, _, _ = env.step(a)
            if done:
                finished.append((jid, env.lazer_accuracy()))
                start(s)
            else:
                slots[s] = (jid, env, obs)


_teacher = None


def teacher(device):
    global _teacher
    if _teacher is None:
        from stable_baselines3 import PPO
        _teacher = PPO.load(TEACHER, device=device)
    return _teacher


def rollouts(jobs, student, drive, procs, device):
    """jobs: [(map, latency ms, seed)]. The student drives each step with probability `drive`; every step is labelled
    with the teacher's action. -> [dict(map, t, cursor, action, lazer_acc)]"""
    t_model = teacher(device)
    k = k_from_obs(t_model.observation_space.shape[0])
    n_max = [int((scene(m).end.max() + 5000) / DT) + 10 for m, _, _ in jobs]
    ts = [np.zeros(n) for n in n_max]
    cur = [np.zeros((n, 2), np.float32) for n in n_max]
    act = [np.zeros((n, 4), np.float32) for n in n_max]
    steps, accs = [0] * len(jobs), {}
    rngs = [np.random.default_rng(seed) for _, _, seed in jobs]
    ctx = get_context("fork" if hasattr(os, "fork") else "spawn")
    conns, workers = [], []
    for w in range(procs):
        mine = [(j, m, seed) for j, (m, _, seed) in enumerate(jobs) if j % procs == w]
        if not mine:
            continue
        a, b = Pipe()
        workers.append(ctx.Process(target=env_worker, args=(b, mine, k), daemon=True))
        workers[-1].start()
        conns.append(a)
    active = set(range(len(conns)))
    while active:
        batch = []
        for w in sorted(active):
            states, finished = conns[w].recv()
            accs.update(finished)
            if not states:
                active.discard(w)
            batch += [(w, *s) for s in states]
        if not batch:
            continue
        obs = np.stack([x[3] for x in batch])
        a_t = t_model.predict(obs, deterministic=True)[0]
        actions = a_t.copy()
        drv = []
        for n, (w, s, jid, _, t, c) in enumerate(batch):
            i = steps[jid]
            ts[jid][i], cur[jid][i], act[jid][i] = t, c, a_t[n]
            steps[jid] += 1
            if student is not None and rngs[jid].random() < drive:
                drv.append(n)
        if drv:
            items = [(jobs[batch[n][2]][0], ts[batch[n][2]], cur[batch[n][2]], steps[batch[n][2]] - 1,
                      jobs[batch[n][2]][1], 0, 2.2) for n in drv]
            with torch.no_grad(), autocast(device):
                m, kl = student(*build(prepare(items), device))
            actions[drv, :2] = m.float().cpu().numpy()
            actions[drv, 2:] = np.where(kl.float().cpu().numpy() > 0, 1.0, -1.0)
        for w in active:
            conns[w].send({s: actions[n] for n, (ww, s, *_) in enumerate(batch) if ww == w})
    for p in workers:
        p.join()
    return [dict(map=m, t=ts[j][:steps[j]], cursor=cur[j][:steps[j]], action=act[j][:steps[j]], lazer_acc=accs[j])
            for j, (m, _, _) in enumerate(jobs)]


# ---------------------------------------------------------------- training

def batches(plays, size, seed, out, stop):
    """Background thread: random (play, step) samples, prepared on the CPU, into the queue `out` until `stop`."""
    rng = np.random.default_rng(seed)
    lens = np.array([len(p["t"]) for p in plays], float)
    prob = lens / lens.sum()
    while not stop.is_set():
        items, labels = [], []
        for j in rng.choice(len(plays), size, p=prob):
            p = plays[j]
            i = int(rng.integers(N_FRAMES, len(p["t"])))
            items.append((p["map"], p["t"], p["cursor"], i, rng.uniform(0, MAX_LAT), int(rng.integers(0, 3)), rng.uniform(1.5, 3.0)))
            labels.append(p["action"][i])
        batch = (prepare(items), np.array(labels, np.float32), int(rng.integers(1 << 30)))
        while not stop.is_set():
            try:
                out.put(batch, timeout=0.5)
                break
            except queue.Full:
                pass


def train(model, plays, updates, device, seed):
    q, stop = queue.Queue(maxsize=4), threading.Event()
    for n in range(2):  # two preparers keep the GPU busy
        threading.Thread(target=batches, args=(plays, 256, seed * 10 + n, q, stop), daemon=True).start()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    t0 = time.time()
    for step in range(updates):
        prepared, labels, aug_seed = q.get()
        f, aux = build(prepared, device, np.random.default_rng(aug_seed))
        a = torch.from_numpy(labels).to(device)
        with autocast(device):
            m, k = model(f, aux)
        loss = F.mse_loss(m.float(), a[:, :2].clamp(-1, 1)) + F.binary_cross_entropy_with_logits(k.float(), (a[:, 2:] > 0).float())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"  update {step}/{updates} loss {loss.item():.4f} ({(time.time() - t0) / max(step, 1) * 1000:.0f} ms/update)", flush=True)
    stop.set()
    model.eval()


def evaluate(model, maps, lat, procs, device, seed=0):
    plays = rollouts([(m, lat, seed + n) for n, m in enumerate(maps)], model, 1.0, procs, device)
    return float(np.mean([p["lazer_acc"] for p in plays])), plays


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--maps", type=int, default=300, help="training maps rolled out per round")
    p.add_argument("--updates", type=int, default=20000, help="gradient updates per round")
    p.add_argument("--procs", type=int, default=12, help="simulator worker processes")
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
    model, plays, best, log = Student().to(device).to(memory_format=torch.channels_last), [], (-1.0, None), []
    os.makedirs(args.out, exist_ok=True)
    for r in range(args.rounds):
        maps = [str(f) for f in rng.choice(pool, args.maps, replace=False)]
        t0 = time.time()
        d = drive[min(r, 3)]
        model.eval()
        new = rollouts([(m, float(rng.uniform(0, MAX_LAT)), int(rng.integers(1 << 30))) for m in maps],
                       model if r else None, d, args.procs, device)
        plays += new
        print(f"round {r}: {len(new)} rollouts ({d:.0%} student) in {time.time() - t0:.0f} s, "
              f"{sum(len(x['t']) for x in plays)} labelled steps", flush=True)
        train(model, plays, args.updates, device, args.seed * 100 + r)
        acc, _ = evaluate(model, val, 0.0, args.procs, device)
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
        acc, ps = evaluate(model, maps, lat, args.procs, device)
        summary[name] = dict(lazer_acc=acc, per_map={os.path.basename(x["map"]): x["lazer_acc"] for x in ps})
        print(f"{name}: lazer acc {acc:.2%}", flush=True)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
