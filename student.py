"""Exp 019: a student that plays from the screen alone, taught by the RL teacher (DAgger, "learning by cheating").

The teacher (Exp 018 seed 1) reads the beatmap; the student sees only the last 4 frames lazer would show (gpu_render.py),
plus where its own cursor is and how old the frames are. Every step of every rollout is labelled with what the
teacher would do there, whoever was driving.

Everything that can runs on the GPU: sampling training steps, looking up what is on screen, frames, the cursor dot,
augmentation, and the teacher's and student's inference for all environments at once. Worker processes only step the
simulator.

    python student.py --rounds 4 --maps 300 --updates 20000      # Linux / WSL with CUDA
    python student.py --smoke                                    # a few minutes, checks the whole loop
"""
import argparse
import glob
import json
import os
import time
from multiprocessing import Pipe, get_context

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from beatmap import parse_osu
from env import OsuEnv, k_from_obs
from gpu_render import MAP_STRIDE, Scene, SceneTable, render_fast
from render import H, W

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

class Maps:
    """Every map seen so far, with its objects on the GPU (gpu_render.SceneTable); rebuilt when maps are added."""

    def __init__(self, device):
        self.device, self.paths, self.index, self.table = device, [], {}, None

    def ids(self, paths):
        new = [m for m in dict.fromkeys(paths) if m not in self.index]
        for m in new:
            self.index[m] = len(self.paths)
            self.paths.append(m)
        if new or self.table is None:
            self.table = SceneTable([scene(m) for m in self.paths], self.device)
        return np.array([self.index[m] for m in paths])


def build(maps, map_ids, times, dots, aux, rng=None):
    """map_ids, times (B * N_FRAMES,), dots (B, N_FRAMES, 3: cursor x, y, radius), aux (B, 3), all on the GPU ->
    frames (B, N_FRAMES, H, W) with the cursor dot drawn in (and augmented when rng is given), aux."""
    img = render_fast(maps.table.pack(map_ids, times), dots.reshape(-1, 3)).view(-1, N_FRAMES, H, W)
    if rng is not None:
        img = augment(img, rng)
    return img, aux


def live_inputs(maps, items):
    """For acting during a rollout: items [(map, times, cursors, i, lat)] with the play so far -> build() arguments."""
    ids = maps.ids([it[0] for it in items])
    times, dots, aux = [], [], []
    for path, ts, cur, i, lat in items:
        for j in range(N_FRAMES):
            t = ts[i] - lat - j * DT
            k = max(0, int(np.searchsorted(ts[:i + 1], t, "right")) - 1)
            times.append(t)
            dots.append((cur[k][0], cur[k][1], 2.2))
        aux.append((cur[i][0] / 512, cur[i][1] / 384, lat / MAX_LAT))
    to = lambda arr, dt: torch.tensor(np.array(arr), dtype=dt, device=maps.device)
    return to(np.repeat(ids, N_FRAMES), torch.long), to(times, torch.float32), to(dots, torch.float32), to(aux, torch.float32)


class Tracks:
    """All labelled plays on the GPU: sampling a training batch (steps, frame times, where the cursor was, labels)
    happens there too."""

    def __init__(self, plays, maps):
        dev = maps.device
        lens = np.array([len(p["t"]) for p in plays])
        first = np.concatenate([[0], np.cumsum(lens)[:-1]])
        pid = np.repeat(np.arange(len(plays)), lens)
        t = np.concatenate([p["t"] for p in plays])
        self.first = torch.from_numpy(first).to(dev)
        self.pid = torch.from_numpy(pid).to(dev)
        self.t = torch.from_numpy(t).to(dev)
        self.keys = torch.from_numpy(pid * MAP_STRIDE + t).to(dev)
        self.cursor = torch.from_numpy(np.concatenate([p["cursor"] for p in plays])).to(dev)
        self.action = torch.from_numpy(np.concatenate([p["action"] for p in plays])).to(dev)
        local = np.arange(len(pid)) - np.repeat(first, lens)
        self.steps = torch.from_numpy(np.nonzero(local >= N_FRAMES)[0]).to(dev)
        self.map_of = torch.from_numpy(maps.ids([p["map"] for p in plays])).to(dev)

    def batch(self, size, gen):
        dev = self.t.device
        g = self.steps[torch.randint(len(self.steps), (size,), generator=gen, device=dev)]
        pid = self.pid[g]
        lat = torch.rand(size, generator=gen, device=dev) * MAX_LAT
        lag = torch.randint(0, 3, (size, 1), generator=gen, device=dev) * DT  # cursor drawn up to 2 frames late
        rad = 1.5 + 1.5 * torch.rand(size, 1, generator=gen, device=dev)
        times = self.t[g][:, None] - lat[:, None] - torch.arange(N_FRAMES, device=dev) * DT  # (size, N_FRAMES)
        k = torch.searchsorted(self.keys, pid[:, None] * MAP_STRIDE + (times - lag).double(), right=True) - 1
        k = torch.maximum(k, self.first[pid][:, None])
        dots = torch.cat([self.cursor[k], rad[:, :, None].expand(-1, N_FRAMES, 1)], 2)
        aux = torch.stack([self.cursor[g, 0] / 512, self.cursor[g, 1] / 384, lat / MAX_LAT], 1)
        ids = self.map_of[pid].repeat_interleave(N_FRAMES)
        return (ids, times.reshape(-1).float(), dots, aux), self.action[g]


def _mix(x, gain, field, noise, sigma):
    return (x * gain + field.unsqueeze(1) + noise * sigma).clamp(0, 1)


try:  # one fused kernel instead of five passes over the batch
    import triton  # noqa: F401
    _mix = torch.compile(_mix, dynamic=False)
except ImportError:
    pass


def augment(x, rng):
    """Brightness, blur, glows and noise the renderer does not have, per sample."""
    b, dev = x.shape[0], x.device
    to = lambda a: torch.from_numpy(np.asarray(a, np.float32)).to(dev)
    # blur, for 30% of the samples: one fixed gaussian (sigma 0.8) mixed in by a per-sample amount
    blur = np.nonzero(rng.random(b) < 0.3)[0]
    if len(blur):
        taps = torch.exp(-torch.arange(-2, 3, device=dev, dtype=torch.float32) ** 2 / (2 * 0.8 ** 2))
        k = (taps[:, None] * taps[None, :]) / taps.sum() ** 2
        idx = torch.from_numpy(blur).to(dev)
        xs = x[idx]
        y = F.conv2d(F.pad(xs.reshape(-1, 1, H, W), (2, 2, 2, 2), mode="replicate"), k.view(1, 1, 5, 5)).view_as(xs)
        x = x.index_copy(0, idx, torch.lerp(xs, y, to(rng.uniform(0.3, 1.0, len(blur))).view(-1, 1, 1, 1)))
    glow = rng.uniform(size=(b, 3, 4)) * [H, W, 4, 0.30] + [0, 0, 2, 0.05]  # centre y, x, size, strength
    glow[:, :, 3] *= (np.arange(3) < rng.integers(1, 4, (b, 1))) & (rng.random((b, 1)) < 0.5)
    g = torch.from_numpy(glow.astype(np.float32)).to(dev)
    yy = torch.arange(H, device=dev, dtype=torch.float32)
    xx = torch.arange(W, device=dev, dtype=torch.float32)
    two_s2 = 2 * g[..., 2:3] ** 2
    gy = g[..., 3:4] * torch.exp(-(yy - g[..., 0:1]) ** 2 / two_s2)  # a 2-d gaussian is a product of two 1-d ones
    gx = torch.exp(-(xx - g[..., 1:2]) ** 2 / two_s2)
    field = torch.einsum("bgh,bgw->bhw", gy, gx)
    gain = to(rng.uniform(0.75, 1.25, b)).view(b, 1, 1, 1)
    sigma = to(rng.uniform(0, 0.03, b)).view(b, 1, 1, 1)
    return _mix(x, gain, field, torch.randn_like(x), sigma)


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


def rollouts(jobs, student, drive, procs, maps):
    """jobs: [(map, latency ms, seed)]. The student drives each step with probability `drive`; every step is labelled
    with the teacher's action. -> [dict(map, t, cursor, action, lazer_acc)]"""
    device = maps.device
    t_model = teacher(device)
    k = k_from_obs(t_model.observation_space.shape[0])
    maps.ids([m for m, _, _ in jobs])
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
            items = [(jobs[batch[n][2]][0], ts[batch[n][2]], cur[batch[n][2]], steps[batch[n][2]] - 1, jobs[batch[n][2]][1])
                     for n in drv]
            with torch.no_grad(), autocast(device):
                m, kl = student(*build(maps, *live_inputs(maps, items)))
            actions[drv, :2] = m.float().cpu().numpy()
            actions[drv, 2:] = np.where(kl.float().cpu().numpy() > 0, 1.0, -1.0)
        for w in active:
            conns[w].send({s: actions[n] for n, (ww, s, *_) in enumerate(batch) if ww == w})
    for p in workers:
        p.join()
    return [dict(map=m, t=ts[j][:steps[j]], cursor=cur[j][:steps[j]], action=act[j][:steps[j]], lazer_acc=accs[j])
            for j, (m, _, _) in enumerate(jobs)]


# ---------------------------------------------------------------- training

def train(model, plays, updates, maps, seed):
    tracks = Tracks(plays, maps)
    gen = torch.Generator(device=maps.device).manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    t0 = time.time()
    for step in range(updates):
        inputs, a = tracks.batch(256, gen)
        f, aux = build(maps, *inputs, rng=rng)
        with autocast(maps.device):
            m, k = model(f, aux)
        loss = F.mse_loss(m.float(), a[:, :2].clamp(-1, 1)) + F.binary_cross_entropy_with_logits(k.float(), (a[:, 2:] > 0).float())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"  update {step}/{updates} loss {loss.item():.4f} ({(time.time() - t0) / max(step, 1) * 1000:.0f} ms/update)", flush=True)
    model.eval()


def evaluate(model, paths, lat, procs, maps, seed=0):
    plays = rollouts([(m, lat, seed + n) for n, m in enumerate(paths)], model, 1.0, procs, maps)
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
    maps = Maps(device)
    os.makedirs(args.out, exist_ok=True)
    for r in range(args.rounds):
        maps = [str(f) for f in rng.choice(pool, args.maps, replace=False)]
        t0 = time.time()
        d = drive[min(r, 3)]
        model.eval()
        new = rollouts([(m, float(rng.uniform(0, MAX_LAT)), int(rng.integers(1 << 30))) for m in maps],
                       model if r else None, d, args.procs, maps)
        plays += new
        print(f"round {r}: {len(new)} rollouts ({d:.0%} student) in {time.time() - t0:.0f} s, "
              f"{sum(len(x['t']) for x in plays)} labelled steps", flush=True)
        train(model, plays, args.updates, maps, args.seed * 100 + r)
        acc, _ = evaluate(model, val, 0.0, args.procs, maps)
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
    for name, paths, lat in [("real", real, 0.0), ("hf", hf, 0.0), ("real_lat20", real, 20.0)]:
        acc, ps = evaluate(model, paths, lat, args.procs, maps)
        summary[name] = dict(lazer_acc=acc, per_map={os.path.basename(x["map"]): x["lazer_acc"] for x in ps})
        print(f"{name}: lazer acc {acc:.2%}", flush=True)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
