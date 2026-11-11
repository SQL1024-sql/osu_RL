"""Exp 019: a student that plays from the screen alone, taught by the RL teacher (DAgger, "learning by cheating").

The teacher (Exp 018 seed 1) reads the beatmap; the student sees only the last 4 frames lazer would show (gpu_render.py),
plus where its own cursor is and how old the frames are. Every step of every rollout is labelled with what the
teacher would do there, whoever was driving.

Everything that can runs on the GPU: sampling training steps, looking up what is on screen, frames, the cursor dot,
augmentation, and the teacher's and student's inference for all environments at once. Worker processes only step the
simulator.

    python -m vision.student --rounds 4 --maps 300 --updates 20000      # Linux / WSL with CUDA
    python -m vision.student --smoke                                    # a few minutes, checks the whole loop
"""
import argparse
import glob
import json
import os
import pickle
import time
from dataclasses import replace
from multiprocessing import Pipe, get_context

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from beatmap import CIRCLE, SPINNER, parse_osu
from state.env import OsuEnv, k_from_obs
from vision.gpu_render import MAP_STRIDE, Scene, SceneTable, render_fast
from vision.render import PX, H, W

HZ = 120
DT = 1000 / HZ
N_FRAMES = 4
FRAME_STEPS = [0, 1, 2, 3]  # how many steps back each frame is (--frames)
FOVEA = 0  # --fovea: side in pixels of a close-up around the cursor (0: none), so approach circles are sharp
FOVEA_PX = 0.75  # close-up pixels per osu! pixel (the full frame has 0.25): 96 px span 128 osu! px, a CS2 circle's ring fits
MAX_LAT = 30.0  # ms
TEACHER = "results/experiments/018-room/seed1/model.zip"
SEL_SEED = 20_000  # the HF test sample of eval_hf.py / select_ckpt.py
SLOTS = 8  # environments per worker process
MAP_KEYS = False  # --map-keys: click / hold labels from the map (object starts, slider and spinner spans), not the teacher
TEACHER_CLICKS = False  # --teacher-clicks: click labels from the teacher's own key transitions (see Tracks.labels)
CLICK_TH = 0.0  # click logit a peak must reach to fire a click
CLICK_GAP = 0  # steps a click must wait after the last one (a logit that dips below the threshold and back fires twice)
# steps after a click that its key stays held whatever the hold logit says (Exp 031): the hold logit only rises once
# the frames show the slider started, a few steps after its head was clicked, and a key let go is never pressed again
MIN_HOLD = 0
AUX = 3  # cursor x, y, latency (the last move or held keys as inputs get copied: the student keeps doing what it did)

_scenes = {}


def load(key):
    """A job's map: a path, or "path|cs|ar" for that map with its circle size and approach rate changed (stacking
    stays as parsed)."""
    path, *mod = key.split("|")
    bm = parse_osu(path)
    return replace(bm, cs=float(mod[0]), ar=float(mod[1])) if mod else bm


def scene(path):
    if path not in _scenes:
        _scenes[path] = Scene(load(path))
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
    """map_ids, times (B * N_FRAMES,), dots (B, N_FRAMES, 3: cursor x, y, radius), aux (B, AUX), all on the GPU ->
    frames (B, N_FRAMES, H, W) with the cursor dot drawn in (and augmented when rng is given), aux."""
    p = maps.table.pack(map_ids, times)
    dots = dots.view(-1, N_FRAMES, 3)  # live_inputs gives (B * N_FRAMES, 3)
    img = render_fast(p, dots.reshape(-1, 3)).view(-1, N_FRAMES, H, W)
    if rng is not None:
        img = augment(img, rng)
    if not FOVEA:
        return img, aux
    # the same 4 moments up close, all centred where the cursor is in the newest frame (a still circle stays put)
    origin = (dots[:, :1, :2] - FOVEA / 2 / FOVEA_PX).expand(-1, N_FRAMES, -1).reshape(-1, 2)
    big = torch.cat([dots[..., :2], dots[..., 2:] * FOVEA_PX / PX], 2).reshape(-1, 3)
    crop = render_fast(p, big, origin, FOVEA_PX, FOVEA, FOVEA).view(-1, N_FRAMES, FOVEA, FOVEA)
    if rng is not None:
        b = crop.shape[0]
        gain = torch.empty(b, 1, 1, 1, device=crop.device).uniform_(0.75, 1.25)
        sigma = torch.empty(b, 1, 1, 1, device=crop.device).uniform_(0, 0.03)
        crop = (crop * gain + torch.randn_like(crop) * sigma).clamp(0, 1)
    return img, crop, aux


def live_inputs(maps, items):
    """For acting during a rollout: items [(map, times, cursors, keys, i, lat)] with the play so far -> build() arguments."""
    ids = maps.ids([it[0] for it in items])
    times, dots, aux = [], [], []
    for path, ts, cur, held, i, lat in items:
        for j in range(N_FRAMES):
            t = ts[i] - lat - FRAME_STEPS[j] * DT
            k = max(0, int(np.searchsorted(ts[:i + 1], t, "right")) - 1)
            times.append(t)
            dots.append((cur[k][0], cur[k][1], 2.2))
        aux.append((cur[i][0] / 512, cur[i][1] / 384, lat / MAX_LAT))
    to = lambda arr, dt: torch.tensor(np.array(arr), dtype=dt, device=maps.device)
    return to(np.repeat(ids, N_FRAMES), torch.long), to(times, torch.float32), to(dots, torch.float32), to(aux, torch.float32)


_w50 = {}


def map_keys(sc, t, cursor, key):
    """(steps, 2) float: click and hold labels from the map and where the cursor actually was. An object is clicked at
    the step nearest its start on which the cursor is on it, within the 50 window either side (none if it never is;
    the teacher often clicks a little early and is gone by the start); the click label covers the two steps before it, as the click rule fires the step after the
    logit's peak (lines up with the teacher's own clicks). Hold: from that click to the object's end for sliders and spinners. The teacher's keys depend on
    the key state it sees, which the student's own key handling makes unlike anything the teacher was trained on, so
    in student-driven plays they are no guide to when a click is due (Exp 019-024); clicking at the start whatever
    the cursor does wastes the click when the cursor is late (Exp 025)."""
    if key not in _w50:
        _w50[key] = load(key).windows[2]
    click, d = np.zeros(len(t), bool), np.zeros(len(t) + 1)
    for j in range(len(sc.start)):
        s0 = sc.start[j]
        if sc.kind[j] == SPINNER:
            k = min(int(np.searchsorted(t, s0)), len(t) - 1)
        else:
            k0, k1 = int(np.searchsorted(t, s0 - _w50[key])), int(np.searchsorted(t, s0 + _w50[key], "right"))
            on = np.nonzero(np.hypot(*(cursor[k0:k1] - sc.path[j, 0]).T) <= sc.r)[0]
            if not len(on):
                continue
            k = k0 + int(on[np.argmin(np.abs(t[k0 + on] - s0))])
        click[max(0, k - 2):k] = True  # the two steps before it: the rule fires the step after the peak
        if sc.kind[j] != CIRCLE:
            d[k] += 1
            d[max(k + 1, int(np.searchsorted(t, sc.end[j])))] -= 1
    return np.stack([click, np.cumsum(d)[:-1] > 0], 1).astype(np.float32)


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
        self.held = torch.from_numpy(np.concatenate([p["keys"] for p in plays])).to(dev)
        local = np.arange(len(pid)) - np.repeat(first, lens)
        self.steps = torch.from_numpy(np.nonzero(local >= N_FRAMES)[0]).to(dev)
        self.map_of = torch.from_numpy(maps.ids([p["map"] for p in plays])).to(dev)
        nxt = []  # from the map, not the teacher: where the next object to hit is, and how long until it is due
        for p in plays:
            sc = scene(p["map"])
            j = np.minimum(np.searchsorted(sc.start, p["t"]), len(sc.start) - 1)
            nxt.append(np.stack([sc.path[j, 0, 0] / 512, sc.path[j, 0, 1] / 384,
                                 np.clip((sc.start[j] - p["t"]) / 1000, 0, 1)], 1).astype(np.float32))
        self.next = torch.from_numpy(np.concatenate(nxt)).to(dev, torch.float16)
        if MAP_KEYS:
            self.mkeys = torch.from_numpy(np.concatenate([map_keys(scene(p["map"]), p["t"], p["cursor"], p["map"]) for p in plays])).to(dev)  # half: 43M steps would be 0.5 GB

    def batch(self, size, gen):
        dev = self.t.device
        g = self.steps[torch.randint(len(self.steps), (size,), generator=gen, device=dev)]
        pid = self.pid[g]
        lat = torch.rand(size, generator=gen, device=dev) * MAX_LAT
        lag = torch.randint(0, 3, (size, 1), generator=gen, device=dev) * DT  # cursor drawn up to 2 frames late
        rad = 1.5 + 1.5 * torch.rand(size, 1, generator=gen, device=dev)
        times = self.t[g][:, None] - lat[:, None] - torch.tensor(FRAME_STEPS, device=dev) * DT  # (size, N_FRAMES)
        k = torch.searchsorted(self.keys, pid[:, None] * MAP_STRIDE + (times - lag).double(), right=True) - 1
        k = torch.maximum(k, self.first[pid][:, None])
        dots = torch.cat([self.cursor[k], rad[:, :, None].expand(-1, N_FRAMES, 1)], 2)
        aux = torch.stack([self.cursor[g, 0] / 512, self.cursor[g, 1] / 384, lat / MAX_LAT], 1)
        ids = self.map_of[pid].repeat_interleave(N_FRAMES)
        return (ids, times.reshape(-1).float(), dots, aux), torch.cat([self.labels(g), self.next[g].float()], 1)

    def labels(self, g):
        if MAP_KEYS:
            return torch.cat([self.action[g, :2].clamp(-1, 1), self.mkeys[g]], 1)
        """(move x, move y, click, hold). The teacher's keys are turned into events: a click is a key going down (on
        this step or the next, so a click is not split over two steps), hold is any key held. Copying held keys
        teaches a student to keep doing what it did, and a held key never clicks.
        A key goes down when the teacher presses it after its own previous action had it up (--teacher-clicks). Measured
        against the game's keys instead, every step the student had a key up that the teacher holds counts as a click:
        22 per hit object in student-driven data against 1.1 in teacher-driven (Exp 019-023)."""
        nxt = torch.clamp(g + 1, max=len(self.t) - 1)
        if TEACHER_CLICKS:
            def down(i):
                prv = torch.clamp(i - 1, min=0)
                was = torch.where((self.pid[prv] == self.pid[i])[:, None], self.action[prv, 2:] > 0, self.held[i] > 0.5)
                return ((self.action[i, 2:] > 0) & ~was).any(1)
        else:
            down = lambda i: ((self.action[i, 2:] > 0) & (self.held[i] < 0.5)).any(1)
        click = down(g) | (down(nxt) & (self.pid[nxt] == self.pid[g]))
        hold = (self.action[g, 2:] > 0).any(1)
        return torch.cat([self.action[g, :2].clamp(-1, 1), click[:, None].float(), hold[:, None].float()], 1)


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
    def __init__(self, where=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(N_FRAMES, 32, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),  # (128, 8, 10): keeps where things are
        )
        self.fovea = None
        extra = 0
        if FOVEA:
            self.fovea = nn.Sequential(
                nn.Conv2d(N_FRAMES, 32, 5, 2, 2), nn.ReLU(),
                nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(),
            )
            extra = 64 * ((FOVEA + 15) // 16) ** 2
        self.head = nn.Sequential(nn.Linear(128 * 8 * 10 + extra + AUX, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 4))
        # training only: predict the next object's x, y and time to hit, so the CNN has to find it on screen
        self.where = nn.Linear(256, 3) if where else None

    def forward(self, frames, *rest):
        z = self.conv(frames.to(memory_format=torch.channels_last)).flatten(1)
        aux = rest[-1]
        if self.fovea is not None:
            z = torch.cat([z, self.fovea(rest[0].to(memory_format=torch.channels_last)).flatten(1)], 1)
        h = self.head[:-1](torch.cat([z, aux], 1))
        out = self.head[-1](h)
        if self.training and self.where is not None:
            return torch.tanh(out[:, :2]), out[:, 2:], self.where(h)
        return torch.tanh(out[:, :2]), out[:, 2:]  # move in [-1, 1], logits of (click, hold)


def keys_for(click, hold, held, last, clicked):
    """The student's (click, hold) -> key actions (+1 held, -1 up) and the key a click used. A click presses a key
    that is up, the other one than last time, like a player alternating; no two clicks on consecutive steps."""
    if click and not clicked:
        up = [k for k in (1 - last, last) if not held[k]]
        if not up:
            return np.array([-1.0, -1.0]), last, False  # both held: let go now, click on the next step
        out = np.full(2, -1.0)
        out[up[0]] = 1.0
        return out, up[0], True
    return (np.where(held, 1.0, -1.0) if hold else np.array([-1.0, -1.0])), last, False


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
            env = OsuEnv(load(path), hz=HZ, k=k, classic=False)
            obs, _ = env.reset(seed=seed)
            slots[slot] = (jid, env, obs)

    for s in range(SLOTS):
        start(s)
    while True:
        conn.send(([(s, jid, obs, env.t, env.cursor.copy(), env.keys) for s, (jid, env, obs) in slots.items()], finished))
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
    held = [np.zeros((n, 2), np.float32) for n in n_max]
    steps, accs = [0] * len(jobs), {}
    rngs = [np.random.default_rng(seed) for _, _, seed in jobs]
    last, clicked = [0] * len(jobs), [False] * len(jobs)
    prev, fired = [-np.inf] * len(jobs), [False] * len(jobs)
    last_fire = [-10 ** 9] * len(jobs)
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
        for n, (w, s, jid, _, t, c, kh) in enumerate(batch):
            i = steps[jid]
            ts[jid][i], cur[jid][i], act[jid][i], held[jid][i] = t, c, a_t[n], kh
            steps[jid] += 1
            if student is not None and rngs[jid].random() < drive:
                drv.append(n)
        if drv:
            items = [(jobs[batch[n][2]][0], ts[batch[n][2]], cur[batch[n][2]], held[batch[n][2]], steps[batch[n][2]] - 1, jobs[batch[n][2]][1])
                     for n in drv]
            with torch.no_grad(), autocast(device):
                m, kl = student(*build(maps, *live_inputs(maps, items)))
            actions[drv, :2] = m.float().cpu().numpy()
            for n, (c, h) in zip(drv, kl.float().cpu().numpy()):
                jid = batch[n][2]
                # click once per stretch of click logit > 0, on the step after its peak: clicking as soon as it
                # crosses 0 lands 2-3 steps early, since the logit rises over several steps before the click
                fire = prev[jid] > CLICK_TH and c < prev[jid] and not fired[jid] and steps[jid] - last_fire[jid] >= CLICK_GAP
                if fire:
                    last_fire[jid] = steps[jid]
                fired[jid] = (fired[jid] or fire) and c > CLICK_TH
                prev[jid] = c
                hold = h > 0 or steps[jid] - last_fire[jid] <= MIN_HOLD
                actions[n, 2:], last[jid], clicked[jid] = keys_for(fire, hold, batch[n][6], last[jid], clicked[jid])
        for w in active:
            conns[w].send({s: actions[n] for n, (ww, s, *_) in enumerate(batch) if ww == w})
    for p in workers:
        p.join()
    return [dict(map=m, t=ts[j][:steps[j]], cursor=cur[j][:steps[j]], keys=held[j][:steps[j]], action=act[j][:steps[j]], lazer_acc=accs[j])
            for j, (m, _, _) in enumerate(jobs)]


# ---------------------------------------------------------------- training

def train(model, plays, updates, maps, seed, decay=False):
    tracks = Tracks(plays, maps)
    gen = torch.Generator(device=maps.device).manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    # --lr-decay: cosine down to 1% within the round, so each round ends on settled weights instead of a noisy point
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, updates, eta_min=3e-6) if decay else None
    model.train()
    t0 = time.time()
    for step in range(updates):
        inputs, a = tracks.batch(256, gen)
        x = build(maps, *inputs, rng=rng)
        with autocast(maps.device):
            m, k, *w = model(*x)
        loss = F.mse_loss(m.float(), a[:, :2]) + F.binary_cross_entropy_with_logits(k.float(), a[:, 2:4])
        if w:
            loss = loss + F.mse_loss(w[0].float(), a[:, 4:])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sched:
            sched.step()
        if step % 500 == 0:
            print(f"  update {step}/{updates} loss {loss.item():.4f} ({(time.time() - t0) / max(step, 1) * 1000:.0f} ms/update)", flush=True)
    model.eval()


def evaluate(model, paths, lat, procs, maps, seed=0):
    plays = rollouts([(m, lat, seed + n) for n, m in enumerate(paths)], model, 1.0, procs, maps)
    return float(np.mean([p["lazer_acc"] for p in plays])), plays


def usable_cpus():
    """CPU cores this process may use: its affinity mask, capped by a container's CPU quota (cgroup v2)."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            n = min(n, int(quota) // int(period))
    except (OSError, ValueError):
        pass
    return max(1, n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--init", help="start from these weights (a round*.pt) instead of a new student")
    p.add_argument("--first-round", type=int, default=0, help="number of the first round (sets how often the student drives)")
    p.add_argument("--maps", type=int, default=300, help="training maps rolled out per round")
    p.add_argument("--updates", type=int, default=20000, help="gradient updates per round")
    p.add_argument("--procs", type=int, help="simulator worker processes (default: the usable cores minus 2, split by --share)")
    p.add_argument("--share", type=int, default=1, help="how many runs share this machine")
    p.add_argument("--vary", type=float, default=0.5, help="share of training maps given a random CS (2-9) and AR (2-10)")
    p.add_argument("--where", action="store_true", help="train an extra head to find the next object (Exp 021)")
    p.add_argument("--frames", default="0,1,2,3", help="steps back of the 4 frames (Exp 022: 0,2,6,16)")
    p.add_argument("--fovea", type=int, default=0, help="side of a sharp close-up around the cursor (Exp 023: 96)")
    p.add_argument("--lr-decay", action="store_true", help="cosine learning-rate decay within each round (Exp 024)")
    p.add_argument("--map-keys", action="store_true", help="click / hold labels from the map (Exp 025)")
    p.add_argument("--teacher-clicks", action="store_true", help="click labels from the teacher's own key presses (Exp 024)")
    p.add_argument("--resume", action="store_true", help="continue a run in --out from its saved rounds and data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="default: results/experiments/019-student/seed<seed>")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    FRAME_STEPS[:] = map(int, args.frames.split(","))
    global FOVEA, TEACHER_CLICKS, MAP_KEYS
    FOVEA, TEACHER_CLICKS, MAP_KEYS = args.fovea, args.teacher_clicks, args.map_keys
    assert len(FRAME_STEPS) == N_FRAMES
    if args.smoke:
        args.rounds, args.maps, args.updates = 2, 4, 200
    args.procs = args.procs or max(1, (usable_cpus() - 2) // args.share)  # main process + GPU driver keep the rest
    args.out = args.out or f"results/experiments/019-student/seed{args.seed}"
    print(f"{args.procs} simulator processes, results in {args.out}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":  # stay inside this run's share of VRAM instead of spilling into shared system memory
        torch.cuda.set_per_process_memory_fraction(0.9 / args.share)
    pool = sorted(glob.glob("maps/hf/train/*.osu"))
    val = sorted(glob.glob("maps/hf/val/*.osu"))
    val = [str(f) for f in np.random.default_rng(1).choice(val, 3 if args.smoke else 20, replace=False)]
    drive = [0.0, 0.5, 0.75, 1.0]
    model, plays, best, log = Student(args.where).to(device).to(memory_format=torch.channels_last), [], (-1.0, None), []
    maps = Maps(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device))
    os.makedirs(args.out, exist_ok=True)
    at = lambda name: os.path.join(args.out, name)
    if args.resume:  # DAgger trains on every round's data, so a continued run needs all of it back
        while os.path.exists(at(f"plays{args.first_round}.pkl")) and os.path.exists(at(f"round{args.first_round}.pt")):
            with open(at(f"plays{args.first_round}.pkl"), "rb") as f:
                plays += pickle.load(f)
            args.first_round += 1
            args.rounds -= 1
        model.load_state_dict(torch.load(at(f"round{args.first_round - 1}.pt"), map_location=device))
        log = json.load(open(at("log.json")))[:args.first_round]
        best = max((x["val_lazer_acc"], x["round"]) for x in log)
        print(f"resumed after round {args.first_round - 1}, {sum(len(x['t']) for x in plays)} labelled steps", flush=True)
    for r in range(args.first_round, args.first_round + args.rounds):
        rng = np.random.default_rng([args.seed, r])
        picked = [str(f) for f in rng.choice(pool, args.maps, replace=False)]
        # small circles and low AR are rare in the pool (CS >= 7 is 0.3% of it), so many maps get a random CS and AR
        picked = [f"{m}|{rng.uniform(2, 9):.1f}|{rng.uniform(2, 10):.1f}" if rng.random() < args.vary else m for m in picked]
        t0 = time.time()
        d = drive[min(r, 3)]
        model.eval()
        new = rollouts([(m, float(rng.uniform(0, MAX_LAT)), int(rng.integers(1 << 30))) for m in picked],
                       model if r or args.init else None, d, args.procs, maps)
        plays += new
        with open(at(f"plays{r}.pkl"), "wb") as f:
            pickle.dump(new, f)
        print(f"round {r}: {len(new)} rollouts ({d:.0%} student) in {time.time() - t0:.0f} s, "
              f"{sum(len(x['t']) for x in plays)} labelled steps", flush=True)
        train(model, plays, args.updates, maps, args.seed * 100 + r, args.lr_decay)
        torch.cuda.empty_cache()  # give the training batch memory back before rollouts, so nothing spills into system RAM
        acc, _ = evaluate(model, val, 0.0, args.procs, maps)
        print(f"round {r}: validation lazer acc {acc:.2%}", flush=True)
        log.append(dict(round=r, val_lazer_acc=acc, steps=sum(len(x["t"]) for x in plays)))
        torch.save(model.state_dict(), at(f"round{r}.pt"))
        json.dump(log, open(at("log.json"), "w"), indent=1)
        if acc > best[0]:
            best = (acc, r)
    model.load_state_dict(torch.load(os.path.join(args.out, f"round{best[1]}.pt")))
    real = sorted(glob.glob("maps/test/*.osu"))
    hf = sorted(glob.glob("maps/hf/test/*.osu"))
    hf = [str(f) for f in np.random.default_rng(SEL_SEED).choice(hf, min(60, len(hf)), replace=False)]
    if args.smoke:
        real, hf = real[:2], hf[:2]
    summary = dict(best_round=best[1], rounds=log, frames=FRAME_STEPS, fovea=FOVEA, teacher_clicks=TEACHER_CLICKS, map_keys=MAP_KEYS)
    for name, paths, lat in [("real", real, 0.0), ("hf", hf, 0.0), ("real_lat20", real, 20.0)]:
        acc, ps = evaluate(model, paths, lat, args.procs, maps)
        summary[name] = dict(lazer_acc=acc, per_map={os.path.basename(x["map"]): x["lazer_acc"] for x in ps})
        print(f"{name}: lazer acc {acc:.2%}", flush=True)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
