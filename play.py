import argparse
from collections import Counter

import numpy as np
from stable_baselines3 import PPO

from beatmap import SPINNER, parse_osu
from env import OsuEnv, k_from_obs
from replay import write_osr


def oracle(env, noise=0.0, rng=None):
    """Scripted player (autoplay baseline): alternates keys, follows slider balls, spins spinners.
    noise > 0 jitters aim and keys, used to fuzz the simulator against lazer."""
    keys = [False, False]
    nxt = next((i for i in range(env.next, len(env.objs)) if not env.started[i] and env.kinds[i] != SPINNER), None)
    t1, t2 = round(env.clock + env.dt), round(env.clock + 2 * env.dt)  # next two frame times
    if nxt is not None and (env.t + t1) / 2 < env.objs[nxt, 2] <= (t1 + t2) / 2:  # next frame is the closest one
        target = env.objs[nxt, :2]
        k = nxt % 2 if not env.keys[nxt % 2] else 1 - nxt % 2
        keys[k] = True
        if env.active and env.active[0]["hit_key"] is not None:
            keys[env.active[0]["hit_key"]] = True
    elif env.active and env.active[0]["pending"]:
        s = env.active[0]
        target = s["sl"].ball(env.t + env.dt)
        keys[s["hit_key"] if s["hit_key"] is not None else 0] = True
    elif env.spinning:
        c = env.objs[env.spinning[0]["idx"], :2]
        ang = np.arctan2(*(env.cursor - c)[::-1]) + 0.6
        target, keys = c + 50 * np.array([np.cos(ang), np.sin(ang)]), [True, False]
    elif nxt is not None:
        target = env.objs[nxt, :2]
    else:
        target = env.cursor
    move = np.clip((target - env.cursor) / env.max_speed, -1, 1)
    if noise:
        move = move + rng.normal(0, noise, 2)
        keys = [k != (rng.random() < noise / 4) for k in keys]
    return np.array([*move, *(1.0 if k else -1.0 for k in keys)])


def run(env, policy):
    obs, _ = env.reset(seed=0)
    frames = [(env.t, *env.cursor, env.keys)]
    done = False
    while not done:
        obs, _, done, _, _ = env.step(policy(obs))
        frames.append((env.t, *env.cursor, env.keys))
    frames = sorted(frames + env.extra_frames, key=lambda f: f[0])
    counts = Counter(s for *_, s in env.judgements)
    acc = sum(k * v for k, v in counts.items()) / (300 * len(env.objs))
    return frames, counts, acc


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("map")
    p.add_argument("--model", default="models/osu_ppo")
    p.add_argument("--out", default="replay.osr")
    p.add_argument("--hz", type=int, default=60)  # must match training
    p.add_argument("--k", type=int, default=None, help="upcoming objects (default: auto-detect from model)")
    args = p.parse_args()

    bm = parse_osu(args.map)
    model = PPO.load(args.model, device="cpu")
    k = args.k or k_from_obs(model.observation_space.shape[0])
    env = OsuEnv(bm, hz=args.hz, k=k)
    frames, counts, acc = run(env, lambda o: model.predict(o, deterministic=True)[0])
    print(f"{bm.title}: acc {acc:.2%}  300/100/50/miss = {counts[300]}/{counts[100]}/{counts[50]}/{counts[0]}  max combo {env.max_combo}")
    write_osr(args.out, args.map, frames, counts, env.max_combo)
    print(f"replay -> {args.out}")
