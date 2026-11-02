import argparse
import glob
import json
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import SubprocVecEnv

import numpy as np
import torch

import env as env_mod
from beatmap import parse_osu, random_beatmap, window
from env import OsuEnv, K
from grouped_vec_env import GroupedSubprocVecEnv
from play import run


VAL_SEEDS = range(10_000, 10_005)  # fixed random maps for picking checkpoints; the real map stays a pure test


def val_maps(diff, pool=None):
    """Checkpoint-picking maps: 5 fixed random maps, plus 5 fixed 100-object windows of held-out real maps (the
    pool's sibling val/ folder) when training on a real-map pool."""
    maps = [random_beatmap(np.random.default_rng(s), n=100, diff=diff) for s in VAL_SEEDS]
    if pool:
        rng = np.random.default_rng(VAL_SEEDS[0])
        files = sorted(glob.glob(os.path.join(os.path.dirname(os.path.normpath(pool)), "val", "*.osu")))
        for f in rng.choice(files, 5, replace=False):
            bm = parse_osu(f)
            maps.append(window(bm, int(rng.integers(max(1, len(bm.objects) - 99))), 100))
    return maps


class HybridPPO(PPO):
    """Rollouts on the CPU (8 observations per step: the GPU's transfer latency would dominate), updates on the GPU."""

    def collect_rollouts(self, *args, **kwargs):
        device = self.device
        self.policy.to("cpu")
        self.device = torch.device("cpu")
        try:
            return super().collect_rollouts(*args, **kwargs)
        finally:
            self.device = device
            self.policy.to(device)


class RealMapEval(BaseCallback):
    """Every `freq` steps: plays the real test map (eval/*) and the validation maps (val/*), logs judgement stats."""

    def __init__(self, map_path, freq, hz, diff, pool, k=K):
        super().__init__()
        self.env = OsuEnv(parse_osu(map_path), hz=hz, k=k)
        self.val_envs = [OsuEnv(bm, hz=hz, k=k) for bm in val_maps(diff, pool)]
        self.freq, self.last = freq, 0

    def _on_step(self):
        if self.num_timesteps - self.last >= self.freq:
            self.last = self.num_timesteps
            policy = lambda o: self.model.predict(o, deterministic=True)[0]
            _, counts, acc = run(self.env, policy)
            self.logger.record("eval/acc", acc)
            for k, name in ((300, "n300"), (100, "n100"), (50, "n50"), (0, "miss")):
                self.logger.record(f"eval/{name}", counts[k])
            self.logger.record("eval/max_combo", self.env.max_combo)
            val = [run(e, policy)[1:] for e in self.val_envs]
            self.logger.record("val/acc", float(np.mean([acc for _, acc in val])))
            self.logger.record("val/miss", sum(c[0] for c, _ in val))
        return True


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=float, default=5e6)
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--workers", type=int, default=0, help="worker processes for the envs; 0 = one process per env")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hz", type=int, default=60)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eval-map", default="maps/bakenohana.osu")
    p.add_argument("--name", default="")
    p.add_argument("--timing", type=float, default=0.0, help="max timing bonus per circle hit (env.TIMING_MS)")
    p.add_argument("--spin-shaping", action="store_true", help="distance shaping toward the spinner ring (env.SPIN_RADIUS)")
    p.add_argument("--spin-free-press", action="store_true", help="no empty-click penalty while a spinner runs")
    p.add_argument("--device", default="cpu", choices=["cpu", "hybrid"], help="hybrid: PPO updates on the GPU (needs CUDA torch)")
    p.add_argument("--tail-miss", type=float, default=None, help="reward for a missed slider tail (default: same as other slider parts)")
    p.add_argument("--pool", default="", help="folder of real .osu maps to train on, e.g. maps/hf/train")
    p.add_argument("--pool-frac", type=float, default=0.5, help="share of episodes taken from --pool")
    p.add_argument("--diff", default="narrow", choices=["narrow", "wide"], help="CS / AR / OD range of training maps")
    p.add_argument("--k", type=int, default=K, help="number of upcoming objects in observation (default: 3)")
    p.add_argument("--lr-decay", action="store_true", help="learning rate falls linearly from 3e-4 to 0 over --steps")
    args = p.parse_args()
    torch.set_num_threads(4)  # the default (all cores) fights the env worker processes: 4 was fastest on i7-13620H with 8 envs

    run_dir = os.path.join("runs", time.strftime("%Y%m%d-%H%M") + (f"-{args.name}" if args.name else ""))
    hparams = dict(n_steps=8192 // args.envs,  # 8192 samples per update whatever the env count
                    batch_size=2048, gamma=args.gamma, policy_kwargs=dict(net_arch=[256, 256]))
    os.makedirs(run_dir)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        consts = {k: getattr(env_mod, k) for k in ("MAX_CURSOR_SPEED", "K", "MISS_WINDOW", "REWARD", "NESTED_REWARD", "SPIN_REWARD", "CLICK_PENALTY", "SHAPING", "CURSOR_MARGIN", "LOOKAHEAD_MS", "TIMING_MS", "SPIN_RADIUS")}
        json.dump(dict(args=vars(args), ppo=hparams, env=consts, torch=torch.__version__), f, indent=2, default=str)

    vec = dict(vec_env_cls=GroupedSubprocVecEnv, vec_env_kwargs=dict(workers=args.workers)) if args.workers else dict(vec_env_cls=SubprocVecEnv)
    env = make_vec_env(OsuEnv, n_envs=args.envs, seed=args.seed, env_kwargs=dict(hz=args.hz, diff=args.diff, timing=args.timing, spin_shaping=args.spin_shaping, spin_free_press=args.spin_free_press, tail_miss=args.tail_miss,
                                                                  pool=sorted(glob.glob(os.path.join(args.pool, "*.osu"))) if args.pool else (),
                                                                  pool_frac=args.pool_frac, k=args.k), **vec)
    # small MLP: CPU is faster than GPU for rollouts; only the updates gain from the GPU
    cls, device = (HybridPPO, "cuda") if args.device == "hybrid" else (PPO, "cpu")
    lr = get_linear_fn(3e-4, 0.0, 1.0) if args.lr_decay else 3e-4  # 3e-4: SB3's default
    model = cls("MlpPolicy", env, verbose=1, device=device, seed=args.seed, learning_rate=lr, **hparams)
    model.set_logger(configure(run_dir, ["stdout", "csv"]))
    model.learn(int(args.steps), callback=[
        CheckpointCallback(500_000 // args.envs, os.path.join(run_dir, "ckpt")),
        RealMapEval(args.eval_map, 250_000, args.hz, args.diff, args.pool, k=args.k),
    ])
    model.save(os.path.join(run_dir, "model"))
