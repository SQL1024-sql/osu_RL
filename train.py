import argparse
import json
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv

import numpy as np
import torch

import env as env_mod
from beatmap import parse_osu, random_beatmap
from env import OsuEnv
from play import run


VAL_SEEDS = range(10_000, 10_005)  # fixed random maps for picking checkpoints; the real map stays a pure test


class RealMapEval(BaseCallback):
    """Every `freq` steps: plays the real test map (eval/*) and the validation maps (val/*), logs judgement stats."""

    def __init__(self, map_path, freq, hz, diff):
        super().__init__()
        self.env = OsuEnv(parse_osu(map_path), hz=hz)
        self.val_envs = [OsuEnv(random_beatmap(np.random.default_rng(s), n=100, diff=diff), hz=hz) for s in VAL_SEEDS]
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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hz", type=int, default=60)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eval-map", default="maps/bakenohana.osu")
    p.add_argument("--name", default="")
    p.add_argument("--diff", default="narrow", choices=["narrow", "wide"], help="CS / AR / OD range of training maps")
    args = p.parse_args()
    torch.set_num_threads(4)  # the default (all cores) fights the env worker processes: 4 was fastest on i7-13620H with 8 envs

    run_dir = os.path.join("runs", time.strftime("%Y%m%d-%H%M") + (f"-{args.name}" if args.name else ""))
    hparams = dict(n_steps=1024, batch_size=2048, gamma=args.gamma, policy_kwargs=dict(net_arch=[256, 256]))
    os.makedirs(run_dir)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        consts = {k: getattr(env_mod, k) for k in ("MAX_CURSOR_SPEED", "K", "MISS_WINDOW", "REWARD", "NESTED_REWARD", "SPIN_REWARD", "CLICK_PENALTY", "SHAPING", "CURSOR_MARGIN", "LOOKAHEAD_MS")}
        json.dump(dict(args=vars(args), ppo=hparams, env=consts), f, indent=2, default=str)

    env = make_vec_env(OsuEnv, n_envs=args.envs, vec_env_cls=SubprocVecEnv, seed=args.seed, env_kwargs=dict(hz=args.hz, diff=args.diff))
    # small MLP: CPU is faster than GPU here
    model = PPO("MlpPolicy", env, verbose=1, device="cpu", seed=args.seed, **hparams)
    model.set_logger(configure(run_dir, ["stdout", "csv"]))
    model.learn(int(args.steps), callback=[
        CheckpointCallback(500_000 // args.envs, os.path.join(run_dir, "ckpt")),
        RealMapEval(args.eval_map, 250_000, args.hz, args.diff),
    ])
    model.save(os.path.join(run_dir, "model"))
