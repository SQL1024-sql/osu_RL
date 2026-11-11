import multiprocessing as mp

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper


def _worker(remote, fns):
    venv = DummyVecEnv(fns.var)
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            remote.send(venv.step(data))
        elif cmd == "reset":
            venv._seeds = data  # per-env seeds, as the parent VecEnv assigned them
            remote.send(venv.reset())
        elif cmd == "spaces":
            remote.send((venv.observation_space, venv.action_space))
        elif cmd == "call":
            name, args, indices = data
            remote.send(getattr(venv, name)(*args, indices=indices))
        elif cmd == "close":
            venv.close()
            remote.close()
            break


class GroupedSubprocVecEnv(VecEnv):
    """SubprocVecEnv with several envs per worker process: one policy call covers many envs while the number of
    processes stays at the number of fast cores (SB3's per-step overhead, not the env, dominates training time)."""

    def __init__(self, env_fns, workers=8):
        self.groups = [g.tolist() for g in np.array_split(np.arange(len(env_fns)), workers)]
        ctx = mp.get_context("spawn")
        self.remotes, self.processes = [], []
        for g in self.groups:
            remote, work = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(work, CloudpickleWrapper([env_fns[i] for i in g])), daemon=True)
            p.start()
            work.close()
            self.remotes.append(remote)
            self.processes.append(p)
        self.remotes[0].send(("spaces", None))
        super().__init__(len(env_fns), *self.remotes[0].recv())

    def reset(self):
        for r, g in zip(self.remotes, self.groups):
            r.send(("reset", [self._seeds[i] for i in g]))
        obs = np.concatenate([r.recv() for r in self.remotes])
        self._reset_seeds()
        self._reset_options()
        return obs

    def step_async(self, actions):
        for r, g in zip(self.remotes, self.groups):
            r.send(("step", actions[g[0]:g[-1] + 1]))

    def step_wait(self):
        results = [r.recv() for r in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return np.concatenate(obs), np.concatenate(rews), np.concatenate(dones), [i for part in infos for i in part]

    def _call(self, name, args, indices):
        wanted = set(self._get_indices(indices))
        out = []
        for r, g in zip(self.remotes, self.groups):
            local = [j for j, i in enumerate(g) if i in wanted]
            if local:
                r.send(("call", (name, args, local)))
                out += r.recv() or []  # set_attr returns None
        return out

    def get_attr(self, attr_name, indices=None):
        return self._call("get_attr", (attr_name,), indices)

    def set_attr(self, attr_name, value, indices=None):
        self._call("set_attr", (attr_name, value), indices)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        assert not method_kwargs, "keyword arguments are not forwarded"
        return self._call("env_method", (method_name, *method_args), indices)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return self._call("env_is_wrapped", (wrapper_class,), indices)

    def close(self):
        for r in self.remotes:
            r.send(("close", None))
        for p in self.processes:
            p.join()
