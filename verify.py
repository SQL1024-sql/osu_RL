"""Replays plays in osu!lazer's real judgement engine (headless) and compares every judgement with the simulator.

    python verify.py maps/test/*.osu --model runs/y/model --hz 120 --out runs/y/verify     # a model on real maps
    python verify.py maps/test/*.osu --noise 0 0.3 1 --hz 120 --out runs/verify/real     # scripted player on real maps
    python verify.py --fuzz 30 --noise 0 0.3 1 --hz 120 --out runs/verify/fuzz/f        # random maps
"""
import argparse
import os
import subprocess
from collections import Counter

import numpy as np
from stable_baselines3 import PPO

from beatmap import SLIDER, SPINNER, parse_osu, random_osu
from env import OsuEnv, k_from_obs
from play import oracle, run
from replay import write_osr

OSU_REPO = os.path.abspath(os.environ.get("OSU_REPO", "../ppy-osu"))
DOTNET_ROOT = os.environ.get("DOTNET_ROOT") or os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\dotnet10")
DOTNET = os.path.join(DOTNET_ROOT, "dotnet.exe" if os.name == "nt" else "dotnet")
SIM_TO_LAZER = {300: "Great", 100: "Ok", 50: "Meh", 0: "Miss"}
NESTED_KIND = {"SliderHeadCircle": "head", "SliderTick": "tick", "SliderRepeat": "repeat", "SliderTailCircle": "tail"}
OBJECT_KIND = {0: "HitCircle", SLIDER: "Slider", SPINNER: "Spinner"}


def lazer_judge(jobs, jobs_file):
    """jobs: [(map.osu, replay.osr, out.csv)] -> [(header, rows)], all in one headless lazer run."""
    with open(jobs_file, "w") as f:
        f.write("\n".join("|".join(os.path.abspath(p) for p in job) for job in jobs))
    subprocess.run([DOTNET, "test", "osu.Game.Rulesets.Osu.Tests", "--filter", "FullyQualifiedName~TestSceneOsuRLReplay"],
                   cwd=OSU_REPO, env=dict(os.environ, DOTNET_ROOT=DOTNET_ROOT, OSU_RL_JOBS=os.path.abspath(jobs_file)), check=True, capture_output=True)
    out = []
    for *_, csv in jobs:
        with open(csv) as f:
            header = dict(kv.split("=") for kv in f.readline()[2:].strip().split(","))
            out.append((header, [line.strip().split(",") for line in f]))
    return out


def compare(bm, env, header, rows):
    agree, diffs = Counter(), []

    def check(kind, same, info):
        agree[kind, same] += 1
        if not same:
            diffs.append((kind, *info))

    lazer_obj = {(round(float(t)), kind): res for t, kind, res, _ in rows if kind in OBJECT_KIND.values()}
    for i, _, score in env.judgements:
        kind = OBJECT_KIND[bm.kinds[i]]
        res = lazer_obj.get((round(bm.objects[i, 2]), kind))
        check(kind, res == SIM_TO_LAZER[score], (i, score, res))

    lazer_nested = {(round(float(t)), NESTED_KIND[kind]): res.endswith("Hit") for t, kind, res, _ in rows if kind in NESTED_KIND}
    sim_spin = Counter()
    for time, kind, hit in env.nested_log:
        if kind == "spin":
            sim_spin[round(time)] += hit
        else:
            got = lazer_nested.get((round(time), kind))
            check(kind, got == hit, (round(time, 1), hit, got))
    for i, sp in bm.spinners.items():
        lazer_hits = sum(1 for t, kind, res, _ in rows if kind.startswith("SpinnerTick") or kind.startswith("SpinnerBonusTick")
                         if sp.start <= float(t) <= sp.end and not res.endswith("Miss"))
        check("spins", sim_spin[round(sp.start)] == lazer_hits, (i, sim_spin[round(sp.start)], lazer_hits))
    check("max_combo", env.max_combo == int(header["max_combo"]), (env.max_combo, header["max_combo"]))
    return agree, diffs


def report(agree, diffs):
    for kind in ("HitCircle", "Slider", "Spinner", "head", "tick", "repeat", "tail", "spins", "max_combo"):
        n = agree[kind, True] + agree[kind, False]
        if n:
            print(f"{kind:>10}: {agree[kind, True]}/{n} agree")
    for d in diffs[:20]:
        print("  mismatch:", d)


def verify(plays, hz, out, k=None):
    """plays: [(name, map path, policy_fn)], policy_fn(env) -> policy. Plays each in the simulator, writes the replay,
    judges all replays in one lazer run. Returns (agree Counter, diffs, [(name, bm, env, lazer header)])."""
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    done = []
    for name, map_path, policy_fn in plays:
        bm = parse_osu(map_path)
        cur_k = k if k is not None else getattr(policy_fn, "k", None)
        env = OsuEnv(bm, hz=hz, k=cur_k)
        frames, counts, _ = run(env, policy_fn(env))
        write_osr(f"{out}_{name}.osr", map_path, frames, counts, env.max_combo)
        done.append((name, bm, env, map_path))
    results = lazer_judge([(m, f"{out}_{name}.osr", f"{out}_{name}_lazer.csv") for name, _, _, m in done], out + "_jobs.txt")
    agree, diffs, summary = Counter(), [], []
    for (name, bm, env, _), (header, rows) in zip(done, results):
        a, d = compare(bm, env, header, rows)
        agree += a
        diffs += [(name, *x) for x in d]
        summary.append((name, bm, env, header))
    return agree, diffs, summary


def oracle_policy(noise, seed):
    rng = np.random.default_rng(seed)
    return lambda e: (lambda o: oracle(e, noise, rng))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("maps", nargs="*")
    p.add_argument("--model", default="oracle", help='model path, or "oracle" for the scripted player')
    p.add_argument("--noise", type=float, nargs="+", default=[0.0], help="oracle noise levels")
    p.add_argument("--hz", type=int, default=120)
    p.add_argument("--out", required=True, help="output prefix")
    p.add_argument("--fuzz", type=int, default=0, help="number of random maps")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    maps = [(os.path.splitext(os.path.basename(m))[0], m) for m in args.maps]
    for seed in range(args.fuzz):
        path = f"{args.out}_{seed}.osu"
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(random_osu(np.random.default_rng(seed), n=60)))
        maps.append((str(seed), path))

    if args.model == "oracle":
        plays = [(f"{name}_n{noise}", path, oracle_policy(noise, i)) for i, (name, path) in enumerate(maps) for noise in args.noise]
        k = None
    else:
        model = PPO.load(args.model, device="cpu")
        k = k_from_obs(model.observation_space.shape[0])
        plays = [(name, path, lambda e: (lambda o: model.predict(o, deterministic=True)[0])) for name, path in maps]
    agree, diffs, _ = verify(plays, args.hz, args.out, k=k)
    report(agree, diffs)
