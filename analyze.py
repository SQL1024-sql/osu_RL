"""Error analysis: where does a model lose judgements? Counts misses / 100s / 50s / slider breaks by situation.

    python analyze.py results/<run>/best_model.zip maps/test/*.osu --hz 120
"""
import argparse
import collections
import math

from stable_baselines3 import PPO

from beatmap import SLIDER, SPINNER, parse_osu
from env import OsuEnv, k_from_obs
from evaluate import model_policy
from play import run


def situation(bm, i):
    """Coarse label for object i from its neighbourhood: what a player would call it."""
    if bm.kinds[i] == SPINNER:
        return "spinner"
    x, y, t = bm.objects[i]
    if i == 0:
        return "first object"
    px, py, pt = bm.objects[i - 1]
    gap, dist = t - bm.ends[i - 1], math.hypot(x - px, y - py)
    if bm.kinds[i - 1] == SPINNER:
        return "after spinner"
    if gap < 110 and dist < 2 * bm.radius:
        return "stream / stack"
    if dist > 3 * bm.radius and dist / max(gap, 1) > 1.2:
        return "jump"
    return "slider" if bm.kinds[i] == SLIDER else "normal circle"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("maps", nargs="+")
    p.add_argument("--hz", type=int, default=120)
    args = p.parse_args()
    model = PPO.load(args.model, device="cpu")
    k = k_from_obs(model.observation_space.shape[0])
    judged, objects = collections.Counter(), collections.Counter()
    nested_miss, nested_all = collections.Counter(), collections.Counter()
    for path in args.maps:
        bm = parse_osu(path)
        env = OsuEnv(bm, hz=args.hz, k=k)
        run(env, model_policy(model)(env))
        for i, _, score in env.judgements:
            s = situation(bm, i)
            objects[s] += 1
            if score != 300:
                judged[(s, score)] += 1
        for _, kind, hit in env.nested_log:
            nested_all[kind] += 1
            nested_miss[kind] += not hit
    print(f"{'situation':16s} {'objects':>8s} {'miss':>6s} {'50':>5s} {'100':>5s}")
    for s, n in objects.most_common():
        print(f"{s:16s} {n:8d} {judged[(s, 0)]:6d} {judged[(s, 50)]:5d} {judged[(s, 100)]:5d}")
    print("slider parts / spinner ticks missed:", {k: f"{nested_miss[k]}/{nested_all[k]}" for k in nested_all})
