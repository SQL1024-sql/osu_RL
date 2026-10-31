"""Plays a run's best model on a fixed sample of the held-out HF test maps (maps/hf/test), next to autoplay.

    python eval_hf.py results/20261031-1933-timing120 --hz 120
Writes hf_test.json into that results folder. Full maps, simulator only (osu!lazer re-judging stays on maps/test).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from stable_baselines3 import PPO

from beatmap import parse_osu
from evaluate import model_policy, pct, play
from play import oracle

N_MAPS = 60

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("results_dir")
    p.add_argument("--hz", type=int, default=120)
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    files = sorted(glob.glob("maps/hf/test/*.osu"))
    files = [str(f) for f in np.random.default_rng(20_000).choice(files, min(N_MAPS, len(files)), replace=False)]
    model = PPO.load(os.path.join(args.results_dir, "best_model.zip"), device="cpu")
    rows = []
    for f in files:
        bm = parse_osu(f)
        m, o = play(bm, model_policy(model), args.hz), play(bm, lambda e: (lambda ob: oracle(e)), args.hz)
        rows.append(dict(map=os.path.basename(f)[:-4], title=bm.title, objects=len(bm.objects), spinners=len(bm.spinners),
                         cs=bm.cs, ar=bm.ar, od=bm.od, model=m, autoplay=o))
        print(f"hf {rows[-1]['map']}: model {pct(m['acc'])} miss {m['miss']} | autoplay {pct(o['acc'])}")
    n = sum(r["objects"] for r in rows)
    summary = dict(maps=len(rows), model_acc=float(np.mean([r["model"]["acc"] for r in rows])),
                   model_acc_median=float(np.median([r["model"]["acc"] for r in rows])), fc=sum(r["model"]["fc"] for r in rows),
                   miss_per_100=100 * sum(r["model"]["miss"] for r in rows) / n,
                   spinner_share_of_misses=sum(min(r["spinners"], r["model"]["miss"]) for r in rows) / max(1, sum(r["model"]["miss"] for r in rows)),
                   autoplay_acc=float(np.mean([r["autoplay"]["acc"] for r in rows])))
    json.dump(dict(summary=summary, maps=rows), open(os.path.join(args.results_dir, "hf_test.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
