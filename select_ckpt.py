"""Re-picks a run's checkpoint with a bigger, harder validation set (20 windows of held-out HF val maps + the 5 random
maps), then scores only that checkpoint on the test sets. Selection never looks at test maps.

    python select_ckpt.py runs/<run> --hz 120
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from stable_baselines3 import PPO

from beatmap import parse_osu, window
from evaluate import model_policy, pct, play
from train import val_maps

N_HF_VAL = 20

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--hz", type=int, default=120)
    p.add_argument("--every", type=int, default=1_000_000, help="checkpoint spacing to consider (steps)")
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    run_args = json.load(open(os.path.join(args.run_dir, "config.json")))["args"]
    classic = not run_args.get("room", False)  # judge everything by the rules the run trained under
    key = "acc" if classic else "lazer_acc"  # a room scores by lazer's accuracy
    rng = np.random.default_rng(10_100)  # not the training val windows' rng
    files = sorted(glob.glob("maps/hf/val/*.osu"))
    vmaps = val_maps(run_args.get("diff", "narrow"))
    for f in rng.choice(files, N_HF_VAL, replace=False):
        bm = parse_osu(f)
        vmaps.append(window(bm, int(rng.integers(max(1, len(bm.objects) - 99))), 100))
    ckpts = sorted((c for c in glob.glob(os.path.join(args.run_dir, "ckpt", "*.zip")) if int(re.findall(r"(\d+)_steps", c)[0]) % args.every == 0),
                   key=lambda c: int(re.findall(r"(\d+)_steps", c)[0]))
    val = []
    for c in ckpts:
        model = PPO.load(c, device="cpu")
        r = [play(bm, model_policy(model), args.hz, classic=classic) for bm in vmaps]
        val.append(dict(ckpt=os.path.basename(c), acc=float(np.mean([x[key] for x in r])), miss=sum(x["miss"] for x in r)))
        print(f"val25 {val[-1]['ckpt']}: acc {pct(val[-1]['acc'])} miss {val[-1]['miss']}")
    best = max(val, key=lambda v: (v["acc"], -v["miss"]))
    model = PPO.load(os.path.join(args.run_dir, "ckpt", best["ckpt"]), device="cpu")
    real = [play(parse_osu(f), model_policy(model), args.hz, classic=classic) for f in sorted(glob.glob("maps/test/*.osu"))]
    hf_files = sorted(glob.glob("maps/hf/test/*.osu"))
    hf_files = [str(f) for f in np.random.default_rng(20_000).choice(hf_files, min(60, len(hf_files)), replace=False)]  # same sample as eval_hf.py
    hf = [play(parse_osu(f), model_policy(model), args.hz, classic=classic) for f in hf_files]
    out = dict(best=best, val=val,
               real=dict(acc=float(np.mean([x["acc"] for x in real])), lazer_acc=float(np.mean([x["lazer_acc"] for x in real])), **{k: int(sum(x[k] for x in real)) for k in
                                                                         ("fc", "miss", "n100", "head_miss", "tail_miss", "spinner_miss")}),
               hf=dict(acc=float(np.mean([x["acc"] for x in hf])), lazer_acc=float(np.mean([x["lazer_acc"] for x in hf])), fc=sum(x["fc"] for x in hf), miss=sum(x["miss"] for x in hf)))
    name = os.path.basename(os.path.normpath(args.run_dir))
    json.dump(out, open(os.path.join("results", name, "select25.json"), "w"), indent=1)
    print(json.dumps(dict(best=best, real=out["real"], hf=out["hf"]), indent=1))
