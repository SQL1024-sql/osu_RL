"""Runs one seed of an experiment config end to end, so anyone can reproduce or extend a hypothesis test.

    python -m state.run experiments/015-tail-miss.json --seed 2
    python -m state.run experiments/015-tail-miss.json --seed 2 --dry-run     # print the commands only
train.py with the config's settings -> checkpoint picked on the 25-map validation set (select_ckpt.py, H6) -> the
26 real test maps and 60 HF test maps -> results/experiments/<id>/seed<N>/summary.json (+ the picked model).
The criteria in a config are the machine-checkable part of the pre-registration in RESEARCH.md; they are meant for the
mean over seeds, so a single seed's pass / fail is only a hint.
"""
import argparse
import glob
import json
import operator
import os
import shutil
import subprocess
import sys

OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}


def train_flags(train):
    flags = []
    for k, v in train.items():
        flag = "--" + k.replace("_", "-")
        if v is True:
            flags.append(flag)
        elif v is not False and v is not None:
            flags += [flag, str(v)]
    return flags


def metric(summary, name):
    part, key = name.split(".")
    return summary[part][key]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="hybrid", choices=["cpu", "hybrid"], help="hybrid needs CUDA torch (see README)")
    p.add_argument("--steps", type=float, help="override the config's steps (e.g. a quick smoke test)")
    p.add_argument("--every", type=int, default=1_000_000, help="checkpoint spacing for selection")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = json.load(open(args.config, encoding="utf-8"))
    train = dict(cfg["train"], **({"steps": args.steps} if args.steps else {}))
    name = f"{cfg['id']}-s{args.seed}"
    cmd = [sys.executable, "-m", "state.train", *train_flags(train), "--seed", str(args.seed), "--device", args.device, "--name", name]
    print(" ".join(cmd))
    if args.dry_run:
        sys.exit(0)

    subprocess.run(cmd, check=True)
    run_dir = sorted(glob.glob(os.path.join("runs", f"*-{name}")))[-1]
    os.makedirs(os.path.join("results", os.path.basename(run_dir)), exist_ok=True)
    subprocess.run([sys.executable, "-m", "state.select_ckpt", run_dir, "--hz", str(train["hz"]), "--every", str(args.every)], check=True)
    sel = json.load(open(os.path.join("results", os.path.basename(run_dir), "select25.json")))

    out = os.path.join("results", "experiments", cfg["id"], f"seed{args.seed}")
    os.makedirs(out, exist_ok=True)
    shutil.copy(os.path.join(run_dir, "ckpt", sel["best"]["ckpt"]), os.path.join(out, "model.zip"))
    summary = dict(experiment=cfg["id"], seed=args.seed, run=os.path.basename(run_dir), ckpt=sel["best"]["ckpt"],
                   val=sel["best"], real=sel["real"], hf=sel["hf"],
                   config=json.load(open(os.path.join(run_dir, "config.json"))))
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(f"-> {out}/summary.json")
    for c in cfg["criteria"]:
        v = metric(summary, c["metric"])
        print(f"  {c['metric']} = {v:.4g}  {c['op']} {c['value']}: {'pass' if OPS[c['op']](v, c['value']) else 'fail'} (this seed only)")
