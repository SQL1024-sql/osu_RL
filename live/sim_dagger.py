"""Fresh rendered DAgger data for live.finetune_real: the model playing now drives plays in the simulator (rendered
frames, no game, faster than real time) and the teacher labels every step, on many more maps than the real game gets
through in a round.

Until now the rendered half of every training batch came from results/experiments/020-student/seed0/plays7.pkl: plays
of the student as it was on 2026-11-07, never renewed, so half of what each candidate learnt was about states an old
student got into.

    python -m live.sim_dagger --ckpt runs/improve/r001/best.pt --n 200 --out runs/improve/sim/r034.pkl

Maps: half from the loop's training maps of every tier unlocked so far (runs/improve/state.json), half drawn from the
HF training split (maps/hf) up to the current tier's top star rating; never a gate map.
"""
import argparse
import csv
import glob
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

import vision.student as S
from live.vision_play import CLICK_TH, MIN_HOLD

TIER_STARS = [4.5, 5.0, 5.5, 6.0]  # top star rating of each of live.improve's tiers


def pick_maps(n, rng, state_path="runs/improve/state.json"):
    from live import improve as I
    tier = json.load(open(state_path)).get("tier", 0) if os.path.exists(state_path) else 0
    gate = {I.md5(p) for _, _, g in I.TIERS for p in I.maps_of(g)}
    loop = I.maps_of([g for _, tr, _ in I.TIERS[: tier + 1] for g in tr])
    hf = []
    for r in csv.DictReader(open("maps/hf/index.csv", encoding="utf-8")):
        try:
            st, ln, ar, cs = float(r["stars"]), float(r["hit_length"]), float(r["ar"]), float(r["cs"])
        except ValueError:
            continue
        path = f"maps/hf/{r['split']}/{r['beatmap_id']}.osu"
        if r["split"] == "train" and 3.0 <= st <= TIER_STARS[tier] and 40 <= ln <= 180 and 3.0 <= cs <= 5.0 and ar >= 7 \
                and os.path.exists(path):
            hf.append(path)
    a = [str(x) for x in rng.choice(loop, min(len(loop), n // 2), replace=False)]
    b = [p for p in rng.choice(hf, min(len(hf), 2 * (n - len(a))), replace=False) if I.md5(p) not in gate][: n - len(a)]
    return a + b, tier


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n", type=int, default=200, help="plays")
    p.add_argument("--out", required=True)
    p.add_argument("--procs", type=int, default=6, help="simulator processes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smooth", type=float, default=0.5)
    p.add_argument("--hold-press", type=float, default=8.0)
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rng = np.random.default_rng(a.seed)
    maps_list, tier = pick_maps(a.n, rng)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # the live player's rules, so the simulated plays go where live ones would
    S.CLICK_TH, S.MIN_HOLD, S.MAP_KEYS = CLICK_TH, MIN_HOLD, True
    S.SMOOTH, S.HOLD_PRESS = a.smooth, a.hold_press
    model = S.make_model(a.ckpt, dev)
    S.FOVEA = model.fovea_side or 0  # student.build renders the close-ups a fovea model takes
    maps = S.Maps(dev)
    t0 = time.time()
    jobs = [(m, float(rng.uniform(0, S.MAX_LAT)), int(rng.integers(1 << 30))) for m in maps_list]
    plays = S.rollouts(jobs, model, 1.0, a.procs, maps)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(plays, f)
    accs = [p["lazer_acc"] for p in plays]
    print(f"{len(plays)} simulated plays (tier {tier + 1}) with {a.ckpt} in {time.time() - t0:.0f} s, "
          f"{sum(len(p['t']) for p in plays)} labelled steps, simulator accuracy {np.mean(accs):.2%} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
