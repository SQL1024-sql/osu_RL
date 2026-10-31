"""Full evaluation of a training run, written to results/<run name>/.

1. every checkpoint is scored on the validation maps (fixed random maps) -> best checkpoint
2. the best checkpoint plays the test set: real maps (maps/test/*.osu) + unseen random maps, next to the scripted
   autoplay baseline
3. the real-map plays are re-judged by osu!lazer (verify.py) and the replays are kept

    python evaluate.py runs/20261031-0300-full120 --hz 120
"""
import argparse
import glob
import json
import os
import re
import shutil

import numpy as np
from stable_baselines3 import PPO

from beatmap import SLIDER, SPINNER, parse_osu, random_beatmap
from env import OsuEnv
from play import oracle, run
from train import VAL_SEEDS
from verify import verify

TEST_SEEDS = range(20_000, 20_050)


def max_combo(bm):
    """Combo of a full combo: circles, spinners, and per slider its head, ticks, repeats and the slider itself."""
    nested = sum(sum(k != "tail" for _, k in s.nested) for s in bm.sliders.values())
    return int((bm.kinds != SLIDER).sum() + 2 * len(bm.sliders) + nested)


def play(bm, policy_fn, hz):
    env = OsuEnv(bm, hz=hz)
    _, counts, acc = run(env, policy_fn(env))
    return dict(acc=acc, n300=counts[300], n100=counts[100], n50=counts[50], miss=counts[0], combo=env.max_combo,
                max_combo=max_combo(bm), fc=counts[0] == 0 and env.max_combo == max_combo(bm))


def model_policy(model):
    return lambda e: (lambda o: model.predict(o, deterministic=True)[0])


def table(rows, cols):
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "".join("| " + " | ".join(str(r[c]) for c in cols) + " |\n" for r in rows)


def pct(x):
    return f"{100 * x:.2f}%"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--hz", type=int, default=120)
    p.add_argument("--maps", default="maps/test/*.osu", help="glob of real test maps")
    p.add_argument("--random", type=int, default=len(TEST_SEEDS), help="number of random test maps")
    args = p.parse_args()
    name = os.path.basename(os.path.normpath(args.run_dir))
    out = os.path.join("results", name)
    os.makedirs(os.path.join(out, "replays"), exist_ok=True)

    # 1. checkpoint selection on validation maps
    val_maps = [random_beatmap(np.random.default_rng(s), n=100) for s in VAL_SEEDS]
    ckpts = sorted(glob.glob(os.path.join(args.run_dir, "ckpt", "*.zip")), key=lambda f: int(re.findall(r"(\d+)_steps", f)[0]))
    ckpts.append(os.path.join(args.run_dir, "model.zip"))
    val = []
    for c in ckpts:
        model = PPO.load(c, device="cpu")
        r = [play(bm, model_policy(model), args.hz) for bm in val_maps]
        steps = re.findall(r"(\d+)_steps", c)
        val.append(dict(ckpt=os.path.basename(c), steps=int(steps[0]) if steps else -1,
                        acc=float(np.mean([x["acc"] for x in r])), miss=sum(x["miss"] for x in r)))
        print(f"val {val[-1]['ckpt']}: acc {pct(val[-1]['acc'])} miss {val[-1]['miss']}")
    best = max(range(len(val)), key=lambda i: (val[i]["acc"], -val[i]["miss"]))
    best_path = ckpts[best]
    shutil.copy(best_path, os.path.join(out, "best_model.zip"))
    model = PPO.load(best_path, device="cpu")
    print("best:", val[best])

    # 2. test set, model vs scripted autoplay
    real = sorted(glob.glob(args.maps))
    test = []
    for path in real:
        bm = parse_osu(path)
        m, o = play(bm, model_policy(model), args.hz), play(bm, lambda e: (lambda ob: oracle(e)), args.hz)
        test.append(dict(map=os.path.splitext(os.path.basename(path))[0], kind="real", objects=len(bm.objects),
                         sliders=len(bm.sliders), spinners=len(bm.spinners), cs=bm.cs, ar=bm.ar, od=bm.od, model=m, autoplay=o))
        print(f"test {test[-1]['map']}: model {pct(m['acc'])} miss {m['miss']} | autoplay {pct(o['acc'])}")
    for s in TEST_SEEDS[:args.random]:
        bm = random_beatmap(np.random.default_rng(s), n=100)
        test.append(dict(map=f"random_{s}", kind="random", objects=len(bm.objects), sliders=len(bm.sliders),
                         spinners=len(bm.spinners), cs=bm.cs, ar=bm.ar, od=bm.od,
                         model=play(bm, model_policy(model), args.hz), autoplay=play(bm, lambda e: (lambda ob: oracle(e)), args.hz)))

    # 3. lazer re-judges the real-map plays; replays are kept
    agree, diffs, summary = verify([(os.path.splitext(os.path.basename(p))[0], p, model_policy(model)) for p in real],
                                   args.hz, os.path.join(out, "replays", "model"))
    lazer = {n: dict(acc=float(h["accuracy"]), combo=int(h["max_combo"])) for n, _, _, h in summary}
    n_checked = sum(agree.values())
    n_agree = sum(v for (k, same), v in agree.items() if same)
    for f in glob.glob(os.path.join(out, "replays", "*")):
        if not f.endswith(".osr"):
            os.remove(f)

    json.dump(dict(val=val, best=val[best], test=test, lazer=lazer, lazer_agreement=[n_agree, n_checked],
                   lazer_mismatches=[[str(x) for x in d] for d in diffs]), open(os.path.join(out, "results.json"), "w"), indent=1)
    if os.path.exists(os.path.join(args.run_dir, "progress.csv")):
        os.system(f'python plot.py "{args.run_dir}" > nul')
        shutil.copy(os.path.join(args.run_dir, "curves.png"), os.path.join(out, "curves.png"))
    shutil.copy(os.path.join(args.run_dir, "config.json"), os.path.join(out, "config.json"))

    # markdown summary
    def summarize(kind):
        rows = [t for t in test if t["kind"] == kind]
        f = lambda who, k: np.mean([t[who][k] for t in rows])
        return dict(set=f"{kind} ({len(rows)} maps)",
                    model_acc=pct(f("model", "acc")), model_fc=f"{sum(t['model']['fc'] for t in rows)}/{len(rows)}",
                    model_miss=f"{f('model', 'miss'):.1f}", autoplay_acc=pct(f("autoplay", "acc")),
                    autoplay_fc=f"{sum(t['autoplay']['fc'] for t in rows)}/{len(rows)}")

    real_rows = [dict(map=t["map"], objects=t["objects"], sliders=t["sliders"], spinners=t["spinners"],
                      diff=f"CS{t['cs']:g} AR{t['ar']:g} OD{t['od']:g}", acc=pct(t["model"]["acc"]),
                      judgements=f"{t['model']['n300']}/{t['model']['n100']}/{t['model']['n50']}/{t['model']['miss']}",
                      combo=f"{t['model']['combo']}/{t['model']['max_combo']}", lazer_acc=pct(lazer[t["map"]]["acc"]),
                      autoplay=pct(t["autoplay"]["acc"])) for t in test if t["kind"] == "real"]
    md = (f"# {name}\n\nbest checkpoint: `{val[best]['ckpt']}` (validation acc {pct(val[best]['acc'])}, {val[best]['miss']} misses)\n\n"
          f"## Test set\n\n{table([summarize('real'), summarize('random')], ['set', 'model_acc', 'model_fc', 'model_miss', 'autoplay_acc', 'autoplay_fc'])}\n"
          f"acc = osu!stable-style accuracy over objects (300/100/50/miss). autoplay = scripted perfect player, the ceiling.\n\n"
          f"## Real maps\n\n{table(real_rows, ['map', 'objects', 'sliders', 'spinners', 'diff', 'acc', 'judgements', 'combo', 'lazer_acc', 'autoplay'])}\n"
          f"judgements = 300/100/50/miss. lazer_acc = accuracy osu!lazer shows for the same replay (it also weights slider ticks).\n\n"
          f"## Simulator vs osu!lazer\n\n{n_agree}/{n_checked} judgements identical on the real-map replays.\n\n"
          f"## Checkpoints (validation)\n\n{table([dict(v, acc=pct(v['acc'])) for v in val], ['ckpt', 'acc', 'miss'])}")
    open(os.path.join(out, "results.md"), "w", encoding="utf-8").write(md)
    print(md)
