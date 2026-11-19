"""Training loop v2 (RESEARCH.md, "設計 — 迴圈訓練 v2"): one cycle is collect -> label -> train -> calibrate ->
offline filter -> real gate -> promote, on fork copies 1-3 (copy 0 stays with the stream). It runs the steps that
were done by hand on 2026-11-18 (Exp 033-035) and keeps what made those work: every close-up recording used in turn
(live.finetune_real --rotate), candidates screened by live.offline_eval, judged in a Latin square (live.ablate) map by
map, promoted only if the paired gain is clear (mean - 2 se > 0 and at least 4 of 6 maps; 5 of 6 until
cycle 3, which held back two +0.8 candidates at 4/6).

    python -m live.cycle                       # cycles until Ctrl+C
    python -m live.cycle --cycles 1 --no-collect --dry-gate   # train and screen only (no game needed)
    python -m live.cycle --recipe --aim-rule oracle            # a recipe change: extra live.finetune_real arguments

State: runs/cycle/state.json (the model playing now), one line per cycle in runs/cycle/cycles.jsonl.
"""
import argparse
import glob
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np

STATE = os.path.join("runs", "cycle", "state.json")
LOG = os.path.join("runs", "cycle", "cycles.jsonl")
RECIPES = os.path.join("runs", "cycle", "recipes.json")  # read at the start of every cycle, if there
STOP_FILE = os.path.join("runs", "cycle", "STOP")
# 10 gate maps (tiers 3-5 from cycle 7; tiers 2-4 in cycle 6; tiers 3-4 until cycle 5): the se is over maps, so more rounds alone barely shrank it
# (cycle 5's confirmation: +0.67 +- 0.64 over 18 plays -> +0.50 +- 0.46 over 36)
GATE_MAPS = ["runs/maps/tier4/gate/*.osu", "runs/maps/tier5/gate/*.osu", "runs/maps/tier6/gate/*.osu"]
# moving up in stars (2026-11-18, cycle 7 on): tier 5 (6.0-6.5 stars, live.make_pool) takes half of each collection,
# and its 4 gate maps replace tier 2's (4.5-5 stars, near the ceiling at 95-97%)
# from cycle 9: tier 6 (6.5-7 stars) is the new one; its gate maps replace tier 3's (5-5.5 stars, 92-97%)
NEW_TIER_TRAIN = ["runs/maps/tier6/train/*.osu"]
NEW_FRAC = 0.5
CONFIRM_MIN = 0.3  # a gate leader this far ahead (points) but not clear gets three more rounds
SCREEN_SLACK = 0.01  # offline composite a candidate may fall short of the model playing and still be judged in the game
COPIES = [1, 2, 3]
PLAY = ["--click-th", "-0.75", "--min-hold", "5", "--hold-press", "8", "--smooth", "0.5", "--hold-th", "-1.5"]


def say(text):
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def load_state(a):
    if os.path.exists(STATE):
        return json.load(open(STATE))
    from live.stream import improve_current
    return dict(cycle=0, current=a.start or improve_current(), history=[])


def save_state(s):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    json.dump(s, open(STATE + ".tmp", "w"), indent=1)
    os.replace(STATE + ".tmp", STATE)


def train_maps(n):
    """n training maps: NEW_FRAC of them from the newest tier (the stars the student is moving up to), the rest from
    every tier before it (so it keeps what it learnt)."""
    from live.improve import TIERS, maps_of
    new = maps_of(NEW_TIER_TRAIN)
    old = [m for m in maps_of([g for _, tr, _ in TIERS for g in tr]) if m not in set(new)]
    k = min(len(new), int(round(n * NEW_FRAC)))
    return random.sample(new, k) + random.sample(old, min(n - k, len(old)))


def gate_maps():
    return sorted(glob.glob("runs/maps/tier3/gate/*.osu") + glob.glob("runs/maps/tier4/gate/*.osu"))


def collect(ckpt, n_per_copy, out):
    """The model playing now plays n_per_copy training maps on each copy, recorded. -> recordings with a score."""
    procs = []
    for inst in COPIES:
        maps = train_maps(n_per_copy)
        log = open(os.path.join(out, f"collect-copy{inst}.log"), "w", encoding="utf-8")
        cmd = [sys.executable, "-u", "-m", "live.collect", "--background", "--maps", *map(glob.escape, maps),
               "--plays", str(len(maps)), "--ckpt", ckpt, "--out", os.path.join(out, "plays"), *PLAY]
        procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                       env=dict(os.environ, OSU_RL_INSTANCE=str(inst), PYTHONIOENCODING="utf-8")), log))
        time.sleep(20)
    for p, log in procs:
        p.wait()
        log.close()
    from live.ablate import stale_steps, STALE_MAX
    recs = []
    for f in glob.glob(os.path.join(out, "plays", "*", "plays.jsonl")):
        for line in open(f, encoding="utf-8"):
            p = json.loads(line)
            if p.get("accuracy") is None or not p.get("rec") or (p.get("play_capture_fps") or 0) < 100 \
                    or (p.get("step_hz") or 0) < 100 or stale_steps(p) > STALE_MAX:
                continue
            recs.append(p["rec"])
    return recs


def label(recs):
    todo = [r for r in recs if not os.path.exists(r[:-4] + ".labels.npz") and not os.path.exists(r[:-4] + ".labels.failed")]
    for i in range(0, len(todo), 40):
        subprocess.run([sys.executable, "-m", "live.dagger_label", *todo[i:i + 40]], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return [r for r in recs if os.path.exists(r[:-4] + ".labels.npz")]


def training_recs():
    """Every labelled training-map recording with close-ups that played at full speed, newest first."""
    from live.improve import TIERS, maps_of, md5, has_fovea, step_hz
    gate = {md5(p) for _, _, g in TIERS for p in maps_of(g)}
    recs = glob.glob("runs/improve/plays/*/*-rec.npz") + glob.glob("runs/collect/*/*-rec.npz") + \
        glob.glob("runs/stream/plays/*/*-rec.npz") + glob.glob("runs/cycle/c*/plays/*/*-rec.npz")
    out = []
    for f in sorted(set(recs), key=os.path.getmtime, reverse=True):
        if not os.path.exists(f[:-4] + ".labels.npz") or not has_fovea(f):
            continue
        try:
            if str(np.load(f)["md5"]) in gate or step_hz(f[:-len("-rec.npz")] + ".npz") < 100:
                continue
        except Exception:
            continue
        out.append(f)
    return out


def held_out(current, n=8):
    """The newest labelled gate plays with close-ups (of any model): picks the training step."""
    from live.offline_eval import gate_recs
    return [r for r in gate_recs(60) if os.path.exists(r[:-4] + ".labels.npz")][:n]


def train(current, out, recipe, updates, lr, init=None):
    """init: the weights to start from (default the model playing; the lead candidate of the last cycle when it was
    ahead but not promoted). Calibration is always against the model playing."""
    init = init or current
    train_recs, val = training_recs(), held_out(current)
    cal = train_recs[:6]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out + "-recs.txt", "w", encoding="utf-8") as f:  # 300+ paths overflow a Windows command line
        f.write("\n".join(train_recs) + "\n")
    cmd = [sys.executable, "-u", "-m", "live.finetune_real", "@" + out + "-recs.txt", "--val", *val, "--clicks", "map",
           "--init", init, "--updates", str(updates), "--lr", str(lr), "--frames", "cpu", "--rotate", "2000",
           "--chunk-gb", "9", "--spin-frac", "0.1", "--spin-rule", "--aim-w", "6", "--aim-ms", "60", "--click-steps", "2",
           "--every", "1000", "--out", out, *recipe]
    say(f"training from {current} on {len(train_recs)} recordings ({' '.join(recipe) or 'the usual recipe'}) -> {out}")
    with open(out + "-train.log", "w", encoding="utf-8") as log:
        if subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode:
            return []
    cands = []
    try:
        best_at = json.load(open(os.path.join(out, "log.json"))).get("best_update")
    except Exception:
        best_at = None
    for name in ("best", "last", "ema"):
        if name == "best" and best_at == 0:  # never beat the weights it started from: it is the model playing
            continue
        if not os.path.exists(os.path.join(out, f"{name}.pt")):  # ema.pt only with --ema
            continue
        r = subprocess.run([sys.executable, "-m", "live.calibrate", os.path.join(out, f"{name}.pt"), "--ref", current,
                            "--recs", *cal, "--out", os.path.join(out, f"{name}_cal.pt")], capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode == 0:
            cands.append(os.path.join(out, f"{name}_cal.pt"))
    import vision.student as S
    for c in cands:  # the aim offset of the model they were trained from, until their own plays measure theirs
        S.save_aim_offset(c, S.aim_offset_of(init), f"inherited from {init}")
    return cands


def screen(current, cands, exclude):
    """live.offline_eval's composite (found - extra clicks per object) for the model playing now and each candidate.
    -> {ckpt: composite}"""
    import torch
    from live.dagger_label import MAP_GLOBS, map_index
    from live.offline_eval import evaluate, gate_recs
    index, dev = map_index(MAP_GLOBS), "cuda" if torch.cuda.is_available() else "cpu"
    recs = gate_recs(12, set(exclude))
    out = {}
    for ck in [current, *cands]:
        r = evaluate(ck, recs, index, dev)
        out[ck] = r["found"] - r["extra_per_play"] / (r["objects"] / len(recs))
        say(f"offline {ck}: found {r['found']:.3f}, extra {r['extra_per_play']:.1f} a play -> {out[ck]:.3f}")
    return out


def run_gate(current, cand, cand2=None):
    """live.ablate in a Latin square on copies 1-3: the model playing now, the candidate, and a second candidate (another
    recipe's) or the same one again. -> the run's folder, or None"""
    conds = [f"current={current}", f"cand={cand}", f"cand2={cand2 or cand}"]
    r = subprocess.run([sys.executable, "-u", "-m", "live.ablate", "--ckpt", current, "--conditions", *conds, "--maps", *GATE_MAPS],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    folder = r.stdout.splitlines()[0].split(":")[0].strip() if r.stdout else None
    if not folder or not os.path.isdir(folder):
        say(f"gate failed: {(r.stdout or '')[-300:]} {(r.stderr or '')[-300:]}")
        return None
    return folder


def paired(current, cands, folders):
    """Each candidate against the model playing now over every live.ablate run in folders (pooled: the arms are told
    apart by the checkpoint each played, from setup.json, so a confirmation run adds to the first).
    -> {cand: (mean paired gain in points over the maps, its se, maps won, maps)}"""
    from live.ablate import stale_steps, STALE_MAX
    acc = {}
    norm = lambda p: os.path.normcase(os.path.normpath(p))
    for folder in folders:
        setup = json.load(open(os.path.join(folder, "setup.json"), encoding="utf-8"))
        ck = {arm: norm(args[args.index("--ckpt") + 1]) for arm, args in setup["args"].items() if "--ckpt" in args}
        for f in glob.glob(os.path.join(folder, "r*-*", "*", "plays.jsonl")):
            arm = os.path.basename(os.path.dirname(os.path.dirname(f))).split("-", 1)[1]
            for line in open(f, encoding="utf-8"):
                p = json.loads(line)
                if p.get("accuracy") is not None and stale_steps(p) <= STALE_MAX and arm in ck:
                    acc.setdefault((ck[arm], os.path.basename(p["map"])), []).append(p["accuracy"])
    res, cur = {}, norm(current)
    for c in cands:
        maps = sorted({m for k, m in acc if k == cur and (norm(c), m) in acc})
        if len(maps) < 4:
            say(f"gate: only {len(maps)} maps scored by both the model playing and {c}: no decision for it")
            continue
        d = np.array([np.mean(acc[(norm(c), m)]) - np.mean(acc[(cur, m)]) for m in maps]) * 100
        res[c] = (float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))), int((d > 0).sum()), len(maps))
    return res


def passes(r):
    mean, se, won, n = r
    return mean - 2 * se > 0 and won >= n - max(2, n // 3)


def promote(state, cand, why, gate_folder=None):
    """The candidate plays from now on: in the loop, on stream / requests / rooms (runs/stream/state.json) and in
    live.improve's state, each backed up first. Its aim offset is measured again from its gate plays (live.aim_cal)."""
    import vision.student as S
    from live.aim_cal import measure
    if gate_folder:
        setup = json.load(open(os.path.join(gate_folder, "setup.json"), encoding="utf-8"))
        same = lambda a_, b_: os.path.normcase(os.path.normpath(a_)) == os.path.normcase(os.path.normpath(b_))
        arms = [arm for arm, args in setup["args"].items() if "--ckpt" in args and same(args[args.index("--ckpt") + 1], cand)]
        off, n = measure([d for arm in arms for d in glob.glob(os.path.join(gate_folder, f"r*-{arm}")) if os.path.isdir(d)])
        if n >= 500:
            S.save_aim_offset(cand, off, f"live.aim_cal from {n} clicks in its gate plays ({gate_folder})")
            say(f"aim offset of {cand}: {off.round(2)} ({n} clicks)")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dst = os.path.normpath(f"runs/stream/models/{stamp}-cycle{state['cycle']}.pt")
    shutil.copy(cand, dst)
    if os.path.exists(cand + ".aim.json"):
        shutil.copy(cand + ".aim.json", dst + ".aim.json")
    for path, key in (("runs/stream/state.json", dst), ("runs/improve/state.json", cand)):
        if os.path.exists(path):
            shutil.copy(path, path + f".bak-{stamp}")
            s = json.load(open(path))
            if "history" in s and path.startswith("runs/stream"):
                s["history"].append(dict(model=s["current"], until=time.time()))
                s["since"] = time.time()
            s["current"] = key
            json.dump(s, open(path, "w"), indent=1)
    state["history"].append(dict(cycle=state["cycle"], old=state["current"], new=cand, why=why, time=stamp))
    state["current"] = cand


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--cycles", type=int, default=0, help="0: until Ctrl+C")
    p.add_argument("--start", help="the model to start from (default: live.improve's current)")
    p.add_argument("--collect", type=int, default=5, help="training maps per copy per cycle")
    p.add_argument("--no-collect", dest="do_collect", action="store_false")
    p.add_argument("--updates", type=int, default=8000)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--dry-gate", action="store_true", help="stop after the offline screen (no game needed)")
    p.add_argument("--no-promote", dest="do_promote", action="store_false", help="judge, but never swap models")
    # the f4ws recipe: the when head as a second target and streams oversampled (+2.39 in the game over fovea1, Exp 039)
    p.add_argument("--recipe", nargs=argparse.REMAINDER, default=["--when", "0.15", "--stream-frac", "0.3"],
                   help="extra live.finetune_real arguments (last; replaces the default recipe)")
    p.add_argument("--recipe2", type=shlex.split, default=None,
                   help='a second recipe as one string (--recipe2="--when 0.15 --click-w 3"): its candidate takes '
                        "the gate's third copy instead of a repeat of the first")
    a = p.parse_args()
    state = load_state(a)
    done = 0
    while not a.cycles or done < a.cycles:
        if os.path.exists(STOP_FILE):  # stop between cycles (the file is removed)
            os.remove(STOP_FILE)
            say(f"{STOP_FILE}: stopping between cycles")
            break
        if os.path.exists(RECIPES):  # recipes changed while the loop runs: {"recipe": [...], "recipe2": [...] or null}
            r = json.load(open(RECIPES, encoding="utf-8"))
            a.recipe, a.recipe2 = r.get("recipe", a.recipe), r.get("recipe2", a.recipe2)
            say(f"recipes from {RECIPES}: {' '.join(a.recipe)} | {' '.join(a.recipe2 or []) or '-'}")
        state["cycle"] += 1
        out = os.path.join("runs", "cycle", f"c{state['cycle']:03d}")
        os.makedirs(out, exist_ok=True)
        row = dict(cycle=state["cycle"], current=state["current"], recipe=a.recipe, start=time.strftime("%Y%m%d-%H%M%S"))
        if a.do_collect:
            recs = collect(state["current"], a.collect, out)
            row["collected"] = len(recs)
            say(f"collected {len(recs)} usable recordings")
            if not recs:
                say("nothing usable collected (capture < 100 fps? Parsec or a display must be on): waiting 10 min")
                state["cycle"] -= 1
                time.sleep(600)
                continue
            row["labelled"] = len(label(recs))
        if os.path.exists(RECIPES):  # again just before training: a change made while it collected counts now
            r = json.load(open(RECIPES, encoding="utf-8"))
            a.recipe, a.recipe2 = r.get("recipe", a.recipe), r.get("recipe2", a.recipe2)
        # from the last cycle's lead candidate when it was ahead of the model playing but not promoted: cycle 6 went on
        # from cycle 5's (+0.50) and was promoted at +0.82, while starting over from the model playing gave +0.46
        start = state.get("lead") if state.get("lead") and os.path.exists(state["lead"]) else state["current"]
        row["init"] = start
        if start != state["current"]:
            say(f"training from the lead candidate {start} (the model playing: {state['current']})")
        cands = train(state["current"], os.path.join(out, "train"), a.recipe, a.updates, a.lr, init=start)
        cands2 = train(state["current"], os.path.join(out, "train2"), a.recipe2, a.updates, a.lr, init=start) if a.recipe2 else []
        state["lead"] = None  # set again below if a candidate is ahead but not promoted
        row["candidates"], row["candidates2"] = cands, cands2
        if cands or cands2:
            scores = screen(state["current"], cands + cands2, held_out(state["current"]))
            row["offline"] = scores
            # to the gate unless clearly worse offline: real +0.8 points showed as +0.001 offline (cycle 3), and the
            # strict "better than the model playing" kept every candidate of cycle 4 out of the game
            pick = lambda cs: max(cs, key=scores.get) if cs and max(scores[c] for c in cs) > scores[state["current"]] - SCREEN_SLACK else None
            b1, b2 = pick(cands), pick(cands2)
            if b1 and not cands2:  # one recipe: the third copy plays its next-best candidate (e.g. ema vs best), not b1 again
                rest = [c for c in cands if c != b1 and scores[c] > scores[state["current"]] - SCREEN_SLACK]
                b2 = max(rest, key=scores.get) if rest else None
            row["screened"], row["screened2"] = b1, b2
            if (b1 or b2) and not a.dry_gate:
                folder = run_gate(state["current"], b1 or b2, b2 if b1 else None)
                if folder:
                    folders = [folder]
                    cands_g = [c for c in (b1, b2) if c]
                    res = paired(state["current"], cands_g, folders)
                    for c, (mean, se, won, n) in res.items():
                        say(f"gate {c}: {mean:+.2f} +- {se:.2f} points, {won}/{n} maps ({folder})")
                    row["gate"], row["gate_folder"] = res, folder
                    winners = [c for c in res if passes(res[c])]
                    lead = max(res, key=lambda c: res[c][0]) if res else None
                    # ahead but not clear: three more rounds of the leader (two arms of it), pooled with the first run.
                    # Cycles 2-4: every candidate +0.6 to +0.9 over the model playing, held back at 2 se by 18 plays
                    if not winners and lead and res[lead][0] >= CONFIRM_MIN and a.do_promote:
                        say(f"confirming {lead} ({res[lead][0]:+.2f}): three more rounds")
                        f2 = run_gate(state["current"], lead)
                        if f2:
                            folders.append(f2)
                            res2 = paired(state["current"], [lead], folders)
                            row["confirm"], row["confirm_folder"] = res2, f2
                            for c, (mean, se, won, n) in res2.items():
                                say(f"gate pooled {c}: {mean:+.2f} +- {se:.2f} points, {won}/{n} maps ({', '.join(folders)})")
                            winners = [c for c in res2 if passes(res2[c])]
                            res = res2
                    if winners and a.do_promote:
                        best = max(winners, key=lambda c: res[c][0])
                        mean, se, won, n = res[best]
                        promote(state, best, f"{mean:+.2f} +- {se:.2f}, {won}/{n} maps ({', '.join(folders)})", folders[-1])
                        row["promoted"] = best
                        say(f"promoted {best}")
                    elif res:
                        lead = max(res, key=lambda c: res[c][0])
                        if res[lead][0] > 0:
                            state["lead"] = lead
                            say(f"lead candidate for the next cycle: {lead} ({res[lead][0]:+.2f})")
        row["end"] = time.strftime("%Y%m%d-%H%M%S")
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        save_state(state)
        done += 1


if __name__ == "__main__":
    main()
