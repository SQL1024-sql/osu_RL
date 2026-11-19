"""Where the student's misses come from, in the real game: the same model plays the gate maps as usual, with perfect
keys from the map (it only aims), and with perfect aim from the map (it only clicks) (live.oracle). Each condition
gets its own copy of the fork, and the copies swap conditions every round (a Latin square), so a copy that plays
better than the others counts for every condition alike.

    python -m live.ablate --ckpt runs/improve/r033_aim6/best_cal.pt            # 3 rounds on copies 1-3
    python -m live.ablate --report runs/ablate/<stamp>                         # the table again
    python -m live.ablate --makeup runs/ablate/<stamp>                         # replay plays left out (window minimized)
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

# name -> extra live.collect arguments; the default run compares the first three
CONDITIONS = {"student": [], "keys": ["--oracle", "keys"], "aim": ["--oracle", "aim"],
              # hold logit under which a held key is let go (live.vision_play VisionPlayer.hold_th; -1.5 is the default since 2026-11-18)
              "hold0": ["--hold-th", "0"], "hold-1.5": ["--hold-th", "-1.5"], "hold-3": ["--hold-th", "-3"],
              # where the cursor is sent, moved by (dx, dy) osu! px: the students click ~(-13, +10) px off the circles
              "offset-half": ["--aim-offset", "6.5", "-5"], "offset-full": ["--aim-offset", "13", "-10"]}
DEFAULT = ["student", "keys", "aim"]
GATE = ["runs/maps/tier3/gate/*.osu", "runs/maps/tier4/gate/*.osu"]
# the loop's play options (live.improve defaults)
PLAY = ["--click-th", "-0.75", "--min-hold", "5", "--hold-press", "8", "--smooth", "0.5", "--hold-th", "-1.5"]


def parse_condition(c):
    """'keys' -> ('keys', its live.collect arguments); 'fovea=path/best.pt' -> ('fovea', ['--ckpt', path])."""
    if "=" in c:  # name=checkpoint, optionally followed by +more live.collect arguments: f2w=x.pt+--decoder+sched
        name, rest = c.split("=", 1)
        ckpt, *more = rest.split("+")
        assert os.path.exists(ckpt), ckpt
        return name, ["--ckpt", ckpt, *more]
    assert c in CONDITIONS, f"{c}: not one of {list(CONDITIONS)} and not name=checkpoint"
    return c, CONDITIONS[c]


def run(a):
    out = os.path.join("runs", "ablate", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out, exist_ok=True)
    maps = sorted({f for g in a.maps for f in glob.glob(g)})
    conds = [parse_condition(c) for c in a.conditions]
    json.dump(dict(ckpt=a.ckpt, maps=maps, conditions=[c[0] for c in conds], args={n: x for n, x in conds},
                   instances=a.instances), open(os.path.join(out, "setup.json"), "w"))
    print(f"{out}: {len(maps)} maps, {len(conds)} conditions on copies {a.instances}, {a.rounds} rounds", flush=True)
    for r in range(a.rounds):
        procs = []
        for i, inst in enumerate(a.instances):
            name, extra = conds[(i + r) % len(conds)]
            log = open(os.path.join(out, f"r{r}-copy{inst}-{name}.log"), "w", encoding="utf-8")
            cmd = [sys.executable, "-u", "-m", "live.collect", "--background", "--no-record", "--maps", *map(glob.escape, maps),
                   "--plays", str(len(maps)), "--ckpt", a.ckpt, "--out", os.path.join(out, f"r{r}-{name}"), *PLAY] \
                + extra
            procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                           env=dict(os.environ, OSU_RL_INSTANCE=str(inst), PYTHONIOENCODING="utf-8")), log))
            time.sleep(20)  # the copies start one after another
        for p, log in procs:
            p.wait()
            log.close()
        print(f"round {r + 1} done", flush=True)
        report(out)


def makeup(out):
    """Plays the left-out plays (stale_steps) again: the same condition and map on the same copy, into the same round's
    folder, so report() counts the new play instead. Copies run at once, each its own list in turn."""
    setup = json.load(open(os.path.join(out, "setup.json")))
    todo = {}  # copy -> [(round folder, condition, map)]
    for f in glob.glob(os.path.join(out, "r*-*", "*", "plays.jsonl")):
        folder = os.path.dirname(os.path.dirname(f))
        rnd, name = os.path.basename(folder).split("-", 1)
        logs = glob.glob(os.path.join(out, f"{rnd}-copy*-{name}.log"))
        inst = int(os.path.basename(logs[0]).split("-copy")[1].split("-")[0])
        for line in open(f, encoding="utf-8"):
            p = json.loads(line)
            if p.get("accuracy") is not None and stale_steps(p) > STALE_MAX:
                todo.setdefault(inst, []).append((folder, name, p["map"]))
    if not todo:
        print("nothing to play again")
        return
    print("again: " + "; ".join(f"copy {i}: " + ", ".join(f"{n} {os.path.basename(m)[:30]}" for _, n, m in v) for i, v in todo.items()), flush=True)
    procs = []
    for inst, items in todo.items():
        cmds = [[sys.executable, "-u", "-m", "live.collect", "--background", "--no-record", "--maps", glob.escape(m),
                 "--plays", "1", "--ckpt", setup["ckpt"], "--out", folder, *PLAY, *setup.get("args", CONDITIONS)[n]]
                for folder, n, m in items]
        log = open(os.path.join(out, f"makeup-copy{inst}.log"), "a", encoding="utf-8")
        env = dict(os.environ, OSU_RL_INSTANCE=str(inst), PYTHONIOENCODING="utf-8")
        # one copy's plays one after another, in a small driver process; copies side by side
        driver = "import subprocess, sys, json\nfor c in json.loads(sys.argv[1]):\n    subprocess.run(c)\n"
        procs.append((subprocess.Popen([sys.executable, "-c", driver, json.dumps(cmds)], stdout=log, stderr=subprocess.STDOUT, env=env), log))
        time.sleep(20)
    for p, log in procs:
        p.wait()
        log.close()
    report(out)


def classify(lazer_path, map_path):
    """Per play: accuracy and each miss's kind. A circle / slider head miss is 'no click' (no key went down within
    the 50 window), 'off' (keys went down but the cursor was never on the circle at one of them) or 'on' (it was, and
    it still missed: the click went to another object, or note lock)."""
    from beatmap import radius
    import vision.student as S
    d = json.load(open(lazer_path, encoding="utf-8"))
    bm = S.load(map_path)
    r, w50 = radius(bm.cs), bm.windows[2]
    fr = np.array(d["frames"], float)
    t, xy, b = fr[:, 0], fr[:, 1:3], fr[:, 3].astype(int)
    down = np.nonzero(b[1:] & ~b[:-1])[0] + 1  # frames where some key went down
    c = dict(acc=d["accuracy"], head_miss=0, no_click=0, off=0, on=0, tail_miss=0, tick_miss=0, spin_miss=0)
    for start, kind, result, *_rest, x, y, _hit in d["judgements"]:
        if kind in ("HitCircle", "SliderHeadCircle") and result == "Miss":
            c["head_miss"] += 1
            k = down[(t[down] >= start - w50) & (t[down] <= start + w50)]
            if not len(k):
                c["no_click"] += 1
            elif np.any(np.hypot(*(xy[k] - (x, y)).T) <= r):
                c["on"] += 1
            else:
                c["off"] += 1
        elif kind == "SliderTailCircle" and result in ("IgnoreMiss", "Miss"):
            c["tail_miss"] += 1
        elif kind in ("SliderTick", "SliderRepeat") and result.endswith("Miss"):
            c["tick_miss"] += 1
        elif kind == "Spinner" and result == "Miss":
            c["spin_miss"] += 1
    return c


COLS = [("acc", "準確率"), ("head_miss", "圈+滑條頭 Miss"), ("off", "游標不在圈上"), ("on", "在圈上卻 Miss"),
        ("no_click", "窗內沒點"), ("tail_miss", "滑條尾"), ("tick_miss", "tick/repeat"), ("spin_miss", "轉盤")]


STALE_MS = 100  # a step whose newest frame is older than this saw a frozen picture
STALE_MAX = 12  # steps of those a play may have (0.1 s) before it is left out


def stale_steps(play):
    """Steps of a play (live.collect row) that decided on a frame older than STALE_MS: a minimized window gives the
    capture no frames, and the student keeps playing on the last one."""
    if not play.get("out") or not os.path.exists(play["out"]):
        return 0
    log = np.load(play["out"])["log"]
    return int(((log[:, 0] - log[:, 11]) * 1000 > STALE_MS).sum())


def report(out):
    setup = json.load(open(os.path.join(out, "setup.json")))
    by_md5 = {}
    from live.bridge import md5_of
    for m in setup["maps"]:
        by_md5[md5_of(m)] = m
    rows, excluded = {}, []
    for f in glob.glob(os.path.join(out, "r*-*", "*", "plays.jsonl")):
        name = os.path.basename(os.path.dirname(os.path.dirname(f))).split("-", 1)[1]
        for line in open(f, encoding="utf-8"):
            p = json.loads(line)
            if p.get("accuracy") is None or not p.get("lazer") or not os.path.exists(p["lazer"]):
                continue
            if stale_steps(p) > STALE_MAX:  # the window was minimized for a while: the student played blind
                excluded.append((name, os.path.basename(p["map"]), p["time"]))
                continue
            c = classify(p["lazer"], by_md5.get(p["md5"], p["map"]))
            c["map"] = os.path.basename(p["map"])
            rows.setdefault(name, []).append(c)
    if not rows:
        print("no scored plays yet")
        return
    lines = ["| 條件 | 場數 | " + " | ".join(h for _, h in COLS) + " |", "|---|---|" + "---|" * len(COLS)]
    for name in [c for c in setup["conditions"] if c in rows]:
        rs = rows[name]
        cells = []
        for k, _ in COLS:
            v = np.array([x[k] for x in rs], float)
            cells.append(f"{100 * v.mean():.2f}% ± {100 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0:.2f}" if k == "acc" else f"{v.mean():.1f}")
        lines.append(f"| {name} | {len(rs)} | " + " | ".join(cells) + " |")
    # per map, so a map that one condition played more often does not tilt the means
    lines += ["", "| 譜 | " + " | ".join(c for c in setup["conditions"] if c in rows) + " |", "|---|" + "---|" * len(rows)]
    for m in sorted({x["map"] for rs in rows.values() for x in rs}):
        cells = []
        for name in [c for c in setup["conditions"] if c in rows]:
            v = [x["acc"] for x in rows[name] if x["map"] == m]
            cells.append(f"{100 * np.mean(v):.2f}% ({len(v)})" if v else "-")
        lines.append(f"| {m[:48]} | " + " | ".join(cells) + " |")
    # each condition against the first, map by map (each map's mean difference; the error over maps)
    names = [c for c in setup["conditions"] if c in rows]
    maps = sorted({x["map"] for rs in rows.values() for x in rs})
    for name in names[1:]:
        d = [100 * (np.mean([x["acc"] for x in rows[name] if x["map"] == m]) - np.mean([x["acc"] for x in rows[names[0]] if x["map"] == m]))
             for m in maps if any(x["map"] == m for x in rows[name]) and any(x["map"] == m for x in rows[names[0]])]
        if len(d) > 1:
            lines.append(f"\n{name} − {names[0]}（逐譜）：{np.mean(d):+.2f} ± {np.std(d, ddof=1) / np.sqrt(len(d)):.2f} 分，"
                         f"{sum(v > 0 for v in d)}/{len(d)} 首比較高")
    if excluded:
        lines += ["", f"排除 {len(excluded)} 場（畫面停住超過 {STALE_MAX} 步，視窗被縮小）："]
        lines += [f"- {n} {m[:48]} {t}" for n, m, t in sorted(excluded)]
    text = "\n".join(lines)
    open(os.path.join(out, "report.md"), "w", encoding="utf-8").write(text + "\n")
    print(text, flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/improve/r033_aim6/best_cal.pt")
    p.add_argument("--maps", nargs="*", default=GATE)
    p.add_argument("--conditions", nargs="*", default=DEFAULT,
                   help=f"{', '.join(CONDITIONS)}, or name=checkpoint for another model (the first one is the baseline)")
    p.add_argument("--instances", type=int, nargs="*", default=[1, 2, 3], help="fork copies (OSU_RL_INSTANCE); 0 is the stream's")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--report", help="only print the table of this run folder")
    p.add_argument("--makeup", help="play the left-out plays of this run folder again, then print the table")
    a = p.parse_args()
    if a.report:
        report(a.report)
    elif a.makeup:
        makeup(a.makeup)
    else:
        run(a)


if __name__ == "__main__":
    main()
