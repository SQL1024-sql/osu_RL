"""The student getting better on its own, round after round, with nobody (and no Claude) in the loop:

    collect -> label -> retrain -> gate -> (swap) -> collect -> ...

    python -m live.improve                       # until Ctrl+C
    python -m live.improve --rounds 1 --collect 2 --gate 1 --updates 500    # one small round, to check it runs

Each round, in runs/improve/:
1. collect: the current model plays --collect maps drawn from the training maps (TRAIN_MAPS) in the fork (live.collect.Session), recorded;
2. label: live.dagger_label asks the teacher what to do in every state those plays went through;
3. retrain: live.finetune_real trains a candidate from --init on the teacher's recordings plus the newest student
   recordings (at most --max-recs, all of them on the GPU), held-out loss on the validation recording;
4. gate: the current model and the candidate play the gate maps in turn, --gate times each; the candidate replaces
   the current model only if its mean accuracy is higher by --margin points or more.
While the candidate trains the game stands still (training takes the whole GPU); --play-while-training keeps the
current model playing meanwhile, which needs a second GPU. state.json says which model plays now;
rounds.jsonl has one line per round. Restarting picks up from state.json.
"""
import argparse
import ctypes
import glob
import hashlib
import json
import os
import random
import subprocess
import sys
import time

import numpy as np

from live.collect import Session
from live.tablet import is_fork, lazer_window, window_exe

TEACHER_RECS = ["runs/live_rec/[1-4]_train_*-2026*[0-9].npz", "runs/live_rec/renatus_hard-20261111-002535.npz",
                "runs/live_rec/renatus_hard-20261111-004231.npz"]
VAL_REC = "runs/live_rec/5_val_-_103019_-20261111-012426.npz"
# runs/maps/pool: maps picked by live.make_pool (the live set alone had 5, and the student learnt them by heart)
TRAIN_MAPS = ["runs/maps/live_set/[1-4]*.osu", "runs/maps/pool/train/*.osu"]
GATE_MAPS = ["runs/maps/live_set/5_*.osu", "runs/maps/pool/gate/*.osu"]
# difficulty tiers (live.make_pool --out runs/maps/tierN): the loop starts on the first; once the model playing scores
# --promote on the gate maps of its tier --promote-rounds gates in a row, the next tier (if its maps exist) joins: half
# of each round's training maps from the newest tier, half from the earlier ones (so it keeps what it learnt), and the
# gate on the newest tier's gate maps and the one before's
TIERS = [("第 1 級（3–4.5 星）", TRAIN_MAPS, GATE_MAPS)] + [
    (f"第 {n} 級（{lo}–{hi} 星）", [f"runs/maps/tier{n}/train/*.osu"], [f"runs/maps/tier{n}/gate/*.osu"])
    for n, lo, hi in ((2, 4.5, 5), (3, 5, 5.5), (4, 5.5, 6))]

START = "results/experiments/030-real-dagger-mapkeys/seed0/best.pt"


def maps_of(globs):
    return sorted({f for g in globs for f in glob.glob(g)})


def step_hz(log_path):
    """How many decisions a second the student made in a play (its vision_play log; 120 when nothing slowed it)."""
    z = np.load(log_path)
    wall = z["log"][:, list(z["cols"]).index("wall")]
    # steps over the whole play (as live.collect's step_hz): a slowed student mostly stalls, its median step looks fine
    return (len(wall) - 1) / float(wall[-1] - wall[0]) if len(wall) > 1 and wall[-1] > wall[0] else 0.0


# GPU memory a live.finetune_real run takes besides the recordings (measured 2026-11-13, 16 GB card): the rendered pool
# (300 batches) 3.3 GB, training (batch 256, the held-out batches of 512, AdamW) 4.0 GB, its CUDA context 0.5 GB
TRAIN_FIXED_GB = 7.8


def rec_gb(path):
    """GPU memory a recording takes in training: its arrays' size, read from the .npz headers (nothing is loaded)."""
    import zipfile
    from numpy.lib import format as F
    total = 0
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            with z.open(n) as f:
                v = F.read_magic(f)
                shape, _, dtype = (F.read_array_header_1_0 if v == (1, 0) else F.read_array_header_2_0)(f)
                total += int(np.prod(shape)) * dtype.itemsize
    return total / 2 ** 30


def fork_minimized():
    """True if the fork is running with its window minimized."""
    try:
        h = getattr(lazer_window(), "value", None)
    except SystemExit:  # no window yet: the fork is not up (the session starts it)
        return False
    return bool(h and ctypes.windll.user32.IsIconic(ctypes.c_void_p(h)) and is_fork(window_exe(h)))


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


class Improver:
    def __init__(self, a):
        self.a = a
        os.makedirs(a.out, exist_ok=True)
        self.state_path = os.path.join(a.out, "state.json")
        self.state = json.load(open(self.state_path)) if os.path.exists(self.state_path) else \
            dict(round=0, current=a.start, history=[])
        self.state.setdefault("tier", 0)
        self.state.setdefault("promote_streak", 0)
        self.gate_md5 = {md5(p) for _, _, g in TIERS for p in maps_of(g)}  # never trained on, whatever the tier
        sargs = argparse.Namespace(maps=TRAIN_MAPS, shuffle=True, hours=0, plays=0, mods=["NF"], ckpt=self.state["current"],
                                   click_th=a.click_th, min_hold=a.min_hold, press="first", record=True, min_fps=50,
                                   max_gb=0, restart_fork_mb=6000, out=os.path.join(a.out, "plays"),
                                   background=not a.foreground, view=a.view or None, view_style=a.view_style,
                                   overlay=os.path.join(a.out, "overlay.txt"))
        self.session = Session(sargs)
        self.phase = "start"
        self.session.overlay_head = self.overlay_lines

    def say(self, text):
        self.session.say(f"[improve] {text}")

    PHASES = dict(collect="練習中", label="整理這輪的錄影", retrain="訓練新模型中（舊模型繼續打）",
                  gate="考試：目前的模型 vs 新訓練的模型")

    def set_phase(self, phase):
        self.phase = phase
        self.session.status("improving", phase=phase)

    def overlay_lines(self):
        """The overlay's first lines: the round, what the loop is doing, and which model is playing."""
        playing = os.path.basename(os.path.dirname(self.session.args.ckpt or "")) or "?"
        if self.phase == "gate":
            playing += "（目前的模型）" if self.session.args.ckpt == self.state["current"] else "（新模型）"
        return [f"osu_rl 自己練習中・第 {self.state['round'] + 1} 輪・{TIERS[self.state['tier']][0]}",
                self.PHASES.get(self.phase, self.phase),
                f"上場模型：{playing}"]

    def save_state(self):
        tmp = self.state_path + ".tmp"
        json.dump(self.state, open(tmp, "w"), indent=1)
        os.replace(tmp, self.state_path)

    def use(self, ckpt):
        self.session.player.load(ckpt)
        self.session.args.ckpt = ckpt

    def play_maps(self, maps, n):
        """Plays n maps (cycling through maps) with the loaded model. -> rows"""
        rows = []
        i = 0
        while i < n:
            path = maps[i % len(maps)]
            if self.wait_unminimized():
                continue
            try:
                row = self.session.play(path)
            except Exception as e:
                row = dict(map=path, error=f"{type(e).__name__}: {e}")
                self.session.clock = None if self.session.clock is None or not self.session.clock.connected else self.session.clock
            if row.get("accuracy") is None and fork_minimized():
                continue  # not the map's fault: the same map again once the window is back
            i += 1
            self.session.plays.append(row)
            with open(os.path.join(self.session.dir, "plays.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            acc = row.get("accuracy")
            self.say(f"{os.path.basename(path)} with {os.path.basename(os.path.dirname(row.get('model', '') or '')) or '?'}: "
                     + (f"{acc:.2%}" if acc is not None else f"no score ({row.get('error')})"))
            self.session.status("improving", phase=self.phase)
            self.session.housekeeping(row)  # frames cap, fork restart when its memory grew
            rows.append(row)
        return rows

    def play_groups(self, groups):
        """Several copies of the fork at once (--instances): groups[0] = (model, maps) plays here, on the first copy
        (the one on stream); each other group in a live.collect process of its own, on copy 1, 2, ... in the
        background. -> one list of rows per group"""
        workers = []
        for i, (ckpt, maps) in enumerate(groups[1:], 1):
            log_path = os.path.join(self.a.out, "workers", f"copy{i}-{time.strftime('%Y%m%d-%H%M%S')}.log")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            cmd = [sys.executable, "-u", "-m", "live.collect", "--background", "--maps", *map(glob.escape, maps),
                   "--plays", str(len(maps)),
                   "--ckpt", ckpt, "--click-th", str(self.a.click_th), "--min-hold", str(self.a.min_hold),
                   "--out", os.path.join(self.a.out, "plays")]
            log = open(log_path, "w", encoding="utf-8")
            workers.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                             env=dict(os.environ, OSU_RL_INSTANCE=str(i), PYTHONIOENCODING="utf-8")),
                            log, log_path))
        ckpt, maps = groups[0]
        if ckpt != self.session.args.ckpt:
            self.use(ckpt)
        results = [self.play_maps(maps, len(maps))]
        for proc, log, log_path in workers:
            proc.wait()
            log.close()
            results.append(self.worker_rows(log_path))
        return results

    def worker_rows(self, log_path):
        """The rows a live.collect process wrote (its session folder is named on its first line)."""
        text = open(log_path, encoding="utf-8", errors="replace").read()
        found = [l.split("session ", 1)[1].split(": model", 1)[0] for l in text.splitlines() if "] session " in l]
        if not found:
            self.say(f"{log_path}: the copy never started a session; see its log")
            return []
        path = os.path.join(found[0], "plays.jsonl")
        rows = [json.loads(l) for l in open(path, encoding="utf-8")] if os.path.exists(path) else []
        name = os.path.basename(log_path).split("-")[0]
        for row in rows:
            acc = row.get("accuracy")
            self.say(f"({name}) {os.path.basename(row['map'])} with {os.path.basename(os.path.dirname(row.get('model') or ''))}: "
                     + (f"{acc:.2%}" if acc is not None else f"no score ({row.get('error')})"))
        return rows

    def wait_unminimized(self):
        """A minimized window gives the capture no frames: wait (without touching the window) until someone restores
        it. -> True if it waited."""
        if not fork_minimized():
            return False
        self.say("the fork's window is minimized (the capture gets no frames): waiting until it is restored")
        self.session.status("waiting", reason="the fork's window is minimized")
        while fork_minimized():
            time.sleep(5)
        self.say("the fork's window is back")
        time.sleep(2)
        return True

    def student_recs(self):
        """Labelled student recordings of training maps (never the gate maps), newest first."""
        recs = []
        for f in glob.glob("runs/live_play/*-rec.npz") + glob.glob(os.path.join(self.a.out, "plays", "*", "*-rec.npz")) + \
                glob.glob("runs/collect/*/*-rec.npz") + glob.glob("runs/stream/plays/*/*-rec.npz"):
            if not os.path.exists(f[:-4] + ".labels.npz"):
                continue
            try:
                if str(np.load(f)["md5"]) in self.gate_md5:
                    continue
                if step_hz(f[: -len("-rec.npz")] + ".npz") < 100:
                    continue  # the student ran slow (the GPU shared with training): not how it plays
            except Exception:
                continue
            recs.append(f)
        return sorted(set(recs), key=os.path.getmtime, reverse=True)

    def label(self, recs):
        # not yet labelled, not given up on (.labels.failed: live.dagger_label could not), and not of a gate map
        # (never trained on: labelling them would only take time)
        todo = []
        for r in recs:
            if os.path.exists(r[:-4] + ".labels.npz") or os.path.exists(r[:-4] + ".labels.failed"):
                continue
            try:
                if str(np.load(r)["md5"]) in self.gate_md5:
                    continue
            except Exception:
                pass
            todo.append(r)
        if not todo:
            return
        self.say(f"labelling {len(todo)} recordings")
        done = 0
        for i in range(0, len(todo), 40):  # a few at a time: Windows caps a command line at 32k characters
            r = subprocess.run([sys.executable, "-m", "live.dagger_label", *todo[i: i + 40]], capture_output=True, text=True)
            for line in (r.stdout + r.stderr).splitlines():
                if "Error" in line or "Traceback" in line:
                    self.say(line.strip())
                done += "lazer acc" in line
        self.say(f"labelled {done} of {len(todo)}")

    def fit_students(self, teacher):
        """The newest student recordings that fit on the GPU next to the teacher's, the validation recording, the
        rendered pool and training itself (live.finetune_real keeps all of them on the GPU; past the card's memory
        Windows spills into system RAM and training runs 2-4x slower), at most --max-recs in all."""
        students = self.student_recs()[: max(0, self.a.max_recs - len(teacher))]
        try:
            import torch
            free = torch.cuda.mem_get_info()[0] / 2 ** 30
        except Exception:
            return students
        budget = free - TRAIN_FIXED_GB - self.a.vram_margin_gb - sum(map(rec_gb, teacher + [VAL_REC]))
        fit = []
        for f in students:
            if rec_gb(f) > budget:
                break
            budget -= rec_gb(f)
            fit.append(f)
        self.say(f"GPU free {free:.1f} GB: {len(fit)} of {len(students)} student recordings fit")
        return fit

    def retrain(self, rnd):
        teacher = sorted({f for g in TEACHER_RECS for f in glob.glob(g) if not f.endswith(".labels.npz")})
        students = self.fit_students(teacher)
        out = os.path.join(self.a.out, f"r{rnd:03d}")
        while os.path.exists(out):  # a round that was stopped before it finished: keep what it left
            out += "b"
        init = self.state["current"] if self.a.init == "current" else self.a.init
        cmd = [sys.executable, "-u", "-m", "live.finetune_real", *teacher, *students, "--val", VAL_REC, "--clicks", "map",
               "--init", init, "--updates", str(self.a.updates), "--lr", str(self.a.lr), "--out", out]
        self.say(f"retraining from {init} on {len(teacher)} teacher + {len(students)} student recordings -> {out}")
        with open(out + "-train.log", "w", encoding="utf-8") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            # the stream never stops: the current model keeps playing training maps while the candidate trains
            # (recorded like any play: next round's labels and training use them too)
            played = 0
            while proc.poll() is None:
                if self.a.pause_while_training:
                    time.sleep(5)
                    continue
                train = maps_of(TRAIN_MAPS)
                if not any(r.get("accuracy") is not None for r in self.play_maps([random.choice(train)], 1)):
                    time.sleep(5)  # a failed play: do not spin
                played += 1
            returncode = proc.wait()
        if played:
            self.say(f"{played} plays while training")
        best = os.path.join(out, "best.pt")
        if returncode != 0 or not os.path.exists(best):
            self.say(f"retraining failed (exit {returncode}); see {out}-train.log")
            return None
        try:
            if json.load(open(os.path.join(out, "log.json")))["best_update"] == 0:
                # the held-out loss never beat the starting weights: the candidate would be them; no gate needed
                self.say("retraining never beat its starting weights on the held-out recording: no candidate this round")
                return None
        except (OSError, KeyError, ValueError):
            pass
        if self.a.calibrate:
            # its click and hold outputs lined up with the model playing now (live.calibrate), on the newest labelled
            # student recordings: without it the thresholds meant fewer clicks and every candidate lost (rounds 5-9)
            cal = os.path.join(out, "best_cal.pt")
            recs = self.student_recs()[:6]
            r = subprocess.run([sys.executable, "-m", "live.calibrate", best, "--ref", self.state["current"], "--recs", *recs,
                                "--out", cal], capture_output=True, text=True)
            if r.returncode == 0 and os.path.exists(cal):
                self.say("calibrated: " + (r.stdout.strip().splitlines() or ["?"])[-1])
                return cal
            self.say(f"calibration failed (exit {r.returncode}): {(r.stdout + r.stderr).strip()[-300:]}; the uncalibrated candidate goes to the gate")
        return best

    def train_draw(self, n):
        """n training maps for this round: from the tier the loop is on, half from the earlier tiers once past the first."""
        t = self.state["tier"]
        new = maps_of(TIERS[t][1])
        old = maps_of([g for _, tr, _ in TIERS[:t] for g in tr])
        pick = lambda maps, k: (random.sample(maps, min(k, len(maps))) * (k // max(1, len(maps)) + 1))[:k] if maps and k else []
        draw = pick(new, n - n // 2) + pick(old, n // 2) if t else pick(new, n)
        random.shuffle(draw)
        return draw

    def gate_maps(self):
        t = self.state["tier"]
        return maps_of(TIERS[t][2] + (TIERS[t - 1][2] if t else []))

    def promote(self, by_map, who):
        """After a gate: did the model now playing (who) score --promote on its tier's gate maps, rounds in a row?"""
        t = self.state["tier"]
        if t + 1 >= len(TIERS) or not maps_of(TIERS[t + 1][1]) or not maps_of(TIERS[t + 1][2]):
            return  # no harder tier ready (live.make_pool --out runs/maps/tierN)
        tier_maps = set(maps_of(TIERS[t][2]))
        accs = [acc for m, v in by_map[who].items() if m in tier_maps for acc in v]
        score = float(np.mean(accs)) if accs else 0.0
        self.state["promote_streak"] = self.state["promote_streak"] + 1 if score >= self.a.promote else 0
        self.say(f"tier {t + 1} gate maps: {score:.2%} ({self.state['promote_streak']}/{self.a.promote_rounds} rounds at "
                 f"{self.a.promote:.0%} or more)")
        if self.state["promote_streak"] >= self.a.promote_rounds:
            self.state["tier"], self.state["promote_streak"] = t + 1, 0
            self.say(f"up a tier: {TIERS[t + 1][0]} joins the training and the gate")

    def gate(self, cand):
        """Current and candidate in turn on the gate maps. -> (current mean, candidate mean) accuracy, scores, and
        scores by map {who: {map: [accuracy, ...]}}."""
        gate_maps = self.gate_maps()
        scores = {"current": [], "candidate": []}
        by_map = {"current": {}, "candidate": {}}
        models = {"current": self.state["current"], "candidate": cand}
        # each repetition: both models on every gate map, in turn on one copy, at the same time on two; as many
        # repetitions at once as copies allow, sides swapped so that over the gate each model plays on every copy as
        # often as the other (a copy that scores higher or lower then favours neither; with 4 copies use --gate 4)
        per = max(1, self.a.instances // 2)
        reps = [("current", "candidate") if (i % per + i // per) % 2 == 0 else ("candidate", "current") for i in range(self.a.gate)]
        for b in range(0, len(reps), per):
            if self.a.instances < 2:
                for who in reps[b]:
                    for row in self.play_groups([(models[who], gate_maps)])[0]:
                        if row.get("accuracy") is not None:
                            scores[who].append(row["accuracy"])
                            by_map[who].setdefault(row["map"], []).append(row["accuracy"])
                continue
            order = [who for rep in reps[b: b + per] for who in rep]
            for who, rows in zip(order, self.play_groups([(models[who], gate_maps) for who in order])):
                for r in rows:
                    if r.get("accuracy") is not None:
                        scores[who].append(r["accuracy"])
                        by_map[who].setdefault(r["map"], []).append(r["accuracy"])
        self.use(self.state["current"])
        mean = lambda v: float(np.mean(v)) if v else float("nan")
        return mean(scores["current"]), mean(scores["candidate"]), scores, by_map

    def run(self):
        a = self.a
        self.say(f"start: round {self.state['round']}, current model {self.state['current']}")
        self.use(self.state["current"])
        rounds = 0
        try:
            while not a.rounds or rounds < a.rounds:
                rnd = self.state["round"] + 1
                t0 = time.time()
                self.set_phase("collect")
                self.say(f"round {rnd}: collecting {a.collect} plays with {self.state['current']}")
                # a different draw of the training maps every round (too many to play them all each round)
                draw = self.train_draw(a.collect)
                # split between the copies of the fork (--instances), all with the current model
                n = max(1, min(a.instances, len(draw)))
                rows = [r for g in self.play_groups([(self.state["current"], draw[i::n]) for i in range(n)]) for r in g]
                # every recording of this loop still unlabelled, also those of a run that was stopped mid-round, and
                # those brought back from the stream machine (live.stream --record, copied into runs/stream/plays/)
                recs = [r["rec"] for r in rows if r.get("rec")] + glob.glob(os.path.join(a.out, "plays", "*", "*-rec.npz")) \
                    + glob.glob("runs/stream/plays/*/*-rec.npz")
                self.set_phase("label")
                self.label(sorted(set(recs)))
                self.set_phase("retrain")
                if a.pause_while_training:
                    self.session.player.close()  # all of the GPU to training; nothing is sent to the game meanwhile
                if any(r.get("rec") for r in rows):
                    cand = self.retrain(rnd)
                else:
                    self.say("no new recordings this round: no retraining")
                    cand = None
                self.session.player.attach(wait=False)
                result = dict(round=rnd, current=self.state["current"], candidate=cand,
                              collected=[r.get("accuracy") for r in rows])
                if cand:
                    self.set_phase("gate")
                    cur, new, scores, by_map = self.gate(cand)
                    better = np.isfinite(new) and np.isfinite(cur) and (new - cur) * 100 >= a.margin
                    result.update(gate_current=cur, gate_candidate=new, gate_scores=scores, swapped=bool(better))
                    self.say(f"round {rnd} gate: current {cur:.2%}, candidate {new:.2%} -> "
                             + ("candidate takes over" if better else "current stays"))
                    if better:
                        self.state["current"] = cand
                        self.use(cand)
                    self.promote(by_map, "candidate" if better else "current")
                result["tier"] = self.state["tier"]
                result["minutes"] = round((time.time() - t0) / 60, 1)
                self.state["round"] = rnd
                self.state["history"].append({k: result[k] for k in result if k != "gate_scores"})
                self.save_state()
                with open(os.path.join(a.out, "rounds.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(result) + "\n")
                rounds += 1
        except KeyboardInterrupt:
            self.say("stopped (Ctrl+C)")
        finally:
            self.session.player.close()
            self.session.status("stopped")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=0, help="0: until Ctrl+C")
    p.add_argument("--collect", type=int, default=12, help="training-map plays per round")
    p.add_argument("--gate", type=int, default=2, help="plays of each gate map per model per round")
    p.add_argument("--margin", type=float, default=2.0, help="accuracy points the candidate must win by")
    p.add_argument("--updates", type=int, default=6000)
    p.add_argument("--lr", type=float, default=3e-5,
                   help="fine-tuning learning rate (2026-11-13, calibrated, vs r001 on the gate: 3e-5 x 6000 86.1%%, 3e-5 x 2000 "
                        "84.8%%, 1e-4 x 1000 84.2%%, r001 85.5%%; 1e-4 x 6000 was the old default)")
    p.add_argument("--init", default="current",
                   help="what each candidate trains from: the model playing now (current), or a weights file such as "
                        "results/experiments/020-student/seed0/round6.pt (from round 3 on: candidates from round6 kept losing)")
    p.add_argument("--start", default=START, help="the model that plays first (when there is no state.json yet)")
    p.add_argument("--max-recs", type=int, default=64,
                   help="at most this many recordings per training run; fewer when they do not fit on the GPU")
    p.add_argument("--vram-margin-gb", type=float, default=0.8,
                   help="GPU memory left free when picking how many recordings train (so nothing spills to system RAM)")
    p.add_argument("--click-th", type=float, default=-0.75)
    p.add_argument("--min-hold", type=int, default=5)
    p.add_argument("--play-while-training", dest="pause_while_training", action="store_false",
                   help="keep playing while the candidate trains, sharing the GPU. On one 16 GB card (2026-11-12) the card "
                        "filled up, the student ran at 61 Hz instead of 120 (89%% -> 53-60%%), training took 2.5x as "
                        "long and the fork shut down: only with a second GPU or a much smaller training run")
    p.add_argument("--view", type=int, default=8770, metavar="PORT",
                   help="serve what the student sees for OBS (Browser source http://127.0.0.1:PORT/); 0: off")
    p.add_argument("--view-style", choices=["neuro", "input"], default="neuro",
                   help="neuro: the map drawn plain and blurred from the game state, like the AI view on Neuro-sama's osu! streams; input: the frames the model got, sharp, with its move / clicks / keys")
    p.add_argument("--no-calibrate", dest="calibrate", action="store_false",
                   help="send the candidate to the gate as trained, its click/hold outputs not lined up (live.calibrate)")
    p.add_argument("--promote", type=float, default=0.90,
                   help="gate accuracy on its tier's gate maps that moves the loop up a difficulty tier (TIERS)")
    p.add_argument("--promote-rounds", type=int, default=2, help="... in this many gates in a row")
    p.add_argument("--instances", type=int, default=1,
                   help="copies of the fork playing at once (background input only; the first is the one on stream)")
    p.add_argument("--foreground", action="store_true",
                   help="play through the virtual tablet with the fork in front (default: in the background)")
    p.add_argument("--out", default="runs/improve")
    a = p.parse_args()
    imp = Improver(a)
    imp.phase = "start"
    imp.run()


if __name__ == "__main__":
    main()
