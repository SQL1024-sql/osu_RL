"""The student getting better on its own, round after round, with nobody (and no Claude) in the loop:

    collect -> label -> retrain -> gate -> (swap) -> collect -> ...

    python -m live.improve                       # until Ctrl+C
    python -m live.improve --rounds 1 --collect 2 --gate 1 --updates 500    # one small round, to check it runs

Each round, in runs/improve/:
1. collect: the current model plays --collect maps of the training set in the fork (live.collect.Session), recorded;
2. label: live.dagger_label asks the teacher what to do in every state those plays went through;
3. retrain: live.finetune_real trains a candidate from --init on the teacher's recordings plus the newest student
   recordings (at most --max-recs, all of them on the GPU), held-out loss on the validation recording;
4. gate: the current model and the candidate play the gate maps in turn, --gate times each; the candidate replaces
   the current model only if its mean accuracy is higher by --margin points or more.
Playing and training never overlap (training needs the GPU the player uses). state.json says which model plays now;
rounds.jsonl has one line per round. Restarting picks up from state.json.
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

from live.collect import Session

TEACHER_RECS = ["runs/live_rec/[1-4]_train_*-2026*[0-9].npz", "runs/live_rec/renatus_hard-20261111-002535.npz",
                "runs/live_rec/renatus_hard-20261111-004231.npz"]
VAL_REC = "runs/live_rec/5_val_-_103019_-20261111-012426.npz"
TRAIN_MAPS = "runs/maps/live_set/[1-4]*.osu"
GATE_MAPS = "runs/maps/live_set/5_*.osu"
START = "results/experiments/030-real-dagger-mapkeys/seed0/best.pt"


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


class Improver:
    def __init__(self, a):
        self.a = a
        os.makedirs(a.out, exist_ok=True)
        self.state_path = os.path.join(a.out, "state.json")
        self.state = json.load(open(self.state_path)) if os.path.exists(self.state_path) else \
            dict(round=0, current=a.start, history=[])
        self.gate_md5 = {md5(p) for p in glob.glob(GATE_MAPS)}
        sargs = argparse.Namespace(maps=[TRAIN_MAPS], shuffle=True, hours=0, plays=0, mods=["NF"], ckpt=self.state["current"],
                                   click_th=a.click_th, min_hold=a.min_hold, press="first", record=True, min_fps=50,
                                   max_gb=0, restart_fork_mb=6000, out=os.path.join(a.out, "plays"),
                                   background=not a.foreground)
        self.session = Session(sargs)

    def say(self, text):
        self.session.say(f"[improve] {text}")

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
        for i in range(n):
            path = maps[i % len(maps)]
            try:
                row = self.session.play(path)
            except Exception as e:
                row = dict(map=path, error=f"{type(e).__name__}: {e}")
                self.session.clock = None if self.session.clock is None or not self.session.clock.connected else self.session.clock
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

    def student_recs(self):
        """Labelled student recordings of training maps (never the gate maps), newest first."""
        recs = []
        for f in glob.glob("runs/live_play/*-rec.npz") + glob.glob(os.path.join(self.a.out, "plays", "*", "*-rec.npz")) + \
                glob.glob("runs/collect/*/*-rec.npz"):
            if not os.path.exists(f[:-4] + ".labels.npz"):
                continue
            try:
                if str(np.load(f)["md5"]) in self.gate_md5:
                    continue
            except Exception:
                continue
            recs.append(f)
        return sorted(set(recs), key=os.path.getmtime, reverse=True)

    def label(self, recs):
        todo = [r for r in recs if not os.path.exists(r[:-4] + ".labels.npz")]
        if not todo:
            return
        self.say(f"labelling {len(todo)} recordings")
        r = subprocess.run([sys.executable, "-m", "live.dagger_label", *todo], capture_output=True, text=True)
        for line in (r.stdout + r.stderr).splitlines():
            if "lazer acc" in line or "Error" in line:
                self.say(line.strip())

    def retrain(self, rnd):
        teacher = sorted({f for g in TEACHER_RECS for f in glob.glob(g) if not f.endswith(".labels.npz")})
        students = self.student_recs()[: max(0, self.a.max_recs - len(teacher))]
        out = os.path.join(self.a.out, f"r{rnd:03d}")
        init = self.state["current"] if self.a.init == "current" else self.a.init
        cmd = [sys.executable, "-u", "-m", "live.finetune_real", *teacher, *students, "--val", VAL_REC, "--clicks", "map",
               "--init", init, "--updates", str(self.a.updates), "--out", out]
        self.say(f"retraining from {init} on {len(teacher)} teacher + {len(students)} student recordings -> {out}")
        with open(os.path.join(self.a.out, f"r{rnd:03d}-train.log"), "w", encoding="utf-8") as log:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        best = os.path.join(out, "best.pt")
        if r.returncode != 0 or not os.path.exists(best):
            self.say(f"retraining failed (exit {r.returncode}); see r{rnd:03d}-train.log")
            return None
        try:
            if json.load(open(os.path.join(out, "log.json")))["best_update"] == 0:
                # the held-out loss never beat the starting weights: the candidate would be them; no gate needed
                self.say("retraining never beat its starting weights on the held-out recording: no candidate this round")
                return None
        except (OSError, KeyError, ValueError):
            pass
        return best

    def gate(self, cand):
        """Current and candidate in turn on the gate maps. -> (current mean, candidate mean) accuracy."""
        gate_maps = sorted(glob.glob(GATE_MAPS))
        scores = {"current": [], "candidate": []}
        for i in range(self.a.gate):
            for who, ckpt in (("current", self.state["current"]), ("candidate", cand))[:: 1 if i % 2 == 0 else -1]:
                self.use(ckpt)
                for row in self.play_maps(gate_maps, len(gate_maps)):
                    if row.get("accuracy") is not None:
                        scores[who].append(row["accuracy"])
        self.use(self.state["current"])
        mean = lambda v: float(np.mean(v)) if v else float("nan")
        return mean(scores["current"]), mean(scores["candidate"]), scores

    def run(self):
        a = self.a
        self.say(f"start: round {self.state['round']}, current model {self.state['current']}")
        self.use(self.state["current"])
        rounds = 0
        try:
            while not a.rounds or rounds < a.rounds:
                rnd = self.state["round"] + 1
                t0 = time.time()
                self.phase = "collect"
                self.say(f"round {rnd}: collecting {a.collect} plays with {self.state['current']}")
                rows = self.play_maps(sorted(glob.glob(TRAIN_MAPS)), a.collect)
                # every recording of this loop still unlabelled, also those of a run that was stopped mid-round
                recs = [r["rec"] for r in rows if r.get("rec")] + glob.glob(os.path.join(a.out, "plays", "*", "*-rec.npz"))
                self.phase = "label"
                self.label(sorted(set(recs)))
                self.phase = "retrain"
                self.session.player.close()  # the GPU goes to training; nothing is sent to the game meanwhile
                cand = self.retrain(rnd)
                self.session.player.attach(wait=False)
                result = dict(round=rnd, current=self.state["current"], candidate=cand,
                              collected=[r.get("accuracy") for r in rows])
                if cand:
                    self.phase = "gate"
                    cur, new, scores = self.gate(cand)
                    better = np.isfinite(new) and np.isfinite(cur) and (new - cur) * 100 >= a.margin
                    result.update(gate_current=cur, gate_candidate=new, gate_scores=scores, swapped=bool(better))
                    self.say(f"round {rnd} gate: current {cur:.2%}, candidate {new:.2%} -> "
                             + ("candidate takes over" if better else "current stays"))
                    if better:
                        self.state["current"] = cand
                        self.use(cand)
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
    p.add_argument("--gate", type=int, default=3, help="plays of each gate map per model per round")
    p.add_argument("--margin", type=float, default=2.0, help="accuracy points the candidate must win by")
    p.add_argument("--updates", type=int, default=6000)
    p.add_argument("--init", default="current",
                   help="what each candidate trains from: the model playing now (current), or a weights file such as "
                        "results/experiments/020-student/seed0/round6.pt (from round 3 on: candidates from round6 kept losing)")
    p.add_argument("--start", default=START, help="the model that plays first (when there is no state.json yet)")
    p.add_argument("--max-recs", type=int, default=24,
                   help="recordings per training run (all on the GPU: 32 filled the 16 GB card and ran 4x slower)")
    p.add_argument("--click-th", type=float, default=-0.75)
    p.add_argument("--min-hold", type=int, default=5)
    p.add_argument("--foreground", action="store_true",
                   help="play through the virtual tablet with the fork in front (default: in the background)")
    p.add_argument("--out", default="runs/improve")
    a = p.parse_args()
    imp = Improver(a)
    imp.phase = "start"
    imp.run()


if __name__ == "__main__":
    main()
