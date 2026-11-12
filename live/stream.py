"""The stream machine's side when training happens elsewhere: the student only plays, map after map, and a new model
dropped into runs/stream/new/ takes over before the next map. Nothing trains here, so the GPU needs only what playing
takes (the fork, the student, OBS: about 2-3 GB).

    python -m live.stream --foreground          # on the stream machine (virtual tablet, game in front)
    python -m live.stream                       # in the background (bridge input), as live.improve does by default
    python -m live.stream --export              # on the training machine: copy the model live.improve plays now to
                                                # runs/stream/outbox/, named after its round, to send over

Handing over a model: put the .pt file (from --export, or any runs/improve/rNNN/best.pt) into runs/stream/new/ on
the stream machine. Between maps it is checked (the same network, weights that load); a good one is moved to
runs/stream/models/ and plays from the next map on, a bad one is moved to runs/stream/bad/ and the old one keeps
playing. runs/stream/state.json says which model plays; restarting picks it up. Without one yet, the model live.improve
was playing when the bundle was made (runs/improve/state.json) starts.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

import torch

import vision.student as S
from live.collect import Session
from live.improve import GATE_MAPS, START, TRAIN_MAPS

DIR = os.path.join("runs", "stream")
NEW, MODELS, BAD, OUTBOX = (os.path.join(DIR, d) for d in ("new", "models", "bad", "outbox"))
STATE = os.path.join(DIR, "state.json")
IMPROVE_STATE = os.path.join("runs", "improve", "state.json")


def improve_current():
    """The model live.improve plays now (runs/improve/state.json), else its starting model."""
    try:
        return json.load(open(IMPROVE_STATE))["current"]
    except (OSError, KeyError, ValueError):
        return START


def model_name(path):
    """runs/improve/r005/best.pt -> r005; runs/stream/models/20261113-2210-r005.pt -> r005."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem in ("best", "last", "model"):
        stem = os.path.basename(os.path.dirname(path)) or stem
    parts = stem.split("-", 2)
    return parts[2] if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit() else stem


def check_weights(path):
    """None if this file holds weights for the student's network, else what is wrong with it."""
    try:
        sd = torch.load(path, map_location="cpu")
        S.Student().load_state_dict(sd)  # strict: every weight there, every shape right
        return None
    except Exception as e:
        return f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"


def export():
    src = improve_current()
    os.makedirs(OUTBOX, exist_ok=True)
    name = os.path.basename(os.path.dirname(src)) if os.path.basename(src) == "best.pt" else model_name(src)
    dst = os.path.join(OUTBOX, f"{name}.pt")
    shutil.copy2(src, dst)
    print(f"目前上場的模型：{src}\n已複製到：{os.path.abspath(dst)}\n把這個檔案放到直播機的 runs\\stream\\new\\ 資料夾，下一首就會換上。")


class Streamer:
    def __init__(self, a):
        self.a = a
        for d in (NEW, MODELS, BAD):
            os.makedirs(d, exist_ok=True)
        self.state = json.load(open(STATE)) if os.path.exists(STATE) else dict(current=improve_current(), since=time.time(), history=[])
        sargs = argparse.Namespace(maps=TRAIN_MAPS + GATE_MAPS, shuffle=True, hours=0, plays=0, mods=["NF"],
                                   ckpt=self.state["current"], click_th=a.click_th, min_hold=a.min_hold, press="first",
                                   record=a.record, min_fps=50, max_gb=a.max_gb, restart_fork_mb=6000,
                                   out=os.path.join(DIR, "plays"), background=not a.foreground, view=a.view or None,
                                   view_style=a.view_style, overlay=os.path.join(DIR, "overlay.txt"))
        self.session = Session(sargs)
        self.session.overlay_head = self.overlay_lines
        self._session_play = self.session.play
        self.session.play = self.play  # Session.run's loop, with the inbox checked before every map
        self.save()

    def save(self):
        tmp = STATE + ".tmp"
        json.dump(self.state, open(tmp, "w"), indent=1)
        os.replace(tmp, STATE)

    def overlay_lines(self):
        since = time.strftime("%m/%d %H:%M", time.localtime(self.state.get("since", time.time())))
        return ["osu_rl 直播中", f"上場模型：{model_name(self.state['current'])}（{since} 換上）"]

    def check_inbox(self):
        """A model dropped into runs/stream/new/ (the newest, if several) takes over; the others are set aside."""
        dropped = sorted(glob.glob(os.path.join(NEW, "*.pt")), key=os.path.getmtime)
        if not dropped:
            return
        sizes = [os.path.getsize(f) for f in dropped]
        time.sleep(2)
        if sizes != [os.path.getsize(f) if os.path.exists(f) else -1 for f in dropped]:
            return  # still being copied in: next map
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for f in dropped[:-1]:  # older drops: never played, kept with the models
            shutil.move(f, os.path.join(MODELS, f"{stamp}-skipped-{os.path.basename(f)}"))
        f = dropped[-1]
        problem = check_weights(f)
        if problem:
            dst = os.path.join(BAD, f"{stamp}-{os.path.basename(f)}")
            shutil.move(f, dst)
            self.session.say(f"[stream] {os.path.basename(f)} is not a student model ({problem}): moved to {dst}; "
                             f"{model_name(self.state['current'])} keeps playing")
            return
        stem = os.path.splitext(os.path.basename(f))[0]  # r007.pt from --export; a plain best.pt has no round in it
        dst = os.path.join(MODELS, f"{time.strftime('%Y%m%d-%H%M')}-{'model' if stem in ('best', 'last') else stem}.pt")
        shutil.move(f, dst)
        old = self.state["current"]
        self.session.player.load(dst)
        self.session.args.ckpt = dst
        self.state["history"].append(dict(model=old, until=time.time()))
        self.state.update(current=dst, since=time.time())
        self.save()
        self.session.say(f"[stream] new model {model_name(dst)} takes over from {model_name(old)} ({dst})")

    def play(self, path):
        try:
            self.check_inbox()
        except Exception as e:  # a model handover must never stop the stream
            self.session.say(f"[stream] checking {NEW} failed: {type(e).__name__}: {e}")
        return self._session_play(path)

    def run(self):
        self.session.say(f"[stream] playing with {self.state['current']}; drop a new model into {os.path.abspath(NEW)}")
        self.session.run()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--export", action="store_true", help="training machine: copy live.improve's current model to runs/stream/outbox/")
    p.add_argument("--foreground", action="store_true",
                   help="play through the virtual tablet with the fork in front (default: in the background)")
    p.add_argument("--record", action="store_true",
                   help="keep the frames of every play (~130 MB a map), to bring back for training")
    p.add_argument("--max-gb", type=float, default=50, help="with --record: stop keeping frames past this much")
    p.add_argument("--click-th", type=float, default=-0.75)
    p.add_argument("--min-hold", type=int, default=5)
    p.add_argument("--view", type=int, default=8770, metavar="PORT", help="the student's view for OBS; 0: off")
    p.add_argument("--view-style", choices=["neuro", "input"], default="neuro")
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # Chinese messages in any console
    if a.export:
        export()
        return
    Streamer(a).run()


if __name__ == "__main__":
    main()
