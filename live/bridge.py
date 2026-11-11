"""Talks to our own osu!lazer build (../ppy-osu, osu_rl fork) over its local bridge: start a map with mods, read the
gameplay clock exactly (the game sends (QPC seconds, gameplay ms) pairs taken in the same update, and QPC is
time.perf_counter's clock), and get every judgement plus the game's own record of the cursor when the play ends.

    python -m live.bridge launch                         # start the fork (its own data folder, %APPDATA%/osu-rl)
    python -m live.bridge setup                          # import the live set and the student skin, pick it, hide the HUD
    python -m live.bridge status
    python -m live.bridge play runs/maps/live_set/5_*.osu --mods NF   # start a map (no input sent: you or a player play it)

BridgeClock has TosuClock's interface (model, playing, map_md5), so live.vision_play can use it (--clock bridge).
"""
import argparse
import glob
import hashlib
import itertools
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import zipfile
from collections import deque

import numpy as np

PORT = 24051
EXE = os.path.join("..", "ppy-osu", "osu.Desktop", "bin", "Release", "net10.0", "osu!.exe")
DATA = os.path.join(os.environ.get("APPDATA", ""), "osu-rl")
LIVE_SET = "runs/maps/live_set"
SKIN = "runs/skin/Neuro-sama number 1 (AI) (osu_rl student).osk"
SKIN_NAME = "Neuro-sama number 1 (AI) (osu_rl student)"


def md5_of(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


class BridgeClock(threading.Thread):
    """The connection to the game. call() sends a command and waits for its reply; events update state, the clock
    model (wall = icept + game_s, from the last samples) and, when a play ends, done (its summary) and done_event."""

    def __init__(self, port=PORT, connect_timeout=60):
        super().__init__(daemon=True)
        t0 = time.time()
        while True:
            try:
                self.sock = socket.create_connection(("127.0.0.1", port), timeout=5)
                break
            except OSError:
                if time.time() - t0 > connect_timeout:
                    raise SystemExit(f"no osu_rl bridge on 127.0.0.1:{port}: start the fork with python -m live.bridge launch")
                time.sleep(0.5)
        self.sock.settimeout(None)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.file = self.sock.makefile("rb")
        self.lock = threading.Lock()
        self.ids = itertools.count(1)
        self.replies, self.reply_ready = {}, threading.Condition()
        self.state, self.map_md5, self.done, self.error = None, None, None, None
        self.done_event = threading.Event()
        self.samples = deque(maxlen=64)  # (qpc s, gameplay ms)
        self.judgements = []
        self.start()

    # TosuClock's interface
    @property
    def model(self):
        if not self.samples:
            return None
        s = np.array(self.samples)
        return float(np.median(s[:, 0] - s[:, 1] / 1000)), 1.0

    @property
    def playing(self):
        return self.state == "playing"

    def call(self, timeout=60, **cmd):
        cmd["id"] = i = next(self.ids)
        with self.lock:
            self.sock.sendall((json.dumps(cmd) + "\n").encode())
        with self.reply_ready:
            if not self.reply_ready.wait_for(lambda: i in self.replies, timeout):
                raise TimeoutError(f"no reply to {cmd['cmd']} in {timeout} s")
            return self.replies.pop(i)

    def play(self, osu_path, mods=("NF",), out=None):
        """Start a map; returns once the game took the command (the play itself starts after loading)."""
        self.map_md5 = md5_of(osu_path)
        self.done, self.judgements = None, []
        self.done_event.clear()
        self.samples.clear()
        r = self.call(cmd="play", md5=self.map_md5, mods=list(mods), out=os.path.abspath(out) if out else None)
        if not r.get("ok"):
            raise RuntimeError(f"play {osu_path}: {r.get('error')}")
        return r

    def run(self):
        for line in self.file:
            m = json.loads(line)
            ev = m.get("ev")
            if ev is None:
                with self.reply_ready:
                    self.replies[m.get("id")] = m
                    self.reply_ready.notify_all()
            elif ev == "clock":
                self.samples.append((m["qpc"], m["t"]))
            elif ev == "state":
                self.state = m["state"]
            elif ev == "judge":
                self.judgements.append(m)
            elif ev == "done":
                self.state, self.done = "done", m
                self.done_event.set()
            elif ev == "error":
                self.error = m["error"]
        self.state = None


def launch(port=PORT):
    """Start the fork with its bridge on, copying the installed lazer's settings into its data folder the first time."""
    if not os.path.exists(EXE):
        raise SystemExit(f"{EXE} is missing: dotnet build ../ppy-osu/osu.Desktop -c Release")
    src = os.path.join(os.environ["APPDATA"], "osu")
    os.makedirs(DATA, exist_ok=True)
    for f in ("framework.ini", "game.ini", "input.json"):
        if not os.path.exists(os.path.join(DATA, f)) and os.path.exists(os.path.join(src, f)):
            with open(os.path.join(src, f), "rb") as a, open(os.path.join(DATA, f), "wb") as b:
                b.write(a.read())
    env = dict(os.environ, OSU_RL_BRIDGE=str(port))
    subprocess.Popen([os.path.abspath(EXE)], cwd=os.path.dirname(os.path.abspath(EXE)), env=env,
                     creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["launch", "setup", "status", "play", "menu"])
    p.add_argument("map", nargs="?")
    p.add_argument("--mods", nargs="*", default=["NF"])
    p.add_argument("--out", help="where the game writes the play's judgements, cursor and clock (json)")
    p.add_argument("--port", type=int, default=PORT)
    a = p.parse_args()
    if a.cmd == "launch":
        launch(a.port)
        BridgeClock(a.port)
        print(f"the fork is up, bridge on 127.0.0.1:{a.port}; data in {DATA}")
        return
    b = BridgeClock(a.port)
    if a.cmd == "setup":
        # lazer deletes an .osz / .osk it imported: hand it copies (the set is zipped from the exact .osu files)
        tmp = tempfile.mkdtemp(prefix="osu_rl_import_")
        osz = os.path.join(tmp, "osu_rl live set.osz")
        with zipfile.ZipFile(osz, "w") as z:
            for f in sorted(glob.glob(os.path.join(LIVE_SET, "*.osu"))):
                z.write(f, os.path.basename(f))
        osk = os.path.join(tmp, os.path.basename(SKIN))
        shutil.copy(SKIN, osk)
        print(b.call(cmd="import", paths=[osz, osk], timeout=300))
        time.sleep(2)
        print(b.call(cmd="setup", skin=SKIN_NAME, hud=False))
    elif a.cmd == "status":
        print(b.call(cmd="status"))
    elif a.cmd == "menu":
        print(b.call(cmd="menu"))
    elif a.cmd == "play":
        path = sorted(glob.glob(a.map))[0]
        print(b.play(path, a.mods, a.out))
        last = None
        while not b.done_event.wait(0.5):
            if b.state != last:
                print("state", b.state, flush=True)
                last = b.state
        d = b.done
        print(f"done: accuracy {d['accuracy']:.2%}, {d['statistics']}, file {d['file']}")


if __name__ == "__main__":
    main()
