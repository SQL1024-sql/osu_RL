"""The student playing one map after another in our lazer fork, unattended: the base of a 24-hour stream and of the
data the student keeps learning from.

    python -m live.collect                                   # the live set, round and round, until Ctrl+C
    python -m live.collect --maps "runs/maps/live_set/[1-4]*.osu" --hours 2 --shuffle

Each play goes to runs/collect/<session>/: the student's log (.npz), its frames (-rec.npz, for live.dagger_label;
--no-record to skip, ~20 MB a map) and the game's own record of it (-lazer.json: every judgement, the cursor, the
clock). plays.jsonl gets one line per map (accuracy, statistics, capture speed, what went wrong), status.json the
state now (for a stream overlay).

What it looks after, so nobody has to:
- the fork: started if it is not running (live.bridge.launch), started again if it quits or crashes;
- the window: if something takes the focus, the game is asked to come back (bridge "focus"); a map that was
  interrupted is abandoned and logged;
- the screen capture: a map starts only if the playfield is captured at --min-fps or more (it drops to 30 fps when
  the display is off or not driven), otherwise it waits and tries again;
- maps that do not start, plays that never end (longer than the map plus a margin), and errors: logged, and the
  next map is played;
- Windows: kept from sleeping or turning the display off while it runs.
"""
import argparse
import ctypes
import glob
import json
import os
import re
import sys
import random
import tempfile
import time
import traceback
import zipfile

import numpy as np

import vision.student as S
import live.tablet
from live.bridge import INSTANCE, SKIN_NAME, BridgeClock, import_skin, launch, md5_of
from live.tablet import is_fork, lazer_window, window_exe
from live.vision_play import CKPT, CLICK_TH, MIN_HOLD, VisionPlayer

ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x1, 0x2


class _MemCounters(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t), ("PrivateUsage", ctypes.c_size_t)]


def private_mb(pid=None):
    """A process's private memory in MB (this process if pid is None); None if it cannot be read."""
    k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    k32.OpenProcess.restype = ctypes.c_void_p
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    h = k32.GetCurrentProcess() if pid is None else k32.OpenProcess(0x1000 | 0x0010, False, pid)
    if not h:
        return None
    try:
        c = _MemCounters()
        c.cb = ctypes.sizeof(c)
        if not psapi.GetProcessMemoryInfo(ctypes.c_void_p(h), ctypes.byref(c), c.cb):
            return None
        return round(c.PrivateUsage / 2 ** 20)
    finally:
        if pid is not None:
            k32.CloseHandle(ctypes.c_void_p(h))


def fork_pid():
    """The fork's process id, or None: never the installed lazer's (its window is only taken if its program lives
    is live.bridge.EXE)."""
    try:
        h = lazer_window()
    except SystemExit:
        return None
    if not is_fork(window_exe(h.value)):
        return None
    pid = ctypes.c_ulong()
    ctypes.windll.user32.GetWindowThreadProcessId(h, ctypes.byref(pid))
    return pid.value


def keep_awake(on=True):
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED if on else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(flags)


def map_title(path):
    """The song and difficulty of one of our .osu files ("pool train - Title Version (id)" -> "Title Version")."""
    try:
        with open(path, encoding="utf-8-sig", errors="replace") as f:
            ver = next((l[8:].strip() for l in f if l.startswith("Version:")), "")
    except OSError:
        ver = ""
    ver = ver.split(" - ", 1)[-1]
    return re.sub(r"\s*\(\d+\)$", "", ver) or os.path.basename(path)[:-4]


class Session:
    def __init__(self, args):
        self.args = args
        self.dir = os.path.join(args.out, time.strftime("%Y%m%d-%H%M%S") + (f"-i{INSTANCE}" if INSTANCE else ""))
        os.makedirs(self.dir, exist_ok=True)
        self.started = time.time()
        self.plays, self.clock = [], None
        self.bytes = 0  # written into the session folder so far
        self.recording = args.record
        self.player = VisionPlayer(args.ckpt, args.click_th, args.min_hold, args.press, view=getattr(args, "view", None),
                                   view_style=getattr(args, "view_style", "input"),
                                   background=args.background)
        self.overlay_head = lambda: []  # -> first lines of the overlay text (live.improve: the round, what it does)
        self.say(f"session {self.dir}: model {args.ckpt}, click threshold {args.click_th}, min hold {args.min_hold}, press {args.press}")

    def say(self, text):
        line = f"[{time.strftime('%H:%M:%S')}] {text}"
        print(line, flush=True)
        with open(os.path.join(self.dir, "session.log"), "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def status(self, state, **more):
        accs = [p["accuracy"] for p in self.plays if p.get("accuracy") is not None]
        s = dict(state=state, session=self.dir, model=self.args.ckpt, since=self.started, now=time.time(),
                 plays=len(self.plays), completed=len(accs), mean_accuracy_last10=float(np.mean(accs[-10:])) if accs else None,
                 last=self.plays[-1] if self.plays else None, **more)
        tmp = os.path.join(self.dir, "status.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=1)
        os.replace(tmp, os.path.join(self.dir, "status.json"))
        if getattr(self.args, "overlay", None):
            self.write_overlay(state, more.get("map"), accs)

    def write_overlay(self, state, map_path, accs):
        """A few lines for the stream (OBS: Text source, "Read from file"), rewritten whenever the state changes."""
        lines = list(self.overlay_head())
        if state == "playing" and map_path:
            lines.append(f"正在打：{map_title(map_path)}")
        elif state == "waiting":
            lines.append("暫停中（遊戲視窗被縮小了）")
        if accs:
            lines.append(f"上一首：{accs[-1]:.2%}")
            lines.append(f"最近 {min(10, len(accs))} 首平均：{np.mean(accs[-10:]):.2%}")
        tmp = self.args.overlay + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(tmp, self.args.overlay)

    # ------------------------------------------------------------- the game
    def connect(self):
        """A live connection to the fork, starting it (again) if needed, with the playfield captured."""
        if self.clock is not None and self.clock.connected:
            return
        try:
            self.clock = BridgeClock(connect_timeout=3)
        except SystemExit:
            self.say("the fork is not running: starting it")
            launch()
            self.clock = BridgeClock(connect_timeout=180)
            time.sleep(10)  # the intro
        st = self.clock.call(cmd="status")
        live.tablet.FORK_PID = st.get("pid")  # several copies may run: capture and watch only this one's window
        self.say(f"connected to the fork: {st}")
        # the student's skin, no HUD, a borderless window (stays up behind other windows; fullscreen minimizes); in the
        # background nobody is at the mouse, so no menu cursor follows it on the stream
        setup = dict(cmd="setup", skin=SKIN_NAME, window="borderless", hud=False)
        if self.args.background:
            setup["cursor"] = False
        r = self.clock.call(**setup)
        if not r.get("ok") and "no skin" in str(r.get("error")):  # a fresh data folder: import it once
            self.say(f"importing the student skin: {import_skin(self.clock)}")
            r = self.clock.call(**setup)
        if not r.get("ok"):
            self.say(f"setup failed: {r}")

    @staticmethod
    def in_front():
        try:
            return ctypes.windll.user32.GetForegroundWindow() == lazer_window().value
        except SystemExit:  # no window yet
            return False

    def focus(self, wait=10.0, tries=3):
        """The fork in front (asking it to come back if it is not). -> bool"""
        for _ in range(tries):
            if self.in_front():
                return True
            try:
                r = self.clock.call(cmd="focus", timeout=10)
                if not r.get("front"):
                    self.say(f"asked the fork to come in front: {r}")
            except Exception as e:
                self.say(f"focus request failed: {e}")
            t0 = time.time()
            while time.time() - t0 < wait / tries:
                if self.in_front():
                    return True
                time.sleep(0.1)
        return False

    def import_map(self, path):
        """lazer deletes what it imported: give it a zipped copy of the .osu."""
        tmp = tempfile.mkdtemp(prefix="osu_rl_import_")
        osz = os.path.join(tmp, os.path.basename(path)[:-4] + ".osz")
        with zipfile.ZipFile(osz, "w") as z:
            z.write(path, os.path.basename(path))
        self.say(f"importing {os.path.basename(path)}: {self.clock.call(cmd='import', paths=[osz], timeout=300)}")
        time.sleep(2)

    # ------------------------------------------------------------- one map
    def play(self, path):
        a = self.args
        name = os.path.basename(path)[:-4]
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out = os.path.join(self.dir, f"{stamp}-{name[:40]}.npz")
        lazer_out = out[:-4] + "-lazer.json"
        row = dict(time=stamp, map=path, md5=md5_of(path), model=a.ckpt, out=out)

        self.connect()
        if not a.background and not self.focus():
            row["error"] = "the fork could not be brought in front"
            return row
        self.player.attach(wait=False)
        if self.player.view is not None:
            self.player.view.set_map(path)  # the "neuro" view draws the map itself
        time.sleep(0.5)
        fps = self.player.cap.fps()
        for _ in range(3):  # just after the game started or the capture restarted it can be briefly slow
            if fps >= a.min_fps:
                break
            time.sleep(2)
            fps = self.player.cap.fps()
        row["capture_fps"] = round(fps, 1)
        if fps < a.min_fps:
            row["error"] = f"capture at {fps:.0f} fps (< {a.min_fps}): display off or not driven?"
            return row

        try:
            self.clock.play(path, a.mods, lazer_out, "bridge" if a.background else None)
        except RuntimeError as e:
            if "no beatmap" not in str(e):
                raise
            self.import_map(path)
            self.clock.play(path, a.mods, lazer_out, "bridge" if a.background else None)
        length = (S.scene(path).end.max() - min(0, S.scene(path).start.min())) / 1000
        self.status("playing", map=path)
        self.say(f"playing {name} ({length:.0f} s)")

        def lost():
            self.say("the fork lost the focus mid-map: asking it back; the map is abandoned")
            return self.focus()

        result = self.player.play(self.clock, record=self.recording, start_timeout=60, on_lost_focus=lost,
                                  max_seconds=length + 30)
        if result is None:
            row["error"] = "the map did not start within 60 s"
            self.clock.call(cmd="menu")
            return row
        self.clock.done_event.wait(15)
        done = self.clock.done
        if result["lost_focus"] or result["timed_out"] or not done or done.get("quit"):
            self.clock.call(cmd="menu")  # leave the paused or stuck play
            self.clock.done_event.wait(10)
            done = self.clock.done
        arr = result["log"]
        extra = {"lazer": done["file"]} if done and done.get("file") else {}
        rec = self.player.save(result, out, row["md5"], "bridge", extra) if len(arr) else None
        row.update(steps=len(arr), rec=rec, lost_focus=result["lost_focus"], timed_out=result["timed_out"])
        frames = result.get("frames")
        if frames and len(frames) > 2:  # how fast the playfield was captured during the play itself
            row["play_capture_fps"] = round((len(frames) - 1) / (frames[-1][0] - frames[0][0]), 1)
        if len(arr):
            w = arr[:, 0]
            row.update(step_hz=round(len(w) / max(1e-9, w[-1] - w[0]), 1), frame_age_ms=round(float(np.median(arr[:, 10])), 1),
                       clicks=int(arr[:, 8].sum()))
        if done:
            row.update(accuracy=done["accuracy"], statistics=done["statistics"], quit=done.get("quit"), lazer=done.get("file"))
        for f in (out, rec, done.get("file") if done else None):
            if f and os.path.exists(f):
                self.bytes += os.path.getsize(f)
        row.update(session_gb=round(self.bytes / 2 ** 30, 2), python_mb=private_mb(), fork_mb=private_mb(fork_pid()))
        if result["lost_focus"] or result["timed_out"] or (done and done.get("quit")):
            row["accuracy"] = None  # an interrupted play is not a score
            row["error"] = "interrupted (focus lost)" if result["lost_focus"] else ("timed out" if result["timed_out"] else "quit")
        return row

    def housekeeping(self, row):
        """Between maps: stop keeping frames past --max-gb (nothing is deleted), and start the fork afresh when its
        memory grew past --restart-fork-mb (a leak would otherwise end a long run)."""
        a = self.args
        if self.recording and a.max_gb and self.bytes > a.max_gb * 2 ** 30:
            self.recording = False
            self.say(f"the session holds {self.bytes / 2 ** 30:.1f} GB (> --max-gb {a.max_gb}): frames are no longer kept")
        mb = row.get("fork_mb")
        if mb and a.restart_fork_mb and mb > a.restart_fork_mb and self.clock is not None:
            self.say(f"the fork uses {mb} MB (> {a.restart_fork_mb}): restarting it")
            pid = fork_pid()
            if pid:
                os.kill(pid, 9)
            self.clock = None

    def run(self):
        a = self.args
        maps = sorted({f for g in a.maps for f in glob.glob(g)})
        assert maps, f"no maps in {a.maps}"
        self.say(f"{len(maps)} maps: " + ", ".join(os.path.basename(m) for m in maps))
        end = self.started + a.hours * 3600 if a.hours else None
        keep_awake(True)
        fails = 0
        try:
            while True:
                order = random.sample(maps, len(maps)) if a.shuffle else maps
                for path in order:
                    if (end and time.time() > end) or (a.plays and len(self.plays) >= a.plays):
                        return
                    try:
                        row = self.play(path)
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        row = dict(time=time.strftime("%Y%m%d-%H%M%S"), map=path, error=f"{type(e).__name__}: {e}")
                        self.say(traceback.format_exc())
                        self.clock = None if self.clock is None or not self.clock.connected else self.clock
                    self.plays.append(row)
                    with open(os.path.join(self.dir, "plays.jsonl"), "a", encoding="utf-8") as f:
                        f.write(json.dumps(row) + "\n")
                    if row.get("accuracy") is not None:
                        fails = 0
                        s = row["statistics"]
                        self.say(f"{os.path.basename(path)}: {row['accuracy']:.2%} (miss {s.get('Miss', 0)}, slider ends "
                                 f"{s.get('SliderTailHit', 0)}), capture {row.get('capture_fps')} fps, frame age {row.get('frame_age_ms')} ms")
                    else:
                        fails += 1
                        self.say(f"{os.path.basename(path)}: no score: {row.get('error')} ({fails} in a row)")
                        time.sleep(min(300, 10 * fails))  # back off while something is wrong
                    self.housekeeping(row)
                    self.status("between maps")
        except KeyboardInterrupt:
            self.say("stopped (Ctrl+C)")
        finally:
            keep_awake(False)
            self.player.close()
            self.status("stopped")
            accs = [p["accuracy"] for p in self.plays if p.get("accuracy") is not None]
            self.say(f"{len(self.plays)} maps, {len(accs)} scored" + (f", mean accuracy {np.mean(accs):.2%}" if accs else ""))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # map names and messages may not be ASCII
    p = argparse.ArgumentParser()
    p.add_argument("--maps", nargs="*", default=["runs/maps/live_set/*.osu"], help=".osu files (globs) to play in turn")
    p.add_argument("--shuffle", action="store_true", help="a new random order every round")
    p.add_argument("--hours", type=float, default=0, help="stop after this long (0: until Ctrl+C)")
    p.add_argument("--plays", type=int, default=0, help="stop after this many maps (0: no limit)")
    p.add_argument("--mods", nargs="*", default=["NF"])
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--click-th", type=float, default=CLICK_TH)
    p.add_argument("--min-hold", type=int, default=MIN_HOLD)
    p.add_argument("--press", choices=["first", "after", "same"], default="first")
    p.add_argument("--no-record", dest="record", action="store_false", help="do not keep the frames")
    p.add_argument("--min-fps", type=float, default=50,
                   help="capture speed a map needs to start (measured on the results screen, where it runs lower than in "
                        "play; the failure it catches is ~30 fps)")
    p.add_argument("--max-gb", type=float, default=0,
                   help="stop keeping frames once the session folder holds this much (0: no limit; nothing is deleted)")
    p.add_argument("--restart-fork-mb", type=float, default=6000,
                   help="restart the fork between maps when its private memory passes this (0: never)")
    p.add_argument("--background", action="store_true",
                   help="play behind other windows: capture the fork's window itself, input over the bridge (the "
                        "computer stays free to use)")
    p.add_argument("--out", default="runs/collect")
    Session(p.parse_args()).run()


if __name__ == "__main__":
    main()
