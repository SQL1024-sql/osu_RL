"""The vision student playing lazer from the screen alone: every 1/120 s it takes the frames lazer showed (the 4:3 middle,
160 x 120 grey, as live.record captures them), the cursor drawn where it is, and decides the next move and keys, which
go out through the virtual tablet. Nothing is read from the game but its picture; the clock (tosu, or our fork's
bridge) only says when a map is being played and its time, so the student is idle in menus.

    python -m live.vision_play --dry-run     # the student decides but sends nothing (cursor: wherever it is)
    python -m live.vision_play               # the student plays
    python -m live.vision_play --view        # ... and streams what it sees for OBS (live.vision_view)
    python -m live.vision_play --clock bridge --map runs/maps/live_set/5_*.osu --record   # one map in the fork, unattended

Use the training-look skin (live.student_skin), HUD off, No Fail. A capture thread grabs frames as fast as it can;
the main loop runs at 120 Hz on the newest ones. Every step is logged; the log goes to runs/live_play/ at the end, and
when the map is one with a prepared teacher play (live.prepare), the student's clicks and moves are compared with the
teacher's (in a dry run while the teacher plays through live.screen_play, the cursor is the teacher's).
VisionPlayer keeps the model, capture and tablet between maps (live.collect plays one map after another with it).
"""
import argparse
import ctypes
import os
import threading
import time
from collections import deque

import cv2
import mss
import numpy as np
import torch

import vision.student as S
from state.env import CURSOR_MARGIN, CURSOR_MAX, MAX_CURSOR_SPEED
from vision.gpu_render import draw_dots
from live.screen_play import Playfield, wait_front
from live.tablet import M1, M2, LostFocus, Tablet
from live.tosu_clock import TosuClock
from vision.render import H, PX, W, X0, Y0

CKPT = "results/experiments/030-real-dagger-mapkeys/seed0/best.pt"
CLICK_TH = -0.75  # the real-frame students click with lower logits (live.vision_eval --click-th sweep: 030 clicks 94% of objects, no stray)
MIN_HOLD = 5  # Exp 031: steps a clicked key stays held (student.MIN_HOLD); 030 sim val 74.8% -> 77.7%, live log re-keyed 63.1% -> 70.0%
COLS = "wall game x y mx my click hold fire buttons lat newest".split()


def user32_metrics():
    m = ctypes.WinDLL("user32").GetSystemMetrics
    return m(0), m(1)


class Capture(threading.Thread):
    """Grabs the frame region as fast as it can: a ring of (capture time, 160 x 120 uint8)."""

    def __init__(self, box, keep=64):
        super().__init__(daemon=True)
        self.box, self.ring, self.stop = box, deque(maxlen=keep), False
        self.keep_all = None  # a list while recording: every (capture time, frame)

    def run(self):
        with mss.MSS() as sct:
            while not self.stop:
                t0 = time.perf_counter()
                img = np.asarray(sct.grab(self.box))
                t1 = time.perf_counter()
                g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY), (W, H), interpolation=cv2.INTER_AREA)
                self.ring.append(((t0 + t1) / 2, g))
                if self.keep_all is not None:
                    self.keep_all.append(self.ring[-1])

    def frames(self, times):
        """The latest frame captured at or before each time (oldest kept if none)."""
        ring = list(self.ring)
        ts = np.array([r[0] for r in ring])
        k = np.clip(np.searchsorted(ts, times, side="right") - 1, 0, len(ring) - 1)
        return np.stack([ring[i][1] for i in k]), ts[k]

    def fps(self):
        ts = [r[0] for r in list(self.ring)]
        return (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 2 and ts[-1] > ts[0] else 0.0


class WindowCapture:
    """Capture's interface over Windows Graphics Capture of one window (windows-capture): the window's own picture,
    also while other windows cover it (not while it is minimized). box: the playfield region in screen pixels."""

    def __init__(self, hwnd, box, keep=64):
        from windows_capture import WindowsCapture
        r = (ctypes.c_long * 4)()
        ctypes.windll.user32.GetWindowRect(ctypes.c_void_p(hwnd), r)
        x, y = box["left"] - r[0], box["top"] - r[1]  # the frame is the window, from its top left corner
        self.crop = (y, y + box["height"], x, x + box["width"])
        self.box, self.ring, self.stop, self.keep_all = box, deque(maxlen=keep), False, None
        self._wgc = WindowsCapture(cursor_capture=False, draw_border=False, window_hwnd=hwnd, minimum_update_interval=1)

        @self._wgc.event
        def on_frame_arrived(frame, control):
            t = time.perf_counter()
            if self.stop:
                control.stop()
                return
            y0, y1, x0, x1 = self.crop
            img = frame.frame_buffer[max(0, y0):y1, max(0, x0):x1]
            if img.shape[0] < 2 or img.shape[1] < 2:
                return
            g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY), (W, H), interpolation=cv2.INTER_AREA)
            self.ring.append((t, g))
            if self.keep_all is not None:
                self.keep_all.append(self.ring[-1])

        @self._wgc.event
        def on_closed():
            self.stop = True

        self._control = None

    def start(self):
        self._control = self._wgc.start_free_threaded()

    def is_alive(self):
        return self._control is not None and not self.stop and not self._control.is_finished()

    frames = Capture.frames
    fps = Capture.fps


class VisionPlayer:
    """The student with its model, screen capture and tablet, kept between maps. attach() finds lazer's playfield
    and starts capturing it; play() plays one map, from when the clock says it started to when it says it ended."""

    def __init__(self, ckpt=CKPT, click_th=CLICK_TH, min_hold=MIN_HOLD, press="first", dry_run=False, view=None,
                 view_raw=False, background=False, view_style="input"):
        """background: our fork only. The window is captured even while covered (WindowCapture) and the input goes
        over the bridge (live.bridge.BridgeInput) instead of the virtual tablet, so nothing needs the game in front."""
        S.CLICK_TH, S.MIN_HOLD = click_th, min_hold
        self.background = background
        self._cur = np.array([256.0, 192.0])  # where the student left the cursor (background: there is no OS cursor)
        self.ckpt, self.click_th, self.min_hold, self.press, self.dry_run = ckpt, click_th, min_hold, press, dry_run
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = S.Student().to(self.dev).to(memory_format=torch.channels_last)
        self.model.load_state_dict(torch.load(ckpt, map_location=self.dev))
        self.model.eval()
        with torch.no_grad(), S.autocast(self.dev):  # warm up (the first CUDA calls are slow)
            self.model(torch.zeros(1, S.N_FRAMES, H, W, device=self.dev), torch.zeros(1, S.AUX, device=self.dev))
        self.view_port, self.view_raw, self.view, self.view_style = view, view_raw, None, view_style
        self.field = self.cap = self.tablet = self.box = None
        self._pt = (ctypes.c_long * 2)()

    def load(self, ckpt):
        """Play with other weights from now on (same network)."""
        self.model.load_state_dict(torch.load(ckpt, map_location=self.dev))
        self.model.eval()
        self.ckpt = ckpt

    def attach(self, bench=False, wait=True):
        """Find lazer's window and playfield (it must be in front unless bench; wait: until it is), start capturing
        it, open the tablet. Again after the window moved or the game restarted: the capture follows the new playfield."""
        if not bench and wait and not self.background:
            wait_front("lazer")
        field = Playfield()
        x0, y0 = field(X0, Y0)
        box = {"left": round(x0), "top": round(y0), "width": round(W / PX * field.scale), "height": round(H / PX * field.scale)}
        if bench and field.scale == 0:  # lazer minimized (fullscreen, another window in front): time on the screen's 4:3 middle
            sw, sh = user32_metrics()
            box = {"left": (sw - sh * 4 // 3) // 2, "top": 0, "width": sh * 4 // 3, "height": sh}
            field.scale = sh / (H / PX)
            field.x0, field.y0 = box["left"] - X0 * field.scale, box["top"] - Y0 * field.scale
        if box != self.box or self.cap is None or not self.cap.is_alive():
            if self.cap:
                self.cap.stop = True
            self.cap = WindowCapture(field.hwnd, box) if self.background else Capture(box)
            self.cap.start()
            t0 = time.time()
            while not self.cap.ring:
                if time.time() - t0 > 10:
                    raise RuntimeError("no frames from the capture in 10 s (minimized window?)")
                time.sleep(0.01)
        self.field, self.box = field, box
        if self.view_port and self.view is None:
            from live.vision_view import View
            # between maps: the raw capture, but only while lazer is in front (the desktop never goes out on stream)
            self.view = View(self.view_port, hud=not self.view_raw, style=self.view_style,
                             idle=lambda: self.cap.ring[-1][1] if self.field.in_front() else None)
            print(f"student's view for OBS: {self.view.url}", flush=True)
        if not self.dry_run and not self.background:  # background: the bridge input is made per play (it needs the clock)
            if self.tablet:
                self.tablet.close()
            self.tablet = Tablet(self.press)

    def os_cursor(self):
        if self.background:
            return self._cur.copy()
        ctypes.windll.user32.GetCursorPos(self._pt)
        return np.array([(self._pt[0] - self.field.x0) / self.field.scale, (self._pt[1] - self.field.y0) / self.field.scale])

    def play(self, clock, record=False, start_timeout=None, bench=0, on_lost_focus=None, max_seconds=None):
        """One map: idle until clock.playing (None if it does not start within start_timeout s), then play until it
        stops (the map ended, or the game paused) or for max_seconds at most. on_lost_focus(): called when the game is
        not in front; returns True once it is back, False to give up on the map. -> dict(log=(n, len(COLS)),
        frames=[(capture time, frame)] if record, lost_focus=bool, timed_out=bool)."""
        t_live, timed_out = None, False
        dev, model, cap, field, tablet = self.dev, self.model, self.cap, self.field, self.tablet
        if self.background and not self.dry_run and not bench:
            from live.bridge import BridgeInput
            tablet = BridgeInput(clock, self.press)  # positions in osu! px, straight into the play
        dt = S.DT / 1000
        log, live, lost = [], False, False
        cur = self.os_cursor()
        hist = deque([(time.perf_counter(), cur.copy())], maxlen=16)  # (wall, osu! px) where the cursor was
        held, last, clicked = np.zeros(2, bool), 0, False
        prev, fired, last_fire, step = -np.inf, False, -10 ** 9, 0
        t_wait = nxt = time.perf_counter()
        try:
            while True:
                playing = bool(bench) or (clock.playing and clock.model is not None)
                if bench and step >= bench:
                    break
                if not playing:
                    if live:
                        if tablet:
                            tablet.release()
                        break  # the map ended or paused
                    if start_timeout is not None and time.perf_counter() - t_wait > start_timeout:
                        return None
                    time.sleep(0.005)
                    nxt = time.perf_counter()
                    continue
                if live and max_seconds is not None and time.perf_counter() - t_live > max_seconds:
                    timed_out = True
                    break
                if not live:
                    live, cur, t_live = True, self.os_cursor(), time.perf_counter()
                    if record:
                        cap.keep_all = []
                    hist.clear()
                    hist.append((time.perf_counter(), cur.copy()))
                now = time.perf_counter()
                if self.dry_run:  # the cursor is whoever else moves it
                    cur = self.os_cursor()
                    hist.append((now, cur.copy()))
                # frames: the newest, and the ones FRAME_STEPS steps before it
                newest = cap.ring[-1][0]
                imgs, shown = cap.frames(newest - np.array(S.FRAME_STEPS) * dt)
                lat_ms = float(np.clip((now - newest) * 1000, 0, S.MAX_LAT))
                hs_t = np.array([h[0] for h in hist])
                dots = np.array([hist[int(np.clip(np.searchsorted(hs_t, s, side="right") - 1, 0, len(hist) - 1))][1] for s in shown])
                img = torch.tensor(imgs, device=dev).float().div_(255).view(S.N_FRAMES, H, W)
                d = torch.tensor(np.c_[dots, np.full(S.N_FRAMES, 2.2)], dtype=torch.float32, device=dev)
                img = draw_dots(img, d).view(1, S.N_FRAMES, H, W)
                aux = torch.tensor([[cur[0] / 512, cur[1] / 384, lat_ms / S.MAX_LAT]], dtype=torch.float32, device=dev)
                with torch.no_grad(), S.autocast(dev):
                    m, kl = model(img, aux)
                m, (c, h) = m.float().cpu().numpy()[0], kl.float().cpu().numpy()[0]
                # keys exactly as student.rollouts: one click per peak of the click logit above the threshold
                fire = prev > S.CLICK_TH and c < prev and not fired and step - last_fire >= S.CLICK_GAP
                if fire:
                    last_fire = step
                fired = (fired or fire) and c > S.CLICK_TH
                prev = c
                act, last, clicked = S.keys_for(fire, h > 0 or step - last_fire <= S.MIN_HOLD, held, last, clicked)
                held = act > 0
                buttons = (M1 if held[0] else 0) | (M2 if held[1] else 0)
                if not self.dry_run:
                    cur = np.clip(cur + m * MAX_CURSOR_SPEED * S.DT, -CURSOR_MARGIN, CURSOR_MAX)
                    try:
                        tablet.send(*(cur if self.background else field(*cur)), buttons)
                    except LostFocus:
                        held[:] = False
                        lost = True
                        if on_lost_focus is None:
                            print("osu! lost focus: paused; click back into osu! to resume (Ctrl+C stops)", flush=True)
                            while not field.in_front():
                                time.sleep(0.01)
                            print("resumed", flush=True)
                        elif not on_lost_focus():
                            break
                    hist.append((time.perf_counter(), cur.copy()))
                game = (now - clock.model[0]) * 1000 if clock is not None and clock.model is not None else np.nan
                if self.view and self.view.want():  # the newest input frame, copied only as often as it is shown
                    self.view.put(img[0, 0].mul(255).byte().cpu().numpy(), cur, m, held, fire, game)
                log.append((now, game, *cur, *m, c, h, fire, buttons, lat_ms, newest))
                step += 1
                nxt += dt
                wait = nxt - time.perf_counter()
                if wait > 0.0015:
                    time.sleep(wait - 0.001)
                while time.perf_counter() < nxt:
                    pass
                if wait < -0.05:
                    nxt = time.perf_counter()  # fell behind: do not try to catch up
        finally:
            if tablet:
                tablet.release()
            frames, cap.keep_all = cap.keep_all, None
            self._cur = np.array(cur, float)
        return dict(log=np.array(log, dtype=np.float64).reshape(-1, len(COLS)), frames=frames if record else None,
                    lost_focus=lost, timed_out=timed_out)

    def save(self, result, out, md5, clock_name, extra=None):
        """Writes a play's log to out (.npz) and, if it was recorded, its frames to <out>-rec.npz. -> rec path or None."""
        arr = result["log"]
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        np.savez_compressed(out, log=arr, cols=np.array(COLS), md5=str(md5), ckpt=self.ckpt, click_th=self.click_th,
                            min_hold=self.min_hold, clock=clock_name, press=self.press, **(extra or {}))
        frames = result.get("frames")
        if not frames:
            return None
        # capture times -> game times with the clock the steps used (wall - game, robust over the play)
        ok = np.isfinite(arr[:, 1])
        icept = float(np.median(arr[ok, 0] - arr[ok, 1] / 1000))
        ct = np.array([f[0] for f in frames])
        rec = out[:-4] + "-rec.npz"
        np.savez_compressed(rec, frames=np.stack([f[1] for f in frames]), game_ms=(ct - icept) * 1000, md5=str(md5),
                            box=np.array([self.box["left"], self.box["top"], self.box["width"], self.box["height"]]), play=out)
        return rec

    def close(self):
        if self.cap:
            self.cap.stop = True
            self.cap = None
        if self.tablet:
            self.tablet.close()
            self.tablet = None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--click-th", type=float, default=CLICK_TH)
    p.add_argument("--min-hold", type=int, default=MIN_HOLD,
                   help="steps a clicked key stays held before the hold logit decides (Exp 031; 0: as before)")
    p.add_argument("--clock", choices=["tosu", "bridge"], default="tosu",
                   help="where the game clock comes from: tosu (installed lazer) or our fork's bridge (live.bridge)")
    p.add_argument("--map", help="--clock bridge: start this .osu in the fork once the student is ready")
    p.add_argument("--mods", nargs="*", default=["NF"], help="--map: mods to play it with")
    p.add_argument("--press", choices=["first", "after", "same"], default="first",
                   help="a step where a key goes down: press where the cursor was, then move (first), the other way round"
                        " (after), or one report (same: lazer picks either way); live.tablet.Tablet")
    p.add_argument("--background", action="store_true",
                   help="our fork only (--clock bridge): capture its window even when covered, input over the bridge; "
                        "the computer stays free to use")
    p.add_argument("--dry-run", action="store_true", help="decide but send no input")
    p.add_argument("--record", action="store_true",
                   help="also save every captured frame as a live.record recording (for live.dagger_label)")
    p.add_argument("--bench", type=int, default=0, help="no clock, no input: run this many steps on the screen as it is and report the speed")
    p.add_argument("--view", type=int, nargs="?", const=8770, default=None, metavar="PORT",
                   help="serve what the student sees for OBS (Browser source http://127.0.0.1:PORT/; live.vision_view)")
    p.add_argument("--view-raw", action="store_true", help="the view without the move arrow / click ring / key lights")
    p.add_argument("--maps", nargs="*", default=["maps/test/*.osu", "runs/maps/live_set/*.osu"],
                   help="maps with prepared teacher plays, to compare against when one of them is played")
    args = p.parse_args()
    if args.bench:
        args.dry_run = True
    assert not args.background or args.clock == "bridge", "--background needs --clock bridge (our fork)"
    player = VisionPlayer(args.ckpt, args.click_th, args.min_hold, args.press, args.dry_run, args.view, args.view_raw,
                          background=args.background)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    clock = None
    if not args.bench:
        if args.clock == "bridge":
            from live.bridge import BridgeClock
            clock = BridgeClock()  # the fork's own gameplay clock, judgements and cursor record
        else:
            clock = TosuClock()  # any map: the student only needs to know a map is being played
            clock.start()
    player.attach(bench=bool(args.bench))
    print(f"{'dry run' if args.dry_run else 'playing'}: {args.ckpt}, click threshold {args.click_th}, min hold {args.min_hold}, "
          f"press {args.press}; capture {player.cap.fps():.0f} fps; start a map in lazer (Ctrl+C stops)", flush=True)
    if args.map:
        assert args.clock == "bridge", "--map needs --clock bridge"
        lazer_out = os.path.join("runs", "live_play", stamp + ("-dry" if args.dry_run else "") + "-lazer.json")
        print(f"starting {args.map} with {' '.join(args.mods) or 'no mods'}: {clock.play(args.map, args.mods, lazer_out, 'bridge' if args.background else None)}", flush=True)
    try:
        result = player.play(clock, record=args.record, bench=args.bench)
    except KeyboardInterrupt:
        result = None
    finally:
        player.close()
    if result is None or not len(result["log"]):
        return
    arr = result["log"]
    if args.bench:
        gaps = np.diff(arr[:, 0]) * 1000
        print(f"bench: {len(arr)} steps at {len(arr) / (arr[-1, 0] - arr[0, 0]):.0f} Hz; step gap median {np.median(gaps):.2f} ms, "
              f"p99 {np.percentile(gaps, 99):.2f} ms; frame age median {np.median(arr[:, 10]):.1f} ms")
        return
    print("map over or paused: idle", flush=True)
    out = os.path.join("runs", "live_play", stamp + ("-dry" if args.dry_run else "") + ".npz")
    extra = {}
    if args.clock == "bridge":
        clock.done_event.wait(15)  # the fork writes its record of the play when the results come up
        if clock.done:
            d = clock.done
            print(f"lazer: accuracy {d['accuracy']:.2%}, {d['statistics']}" + (f" -> {d['file']}" if d.get("file") else ""), flush=True)
            if d.get("file"):
                extra["lazer"] = d["file"]
    if args.background:
        extra["input"] = "bridge"
    rec = player.save(result, out, clock.map_md5, args.clock, extra)
    if rec:
        print(f"recorded {len(result['frames'])} frames -> {rec}", flush=True)
    w = arr[:, 0]
    print(f"{len(arr)} steps ({len(w) / max(1e-9, w[-1] - w[0]):.0f} Hz), frame age median {np.median(arr[:, 10]):.1f} ms, "
          f"{int(arr[:, 8].sum())} clicks -> {out}", flush=True)
    judge(arr, str(clock.map_md5), args.maps)


def judge(arr, md5, map_globs):
    """Against the teacher's prepared play of the same map: its key-downs found by the student's clicks within 3
    steps, the student's extra clicks, and (dry run) how the student's moves agree with the teacher's."""
    from live.dagger_label import map_index
    from live.vision_eval import at, teacher_track
    index = map_index(map_globs)
    if md5 not in index:
        print("no prepared teacher play of this map to compare with")
        return
    tt, pos, keys = teacher_track(open(index[md5], "rb").read(), "expert")
    game = arr[:, 1]
    ok = (game > tt[0]) & (game < tt[-1])
    if ok.sum() < 2:
        print("the play does not overlap the teacher's play of this map: nothing to compare")
        return
    steps = np.arange(game[ok][0], game[ok][-1], S.DT)
    k0, k1 = at(tt, keys, steps), at(tt, keys, steps + S.DT)
    t_ev = steps[(k1 & ~k0) != 0]
    s_ev = game[arr[:, 8] > 0]
    tol = 3 * S.DT
    found = np.mean([np.any(np.abs(s_ev - t) <= tol) for t in t_ev]) if len(t_ev) else np.nan
    extra = int(sum(not np.any(np.abs(t_ev - s) <= tol) for s in s_ev))
    nxt = at(tt, pos, game[ok] + S.DT)
    move = np.clip((nxt - at(tt, pos, game[ok])) / (MAX_CURSOR_SPEED * S.DT), -1, 1)
    m = arr[ok][:, 4:6]
    moving = np.hypot(*move.T) > 0.05
    cos = (m * move).sum(1)[moving] / (np.hypot(*m[moving].T) * np.hypot(*move[moving].T) + 1e-9)
    print(f"vs the teacher on {os.path.basename(index[md5])}: clicks found {found:.1%} of {len(t_ev)}, "
          f"extra {extra} of {len(s_ev)}; move direction within 45 deg {np.mean(cos > 0.7):.1%}")


if __name__ == "__main__":
    main()
