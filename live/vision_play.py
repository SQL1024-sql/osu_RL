"""The vision student playing lazer from the screen alone: every 1/120 s it takes the frames lazer showed (the 4:3 middle,
160 x 120 grey, as live.record captures them), the cursor drawn where it is, and decides the next move and keys, which
go out through the virtual tablet. Nothing is read from the game but its picture; tosu only says when a map is being
played, so the student is idle in menus.

    python -m live.vision_play --dry-run     # the student decides but sends nothing (cursor: wherever it is)
    python -m live.vision_play               # the student plays
    python -m live.vision_play --view        # ... and streams what it sees for OBS (live.vision_view)

Use the training-look skin (live.student_skin), HUD off, No Fail. A capture thread grabs frames as fast as it can;
the main loop runs at 120 Hz on the newest ones. Every step is logged; the log goes to runs/live_play/ at the end, and
when the map is one with a prepared teacher play (live.prepare), the student's clicks and moves are compared with the
teacher's (in a dry run while the teacher plays through live.screen_play, the cursor is the teacher's).
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
    p.add_argument("--dry-run", action="store_true", help="decide but send no input")
    p.add_argument("--record", action="store_true",
                   help="also save every captured frame as a live.record recording (for live.dagger_label)")
    p.add_argument("--bench", type=int, default=0, help="no tosu, no input: run this many steps on the screen as it is and report the speed")
    p.add_argument("--view", type=int, nargs="?", const=8770, default=None, metavar="PORT",
                   help="serve what the student sees for OBS (Browser source http://127.0.0.1:PORT/; live.vision_view)")
    p.add_argument("--view-raw", action="store_true", help="the view without the move arrow / click ring / key lights")
    p.add_argument("--maps", nargs="*", default=["maps/test/*.osu", "runs/maps/live_set/*.osu"],
                   help="maps with prepared teacher plays, to compare against when one of them is played")
    args = p.parse_args()
    if args.bench:
        args.dry_run = True
    S.CLICK_TH, S.MIN_HOLD = args.click_th, args.min_hold
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = S.Student().to(dev).to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev))
    model.eval()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if args.clock == "bridge":
        from live.bridge import BridgeClock
        clock = BridgeClock()  # the fork's own gameplay clock, judgements and cursor record
    else:
        clock = TosuClock()  # any map: the student only needs to know a map is being played
        clock.start()
    if not args.bench:
        wait_front("lazer")
    field = Playfield()
    x0, y0 = field(X0, Y0)
    box = {"left": round(x0), "top": round(y0), "width": round(W / PX * field.scale), "height": round(H / PX * field.scale)}
    if args.bench and field.scale == 0:  # lazer minimized (fullscreen, another window in front): time on the screen's 4:3 middle
        sw, sh = user32_metrics()
        box = {"left": (sw - sh * 4 // 3) // 2, "top": 0, "width": sh * 4 // 3, "height": sh}
        field.scale = sh / (H / PX)
        field.x0, field.y0 = box["left"] - X0 * field.scale, box["top"] - Y0 * field.scale
    cap = Capture(box)
    cap.start()
    while not cap.ring:
        time.sleep(0.01)
    view = None
    if args.view:
        from live.vision_view import View
        # between maps: the raw capture, but only while lazer is in front (the desktop never goes out on stream)
        view = View(args.view, hud=not args.view_raw, idle=lambda: cap.ring[-1][1] if field.in_front() else None)
        print(f"student's view for OBS: {view.url}", flush=True)
    tablet = None if args.dry_run else Tablet(args.press)
    user32 = ctypes.WinDLL("user32")
    pt = (ctypes.c_long * 2)()

    def os_cursor():
        user32.GetCursorPos(pt)
        return np.array([(pt[0] - field.x0) / field.scale, (pt[1] - field.y0) / field.scale])

    # warm up the model (first CUDA calls are slow)
    with torch.no_grad(), S.autocast(dev):
        model(torch.zeros(1, S.N_FRAMES, H, W, device=dev), torch.zeros(1, S.AUX, device=dev))
    print(f"{'dry run' if args.dry_run else 'playing'}: {args.ckpt}, click threshold {args.click_th}, min hold {args.min_hold}; "
          f"start a map in lazer (Ctrl+C stops)", flush=True)
    lazer_out = None
    if args.map:
        assert args.clock == "bridge", "--map needs --clock bridge"
        lazer_out = os.path.join("runs", "live_play", stamp + ("-dry" if args.dry_run else "") + "-lazer.json")
        print(f"starting {args.map} with {' '.join(args.mods) or 'no mods'}: {clock.play(args.map, args.mods, lazer_out)}", flush=True)

    dt = S.DT / 1000
    log, live = [], False
    cur = os_cursor()
    hist = deque([(time.perf_counter(), cur.copy())], maxlen=16)  # (wall, osu! px) where the cursor was
    held, last, clicked = np.zeros(2, bool), 0, False
    prev, fired, last_fire, step = -np.inf, False, -10 ** 9, 0
    nxt = time.perf_counter()
    try:
        while True:
            playing = bool(args.bench) or (clock.playing and clock.model is not None)
            if args.bench and step >= args.bench:
                break
            if not playing:
                if live:
                    print("map over or paused: idle", flush=True)
                    if tablet:
                        tablet.release()
                    held[:] = False
                    live = False
                    if log:
                        break  # one map per run, so its log can be judged
                time.sleep(0.005)
                nxt = time.perf_counter()
                continue
            if not live:
                print("playing: the student has the cursor" if tablet else "map started: dry run", flush=True)
                live, cur = True, os_cursor()
                if args.record:
                    cap.keep_all = []
                hist.clear()
                hist.append((time.perf_counter(), cur.copy()))
            now = time.perf_counter()
            if args.dry_run:  # the cursor is whoever else moves it
                cur = os_cursor()
                hist.append((now, cur.copy()))
            # frames: the newest, and the ones FRAME_STEPS steps before it
            newest = cap.ring[-1][0]
            want = newest - np.array(S.FRAME_STEPS) * dt
            imgs, shown = cap.frames(want)
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
            if not args.dry_run:
                cur = np.clip(cur + m * MAX_CURSOR_SPEED * S.DT, -CURSOR_MARGIN, CURSOR_MAX)
                try:
                    tablet.send(*field(*cur), buttons)
                except LostFocus:
                    print("osu! lost focus: paused; click back into osu! to resume (Ctrl+C stops)", flush=True)
                    while not field.in_front():
                        time.sleep(0.01)
                    print("resumed", flush=True)
                hist.append((time.perf_counter(), cur.copy()))
            if view and view.want():  # the newest input frame (FRAME_STEPS[0] == 0), copied only as often as it is shown
                view.put(img[0, 0].mul(255).byte().cpu().numpy(), cur, m, held, fire)
            game =(now - clock.model[0]) * 1000 if clock.model is not None else np.nan
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
    except KeyboardInterrupt:
        pass
    finally:
        cap.stop = True
        if tablet:
            tablet.close()
    if not log or args.bench:
        if log:
            arr = np.array(log)
            gaps = np.diff(arr[:, 0]) * 1000
            print(f"bench: {len(arr)} steps at {len(arr) / (arr[-1, 0] - arr[0, 0]):.0f} Hz; step gap median {np.median(gaps):.2f} ms, "
                  f"p99 {np.percentile(gaps, 99):.2f} ms; frame age median {np.median(arr[:, 10]):.1f} ms; capture ring {len(cap.ring)}")
        return
    cols = "wall game x y mx my click hold fire buttons lat newest".split()
    arr = np.array(log, dtype=np.float64)
    os.makedirs(os.path.join("runs", "live_play"), exist_ok=True)
    out = os.path.join("runs", "live_play", stamp + ("-dry" if args.dry_run else "") + ".npz")
    extra = {}
    if args.clock == "bridge":
        clock.done_event.wait(15)  # the fork writes its record of the play when the results come up
        if clock.done:
            d = clock.done
            print(f"lazer: accuracy {d['accuracy']:.2%}, {d['statistics']}" + (f" -> {d['file']}" if d.get("file") else ""), flush=True)
            if d.get("file"):
                extra["lazer"] = d["file"]
    np.savez_compressed(out, log=arr, cols=np.array(cols), md5=str(clock.map_md5), ckpt=args.ckpt, click_th=args.click_th,
                        min_hold=args.min_hold, clock=args.clock, press=args.press, **extra)
    if args.record and cap.keep_all:
        # capture times -> game times with the clock the steps used (wall - game, robust over the play)
        ok = np.isfinite(arr[:, 1])
        icept = float(np.median(arr[ok, 0] - arr[ok, 1] / 1000))
        ct = np.array([f[0] for f in cap.keep_all])
        rec = out[:-4] + "-rec.npz"
        np.savez_compressed(rec, frames=np.stack([f[1] for f in cap.keep_all]), game_ms=(ct - icept) * 1000,
                            md5=str(clock.map_md5), box=np.array([box["left"], box["top"], box["width"], box["height"]]),
                            play=out)
        print(f"recorded {len(ct)} frames -> {rec}", flush=True)
    w = arr[:, 0]
    rate = len(w) / max(1e-9, w[-1] - w[0])
    print(f"{len(arr)} steps ({rate:.0f} Hz), frame age median {np.median(arr[:, 10]):.1f} ms, "
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
