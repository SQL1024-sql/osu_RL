"""Records what lazer shows during a play, as the vision student sees it: the 4:3 middle of the screen (osu! pixels
-64..576 x -56..424, render.py's frame) in grayscale at 160 x 120, each frame with its game time from tosu. Sends no
input, so it can run beside live.screen_play (the teacher playing) to collect real frames with the teacher's moves.

    python -m live.record maps/test/renatus_hard.osu          # tosu.exe must be running; Ctrl+C or map end stops

Writes runs/live_rec/<map>-<time>.npz: frames (N, 120, 160) uint8, game_ms (N,) when each was captured (the middle
of the grab), and the map's MD5. Capture runs as fast as GDI allows (about 70 fps at 1440 x 1080; the screen itself
may refresh slower).
"""
import argparse
import ctypes
import hashlib
import os
import time

import cv2
import mss
import numpy as np

from beatmap import parse_osu
from live.screen_play import Playfield, wait_front
from live.tosu_clock import TosuClock
from vision.render import H, PX, W, X0, Y0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("map")
    p.add_argument("--out", help="default: runs/live_rec/<map>-<time>.npz")
    args = p.parse_args()
    osu = open(args.map, "rb").read()
    bm = parse_osu(args.map)
    end_ms = float(bm.ends.max()) + 1000
    out = args.out or os.path.join("runs", "live_rec", f"{os.path.basename(args.map)[:-4]}-{time.strftime('%Y%m%d-%H%M%S')}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    clock = TosuClock(hashlib.md5(osu).hexdigest())
    clock.start()
    wait_front("song select of this map")
    f = Playfield()
    x0, y0 = f(X0, Y0)
    box = {"left": round(x0), "top": round(y0), "width": round(W / PX * f.scale), "height": round(H / PX * f.scale)}
    print(f"recording {box} -> {W}x{H}; start the map in lazer (Ctrl+C stops)", flush=True)

    frames, game_ms, started = [], [], False
    try:
        with mss.MSS() as sct:
            while True:
                model = clock.model
                if model is None:
                    if started and not clock.playing:
                        break  # left the map after recording it
                    time.sleep(0.002)
                    continue
                if not started:
                    print("game clock found: recording", flush=True)
                    started = True
                t0 = time.perf_counter()
                img = np.asarray(sct.grab(box))
                t1 = time.perf_counter()
                if not f.in_front():
                    continue  # another window covers lazer: not the game
                g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY), (W, H), interpolation=cv2.INTER_AREA)
                game = ((t0 + t1) / 2 - model[0]) / (1 + model[1]) * 1000
                frames.append(g)
                game_ms.append(game)
                if len(frames) % 600 == 0:
                    print(f"  {len(frames)} frames, game {game / 1000:.1f} s", flush=True)
                if game > end_ms:
                    break
    except KeyboardInterrupt:
        pass
    if not frames:
        raise SystemExit("nothing recorded")
    np.savez_compressed(out, frames=np.stack(frames), game_ms=np.array(game_ms), md5=hashlib.md5(osu).hexdigest(),
                        box=np.array([box["left"], box["top"], box["width"], box["height"]]))
    g = np.array(game_ms)
    print(f"saved {len(frames)} frames ({g[0] / 1000:.1f}-{g[-1] / 1000:.1f} s game, {len(g) / max(1e-9, (g[-1] - g[0]) / 1000):.0f} fps)"
          f" -> {out}", flush=True)


if __name__ == "__main__":
    main()
