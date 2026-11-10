"""The game clock read from tosu (https://github.com/tosuapp/tosu), which reads lazer's memory: whether a map is being
played, which one (MD5), and its current time, instead of guessing all of that from the screen.

    python -m live.tosu_clock maps/test/renatus_hard.osu      # no input: prints what it reads while you play the map

tosu serves ws://127.0.0.1:24050/websocket/v2 (state, beatmap with time.live in ms, about every 150 ms) and
/websocket/v2/precise (keys and hit errors, about every 10 ms; tosu 4.26 sends no currentTime there for lazer).
Messages arrive a little after tosu read the time, never before, so wall - game is the true offset plus a delay that
is never negative (10-20 ms here): the clock keeps a low percentile of the recent offsets, which tracks the true one.
"""
import argparse
import asyncio
import hashlib
import json
import threading
import time
from collections import deque

import aiohttp
import numpy as np

URL = "ws://127.0.0.1:24050/websocket"
PLAYING = 2  # tosu's state number for gameplay
JUMP = 0.25  # s: an offset this far from the clock means a seek, pause or restart; start over
ARM_AFTER = 5.0  # s of gameplay state after which a time far from the start is trusted (the preview lasts ~1 s)


class TosuClock(threading.Thread):
    """Same use as screen_play.Clock: model is None until the game clock is known, then at(game_s) + game_s is the
    wall clock (time.perf_counter) of game time game_s. state, md5 and keys follow tosu."""

    def __init__(self, md5=None):
        super().__init__(daemon=True)
        self.md5 = md5  # the map that must be playing, or None for any
        self.state, self.map_md5, self.map_name, self.keys, self.hit_errors = None, None, "", {}, []
        self.offsets = deque(maxlen=400)  # (wall - game) seconds, the last few seconds of play
        self.model, self.error = None, None
        self.last_game, self.armed, self.play_since = None, False, None

    @property
    def offset(self):
        return None if self.model is None else self.model[0]

    def at(self, game):
        return self.model[0] + self.model[1] * game

    @property
    def playing(self):
        return self.state == PLAYING and (self.md5 is None or self.map_md5 == self.md5)

    def run(self):
        while True:  # tosu not running yet, restarted, ...: keep trying; the owner sees the last failure in .error
            try:
                asyncio.run(self._main())
            except Exception as e:
                self.error = e
            self.state, self.model = None, None
            self.offsets.clear()
            time.sleep(1)

    async def _main(self):
        async with aiohttp.ClientSession() as s:
            await asyncio.gather(self._v2(s), self._precise(s))

    async def _v2(self, s):
        async with s.ws_connect(URL + "/v2") as ws:
            self.error = None
            async for msg in ws:
                wall = time.perf_counter()
                d = json.loads(msg.data)
                self.state = d.get("state", {}).get("number")
                b = d.get("beatmap", {})
                self.map_md5 = b.get("checksum")
                self.map_name = f"{b.get('artist')} - {b.get('title')} [{b.get('version')}]"
                if not self.playing:
                    self.offsets.clear()
                    self.model, self.last_game, self.armed, self.play_since = None, None, False, None
                    continue
                if self.play_since is None:
                    self.play_since = wall
                t = b.get("time", {})
                game, first = t.get("live"), t.get("firstObject") or 0
                if game is None:
                    continue
                # gameplay starts while time.live still shows the song select preview (say 169 s into the song) for
                # about a second, and only then jumps back to the start of the map: no clock until the time is seen
                # near the start, or (joined in the middle) after ARM_AFTER seconds of play
                if not self.armed:
                    if game > first + 2000 and wall - self.play_since < ARM_AFTER:
                        continue
                    self.armed = True
                game /= 1000
                if game == self.last_game:
                    continue  # the same time read again (paused, or loading): says nothing about the clock
                self.last_game = game
                off = wall - game
                if self.offsets and abs(off - self.offsets[-1]) > JUMP:
                    self.offsets.clear()  # seek, pause or restart: the old offsets no longer hold
                    self.model = None
                self.offsets.append(off)
                if len(self.offsets) >= 5:  # under a second of play: the first notes need a clock already
                    o = np.array(self.offsets)
                    self.model = (float(o.min() if len(o) < 40 else np.percentile(o, 5)), 0.0)

    async def _precise(self, s):
        async with s.ws_connect(URL + "/v2/precise") as ws:
            async for msg in ws:
                d = json.loads(msg.data)
                self.keys, self.hit_errors = d.get("keys", {}), d.get("hitErrors", [])


def osu_md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("map", nargs="?", help="only count this map as playing (checked by MD5)")
    args = p.parse_args()
    clock = TosuClock(osu_md5(args.map) if args.map else None)
    clock.start()
    print("reading tosu at 127.0.0.1:24050; play the map in lazer (Ctrl+C stops)", flush=True)
    last = None
    while True:
        time.sleep(1)
        if clock.error is not None and clock.state is None:
            line = f"waiting for tosu ({type(clock.error).__name__}); start tosu.exe"
            if line != last:
                print(line, flush=True)
                last = line
            continue
        line = (f"state {clock.state} map {clock.map_name} md5 {'ok' if clock.md5 in (None, clock.map_md5) else 'OTHER MAP'}"
                f" playing {clock.playing}")
        if clock.model is not None:
            o = np.array(clock.offsets)
            game = time.perf_counter() - clock.offset
            he = clock.hit_errors
            line += (f" | game {game:7.2f}s from {len(o)} reads (p50 {(np.percentile(o, 50) - clock.offset) * 1000:.1f} ms"
                     f" above the clock), keys {[(k, v.get('count')) for k, v in clock.keys.items()]},"
                     f" hit error mean {np.mean(he) if he else float('nan'):.1f} ms of {len(he)}")
        if line != last:
            print(line, flush=True)
            last = line


if __name__ == "__main__":
    main()
