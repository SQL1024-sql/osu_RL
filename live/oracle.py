"""Perfect aim or perfect keys from the map, to split the student's misses between aiming and clicking: the student
plays as usual and one half of what it does is replaced by what the map says (live.vision_play VisionPlayer.oracle).

    aim:  the cursor goes where the map says (on each object by its start, along the slider ball, round the spinner);
          the student only decides the keys. It still sees its own cursor where the map put it.
    keys: a key goes down on each object's start (alternating keys, held through sliders and spinners); the student
          only moves the cursor.
    both: the map plays alone (a check of the oracle and the game clock: it should be near 100%).

Like autopilot, nothing here looks at the screen: it is a diagnostic, never for the stream.
"""
import numpy as np

import vision.student as S
from beatmap import CENTRE, SLIDER, SPINNER

SPIN_R = 60.0  # osu! px from the spinner's centre
SPIN_RATE = 0.045  # rad / ms: 7 turns a second, under lazer's cap (477 rpm)
EARLY_MS = 30.0  # the cursor arrives on an object this long before its start (at most 30% of the way there)
CIRCLE_HOLD_MS = 40.0  # a circle's key stays down this long
RELEASE_GAP_MS = 10.0  # a key is up at least this long before it goes down again


class Oracle:
    def __init__(self, path, mode):
        assert mode in ("aim", "keys", "both"), mode
        self.mode, self.bm = mode, S.load(path)
        bm = self.bm
        self.start = bm.objects[:, 2].astype(float)
        self.end = np.maximum(bm.ends.astype(float), self.start)
        self.kind = bm.kinds
        n = len(self.start)
        # where each object begins and ends (a spinner: on its circle, where the spin starts and where it stops)
        self.p0 = bm.objects[:, :2].astype(float).copy()
        self.p1 = self.p0.copy()
        for i in range(n):
            if self.kind[i] == SLIDER:
                self.p1[i] = bm.sliders[i].ball(self.end[i])
            elif self.kind[i] == SPINNER:
                self.p0[i] = self.spin(i, self.start[i])
                self.p1[i] = self.spin(i, self.end[i])
        # keys: object i on key i % 2, down from its start to its release
        rel = np.where(self.kind == 0, self.start + CIRCLE_HOLD_MS, self.end + 20.0)
        for i in range(n - 2):  # the same key goes down again on object i + 2
            rel[i] = min(rel[i], self.start[i + 2] - RELEASE_GAP_MS)
        self.release = np.maximum(rel, self.start + S.DT)

    def spin(self, i, t):
        a = (t - self.start[i]) * SPIN_RATE
        return CENTRE + SPIN_R * np.array([np.cos(a), np.sin(a)])

    def pos(self, t):
        """Where the cursor should be at game time t (ms)."""
        j = int(np.searchsorted(self.start, t, "right")) - 1  # the last object started by t
        if j >= 0 and t <= self.end[j]:
            if self.kind[j] == SLIDER:
                return self.bm.sliders[j].ball(t)
            if self.kind[j] == SPINNER:
                return self.spin(j, t)
            return self.p0[j]
        if j + 1 >= len(self.start):
            return self.p1[-1]
        if j < 0:
            return self.p0[0]
        # between object j's end and object j + 1's start: a straight line, there a little early
        t0, t1 = self.end[j], self.start[j + 1]
        arrive = t1 - min(EARLY_MS, 0.3 * (t1 - t0))
        f = 1.0 if arrive <= t0 else float(np.clip((t - t0) / (arrive - t0), 0, 1))
        f = f * f * (3 - 2 * f)  # smooth start and stop
        return self.p1[j] + f * (self.p0[j + 1] - self.p1[j])

    def buttons(self, t):
        """Keys held at game time t (bit 0: M1, bit 1: M2)."""
        lo = int(np.searchsorted(self.start, t - 30_000))
        hi = int(np.searchsorted(self.start, t, "right"))
        b = 0
        for i in range(lo, hi):
            if t < self.release[i]:
                b |= 1 << (i % 2)
        return b
