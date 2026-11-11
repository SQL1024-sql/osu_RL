"""What lazer shows of a beatmap at a given moment, as the small grayscale frame a vision model would see: the 4:3
middle of a 16:9 screen, background fully dimmed, no cursor (the cursor is drawn in separately in training).

    python -m vision.render maps/test/renatus_hard.osu 12.5     # writes runs/render.png for that time (seconds)
"""
import sys

import cv2
import numpy as np

from beatmap import SLIDER, SPINNER, parse_osu

W, H = 160, 120
PX = 0.25  # frame pixels per osu! pixel: the frame spans 640x480 osu! pixels
X0, Y0 = -64.0, -56.0  # osu! pixel at the frame's top-left (lazer: playfield 80% of the height, 8 px lower)
SUB = 16  # cv2 sub-pixel drawing (shift=4)
FADE_OUT = 60  # ms a hit object takes to disappear after it is hit (this skin barely animates it)
# brightness 0..1 of each part over the black background
# grid-searched against 14 lazer screenshots (Prismatix): mean blurred correlation 0.615 -> 0.653
FILL, NUMBER, RIM, APPROACH, BODY, BORDER, SPIN = 0.15, 0.6, 0.95, 0.4, 0.05, 0.85, 0.5
RIM_W, APPROACH_W = 0.12, 0.1  # line widths as a fraction of the circle radius


def px(p):
    return (np.asarray(p, float) - (X0, Y0)) * PX * SUB


class Renderer:
    def __init__(self, bm):
        self.bm = bm
        self.r = bm.radius
        self.preempt = bm.preempt
        self.fade = 400 * min(1, self.preempt / 450)
        self.paths = {i: px(s.path).astype(np.int32) for i, s in bm.sliders.items()}

    def frame(self, t):
        """t in ms -> (H, W) uint8."""
        bm, pre = self.bm, self.preempt
        out = np.zeros((H, W), np.float32)
        starts, ends = bm.objects[:, 2], bm.ends
        live = np.nonzero((starts - pre <= t) & (ends + FADE_OUT >= t))[0]
        for i in live[::-1]:  # earlier objects are drawn on top
            if bm.kinds[i] == SPINNER:
                self.spinner(out, i, t)
            else:
                self.hit_object(out, i, t)
        return (np.clip(out, 0, 1) * 255).astype(np.uint8)

    def blend(self, out, alpha, draw):
        """draw(layer, mask) paints the object's own brightness and coverage; composite it at `alpha`."""
        layer, mask = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
        draw(layer, mask)
        a = alpha * mask
        out *= 1 - a
        out += alpha * layer

    def hit_object(self, out, i, t):
        bm, r = self.bm, self.r
        start, end = bm.objects[i, 2], bm.ends[i]
        appear = start - self.preempt
        alpha = min(1.0, (t - appear) / self.fade)
        if t > end:
            alpha *= max(0.0, 1 - (t - end) / FADE_OUT)
        head = px(bm.objects[i, :2]).astype(np.int32)
        rad = int(r * PX * SUB)
        slider = bm.kinds[i] == SLIDER

        def draw(layer, mask):
            if slider:
                path = self.paths[i]
                n = max(2, int(len(path) * min(1.0, (t - appear) / (self.preempt / 3))))  # snaking in
                body = np.zeros((H, W), np.float32)
                cv2.polylines(body, [path[:n]], False, 1.0, max(1, round(2 * r * PX)), cv2.LINE_AA, 4)
                inner = np.zeros((H, W), np.float32)
                cv2.polylines(inner, [path[:n]], False, 1.0, max(1, round(1.7 * r * PX)), cv2.LINE_AA, 4)
                layer[:] = np.maximum(layer, BORDER * (body - inner) + BODY * inner)
                mask[:] = np.maximum(mask, body)
            # the head; while sliding, the ball, which this skin draws like a hit circle
            c = head if t <= start or not slider else px(bm.sliders[i].ball(t)).astype(np.int32)
            cv2.circle(layer, tuple(c), rad, FILL, -1, cv2.LINE_AA, 4)
            cv2.circle(layer, tuple(c), rad // 4, NUMBER, -1, cv2.LINE_AA, 4)  # stands in for the combo number
            cv2.circle(layer, tuple(c), rad, RIM, max(1, round(RIM_W * r * PX)), cv2.LINE_AA, 4)
            cv2.circle(mask, tuple(c), rad, 1.0, -1, cv2.LINE_AA, 4)
            if t < start:  # approach circle: 4r down to r
                ar = int(rad * (1 + 3 * (start - t) / self.preempt))
                a = min(1.0, (t - appear) / min(2 * self.fade, self.preempt))
                cv2.circle(layer, tuple(head), ar, APPROACH * a, max(1, round(APPROACH_W * r * PX)), cv2.LINE_AA, 4)
                cv2.circle(mask, tuple(head), ar, a, max(1, round(APPROACH_W * r * PX)), cv2.LINE_AA, 4)

        self.blend(out, max(0.0, alpha), draw)

    def spinner(self, out, i, t):
        s = self.bm.spinners[i]
        alpha = min(1.0, max(0.0, (t - (s.start - 500)) / 500)) * (1 if t <= s.end else max(0.0, 1 - (t - s.end) / FADE_OUT))
        c = tuple(px((256, 192)).astype(np.int32))

        def draw(layer, mask):
            for rr in (40, 120, 170):
                cv2.circle(layer, c, int(rr * PX * SUB), SPIN, 1, cv2.LINE_AA, 4)
                cv2.circle(mask, c, int(rr * PX * SUB), 1.0, 1, cv2.LINE_AA, 4)

        self.blend(out, alpha, draw)


if __name__ == "__main__":
    img = Renderer(parse_osu(sys.argv[1])).frame(float(sys.argv[2]) * 1000)
    cv2.imwrite("runs/render.png", cv2.resize(img, (W * 4, H * 4), interpolation=cv2.INTER_NEAREST))
