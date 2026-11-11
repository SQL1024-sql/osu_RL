"""What the vision student sees, as a video stream for OBS: put it in a corner of the stream.

    python -m live.vision_play --view          # play, and serve the student's view at http://127.0.0.1:8770/
    python -m live.vision_play --view --view-raw   # the bare model input, no move arrow / key lights
    python -m live.vision_view --demo          # no model, no tosu: lazer's playfield as the student would get it

OBS: Sources -> + -> Browser, URL http://127.0.0.1:8770/, width 640, height 480 (any 4:3 size; the picture scales
with sharp pixels), and drag it into a corner. Once loaded the page reconnects by itself, so it can stay in the scene
between runs (transparent while nothing streams); if OBS was opened before the first stream, refresh the source once
(source properties -> Refresh cache of current page). A Local file source does not work: OBS's browser will not let
a file page read the localhost stream. The stream is the newest frame the student decided on (160 x 120 grey, the
cursor dot drawn in as the model gets it), scaled up; while no map is being played it shows the raw capture.
The HUD adds what the student is about to do: an arrow for its move, a ring when it clicks, and K1/K2 while held.
"""
import argparse
import http.server
import os
import socketserver
import threading
import time

import cv2
import numpy as np

from vision.render import H, PX, W, X0, Y0

PAGE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vision_view.html"), "rb").read()

ARROW = (255, 200, 0)    # BGR: the move
RING = (80, 80, 255)     # a click
KEY_ON = (80, 220, 80)
KEY_OFF = (70, 70, 70)
STALE = 0.25             # s without a decision -> show the raw capture


class View:
    """Serves the student's view as MJPEG. Call want() / put() from the play loop; everything else runs on its own threads."""

    def __init__(self, port=8770, scale=4, hud=True, fps=60, idle=None, host="127.0.0.1"):
        self.scale, self.hud, self.dt = scale, hud, 1 / fps
        self.idle = idle            # () -> newest raw (H, W) uint8 frame, shown while the student is not deciding
        self.state, self.t_state, self.taken = None, 0.0, True
        self.last_fire = -1.0
        self.jpeg, self.seq = None, 0
        self.cond = threading.Condition()
        threading.Thread(target=self._render, daemon=True).start()
        view = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                path = self.path.split("?")[0]
                if path == "/":
                    self._send(200, "text/html; charset=utf-8", PAGE)
                elif path == "/frame.jpg":
                    with view.cond:
                        view.cond.wait_for(lambda: view.jpeg is not None, timeout=2)
                        data = view.jpeg
                    self._send(200, "image/jpeg", data or b"")
                elif path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    seen = -1
                    try:
                        while True:
                            with view.cond:
                                view.cond.wait_for(lambda: view.seq != seen, timeout=2)
                                seen, data = view.seq, view.jpeg
                            if data is None:
                                continue
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(data))
                            self.wfile.write(data)
                            self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        pass
                else:
                    self._send(404, "text/plain", b"not found")

            def _send(self, code, ctype, body):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

        class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        self.server = Server((host, port), Handler)
        self.url = f"http://{host}:{port}/"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def want(self):
        """True when the last decision has been drawn, so the play loop copies a frame only as often as it is shown."""
        return self.taken

    def put(self, frame, cursor, move, held, fire):
        """frame (H, W) uint8: the newest model input frame; cursor in osu! px; move in [-1, 1] (x max speed); held (2,) bool."""
        now = time.perf_counter()
        if fire:
            self.last_fire = now
        self.state = (frame, np.array(cursor, float), np.array(move, float), np.array(held, bool))
        self.t_state, self.taken = now, False

    def _render(self):
        nxt = time.perf_counter()
        while True:
            now = time.perf_counter()
            img = None
            if self.state is not None and now - self.t_state < STALE:
                img = self.draw(*self.state, ring=now - self.last_fire < 0.12)
                self.taken = True
            elif self.idle is not None:
                try:
                    raw = self.idle()
                except Exception:
                    raw = None
                if raw is not None:
                    img = self.upscale(raw)
            if img is not None:
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if ok:
                    with self.cond:
                        self.jpeg, self.seq = buf.tobytes(), self.seq + 1
                        self.cond.notify_all()
            nxt += self.dt
            wait = nxt - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            else:
                nxt = time.perf_counter()

    def upscale(self, frame):
        big = cv2.resize(frame, (W * self.scale, H * self.scale), interpolation=cv2.INTER_NEAREST)
        return cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)

    def draw(self, frame, cursor, move, held, ring=False):
        img = self.upscale(frame)
        if not self.hud:
            return img
        s = self.scale
        c = ((cursor[0] - X0) * PX * s, (cursor[1] - Y0) * PX * s)
        if np.hypot(*move) > 0.05:
            tip = (c[0] + move[0] * 18 * s, c[1] + move[1] * 18 * s)
            cv2.arrowedLine(img, tuple(map(round, c)), tuple(map(round, tip)), ARROW, max(1, s // 2), cv2.LINE_AA, tipLength=0.3)
        if ring:
            cv2.circle(img, tuple(map(round, c)), 6 * s, RING, max(1, s // 2), cv2.LINE_AA)
        bw, bh, pad = 9 * s, 6 * s, 2 * s
        for i, name in enumerate(("K1", "K2")):
            x = W * s - (2 - i) * (bw + pad)
            y = H * s - bh - pad
            on = bool(held[i])
            cv2.rectangle(img, (x, y), (x + bw, y + bh), KEY_ON if on else KEY_OFF, -1)
            cv2.putText(img, name, (x + s, y + bh - 2 * s), cv2.FONT_HERSHEY_SIMPLEX, 0.09 * s, (0, 0, 0) if on else (200, 200, 200),
                        max(1, s // 3), cv2.LINE_AA)
        return img


def dot(frame, x, y, r=2.2):
    """gpu_render.draw_dots on the CPU: a white dot of radius r frame px at (x, y) osu! px."""
    gx = X0 + (np.arange(W) + 0.5) / PX
    gy = (Y0 + (np.arange(H) + 0.5) / PX)[:, None]
    d = np.clip(r - np.hypot(gx - x, gy - y) * PX + 0.5, 0, 1)
    return np.maximum(frame, (d * 255).astype(np.uint8))


def frame_box(field):
    """The screen region of the student's frame (as live.vision_play grabs it) for a lazer Playfield."""
    x0, y0 = field(X0, Y0)
    return {"left": round(x0), "top": round(y0), "width": round(W / PX * field.scale), "height": round(H / PX * field.scale)}


def demo(args):
    """lazer's playfield as the student gets it, with the OS cursor as its dot: to set OBS up without the model.
    Only while lazer is in front; otherwise the view holds its last frame (the desktop never goes out on stream)."""
    import ctypes

    import mss

    from live.screen_play import Playfield, wait_front
    user32 = ctypes.WinDLL("user32")
    view = View(args.port, args.scale, not args.raw, args.fps)
    print(f"demo view at {view.url} (Ctrl+C stops)", flush=True)
    pt = (ctypes.c_long * 2)()
    prev, field = None, None
    with mss.MSS() as sct:
        try:
            while True:
                if field is None or not field.in_front():
                    wait_front("demo view")  # a fullscreen lazer is minimized behind other windows: nothing to grab
                    field = Playfield()     # laid out again: the window may have changed
                    box = frame_box(field)
                    print(f"grabbing lazer's playfield {box}", flush=True)
                g = cv2.resize(cv2.cvtColor(np.asarray(sct.grab(box)), cv2.COLOR_BGRA2GRAY), (W, H), interpolation=cv2.INTER_AREA)
                user32.GetCursorPos(pt)
                cur = np.array([(pt[0] - field.x0) / field.scale, (pt[1] - field.y0) / field.scale])
                move = np.zeros(2) if prev is None else np.clip((cur - prev) / 15, -1, 1)
                prev = cur
                buttons = user32.GetAsyncKeyState(0x01) & 0x8000, user32.GetAsyncKeyState(0x02) & 0x8000
                view.put(dot(g, *cur), cur, move, buttons, False)
                time.sleep(1 / 120)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="no model: stream lazer's playfield with the OS cursor")
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--scale", type=int, default=4, help="frame pixels -> stream pixels (4: 640 x 480)")
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--raw", action="store_true", help="no HUD")
    a = p.parse_args()
    if not a.demo:
        p.error("run it through live.vision_play --view, or try it with --demo")
    demo(a)
