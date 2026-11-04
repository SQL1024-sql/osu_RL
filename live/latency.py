"""Loop latency of the real setup: the virtual tablet moves the cursor, lazer draws it, the screen shows it, a
screen capture sees it. The time from the move to the capture seeing lazer's cursor is what the player has to
lead by (phase 1) and what a vision model sees the world late by (phase 2).

    python -m live.latency            # lazer in front, on a screen where its cursor shows (song select is fine)
"""
import argparse
import ctypes
import time

import mss
import numpy as np

from live.tablet import Tablet, lazer_window, user32


def client_rect():
    hwnd = lazer_window()
    r, p = (ctypes.c_long * 4)(), (ctypes.c_long * 2)()
    user32.GetClientRect(hwnd, r)
    user32.ClientToScreen(hwnd, p)
    return p[0], p[1], r[2], r[3]


class Dxcam:
    """DXGI desktop duplication ("dxgi") or Windows Graphics Capture ("winrt") through dxcam; a frame only when the
    screen changed. DXGI duplication fails on hybrid-GPU laptops; winrt needs pip install "dxcam[winrt]"."""

    def __init__(self, box, backend):
        import dxcam
        self.cam = dxcam.create(output_color="BGR", backend=backend)
        self.region = (box["left"], box["top"], box["left"] + box["width"], box["top"] + box["height"])
        self.last = None

    def grab(self, _box):
        f = self.cam.grab(region=self.region)
        if f is not None:
            self.last = f
        return self.last

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cam.release()


def measure(trials=60, threshold=8.0, timeout=0.25, backend="mss"):
    left, top, w, h = client_rect()
    a = (left + 0.12 * w, top + 0.55 * h)
    b = (left + 0.12 * w, top + 0.80 * h)
    half = 12
    # lazer's cursor is drawn down and right of its hot spot
    box = {"left": int(b[0]) + 4, "top": int(b[1]) + 4, "width": 2 * half, "height": 2 * half}
    tablet = Tablet()
    lat, grabs, lost = [], [], 0
    try:
        with (mss.mss() if backend == "mss" else Dxcam(box, backend)) as sct:
            def grab():
                s = time.perf_counter()
                v = np.asarray(sct.grab(box))[:, :, :3].astype(np.float32)
                e = time.perf_counter()
                grabs.append(e - s)
                return v, e  # GDI capture waits for the next composition, so the image is from about the end

            for _ in range(trials):
                tablet.send(*a)
                time.sleep(0.2)
                base = np.mean([grab()[0] for _ in range(3)], axis=0)
                moved = time.perf_counter()
                tablet.send(*b)
                while True:
                    v, s = grab()
                    if np.abs(v - base).mean() > threshold:
                        lat.append((s - moved) * 1000)
                        break
                    if s - moved > timeout:
                        lost += 1
                        break
    finally:
        tablet.close()
    lat = np.array(lat)
    g = np.array(grabs) * 1000
    print(f"{len(lat)}/{trials} seen ({lost} not seen within {timeout * 1000:.0f} ms)")
    if len(lat):
        print(f"move -> seen: median {np.median(lat):.1f} ms, p10 {np.percentile(lat, 10):.1f}, "
              f"p90 {np.percentile(lat, 90):.1f}, min {lat.min():.1f}, max {lat.max():.1f}")
    print(f"one capture takes median {np.median(g):.2f} ms, p90 {np.percentile(g, 90):.2f} ms")
    return lat


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=60)
    p.add_argument("--backend", default="mss", choices=["mss", "dxgi", "winrt"])
    args = p.parse_args()
    measure(args.trials, backend=args.backend)
