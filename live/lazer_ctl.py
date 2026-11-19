"""Drives the osu!lazer window from a script, for unattended data collection: bring it to the front, look at it
(screenshot), press keys, type, click. Every action first makes sure lazer is the window in front, so nothing goes
to another window.

    python -m live.lazer_ctl shot runs/lazer.png      # bring lazer to the front and save a screenshot
    python -m live.lazer_ctl keys down down enter     # key presses (names below)
    python -m live.lazer_ctl type osu_rl              # type text (the song select search box takes typing)
    python -m live.lazer_ctl click 960 540            # left click at screen pixels
"""
import argparse
import ctypes
import sys
import time
from ctypes import wintypes

import cv2
import mss
import numpy as np

from live.tablet import lazer_window

user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.GetForegroundWindow.restype = ctypes.c_void_p
VK = {"enter": 0x0D, "esc": 0x1B, "space": 0x20, "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
      "tab": 0x09, "back": 0x08, "f1": 0x70, "f2": 0x71, "ctrl": 0x11, "shift": 0x10, "alt": 0x12, "delete": 0x2E}
KEYEVENTF_KEYUP, KEYEVENTF_UNICODE = 0x2, 0x4
MOUSEEVENTF_MOVE, MOUSEEVENTF_ABSOLUTE, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP = 0x1, 0x8000, 0x2, 0x4


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.c_size_t)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class INPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT), ("pad", ctypes.c_byte * 32)]
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _U)]


def send(*inputs):
    arr = (INPUT * len(inputs))(*inputs)
    user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT))


def front():
    """Bring lazer to the front (restoring it if a fullscreen lazer minimized itself); True once it is."""
    hwnd = lazer_window().value
    for _ in range(20):
        if user32.GetForegroundWindow() == hwnd:
            return True
        user32.ShowWindow(ctypes.c_void_p(hwnd), 9)  # SW_RESTORE
        # Windows only lets the foreground window hand the focus over; a tap of Alt counts as input from us
        send(INPUT(1, INPUT._U(ki=KEYBDINPUT(VK["alt"], 0, 0, 0, 0))),
             INPUT(1, INPUT._U(ki=KEYBDINPUT(VK["alt"], 0, KEYEVENTF_KEYUP, 0, 0))))
        user32.SetForegroundWindow(ctypes.c_void_p(hwnd))
        time.sleep(0.25)
    return user32.GetForegroundWindow() == hwnd


def must_front():
    if not front():
        sys.exit("could not bring osu!lazer to the front; nothing was sent")


def key(name):
    vk = VK[name]
    send(INPUT(1, INPUT._U(ki=KEYBDINPUT(vk, 0, 0, 0, 0))), INPUT(1, INPUT._U(ki=KEYBDINPUT(vk, 0, KEYEVENTF_KEYUP, 0, 0))))


def type_text(text):
    for ch in text:
        send(INPUT(1, INPUT._U(ki=KEYBDINPUT(0, ord(ch), KEYEVENTF_UNICODE, 0, 0))),
             INPUT(1, INPUT._U(ki=KEYBDINPUT(0, ord(ch), KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, 0, 0))))
        time.sleep(0.03)


def click(x, y):
    m = user32.GetSystemMetrics
    nx, ny = round(x * 65535 / (m(0) - 1)), round(y * 65535 / (m(1) - 1))
    send(INPUT(0, INPUT._U(mi=MOUSEINPUT(nx, ny, 0, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, 0, 0))))
    time.sleep(0.05)
    send(INPUT(0, INPUT._U(mi=MOUSEINPUT(nx, ny, 0, MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_ABSOLUTE, 0, 0))),
         INPUT(0, INPUT._U(mi=MOUSEINPUT(nx, ny, 0, MOUSEEVENTF_LEFTUP | MOUSEEVENTF_ABSOLUTE, 0, 0))))


def shot(path, scale=0.5):
    with mss.MSS() as sct:
        img = np.asarray(sct.grab(sct.monitors[1]))[:, :, :3]
    cv2.imwrite(path, cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["shot", "keys", "type", "click"])
    p.add_argument("args", nargs="*")
    p.add_argument("--wait", type=float, default=0.4, help="seconds after each action")
    a = p.parse_args()
    must_front()
    if a.cmd == "shot":
        time.sleep(a.wait)
        print(shot(a.args[0] if a.args else "runs/lazer.png"))
    elif a.cmd == "keys":
        for k in a.args:
            key(k)
            time.sleep(a.wait)
    elif a.cmd == "type":
        type_text(" ".join(a.args))
        time.sleep(a.wait)
    elif a.cmd == "click":
        click(float(a.args[0]), float(a.args[1]))
        time.sleep(a.wait)


if __name__ == "__main__":
    main()
