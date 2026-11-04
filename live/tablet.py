"""Virtual tablet output through the VMulti driver (https://github.com/X9VoiD/vmulti-bin): an OS-level absolute
pointer with two buttons, like a pen tablet. Nothing touches the game process.

    python -m live.tablet           # draws a circle around the screen centre, prints the OS cursor lag
    python -m live.tablet --click   # also alternates M1 / M2 (only run it with lazer in front)
"""
import ctypes
import math
import sys
import time
from ctypes import wintypes

HID_GUID = (ctypes.c_byte * 16)(*bytes.fromhex("b2551e4d6ff1cf1188cb001111000030"))  # {4d1e55b2-f16f-11cf-88cb-001111000030}
VMULTI = (0x00FF, 0xBACC)  # vendor, product
REPORT_LEN = 65  # the control collection's output report, report id included
ABS_REPORT = 0x09  # absolute pointer: header, x, y, pressure (VoiDPlugins AbsoluteInputReport)
M1, M2 = 1, 2

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.CreateFileW.restype = wintypes.HANDLE
user32 = ctypes.WinDLL("user32")
user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # per-monitor v2: real pixels, not scaled ones


class HiddAttributes(ctypes.Structure):
    _fields_ = [("size", wintypes.ULONG), ("vid", wintypes.USHORT), ("pid", wintypes.USHORT), ("version", wintypes.USHORT)]


def control_path():
    cfg = ctypes.WinDLL("cfgmgr32")
    size = wintypes.ULONG()
    cfg.CM_Get_Device_Interface_List_SizeW(ctypes.byref(size), HID_GUID, None, 0)
    buf = ctypes.create_unicode_buffer(size.value)
    cfg.CM_Get_Device_Interface_ListW(HID_GUID, None, buf, size.value, 0)
    hid = ctypes.WinDLL("hid")
    # the control collection is VMulti's vendor-defined one; writing to it is how VMulti takes reports
    for p in filter(None, buf[:size.value].split("\0")):
        h = k32.CreateFileW(p, 0x40000000, 3, None, 3, 0, None)  # GENERIC_WRITE, share read|write, OPEN_EXISTING
        if h is None or h == wintypes.HANDLE(-1).value:
            continue
        attrs = HiddAttributes(ctypes.sizeof(HiddAttributes))
        caps = (ctypes.c_ushort * 32)()
        pp = ctypes.c_void_p()
        if hid.HidD_GetAttributes(wintypes.HANDLE(h), ctypes.byref(attrs)) and (attrs.vid, attrs.pid) == VMULTI \
                and hid.HidD_GetPreparsedData(wintypes.HANDLE(h), ctypes.byref(pp)):
            hid.HidP_GetCaps(pp, caps)
            hid.HidD_FreePreparsedData(pp)
            if caps[3] == REPORT_LEN:  # HIDP_CAPS.OutputReportByteLength
                return h
        k32.CloseHandle(wintypes.HANDLE(h))
    sys.exit("VMulti control device not found. Install https://github.com/X9VoiD/vmulti-bin/releases/latest")


class Tablet:
    """Absolute position in screen pixels plus M1/M2, sent as one VMulti report per call."""

    def __init__(self):
        self.h = wintypes.HANDLE(control_path())
        m = user32.GetSystemMetrics
        self.left, self.top, self.width, self.height = m(76), m(77), m(78), m(79)  # the virtual screen
        self.buf = (ctypes.c_ubyte * REPORT_LEN)()
        self.buttons = 0

    def send(self, x, y, buttons=None):
        if buttons is not None:
            self.buttons = buttons
        nx = round((x - self.left) * 32767 / (self.width - 1))
        ny = round((y - self.top) * 32767 / (self.height - 1))
        nx, ny = min(max(nx, 0), 32767), min(max(ny, 0), 32767)
        report = bytes([0x40, 9, ABS_REPORT, self.buttons]) + nx.to_bytes(2, "little") + ny.to_bytes(2, "little") + b"\0\0"
        ctypes.memmove(self.buf, report, len(report))
        written = wintypes.DWORD()
        if not k32.WriteFile(self.h, self.buf, REPORT_LEN, ctypes.byref(written), None):
            raise OSError(ctypes.get_last_error(), "VMulti write failed")

    def close(self):
        p = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(p))
        self.send(p.x, p.y, 0)  # never leave a button held
        k32.CloseHandle(self.h)


def cursor():
    p = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(p))
    return p.x, p.y


def demo(click, seconds=5.0, hz=120):
    t = Tablet()
    cx, cy, r = t.left + t.width / 2, t.top + t.height / 2, t.height / 4
    k32.SetThreadPriority(k32.GetCurrentThread(), 2)
    ctypes.WinDLL("winmm").timeBeginPeriod(1)
    lags, errs = [], []
    try:
        start = time.perf_counter()
        for i in range(int(seconds * hz)):
            due = start + i / hz
            while time.perf_counter() < due:
                pass
            a = 2 * math.pi * i / hz
            x, y = round(cx + r * math.cos(a)), round(cy + r * math.sin(a))
            buttons = (M1 if (i // 15) % 2 == 0 else M2) if click and (i // 15) % 4 < 2 else 0
            sent = time.perf_counter()
            t.send(x, y, buttons)
            while cursor() != (x, y) and time.perf_counter() - sent < 0.004:
                pass
            if cursor() == (x, y):
                lags.append((time.perf_counter() - sent) * 1000)
            else:
                cur = cursor()
                errs.append(math.hypot(cur[0] - x, cur[1] - y))
    finally:
        t.close()
        ctypes.WinDLL("winmm").timeEndPeriod(1)
    n = int(seconds * hz)
    lags.sort()
    print(f"screen {t.width}x{t.height} at ({t.left},{t.top}), {n} reports at {hz} Hz")
    if lags:
        print(f"cursor reached the target {len(lags)}/{n}: median {lags[len(lags) // 2]:.2f} ms, max {lags[-1]:.2f} ms")
    if errs:
        print(f"missed {len(errs)}/{n}, off by up to {max(errs):.1f} px")


if __name__ == "__main__":
    demo("--click" in sys.argv)
