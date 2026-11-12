"""Checks a machine before it runs the stream (live.improve): Python packages, the GPU, the VMulti virtual tablet, the
fork, the files that are not in git (live.bundle carries them), the display and the sleep settings. Changes nothing.

    python -m live.doctor
"""
import ctypes
import glob
import importlib
import os
import subprocess
import sys
from ctypes import wintypes

OK, BAD, WARN = "[OK]  ", "[缺]  ", "[注意]"
problems = []


def say(mark, text, fix=None):
    print(f"{mark} {text}")
    if fix:
        print(f"        -> {fix}")
    if mark == BAD:
        problems.append(text)


def python_and_packages():
    v = sys.version_info
    say(OK if v >= (3, 11) else BAD, f"Python {v.major}.{v.minor}.{v.micro}", None if v >= (3, 11) else "裝 Python 3.11")
    for mod, pip in [("numpy", "numpy"), ("torch", "torch（CUDA 版，見安裝說明）"), ("cv2", "opencv-python-headless"),
                     ("stable_baselines3", "stable-baselines3"), ("gymnasium", "gymnasium"), ("aiohttp", "aiohttp"),
                     ("mss", "mss"), ("windows_capture", "windows-capture")]:
        try:
            importlib.import_module(mod)
            say(OK, f"套件 {mod}")
        except Exception as e:
            say(BAD, f"套件 {mod} 載入失敗（{type(e).__name__}）", f"pip install {pip}；或 pip install -r requirements.txt")


def gpu():
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        say(BAD, f"PyTorch 看不到 CUDA GPU（torch {torch.__version__}）", "裝 CUDA 版 torch，並確認 NVIDIA 驅動是新的")
        return
    p = torch.cuda.get_device_properties(0)
    gb = p.total_memory / 2**30
    if gb >= 14:
        say(OK, f"GPU {p.name}，{gb:.1f} GB")
    elif gb >= 8:
        say(WARN, f"GPU {p.name}，{gb:.1f} GB：打歌沒問題（用 live.stream，訓練放在別台）；"
                  "要在這台訓練，live.improve 的 --max-recs 要調小（16 GB 用 24）")
    else:
        say(BAD, f"GPU {p.name}，只有 {gb:.1f} GB", "打歌勉強，訓練跑不動")


def vmulti():
    try:
        from live import tablet
    except SystemExit:
        return
    try:
        h = tablet.control_path()
        tablet.k32.CloseHandle(wintypes.HANDLE(h))
        say(OK, "VMulti 虛擬繪圖板驅動")
    except SystemExit:
        say(BAD, "找不到 VMulti 虛擬繪圖板驅動（前景模式要用）",
            "從 https://github.com/X9VoiD/vmulti-bin/releases/latest 下載，解壓後以系統管理員身分執行 "
            "install_hiddriver.bat，裝完重開機；只用背景模式可以先不裝")


def fork():
    from live import bridge
    if os.path.exists(bridge.EXE):
        say(OK, f"fork：{os.path.abspath(bridge.EXE)}")
    else:
        say(BAD, "找不到 fork 的 osu!.exe", "把 live.bundle 做的包解壓到這個資料夾（會有 fork\\osu!.exe），或設 OSU_RL_FORK_EXE")
    from live.collect import fork_pid  # by its window: never connects to the bridge (a running loop is using it)
    pid = fork_pid()
    say(OK, f"fork 正在跑（pid {pid}）" if pid else "fork 沒開（live.improve 會自己開）")


def files():
    import vision.student as S
    from live import bridge, improve
    need = [(S.TEACHER, "老師模型"), (improve.START, "起始學生模型"), (improve.VAL_REC, "驗證錄影"),
            (improve.VAL_REC[:-4] + ".labels.npz", "驗證錄影的標籤"), (bridge.SKIN, "學生用的 skin"),
            ("results/experiments/020-student/seed0/plays7.pkl", "訓練用的模擬資料")]
    for path, what in need:
        say(OK if os.path.exists(path) else BAD, f"{what}：{path}", None if os.path.exists(path) else "從 live.bundle 的包裡補上")
    teacher = sorted({f for g in improve.TEACHER_RECS for f in glob.glob(g) if not f.endswith(".labels.npz")})
    say(OK if len(teacher) >= 6 else BAD, f"老師的實機錄影 {len(teacher)} 個（要 6 個）")
    train, gate = improve.maps_of(improve.TRAIN_MAPS), improve.maps_of(improve.GATE_MAPS)
    say(OK if len(train) >= 4 and gate else BAD, f"訓練譜 {len(train)} 首、關卡譜 {len(gate)} 首")
    n_hf = len(glob.glob("maps/hf/train/*.osu"))
    say(OK if n_hf else BAD, f"maps/hf/train 有 {n_hf} 首（訓練時模擬那一半要用）")
    state = os.path.join("runs", "improve", "state.json")
    if os.path.exists(state):
        import json
        cur = json.load(open(state))["current"]
        say(OK if os.path.exists(cur) else BAD, f"迴圈接續：第 {json.load(open(state))['round']} 輪，上場模型 {cur}")
    else:
        say(OK, "沒有 runs/improve/state.json：迴圈會從起始模型開始")


def display():
    user32 = ctypes.windll.user32
    user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))

    class DEVMODE(ctypes.Structure):
        _fields_ = [("name", wintypes.WCHAR * 32), ("spec", wintypes.WORD), ("drv", wintypes.WORD), ("size", wintypes.WORD),
                    ("extra", wintypes.WORD), ("fields", wintypes.DWORD), ("pos", ctypes.c_long * 2), ("orient", wintypes.DWORD),
                    ("fixed", wintypes.DWORD), ("color", ctypes.c_short), ("duplex", ctypes.c_short), ("yres", ctypes.c_short),
                    ("tt", ctypes.c_short), ("collate", ctypes.c_short), ("form", wintypes.WCHAR * 32), ("logpix", wintypes.WORD),
                    ("bpp", wintypes.DWORD), ("w", wintypes.DWORD), ("h", wintypes.DWORD), ("flags", wintypes.DWORD),
                    ("hz", wintypes.DWORD)]
    dm = DEVMODE()
    dm.size = ctypes.sizeof(DEVMODE)
    user32.EnumDisplaySettingsW(None, -1, ctypes.byref(dm))  # ENUM_CURRENT_SETTINGS
    n = user32.GetSystemMetrics(80)  # SM_CMONITORS
    say(OK if dm.hz >= 60 else WARN, f"主螢幕 {dm.w}x{dm.h} @ {dm.hz} Hz，共 {n} 個螢幕")
    if dm.hz < 120:
        say(WARN, "更新率低於 120 Hz：學生每秒決定 120 次，畫面最好 120 Hz 以上（沒接螢幕時用顯示卡欺騙器）")


def power():
    try:
        out = subprocess.run(["powercfg", "/query", "SCHEME_CURRENT", "SUB_SLEEP", "STANDBYIDLE"], capture_output=True,
                             text=True, errors="replace").stdout
        ac = [l for l in out.splitlines() if "0x" in l]
        secs = int(ac[-2].split()[-1], 16) if len(ac) >= 2 else None  # "Current AC Power Setting Index"
    except Exception:
        secs = None
    if secs is None:
        say(WARN, "讀不到睡眠設定", "自己確認：設定 -> 電源 -> 插電時「永不」睡眠")
    elif secs == 0:
        say(OK, "插電時不會睡眠")
    else:
        say(BAD, f"插電時 {secs // 60} 分鐘沒動作就睡眠", "設定 -> 系統 -> 電源 -> 螢幕與睡眠：插電時都改成「永不」")


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding="utf-8")
    for step in (python_and_packages, gpu, vmulti, fork, files, display, power):
        print(f"--- {step.__name__}")
        try:
            step()
        except Exception as e:
            say(BAD, f"{step.__name__} 檢查時出錯：{type(e).__name__}: {e}")
    print()
    if problems:
        print(f"還缺 {len(problems)} 項，補完再跑 python -m live.improve")
    else:
        print("都齊了：python -m live.improve --foreground（前景、虛擬繪圖板）或 python -m live.improve（背景）")


if __name__ == "__main__":
    main()
