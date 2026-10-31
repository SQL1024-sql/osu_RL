import hashlib
import lzma
import struct
import time

KEY_BITS = (5, 10)  # K1 = M1|K1, K2 = M2|K2


def _str(s):
    b = s.encode()
    out, n = bytearray(b"\x0b"), len(b)
    while True:
        byte, n = n & 0x7F, n >> 7
        out.append(byte | (0x80 if n else 0))
        if not n:
            return bytes(out) + b


def write_osr(path, osu_path, frames, counts, max_combo, player="osu_rl"):
    """frames: [(time_ms, x, y, (key1, key2))]; counts: {300: n, 100: n, 50: n, 0: n}"""
    with open(osu_path, "rb") as f:
        map_md5 = hashlib.md5(f.read()).hexdigest()

    parts, elapsed = [], 0
    for t, x, y, keys in frames:
        d = round(t - elapsed)  # accumulate rounded deltas so timing doesn't drift
        elapsed += d
        parts.append(f"{d}|{x:.2f}|{y:.2f}|{sum(b for b, k in zip(KEY_BITS, keys) if k)}")
    parts.append("-12345|0|0|0")
    data = lzma.compress((",".join(parts) + ",").encode(), format=lzma.FORMAT_ALONE)

    n300, n100, n50, miss = (counts.get(k, 0) for k in (300, 100, 50, 0))
    score = 300 * n300 + 100 * n100 + 50 * n50  # shortcut: not real osu! scoring, lazer re-judges on playback
    ticks = int(time.time() * 1e7) + 621355968000000000  # .NET DateTime ticks
    replay_md5 = hashlib.md5(f"{map_md5}{player}{ticks}".encode()).hexdigest()
    body = (
        struct.pack("<bi", 0, 20240101)
        + _str(map_md5) + _str(player) + _str(replay_md5)
        + struct.pack("<6hih?i", n300, n100, n50, 0, 0, miss, score, max_combo, miss == 0, 0)
        + b"\x00"  # no life bar graph
        + struct.pack("<qi", ticks, len(data)) + data
        + struct.pack("<q", 0)
    )
    with open(path, "wb") as f:
        f.write(body)
