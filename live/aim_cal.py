"""A model's own aim offset from its plays: where the student thought its cursor was (the play log's x, y, before any
offset was added) when it fired, against the circle / slider head it fired for (lazer's judgements). The offset to
send is minus the median of that, whatever offset the plays were made with. student.aim_offset_of reads it back from
<ckpt>.aim.json, and live.vision_play.VisionPlayer adds it to where the cursor is sent.

    python -m live.aim_cal runs/fovea/f4ws/last_cal.pt runs/ablate/20261118-075015/r*-student --write
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

import vision.student as S

MATCH_MS = 80.0  # a click belongs to the circle / slider head starting nearest to it, if within this


def shifts(play_dirs):
    """(where lazer saw the key go down - object centre - the offset the play sent) for each press matched to a
    circle / slider head, over every play under the folders (a play: X.npz with its log and X-lazer.json with lazer's
    judgements and input frames). lazer's own record of the press, not the log's cursor at the step that decided it:
    that one reads ~2 px short along x (2026-11-18: the log said (9.3, -9.7), the game's residuals after sending
    (6.5, -5) and (13, -10) both put the right offset at (11.4, -10.0)). The offset a play sent is the median of lazer's
    cursor minus the log's at the same game times."""
    out = []
    for d in play_dirs:
        for j_path in glob.glob(os.path.join(d, "**", "*-lazer.json"), recursive=True):
            npz = j_path[:-len("-lazer.json")] + ".npz"
            if not os.path.exists(npz):
                continue
            log = np.load(npz)["log"]
            log = log[np.isfinite(log[:, 1])]
            with open(j_path, encoding="utf-8") as f:
                j = json.load(f)
            fr = np.array(j["frames"], float)
            heads = np.array([(r[0], r[5], r[6]) for r in j["judgements"] if r[1] in ("HitCircle", "SliderHeadCircle")], float)
            if len(log) < 100 or len(fr) < 100 or not len(heads):
                continue
            inside = (fr[:, 0] > log[0, 1]) & (fr[:, 0] < log[-1, 1])
            lx = np.interp(fr[inside, 0], log[:, 1], log[:, 2])
            ly = np.interp(fr[inside, 0], log[:, 1], log[:, 3])
            sent = np.median(np.c_[fr[inside, 1] - lx, fr[inside, 2] - ly], 0)
            b = fr[:, 3].astype(int)
            press = np.nonzero((b & ~np.r_[0, b[:-1]]) & 3)[0]
            for i in press:
                k = int(np.argmin(np.abs(heads[:, 0] - fr[i, 0])))
                if abs(heads[k, 0] - fr[i, 0]) <= MATCH_MS:
                    out.append(fr[i, 1:3] - heads[k, 1:] - sent)
    return np.array(out).reshape(-1, 2)


def measure(play_dirs):
    """-> (offset to send, clicks it is from)"""
    s = shifts(play_dirs)
    return (-np.median(s, 0) if len(s) else np.zeros(2)), len(s)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("plays", nargs="+", help="folders with the model's plays (globs are fine)")
    p.add_argument("--write", action="store_true", help="save it as <ckpt>.aim.json")
    a = p.parse_args()
    dirs = [d for g in a.plays for d in (glob.glob(g) or [g]) if os.path.isdir(d)]
    off, n = measure(dirs)
    print(f"{a.ckpt}: now {S.aim_offset_of(a.ckpt)}, measured {off.round(2)} from {n} clicks in {len(dirs)} folders")
    if a.write and n >= 500:
        S.save_aim_offset(a.ckpt, off, f"live.aim_cal from {n} clicks: {', '.join(dirs)}")
        print("written")
    elif a.write:
        print("too few clicks (< 500): not written")


if __name__ == "__main__":
    main()
