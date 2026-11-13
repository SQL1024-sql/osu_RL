"""More maps for the improve loop: picks maps from the local HF set (maps/hf, .osu only: they play silent, like the live
set) close to the live set's difficulty, keeps those the teacher plays well in the simulator (its actions are the
student's labels: a map it plays badly would teach mistakes), and writes them with the live set's metadata.

    python -m live.make_pool                        # 24 training maps (HF train split) + 2 gate maps (HF val split)
    python -m live.make_pool --train 40 --gate 0    # more training maps; existing files are kept
    python -m live.make_pool --out runs/maps/tier2 --min-stars 4.5 --max-stars 5.0 --max-ar 9.6   # a harder tier

Writes runs/maps/pool/train/*.osu and runs/maps/pool/gate/*.osu, plus pool.json (each map's stars, length and the
teacher's simulator accuracy). The fork imports a map the first time it is asked to play it (live.collect).
"""
import argparse
import csv
import ctypes
import glob
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np

OUT = "runs/maps/pool"
LIVE_IDS = {"776730", "921152", "531556", "3795827", "103019"}  # already in runs/maps/live_set


def write_map(src, dst_dir, tag):
    """The .osu with the live set's metadata (no online IDs: lazer must not look it up); gameplay untouched."""
    txt = open(src, encoding="utf-8-sig", errors="replace").read()
    title = re.search(r"^Title:(.*)$", txt, re.M)
    ver = re.search(r"^Version:(.*)$", txt, re.M)
    name = f"{tag} - {(title.group(1).strip() if title else '')} {(ver.group(1).strip() if ver else '')} " \
           f"({os.path.basename(src)[:-4]})".strip()
    meta = (f"[Metadata]\nTitle:osu_rl live set\nTitleUnicode:osu_rl live set\nArtist:osu_rl\nArtistUnicode:osu_rl\n"
            f"Creator:osu_rl\nVersion:{name}\nSource:\nTags:osu_rl\nBeatmapID:0\nBeatmapSetID:-1\n\n")
    if "[Metadata]" in txt:
        txt = re.sub(r"\[Metadata\].*?(?=\n\[)", meta.rstrip("\n"), txt, flags=re.S)
    else:
        txt = txt.replace("[Difficulty]", meta + "[Difficulty]", 1)
    dst = os.path.join(dst_dir, re.sub(r"[^\w.-]+", "_", name)[:120] + ".osu")
    open(dst, "w", encoding="utf-8", newline="\r\n").write(txt.replace("\r\n", "\n"))
    return dst


def teacher_acc(path):
    """The teacher's lazer accuracy on the map in the simulator (as student.rollouts plays it)."""
    if os.name == "nt":  # the student may be playing live meanwhile: stay out of its way
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL
    import torch
    torch.set_num_threads(1)
    import vision.student as S
    from state.env import OsuEnv
    from state.evaluate import model_policy
    model = S.teacher("cpu")
    env = OsuEnv(S.load(path), hz=S.HZ, k=model_policy(model).k, classic=False)
    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        obs, _, done, _, _ = env.step(model.predict(obs, deterministic=True)[0])
    return env.lazer_accuracy()


def candidates(a, split):
    rows = []
    for r in csv.DictReader(open("maps/hf/index.csv", encoding="utf-8")):
        if r["split"] != split or r["beatmap_id"] in LIVE_IDS or r["approved"] not in ("1", "2"):
            continue
        st, ln, cs, ar = float(r["stars"]), float(r["hit_length"]), float(r["cs"]), float(r["ar"])
        if a.min_stars <= st <= a.max_stars and a.min_len <= ln <= a.max_len and 3.0 <= cs <= a.max_cs and 7.0 <= ar <= a.max_ar:
            r["path"] = f"maps/hf/{split}/{r['beatmap_id']}.osu"
            if os.path.exists(r["path"]):
                rows.append(r)
    rng = np.random.default_rng(a.seed)
    rng.shuffle(rows)
    seen, out = set(), []
    for r in rows:  # one difficulty per set
        if r["beatmapset_id"] not in seen:
            seen.add(r["beatmapset_id"])
            out.append(r)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=int, default=24, help="training maps wanted in runs/maps/pool/train")
    p.add_argument("--gate", type=int, default=2, help="gate maps wanted in runs/maps/pool/gate")
    p.add_argument("--min-stars", type=float, default=3.0)
    p.add_argument("--max-stars", type=float, default=4.5)
    p.add_argument("--min-len", type=float, default=70, help="drain seconds")
    p.add_argument("--max-len", type=float, default=140)
    p.add_argument("--max-ar", type=float, default=9.3, help="harder tiers need faster approach rates")
    p.add_argument("--max-cs", type=float, default=4.5)
    p.add_argument("--out", default=OUT, help="folder for train/, gate/ and pool.json (a difficulty tier of its own)")
    p.add_argument("--teacher-min", type=float, default=0.95, help="the teacher's simulator accuracy a map needs")
    p.add_argument("--procs", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    tag = "pool" if os.path.normpath(a.out) == os.path.normpath(OUT) else os.path.basename(os.path.normpath(a.out))
    info_path = os.path.join(a.out, "pool.json")
    os.makedirs(a.out, exist_ok=True)
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    for split, sub, want in (("train", "train", a.train), ("val", "gate", a.gate)):
        d = os.path.join(a.out, sub)
        os.makedirs(d, exist_ok=True)
        have = len(glob.glob(os.path.join(d, "*.osu")))
        tried = {v["id"] for v in info.values()}
        todo = [r for r in candidates(a, split) if r["beatmap_id"] not in tried]
        with ProcessPoolExecutor(a.procs) as ex:
            while have < want and todo:
                batch, todo = todo[: a.procs * 2], todo[a.procs * 2:]
                for r, acc in zip(batch, ex.map(teacher_acc, [r["path"] for r in batch])):
                    ok = acc >= a.teacher_min and have < want
                    dst = write_map(r["path"], d, f"{tag} {sub}") if ok else None
                    have += ok
                    info[r["path"]] = dict(id=r["beatmap_id"], split=sub, kept=bool(ok), file=dst, teacher=round(acc, 4),
                                           stars=float(r["stars"]), seconds=float(r["hit_length"]), bpm=float(r["bpm"]),
                                           cs=float(r["cs"]), ar=float(r["ar"]), title=f"{r['artist']} - {r['title']} [{r['version']}]")
                    print(f"{sub} {r['beatmap_id']:>8} {float(r['stars']):.2f}* teacher {acc:.2%} "
                          f"{'kept' if ok else 'skipped'}  {r['artist']} - {r['title']} [{r['version']}]", flush=True)
                json.dump(info, open(info_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print(f"{sub}: {have} maps in {d}")


if __name__ == "__main__":
    main()
