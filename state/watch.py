"""Plays the osu!standard difficulties of a .osz (or one .osu) with a model and writes a replay per difficulty.

    python -m state.watch "path/to/map.osz" --out replays/renatus
    python -m state.watch "path/to/map.osz" --out replays/renatus --diff Hard --open
--open starts the local osu!lazer build (OSU_REPO) with the .osz and the replays: it imports the map first, then the
replays, which it can then play back. Replays only match the exact .osu they were made from (osu! looks maps up by MD5).
"""
import argparse
import os
import shutil
import subprocess
import tempfile
import zipfile

from stable_baselines3 import PPO

from beatmap import parse_osu
from state.env import OsuEnv, k_from_obs
from get_maps import is_standard
from state.play import run
from replay import write_osr
from state.verify import DOTNET_ROOT, OSU_REPO

LAZER = os.path.join(OSU_REPO, "osu.Desktop", "bin", "Debug", "net10.0", "osu!.exe")


def charts(path):
    """[(name, .osu bytes)] of the osu!standard difficulties in a .osz, or the single .osu."""
    if path.lower().endswith(".osu"):
        return [(os.path.basename(path)[:-4], open(path, "rb").read())]
    with zipfile.ZipFile(path) as z:
        return [(os.path.basename(n)[:-4], z.read(n)) for n in z.namelist()
                if n.lower().endswith(".osu") and is_standard(z.read(n).decode("utf-8-sig", "ignore"))]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("map", help=".osz or .osu")
    p.add_argument("--out", required=True, help="folder for the replays")
    p.add_argument("--model", default="results/20261101-0021-pool120/best_model_h6.zip")
    p.add_argument("--hz", type=int, default=120)  # must match training
    p.add_argument("--diff", default="", help="only difficulties whose name contains this")
    p.add_argument("--open", action="store_true", help="open the map and replays in the local osu!lazer build")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = PPO.load(args.model, device="cpu")
    k = k_from_obs(model.observation_space.shape[0])
    replays = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, data in charts(args.map):
            if args.diff.lower() not in name.lower():
                continue
            osu_path = os.path.join(tmp, "map.osu")
            open(osu_path, "wb").write(data)
            env = OsuEnv(parse_osu(osu_path), hz=args.hz, k=k)
            frames, counts, acc = run(env, lambda o: model.predict(o, deterministic=True)[0])
            out = os.path.join(args.out, f"{name}.osr")
            write_osr(out, osu_path, frames, counts, env.max_combo)
            replays.append(out)
            print(f"{name}: {acc:.2%}  300/100/50/miss {counts[300]}/{counts[100]}/{counts[50]}/{counts[0]}  combo {env.max_combo} -> {out}")
    if not replays:
        raise SystemExit("no osu!standard difficulty matched")
    if args.open:
        # lazer deletes an imported .osz, so it gets a copy; beatmaps are imported before scores, so one launch works
        osz = shutil.copy(args.map, os.path.join(args.out, os.path.basename(args.map)))
        subprocess.Popen([LAZER, os.path.abspath(osz), *map(os.path.abspath, replays)], cwd=os.path.dirname(LAZER),
                         env=dict(os.environ, DOTNET_ROOT=DOTNET_ROOT))
        print("opening in osu!lazer")
