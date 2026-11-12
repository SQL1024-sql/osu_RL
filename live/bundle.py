"""Everything another machine needs to run the stream (live.improve), in one zip: this repo's tracked files as they are
now (uncommitted changes included), the files git does not carry (maps, skin, the teacher's recordings, the rendered
training plays, the loop's current model and state), and the fork published self-contained (no .NET install needed),
in fork/ (live.bridge finds it there).

    python -m live.bundle                      # -> runs/bundle/osu_rl_bundle.zip
    python -m live.bundle --no-fork            # the fork is built on the other machine instead
    python -m live.bundle --no-state           # the other machine starts the loop over from the starting model

On the other machine: unzip, set up Python (live/SETUP_STREAM_PC.md), python -m live.doctor, python -m live.improve.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import tempfile
import zipfile

FORK_SRC = os.path.join("..", "ppy-osu", "osu.Desktop")
EXTRA = ["runs/maps/live_set/*.osu", "runs/maps/pool/*/*.osu", "runs/maps/pool/pool.json", "runs/skin/*.osk",
         "results/experiments/020-student/seed0/plays7.pkl", "maps/hf/index.csv", "maps/hf/*/*.osu", "maps/test/*.osu",
         "maps/*.osu"]


def teacher_files():
    from live import improve
    recs = sorted({f for g in improve.TEACHER_RECS for f in glob.glob(g) if not f.endswith(".labels.npz")})
    recs.append(improve.VAL_REC)
    return [p for r in recs for p in (r, r[:-4] + ".labels.npz") if os.path.exists(p)]


def state_files():
    path = os.path.join("runs", "improve", "state.json")
    if not os.path.exists(path):
        return []
    cur = json.load(open(path))["current"]
    return [p for p in (path, os.path.join("runs", "improve", "rounds.jsonl"), cur,
                        os.path.join(os.path.dirname(cur), "log.json")) if os.path.exists(p)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/bundle/osu_rl_bundle.zip")
    p.add_argument("--no-fork", action="store_true")
    p.add_argument("--no-state", action="store_true")
    a = p.parse_args()
    if os.name == "nt":  # zipping a GB is heavy: a loop playing on this machine comes first
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL

    # tracked files and new ones not committed yet (not ignored), minus what does not belong on a stream machine
    listed = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], capture_output=True, text=True,
                            check=True).stdout.splitlines()
    skip = (".impeccable/", "live/lazer_ctl.py", "scratch/")
    tracked = [f for f in listed if os.path.isfile(f) and not f.startswith(skip) and not f.endswith("last.pt")]
    extra = [f for g in EXTRA for f in glob.glob(g)] + teacher_files() + ([] if a.no_state else state_files())
    files = sorted(set(os.path.normpath(f) for f in tracked + extra))

    fork_dir = None
    if not a.no_fork:
        fork_dir = tempfile.mkdtemp(prefix="osu_rl_fork_")
        print(f"publishing the fork (self-contained) from {FORK_SRC} ...", flush=True)
        env = dict(os.environ)
        env.setdefault("DOTNET_ROOT", r"C:\Program Files\dotnet")
        r = subprocess.run(["dotnet", "publish", FORK_SRC, "-c", "Release", "-r", "win-x64", "--self-contained", "true",
                            "-o", fork_dir], env=env, capture_output=True, text=True,
                           creationflags=0x4000)  # BELOW_NORMAL_PRIORITY_CLASS: a loop may be playing on this machine
        if r.returncode != 0 or not os.path.exists(os.path.join(fork_dir, "osu!.exe")):
            print(r.stdout[-3000:], r.stderr[-3000:])
            raise SystemExit("dotnet publish failed")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    total = 0
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in files:
            z.write(f, os.path.join("osu_rl", f))
            total += os.path.getsize(f)
        if fork_dir:
            for root, _, names in os.walk(fork_dir):
                for n in names:
                    src = os.path.join(root, n)
                    z.write(src, os.path.join("osu_rl", "fork", os.path.relpath(src, fork_dir)))
                    total += os.path.getsize(src)
    if fork_dir:
        shutil.rmtree(fork_dir, ignore_errors=True)
    print(f"{a.out}: {len(files)} files{' + the fork' if fork_dir else ''}, {total / 2**20:.0f} MB in, "
          f"{os.path.getsize(a.out) / 2**20:.0f} MB zipped")


if __name__ == "__main__":
    main()
