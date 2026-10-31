"""Collects the real osu!standard beatmaps bundled with ppy/osu's test resources into maps/test/.

Beatmaps belong to their mappers, so they are not committed here; run this after cloning ppy/osu next to this repo.
"""
import glob
import os
import re
import zipfile

OSU_REPO = os.environ.get("OSU_REPO", "../ppy-osu")
OUT = "maps/test"


def is_standard(text):
    m = re.search(r"^Mode:\s*(\d)", text, re.M)
    return m is None or m.group(1) == "0"


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    seen, n = set(), 0
    for f in sorted(glob.glob(f"{OSU_REPO}/osu.Game.Rulesets.*.Tests/Resources/Testing/Beatmaps/*.osu")):
        name = os.path.basename(f)
        if not re.fullmatch(r"\d+\.osu", name):  # numeric names are real ranked maps; the rest are hand-made edge cases
            continue
        data = open(f, "rb").read()
        if data in seen or not is_standard(data.decode("utf-8-sig", "ignore")):
            continue
        seen.add(data)
        open(os.path.join(OUT, name), "wb").write(data)
        n += 1
    with zipfile.ZipFile(f"{OSU_REPO}/osu.Game.Tests/Resources/Archives/241526 Soleily - Renatus.osz") as z:
        for name in z.namelist():
            data = z.read(name)
            if name.endswith(".osu") and is_standard(data.decode("utf-8-sig", "ignore")):
                diff = re.search(r"\[(.*)\]", name).group(1).lower()
                open(os.path.join(OUT, f"renatus_{diff}.osu"), "wb").write(data)
                n += 1
    print(f"{n} maps -> {OUT}")
