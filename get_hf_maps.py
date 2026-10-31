"""Extracts osu!standard charts from project-riz/osu-beatmaps WebDataset shards into maps/hf/{train,val,test}/.

    python get_hf_maps.py data/hf/data-000000.tar
    curl -sL <shard url> | python get_hf_maps.py -     # streams: the audio never touches the disk
Audio is skipped. Maps that share a beatmapset with any maps/test/ map are dropped, so the original test set stays
unseen. The split is by beatmapset, so difficulties of one song never land on both sides.
"""
import csv
import glob
import hashlib
import json
import os
import re
import sys
import tarfile

OUT = "maps/hf"


def test_set_keys():
    md5s, ids, sets = set(), set(), set()
    for f in glob.glob("maps/test/*.osu"):
        data = open(f, "rb").read()
        md5s.add(hashlib.md5(data).hexdigest())
        text = data.decode("utf-8-sig", "ignore")
        for key, dest in (("BeatmapID", ids), ("BeatmapSetID", sets)):
            m = re.search(rf"^{key}:\s*(\d+)", text, re.M)
            if m:
                dest.add(int(m.group(1)))
        if re.fullmatch(r"\d+\.osu", os.path.basename(f)):  # ppy test resources are named by beatmap id
            ids.add(int(os.path.basename(f)[:-4]))
    return md5s, ids, sets


def split_of(beatmapset_id):
    h = int(hashlib.sha256(str(beatmapset_id).encode()).hexdigest(), 16) % 20
    return "test" if h == 0 else "val" if h == 1 else "train"


if __name__ == "__main__":
    md5s, ids, sets = test_set_keys()
    for s in ("train", "val", "test"):
        os.makedirs(os.path.join(OUT, s), exist_ok=True)
    index_path = os.path.join(OUT, "index.csv")
    new = not os.path.exists(index_path)
    index = open(index_path, "a", newline="", encoding="utf-8")
    w = csv.writer(index)
    if new:
        w.writerow(["beatmap_id", "beatmapset_id", "split", "cs", "ar", "od", "stars", "bpm", "hit_length", "max_combo", "approved", "artist", "title", "version"])
    kept = skipped_test = 0
    for shard in sys.argv[1:]:
        with (tarfile.open(fileobj=sys.stdin.buffer, mode="r|") if shard == "-" else tarfile.open(shard)) as tar:
            for member in tar:
                if not member.name.endswith(".json"):
                    continue
                for b in json.load(tar.extractfile(member))["beatmaps"]:
                    if b["mode"] != 0:
                        continue
                    content = b["content"].encode("utf-8")
                    if b["beatmap_id"] in ids or b["beatmapset_id"] in sets or hashlib.md5(content).hexdigest() in md5s:
                        skipped_test += 1
                        continue
                    split = split_of(b["beatmapset_id"])
                    with open(os.path.join(OUT, split, f"{int(b['beatmap_id'])}.osu"), "wb") as f:
                        f.write(content)
                    w.writerow([b["beatmap_id"], b["beatmapset_id"], split, b["diff_size"], b["diff_approach"], b["diff_overall"],
                                b["difficultyrating"], b["bpm"], b["hit_length"], b["max_combo"], b["approved"], b["artist"], b["title"], b["version"]])
                    kept += 1
    index.close()
    print(f"kept {kept} osu!standard maps -> {OUT}, dropped {skipped_test} that overlap the test set")
