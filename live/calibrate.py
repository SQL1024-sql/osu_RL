"""Lines a newly trained student's click and hold outputs up with the model playing now, so the fixed thresholds play
mean the same for it (live.vision_play: a click fires on a click-logit peak above CLICK_TH, a key stays held while the
hold logit is above 0).

Why: fine-tuning made the click logits more confident on the negative side (2026-11-13: rounds 5-9, every candidate
from r001). Past the same -0.75 threshold the candidates clicked 18% less often: 7 more missed circles and nearly every
spinner failed per gate play, 2 accuracy points down, and none of them ever took over.

    python -m live.calibrate cand.pt --ref current.pt --recs <labelled recordings> --out cand_cal.pt

Both models see the same real frames (sampled from the recordings as in training); the candidate's last-layer biases
for click and hold are shifted so that each crosses its threshold as often as the reference's does.
"""
import argparse

import numpy as np
import torch

import vision.student as S
from live.dagger_label import MAP_GLOBS, map_index
from live.finetune_real import RealTracks
from live.vision_play import CLICK_TH


def logits(model, tracks, n_batches, size, seed=12345):
    gen = torch.Generator(device=tracks.dev).manual_seed(seed)
    cs, hs = [], []
    with torch.no_grad():
        for _ in range(n_batches):
            img, aux, _ = tracks.batch(size, gen)
            with S.autocast(tracks.dev):
                kl = model(img, aux)[1].float()
            cs.append(kl[:, 0].cpu().numpy())
            hs.append(kl[:, 1].cpu().numpy())
    return np.concatenate(cs), np.concatenate(hs)


def calibrate(cand, ref, recs, out, n_batches=40, size=512):
    """-> (click shift, hold shift) added to cand's biases, written to out."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tracks = RealTracks(recs, map_index(MAP_GLOBS), dev, clicks="map")
    found = {}
    for name, path in (("ref", ref), ("cand", cand)):
        model = S.Student().to(dev).to(memory_format=torch.channels_last)
        model.load_state_dict(torch.load(path, map_location=dev))
        model.eval()
        found[name] = logits(model, tracks, n_batches, size)
    (rc, rh), (cc, ch) = found["ref"], found["cand"]
    pc, ph = (rc > CLICK_TH).mean(), (rh > 0).mean()
    dc = float(CLICK_TH - np.quantile(cc, 1 - pc))
    dh = float(0 - np.quantile(ch, 1 - ph))
    sd = torch.load(cand, map_location="cpu")
    bias = [k for k in sd if k.startswith("head.") and k.endswith(".bias")][-1]  # the last layer: (move x, y, click, hold)
    sd[bias][2] += dc
    sd[bias][3] += dh
    torch.save(sd, out)
    print(f"reference: click over {CLICK_TH} {pc:.2%} of steps, hold over 0 {ph:.2%}; candidate before: "
          f"{(cc > CLICK_TH).mean():.2%}, {(ch > 0).mean():.2%} -> click {dc:+.3f}, hold {dh:+.3f} -> {out}", flush=True)
    return dc, dh


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cand")
    p.add_argument("--ref", required=True)
    p.add_argument("--recs", nargs="+", required=True, help="labelled recordings (live.dagger_label), real frames")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    calibrate(a.cand, a.ref, a.recs, a.out)


if __name__ == "__main__":
    main()
