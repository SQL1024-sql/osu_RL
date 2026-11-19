"""A cheap look at a candidate before the game: run it over recorded plays (what the screen showed, the cursor where it
was) step by step as live.vision_play would, decode its clicks with the live rule, and score them against the map.
The recordings were played by another model, so this is off-policy: it says how well the network sees *when* to click
(and, against live.oracle, where to go), not what it would score in a play of its own. It is meant as a filter in front
of live.ablate, not a replacement: check how well it ranks models whose real scores are known first.

    python -m live.offline_eval runs/fovea/f1/last_cal.pt runs/improve/r033_aim6/best_cal.pt --recs 12
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

import vision.student as S
from beatmap import SPINNER
from live.dagger_label import MAP_GLOBS, map_index
from vision.gpu_render import draw_dots
from vision.render import PX, H, W

CLICK_TH = -0.75
LAT_MS = 5.0  # frame age the student is told (live plays: about 5 ms)
GAPS = ((0, 100), (100, 150), (150, 250), (250, 1e9))


def gate_recs(n, exclude=()):
    """Recorded gate-map plays with close-ups (any model), newest first: maps the students never train on."""
    from live.improve import TIERS, maps_of, md5
    gate = {md5(p) for _, _, g in TIERS for p in maps_of(g)}
    out = []
    for f in sorted(glob.glob("runs/improve/plays/*/*-rec.npz"), key=os.path.getmtime, reverse=True):
        if f in exclude:
            continue
        try:
            z = np.load(f)
            if "fovea" not in z.files or str(z["md5"]) not in gate or not os.path.exists(str(z["play"])):
                continue
        except Exception:
            continue
        out.append(f)
        if len(out) >= n:
            break
    return out


def outputs(model, rec, dev):
    """The model's (move, click logit, hold logit) on every 120 Hz step of a recording, and those steps' game times."""
    z = np.load(rec)
    frames, gm = z["frames"], z["game_ms"]
    fov, fo = (z["fovea"], z["fovea_origin"]) if "fovea" in z.files else (None, None)
    log = np.load(str(z["play"]))["log"]
    ok = np.isfinite(log[:, 1])
    g, xy = log[ok, 1], log[ok, 2:4]
    t = np.arange(max(g[0], gm[0]) + 300, min(g[-1], gm[-1]) - 20, S.DT)
    cur = lambda tt: np.stack([np.interp(tt, g, xy[:, 0]), np.interp(tt, g, xy[:, 1])], -1)
    ms, cs, hs, ws = [], [], [], []
    use_fov = model.fovea is not None
    for i in range(0, len(t), 1024):
        tb = t[i:i + 1024]
        times = tb[:, None] - LAT_MS - np.array(S.FRAME_STEPS) * S.DT
        k = np.clip(np.searchsorted(gm, times, "right") - 1, 0, len(gm) - 1)
        dots = cur(times)
        img = torch.from_numpy(frames[k.reshape(-1)]).to(dev).float().div_(255)
        d = torch.tensor(np.c_[dots.reshape(-1, 2), np.full(dots.size // 2, 2.2)], dtype=torch.float32, device=dev)
        img = draw_dots(img, d).view(len(tb), S.N_FRAMES, H, W)
        crop = None
        if use_fov:
            side = model.fovea_side
            cf = torch.from_numpy(fov[k.reshape(-1)]).to(dev).float().div_(255)
            db = torch.tensor(np.c_[dots.reshape(-1, 2), np.full(dots.size // 2, 2.2 * S.FOVEA_PX / PX)], dtype=torch.float32, device=dev)
            org = torch.from_numpy(fo[k.reshape(-1)]).to(dev).float()
            crop = draw_dots(cf, db, org, S.FOVEA_PX).view(len(tb), S.N_FRAMES, side, side)
        c0 = cur(tb)
        aux = torch.tensor(np.c_[c0[:, 0] / 512, c0[:, 1] / 384, np.full(len(tb), LAT_MS / S.MAX_LAT)], dtype=torch.float32, device=dev)
        with torch.no_grad(), S.autocast(dev):
            m, kl = S.apply_model(model, img, crop, aux)
        ms.append(m.float().cpu().numpy())
        if model.when is not None:
            with torch.no_grad(), S.autocast(dev):
                ws.append(torch.softmax(model.when(model.last_h).float(), 1).cpu().numpy())
        kl = kl.float().cpu().numpy()
        cs.append(kl[:, 0])
        hs.append(kl[:, 1])
    outputs.when = np.concatenate(ws) if ws else None  # the when head's class probabilities per step, if it has one
    return t, np.concatenate(ms), np.concatenate(cs), np.concatenate(hs), cur(t), str(z["md5"])


def fires(c):
    """Steps where the live rule fires a click: the step after a peak of the click logit over CLICK_TH, once per stretch."""
    out, prev, fired = [], -np.inf, False
    for i, v in enumerate(c):
        if prev > CLICK_TH and v < prev and not fired:
            out.append(i)
            fired = True
        fired = fired and v > CLICK_TH
        prev = v
    return np.array(out, int)


def fires_when(t, w, th=0.5, lead_classes=3, same_ms=25.0):
    """Clicks from the when head: fire on a step where the chance that the next object is due within lead_classes x
    WHEN_MS (the key goes down a step later) is over th, unless the object it expects is the one it last fired for
    (its expected time within same_ms of that one's): a click is remembered, so a stream note is not clicked twice and
    the next one is not skipped for want of a dip."""
    centres = (np.arange(w.shape[1]) + 0.5) * S.WHEN_MS
    soon = w[:, :lead_classes].sum(1)
    near = w[:, :13]  # within ~50 ms: where it thinks the next object is
    out, last_target = [], -1e9
    for i in range(len(t)):
        if soon[i] > th:
            target = t[i] + (near[i] * centres[:13]).sum() / max(near[i].sum(), 1e-6)
            if abs(target - last_target) > same_ms:
                out.append(i)
                last_target = target
    return np.array(out, int)


def fires_sched(t, w, min_mass=0.2):
    """student.WhenScheduler over a play's when-head outputs (the key goes down a step after the decision)."""
    sch = S.WhenScheduler(min_mass=min_mass)
    return np.array([i for i in range(len(t)) if sch.step(t[i], w[i], t[i] + S.DT)], int)


def score(t, c, md5, index, fire_steps=None):
    """Clicks against the map: each circle / slider head takes the first unused click within its 50 window (like
    lazer, in time order). -> per object (gap to the one before, found, click time - start) and the clicks left over."""
    bm = S.load(index[md5])
    w50 = bm.windows[2]
    ft = t[fires(c) if fire_steps is None else fire_steps] + S.DT  # the key goes down on the step after the decision... sent within that step
    used = np.zeros(len(ft), bool)
    objs = []
    starts, kinds = bm.objects[:, 2], bm.kinds
    for j in range(len(starts)):
        if kinds[j] == SPINNER or starts[j] < t[0] + 200 or starts[j] > t[-1] - 200:
            continue
        gap = starts[j] - starts[j - 1] if j else 1e9
        cand = np.nonzero(~used & (ft >= starts[j] - w50) & (ft <= starts[j] + w50))[0]
        if len(cand):
            used[cand[0]] = True
            objs.append((gap, 1, ft[cand[0]] - starts[j]))
        else:
            objs.append((gap, 0, np.nan))
    return np.array(objs, float), int((~used).sum())


def oracle_agreement(t, m, pos, md5, index):
    """How the model's moves point against live.oracle's (finetune_real.oracle_moves): share within 45 degrees, on
    steps where the oracle moves."""
    from live.finetune_real import oracle_moves
    om = oracle_moves(index[md5], t, pos)
    mv = np.hypot(*om.T) > 0.05
    cos = (m[mv] * om[mv]).sum(1) / (np.hypot(*m[mv].T) * np.hypot(*om[mv].T) + 1e-9)
    return float(np.mean(cos > 0.7))


def evaluate(ckpt, recs, index, dev, decoder="logit", th=0.5):
    """decoder: "logit" (the live rule on the click logit), "when" (fires_when on the when head, a threshold: worse)
    or "sched" (student.WhenScheduler on the when head)."""
    model = S.make_model(ckpt, dev)
    objs, extra, agree = [], 0, []
    for rec in recs:
        t, m, c, h, pos, md5 = outputs(model, rec, dev)
        if decoder == "when":
            fs = fires_when(t, outputs.when, th)
        elif decoder == "sched":
            fs = fires_sched(t, outputs.when, min_mass=th)
        else:
            fs = None
        o, e = score(t, c, md5, index, fs)
        objs.append(o)
        extra += e
        agree.append(oracle_agreement(t, m, pos, md5, index))
    o = np.concatenate(objs)
    res = dict(ckpt=ckpt, objects=len(o), found=float(o[:, 1].mean()), extra_per_play=extra / len(recs),
               abs_err=float(np.nanmean(np.abs(o[:, 2]))), oracle_dir=float(np.mean(agree)))
    for lo, hi in GAPS:
        m_ = (o[:, 0] >= lo) & (o[:, 0] < hi)
        res[f"found_{lo}"] = float(o[m_, 1].mean()) if m_.any() else float("nan")
    return res


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--recs", type=int, default=12, help="this many recorded gate plays (newest with close-ups)")
    p.add_argument("--exclude", nargs="*", default=[], help="recordings to leave out (e.g. a training run's held-out ones)")
    p.add_argument("--out", help="also append the results here (jsonl)")
    p.add_argument("--decoder", choices=["logit", "when", "sched"], default="logit")
    p.add_argument("--th", type=float, nargs="*", default=[0.5], help="--decoder when: thresholds to try")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    index = map_index(MAP_GLOBS)
    recs = gate_recs(a.recs, set(a.exclude))
    print(f"{len(recs)} recorded gate plays", flush=True)
    print(f"{'model':40s} found  <100  100-150 150-250 250+   extra/play |err| ms  oracle-dir")
    for ck, th in [(c, x) for c in a.ckpts for x in (a.th if a.decoder in ("when", "sched") else [None])]:
        r = evaluate(ck, recs, index, dev, a.decoder, th)
        r["decoder"], r["th"] = a.decoder, th
        r["composite"] = r["found"] - r["extra_per_play"] / (r["objects"] / len(recs))
        print(f"{(ck[-34:] + (f' th{th}' if th is not None else '')):40s} {r['found']:.3f} {r['found_0']:.3f} {r['found_100']:.3f}   {r['found_150']:.3f}  {r['found_250']:.3f}"
              f"  {r['extra_per_play']:6.1f}    {r['abs_err']:5.1f}   {r['oracle_dir']:.3f}  composite {r['composite']:.3f}", flush=True)
        if a.out:
            with open(a.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(r, recs=recs)) + "\n")


if __name__ == "__main__":
    main()
