"""Fine-tunes the vision student on real lazer frames (live.record recordings of the teacher playing through
live.screen_play), labelled by live.dagger_label (the student's teacher, asked in every state the play went through) exactly
as student.Tracks labels its rendered plays: teacher-driven recordings and student-driven ones (DAgger) alike.
Every batch mixes in rendered DAgger data (a plays*.pkl of a student run) so the student keeps what it knew about
other maps while it learns how lazer's screen looks.

    python -m live.finetune_real runs/live_rec/1_*.npz runs/live_rec/2_*.npz --val runs/live_rec/5_*.npz

Each recording's map is found by its MD5 among --maps. With --val, those recordings (best: other maps) are held out
whole; without it, the part after --split (by game time) of the last recording is held out and only the parts before
it are trained on. The student is judged on the held-out frames every --every updates (loss); the best and last
weights go to --out. Judge clicks and moves with live.vision_eval <val recording> <its map> --ckpt <out>/best.pt.
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F

import vision.student as S
from vision.gpu_render import draw_dots
from live.dagger_label import MAP_GLOBS, load_labels, map_index
from vision.render import PX, H, W

SPIN_CENTER, SPIN_RADIUS, SPIN_SPEED = np.array([256.0, 192.0]), 60.0, 0.9


def spin_moves(pos):
    """Moves for cursor positions inside a spinner: around the playfield centre, clockwise on screen (the way r001
    spins), at SPIN_SPEED of the top speed, steered towards SPIN_RADIUS. The teacher's own spinner moves are no lesson:
    around the centre they average to nothing (it wiggles off-centre, which the simulator accepts), and in the
    student's states it points one way, then the other; trained on them, every candidate stopped spinning
    (0.12 turns/s against r001's 2, 2026-11-14)."""
    r = pos - SPIN_CENTER
    d = np.linalg.norm(r, axis=1, keepdims=True) + 1e-6
    tangent = np.c_[r[:, 1], -r[:, 0]] / d  # clockwise on screen (y down)
    radial = -r / d * np.clip((d - SPIN_RADIUS) / SPIN_RADIUS, -1, 1)
    v = tangent + radial
    return (v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-6) * SPIN_SPEED).astype(np.float32)


class RealTracks:
    """Recordings on the GPU, sampled like student.Tracks: random steps of the teacher's play, the 4 frames the screen
    showed before each (latest recorded frame at or before each frame time), the teacher's cursor drawn in, aux, and
    the labels live.dagger_label wrote for it (the teacher's answer in the state the play was in). spans: per recording, the game time range (ms) to sample from, or None."""

    def __init__(self, recs, index, dev, spans=None, clicks="map", frames_dev=None, spin_frac=0.0, spin_rule=False,
                 aim_w=1.0, aim_ms=60.0, fovea=0):
        """frames_dev "cpu": the frames stay in system memory and each batch's are copied over, so far more
        recordings fit than in GPU memory. spin_frac: that share of each recording's samples drawn from inside its
        spinners (fine-tuning washed spinning out: spinners are a sliver of a play, 2026-11-13)."""
        self.dev, self.recs, self.spin_frac = dev, [], spin_frac
        # aim_w: weight of the move loss on steps within aim_ms before a circle's or slider's start (a fifth label
        # column): 45% of the misses were the cursor just off the circle when it clicked (2026-11-14)
        self.aim_w, self.aim_ms = aim_w, aim_ms
        # fovea: the close-up side the model takes (0: none). Each batch's close-ups go to last_crop (None without);
        # recordings made before live.vision_play kept close-ups (2026-11-14) have none and are left out
        self.fovea, self.last_crop = fovea, None
        frames_dev = frames_dev or dev
        skipped = 0
        for n, path in enumerate(recs):
            r = np.load(path)
            if fovea and "fovea" not in r.files:
                skipped += 1
                continue
            md5 = str(r["md5"])
            assert md5 in index, f"{path}: its map ({md5}) is not among --maps"
            assert os.path.exists(path[:-4] + ".labels.npz"), f"{path}: no labels; run live.dagger_label on it first"
            tt, pos, lab, _ = load_labels(path, clicks)
            t = r["game_ms"].astype(np.float64)
            lo, hi = max(t[0], tt[0]) + 300, min(t[-1], tt[-1]) - 20
            if spans and spans[n]:
                lo, hi = max(lo, spans[n][0]), min(hi, spans[n][1])
            frames = torch.from_numpy(np.ascontiguousarray(r["frames"]))  # uint8 (n, H, W)
            # (not pinned: Windows caps page-locked memory, and pinning 9 GB of frames failed with "CUDA out of memory")
            frames = frames if frames_dev == "cpu" else frames.to(frames_dev)
            sc = S.scene(index[md5])
            spins = [(max(lo, float(a) + 150), min(hi, float(b) - 100)) for a, b, k in zip(sc.start, sc.end, sc.kind) if k == 2]
            spins = [(a, b) for a, b in spins if b - a > 200]
            if spin_rule:  # moves inside spinners from spin_moves, not the teacher
                for a, b, k in zip(sc.start, sc.end, sc.kind):
                    if k == 2:
                        m = (tt >= a) & (tt <= b)
                        lab = lab.copy()
                        lab[m, :2] = spin_moves(pos[m])
            starts = np.sort(np.array([a for a, k in zip(sc.start, sc.kind) if k != 2], np.float64))
            fov = {}
            if fovea:
                fv = torch.from_numpy(np.ascontiguousarray(r["fovea"]))  # uint8 (n, side, side)
                fov = dict(fovea=fv if frames_dev == "cpu" else fv.to(frames_dev),
                           fovea_origin=torch.tensor(r["fovea_origin"], dtype=torch.float32, device=dev))
            self.recs.append(dict(path=path, map=index[md5], lo=lo, hi=hi, spins=spins, frames=frames, **fov,
                                  starts=torch.tensor(starts, device=dev),
                                  rec_t=torch.tensor(t, device=dev), tt=torch.tensor(tt, device=dev),
                                  pos=torch.tensor(pos, dtype=torch.float32, device=dev),
                                  lab=torch.tensor(lab, device=dev)))
        if skipped:
            print(f"{skipped} of {len(recs)} recordings have no close-ups: left out", flush=True)
        assert self.recs, "no usable recordings"
        w = np.array([r["hi"] - r["lo"] for r in self.recs])
        self.weight = torch.tensor(w / w.sum(), device=dev)  # recordings sampled by length

    def batch(self, size, gen):
        dev = self.dev
        which = torch.multinomial(self.weight, size, replacement=True, generator=gen)
        img = torch.empty(size, S.N_FRAMES, H, W, device=dev)
        crop = torch.empty(size, S.N_FRAMES, self.fovea, self.fovea, device=dev) if self.fovea else None
        aux = torch.empty(size, S.AUX, device=dev)
        y = torch.empty(size, 4 if self.aim_w == 1.0 else 5, device=dev)
        for i, r in enumerate(self.recs):
            m = torch.nonzero(which == i).flatten()
            n = len(m)
            if not n:
                continue
            at = lambda t: (torch.searchsorted(r["tt"], t.double().contiguous(), right=True) - 1).clamp(0, len(r["tt"]) - 1)
            t = r["lo"] + (r["hi"] - r["lo"]) * torch.rand(n, generator=gen, device=dev, dtype=torch.float64)
            if self.spin_frac and r["spins"]:
                ns = int(round(self.spin_frac * n))
                if ns:
                    sp = torch.tensor(r["spins"], device=dev, dtype=torch.float64)
                    j = torch.multinomial(sp[:, 1] - sp[:, 0], ns, replacement=True, generator=gen)
                    t[:ns] = sp[j, 0] + (sp[j, 1] - sp[j, 0]) * torch.rand(ns, generator=gen, device=dev, dtype=torch.float64)
            lat = torch.rand(n, generator=gen, device=dev) * S.MAX_LAT
            lag = torch.randint(0, 3, (n, 1), generator=gen, device=dev) * S.DT  # cursor drawn up to 2 frames late
            rad = 1.5 + 1.5 * torch.rand(n, 1, generator=gen, device=dev)
            times = t[:, None] - lat[:, None].double() - torch.tensor(S.FRAME_STEPS, device=dev) * S.DT  # (n, 4)
            k = (torch.searchsorted(r["rec_t"], times.contiguous(), right=True) - 1).clamp(min=0)
            fr = r["frames"]
            frames = (fr[k.cpu()].to(dev, non_blocking=True) if fr.device != k.device else fr[k]).float() / 255
            dots = torch.cat([r["pos"][at(times - lag)], rad[:, :, None].expand(-1, S.N_FRAMES, 1)], 2)
            img[m] = draw_dots(frames.view(-1, H, W), dots.reshape(-1, 3)).view(n, S.N_FRAMES, H, W)
            if crop is not None:  # the close-ups those frames kept, the cursor drawn at the same places, 3x the size
                fv = r["fovea"]
                cf = (fv[k.cpu()].to(dev, non_blocking=True) if fv.device != k.device else fv[k]).float() / 255
                db = torch.cat([dots[..., :2], dots[..., 2:] * S.FOVEA_PX / PX], 2)
                crop[m] = draw_dots(cf.view(-1, self.fovea, self.fovea), db.reshape(-1, 3), r["fovea_origin"][k].reshape(-1, 2),
                                    S.FOVEA_PX).view(n, S.N_FRAMES, self.fovea, self.fovea)
            g = at(t)  # the labelled step: the teacher's answer in the state the play was in
            cur = r["pos"][g]
            aux[m] = torch.stack([cur[:, 0] / 512, cur[:, 1] / 384, lat / S.MAX_LAT], 1)
            y[m, :4] = r["lab"][g]
            if self.aim_w != 1.0:
                nxt = r["starts"][torch.searchsorted(r["starts"], t.contiguous()).clamp(max=len(r["starts"]) - 1)]
                near = (nxt >= t) & (nxt - t <= self.aim_ms)
                y[m, 4] = torch.where(near, torch.tensor(self.aim_w, device=dev), torch.tensor(1.0, device=dev)).float()
        self.last_crop = crop
        return img, aux, y


def augment_crop(crop):
    """The rendered close-ups' augmentation (student.build): brightness and noise."""
    b = crop.shape[0]
    gain = torch.empty(b, 1, 1, 1, device=crop.device).uniform_(0.75, 1.25)
    sigma = torch.empty(b, 1, 1, 1, device=crop.device).uniform_(0, 0.03)
    return (crop * gain + torch.randn_like(crop) * sigma).clamp(0, 1)


def loss_of(model, img, aux, y, crop=None):
    with S.autocast(img.device.type):
        m, k = S.apply_model(model, img, crop, aux)[:2]
    if y.shape[1] > 4:  # per-sample move weights (RealTracks aim_w)
        w = y[:, 4]
        move = (((m.float() - y[:, :2]) ** 2).mean(1) * w).sum() / w.sum()
    else:
        move = F.mse_loss(m.float(), y[:, :2])
    return move + F.binary_cross_entropy_with_logits(k.float(), y[:, 2:4])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("recs", nargs="+", help="live.record recordings of the teacher playing (any maps among --maps)")
    p.add_argument("--val", nargs="*", help="recordings held out whole (default: the last part of the last recording)")
    p.add_argument("--split", type=float, default=0.7, help="without --val: share of each map (by game time) trained on")
    p.add_argument("--maps", nargs="*", default=MAP_GLOBS, help="where the recordings' maps are (found by MD5)")
    p.add_argument("--init", default="results/experiments/020-student/seed0/round6.pt")
    p.add_argument("--clicks", default="map", choices=["map", "teacher"],
                   help="click / hold labels from the map and the cursor (Exp 025), or from the teacher's keys")
    p.add_argument("--plays", nargs="*", default=["results/experiments/020-student/seed0/plays7.pkl"],
                   help="rendered DAgger data mixed into every batch")
    p.add_argument("--real-frac", type=float, default=0.5, help="share of each batch from real frames")
    p.add_argument("--synth-pool", type=int, default=300,
                   help="rendered batches made up front and drawn from at random (rendering without triton is slow)")
    p.add_argument("--frames", choices=["gpu", "cpu"], default="gpu",
                   help="where the recordings' frames are kept: cpu fits many more recordings (each batch is copied over)")
    p.add_argument("--spin-frac", type=float, default=0.0, help="share of real samples drawn from inside spinners")
    p.add_argument("--spin-rule", action="store_true", help="moves inside spinners from spin_moves, not the teacher's")
    p.add_argument("--aim-w", type=float, default=1.0, help="move-loss weight on steps just before an object's start")
    p.add_argument("--aim-ms", type=float, default=60.0, help="... within this many ms before it")
    p.add_argument("--fovea", type=int, default=None,
                   help="close-up side (96) for a model that also looks at one: from an --init without, its weights are "
                        "carried over (student.warm_start_fovea). Default: whatever --init has")
    p.add_argument("--updates", type=int, default=6000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/experiments/027-real-ft/seed0")
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    index = map_index(args.maps)
    init = torch.load(args.init, map_location=dev)
    fovea = S.fovea_of(init) if args.fovea is None else args.fovea
    if fovea and not S.fovea_of(init):
        init = S.warm_start_fovea(init, fovea)
        print(f"close-up branch ({fovea} px) added to {args.init}: it plays as before until it learns to use it", flush=True)
    S.FOVEA = fovea  # student.build renders close-ups for the rendered half too
    if args.val:
        fd = "cpu" if args.frames == "cpu" else dev
        real = RealTracks(args.recs, index, dev, clicks=args.clicks, frames_dev=fd, spin_frac=args.spin_frac,
                          spin_rule=args.spin_rule, aim_w=args.aim_w, aim_ms=args.aim_ms, fovea=fovea)
        held = RealTracks(args.val, index, dev, clicks=args.clicks, frames_dev=fd, fovea=fovea)
        split_ms = None
    else:
        tt = load_labels(args.recs[-1])[0]
        split_ms = float(tt[0] + (tt[-1] - tt[0]) * args.split)
        real = RealTracks(args.recs, index, dev, [(-1e9, split_ms)] * len(args.recs), args.clicks)
        held = RealTracks(args.recs[-1:], index, dev, [(split_ms + 500, 1e9)], args.clicks)
    for name, tr in (("train", real), ("held out", held)):
        for r in tr.recs:
            print(f"{name:>8}: {os.path.basename(r['path'])} ({(r['hi'] - r['lo']) / 1000:.0f} s of {os.path.basename(r['map'])})", flush=True)

    maps = S.Maps(dev)
    plays = []
    for f in args.plays:
        with open(f, "rb") as fh:
            plays += pickle.load(fh)
    S.MAP_KEYS = args.clicks == "map"  # the rendered half labelled the same way as the real half
    synth = S.Tracks(plays, maps) if plays else None

    model = S.make_model(init, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    vgen = torch.Generator(device=dev)
    rng = np.random.default_rng(args.seed)
    n_real = int(256 * args.real_frac) if synth is not None else 256
    pool = None
    if synth is not None and n_real < 256:
        # rendered frames once, as uint8 (no augmentation: that is applied per update, like for the real ones)
        t1 = time.time()
        imgs, auxs, ys, crops = [], [], [], []
        for _ in range(args.synth_pool):
            inputs, y = synth.batch(256 - n_real, gen)
            built = S.build(maps, *inputs)
            img, aux = built[0], built[-1]
            imgs.append((img * 255).round().to(torch.uint8))
            if fovea:
                crops.append((built[1] * 255).round().to(torch.uint8))
            auxs.append(aux)
            ys.append(y[:, :4])
        pool = torch.cat(imgs), torch.cat(auxs), torch.cat(ys), (torch.cat(crops) if fovea else None)
        print(f"rendered a pool of {len(pool[0])} samples in {time.time() - t1:.0f} s", flush=True)

    def held_loss():
        model.eval()
        vgen.manual_seed(12345)  # the same held-out samples every time
        with torch.no_grad():
            ls = []
            for _ in range(8):
                hb = held.batch(512, vgen)
                ls.append(loss_of(model, *hb, crop=held.last_crop).item())
        model.train()
        return float(np.mean(ls))

    log, best = [], (held_loss(), 0)
    print(f"update 0: held-out real loss {best[0]:.4f}", flush=True)
    torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
    model.train()
    t0 = time.time()
    for step in range(1, args.updates + 1):
        img, aux, y = real.batch(n_real, gen)
        crop = real.last_crop
        if pool is not None:
            pick = torch.randint(len(pool[0]), (256 - n_real,), generator=gen, device=dev)
            img = torch.cat([img, pool[0][pick].float() / 255])
            if crop is not None:
                crop = torch.cat([crop, pool[3][pick].float() / 255])
            py = pool[2][pick]
            if y.shape[1] > py.shape[1]:  # the rendered half has no aim weights: 1
                py = torch.cat([py, torch.ones(len(py), y.shape[1] - py.shape[1], device=dev)], 1)
            aux, y = torch.cat([aux, pool[1][pick]]), torch.cat([y, py])
        img = S.augment(img, rng)
        if crop is not None:
            crop = augment_crop(crop)
        loss = loss_of(model, img, aux, y, crop)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % args.every == 0 or step == args.updates:
            hl = held_loss()
            log.append(dict(update=step, train_loss=loss.item(), held_loss=hl))
            mark = ""
            if hl < best[0]:
                best = (hl, step)
                torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
                mark = "  (best)"
            print(f"update {step}: train loss {loss.item():.4f}, held-out real loss {hl:.4f}{mark}"
                  f" ({(time.time() - t0) / step * 1000:.0f} ms/update)", flush=True)
    torch.save(model.state_dict(), os.path.join(args.out, "last.pt"))
    json.dump(dict(args=vars(args), split_ms=split_ms, best_update=best[1], log=log),
              open(os.path.join(args.out, "log.json"), "w"), indent=1)
    print(f"best held-out loss {best[0]:.4f} at update {best[1]} -> {args.out}/best.pt"
          + (f"; split at {split_ms:.0f} ms" if split_ms else ""), flush=True)


if __name__ == "__main__":
    main()
