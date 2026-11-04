"""render.py on the GPU, for whole batches of frames: the CPU only looks up which objects are on screen at each
frame's time (Scene.window); distances, colours and compositing run in one torch call. Same parameters as render.py.

    python gpu_render.py maps/test/renatus_hard.osu      # compares against render.py and times both
"""
import sys

import numpy as np
import torch

from beatmap import SLIDER, SPINNER, parse_osu
from render import (APPROACH, APPROACH_W, BODY, BORDER, FADE_OUT, FILL, NUMBER, PX, RIM, RIM_W, SPIN, H, W, X0, Y0)

V = 12  # objects drawn per frame at most (the earliest V of those on screen)
K = 24  # points a slider path is resampled to
SPIN_RADII = (40.0, 120.0, 170.0)


def resample(path, cum, k):
    d = np.linspace(0, cum[-1], k)
    return np.stack([np.interp(d, cum, path[:, 0]), np.interp(d, cum, path[:, 1])], 1)


class Scene:
    """One beatmap's objects as arrays."""

    def __init__(self, bm):
        n = len(bm.objects)
        self.r, self.preempt = float(bm.radius), float(bm.preempt)
        self.fade = 400 * min(1, self.preempt / 450)
        self.start = bm.objects[:, 2].astype(np.float32)
        self.end = bm.ends.astype(np.float32)
        self.kind = bm.kinds.astype(np.int64)
        self.path = np.repeat(bm.objects[:, None, :2], K, 1).astype(np.float32)
        self.spans = np.ones(n, np.float32)
        for i, s in bm.sliders.items():
            self.path[i] = resample(s.path, s.cum, K)
            self.spans[i] = s.spans
        self.appear = self.start - self.preempt  # sorted, like the start times

    def window(self, t):
        """Indices of the objects on screen at time t (ms), earliest first, at most V."""
        hi = int(np.searchsorted(self.appear, t, "right"))
        idx = np.arange(max(0, hi - 4 * V), hi)
        return idx[self.end[idx] + FADE_OUT >= t][:V]


def pack(frames, device):
    """[(scene, t)] -> tensors for render()."""
    b = len(frames)
    valid = np.zeros((b, V), bool)
    kind, start, end, spans = np.zeros((b, V), np.int64), np.zeros((b, V), np.float32), np.zeros((b, V), np.float32), np.ones((b, V), np.float32)
    path = np.zeros((b, V, K, 2), np.float32)
    per = np.zeros((b, 4), np.float32)  # t, r, preempt, fade
    for j, (s, t) in enumerate(frames):
        idx = s.window(t)
        m = len(idx)
        valid[j, :m] = True
        kind[j, :m], start[j, :m], end[j, :m], spans[j, :m], path[j, :m] = s.kind[idx], s.start[idx], s.end[idx], s.spans[idx], s.path[idx]
        per[j] = t, s.r, s.preempt, s.fade
    to = lambda a: torch.from_numpy(a).to(device, non_blocking=True)
    return dict(valid=to(valid), kind=to(kind), start=to(start), end=to(end), spans=to(spans), path=to(path), per=to(per))


MAP_STRIDE = 1e8  # ms; makes (map, time) one sorted key


class SceneTable:
    """Many maps' objects as GPU tensors, so whole batches of (map, time) find what is on screen without the CPU."""

    def __init__(self, scenes, device):
        cat = lambda name: torch.from_numpy(np.concatenate([getattr(s, name) for s in scenes])).to(device)
        self.start, self.end, self.kind, self.spans, self.path = (cat(n) for n in ("start", "end", "kind", "spans", "path"))
        counts = np.array([len(s.start) for s in scenes])
        self.first = torch.from_numpy(np.concatenate([[0], np.cumsum(counts)[:-1]])).to(device)
        ids = np.repeat(np.arange(len(scenes)), counts)
        self.keys = torch.from_numpy(ids * MAP_STRIDE + np.concatenate([s.appear for s in scenes]).astype(np.float64)).to(device)
        self.per_map = torch.tensor([[s.r, s.preempt, s.fade] for s in scenes], dtype=torch.float32, device=device)

    def pack(self, map_ids, t):
        """map_ids (B,) long, t (B,) ms -> the same dict as pack(), on the GPU."""
        b = len(t)
        hi = torch.searchsorted(self.keys, map_ids.double() * MAP_STRIDE + t.double(), right=True)
        cand = hi[:, None] - 4 * V + torch.arange(4 * V, device=t.device)
        c = cand.clamp(min=0)
        ok = (cand >= self.first[map_ids][:, None]) & (self.end[c] + FADE_OUT >= t[:, None])
        rank = ok.cumsum(1) - 1
        keep = ok & (rank < V)  # the earliest V on screen, like Scene.window
        idx = torch.full((b, V + 1), -1, dtype=torch.long, device=t.device)
        idx.scatter_(1, torch.where(keep, rank, V), torch.where(keep, c, -1))
        idx = idx[:, :V]
        g = idx.clamp(min=0)
        valid = idx >= 0
        return dict(valid=valid, kind=torch.where(valid, self.kind[g], 0), start=self.start[g], end=self.end[g],
                    spans=self.spans[g], path=self.path[g], per=torch.cat([t[:, None].float(), self.per_map[map_ids]], 1))


def _grid(device):
    gx = X0 + (torch.arange(W, device=device) + 0.5) / PX
    gy = Y0 + (torch.arange(H, device=device) + 0.5) / PX
    return gx.view(1, 1, W), gy.view(1, H, 1)


def _disk(d, rad):
    return ((rad - d) * PX + 0.5).clamp(0, 1)  # coverage of a pixel by a disk, 1 px soft edge


def _ring(d, rad, width):
    return ((width / 2 - (d - rad).abs()) * PX + 0.5).clamp(0, 1)


def render(p):
    """-> (B, H, W) float in [0, 1]."""
    gx, gy = _grid(p["per"].device)
    t, r, pre, fade = (p["per"][:, i].view(-1, 1, 1) for i in range(4))
    rim_w = (RIM_W * r * PX).round().clamp(min=1) / PX
    app_w = (APPROACH_W * r * PX).round().clamp(min=1) / PX
    out = torch.zeros(t.shape[0], H, W, device=t.device)
    for v in reversed(range(V)):  # earlier objects end up on top
        ok = p["valid"][:, v].view(-1, 1, 1)
        kind = p["kind"][:, v].view(-1, 1, 1)
        start, end = p["start"][:, v].view(-1, 1, 1), p["end"][:, v].view(-1, 1, 1)
        path = p["path"][:, v]  # (B, K, 2)
        appear = start - pre
        alpha = ((t - appear) / fade).clamp(max=1) * torch.where(t > end, (1 - (t - end) / FADE_OUT).clamp(min=0), 1.0)
        alpha = torch.where(ok, alpha.clamp(min=0), 0.0)
        slider, spinner = kind == SLIDER, kind == SPINNER

        # slider body: distance to the (snaking-in) path
        shown = ((t - appear) / (pre / 3)).clamp(0, 1) * (K - 1)  # path points shown so far
        shown = shown.clamp(min=1)
        d2 = torch.full_like(out, 1e8)
        for k in range(K):  # one point at a time: (B, K, H, W) at once would not fit
            dk = (gx - path[:, k, 0].view(-1, 1, 1)) ** 2 + (gy - path[:, k, 1].view(-1, 1, 1)) ** 2
            d2 = torch.where(shown >= k, torch.minimum(d2, dk), d2)
        db = d2.sqrt()
        body, inner = _disk(db, r) * slider, _disk(db, 0.85 * r) * slider
        layer = BORDER * (body - inner) + BODY * inner
        mask = body

        # the head, or while sliding the ball (drawn like a hit circle by this skin)
        prog = ((t - start) / (end - start).clamp(min=1e-3)).clamp(0, 1) * p["spans"][:, v].view(-1, 1, 1)
        span = prog.floor().clamp(max=p["spans"][:, v].view(-1, 1, 1) - 1)
        f = prog - span
        along = torch.where(span % 2 == 1, 1 - f, f) * (K - 1)
        lo = along.floor().long().clamp(0, K - 2).view(-1)
        w = (along.view(-1) - lo)
        ball = path[torch.arange(len(lo)), lo] * (1 - w[:, None]) + path[torch.arange(len(lo)), lo + 1] * w[:, None]
        c = torch.where((slider & (t > start)).view(-1, 1), ball, path[:, 0])
        dc = torch.sqrt((gx - c[:, 0].view(-1, 1, 1)) ** 2 + (gy - c[:, 1].view(-1, 1, 1)) ** 2)
        disk, num, rim = _disk(dc, r), _disk(dc, r / 4), _ring(dc, r, rim_w)
        layer = torch.lerp(layer, torch.full_like(layer, FILL), disk)
        layer = torch.lerp(layer, torch.full_like(layer, NUMBER), num)
        layer = torch.lerp(layer, torch.full_like(layer, RIM), rim)
        mask = torch.maximum(mask, disk)

        # approach circle: 4r down to r around the head
        dh = torch.sqrt((gx - path[:, 0, 0].view(-1, 1, 1)) ** 2 + (gy - path[:, 0, 1].view(-1, 1, 1)) ** 2)
        a = ((t - appear) / torch.minimum(2 * fade, pre)).clamp(0, 1) * (t < start)
        cov = _ring(dh, r * (1 + 3 * (start - t) / pre), app_w) * (a > 0)  # like cv2: the ring's colour is APPROACH * a, its mask a
        layer = torch.lerp(layer, APPROACH * a.expand_as(layer), cov)
        mask = torch.maximum(mask, cov * a)

        # spinner: three thin rings in the middle
        sp_alpha = torch.where(ok, ((t - (start - 500)) / 500).clamp(0, 1) * torch.where(t > end, (1 - (t - end) / FADE_OUT).clamp(min=0), 1.0), 0.0)
        ds = torch.sqrt((gx - 256) ** 2 + (gy - 192) ** 2)
        srings = sum(_ring(ds, rr, 1 / PX) for rr in SPIN_RADII).clamp(max=1)
        layer = torch.where(spinner, SPIN * srings, layer)
        mask = torch.where(spinner, srings, mask)
        alpha = torch.where(spinner, sp_alpha, alpha)

        out = out * (1 - alpha * mask) + alpha * layer
    return out.clamp(0, 1)


try:
    import triton
    import triton.language as tl
except ImportError:  # no triton (CPU-only torch): render() above still works
    triton = None

if triton is not None:
    @triton.jit
    def _render_kernel(out, per, valid, kind, start, end, spans, path, dots, HAS_DOTS: tl.constexpr,
                       X0_: tl.constexpr, Y0_: tl.constexpr, PX_: tl.constexpr, W_: tl.constexpr, H_: tl.constexpr,
                       V_: tl.constexpr, K_: tl.constexpr, FADE_OUT_: tl.constexpr,
                       FILL_: tl.constexpr, NUMBER_: tl.constexpr, RIM_: tl.constexpr, APPROACH_: tl.constexpr,
                       BODY_: tl.constexpr, BORDER_: tl.constexpr, SPIN_: tl.constexpr, RIM_W_: tl.constexpr,
                       APPROACH_W_: tl.constexpr, BLOCK: tl.constexpr):
        b = tl.program_id(0)
        pix = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        inside = pix < W_ * H_
        gx = X0_ + ((pix % W_).to(tl.float32) + 0.5) / PX_
        gy = Y0_ + ((pix // W_).to(tl.float32) + 0.5) / PX_
        t = tl.load(per + b * 4)
        r = tl.load(per + b * 4 + 1)
        pre = tl.load(per + b * 4 + 2)
        fade = tl.load(per + b * 4 + 3)
        rim_w = tl.maximum(tl.floor(RIM_W_ * r * PX_ + 0.5), 1.0) / PX_
        app_w = tl.maximum(tl.floor(APPROACH_W_ * r * PX_ + 0.5), 1.0) / PX_
        ds = tl.sqrt((gx - 256.0) * (gx - 256.0) + (gy - 192.0) * (gy - 192.0))
        srings = tl.minimum(tl.clamp((0.5 / PX_ - tl.abs(ds - 40.0)) * PX_ + 0.5, 0.0, 1.0)
                            + tl.clamp((0.5 / PX_ - tl.abs(ds - 120.0)) * PX_ + 0.5, 0.0, 1.0)
                            + tl.clamp((0.5 / PX_ - tl.abs(ds - 170.0)) * PX_ + 0.5, 0.0, 1.0), 1.0)
        acc = tl.zeros([BLOCK], tl.float32)
        for vi in tl.static_range(V_):
            v = V_ - 1 - vi  # earlier objects end up on top
            s = b * V_ + v
            ok = tl.load(valid + s)
            if ok > 0.5:  # empty slots cost nothing: most frames show 2-3 objects
                kd = tl.load(kind + s)
                st = tl.load(start + s)
                en = tl.load(end + s)
                sp = tl.load(spans + s)
                pb = path + s * K_ * 2
                appear = st - pre
                out_fade = tl.where(t > en, tl.maximum(1.0 - (t - en) / FADE_OUT_, 0.0), 1.0)
                alpha = tl.maximum(tl.minimum((t - appear) / fade, 1.0) * out_fade, 0.0) * ok
                is_slider = kd == 1
                # slider body
                shown = tl.maximum(tl.clamp((t - appear) / (pre / 3.0), 0.0, 1.0) * (K_ - 1), 1.0)
                d2 = tl.full([BLOCK], 1e8, tl.float32)
                if kd == 1:  # only sliders have a body
                    for k in tl.static_range(K_):
                        qx = tl.load(pb + 2 * k)
                        qy = tl.load(pb + 2 * k + 1)
                        dk = (gx - qx) * (gx - qx) + (gy - qy) * (gy - qy)
                        d2 = tl.where(shown >= k, tl.minimum(d2, dk), d2)
                db = tl.sqrt(d2)
                body = tl.clamp((r - db) * PX_ + 0.5, 0.0, 1.0) * is_slider
                inner = tl.clamp((0.85 * r - db) * PX_ + 0.5, 0.0, 1.0) * is_slider
                layer = BORDER_ * (body - inner) + BODY_ * inner
                mask = body
                # head, or the ball while sliding
                prog = tl.clamp((t - st) / tl.maximum(en - st, 1e-3), 0.0, 1.0) * sp
                span = tl.minimum(tl.floor(prog), sp - 1.0)
                f = prog - span
                odd = (span - 2.0 * tl.floor(span / 2.0)) > 0.5
                along = tl.where(odd, 1.0 - f, f) * (K_ - 1)
                lo = tl.minimum(tl.maximum(tl.floor(along), 0.0), K_ - 2.0)
                w = along - lo
                li = lo.to(tl.int32)
                bx = tl.load(pb + 2 * li) * (1.0 - w) + tl.load(pb + 2 * li + 2) * w
                by = tl.load(pb + 2 * li + 1) * (1.0 - w) + tl.load(pb + 2 * li + 3) * w
                hx = tl.load(pb)
                hy = tl.load(pb + 1)
                sliding = is_slider & (t > st)
                cx = tl.where(sliding, bx, hx)
                cy = tl.where(sliding, by, hy)
                dc = tl.sqrt((gx - cx) * (gx - cx) + (gy - cy) * (gy - cy))
                disk = tl.clamp((r - dc) * PX_ + 0.5, 0.0, 1.0)
                num = tl.clamp((r / 4.0 - dc) * PX_ + 0.5, 0.0, 1.0)
                rim = tl.clamp((rim_w / 2.0 - tl.abs(dc - r)) * PX_ + 0.5, 0.0, 1.0)
                layer = layer + (FILL_ - layer) * disk
                layer = layer + (NUMBER_ - layer) * num
                layer = layer + (RIM_ - layer) * rim
                mask = tl.maximum(mask, disk)
                # approach circle
                dh = tl.sqrt((gx - hx) * (gx - hx) + (gy - hy) * (gy - hy))
                a = tl.clamp((t - appear) / tl.minimum(2.0 * fade, pre), 0.0, 1.0) * (t < st)
                cov = tl.clamp((app_w / 2.0 - tl.abs(dh - r * (1.0 + 3.0 * (st - t) / pre))) * PX_ + 0.5, 0.0, 1.0) * (a > 0)
                layer = layer + (APPROACH_ * a - layer) * cov
                mask = tl.maximum(mask, cov * a)
                # spinner
                is_spin = kd == 2
                sp_alpha = tl.clamp((t - (st - 500.0)) / 500.0, 0.0, 1.0) * out_fade * ok
                layer = tl.where(is_spin, SPIN_ * srings, layer)
                mask = tl.where(is_spin, srings, mask)
                alpha = tl.where(is_spin, sp_alpha, alpha)
                acc = acc * (1.0 - alpha * mask) + alpha * layer
        if HAS_DOTS:  # the cursor: a white dot (x, y in osu! pixels, radius in frame pixels) over everything
            qx = tl.load(dots + b * 3)
            qy = tl.load(dots + b * 3 + 1)
            rad = tl.load(dots + b * 3 + 2)
            dd = tl.sqrt((gx - qx) * (gx - qx) + (gy - qy) * (gy - qy)) * PX_
            acc = tl.maximum(acc, tl.clamp(rad - dd + 0.5, 0.0, 1.0))
        tl.store(out + b * W_ * H_ + pix, tl.clamp(acc, 0.0, 1.0), mask=inside)


def draw_dots(img, dots):
    """img (B, H, W), dots (B, 3): x, y in osu! pixels, radius in frame pixels -> img with a white dot at each."""
    d = dots.view(-1, 3, 1, 1)
    gx = X0 + (torch.arange(W, device=img.device) + 0.5) / PX
    gy = (Y0 + (torch.arange(H, device=img.device) + 0.5) / PX).view(H, 1)
    dist = torch.sqrt((gx - d[:, 0]) ** 2 + (gy - d[:, 1]) ** 2) * PX
    return torch.maximum(img, (d[:, 2] - dist + 0.5).clamp(0, 1))


def render_fast(p, dots=None):
    """render() as one fused kernel on CUDA, with draw_dots() folded in (falls back to the torch code elsewhere)."""
    dev = p["per"].device
    if triton is None or dev.type != "cuda":
        img = render(p)
        return img if dots is None else draw_dots(img, dots)
    b = p["per"].shape[0]
    out = torch.empty(b, H, W, device=dev)
    block = 256
    _render_kernel[(b, triton.cdiv(H * W, block))](
        out, p["per"], p["valid"].float(), p["kind"].int(), p["start"], p["end"], p["spans"], p["path"].contiguous(),
        out if dots is None else dots.contiguous(), dots is not None,
        X0, Y0, PX, W, H, V, K, float(FADE_OUT), FILL, NUMBER, RIM, APPROACH, BODY, BORDER, SPIN, RIM_W, APPROACH_W, block)
    return out


if __name__ == "__main__":
    import time

    from render import Renderer

    bm = parse_osu(sys.argv[1])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cpu, scene = Renderer(bm), Scene(bm)
    ts = np.random.default_rng(0).uniform(0, bm.ends.max(), 1024)
    a = time.perf_counter()
    ref = np.stack([cpu.frame(t) for t in ts]).astype(np.float32) / 255
    t_cpu = time.perf_counter() - a
    p = pack([(scene, t) for t in ts], dev)
    slow = render(p).cpu().numpy()
    render_fast(p)
    if dev == "cuda":
        torch.cuda.synchronize()
    a = time.perf_counter()
    for _ in range(20):
        got = render_fast(p)
    if dev == "cuda":
        torch.cuda.synchronize()
    t_gpu = (time.perf_counter() - a) / 20
    a = time.perf_counter()
    for _ in range(5):
        pack([(scene, t) for t in ts], dev)
    t_pack = (time.perf_counter() - a) / 5
    print(f"fused vs torch version: |diff| max {np.abs(got.cpu().numpy() - slow).max():.4f}; pack {t_pack * 1000:.0f} ms")
    diff = np.abs(got.cpu().numpy() - ref)
    print(f"1024 frames: cpu {t_cpu * 1000:.0f} ms, {dev} {t_gpu * 1000:.0f} ms; |diff| mean {diff.mean() * 255:.2f}/255, "
          f"p99 {np.percentile(diff, 99) * 255:.1f}/255, frames with mean diff > 5/255: {(diff.mean((1, 2)) * 255 > 5).sum()}")
