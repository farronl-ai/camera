#!/usr/bin/env python3
"""L3 — locally-parameterized fusion: guided scale varies PER PIXEL by measured
structure, not a global number.

Compute guided weights at a few discrete scales, then per pixel interpolate
between them by the content-driven local window map S(x,y) (analyze.local_window_map:
small on fine detail, large on smooth). Small-window weights preserve fine detail;
large-window weights are robust on smooth large-CoC regions — in the SAME image.
Then multiband-blend. This is the answer to "no magic global number" — the scale
is measured locally and edge-aware.

Run:  python research/local_fuse.py            # eval vs global-scale + pyramid
"""
from __future__ import annotations
import glob
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from metrics import _ssim_map, _gray32  # noqa: E402
from analyze import local_window_map  # noqa: E402
from focusstack.fusion import _guided_weights, multiband_blend, fuse_blend, fuse_pyramid  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MIX = os.path.join(HERE, "data", "hires_mixed")


def _odd(x):
    x = max(3, int(round(x)))
    return x + 1 - (x % 2)


def fuse_local(frames, focus_method="content_aware", harden=0.5, w_min=6.0, eps=1e-3):
    """Per-pixel scale-adaptive guided multi-band fusion."""
    win = local_window_map(frames, w_min=w_min)                 # (H,W) px, measured
    w_max = float(win.max())
    scales = sorted({int(round(s)) for s in (w_min, w_min * 2, w_min * 4, w_max)})
    if len(scales) < 2:
        scales = [int(w_min), int(w_min) + 2]
    scales = np.array(scales, np.float32)

    # weights at each discrete scale
    Wk = np.stack([_guided_weights(frames, focus_method, int(s), eps, _odd(s), harden)
                   for s in scales], axis=0)                    # (K, N, H, W)
    K, N, H, W = Wk.shape

    # per-pixel interpolate between the two bracketing scales by `win`
    wc = np.clip(win, scales[0], scales[-1])
    hi = np.clip(np.searchsorted(scales, wc), 1, K - 1)         # (H,W)
    lo = hi - 1
    alpha = (wc - scales[lo]) / (scales[hi] - scales[lo] + 1e-9)  # (H,W)

    W_local = np.empty((N, H, W), np.float32)
    for n in range(N):
        Wn = Wk[:, n, :, :]                                     # (K,H,W)
        lo_v = np.take_along_axis(Wn, lo[None], 0)[0]
        hi_v = np.take_along_axis(Wn, hi[None], 0)[0]
        W_local[n] = (1 - alpha) * lo_v + alpha * hi_v
    W_local /= (W_local.sum(0, keepdims=True) + 1e-8)
    return multiband_blend(frames, W_local)


def fuse_perband(frames, radius=6, eps=1e-3, harden=0.0, energy_ksize=7):
    """Per-band EDGE-AWARE weights: pyramid's multi-scale DECISION + guided (halo-free).

    Unlike fuse_blend (one single-scale guided weight broadcast to all bands), this
    computes the focus decision AND an edge-aware guided weight at EACH Laplacian
    band, using that band's own Gaussian image as the guide. A FIXED small radius
    per band => the effective full-res radius grows with scale automatically (the
    pyramid "starts at the finest pixels and moves up") — multi-scale, no magic number.
    """
    from focusstack.fusion import _auto_levels, _laplacian_pyramid, _gaussian_pyramid, guided_filter
    from focusstack.focus import to_gray_float as _t  # noqa
    floats = [f.astype(np.float32) for f in frames]
    N = len(floats)
    levels = _auto_levels(floats[0].shape, None)
    lps = [_laplacian_pyramid(im, levels) for im in floats]
    gps = [_gaussian_pyramid(_gray32(f), levels) for f in frames]   # per-band luminance guide
    fused_bands = []
    for b in range(levels + 1):
        coeffs = [lps[k][b] for k in range(N)]
        if b < levels:
            E = np.stack([cv2.boxFilter((coeffs[k] ** 2).sum(2), cv2.CV_32F, (energy_ksize, energy_ksize))
                          for k in range(N)], 0)
            winner = np.argmax(E, 0)
            conf = None
            if harden > 0:
                srt = np.sort(E, 0)
                conf = np.clip((srt[-1] - srt[-2]) / (srt[-1] + 1e-6), 0, 1)
            W = []
            for k in range(N):
                raw = (winner == k).astype(np.float32)
                wg = np.clip(guided_filter(gps[k][b] / 255.0, raw, radius, eps), 0.0, None)
                if conf is not None:
                    wg = (1 - conf) * wg + conf * raw
                W.append(wg)
            W = np.stack(W, 0)
            W /= (W.sum(0, keepdims=True) + 1e-8)
            fused_bands.append(sum(W[k][..., None] * coeffs[k] for k in range(N)))
        else:
            fused_bands.append(np.mean(np.stack(coeffs, 0), 0))     # base
    result = fused_bands[-1]
    for b in range(levels - 1, -1, -1):
        size = (fused_bands[b].shape[1], fused_bands[b].shape[0])
        result = cv2.pyrUp(result, dstsize=size) + fused_bands[b]
    return np.clip(result, 0, 255).astype(np.uint8)


def _route_map(frames):
    """Per-pixel routing weight toward LOCAL (vs pyramid), from frames only (no GT).

    Pyramid halos exactly at FINE-SCALE focus (depth) boundaries — thin near
    structures where the sharpest-frame decision flips on a fine scale. So route to
    local-guided there, and to pyramid elsewhere (detailed backgrounds / smooth,
    where pyramid's multi-scale detail is strongest). Signal = local density of
    focus-winner transitions, content-normalized (no magic threshold).
    """
    from focusstack.focus import focus_measure
    fm = np.stack([focus_measure(_gray32(f)) for f in frames], 0)
    winner = np.argmax(fm, 0).astype(np.float32)
    gb = cv2.magnitude(cv2.Sobel(winner, cv2.CV_32F, 1, 0), cv2.Sobel(winner, cv2.CV_32F, 0, 1))
    dens = cv2.boxFilter((gb > 0).astype(np.float32), cv2.CV_32F, (21, 21))
    r = np.clip(dens / (np.percentile(dens, 97) + 1e-6), 0, 1)
    return cv2.GaussianBlur(r, (0, 0), 9.0)


def fuse_routed(frames, harden=0.5):
    """Best-of-both: local-guided at fine depth boundaries, pyramid elsewhere."""
    lo = fuse_local(frames, harden=harden).astype(np.float32)
    py = fuse_pyramid(frames).astype(np.float32)
    r = _route_map(frames)[..., None]
    return np.clip(r * lo + (1 - r) * py, 0, 255).astype(np.uint8)


def _load(sid):
    d = os.path.join(MIX, sid)
    frames = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(d, "frame_*.png")))]
    return frames, cv2.imread(os.path.join(d, "gt.png")), cv2.imread(os.path.join(d, "depth.png"), 0)


def worst_region(fused, gt, t=8):
    s = _ssim_map(_gray32(fused), _gray32(gt)); h, w = s.shape
    return min(s[i*(h//t):(i+1)*(h//t), j*(w//t):(j+1)*(w//t)].mean() for i in range(t) for j in range(t))


def main():
    ids = [os.path.basename(os.path.dirname(p)) for p in sorted(glob.glob(os.path.join(MIX, "*", "gt.png")))]
    print(f"{'id':18s} {'global':>8s} {'pyramid':>8s} {'LOCAL':>8s} {'ROUTED':>8s}   near-SSIM g/p/routed")
    agg = {"global": [], "pyramid": [], "local": [], "routed": []}
    for sid in ids:
        frames, gt, depth = _load(sid)
        g = fuse_blend(frames, harden=0.5)          # F19 global auto-scale
        p = fuse_pyramid(frames)
        lo = fuse_local(frames, harden=0.5)
        ro = fuse_routed(frames, harden=0.5)
        for k, im in [("global", g), ("pyramid", p), ("local", lo), ("routed", ro)]:
            agg[k].append(M.ref_ssim(im, gt))
        det = depth < 128                           # fine near structures
        sg = _ssim_map(_gray32(g), _gray32(gt))[det].mean()
        sp = _ssim_map(_gray32(p), _gray32(gt))[det].mean()
        sr = _ssim_map(_gray32(ro), _gray32(gt))[det].mean()
        print(f"{sid:18s} {agg['global'][-1]:8.4f} {agg['pyramid'][-1]:8.4f} {agg['local'][-1]:8.4f} {agg['routed'][-1]:8.4f}   {sg:.3f}/{sp:.3f}/{sr:.3f}")
    print(f"\nMEAN overall   global={np.mean(agg['global']):.4f}  pyramid={np.mean(agg['pyramid']):.4f}  "
          f"LOCAL={np.mean(agg['local']):.4f}  ROUTED={np.mean(agg['routed']):.4f}")


if __name__ == "__main__":
    main()
