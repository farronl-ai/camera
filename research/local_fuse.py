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


def _load(sid):
    d = os.path.join(MIX, sid)
    frames = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(d, "frame_*.png")))]
    return frames, cv2.imread(os.path.join(d, "gt.png")), cv2.imread(os.path.join(d, "depth.png"), 0)


def worst_region(fused, gt, t=8):
    s = _ssim_map(_gray32(fused), _gray32(gt)); h, w = s.shape
    return min(s[i*(h//t):(i+1)*(h//t), j*(w//t):(j+1)*(w//t)].mean() for i in range(t) for j in range(t))


def main():
    ids = [os.path.basename(os.path.dirname(p)) for p in sorted(glob.glob(os.path.join(MIX, "*", "gt.png")))]
    print(f"{'id':18s} {'global':>8s} {'pyramid':>8s} {'LOCAL':>8s}   {'detail-region SSIM (global/local)':>34s}")
    agg = {"global": [], "pyramid": [], "local": []}
    for sid in ids:
        frames, gt, depth = _load(sid)
        g = fuse_blend(frames, harden=0.5)          # F19 global auto-scale
        p = fuse_pyramid(frames)
        lo = fuse_local(frames, harden=0.5)
        agg["global"].append(M.ref_ssim(g, gt)); agg["pyramid"].append(M.ref_ssim(p, gt)); agg["local"].append(M.ref_ssim(lo, gt))
        # SSIM specifically on the DETAILED (near) regions where global must compromise
        det = depth < 128
        sg = _ssim_map(_gray32(g), _gray32(gt))[det].mean()
        sl = _ssim_map(_gray32(lo), _gray32(gt))[det].mean()
        print(f"{sid:18s} {agg['global'][-1]:8.4f} {agg['pyramid'][-1]:8.4f} {agg['local'][-1]:8.4f}   {sg:16.4f} / {sl:.4f}")
    print(f"\nMEAN   overall     global={np.mean(agg['global']):.4f}  pyramid={np.mean(agg['pyramid']):.4f}  LOCAL={np.mean(agg['local']):.4f}")


if __name__ == "__main__":
    main()
