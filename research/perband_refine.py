#!/usr/bin/env python3
"""A/B the refined fuse_perband (weighted base + per-band window caps) vs the
original (v1: mean base, uncapped windows), plus blend/pyramid, on all regimes:
  - Lytro fence (real optical defocus, halo case; composite + pixel diff)
  - Real-MFF low-res (GT-SSIM, n pairs)
  - high-res fine-depth benchmark (GT-SSIM overall + near-structures)

Run:  python research/perband_refine.py [n_realmff]
"""
from __future__ import annotations
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from metrics import _ssim_map, _gray32  # noqa: E402
from focusstack.fusion import (fuse_blend, fuse_perband, fuse_pyramid,  # noqa: E402
                               _auto_levels, _laplacian_pyramid, _gaussian_pyramid, guided_filter)
from focusstack.io import to_gray_float  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RMFF = os.path.join(HERE, "data", "realmff", "extracted", "RealMFF")
MIX = os.path.join(HERE, "data", "hires_mixed")


def perband_v1(images, radius=6, eps=1e-3, energy_ksize=7, harden=0.0):
    """Original fuse_perband: mean base, uncapped per-band windows."""
    floats = [img.astype(np.float32) for img in images]
    n = len(floats)
    levels = _auto_levels(floats[0].shape, None)
    lps = [_laplacian_pyramid(im, levels) for im in floats]
    gps = [_gaussian_pyramid(to_gray_float(f), levels) for f in images]
    bands = []
    for b in range(levels + 1):
        coeffs = [lps[k][b] for k in range(n)]
        if b < levels:
            E = np.stack([cv2.boxFilter((coeffs[k] ** 2).sum(2), cv2.CV_32F,
                                        (energy_ksize, energy_ksize)) for k in range(n)], 0)
            winner = np.argmax(E, 0)
            conf = None
            if harden > 0:
                srt = np.sort(E, 0)
                conf = np.clip((srt[-1] - srt[-2]) / (srt[-1] + 1e-6), 0, 1)
            W = []
            for k in range(n):
                raw = (winner == k).astype(np.float32)
                wg = np.clip(guided_filter(gps[k][b] / 255.0, raw, radius, eps), 0, None)
                if conf is not None:
                    wg = (1 - conf) * wg + conf * raw
                W.append(wg)
            W = np.stack(W, 0)
            W /= (W.sum(0, keepdims=True) + 1e-8)
            bands.append(sum(W[k][..., None] * coeffs[k] for k in range(n)))
        else:
            bands.append(np.mean(np.stack(coeffs, 0), 0))
    result = bands[-1]
    for b in range(levels - 1, -1, -1):
        size = (bands[b].shape[1], bands[b].shape[0])
        result = cv2.pyrUp(result, dstsize=size) + bands[b]
    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    # --- fence ---
    a = cv2.imread(os.path.join(HERE, "data", "standard", "c_05_1.tif"))
    b = cv2.imread(os.path.join(HERE, "data", "standard", "c_05_2.tif"))
    v1 = perband_v1([a, b], harden=0.5)
    v2 = fuse_perband([a, b], harden=0.5)
    bl = fuse_blend([a, b], harden=0.5)
    diff = float(np.abs(v1.astype(np.int16) - v2.astype(np.int16)).mean())
    print("FENCE (real optical, no GT):")
    print(f"  v1 vs v2 mean|diff| = {diff:.3f}  (0 would mean refinement is a no-op!)")
    print(f"  composite: blend={M.composite([a,b],bl):.4f}  v1={M.composite([a,b],v1):.4f}  v2={M.composite([a,b],v2):.4f}")

    # --- Real-MFF low-res, GT ---
    ids = sorted(os.path.basename(p)[:3] for p in glob.glob(os.path.join(RMFF, "imageA", "*_A.png")))[:n]
    s = {"blend": [], "pyramid": [], "v1": [], "v2": []}
    for i in ids:
        A = cv2.imread(os.path.join(RMFF, "imageA", f"{i}_A.png"))
        B = cv2.imread(os.path.join(RMFF, "imageB", f"{i}_B.png"))
        G = cv2.imread(os.path.join(RMFF, "Fusion", f"{i}_F.png"))
        s["blend"].append(M.ref_ssim(fuse_blend([A, B], harden=0.5), G))
        s["pyramid"].append(M.ref_ssim(fuse_pyramid([A, B]), G))
        s["v1"].append(M.ref_ssim(perband_v1([A, B], harden=0.5), G))
        s["v2"].append(M.ref_ssim(fuse_perband([A, B], harden=0.5), G))
    print(f"\nREAL-MFF low-res GT-SSIM ({len(ids)} pairs):")
    for k in s:
        print(f"  {k:8s} {np.mean(s[k]):.4f}")

    # --- high-res fine-depth, GT ---
    hids = [os.path.basename(os.path.dirname(p)) for p in sorted(glob.glob(os.path.join(MIX, "*", "gt.png")))]
    h = {"blend": [], "pyramid": [], "v1": [], "v2": []}
    hn = {"pyramid": [], "v1": [], "v2": []}
    for sid in hids:
        d = os.path.join(MIX, sid)
        frames = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(d, "frame_*.png")))]
        gt = cv2.imread(os.path.join(d, "gt.png"))
        depth = cv2.imread(os.path.join(d, "depth.png"), 0)
        det = depth < 128
        outs = {"blend": fuse_blend(frames, harden=0.5), "pyramid": fuse_pyramid(frames),
                "v1": perband_v1(frames, harden=0.5), "v2": fuse_perband(frames, harden=0.5)}
        for k, im in outs.items():
            h[k].append(M.ref_ssim(im, gt))
            if k in hn:
                hn[k].append(float(_ssim_map(_gray32(im), _gray32(gt))[det].mean()))
    print(f"\nHIGH-RES fine-depth GT-SSIM ({len(hids)} stacks)  overall | near-structures:")
    for k in h:
        near = f" | {np.mean(hn[k]):.4f}" if k in hn else ""
        print(f"  {k:8s} {np.mean(h[k]):.4f}{near}")


if __name__ == "__main__":
    main()
