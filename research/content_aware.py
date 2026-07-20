#!/usr/bin/env python3
"""Content-aware focus measure: best-of-both operator routing.

Finding: laplacian is best on textured/clean content; mod_laplacian is best on
smooth low-contrast content (no 2nd-derivative sign cancellation). Neither wins
everywhere. So route per-pixel by LOCAL CONTRAST: blend the two energies with
weight c = 1 - exp(-ref_contrast/tau), where ref_contrast is the per-pixel max
local std across frames (the region's in-focus contrast, consistent across
frames). Low contrast -> mod_laplacian; high contrast -> laplacian.

Validate it matches the better operator on BOTH regimes (Real-MFF + hard scenes).

Run:  python research/content_aware.py [n_realmff]
"""
from __future__ import annotations
import concurrent.futures as cf
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from focusstack.fusion import fuse_blend, guided_filter, multiband_blend  # noqa: E402
from focusstack.io import to_gray_float  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RMFF = os.path.join(HERE, "data", "realmff", "extracted", "RealMFF")


def _energies(gray):
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    kx = np.array([[-1.0, 2.0, -1.0]], np.float32)
    modl = np.abs(cv2.filter2D(gray, cv2.CV_32F, kx)) + np.abs(cv2.filter2D(gray, cv2.CV_32F, kx.T))
    return lap, modl


def _local_std(g, k=7):
    m = cv2.boxFilter(g, cv2.CV_32F, (k, k))
    m2 = cv2.boxFilter(g * g, cv2.CV_32F, (k, k))
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def fuse_content_aware(frames, tau=8.0, smooth_ksize=9, radius=8, eps=1e-3, levels=None,
                       return_debug=False):
    grays = [to_gray_float(f) for f in frames]
    ref = np.maximum.reduce([_local_std(g) for g in grays])
    c = 1.0 - np.exp(-ref / tau)                       # 0=smooth(mod), 1=textured(lap)
    energies = []
    for g in grays:
        lap, modl = _energies(g)
        e = (1.0 - c) * modl + c * lap
        energies.append(cv2.boxFilter(e, cv2.CV_32F, (smooth_ksize, smooth_ksize)))
    winner = np.argmax(np.stack(energies, 0), axis=0)
    W = []
    for k, f in enumerate(frames):
        raw = (winner == k).astype(np.float32)
        W.append(np.clip(guided_filter(to_gray_float(f) / 255.0, raw, radius, eps), 0.0, None))
    W = np.stack(W, 0)
    W = W / (W.sum(0, keepdims=True) + 1e-8)
    fused = multiband_blend(frames, W)
    return (fused, c) if return_debug else fused


def _rmff_eval(t):
    pa, pb, pf = t
    a, b, gt = cv2.imread(pa), cv2.imread(pb), cv2.imread(pf)
    if a is None:
        return None
    return {
        "lap": M.ref_ssim(fuse_blend([a, b], focus_method="laplacian"), gt),
        "mod": M.ref_ssim(fuse_blend([a, b], focus_method="mod_laplacian"), gt),
        "ca": M.ref_ssim(fuse_content_aware([a, b]), gt),
    }


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    # hard scenes
    from hardbench import (scene_gradient_metal, scene_defocus_spread, defocus_disk,
                           add_noise, SCENES)
    print("hard scenes (GT-SSIM):")
    for name, (gen, nf, mr) in SCENES.items():
        base, depth = gen()
        planes = np.linspace(float(depth.min()), float(depth.max()), nf)
        frames = [add_noise(defocus_disk(base, depth, f, mr), 3.0, i) for i, f in enumerate(planes)]
        lap = M.ref_ssim(fuse_blend(frames, focus_method="laplacian"), base)
        mod = M.ref_ssim(fuse_blend(frames, focus_method="mod_laplacian"), base)
        ca = M.ref_ssim(fuse_content_aware(frames), base)
        star = " <-- best-of-both" if ca >= max(lap, mod) - 0.0008 else ""
        print(f"  {name:15s} lap={lap:.4f}  mod={mod:.4f}  content_aware={ca:.4f}{star}")

    ids = sorted(os.path.basename(p)[:3] for p in glob.glob(os.path.join(RMFF, "imageA", "*_A.png")))[:n]
    trip = [(os.path.join(RMFF, "imageA", f"{i}_A.png"),
             os.path.join(RMFF, "imageB", f"{i}_B.png"),
             os.path.join(RMFF, "Fusion", f"{i}_F.png")) for i in ids]
    rows = []
    with cf.ProcessPoolExecutor(max_workers=min(16, os.cpu_count())) as ex:
        for r in ex.map(_rmff_eval, trip):
            if r:
                rows.append(r)
    print(f"\nReal-MFF ({len(rows)} pairs) GT-SSIM:")
    for key, lbl in [("lap", "laplacian"), ("mod", "mod_laplacian"), ("ca", "content_aware")]:
        print(f"  {lbl:14s} {np.mean([r[key] for r in rows]):.4f}")


if __name__ == "__main__":
    main()
