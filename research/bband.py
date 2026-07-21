#!/usr/bin/env python3
"""E0 — boundary-band error: THE metric of the E-phase.

Global SSIM cannot see boundary-only gains (aggregates hide local defects), so the
phase optimizes error measured WITHIN ±k px of TRUE object boundaries:
  - bband mean|err|: mean per-pixel |fused-GT| (summed over BGR) inside the band;
  - bband SSIM: mean SSIM-map value inside the band.
Boundary GT comes free from our generators (depth/alpha discontinuities).

Baselines (perband/blend/pyramid) over:
  A. hires_mixed  — fine structures at depth (boundary = near-mask edges), 10 stacks
  B. occ          — alpha-matte scenes (boundary = alpha 0.5 crossings), 4 scenes
Run:  python research/bband.py            writes research/bband_baseline.json
"""
from __future__ import annotations
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import _ssim_map, _gray32  # noqa: E402
from focusstack.fusion import fuse_blend, fuse_perband, fuse_pyramid  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
KS = (2, 5, 10)


def boundary_from_mask(mask: np.ndarray) -> np.ndarray:
    """1-px-ish boundary of a binary mask (morphological gradient)."""
    m = (mask > 0).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    return (cv2.dilate(m, k) != cv2.erode(m, k)).astype(np.uint8)


def bband(fused: np.ndarray, gt: np.ndarray, bmap: np.ndarray, k: int):
    """(mean |err| summed-BGR, mean SSIM) within ±k px of boundaries."""
    band = cv2.dilate(bmap, np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0
    err = np.abs(fused.astype(np.float32) - gt.astype(np.float32)).sum(axis=2)
    s = _ssim_map(_gray32(fused), _gray32(gt))
    return float(err[band].mean()), float(s[band].mean())


METHODS = {
    "perband": lambda fr: fuse_perband(fr, harden=0.5),
    "blend": lambda fr: fuse_blend(fr, harden=0.5),
    "pyramid": lambda fr: fuse_pyramid(fr),
}


def bench_hires_mixed():
    out = []
    for d in sorted(glob.glob(os.path.join(HERE, "data", "hires_mixed", "*", ""))):
        gt = cv2.imread(os.path.join(d, "gt.png"))
        depth = cv2.imread(os.path.join(d, "depth.png"), 0)
        frames = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(d, "frame_*.png")))]
        if gt is None or depth is None or not frames:
            continue
        out.append((os.path.basename(d.rstrip("/")), frames, gt, boundary_from_mask(depth < 128)))
    return out


def bench_occ():
    from occ_gen import near_layer, occ_defocus, LONG
    from hires_gen import add_noise
    out = []
    for gp in sorted(glob.glob(os.path.join(HERE, "data", "hires", "*", "gt.png")))[:4]:
        far = cv2.imread(gp)
        h, w = far.shape[:2]
        s = LONG / max(h, w)
        far = cv2.resize(far, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        near, alpha = near_layer(far, seed=hash(gp) % 1000)
        gt = (near * alpha[..., None] + far.astype(np.float32) * (1 - alpha[..., None])).astype(np.uint8)
        max_r = 0.012 * max(gt.shape[:2])
        frames = [add_noise(occ_defocus(far, near, alpha, f, 0.15, 0.85, max_r), 3.0, i)
                  for i, f in enumerate([0.15, 0.85])]
        out.append((os.path.basename(os.path.dirname(gp)) + "_occ", frames, gt,
                    boundary_from_mask(alpha > 0.5)))
    return out


def main():
    results = {}
    for bench_name, stacks in [("hires_mixed", bench_hires_mixed()), ("occ", bench_occ())]:
        agg = {m: {k: [[], []] for k in KS} for m in METHODS}
        for sid, frames, gt, bmap in stacks:
            for m, fn in METHODS.items():
                fused = fn(frames)
                for k in KS:
                    e, s = bband(fused, gt, bmap, k)
                    agg[m][k][0].append(e)
                    agg[m][k][1].append(s)
        results[bench_name] = {m: {str(k): {"err": float(np.mean(agg[m][k][0])),
                                            "ssim": float(np.mean(agg[m][k][1]))}
                                   for k in KS} for m in METHODS}
        print(f"\n{bench_name} ({len(stacks)} stacks) — boundary-band mean|err| / SSIM:")
        print(f"  {'method':9s} " + "  ".join(f"k={k}: err/ssim".rjust(18) for k in KS))
        for m in METHODS:
            row = "  ".join(f"{agg[m][k][0] and np.mean(agg[m][k][0]):7.1f}/{np.mean(agg[m][k][1]):.4f}".rjust(18) for k in KS)
            print(f"  {m:9s} {row}")
    json.dump(results, open(os.path.join(HERE, "bband_baseline.json"), "w"), indent=2)
    print(f"\nwrote bband_baseline.json  (the numbers the E-phase must beat)")


if __name__ == "__main__":
    main()
