#!/usr/bin/env python3
"""E4 — boundary-aware perband: ablation A/B on layered GT scenes.

Variants: baseline | guide-only (b_eps_gain=0) | eps-only (b_lambda=0) | both.
B = fused boundary map (stack ∪ semantic-from-pass1, mean). Reported: boundary-band
err/SSIM at k=2,5 (the phase objective) + global GT-SSIM (non-regression guard).

Run:  python research/e4_integrate.py
"""
from __future__ import annotations
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from boundary import stack_boundary  # noqa: E402
from boundary2 import scenes, load_frames, semantic_boundary  # noqa: E402
from bband import bband  # noqa: E402
import metrics as M  # noqa: E402
from focusstack.fusion import fuse_perband  # noqa: E402


def main():
    variants = {
        "baseline": dict(),
        "guide": dict(b_lambda=0.5, b_eps_gain=0.0),
        "eps": dict(b_lambda=0.0, b_eps_gain=4.0),
        "both": dict(b_lambda=0.5, b_eps_gain=4.0),
    }
    agg = {v: {"e2": [], "s2": [], "e5": [], "s5": [], "g": []} for v in variants}
    print(f"  {'scene':9s} " + "  ".join(f"{v}: e2/s2/glob".rjust(22) for v in variants))
    for d in scenes():
        frames = load_frames(d)
        gt = cv2.imread(os.path.join(d, "gt.png"))
        gt_b = (cv2.imread(os.path.join(d, "boundary.png"), 0) > 0).astype(np.uint8)
        npy = os.path.join(d, "pass1.png.depth.npy")
        B = 0.5 * stack_boundary(frames) + 0.5 * semantic_boundary(np.load(npy))
        line = []
        for v, kw in variants.items():
            fused = fuse_perband(frames, harden=0.5,
                                 boundary=None if v == "baseline" else B, **kw)
            e2, s2 = bband(fused, gt, gt_b, 2)
            e5, s5 = bband(fused, gt, gt_b, 5)
            g = M.ref_ssim(fused, gt)
            for key, val in [("e2", e2), ("s2", s2), ("e5", e5), ("s5", s5), ("g", g)]:
                agg[v][key].append(val)
            line.append(f"{e2:5.1f}/{s2:.3f}/{g:.3f}")
        print(f"  {os.path.basename(d.rstrip('/')):9s} " + "  ".join(x.rjust(22) for x in line))
    print("\n  MEAN (bband k=2 err | k=2 ssim | k=5 err | k=5 ssim | global):")
    for v in variants:
        a = agg[v]
        print(f"    {v:9s} {np.mean(a['e2']):6.1f} | {np.mean(a['s2']):.4f} | "
              f"{np.mean(a['e5']):6.1f} | {np.mean(a['s5']):.4f} | {np.mean(a['g']):.4f}")


if __name__ == "__main__":
    main()
