#!/usr/bin/env python3
"""Quick method-defaulting suite on REAL handheld focal sweeps (mobiledepth).

The small sequences (N=12-14, 1280x720, real optical defocus, graded handheld
motion, NO GT) — same quick protocol used for prior default decisions on no-GT
data (microscopy, F25): q_ssim ordering + disagreement-guided eye crops.
Full pipeline path: align (affine) -> exposure normalize -> fuse.

Run:  python research/mobiledepth_quick.py
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
from focusstack.align import align_stack  # noqa: E402
from focusstack.io import normalize_exposure  # noqa: E402
from focusstack.fusion import fuse_blend, fuse_perband, fuse_pyramid  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, "data", "mobiledepth")
OUT = os.path.join(HERE, "analyze_out", "mobiledepth")
SEQS = ["Figure3/kitchen", "Figure6/zeromotion", "Figure6/smallmotion", "Figure6/largemotion"]


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"{'sequence':22s} N  align_s | q_ssim: perband  blend  pyramid")
    for seq in SEQS:
        paths = sorted(glob.glob(os.path.join(MD, seq, "*.jpg")))
        frames = [cv2.imread(p) for p in paths]
        frames = [f for f in frames if f is not None]
        t0 = time.time()
        frames = align_stack(frames, motion="affine")
        ta = time.time() - t0
        frames = normalize_exposure(frames)
        outs = {"perband": fuse_perband(frames, harden=0.5),
                "blend": fuse_blend(frames, harden=0.5),
                "pyramid": fuse_pyramid(frames)}
        scores = {k: M.q_ssim(frames, v) for k, v in outs.items()}
        name = seq.split("/")[-1]
        for k, v in outs.items():
            cv2.imwrite(os.path.join(OUT, f"{name}_{k}.png"), v)
        cv2.imwrite(os.path.join(OUT, f"{name}_src0.png"), frames[0])
        print(f"{name:22s} {len(frames):2d} {ta:6.1f}s | {scores['perband']:.4f}  "
              f"{scores['blend']:.4f}  {scores['pyramid']:.4f}")
    print(f"\noutputs -> {OUT}")


if __name__ == "__main__":
    main()
