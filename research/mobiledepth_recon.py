#!/usr/bin/env python3
"""Reconstruction on REAL handheld sweeps (mobiledepth small sequences).

Post-fusion boundary reconstruction on top of perband AND blend bases.
No GT: q_ssim (caveat: it rewards sharp stamped pixels), diff mass, WHERE the
matte fired (alpha overlay), and eyetool crops — the eye is the verdict.

Run:  python research/mobiledepth_recon.py
"""
from __future__ import annotations
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from focusstack.align import align_stack  # noqa: E402
from focusstack.io import normalize_exposure  # noqa: E402
from focusstack.fusion import fuse_blend, fuse_perband  # noqa: E402
from focusstack.reconstruct import reconstruct_boundaries, _estimate_matte  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, "data", "mobiledepth")
OUT = os.path.join(HERE, "analyze_out", "mobiledepth")
SEQS = ["Figure3/kitchen", "Figure6/zeromotion", "Figure6/smallmotion", "Figure6/largemotion"]


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"{'sequence':14s} | q_ssim perband->+recon | blend->+recon | diff-mass pb/bl | alpha%")
    for seq in SEQS:
        frames = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(MD, seq, "*.jpg")))]
        frames = [f for f in frames if f is not None]
        frames = normalize_exposure(align_stack(frames, motion="affine"))
        radius = 0.012 * max(frames[0].shape[:2])
        bases = {"perband": fuse_perband(frames, harden=0.5),
                 "blend": fuse_blend(frames, harden=0.5)}
        alpha, owner = _estimate_matte(frames, radius)
        apct = float((alpha > 0.15).mean() * 100)
        name = seq.split("/")[-1]
        line = []
        dm = []
        for k, base in bases.items():
            rec = reconstruct_boundaries(frames, base, radius=radius)
            q0, q1 = M.q_ssim(frames, base), M.q_ssim(frames, rec)
            d = float(np.abs(rec.astype(np.int16) - base.astype(np.int16)).mean())
            dm.append(d)
            line.append(f"{q0:.4f}->{q1:.4f}")
            cv2.imwrite(os.path.join(OUT, f"{name}_{k}_recon.png"), rec)
        # where did the matte fire?
        ov = bases["perband"].copy()
        ov[alpha > 0.15] = (0, 0, 255)
        cv2.imwrite(os.path.join(OUT, f"{name}_alphafire.png"), ov)
        print(f"{name:14s} | {line[0]} | {line[1]} | {dm[0]:.2f}/{dm[1]:.2f} | {apct:.1f}% owner={owner}")
    print(f"\noutputs -> {OUT}")


if __name__ == "__main__":
    main()
