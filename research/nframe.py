#!/usr/bin/env python3
"""B1 — N-frame stress test: does the engine degrade as stacks get deep?

Nearly all validation used N=2. Real stacking uses 5–50 frames. Two specific
mechanisms are hypothesized to degrade with N:

  H1 conf-collapse: harden's confidence = (top1−top2)/top1 focus energy. With many
     focus planes, ADJACENT planes have near-equal energy at most pixels → conf → 0
     → hardening silently disables exactly when stacks get deep.
  H2 weight dilution: soft weights normalize over N frames → mass leaks onto the
     N−1 blurred frames → contamination grows with N.

We measure both DIRECTLY (not just end quality): mean conf and P(conf>0.25) vs N;
mean weight mass on the true-sharpest frame (known from GT depth) vs N; plus
GT-SSIM vs N per method. Scenes: continuous-gradient depth (rail-like) on real
photos, with thin near structures overlaid on one (structure dimension).

Run:  python research/nframe.py
"""
from __future__ import annotations
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from hires_gen import defocus_ca, add_noise  # noqa: E402
from mixed_gen import overlay_fine_near  # noqa: E402
from focusstack.fusion import fuse_blend, fuse_perband, fuse_pyramid, _guided_weights  # noqa: E402
from focusstack.focus import content_aware_energies  # noqa: E402
from focusstack.io import to_gray_float  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "hires")
NS = (2, 4, 8)
LONG_SIDE = 1536


def scenes():
    """Two real-photo scenes: pure gradient depth; gradient + thin near structures."""
    gts = sorted(glob.glob(os.path.join(SRC, "*", "gt.png")))
    out = []
    for i, gp in enumerate(gts[:2]):
        base = cv2.imread(gp)
        h, w = base.shape[:2]
        s = LONG_SIDE / max(h, w)
        base = cv2.resize(base, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        h, w = base.shape[:2]
        if i == 0:
            gt = base
            depth = np.tile(np.linspace(0.05, 0.95, w), (h, 1)).astype(np.float32)
            name = "gradient"
        else:
            gt, near = overlay_fine_near(base, seed=7)
            depth = np.tile(np.linspace(0.05, 0.95, w), (h, 1)).astype(np.float32)
            depth[near > 0] = 0.1
            name = "gradient+wires"
        out.append((name, gt, depth))
    return out


def make_stack(gt, depth, n):
    planes = np.linspace(0.05, 0.95, n)
    max_r = 0.012 * max(gt.shape[:2])
    frames = [add_noise(defocus_ca(gt, depth, float(p), max_r), 3.0, i)
              for i, p in enumerate(planes)]
    return frames, planes


def probes(frames, planes, depth):
    """H1: conf stats. H2: weight mass on the true-sharpest frame."""
    grays = [to_gray_float(f) for f in frames]
    energy = np.stack(content_aware_energies(grays), 0)
    srt = np.sort(energy, 0)
    conf = (srt[-1] - srt[-2]) / (srt[-1] + 1e-6)
    conf = cv2.boxFilter(conf.astype(np.float32), cv2.CV_32F, (15, 15))
    w = _guided_weights(frames, "content_aware", None, 1e-3, None, harden=0.5)
    true_idx = np.argmin(np.abs(planes[:, None, None] - depth[None]), axis=0)  # (H,W)
    hh, ww = true_idx.shape
    yy, xx = np.indices((hh, ww))
    mass = float(w[true_idx, yy, xx].mean())
    agree = float((np.argmax(w, 0) == true_idx).mean())
    return float(conf.mean()), float((conf > 0.25).mean()), mass, agree


def main():
    header = (f"{'scene':15s} {'N':>2s} | {'conf_mean':>9s} {'P(c>.25)':>8s} "
              f"{'true-mass':>9s} {'argmax-ok':>9s} | {'perband':>8s} {'pb(h=0)':>8s} "
              f"{'blend':>8s} {'pyramid':>8s}")
    print(header)
    for name, gt, depth in scenes():
        for n in NS:
            frames, planes = make_stack(gt, depth, n)
            cm, cp, mass, agree = probes(frames, planes, depth)
            r = {
                "perband": M.ref_ssim(fuse_perband(frames, harden=0.5), gt),
                "pb0": M.ref_ssim(fuse_perband(frames, harden=0.0), gt),
                "blend": M.ref_ssim(fuse_blend(frames, harden=0.5), gt),
                "pyramid": M.ref_ssim(fuse_pyramid(frames), gt),
            }
            print(f"{name:15s} {n:2d} | {cm:9.3f} {cp:8.3f} {mass:9.3f} {agree:9.3f} | "
                  f"{r['perband']:8.4f} {r['pb0']:8.4f} {r['blend']:8.4f} {r['pyramid']:8.4f}")


if __name__ == "__main__":
    main()
