#!/usr/bin/env python3
"""16b R2/R3 — C3 reconstruction: radius robustness + all-regime gates.

Gates (plan): occ radius sweep degrades gracefully; hires_mixed (blind) bband
improves; layered scenes no-harm; Real-MFF / N-frame / drift GT-SSIM within
-0.001 of baseline; microscopy diff-mass small. Fence eye-check runs separately.

Run:  python research/recon_gates.py
"""
from __future__ import annotations
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reconstruct import (occ_scenes, contamination_band, reconstruct_band,  # noqa: E402
                         estimate_alpha_v3)
from bband import bband  # noqa: E402
import metrics as M  # noqa: E402
from focusstack.fusion import fuse_perband  # noqa: E402
from focusstack.focus import content_aware_energies  # noqa: E402
from focusstack.io import to_gray_float  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def recon_c3(frames, base, max_r):
    """C3 wrapper for N>=2 frames: owner + most-defocused-at-support as far proxy."""
    a3, owner = estimate_alpha_v3(frames, max_r)
    if a3.max() <= 0:
        return base
    if len(frames) == 2:
        far_idx = 1 - owner
    else:
        E = np.stack(content_aware_energies([to_gray_float(f) for f in frames]), 0)
        sup = a3 > 0.15
        far_idx = int(np.argmin([E[k][sup].mean() for k in range(len(frames))]))
        if far_idx == owner:
            return base
    return reconstruct_band([frames[owner], frames[far_idx]], a3,
                            contamination_band(a3, max_r), base, max_r)


def gate_occ_radius():
    print("== occ: radius robustness (bband e2 base -> C3 @ 0.5x/1x/2x) ==")
    for sc in occ_scenes():
        gt_b = ((cv2.dilate((sc["alpha"] > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8))
                 != cv2.erode((sc["alpha"] > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8)))).astype(np.uint8)
        base = fuse_perband(sc["frames"], harden=0.5)
        e0 = bband(base, sc["gt"], gt_b, 2)[0]
        es = [bband(recon_c3(sc["frames"], base, m * sc["max_r"]), sc["gt"], gt_b, 2)[0]
              for m in (0.5, 1.0, 2.0)]
        print(f"  {sc['sid']:16s} {e0:5.1f} -> {es[0]:5.1f} / {es[1]:5.1f} / {es[2]:5.1f}")


def gate_hires_mixed():
    print("== hires_mixed (blind C3): bband e2 / global ==")
    de, dg = [], []
    for d in sorted(glob.glob(os.path.join(HERE, "data", "hires_mixed", "*", "")))[:5]:
        gt = cv2.imread(os.path.join(d, "gt.png"))
        depth = cv2.imread(os.path.join(d, "depth.png"), 0)
        frames = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(d, "frame_*.png")))]
        m = (depth < 128).astype(np.uint8)
        gt_b = (cv2.dilate(m, np.ones((3, 3), np.uint8)) != cv2.erode(m, np.ones((3, 3), np.uint8))).astype(np.uint8)
        base = fuse_perband(frames, harden=0.5)
        rec = recon_c3(frames, base, 0.012 * max(gt.shape[:2]))
        eb, er = bband(base, gt, gt_b, 2)[0], bband(rec, gt, gt_b, 2)[0]
        gb, gr = M.ref_ssim(base, gt), M.ref_ssim(rec, gt)
        de.append(er - eb); dg.append(gr - gb)
        print(f"  {os.path.basename(d.rstrip('/')):16s} e2 {eb:5.1f}->{er:5.1f}  glob {gb:.4f}->{gr:.4f}")
    print(f"  MEAN delta e2={np.mean(de):+.1f}  glob={np.mean(dg):+.4f}")


def gate_layered():
    print("== layered 5-layer (no-harm): bband e2 / global ==")
    for d in sorted(glob.glob(os.path.join(HERE, "data", "layers", "scene_*", "")))[:3]:
        gt = cv2.imread(os.path.join(d, "gt.png"))
        gt_b = (cv2.imread(os.path.join(d, "boundary.png"), 0) > 0).astype(np.uint8)
        frames = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(d, "frame_*.png")))]
        base = fuse_perband(frames, harden=0.5)
        rec = recon_c3(frames, base, 0.012 * max(gt.shape[:2]))
        print(f"  {os.path.basename(d.rstrip('/')):9s} e2 {bband(base, gt, gt_b, 2)[0]:5.1f}->"
              f"{bband(rec, gt, gt_b, 2)[0]:5.1f}  glob {M.ref_ssim(base, gt):.4f}->{M.ref_ssim(rec, gt):.4f}")


def gate_realmff(n=20):
    print("== Real-MFF (global non-regression) ==")
    root = os.path.join(HERE, "data", "realmff", "extracted", "RealMFF")
    ids = sorted(os.path.basename(p)[:3] for p in glob.glob(os.path.join(root, "imageA", "*_A.png")))
    rng = np.random.default_rng(7)
    db = []
    for i in rng.permutation(ids)[:n]:
        a = cv2.imread(os.path.join(root, "imageA", f"{i}_A.png"))
        b = cv2.imread(os.path.join(root, "imageB", f"{i}_B.png"))
        gt = cv2.imread(os.path.join(root, "Fusion", f"{i}_F.png"))
        if a is None or b is None or gt is None:
            continue
        base = fuse_perband([a, b], harden=0.5)
        rec = recon_c3([a, b], base, 0.012 * max(a.shape[:2]))
        db.append(M.ref_ssim(rec, gt) - M.ref_ssim(base, gt))
    print(f"  mean delta GT-SSIM = {np.mean(db):+.5f}  (gate: > -0.001)   n={len(db)}")


def gate_nframe_drift_micro():
    from nframe import scenes, make_stack
    from drift import apply_drift
    from focusstack.io import normalize_exposure
    print("== N-frame + drift + microscopy ==")
    name, gt, depth = scenes()[0]
    for n in (4, 8):
        frames, _ = make_stack(gt, depth, n)
        base = fuse_perband(frames, harden=0.5)
        rec = recon_c3(frames, base, 0.012 * max(gt.shape[:2]))
        print(f"  nframe N={n}: glob {M.ref_ssim(base, gt):.4f}->{M.ref_ssim(rec, gt):.4f}")
    frames, _ = make_stack(gt, depth, 4)
    dr = normalize_exposure(apply_drift(frames))
    base = fuse_perband(dr, harden=0.5)
    rec = recon_c3(dr, base, 0.012 * max(gt.shape[:2]))
    print(f"  drift+norm:  glob {M.ref_ssim(base, gt):.4f}->{M.ref_ssim(rec, gt):.4f}")
    for d in sorted(glob.glob(os.path.join(HERE, "data", "bbbc006", "*", "")))[:2]:
        frames = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(d, "frame_*.png")))]
        base = fuse_perband(frames, harden=0.5)
        rec = recon_c3(frames, base, 0.012 * max(frames[0].shape[:2]))
        dm = float(np.abs(rec.astype(np.int16) - base.astype(np.int16)).mean())
        print(f"  micro {os.path.basename(d.rstrip('/'))[-6:]}: mean|diff|={dm:.3f} (want small)")


if __name__ == "__main__":
    gate_occ_radius()
    gate_hires_mixed()
    gate_layered()
    gate_realmff()
    gate_nframe_drift_micro()
    print("\ngates done")
