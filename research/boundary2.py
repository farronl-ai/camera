#!/usr/bin/env python3
"""E3 — multi-channel boundary evaluation: stack vs SEMANTIC vs fused.

Stage 1 (main env):   python research/boundary2.py prep
    -> writes composite.png (max-focus, the depth net's input) per layered scene.
Stage 2 (.venv312):   .venv312/bin/python research/bridge_depth.py <composites...>
    -> writes composite.png.depth.npy per scene.
Stage 3 (main env):   python research/boundary2.py eval
    -> P/R/F per channel + fused (max / mean), camo recall, and the
       focus-depth <-> DA-V2 cross-calibration (Spearman on textured pixels).
"""
from __future__ import annotations
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from boundary import stack_boundary, canny_reference, best_f, prf, camo_mask, _robust_norm, _grad_mag  # noqa: E402
from focusstack.fusion import depth_from_focus  # noqa: E402
from focusstack.focus import content_aware_energies  # noqa: E402
from focusstack.io import to_gray_float  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LAYERS = os.path.join(HERE, "data", "layers")


def scenes():
    return sorted(glob.glob(os.path.join(LAYERS, "scene_*", "")))


def load_frames(d):
    return [cv2.imread(p) for p in sorted(glob.glob(os.path.join(d, "frame_*.png")))]


def max_focus_composite(frames):
    E = np.stack(content_aware_energies([to_gray_float(f) for f in frames]), 0)
    w = np.argmax(E, 0)
    yy, xx = np.indices(w.shape)
    return np.stack(frames, 0)[w, yy, xx]


def semantic_boundary(depth_npy):
    """Boundary strength from DA-V2 depth discontinuities (robust-normalized)."""
    d = depth_npy.astype(np.float32)
    d = (d - d.min()) / (np.ptp(d) + 1e-9)
    return _robust_norm(_grad_mag(d), pct=99.5)


def cmd_prep():
    for d in scenes():
        comp = max_focus_composite(load_frames(d))
        cv2.imwrite(os.path.join(d, "composite.png"), comp)
        print(f"  {os.path.basename(d.rstrip('/'))}: composite written")


def cmd_eval(tol=3, src="composite"):
    rows = []
    print(f"tol ±{tol}px, depth input={src} — best-F per channel  (camo recall @ own best)")
    print(f"  {'scene':9s} {'stack':>6s} {'sem':>6s} {'fmax':>6s} {'fmean':>6s} {'canny':>6s} | camo s/sem/f")
    cal = []
    for d in scenes():
        frames = load_frames(d)
        gt_b = (cv2.imread(os.path.join(d, "boundary.png"), 0) > 0).astype(np.uint8)
        npy = os.path.join(d, f"{src}.png.depth.npy")
        if not os.path.exists(npy):
            print(f"  {os.path.basename(d.rstrip('/'))}: no depth npy — run stage 2")
            continue
        sem_d = np.load(npy)
        B_stack = stack_boundary(frames)
        B_sem = semantic_boundary(sem_d)
        B_fmax = np.maximum(B_stack, B_sem)
        B_fmean = 0.5 * B_stack + 0.5 * B_sem
        can = canny_reference(frames)

        res = {}
        th = {}
        for name, B in [("stack", B_stack), ("sem", B_sem), ("fmax", B_fmax), ("fmean", B_fmean)]:
            p, r, f, t = best_f(B, gt_b, tol)
            res[name] = f
            th[name] = t
        pc, rc, fc = prf(can > 0, gt_b, tol)

        cm = camo_mask(d, gt_b)
        k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
        crec = lambda B, t: float(cm[cv2.dilate((B >= t).astype(np.uint8), k) > 0].sum() / (cm.sum() + 1e-9))
        camo = (crec(B_stack, th["stack"]), crec(B_sem, th["sem"]), crec(B_fmax, th["fmax"]))

        # cross-calibration: Spearman(DA-V2 depth, focus-depth) on textured pixels
        fd = depth_from_focus(frames)
        g = to_gray_float(cv2.imread(os.path.join(d, "gt.png")))
        grad = cv2.boxFilter(_grad_mag(g), cv2.CV_32F, (15, 15))
        tex = grad > np.percentile(grad, 75)
        a, b = sem_d[tex].ravel(), fd[tex].ravel()
        def rank(v):
            o = v.argsort(); r_ = np.empty_like(o, float); r_[o] = np.arange(len(v)); return r_
        ra, rb = rank(a), rank(b)
        ra -= ra.mean(); rb -= rb.mean()
        rho = float((ra * rb).sum() / (np.sqrt((ra**2).sum() * (rb**2).sum()) + 1e-9))
        cal.append(rho)

        sid = os.path.basename(d.rstrip("/"))
        rows.append((res["stack"], res["sem"], res["fmax"], res["fmean"], fc))
        print(f"  {sid:9s} {res['stack']:6.2f} {res['sem']:6.2f} {res['fmax']:6.2f} {res['fmean']:6.2f} {fc:6.2f} | "
              f"{camo[0]:.2f}/{camo[1]:.2f}/{camo[2]:.2f}")
    a = np.array(rows)
    print(f"\n  MEAN     {a[:,0].mean():6.3f} {a[:,1].mean():6.3f} {a[:,2].mean():6.3f} {a[:,3].mean():6.3f} {a[:,4].mean():6.3f}")
    print(f"  depth cross-calibration Spearman (DA-V2 vs focus-depth, textured px): "
          f"mean={np.mean(cal):+.3f}  (negative expected: inverted conventions)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "prep":
        cmd_prep()
    else:
        cmd_eval(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
