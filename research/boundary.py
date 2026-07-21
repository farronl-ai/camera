#!/usr/bin/env python3
"""E2 — Boundary Engine v1: stack-evidence channel (no new dependencies).

Three signals no single-frame appearance operator has, fused into a soft
boundary map B(x,y) ∈ [0,1] + near-side tags:

  edges   — per-frame multi-scale gradient magnitude, MAX over frames: an object
            contour is sharpest in its own focal plane, so max-over-frames is
            defocus-robust (a contour blurred in one frame is crisp in another);
  winner  — discontinuities of the focus-argmax label map: where the sharpest
            frame CHANGES, depth changed;
  depth   — gradient of the (guided-smoothed) depth-from-focus map.

Evaluation on E1 layered scenes (true boundary GT): tolerance-t precision/recall/F
with threshold sweep (best-F), vs a Canny-on-sharp-composite reference — the
"parallel vector" baseline. Camouflage-object recall reported SEPARATELY (the
orthogonality probe: appearance sees ~nothing there).

Run:  python research/boundary.py [tolerance_px]
"""
from __future__ import annotations
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from focusstack.fusion import depth_from_focus  # noqa: E402
from focusstack.focus import content_aware_energies  # noqa: E402
from focusstack.io import to_gray_float  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LAYERS = os.path.join(HERE, "data", "layers")


def _robust_norm(x, pct=99.0):
    return np.clip(x / (np.percentile(x, pct) + 1e-6), 0, 1)


def _grad_mag(gray):
    return cv2.magnitude(cv2.Scharr(gray, cv2.CV_32F, 1, 0), cv2.Scharr(gray, cv2.CV_32F, 0, 1))


def stack_boundary(frames, return_parts=False):
    """Soft boundary map from stack evidence. Returns B (and parts, near-side)."""
    grays = [to_gray_float(f) for f in frames]

    # 1) defocus-robust multi-scale edges: max over frames and scales
    edges = None
    for sigma in (0.0, 2.0, 4.0):
        per = []
        for g in grays:
            gs = g if sigma == 0 else cv2.GaussianBlur(g, (0, 0), sigma)
            per.append(_grad_mag(gs))
        e = _robust_norm(np.maximum.reduce(per))
        edges = e if edges is None else np.maximum(edges, e)

    # 2) winner-label discontinuity density
    E = np.stack(content_aware_energies(grays), 0)
    winner = np.argmax(E, 0).astype(np.float32)
    wd = (_grad_mag(winner) > 0).astype(np.float32)
    winner_b = _robust_norm(cv2.boxFilter(wd, cv2.CV_32F, (9, 9)))

    # 3) focus-depth discontinuity
    depth = depth_from_focus(frames)
    depth_b = _robust_norm(_grad_mag(depth))

    B = np.clip(0.5 * edges + 0.25 * winner_b + 0.4 * depth_b, 0, 1)

    # near-side: sign of local depth step (negative = this side nearer)
    dmin = cv2.erode(depth, np.ones((5, 5), np.uint8))
    nearside = np.sign(depth - dmin)

    if return_parts:
        return B, {"edges": edges, "winner": winner_b, "depth": depth_b}, nearside
    return B


def canny_reference(frames):
    """Appearance-only reference: Canny on the max-focus composite."""
    grays = [to_gray_float(f) for f in frames]
    E = np.stack(content_aware_energies(grays), 0)
    winner = np.argmax(E, 0)
    yy, xx = np.indices(winner.shape)
    comp = np.stack(frames, 0)[winner, yy, xx]
    c = cv2.Canny(cv2.cvtColor(comp, cv2.COLOR_BGR2GRAY), 60, 160)
    return (c > 0).astype(np.float32)


def prf(pred_bin, gt_bin, tol):
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    gt_d = cv2.dilate(gt_bin.astype(np.uint8), k) > 0
    pr_d = cv2.dilate(pred_bin.astype(np.uint8), k) > 0
    p = float(pred_bin[gt_d].sum() / (pred_bin.sum() + 1e-9))
    r = float(gt_bin[pr_d].sum() / (gt_bin.sum() + 1e-9))
    f = 2 * p * r / (p + r + 1e-9)
    return p, r, f


def best_f(soft, gt_bin, tol, thresholds=np.linspace(0.15, 0.75, 13)):
    best = (0, 0, 0, 0)
    for t in thresholds:
        pred = (soft >= t).astype(np.uint8)
        pred = cv2.ximgproc.thinning(pred * 255) > 0 if hasattr(cv2, "ximgproc") else pred > 0
        p, r, f = prf(pred, gt_bin, tol)
        if f > best[2]:
            best = (p, r, f, t)
    return best


def camo_mask(scene_dir, gt_bmap):
    """Boundary pixels belonging to the camouflage object (rebuild by seed)."""
    from layers_gen import build_scene
    import glob as g
    photos = [cv2.imread(p) for p in sorted(g.glob(os.path.join(HERE, "data", "hires", "*", "gt.png")))]
    photos = [p for p in photos if p is not None]
    i = int(os.path.basename(scene_dir.rstrip("/")).split("_")[1])
    layers, _ = build_scene(photos, seed=100 + i, long_side=1536)
    camo_alpha = layers[3]["alpha"]  # [bg, rib-behind, blob, CAMO, rib-front]
    k = np.ones((3, 3), np.uint8)
    m = (camo_alpha > 0.5).astype(np.uint8)
    cb = (cv2.dilate(m, k) != cv2.erode(m, k)).astype(np.uint8)
    return (cb & gt_bmap).astype(np.uint8)


def main():
    tol = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    rows = []
    print(f"tolerance ±{tol}px; per scene: [P / R / F @best-thresh]   camo-recall (stack vs canny)")
    for d in sorted(glob.glob(os.path.join(LAYERS, "scene_*", ""))):
        frames = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(d, "frame_*.png")))]
        gt_b = (cv2.imread(os.path.join(d, "boundary.png"), 0) > 0).astype(np.uint8)
        B, parts, _ = stack_boundary(frames, return_parts=True)
        can = canny_reference(frames)

        pb, rb, fb, tb = best_f(B, gt_b, tol)
        pc, rc, fc = prf(can > 0, gt_b, tol)

        cm = camo_mask(d, gt_b)
        k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
        camo_rec = lambda pred: float(cm[cv2.dilate((pred).astype(np.uint8), k) > 0].sum() / (cm.sum() + 1e-9))
        cr_stack = camo_rec(B >= tb)
        cr_canny = camo_rec(can > 0)

        sid = os.path.basename(d.rstrip("/"))
        rows.append((fb, fc, cr_stack, cr_canny))
        print(f"  {sid}: stack {pb:.2f}/{rb:.2f}/{fb:.2f}@{tb:.2f}   canny F={fc:.2f}   camo {cr_stack:.2f} vs {cr_canny:.2f}")
    a = np.array(rows)
    print(f"\nMEAN  stack-F={a[:,0].mean():.3f}  canny-F={a[:,1].mean():.3f}  "
          f"camo-recall stack={a[:,2].mean():.3f} vs canny={a[:,3].mean():.3f}")


if __name__ == "__main__":
    main()
