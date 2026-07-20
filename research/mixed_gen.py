#!/usr/bin/env python3
"""L1 (corrected) — FINE-SCALE-DEPTH high-res GT benchmark.

The local-scale advantage only appears where sharp/blurred content alternates at a
scale FINER than the fusion window — i.e. small structures at VARYING depths, not
whole objects vs background. So: real high-res photo = far background; overlay thin
near structures (wires/hairs, 1–3 px) + small near objects (dots) at a near depth.
GT = the composite with everything sharp. Frames: near-focus (structures sharp,
background blurred) and far-focus (structures defocus-spread, background sharp).

A GLOBAL large window must blur the fine near/far boundaries (halo/soften the thin
structures); a LOCAL small window at those structures preserves them. This is the
data that tests Farron's "small details at varying depths" point.

Run:  python research/mixed_gen.py
Writes data/hires_mixed/<id>/ {gt.png, frame_0..1.png, depth.png} + manifest.json
"""
from __future__ import annotations
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hires_gen import defocus_ca, add_noise  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "hires")
OUT = os.path.join(HERE, "data", "hires_mixed")


def overlay_fine_near(base, seed):
    """Draw thin near structures + small near objects; return (gt_composite, near_mask)."""
    r = np.random.default_rng(seed)
    h, w = base.shape[:2]
    gt = base.copy()
    mask = np.zeros((h, w), np.uint8)
    n_curves = 24
    for _ in range(n_curves):
        pts = np.array([[int(r.integers(w)), int(r.integers(h))] for _ in range(4)], np.int32)
        color = tuple(int(c) for c in (r.integers(0, 60, 3) if r.random() < 0.5
                                       else r.integers(200, 256, 3)))  # dark or bright
        th = int(r.integers(1, 4))
        cv2.polylines(gt, [pts.reshape(-1, 1, 2)], False, color, th, cv2.LINE_AA)
        cv2.polylines(mask, [pts.reshape(-1, 1, 2)], False, 255, th + 2)
    for _ in range(30):  # small near objects
        c = (int(r.integers(w)), int(r.integers(h)))
        rad = int(r.integers(2, 6))
        col = tuple(int(x) for x in r.integers(0, 256, 3))
        cv2.circle(gt, c, rad, col, -1, cv2.LINE_AA)
        cv2.circle(mask, c, rad + 2, 255, -1)
    return gt, mask


def main():
    os.makedirs(OUT, exist_ok=True)
    gts = sorted(glob.glob(os.path.join(SRC, "*", "gt.png")))
    manifest = []
    for idx, gp in enumerate(gts):
        sid = os.path.basename(os.path.dirname(gp))
        base = cv2.imread(gp)
        if base is None:
            continue
        h, w = base.shape[:2]
        gt, near = overlay_fine_near(base, idx)
        depth = np.full((h, w), 0.85, np.float32)          # background far
        depth[near > 0] = 0.15                             # thin near structures
        depth = cv2.GaussianBlur(depth, (0, 0), 1.5)       # slight, keep boundaries fine
        max_r = 0.012 * max(h, w)
        sdir = os.path.join(OUT, sid)
        os.makedirs(sdir, exist_ok=True)
        cv2.imwrite(os.path.join(sdir, "gt.png"), gt)
        cv2.imwrite(os.path.join(sdir, "depth.png"), (depth * 255).astype(np.uint8))
        for fi, fp in enumerate([0.15, 0.85]):
            fr = add_noise(defocus_ca(gt, depth, fp, max_r), 3.0, fi)
            cv2.imwrite(os.path.join(sdir, f"frame_{fi}.png"), fr)
        near_frac = float((near > 0).mean())
        manifest.append({"id": sid, "dims": [h, w], "frames": 2,
                         "max_coc_px": round(max_r, 1), "near_frac": round(near_frac, 3)})
        print(f"  {sid:18s} {w}x{h}  CoC={max_r:.0f}px  near(fine) frac={near_frac:.3f}")
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)
    print(f"\ngenerated {len(manifest)} fine-depth stacks -> {OUT}")


if __name__ == "__main__":
    main()
