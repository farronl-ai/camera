#!/usr/bin/env python3
"""Frontier 3b — occlusion-aware DE-VEILING: unmix the far-focused frame.

Physics (occ_gen): the far-focused frame is a matte composite
    obs = blur(near*alpha, r) + far_sharp * (1 - blur(alpha, r)),  r ~ 0.7*max_r.
The sharp background is VEILED by the defocused near layer — 57% of perband's
error lives in that fringe (task #32 probe). But the mix is invertible:
    far_est = (obs - blur(near_premult_est, r)) / (1 - blur(alpha_est, r)).

Estimation from frames alone (no GT):
  - alpha_est: near structures are where the near-focused frame DECISIVELY wins
    the focus contest (thin sharp structures vs 0.7*max_r blur -> clean dominance).
  - near_premult_est = near_frame * alpha_est (on structure cores alpha~1).
  - r: v1 uses the generator's known r to measure the approach's CEILING; blind r
    estimation is v2 IF the ceiling justifies it.
Guard: where (1 - alpha_blur) < 0.25 the background is (nearly) fully occluded —
nothing to recover; keep the observation (fusion takes the near frame there anyway).

A/B on the 4 occ scenes (GT): perband(raw) vs perband(near, deveiled_far) vs
ORACLE deveil (true alpha/near from the generator) as upper bound.

Run:  python research/deveil.py
"""
from __future__ import annotations
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from occ_gen import near_layer, occ_defocus, LONG  # noqa: E402
from hires_gen import add_noise  # noqa: E402
from hardbench import disk_blur  # noqa: E402
from focusstack.fusion import fuse_perband  # noqa: E402
from focusstack.focus import content_aware_energies  # noqa: E402
from focusstack.io import to_gray_float  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "hires")


def estimate_alpha(near_frame, far_frame):
    """Soft near-structure mask from focus dominance (no GT)."""
    e0, e1 = content_aware_energies([to_gray_float(near_frame), to_gray_float(far_frame)])
    raw = ((e0 > 2.0 * e1) & (e0 > np.percentile(e0, 60))).astype(np.float32)
    # thin structures are 1-6 px: dilate a touch to cover anti-aliased skirts,
    # then a light blur for soft edges
    raw = cv2.dilate(raw, np.ones((3, 3), np.uint8))
    return cv2.GaussianBlur(raw, (0, 0), 1.0)


def deveil(far_frame, alpha_est, near_premult_est, r, guard=0.25):
    """Invert the matte composite in the veil fringe."""
    out = far_frame.astype(np.float32).copy()
    a_blur = disk_blur(alpha_est, r)
    denom = 1.0 - a_blur
    ok = denom > guard
    for c in range(3):
        veil = disk_blur(near_premult_est[..., c].astype(np.float32), r)
        rec = (far_frame[..., c].astype(np.float32) - veil) / np.maximum(denom, guard)
        out[..., c] = np.where(ok, rec, out[..., c])
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    gts = sorted(glob.glob(os.path.join(SRC, "*", "gt.png")))[:4]
    print(f"{'id':18s} | {'base':>7s} {'deveil':>7s} {'oracle':>7s} | fringe mean|err| b/d/o")
    agg = {"base": [], "dev": [], "orc": []}
    fr_agg = {"base": [], "dev": [], "orc": []}
    for gp in gts:
        far = cv2.imread(gp)
        h, w = far.shape[:2]
        s = LONG / max(h, w)
        far = cv2.resize(far, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        near, alpha = near_layer(far, seed=hash(gp) % 1000)
        gt = (near * alpha[..., None] + far.astype(np.float32) * (1 - alpha[..., None])).astype(np.uint8)
        max_r = 0.012 * max(gt.shape[:2])
        r = 0.7 * max_r
        frames = [add_noise(occ_defocus(far, near, alpha, f, 0.15, 0.85, max_r), 3.0, i)
                  for i, f in enumerate([0.15, 0.85])]

        # estimated de-veil (frames only, known r)
        a_est = estimate_alpha(frames[0], frames[1])
        npm_est = frames[0].astype(np.float32) * a_est[..., None]
        dev_far = deveil(frames[1], a_est, npm_est, r)
        # oracle de-veil (true alpha & premult)
        orc_far = deveil(frames[1], alpha, near * alpha[..., None], r)

        outs = {"base": fuse_perband(frames, harden=0.5),
                "dev": fuse_perband([frames[0], dev_far], harden=0.5),
                "orc": fuse_perband([frames[0], orc_far], harden=0.5)}

        a_blur_true = disk_blur(alpha, r)
        fringe = (a_blur_true > 0.03) & (a_blur_true < 0.97) & (alpha < 0.5)
        sid = os.path.basename(os.path.dirname(gp))
        line = {}
        for k, im in outs.items():
            agg[k].append(M.ref_ssim(im, gt))
            err = np.abs(im.astype(np.float32) - gt.astype(np.float32)).sum(2)
            fr_agg[k].append(float(err[fringe].mean()))
            line[k] = (agg[k][-1], fr_agg[k][-1])
        print(f"{sid:18s} | {line['base'][0]:7.4f} {line['dev'][0]:7.4f} {line['orc'][0]:7.4f} | "
              f"{line['base'][1]:5.1f}/{line['dev'][1]:5.1f}/{line['orc'][1]:5.1f}")
    print(f"\nMEAN GT-SSIM     base={np.mean(agg['base']):.4f}  deveil={np.mean(agg['dev']):.4f}  "
          f"oracle={np.mean(agg['orc']):.4f}")
    print(f"MEAN fringe err  base={np.mean(fr_agg['base']):.1f}  deveil={np.mean(fr_agg['dev']):.1f}  "
          f"oracle={np.mean(fr_agg['orc']):.1f}")


if __name__ == "__main__":
    main()
