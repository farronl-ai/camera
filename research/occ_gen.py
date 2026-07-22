#!/usr/bin/env python3
"""B3 — occlusion-aware (alpha-matte) defocus + method re-ranking on honest data.

Prior generators used HARD per-pixel depth indexing: each pixel gets exactly one
depth's blur. Real defocus at a depth edge is a LAYERED composite: the out-of-focus
foreground, and its COVERAGE (alpha), both blur — so a defocused near layer
semi-transparently VEILS the background (blurred alpha < 1 at spread edges), and a
sharp background shows through the fringe. This is what the alpha-matte MFIF papers
target, and getting it right gates every synthetic conclusion (perband's crown
included).

Model (2 layers, near structures over far photo), per color channel (chromatic
aberration = per-channel focus offset):
    out = blur(near_rgb*alpha, CoC_near) + blur(far, CoC_far) * (1 - blur(alpha, CoC_near))
(premultiplied "over"). GT = sharp near over sharp far.

Run:  python research/occ_gen.py        # generate + re-rank perband/blend/pyramid
"""
from __future__ import annotations
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from metrics import _ssim_map, _gray32  # noqa: E402
from hires_gen import add_noise  # noqa: E402
from hardbench import disk_blur  # noqa: E402
from focusstack.fusion import fuse_blend, fuse_perband, fuse_pyramid  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "hires")
OUT = os.path.join(HERE, "data", "hires_occ")
LONG = 1536


def near_layer(far, seed):
    """Thin near structures + small objects as an RGBA layer over `far`."""
    r = np.random.default_rng(seed)
    h, w = far.shape[:2]
    near = np.zeros_like(far)
    alpha = np.zeros((h, w), np.float32)
    for _ in range(24):
        pts = np.array([[int(r.integers(w)), int(r.integers(h))] for _ in range(4)], np.int32).reshape(-1, 1, 2)
        col = tuple(int(c) for c in (r.integers(0, 60, 3) if r.random() < 0.5 else r.integers(200, 256, 3)))
        th = int(r.integers(1, 4))
        cv2.polylines(near, [pts], False, col, th, cv2.LINE_AA)
        cv2.polylines(alpha, [pts], False, 1.0, th, cv2.LINE_AA)
    for _ in range(30):
        c = (int(r.integers(w)), int(r.integers(h)))
        rad = int(r.integers(2, 6))
        cv2.circle(near, c, rad, tuple(int(x) for x in r.integers(0, 256, 3)), -1, cv2.LINE_AA)
        cv2.circle(alpha, c, rad, 1.0, -1, cv2.LINE_AA)
    return near.astype(np.float32), np.clip(alpha, 0, 1)


def occ_defocus(far, near, alpha, focus, near_d, far_d, max_r, ca=0.04, quantize=True):
    offs = (-ca, 0.0, ca)
    out = np.empty_like(far, np.float32)
    for c in range(3):
        rn = max_r * abs(near_d - (focus + offs[c]))
        rf = max_r * abs(far_d - (focus + offs[c]))
        npm = disk_blur((near[..., c] * alpha).astype(np.float32), rn)
        a_b = disk_blur(alpha, rn)
        far_b = disk_blur(far[..., c].astype(np.float32), rf)
        out[..., c] = npm + far_b * (1.0 - a_b)
    out = np.clip(out, 0, 255)
    return out.astype(np.uint8) if quantize else out


def main():
    os.makedirs(OUT, exist_ok=True)
    gts = sorted(glob.glob(os.path.join(SRC, "*", "gt.png")))[:4]
    rows = {"blend": [], "perband": [], "pyramid": []}
    nrows = {"blend": [], "perband": [], "pyramid": []}
    print(f"{'id':18s} | {'blend':>7s} {'perband':>7s} {'pyramid':>7s} | near-struct b/pb/py")
    for gp in gts:
        far = cv2.imread(gp)
        h, w = far.shape[:2]
        s = LONG / max(h, w)
        far = cv2.resize(far, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        near, alpha = near_layer(far, seed=hash(gp) % 1000)
        gt = (near * alpha[..., None] + far.astype(np.float32) * (1 - alpha[..., None])).astype(np.uint8)
        max_r = 0.012 * max(gt.shape[:2])
        frames = [add_noise(occ_defocus(far, near, alpha, f, 0.15, 0.85, max_r), 3.0, i)
                  for i, f in enumerate([0.15, 0.85])]
        outs = {"blend": fuse_blend(frames, harden=0.5), "perband": fuse_perband(frames, harden=0.5),
                "pyramid": fuse_pyramid(frames)}
        det = alpha > 0.3
        sid = os.path.basename(os.path.dirname(gp))
        line = []
        for k, im in outs.items():
            rows[k].append(M.ref_ssim(im, gt))
            nrows[k].append(float(_ssim_map(_gray32(im), _gray32(gt))[det].mean()))
            line.append(rows[k][-1])
        print(f"{sid:18s} | {line[0]:7.4f} {line[1]:7.4f} {line[2]:7.4f} | "
              f"{nrows['blend'][-1]:.3f}/{nrows['perband'][-1]:.3f}/{nrows['pyramid'][-1]:.3f}")
        cv2.imwrite(os.path.join(OUT, f"{sid}_frame1.png"), frames[1])
        cv2.imwrite(os.path.join(OUT, f"{sid}_gt.png"), gt)
    print(f"\nMEAN overall  blend={np.mean(rows['blend']):.4f}  perband={np.mean(rows['perband']):.4f}  "
          f"pyramid={np.mean(rows['pyramid']):.4f}")
    print(f"MEAN near     blend={np.mean(nrows['blend']):.4f}  perband={np.mean(nrows['perband']):.4f}  "
          f"pyramid={np.mean(nrows['pyramid']):.4f}")


if __name__ == "__main__":
    main()
