#!/usr/bin/env python3
"""Does the recommended engine (content_aware + harden) win MORE at high res?

Compares package default-ish baseline (laplacian, harden=0) vs recommended
(content_aware, harden=0.5) on the hard defocus-spread + gradient-metal scenes at
a given resolution. Defocus spread scales with resolution, so structural wins
should hold or grow. Saves worst-region crops for visual inspection.

Run:  python research/hires_recommend.py [SZ]
"""
from __future__ import annotations
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
import hardbench as HB  # noqa: E402
from focusstack.fusion import fuse_blend  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "hires_out")
os.makedirs(OUT, exist_ok=True)


def tile_worst(fused, gt, t=8):
    from metrics import _ssim_map, _gray32
    s = _ssim_map(_gray32(fused), _gray32(gt)); h, w = s.shape
    return min(s[i * (h // t):(i + 1) * (h // t), j * (w // t):(j + 1) * (w // t)].mean()
               for i in range(t) for j in range(t))


def main():
    HB.SZ = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    print(f"resolution SZ={HB.SZ}")
    for name, (gen, nf, mr) in HB.SCENES.items():
        base, depth = gen(); gt = base
        planes = np.linspace(float(depth.min()), float(depth.max()), nf)
        frames = [HB.add_noise(HB.defocus_disk(base, depth, f, mr), 3.0, i) for i, f in enumerate(planes)]
        baseline = fuse_blend(frames, focus_method="laplacian", harden=0.0)
        recommended = fuse_blend(frames, focus_method="content_aware", harden=0.5)
        print(f"\n{name}:")
        print(f"  baseline (laplacian,h0)        GT-SSIM={M.ref_ssim(baseline, gt):.4f}  tile_worst={tile_worst(baseline, gt):.4f}")
        print(f"  recommended (content_aware,h.5) GT-SSIM={M.ref_ssim(recommended, gt):.4f}  tile_worst={tile_worst(recommended, gt):.4f}")
        # worst-region crop (baseline error)
        err = cv2.boxFilter(np.abs(baseline.astype(np.int16) - gt).sum(2).astype(np.float32), cv2.CV_32F, (41, 41))
        y, x = np.unravel_index(err.argmax(), err.shape); h = 160
        y0, x0 = max(0, y - h), max(0, x - h)
        def z(im): return cv2.resize(im[y0:y0 + 2 * h, x0:x0 + 2 * h], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(OUT, f"{name}_{HB.SZ}.png"), np.hstack([z(gt), z(baseline), z(recommended)]))
    print(f"\nwrote crops to {OUT} [ GT | baseline | recommended ]")


if __name__ == "__main__":
    main()
