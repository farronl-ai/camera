#!/usr/bin/env python3
"""16d P0 — WIDE-occluder benchmark (the veil-haze regime nobody measured).

Big matte blobs (radius ~0.15-0.3 of min dim, soft AA edge, optional holes) at a
near depth over real photo backgrounds, rendered with occ_gen's per-channel disk
compositing. Two CoC regimes: moderate (0.012·dim — standard) and GIANT
(0.04·dim — the finger/branch-near-lens case). True alpha + GT emitted in-memory.
"""
from __future__ import annotations
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layers_gen import blob_mask, texture  # noqa: E402
from occ_gen import occ_defocus, LONG  # noqa: E402
from hires_gen import add_noise  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "hires")


def scenes(coc_frac=0.012, n_scenes=4, float_frames=False):
    """Yield dicts: sid, gt, alpha, frames (N=2, noisy), max_r, near, far.

    float_frames=True: frames stay float32 end-to-end (no render or output
    quantization) with the SAME noise seeds — the FRONTIER 19 float condition.
    Default path is byte-identical to before the kw existed.
    """
    photos = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(SRC, "*", "gt.png")))]
    photos = [p for p in photos if p is not None]
    out = []
    for i, gp in enumerate(sorted(glob.glob(os.path.join(SRC, "*", "gt.png")))[:n_scenes]):
        far = cv2.imread(gp)
        h, w = far.shape[:2]
        s = LONG / max(h, w)
        far = cv2.resize(far, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        hh, ww = far.shape[:2]
        rng = np.random.default_rng(500 + i)
        alpha, _ = blob_mask(hh, ww, rng, scale=float(rng.uniform(0.18, 0.30)), holes=True)
        near = texture(photos, rng, hh, ww)
        if near.shape[:2] != (hh, ww):   # source photo smaller than the crop: resize up
            near = cv2.resize(near, (ww, hh), interpolation=cv2.INTER_LINEAR)
        near = near.astype(np.float32)
        gt = (near * alpha[..., None] + far.astype(np.float32) * (1 - alpha[..., None])).astype(np.uint8)
        max_r = coc_frac * max(hh, ww)
        if float_frames:
            frames = [np.clip(occ_defocus(far, near, alpha, f, 0.15, 0.85, max_r, quantize=False)
                              + np.random.default_rng(10 * i + k).normal(0, 3.0, far.shape), 0, 255).astype(np.float32)
                      for k, f in enumerate([0.15, 0.85])]
        else:
            frames = [add_noise(occ_defocus(far, near, alpha, f, 0.15, 0.85, max_r), 3.0, 10 * i + k)
                      for k, f in enumerate([0.15, 0.85])]
        out.append(dict(sid=f"{os.path.basename(os.path.dirname(gp))}_c{coc_frac:g}",
                        gt=gt, alpha=alpha, frames=frames, max_r=max_r,
                        near=near, far=far.astype(np.float32)))
    return out


if __name__ == "__main__":
    for coc in (0.012, 0.04):
        for sc in scenes(coc):
            print(f"  {sc['sid']}: max_r={sc['max_r']:.1f}px  alpha_area={float((sc['alpha']>0.5).mean())*100:.1f}%")
