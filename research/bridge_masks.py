#!/usr/bin/env python3
"""16e T1 — segmentation-mask bridge (runs in .venv312, torch CPU).

Usage:  .venv312/bin/python research/bridge_masks.py IMG [IMG...]
For each IMG writes IMG.masks.npy: uint8 array (K, H, W) of the K largest
automatic masks (K<=24), plus IMG.masks.png visualization.
Backend: FastSAM via ultralytics (weights auto-download on first use);
falls back with a clear error if unavailable.
"""
from __future__ import annotations
import sys

import numpy as np


def main():
    from ultralytics import FastSAM
    import cv2
    model = FastSAM("FastSAM-s.pt")
    for path in sys.argv[1:]:
        res = model(path, device="cpu", retina_masks=True, imgsz=1024,
                    conf=0.4, iou=0.9, verbose=False)
        img = cv2.imread(path)
        h, w = img.shape[:2]
        if not res or res[0].masks is None:
            np.save(path + ".masks.npy", np.zeros((0, h, w), np.uint8))
            print(f"  {path}: no masks")
            continue
        m = res[0].masks.data.cpu().numpy().astype(np.uint8)      # (K,h',w')
        if m.shape[1:] != (h, w):
            m = np.stack([cv2.resize(mi, (w, h), interpolation=cv2.INTER_NEAREST) for mi in m], 0)
        order = np.argsort([-int(mi.sum()) for mi in m])[:24]
        m = m[order]
        np.save(path + ".masks.npy", m)
        vis = img.copy()
        rng = np.random.default_rng(0)
        for mi in m:
            vis[mi > 0] = 0.6 * vis[mi > 0] + 0.4 * rng.integers(60, 255, 3)
        cv2.imwrite(path + ".masks.png", vis)
        print(f"  {path}: {len(m)} masks")


if __name__ == "__main__":
    main()
