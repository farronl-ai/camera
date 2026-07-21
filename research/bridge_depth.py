#!/usr/bin/env python3
"""E3 — semantic monocular-depth bridge (runs in .venv312, torch CPU).

Usage (from the 3.12 env):
    .venv312/bin/python research/bridge_depth.py IMG [IMG...]
For each IMG writes IMG.depth.npy (float32 HxW, higher = nearer per DA-V2
convention) and IMG.depth.png (TURBO vis). First run downloads
depth-anything/Depth-Anything-V2-Small-hf (~100MB) from HuggingFace.
"""
from __future__ import annotations
import sys

import numpy as np
from PIL import Image


def main():
    from transformers import pipeline
    pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf",
                    device=-1)
    for path in sys.argv[1:]:
        img = Image.open(path).convert("RGB")
        out = pipe(img)
        d = np.array(out["predicted_depth"], dtype=np.float32)
        if d.shape != (img.height, img.width):
            import cv2
            d = cv2.resize(d, (img.width, img.height), interpolation=cv2.INTER_LINEAR)
        np.save(path + ".depth.npy", d)
        try:
            import cv2
            dn = ((d - d.min()) / (np.ptp(d) + 1e-9) * 255).astype(np.uint8)
            cv2.imwrite(path + ".depth.png", cv2.applyColorMap(dn, cv2.COLORMAP_TURBO))
        except Exception:
            pass
        print(f"  {path}: depth {d.shape}, range {d.min():.2f}..{d.max():.2f}")


if __name__ == "__main__":
    main()
