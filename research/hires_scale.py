#!/usr/bin/env python3
"""H3 hypothesis: guided-blend underperforms at high-res because its params are
FIXED pixel sizes (radius 8, focus pool 9) — tiny vs a 37px CoC. Scale them with
resolution and blend should match/beat pyramid (which is inherently multi-scale).

Test blend at several radius/pool scales vs pyramid on the high-res GT stacks.
Run:  python research/hires_scale.py
"""
from __future__ import annotations
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from hires_eval import load_stack, manifest  # noqa: E402
from focusstack.fusion import fuse_blend, fuse_pyramid  # noqa: E402


def odd(x):
    x = max(3, int(round(x)))
    return x + 1 - (x % 2)


def main():
    ks = [0.0, 0.006, 0.012, 0.02]  # 0.0 = default (radius8/pool9); else scale*max_dim
    rows = []
    hdr = ["default"] + [f"scale={k}" for k in ks[1:]] + ["pyramid"]
    print(f"{'id':18s} {'ctype':14s} " + " ".join(f"{h:>10s}" for h in hdr))
    for m in manifest():
        frames, gt = load_stack(m["id"])
        md = max(frames[0].shape[:2])
        vals = []
        for k in ks:
            if k == 0.0:
                f = fuse_blend(frames, focus_method="content_aware", harden=0.5)
            else:
                r = max(4, int(md * k))
                f = fuse_blend(frames, focus_method="content_aware", harden=0.5,
                               radius=r, smooth_ksize=odd(md * k))
            vals.append(M.ref_ssim(f, gt))
        vals.append(M.ref_ssim(fuse_pyramid(frames), gt))
        rows.append((m["content_type"], vals))
        print(f"{m['id']:18s} {m['content_type']:14s} " + " ".join(f"{v:10.4f}" for v in vals))
    arr = np.array([r[1] for r in rows])
    print("\nMEAN:              " + " " * 14 + " ".join(f"{v:10.4f}" for v in arr.mean(0)))
    best = np.array(hdr)[arr.mean(0).argmax()]
    print(f"best overall: {best}")


if __name__ == "__main__":
    main()
