#!/usr/bin/env python3
"""Non-regression check: does the recommended high-quality combo hurt clean data?

Compare the package default (blend, laplacian, harden=0) vs the recommended
combo (blend, content_aware focus + harden) on Real-MFF ground truth. The combo
must not regress clean/textured data (its wins are on smooth content + spread/
thin structures). If neutral-or-better here, it is safe to recommend.

Run:  python research/recommend_check.py [n]
"""
from __future__ import annotations
import concurrent.futures as cf
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from focusstack.fusion import fuse_blend  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RMFF = os.path.join(HERE, "data", "realmff", "extracted", "RealMFF")


def _eval(t):
    a, b, gt = (cv2.imread(x) for x in t)
    if a is None:
        return None
    default = fuse_blend([a, b])
    combo = fuse_blend([a, b], focus_method="content_aware", harden=0.5)
    return M.ref_ssim(default, gt), M.ref_ssim(combo, gt)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    ids = sorted(os.path.basename(p)[:3] for p in glob.glob(os.path.join(RMFF, "imageA", "*_A.png")))[:n]
    trip = [(os.path.join(RMFF, "imageA", f"{i}_A.png"),
             os.path.join(RMFF, "imageB", f"{i}_B.png"),
             os.path.join(RMFF, "Fusion", f"{i}_F.png")) for i in ids]
    rows = []
    with cf.ProcessPoolExecutor(max_workers=min(16, os.cpu_count())) as ex:
        for r in ex.map(_eval, trip):
            if r:
                rows.append(r)
    d = np.mean([r[0] for r in rows]); c = np.mean([r[1] for r in rows])
    print(f"Real-MFF {len(rows)} pairs GT-SSIM:")
    print(f"  default (laplacian, harden=0)          {d:.4f}")
    print(f"  combo (content_aware, harden=0.5)      {c:.4f}   (delta {c-d:+.5f})")
    worse = sum(r[1] < r[0] - 1e-4 for r in rows)
    print(f"  combo worse on {worse}/{len(rows)} pairs (regression count)")


if __name__ == "__main__":
    main()
