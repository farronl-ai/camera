#!/usr/bin/env python3
"""Should mod_laplacian replace laplacian as the default focus operator?

Theory: mod_laplacian = |I_xx| + |I_yy| (no sign cancellation) is a strictly more
robust focus signal on low-contrast/smooth content than the signed Laplacian, and
~equal on textured content. The hard benchmark showed a clear win on smooth metal.
Confirm broadly on Real-MFF ground truth before changing the package default.

Run:  python research/compare_operators.py [n_pairs]
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
OPS = ["laplacian", "mod_laplacian"]


def _eval(t):
    pa, pb, pf = t
    a, b, gt = cv2.imread(pa), cv2.imread(pb), cv2.imread(pf)
    if a is None or b is None or gt is None:
        return None
    out = {}
    for op in OPS:
        f = fuse_blend([a, b], focus_method=op)
        out[op] = (M.ref_ssim(f, gt), M.composite([a, b], f))
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    ids = sorted(os.path.basename(p)[:3] for p in glob.glob(os.path.join(RMFF, "imageA", "*_A.png")))
    ids = ids[:n]
    trip = [(os.path.join(RMFF, "imageA", f"{i}_A.png"),
             os.path.join(RMFF, "imageB", f"{i}_B.png"),
             os.path.join(RMFF, "Fusion", f"{i}_F.png")) for i in ids]
    rows = []
    with cf.ProcessPoolExecutor(max_workers=min(16, os.cpu_count())) as ex:
        for r in ex.map(_eval, trip):
            if r:
                rows.append(r)
    print(f"Real-MFF, {len(rows)} pairs:")
    for op in OPS:
        ss = np.mean([r[op][0] for r in rows])
        cc = np.mean([r[op][1] for r in rows])
        print(f"  {op:14s} GT-SSIM={ss:.4f}  composite={cc:.4f}")
    # per-pair: how often does mod_laplacian win/tie on true GT-SSIM?
    wins = sum(r["mod_laplacian"][0] > r["laplacian"][0] + 1e-5 for r in rows)
    ties = sum(abs(r["mod_laplacian"][0] - r["laplacian"][0]) <= 1e-5 for r in rows)
    print(f"  mod_laplacian better on {wins}/{len(rows)}, tie on {ties}")
    d = np.mean([r["mod_laplacian"][0] - r["laplacian"][0] for r in rows])
    print(f"  mean GT-SSIM delta (mod - lap) = {d:+.5f}")


if __name__ == "__main__":
    main()
