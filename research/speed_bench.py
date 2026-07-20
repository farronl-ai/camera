#!/usr/bin/env python3
"""Benchmark weight_scale: speed (high-res) + quality (Real-MFF GT).

Run:  python research/speed_bench.py [n_realmff]
"""
from __future__ import annotations
import concurrent.futures as cf
import glob
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from focusstack.fusion import fuse_blend  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RMFF = os.path.join(HERE, "data", "realmff", "extracted", "RealMFF")
SCALES = [1.0, 0.5, 0.25]


def _pair(sz):
    rng = np.random.default_rng(0)
    a = cv2.GaussianBlur(rng.integers(0, 256, (sz, sz, 3), np.uint8), (0, 0), 1.0)
    b = cv2.GaussianBlur(a, (0, 0), 3.0)
    return a, b


def speed():
    print("SPEED (synthetic, harden=0.5):")
    print(f"  {'res':>6s}  " + "  ".join(f"scale={s}" for s in SCALES))
    for sz in (1024, 2048, 4096):
        row = []
        for s in SCALES:
            a, b = _pair(sz)
            fuse_blend([a, b], harden=0.5, weight_scale=s)  # warm
            t = 1e9
            for _ in range(2):
                t0 = time.time(); fuse_blend([a, b], harden=0.5, weight_scale=s); t = min(t, time.time() - t0)
            row.append(t * 1000)
        base = row[0]
        print(f"  {sz:>6d}  " + "  ".join(f"{r:7.0f}ms({base/r:.1f}x)" for r in row))


def _q(args):
    pa, pb, pf = args
    a, b, gt = cv2.imread(pa), cv2.imread(pb), cv2.imread(pf)
    if a is None:
        return None
    return [M.ref_ssim(fuse_blend([a, b], harden=0.5, weight_scale=s), gt) for s in SCALES]


def quality(n):
    ids = sorted(os.path.basename(p)[:3] for p in glob.glob(os.path.join(RMFF, "imageA", "*_A.png")))[:n]
    trip = [(os.path.join(RMFF, "imageA", f"{i}_A.png"),
             os.path.join(RMFF, "imageB", f"{i}_B.png"),
             os.path.join(RMFF, "Fusion", f"{i}_F.png")) for i in ids]
    rows = []
    with cf.ProcessPoolExecutor(max_workers=min(16, os.cpu_count())) as ex:
        for r in ex.map(_q, trip):
            if r:
                rows.append(r)
    arr = np.array(rows)
    print(f"\nQUALITY (Real-MFF {len(rows)} pairs, GT-SSIM):")
    for i, s in enumerate(SCALES):
        print(f"  scale={s}: {arr[:, i].mean():.4f}   (delta vs full {arr[:, i].mean()-arr[:, 0].mean():+.5f})")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    speed()
    quality(n)
