#!/usr/bin/env python3
"""Is image-space decision fusion both faster AND as-good as multiband blend?

decision skips the Laplacian pyramids (~42% of blend's cost). With harden, it
takes hard-selected pixels directly (no coarse-band spread). Measure speed +
quality (Real-MFF GT + hard scenes) to decide the fast path.

Run:  python research/decision_vs_blend.py [n_realmff]
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
from focusstack.fusion import fuse_blend, fuse_decision  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RMFF = os.path.join(HERE, "data", "realmff", "extracted", "RealMFF")


def _pair(sz):
    r = np.random.default_rng(0)
    a = cv2.GaussianBlur(r.integers(0, 256, (sz, sz, 3), np.uint8), (0, 0), 1.0)
    return a, cv2.GaussianBlur(a, (0, 0), 3.0)


def _time(fn, reps=3):
    fn(); best = 1e9
    for _ in range(reps):
        t0 = time.time(); fn(); best = min(best, time.time() - t0)
    return best * 1000


def speed():
    print("SPEED (harden=0.5):")
    variants = [("blend", dict()), ("blend+ws0.5", dict(weight_scale=0.5)),
                ("decision", dict()), ("decision+ws0.5", dict(weight_scale=0.5))]
    for sz in (2048, 4096):
        a, b = _pair(sz)
        print(f"  {sz}x{sz}:")
        base = None
        for nm, kw in variants:
            fn = (lambda kw=kw, nm=nm: (fuse_blend if nm.startswith("blend") else fuse_decision)([a, b], harden=0.5, **kw))
            t = _time(fn)
            base = base or t
            print(f"    {nm:16s} {t:7.0f} ms ({base/t:.1f}x)")


def _q(args):
    pa, pb, pf = args
    a, b, gt = cv2.imread(pa), cv2.imread(pb), cv2.imread(pf)
    if a is None:
        return None
    return (M.ref_ssim(fuse_blend([a, b], harden=0.5), gt),
            M.ref_ssim(fuse_decision([a, b], harden=0.5), gt),
            M.ref_ssim(fuse_decision([a, b], harden=0.5, weight_scale=0.5), gt))


def quality(n):
    ids = sorted(os.path.basename(p)[:3] for p in glob.glob(os.path.join(RMFF, "imageA", "*_A.png")))[:n]
    trip = [(os.path.join(RMFF, "imageA", f"{i}_A.png"),
             os.path.join(RMFF, "imageB", f"{i}_B.png"),
             os.path.join(RMFF, "Fusion", f"{i}_F.png")) for i in ids]
    rows = []
    with cf.ProcessPoolExecutor(max_workers=min(12, os.cpu_count())) as ex:
        for r in ex.map(_q, trip):
            if r:
                rows.append(r)
    a = np.array(rows)
    print(f"\nReal-MFF ({len(rows)} pairs) GT-SSIM:")
    print(f"  blend            {a[:,0].mean():.4f}")
    print(f"  decision         {a[:,1].mean():.4f}   (delta {a[:,1].mean()-a[:,0].mean():+.5f})")
    print(f"  decision+ws0.5   {a[:,2].mean():.4f}   (delta {a[:,2].mean()-a[:,0].mean():+.5f})")


if __name__ == "__main__":
    speed()
    quality(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
