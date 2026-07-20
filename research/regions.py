#!/usr/bin/env python3
"""M2 — region-adaptive fusion: different tunes for different regions, stitched.

Instead of expensively re-searching parameters per region, we exploit that a
*region* is really "where a particular tune wins." So:

  1. Produce a few diversely-tuned candidate fusions of the whole image
     (crisp/fine ... smooth/robust).
  2. Score each candidate's LOCAL quality per pixel with the calibrated
     composite map (no ground truth needed).
  3. At each pixel, prefer the locally-best candidate; make that choice
     edge-aware with a guided filter (so region borders follow real edges) and
     normalize to weights.
  4. Blend the candidates with those weights via multi-band blending — seamless
     by construction (no hard cuts, no tile seams).

This subsumes "overlapping partitions / cut-and-stitch": the partition is soft,
per-pixel, and content-driven, and the stitch is the same Burt-Adelson blend the
package already uses.

Run:  python research/regions.py [n_val_pairs]   # eval adaptive vs global vs baseline on Real-MFF GT
"""

from __future__ import annotations

import concurrent.futures as cf
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from focusstack.fusion import fuse_blend, guided_filter, multiband_blend  # noqa: E402
from focusstack.io import to_gray_float  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RMFF = os.path.join(HERE, "data", "realmff", "extracted", "RealMFF")

# Diverse tune presets spanning crisp/fine <-> smooth/robust. Refined from M1's
# top diverse winners when best_global.json is available.
PRESETS = [
    {"focus_method": "laplacian", "smooth_ksize": 9, "radius": 8, "eps": 1e-3, "levels": None},
    {"focus_method": "mod_laplacian", "smooth_ksize": 5, "radius": 2, "eps": 1e-3, "levels": 5},
    {"focus_method": "tenengrad", "smooth_ksize": 13, "radius": 16, "eps": 1e-2, "levels": 4},
    {"focus_method": "gradient", "smooth_ksize": 7, "radius": 4, "eps": 3e-4, "levels": 6},
]


def fuse_adaptive(sources, presets=None, guide_radius=16, guide_eps=1e-3, harden=0.0,
                  return_debug=False):
    presets = presets or PRESETS
    cands = [fuse_blend(sources, harden=harden, **p) for p in presets]
    # F12: use q_ssim for per-pixel selection — the global composite's q_abf term
    # ANTI-correlates per-tile and misleads local decisions.
    qmaps = np.stack([M.q_ssim_map(sources, c) for c in cands], axis=0)  # (P,H,W)
    winner = np.argmax(qmaps, axis=0)

    weights = []
    for p, cand in enumerate(cands):
        raw = (winner == p).astype(np.float32)
        guide = to_gray_float(cand) / 255.0
        weights.append(np.clip(guided_filter(guide, raw, guide_radius, guide_eps), 0.0, None))
    w = np.stack(weights, axis=0)
    w = w / (w.sum(axis=0, keepdims=True) + 1e-8)

    fused = multiband_blend(cands, w)
    if return_debug:
        return fused, {"candidates": cands, "winner": winner, "weights": w, "qmaps": qmaps}
    return fused


# --------------------------------------------------------------------------- #
# Evaluation vs global baseline on Real-MFF ground truth
# --------------------------------------------------------------------------- #
def _triples():
    ids = sorted(os.path.basename(p)[:3] for p in glob.glob(os.path.join(RMFF, "imageA", "*_A.png")))
    return [(os.path.join(RMFF, "imageA", f"{i}_A.png"),
             os.path.join(RMFF, "imageB", f"{i}_B.png"),
             os.path.join(RMFF, "Fusion", f"{i}_F.png")) for i in ids]


def _global_best():
    p = os.path.join(HERE, "best_global.json")
    if os.path.exists(p):
        return json.load(open(p))["best"]
    return PRESETS[0]


def _eval_one(args):
    pa, pb, pf, gbest = args
    a, b, gt = cv2.imread(pa), cv2.imread(pb), cv2.imread(pf)
    if a is None or b is None or gt is None:
        return None
    base = fuse_blend([a, b])                       # current package default
    glob_ = fuse_blend([a, b], **gbest)             # M1 tuned global
    adapt = fuse_adaptive([a, b])                   # region-adaptive
    return {
        "base_ssim": M.ref_ssim(base, gt), "base_comp": M.composite([a, b], base),
        "glob_ssim": M.ref_ssim(glob_, gt), "glob_comp": M.composite([a, b], glob_),
        "adapt_ssim": M.ref_ssim(adapt, gt), "adapt_comp": M.composite([a, b], adapt),
    }


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    # held-out split (same perm/offset the tuner used: skip its first 60)
    triples = _triples()
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(triples))
    val = [triples[i] for i in perm[60:60 + n]]
    gbest = _global_best()
    print(f"eval on {len(val)} held-out Real-MFF pairs; global-best={gbest}")

    rows = []
    with cf.ProcessPoolExecutor(max_workers=min(16, os.cpu_count())) as ex:
        for r in ex.map(_eval_one, [(*t, gbest) for t in val]):
            if r:
                rows.append(r)

    def mean(k):
        return float(np.mean([r[k] for r in rows]))

    print(f"\n{'method':10s} {'GT-SSIM':>9s} {'composite':>10s}")
    for tag, name in [("base", "baseline"), ("glob", "global-tuned"), ("adapt", "adaptive")]:
        print(f"{name:10s} {mean(tag+'_ssim'):9.4f} {mean(tag+'_comp'):10.4f}")

    out = {"stage": "M2_adaptive", "n_val": len(rows),
           "baseline_gt_ssim": mean("base_ssim"), "global_gt_ssim": mean("glob_ssim"),
           "adaptive_gt_ssim": mean("adapt_ssim"),
           "presets": PRESETS, "global_best": gbest}
    manifest = os.path.join(HERE, "manifest.json")
    log = json.load(open(manifest)) if os.path.exists(manifest) else []
    log.append(out)
    json.dump(log, open(manifest, "w"), indent=2)
    print("\nappended M2 result to manifest.json")


if __name__ == "__main__":
    main()
