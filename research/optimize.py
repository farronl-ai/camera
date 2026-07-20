#!/usr/bin/env python3
"""M1 — global parameter auto-tuning against the calibrated GT-free objective.

Search the fuse_blend parameter space to maximize the composite metric over a
set of pairs, then VALIDATE the winner on Real-MFF ground truth (does maximizing
the no-reference objective actually raise true GT-SSIM vs the current defaults?).

Run:  python research/optimize.py [n_random] [n_tune_pairs] [n_val_pairs]
Writes research/best_global.json and appends to research/manifest.json.
"""

from __future__ import annotations

import concurrent.futures as cf
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from focusstack.fusion import fuse_blend  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STD = os.path.join(HERE, "data", "standard")
RMFF = os.path.join(HERE, "data", "realmff", "extracted", "RealMFF")

SPACE = {
    "focus_method": ["laplacian", "gradient", "tenengrad", "mod_laplacian"],
    "smooth_ksize": [3, 5, 7, 9, 13, 17],
    "radius": [1, 2, 3, 4, 6, 8, 12, 16],
    "eps": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "levels": [None, 3, 4, 5, 6, 7],
}
BASELINE = {"focus_method": "laplacian", "smooth_ksize": 9, "radius": 8, "eps": 1e-3, "levels": None}

_CACHE: dict = {}


def _load(path: str):
    if path not in _CACHE:
        _CACHE[path] = cv2.imread(path)
    return _CACHE[path]


def std_pairs() -> list[tuple[str, str]]:
    pres = sorted({p.rsplit("_", 1)[0] for p in glob.glob(os.path.join(STD, "*_1.tif"))})
    return [(f"{p}_1.tif", f"{p}_2.tif") for p in pres]


def rmff_triples() -> list[tuple[str, str, str]]:
    ids = sorted(os.path.basename(p)[:3] for p in glob.glob(os.path.join(RMFF, "imageA", "*_A.png")))
    return [(os.path.join(RMFF, "imageA", f"{i}_A.png"),
             os.path.join(RMFF, "imageB", f"{i}_B.png"),
             os.path.join(RMFF, "Fusion", f"{i}_F.png")) for i in ids]


def _eval_params_composite(args):
    """Mean composite over pairs for one param set (worker)."""
    params, pairs = args
    scores = []
    for pa, pb in pairs:
        a, b = _load(pa), _load(pb)
        fused = fuse_blend([a, b], **params)
        scores.append(M.composite([a, b], fused))
    return float(np.mean(scores))


def _sample(rng) -> dict:
    return {k: (v[int(rng.integers(len(v)))] if k != "eps" else float(rng.choice(v)))
            for k, v in SPACE.items()}


def search(pairs, n_random: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    candidates = [BASELINE] + [_sample(rng) for _ in range(n_random)]
    with cf.ProcessPoolExecutor(max_workers=min(16, os.cpu_count())) as ex:
        scores = list(ex.map(_eval_params_composite, [(c, pairs) for c in candidates]))
    best = max(zip(scores, candidates), key=lambda t: t[0])
    best_score, best_params = best

    # Coordinate-descent local refine around the best.
    improved = True
    rounds = 0
    while improved and rounds < 4:
        improved = False
        rounds += 1
        for dim, values in SPACE.items():
            trials = [dict(best_params, **{dim: v}) for v in values if v != best_params[dim]]
            with cf.ProcessPoolExecutor(max_workers=min(16, os.cpu_count())) as ex:
                ts = list(ex.map(_eval_params_composite, [(t, pairs) for t in trials]))
            for s, t in zip(ts, trials):
                if s > best_score:
                    best_score, best_params, improved = s, t, True
    return best_params, best_score


def validate_gt(params, triples):
    """Mean GT-SSIM over Real-MFF triples for a param set."""
    ssims = []
    for pa, pb, pf in triples:
        a, b, gt = _load(pa), _load(pb), _load(pf)
        fused = fuse_blend([a, b], **params)
        ssims.append(M.ref_ssim(fused, gt))
    return float(np.mean(ssims))


def main():
    n_random = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    n_tune = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    n_val = int(sys.argv[3]) if len(sys.argv) > 3 else 120

    # Tune and validate on DISJOINT splits of Real-MFF: tune with the GT-free
    # composite only, validate against held-out ground truth. This directly tests
    # the core hypothesis (maximizing the no-ref objective raises true quality).
    triples = rmff_triples()
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(triples))
    tune_ids = perm[:n_tune]
    val_ids = perm[n_tune:n_tune + n_val]
    tune_pairs = [(triples[i][0], triples[i][1]) for i in tune_ids]
    val = [triples[i] for i in val_ids]

    print(f"tuning on {len(tune_pairs)} Real-MFF pairs (composite only), "
          f"{n_random} random + local refine ...")
    t0 = time.time()
    best_params, best_score = search(tune_pairs, n_random)
    dt = time.time() - t0

    base_score = _eval_params_composite((BASELINE, tune_pairs))
    print(f"\nbaseline composite = {base_score:.4f}  ({BASELINE})")
    print(f"tuned    composite = {best_score:.4f}  ({best_params})")
    print(f"search time = {dt:.0f}s")

    # Ground-truth validation: does the composite-tuned param actually raise true SSIM?
    print(f"\nvalidating on {len(val)} held-out Real-MFF GT pairs ...")
    base_gt = validate_gt(BASELINE, val)
    tuned_gt = validate_gt(best_params, val)
    print(f"  baseline GT-SSIM = {base_gt:.4f}")
    print(f"  tuned    GT-SSIM = {tuned_gt:.4f}   (delta {tuned_gt - base_gt:+.4f})")

    out = {"baseline": BASELINE, "baseline_composite": base_score, "baseline_gt_ssim": base_gt,
           "best": best_params, "best_composite": best_score, "tuned_gt_ssim": tuned_gt,
           "n_random": n_random, "n_tune": n_tune, "n_val": n_val}
    with open(os.path.join(HERE, "best_global.json"), "w") as f:
        json.dump(out, f, indent=2)
    manifest = os.path.join(HERE, "manifest.json")
    log = json.load(open(manifest)) if os.path.exists(manifest) else []
    log.append({"stage": "M1_global", **out})
    json.dump(log, open(manifest, "w"), indent=2)
    print(f"\nwrote best_global.json + manifest.json")


if __name__ == "__main__":
    main()
