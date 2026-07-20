#!/usr/bin/env python3
"""Calibrate the GT-free objective against Real-MFF ground truth.

We will optimize everything against a no-reference metric, so first prove which
metric (or weighted combo) actually predicts TRUE quality. For each Real-MFF
pair (which has an all-in-focus GT), we build several fusion candidates spanning
a quality range, then check how well each no-reference metric ranks them versus
the ground-truth SSIM ranking (mean per-pair Spearman). The best combo's weights
are written to research/metric_weights.json and become the objective for M1+.

This is the ONLY place ground truth is used — to validate the metric during
development. Inference never needs it.

Run:  python research/validate_metrics.py [n_pairs]
"""

from __future__ import annotations

import concurrent.futures as cf
import glob
import itertools
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from focusstack.fusion import fuse_blend, fuse_decision, fuse_max, fuse_pyramid  # noqa: E402
from focusstack.focus import focus_measures  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RMFF = os.path.join(HERE, "data", "realmff", "extracted", "RealMFF")
NO_REF = ["q_abf", "q_mi", "q_ssim", "sharp"]


def candidates(a, b):
    """Fusion candidates spanning a quality range (good -> bad)."""
    srcs = [a, b]
    fm, _ = fuse_max(srcs, focus_measures(srcs))
    return {
        "blend": fuse_blend(srcs),
        "decision": fuse_decision(srcs),
        "pyramid": fuse_pyramid(srcs),
        "max": fm,
        "avg": ((a.astype(np.float32) + b.astype(np.float32)) / 2).astype(np.uint8),  # ghosty
        "srcA": a.copy(),  # half-defocused: anchors the low end
    }


def eval_pair(idx: str):
    a = cv2.imread(os.path.join(RMFF, "imageA", f"{idx}_A.png"))
    b = cv2.imread(os.path.join(RMFF, "imageB", f"{idx}_B.png"))
    gt = cv2.imread(os.path.join(RMFF, "Fusion", f"{idx}_F.png"))
    if a is None or b is None or gt is None:
        return None
    rows = []
    for _name, fused in candidates(a, b).items():
        nr = M.all_scores([a, b], fused)
        rows.append({**{k: nr[k] for k in NO_REF},
                     "ref_ssim": M.ref_ssim(fused, gt)})
    return rows


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    def rank(v):
        order = v.argsort()
        r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(len(v))
        return r
    rx, ry = rank(x), rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 1e-9 else 0.0


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    ids = sorted(os.path.basename(p)[:3] for p in glob.glob(os.path.join(RMFF, "imageA", "*_A.png")))
    rng = np.random.default_rng(0)
    ids = list(rng.permutation(ids)[:n])
    print(f"evaluating {len(ids)} Real-MFF pairs x 6 candidates ...")

    per_pair = []
    with cf.ProcessPoolExecutor(max_workers=min(16, os.cpu_count())) as ex:
        for rows in ex.map(eval_pair, ids):
            if rows:
                per_pair.append(rows)
    print(f"got {len(per_pair)} usable pairs")

    # Per-metric trust: mean within-pair Spearman(no-ref, ref_ssim).
    print("\nindividual metric -> mean per-pair Spearman vs GT-SSIM (1.0 = perfect ranking):")
    indiv = {}
    for m in NO_REF:
        cors = [spearman(np.array([r[m] for r in rows]),
                         np.array([r["ref_ssim"] for r in rows])) for rows in per_pair]
        indiv[m] = float(np.mean(cors))
        print(f"  {m:8s} {indiv[m]:+.3f}")

    # Global z-score each metric to combine on a common scale.
    allrows = [r for rows in per_pair for r in rows]
    stats = {m: (np.mean([r[m] for r in allrows]), np.std([r[m] for r in allrows]) + 1e-9)
             for m in NO_REF}

    def z(rows, m):
        mu, sd = stats[m]
        return (np.array([r[m] for r in rows]) - mu) / sd

    # Search nonneg weights on {q_abf, q_ssim, q_mi} (simplex grid) maximizing
    # mean per-pair Spearman vs GT-SSIM. (sharp excluded — it rewards artifacts.)
    combo = ["q_abf", "q_ssim", "q_mi"]
    best = (-1, None)
    for w in itertools.product(np.arange(0, 1.01, 0.1), repeat=len(combo)):
        if abs(sum(w) - 1.0) > 1e-6:
            continue
        cors = []
        for rows in per_pair:
            comp = sum(wi * z(rows, m) for wi, m in zip(w, combo))
            cors.append(spearman(comp, np.array([r["ref_ssim"] for r in rows])))
        score = float(np.mean(cors))
        if score > best[0]:
            best = (score, dict(zip(combo, [float(x) for x in w])))

    print(f"\nbest composite: {best[1]}")
    print(f"  mean per-pair Spearman vs GT-SSIM = {best[0]:+.3f}")
    out = {"weights": best[1], "spearman": best[0], "individual": indiv,
           "zscore_stats": {m: [float(stats[m][0]), float(stats[m][1])] for m in NO_REF},
           "n_pairs": len(per_pair)}
    with open(os.path.join(HERE, "metric_weights.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {os.path.join(HERE, 'metric_weights.json')}")


if __name__ == "__main__":
    main()
