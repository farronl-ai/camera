#!/usr/bin/env python3
"""B4 — validate multi-scale Q_ABF against GT at BOTH resolution regimes.

Success criteria (decided before running):
  - high-res: q_abf_ms Spearman vs GT-SSIM should recover from plain q_abf's
    collapse (+0.12) toward q_ssim's level (+0.87);
  - low-res: must not regress below plain q_abf (~+0.30).
Also test composites (0.3*abf_variant + 0.7*q_ssim, z-scored per stack is
unnecessary for rank corr within a stack since weights are fixed — we use raw).

Run:  python research/metric_ms.py [n_lowres]
"""
from __future__ import annotations
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from validate_metrics import candidates, spearman  # noqa: E402
from hires_eval import load_stack, manifest  # noqa: E402
from focusstack.fusion import fuse_blend, fuse_perband, fuse_pyramid, fuse_max  # noqa: E402
from focusstack.focus import focus_measures  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RMFF = os.path.join(HERE, "data", "realmff", "extracted", "RealMFF")

VARIANTS = {
    "q_abf": lambda s, f: M.q_abf(s, f),
    "abf_ms_mean": lambda s, f: M.q_abf_ms(s, f, pool="mean"),
    "abf_ms_sum": lambda s, f: M.q_abf_ms(s, f, pool="sum"),
    "q_ssim": lambda s, f: M.q_ssim(s, f),
}


def eval_regime(stacks, label):
    """stacks: list of (sources, {cand_name: fused}, gt). Per-stack Spearman."""
    cors = {v: [] for v in VARIANTS}
    cors["comp_ms"] = []
    cors["comp_old"] = []
    for sources, cands, gt in stacks:
        gtssim = np.array([M.ref_ssim(c, gt) for c in cands.values()])
        vals = {v: np.array([fn(sources, c) for c in cands.values()])
                for v, fn in VARIANTS.items()}
        for v in VARIANTS:
            cors[v].append(spearman(vals[v], gtssim))
        cors["comp_ms"].append(spearman(0.3 * vals["abf_ms_mean"] + 0.7 * vals["q_ssim"], gtssim))
        cors["comp_old"].append(spearman(0.3 * vals["q_abf"] + 0.7 * vals["q_ssim"], gtssim))
    print(f"\n{label} (mean per-stack Spearman vs GT-SSIM, {len(stacks)} stacks):")
    for k, v in cors.items():
        print(f"  {k:12s} {np.mean(v):+.3f}")


def lowres_stacks(n):
    ids = sorted(os.path.basename(p)[:3] for p in glob.glob(os.path.join(RMFF, "imageA", "*_A.png")))
    rng = np.random.default_rng(1)
    out = []
    for i in rng.permutation(ids)[:n]:
        a = cv2.imread(os.path.join(RMFF, "imageA", f"{i}_A.png"))
        b = cv2.imread(os.path.join(RMFF, "imageB", f"{i}_B.png"))
        gt = cv2.imread(os.path.join(RMFF, "Fusion", f"{i}_F.png"))
        if a is None or b is None or gt is None:
            continue
        out.append(([a, b], candidates(a, b), gt))
    return out


def hires_stacks():
    out = []
    for m in manifest():
        frames, gt = load_stack(m["id"])
        fm, _ = fuse_max(frames, focus_measures(frames))
        cands = {
            "perband": fuse_perband(frames, harden=0.5),
            "blend": fuse_blend(frames, harden=0.5),
            "pyramid": fuse_pyramid(frames),
            "max": fm,
            "avg": np.mean(np.stack(frames).astype(np.float32), 0).astype(np.uint8),
            "src0": frames[0],
        }
        out.append((frames, cands, gt))
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    eval_regime(lowres_stacks(n), "LOW-RES Real-MFF")
    eval_regime(hires_stacks(), "HIGH-RES fine-depth")
