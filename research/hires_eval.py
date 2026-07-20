#!/usr/bin/env python3
"""H2 — expert-flow validation at high-res.

1. Re-validate the metric at high-res (do NOT assume it transfers): per-stack
   rank-correlation of no-reference metrics vs true GT-SSIM.
2. Engine comparison by content type: GT-SSIM / PSNR / worst-tile / timing.
Writes crops for the visual failure-mode hunt.

Run:  python research/hires_eval.py [metric|engine|all]
"""
from __future__ import annotations
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from metrics import _ssim_map, _gray32  # noqa: E402
from focusstack.fusion import fuse_blend, fuse_decision, fuse_pyramid, fuse_max  # noqa: E402
from focusstack.focus import focus_measures  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HDIR = os.path.join(HERE, "data", "hires")
OUT = os.path.join(HERE, "hires_out")
os.makedirs(OUT, exist_ok=True)


def load_stack(sid):
    d = os.path.join(HDIR, sid)
    gt = cv2.imread(os.path.join(d, "gt.png"))
    frames = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(d, "frame_*.png")))]
    return frames, gt


def manifest():
    return json.load(open(os.path.join(HDIR, "manifest.json")))


def worst_tile(fused, gt, t=8):
    s = _ssim_map(_gray32(fused), _gray32(gt))
    h, w = s.shape
    return min(s[i*(h//t):(i+1)*(h//t), j*(w//t):(j+1)*(w//t)].mean()
              for i in range(t) for j in range(t))


def spearman(x, y):
    def rk(v):
        o = v.argsort(); r = np.empty_like(o, float); r[o] = np.arange(len(v)); return r
    rx, ry = rk(x), rk(y); rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx*rx).sum()*(ry*ry).sum())
    return float((rx*ry).sum()/d) if d > 1e-9 else 0.0


def cmd_metric():
    print("METRIC re-validation at high-res (per-stack Spearman vs GT-SSIM):")
    per_abf, per_ssim, per_comp = [], [], []
    for m in manifest():
        frames, gt = load_stack(m["id"])
        fm, _ = fuse_max(frames, focus_measures(frames))
        cands = {
            "blend": fuse_blend(frames, harden=0.5),
            "decision": fuse_decision(frames, harden=0.5),
            "pyramid": fuse_pyramid(frames),
            "max": fm,
            "avg": np.mean(np.stack(frames).astype(np.float32), 0).astype(np.uint8),
            "srcA": frames[0],
        }
        gtssim = np.array([M.ref_ssim(c, gt) for c in cands.values()])
        abf = np.array([M.q_abf(frames, c) for c in cands.values()])
        ssim = np.array([M.q_ssim(frames, c) for c in cands.values()])
        comp = np.array([M.composite(frames, c) for c in cands.values()])
        per_abf.append(spearman(abf, gtssim)); per_ssim.append(spearman(ssim, gtssim))
        per_comp.append(spearman(comp, gtssim))
    print(f"  q_abf     {np.mean(per_abf):+.3f}")
    print(f"  q_ssim    {np.mean(per_ssim):+.3f}")
    print(f"  composite {np.mean(per_comp):+.3f}   (low-res calibration was +0.72 vs GT)")


def cmd_engine():
    variants = {
        "baseline(lap,h0)": lambda fr: fuse_blend(fr, focus_method="laplacian", harden=0.0),
        "content_aware": lambda fr: fuse_blend(fr, focus_method="content_aware", harden=0.0),
        "+harden0.5": lambda fr: fuse_blend(fr, focus_method="content_aware", harden=0.5),
        "--fast": lambda fr: fuse_decision(fr, focus_method="content_aware", harden=0.5, weight_scale=0.5),
        "pyramid": lambda fr: fuse_pyramid(fr),
    }
    rows = {}
    print(f"{'id':18s} {'ctype':14s} " + " ".join(f"{v:>16s}" for v in variants))
    for m in manifest():
        frames, gt = load_stack(m["id"])
        res = {}
        for name, fn in variants.items():
            t0 = time.time(); fused = fn(frames); dt = time.time()-t0
            res[name] = (M.ref_ssim(fused, gt), M.ref_psnr(fused, gt), worst_tile(fused, gt), dt)
        rows[m["id"]] = (m["content_type"], res)
        print(f"{m['id']:18s} {m['content_type']:14s} " +
              " ".join(f"{res[v][0]:.4f}" .rjust(16) for v in variants))
    # aggregate by content type + overall (GT-SSIM, worst-tile, time)
    print("\nmean GT-SSIM / worst-tile / ms by variant (overall):")
    for v in variants:
        ss = np.mean([r[1][v][0] for r in rows.values()])
        wt = np.mean([r[1][v][2] for r in rows.values()])
        ms = 1000*np.mean([r[1][v][3] for r in rows.values()])
        print(f"  {v:18s} SSIM={ss:.4f}  worst={wt:.4f}  {ms:6.0f} ms")
    json.dump({k: {"ctype": v[0], "res": {n: [float(x) for x in t] for n, t in v[1].items()}}
               for k, v in rows.items()}, open(os.path.join(OUT, "engine_hires.json"), "w"), indent=2)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("metric", "all"):
        cmd_metric()
    if cmd in ("engine", "all"):
        cmd_engine()
