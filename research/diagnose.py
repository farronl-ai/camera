#!/usr/bin/env python3
"""Visual + conceptual diagnostics — where does adaptive help/hurt, and why?

Metrics guide but don't decide. This finds the Real-MFF pairs where region-
adaptive most helps and most hurts vs the baseline (by true GT-SSIM), renders
them for direct visual inspection, and reports which tune preset dominates each
scene (a scene-type signal). We then LOOK and reason about scene-dependence and
failure modes rather than trusting the aggregate.

Run:  python research/inspect.py [n_scan] [k_each]
Writes research/inspect/{help,hurt}_NN.png montages.
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
from regions import fuse_adaptive, PRESETS  # noqa: E402
from focusstack.fusion import fuse_blend  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RMFF = os.path.join(HERE, "data", "realmff", "extracted", "RealMFF")
OUT = os.path.join(HERE, "inspect")
os.makedirs(OUT, exist_ok=True)


def triples():
    ids = sorted(os.path.basename(p)[:3] for p in glob.glob(os.path.join(RMFF, "imageA", "*_A.png")))
    return [(i, os.path.join(RMFF, "imageA", f"{i}_A.png"),
            os.path.join(RMFF, "imageB", f"{i}_B.png"),
            os.path.join(RMFF, "Fusion", f"{i}_F.png")) for i in ids]


def _scan_one(t):
    idx, pa, pb, pf = t
    a, b, gt = cv2.imread(pa), cv2.imread(pb), cv2.imread(pf)
    if a is None or b is None or gt is None:
        return None
    base = fuse_blend([a, b])
    adapt, dbg = fuse_adaptive([a, b], return_debug=True)
    win = dbg["winner"]
    dom = int(np.bincount(win.ravel(), minlength=len(PRESETS)).argmax())
    return {"idx": idx, "base_ssim": M.ref_ssim(base, gt), "adapt_ssim": M.ref_ssim(adapt, gt),
            "dominant_preset": dom,
            "preset_share": [float((win == p).mean()) for p in range(len(PRESETS))]}


def _montage(idx):
    a = cv2.imread(os.path.join(RMFF, "imageA", f"{idx}_A.png"))
    b = cv2.imread(os.path.join(RMFF, "imageB", f"{idx}_B.png"))
    gt = cv2.imread(os.path.join(RMFF, "Fusion", f"{idx}_F.png"))
    base = fuse_blend([a, b])
    adapt, dbg = fuse_adaptive([a, b], return_debug=True)
    win = dbg["winner"]
    wvis = cv2.applyColorMap((win * (255 // max(1, len(PRESETS) - 1))).astype(np.uint8), cv2.COLORMAP_JET)
    # SSIM-difference heatmap (adapt better = warm) to point the eye at changes
    from metrics import _ssim_map, _gray32
    d = _ssim_map(_gray32(adapt), _gray32(gt)) - _ssim_map(_gray32(base), _gray32(gt))
    dn = np.clip((d - d.min()) / (np.ptp(d) + 1e-9) * 255, 0, 255).astype(np.uint8)
    dvis = cv2.applyColorMap(dn, cv2.COLORMAP_JET)
    row = np.hstack([a, gt, base, adapt, wvis, dvis])
    return row


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    ts = triples()[:n]
    rows = []
    with cf.ProcessPoolExecutor(max_workers=min(16, os.cpu_count())) as ex:
        for r in ex.map(_scan_one, ts):
            if r:
                rows.append(r)
    for r in rows:
        r["delta"] = r["adapt_ssim"] - r["base_ssim"]
    rows.sort(key=lambda r: r["delta"])
    hurt, help_ = rows[:k], rows[-k:][::-1]

    print(f"scanned {len(rows)} pairs. mean delta(adapt-base) SSIM = {np.mean([r['delta'] for r in rows]):+.5f}")
    print(f"adaptive better on {sum(r['delta']>0 for r in rows)}/{len(rows)} pairs")
    print("\n-- adaptive HELPS most --")
    for r in help_:
        print(f"  {r['idx']}  delta={r['delta']:+.4f}  dom_preset={r['dominant_preset']}  share={[round(x,2) for x in r['preset_share']]}")
    print("-- adaptive HURTS most --")
    for r in hurt:
        print(f"  {r['idx']}  delta={r['delta']:+.4f}  dom_preset={r['dominant_preset']}  share={[round(x,2) for x in r['preset_share']]}")

    for tag, group in [("help", help_), ("hurt", hurt)]:
        for r in group:
            cv2.imwrite(os.path.join(OUT, f"{tag}_{r['idx']}.png"), _montage(r["idx"]))
    print(f"\nwrote montages to {OUT}  [ srcA | GT | baseline | adaptive | winner-map | SSIM-diff ]")

    # preset dominance histogram across scenes (scene-type signal)
    hist = np.bincount([r["dominant_preset"] for r in rows], minlength=len(PRESETS))
    print(f"\ndominant-preset histogram across scenes: {list(hist)}  (presets: {[p['focus_method'] for p in PRESETS]})")
    json.dump({"rows": rows}, open(os.path.join(OUT, "scan.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
