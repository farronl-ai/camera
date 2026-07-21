#!/usr/bin/env python3
"""16e T1 — mask matting: the occluder matte = the SAM mask selected by focus
seeds and ordered by semantic depth (objects from masks; F42's fix).

Selection per candidate (frame k, mask m):
  purity  = decisive-k seeds inside m / all decisive seeds inside m
  areafit = seeds_k inside m / area(m)          (mask not vastly larger than seeds)
  near    = median DA-V2 depth inside m > outside (occluder must be nearer)
score = purity * sqrt(areafit), masks failing `near` are dropped. Winning mask
(plus same-owner masks with purity>0.8) -> guided-snap -> alpha.

Run:  python research/maskmatte.py     (after bridge_masks + bridge_depth on pass1)
"""
from __future__ import annotations
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from wideocc_gen import scenes  # noqa: E402
from semalpha import OUT, build_D  # noqa: E402
from veilband import fringe_mask, fuse_perband_weighted_corr  # noqa: E402
from focusstack.fusion import fuse_perband, guided_filter  # noqa: E402
from focusstack.focus import content_aware_energies  # noqa: E402
from focusstack.io import to_gray_float  # noqa: E402


def mask_matte(masks, depth, frames):
    """Returns (alpha, owner) or (zeros, 0) when nothing qualifies."""
    h, w = frames[0].shape[:2]
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h))
    d = (depth - depth.min()) / (np.ptp(depth) + 1e-9)
    grays = [to_gray_float(f) for f in frames]
    E = np.stack(content_aware_energies(grays), 0)
    winner = np.argmax(E, 0)
    srt = np.sort(E, axis=0)
    decisive = ((srt[-1] - srt[-2]) / (srt[-1] + 1e-6) > 0.3) & (srt[-1] > np.median(srt[-1]))
    n = len(frames)

    best = (0.0, None, 0)
    scored = []
    for mi, m in enumerate(masks):
        mm = m > 0
        area = float(mm.sum())
        if area < 400:
            continue
        din = float(np.median(d[mm]))
        dout = float(np.median(d[~mm]))
        if din <= dout:                      # occluder must be nearer
            continue
        dec_in = decisive & mm
        tot = float(dec_in.sum())
        if tot < 50:
            continue
        for k in range(n):
            sk = float((dec_in & (winner == k)).sum())
            purity = sk / tot
            areafit = sk / area
            score = purity * np.sqrt(min(1.0, areafit * 4))
            scored.append((score, mi, k, purity))
            if score > best[0]:
                best = (score, mi, k)
    if best[1] is None:
        return np.zeros((h, w), np.float32), 0
    _, mi0, owner = best
    sel = masks[mi0] > 0
    for score, mi, k, purity in scored:      # union same-owner high-purity masks
        if k == owner and purity > 0.8 and mi != mi0:
            sel = sel | (masks[mi] > 0)
    alpha = np.clip(guided_filter(grays[owner] / 255.0, sel.astype(np.float32), 2, 1e-4), 0.0, 1.0)
    return alpha, owner


def main():
    for coc in (0.04, 0.012):
        print(f"\n===== 16e T1 mask matte, CoC {coc} =====")
        rows = []
        for sc in scenes(coc):
            gt, tru_a, max_r = sc["gt"], sc["alpha"], sc["max_r"]
            base_p = os.path.join(OUT, f"{sc['sid']}_pass1.png")
            mp, dp = base_p + ".masks.npy", base_p + ".depth.npy"
            if not (os.path.exists(mp) and os.path.exists(dp)):
                print(f"  {sc['sid']}: bridge outputs missing")
                continue
            masks = np.load(mp)
            a_est, owner = mask_matte(masks, np.load(dp), sc["frames"])
            fr = fringe_mask(tru_a, max_r)
            base = fuse_perband(sc["frames"], harden=0.5)
            aerr = float(np.abs(a_est - tru_a).mean())
            if a_est.max() > 0:
                D = build_D(sc["frames"], a_est, max_r, owner, 1 - owner)
                cor = fuse_perband_weighted_corr(sc["frames"], D, 0, far_idx=1 - owner)
            else:
                cor = base
            e0 = float(np.abs(base.astype(np.float32) - gt.astype(np.float32)).sum(2)[fr].mean())
            ec = float(np.abs(cor.astype(np.float32) - gt.astype(np.float32)).sum(2)[fr].mean())
            g0, gc = M.ref_ssim(base, gt), M.ref_ssim(cor, gt)
            rows.append((e0, ec, g0, gc, aerr))
            print(f"  {sc['sid']:22s} fringe {e0:5.1f}->{ec:5.1f}  glob {g0:.4f}->{gc:.4f}  "
                  f"|a err|={aerr:.3f} owner={owner} nmasks={len(masks)}")
        if rows:
            a = np.array(rows)
            print(f"  MEAN fringe {a[:,0].mean():5.1f}->{a[:,1].mean():5.1f}  "
                  f"glob {a[:,2].mean():.4f}->{a[:,3].mean():.4f}  |a err|={a[:,4].mean():.3f}")


if __name__ == "__main__":
    main()
