#!/usr/bin/env python3
"""16e T2v2 — PER-CANDIDATE outcome-trained routing (true per-region firing).

Per image: top-K (mask, owner) candidates survive the physics filters
(near-depth, ring-contrast); each gets features + its own outcome label
(apply ONLY that candidate's fringe-clamped D̂ -> did it help). The gate fires
candidates independently; fired candidates' corrections compose (grouped by
far-frame index, D̂ fields summed — fringe-clamped and mostly disjoint).

Fire rate becomes per-component; an image only goes identity if NO candidate
is confident. Run:  python research/t2_candidates.py
"""
from __future__ import annotations
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from semalpha import build_D  # noqa: E402
from veilband import fringe_mask  # noqa: E402
from t2_confidence import scenes, logistic_fit, logistic_p  # noqa: E402
from focusstack.fusion import (fuse_perband, guided_filter, _auto_levels,  # noqa: E402
                               _laplacian_pyramid, _gaussian_pyramid)
from focusstack.focus import content_aware_energies  # noqa: E402
from focusstack.io import to_gray_float  # noqa: E402


def corr_multi(images, D_by_far, radius=6, eps=1e-3, energy_ksize=7, harden=0.5):
    """perband with multi-candidate veil correction: subtract sum_f w[f]*L(D_f)."""
    floats = [img.astype(np.float32) for img in images]
    n = len(floats)
    levels = _auto_levels(floats[0].shape, None)
    ip = [_laplacian_pyramid(im, levels) for im in floats]
    gp = [_gaussian_pyramid(to_gray_float(f), levels) for f in images]
    dp = {f: _laplacian_pyramid(D.astype(np.float32), levels) for f, D in D_by_far.items()}
    fused_bands, w_last = [], None
    for band in range(levels + 1):
        coeffs = [ip[k][band] for k in range(n)]
        bh, bw = coeffs[0].shape[:2]
        if band < levels:
            r_b = max(1, min(radius, min(bh, bw) // 6))
            k_b = max(3, min(energy_ksize, (min(bh, bw) // 4) | 1))
            energy = np.stack([cv2.boxFilter((coeffs[k] ** 2).sum(axis=2), cv2.CV_32F, (k_b, k_b))
                               for k in range(n)], axis=0)
            winner = np.argmax(energy, axis=0)
            srt = np.sort(energy, axis=0)
            conf = np.clip((srt[-1] - srt[-2]) / (srt[-1] + 1e-6), 0.0, 1.0) if harden > 0 else None
            weights = []
            for k in range(n):
                raw = (winner == k).astype(np.float32)
                wg = np.clip(guided_filter(gp[k][band] / 255.0, raw, r_b, eps), 0.0, None)
                if conf is not None:
                    wg = (1.0 - conf) * wg + conf * raw
                weights.append(wg)
            w = np.stack(weights, axis=0)
            w /= (w.sum(axis=0, keepdims=True) + 1e-8)
            fb = sum(w[k][..., None] * coeffs[k] for k in range(n))
            for fidx, pyr in dp.items():
                fb = fb - w[fidx][..., None] * pyr[band]
            fused_bands.append(fb)
            w_last = w
        else:
            wb = np.stack([cv2.pyrDown(w_last[k]) for k in range(n)], axis=0)
            wb = np.clip(wb, 0.0, None)
            wb /= (wb.sum(axis=0, keepdims=True) + 1e-8)
            fb = sum(wb[k][..., None] * coeffs[k] for k in range(n))
            for fidx, pyr in dp.items():
                fb = fb - wb[fidx][..., None] * pyr[levels]
            fused_bands.append(fb)
    result = fused_bands[-1]
    for band in range(levels - 1, -1, -1):
        result = cv2.pyrUp(result, dstsize=(fused_bands[band].shape[1], fused_bands[band].shape[0])) + fused_bands[band]
    return np.clip(result, 0, 255).astype(np.uint8)


def candidates_with_features(sc, topk=4):
    p = os.path.join(sc["dir"], "pass1.png")
    masks, depth = np.load(p + ".masks.npy"), np.load(p + ".depth.npy")
    frames = sc["frames"]
    h, w = frames[0].shape[:2]
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h))
    d = (depth - depth.min()) / (np.ptp(depth) + 1e-9)
    grays = [to_gray_float(f) for f in frames]
    E = np.stack(content_aware_energies(grays), 0)
    winner = np.argmax(E, 0)
    srt = np.sort(E, axis=0)
    decisive = ((srt[-1] - srt[-2]) / (srt[-1] + 1e-6) > 0.3) & (srt[-1] > np.median(srt[-1]))

    cands = {}
    for mi, m in enumerate(masks):
        mm = m > 0
        area = float(mm.sum())
        if area < 400:
            continue
        din, dout = float(np.median(d[mm])), float(np.median(d[~mm]))
        if din <= dout:
            continue
        dec_in = decisive & mm
        tot = float(dec_in.sum())
        if tot < 50:
            continue
        ring = (cv2.dilate(mm.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0) & ~mm
        dec_ring = decisive & ring
        rtot = float(dec_ring.sum())
        for k in range(len(frames)):
            sk = float((dec_in & (winner == k)).sum())
            purity = sk / tot
            areafit = sk / area
            ring_other = (float((dec_ring & (winner != k)).sum()) / rtot) if rtot > 50 else 0.0
            if ring_other < 0.5:
                continue
            score = purity * np.sqrt(min(1.0, areafit * 4)) * ring_other
            if mi not in cands or score > cands[mi]["score"]:
                cands[mi] = dict(score=score, purity=purity, ring_other=ring_other,
                                 areafit=min(1.0, areafit * 4), margin=din - dout,
                                 areafrac=area / (h * w), mi=mi)
    out = []
    for c in sorted(cands.values(), key=lambda r: -r["score"])[:topk]:
        sel = masks[c["mi"]] > 0
        interior = cv2.erode(sel.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        dec_int = decisive & interior
        owner = int(np.bincount(winner[dec_int].ravel(), minlength=len(frames)).argmax()) \
            if dec_int.sum() > 20 else 0
        alpha = np.clip(guided_filter(grays[owner] / 255.0, sel.astype(np.float32), 2, 1e-4), 0.0, 1.0)
        snapped = alpha > 0.5
        iou = float((snapped & sel).sum()) / (float((snapped | sel).sum()) + 1e-6)
        feats = np.array([c["score"], c["purity"], c["ring_other"], c["areafit"],
                          c["margin"], c["areafrac"], iou], np.float32)
        out.append(dict(feats=feats, alpha=alpha, owner=owner))
    return out


def main():
    data = []          # candidate-level rows
    per_scene = []     # scene bookkeeping
    for i, sc in enumerate(scenes()):
        base = fuse_perband(sc["frames"], harden=0.5)
        fr = fringe_mask(sc["alpha"], sc["max_r"])
        e0 = float(np.abs(base.astype(np.float32) - sc["gt"].astype(np.float32)).sum(2)[fr].mean())
        g0 = M.ref_ssim(base, sc["gt"])
        cands = candidates_with_features(sc)
        rows = []
        for c in cands:
            D = build_D(sc["frames"], c["alpha"], sc["max_r"], c["owner"], 1 - c["owner"])
            cor = corr_multi(sc["frames"], {1 - c["owner"]: D})
            ec = float(np.abs(cor.astype(np.float32) - sc["gt"].astype(np.float32)).sum(2)[fr].mean())
            gc = M.ref_ssim(cor, sc["gt"])
            label = 1.0 if (ec < e0 and gc - g0 >= -0.0005) else 0.0
            row = dict(i=i, feats=c["feats"], alpha=c["alpha"], owner=c["owner"], label=label)
            rows.append(row)
            data.append(row)
        per_scene.append(dict(i=i, sc=sc, base=base, e0=e0, g0=g0, fr=fr, cands=rows))
        print(f"  scene_{i:02d}: {len(rows)} candidates, labels={[int(r['label']) for r in rows]}")

    n_tr_scene = int(0.75 * len(per_scene))
    train = [r for r in data if r["i"] < n_tr_scene]
    held = [r for r in data if r["i"] >= n_tr_scene]
    X = np.stack([r["feats"] for r in train])
    y = np.array([r["label"] for r in train])
    wgt, mu, sd = logistic_fit(X, y)
    ps = np.array([logistic_p(wgt, mu, sd, r["feats"]) for r in train])
    bad_p = ps[y == 0]
    tau = float(min(1.0, (bad_p.max() if len(bad_p) else 0.0) + 0.10))
    ph = np.array([logistic_p(wgt, mu, sd, r["feats"]) for r in held])
    yh = np.array([r["label"] for r in held])
    fh = ph >= tau
    print(f"\n  candidates: train={len(train)} held={len(held)}  tau={tau:.3f}")
    print(f"  train fired {int((ps>=tau).sum())}/{len(train)} all-good={bool((y[ps>=tau]==1).all())}")
    print(f"  held  fired {int(fh.sum())}/{len(held)}  good={int(yh[fh].sum())}/{int(fh.sum())}")

    fired_scenes = 0
    worst = 0.0
    for s in per_scene:
        fired = [c for c in s["cands"] if logistic_p(wgt, mu, sd, c["feats"]) >= tau]
        if not fired:
            continue
        fired_scenes += 1
        D_by_far = {}
        for c in fired:
            D = build_D(s["sc"]["frames"], c["alpha"], s["sc"]["max_r"], c["owner"], 1 - c["owner"])
            f = 1 - c["owner"]
            D_by_far[f] = D_by_far.get(f, 0) + D
        out = corr_multi(s["sc"]["frames"], D_by_far)
        ec = float(np.abs(out.astype(np.float32) - s["sc"]["gt"].astype(np.float32)).sum(2)[s["fr"]].mean())
        gc = M.ref_ssim(out, s["sc"]["gt"])
        worst = min(worst, gc - s["g0"])
        print(f"  scene_{s['i']:02d}: {len(fired)} fired  fringe {s['e0']:6.1f}->{ec:6.1f}  "
              f"glob {s['g0']:.4f}->{gc:.4f}")
    print(f"\n  scene coverage: {fired_scenes}/{len(per_scene)}  worst delta={worst:+.4f} (>= -0.001)")
    np.savez("/home/farron/camera/research/t2_gate.npz", w=wgt, mu=mu, sd=sd, tau=tau)


if __name__ == "__main__":
    main()
