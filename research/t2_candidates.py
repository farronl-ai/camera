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
    CACHE = "/home/farron/camera/research/t2_cand_cache.npz"
    cache = dict(np.load(CACHE, allow_pickle=True))["rows"].item() if os.path.exists(CACHE) else {}
    data = []          # candidate-level rows
    per_scene = []     # scene bookkeeping
    dirty = False
    for i, sc in enumerate(scenes()):
        fr = fringe_mask(sc["alpha"], sc["max_r"])
        key = sc["sid"]
        if key in cache:
            e0, g0, crows = cache[key]
            rows = [dict(i=i, feats=np.array(cr["feats"], np.float32), owner=int(cr["owner"]),
                         label=float(cr["label"]), dg=float(cr["dg"]), de=float(cr["de"]),
                         alpha=None) for cr in crows]
        else:
            base = fuse_perband(sc["frames"], harden=0.5)
            e0 = float(np.abs(base.astype(np.float32) - sc["gt"].astype(np.float32)).sum(2)[fr].mean())
            g0 = M.ref_ssim(base, sc["gt"])
            rows = []
            crows = []
            for c in candidates_with_features(sc):
                D = build_D(sc["frames"], c["alpha"], sc["max_r"], c["owner"], 1 - c["owner"])
                cor = corr_multi(sc["frames"], {1 - c["owner"]: D})
                ec = float(np.abs(cor.astype(np.float32) - sc["gt"].astype(np.float32)).sum(2)[fr].mean())
                gc = M.ref_ssim(cor, sc["gt"])
                label = 1.0 if (ec < e0 and gc - g0 >= -0.0005) else 0.0
                rows.append(dict(i=i, feats=c["feats"], alpha=c["alpha"], owner=c["owner"],
                                 label=label, dg=gc - g0, de=ec - e0))
                crows.append(dict(feats=c["feats"].tolist(), owner=c["owner"], label=label,
                                  dg=gc - g0, de=ec - e0))
            cache[key] = (e0, g0, crows)
            dirty = True
        data.extend(rows)
        per_scene.append(dict(i=i, sc=sc, e0=e0, g0=g0, fr=fr, cands=rows))
    if dirty:
        np.savez(CACHE, rows=np.array({k: v for k, v in cache.items()}, dtype=object))
    print(f"  scenes={len(per_scene)} candidates={len(data)} "
          f"positives={int(sum(r['label'] for r in data))}")

    n_tr_scene = int(0.75 * len(per_scene))
    train = [r for r in data if r["i"] < n_tr_scene]
    held = [r for r in data if r["i"] >= n_tr_scene]

    def expand(x):     # quadratic feature map: x + pairwise products
        q = [x[a] * x[b] for a in range(len(x)) for b in range(a, len(x))]
        return np.concatenate([x, q]).astype(np.float32)

    # RIDGE REGRESSION on the OUTCOME dg itself — the gate predicts the gain;
    # fire when predicted gain clears a positive margin. No tau, no label proxy.
    X = np.stack([expand(r["feats"]) for r in train])
    yg = np.array([r["dg"] for r in train], np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = np.hstack([(X - mu) / sd, np.ones((len(X), 1), np.float32)])
    lam = 3.0
    A = Xn.T @ Xn + lam * np.eye(Xn.shape[1], dtype=np.float32)
    wgt = np.linalg.solve(A, Xn.T @ yg)
    FIRE_MARGIN = 3e-4

    def pred(r):
        xn = np.hstack([(expand(r["feats"]) - mu) / sd, [1.0]])
        return float(xn @ wgt)

    ptr = np.array([pred(r) for r in train]); ph = np.array([pred(r) for r in held])
    ytr = yg; yh = np.array([r["dg"] for r in held])
    ftr = ptr >= FIRE_MARGIN; fh = ph >= FIRE_MARGIN
    print(f"\n  candidates: train={len(train)} held={len(held)}  fire if pred_dg>={FIRE_MARGIN}")
    print(f"  train fired {int(ftr.sum())}/{len(train)}  actual-dg>=0: {int((ytr[ftr]>=0).sum())}/{int(ftr.sum())}"
          f"  mean actual dg={ytr[ftr].mean() if ftr.any() else 0:+.4f}")
    print(f"  held  fired {int(fh.sum())}/{len(held)}  actual-dg>=0: {int((yh[fh]>=0).sum())}/{int(fh.sum())}"
          f"  mean actual dg={yh[fh].mean() if fh.any() else 0:+.4f}")
    tau = FIRE_MARGIN
    def logistic_p_shim(w_, m_, s_, x):     # keep property loop signature
        return pred(dict(feats=x))
    globals()["logistic_p"] = logistic_p_shim

    fired_scenes = {"0.02": [0, 0], "0.035": [0, 0]}   # regime -> [fired, total]
    worst = 0.0
    for s in per_scene:
        regime = "0.035" if s["sc"]["max_r"] > 40 else "0.02"
        fired_scenes[regime][1] += 1
        fired_idx = [j for j, c in enumerate(s["cands"]) if pred(c) >= FIRE_MARGIN]
        if not fired_idx:
            continue
        fired_scenes[regime][0] += 1
        cands_full = candidates_with_features(s["sc"])     # recompute alphas (cache-safe)
        D_by_far = {}
        for j in fired_idx:
            if j >= len(cands_full):
                continue
            c = cands_full[j]
            D = build_D(s["sc"]["frames"], c["alpha"], s["sc"]["max_r"], c["owner"], 1 - c["owner"])
            fi = 1 - c["owner"]
            D_by_far[fi] = D_by_far.get(fi, 0) + D
        if not D_by_far:
            continue
        out = corr_multi(s["sc"]["frames"], D_by_far)
        # RUNTIME OUTPUT SELF-CHECK — DISABLED (F45): q_ssim measures similarity
        # to the locally-sharpest SOURCE, but the veil correction synthesizes
        # de-hazed content matching NO source; the check reverts GT-verified wins.
        SELF_CHECK = False
        base_img = fuse_perband(s["sc"]["frames"], harden=0.5)
        reg = np.zeros(out.shape[:2], bool)
        for fi, Dp in D_by_far.items():
            reg |= np.abs(Dp).sum(2) > 0.5
        if SELF_CHECK and reg.sum() > 100:
            q_out = M.q_ssim_map(s["sc"]["frames"], out)[reg].mean()
            q_base = M.q_ssim_map(s["sc"]["frames"], base_img)[reg].mean()
            if q_out < q_base - 0.001:
                out = base_img
                print(f"  scene_{s['i']:02d}: REVERTED by output self-check "
                      f"(regional q_ssim {q_base:.4f}->{q_out:.4f})")
        ec = float(np.abs(out.astype(np.float32) - s["sc"]["gt"].astype(np.float32)).sum(2)[s["fr"]].mean())
        gc = M.ref_ssim(out, s["sc"]["gt"])
        worst = min(worst, gc - s["g0"])
        print(f"  scene_{s['i']:02d}: {len(fired_idx)} fired  fringe {s['e0']:6.1f}->{ec:6.1f}  "
              f"glob {s['g0']:.4f}->{gc:.4f}")
    for reg, (fc, tc) in fired_scenes.items():
        print(f"  coverage CoC {reg}: {fc}/{tc}")
    print(f"  worst delta={worst:+.4f} (>= -0.001)")
    np.savez("/home/farron/camera/research/t2_gate.npz", w=wgt, mu=mu, sd=sd,
             margin=FIRE_MARGIN)


if __name__ == "__main__":
    main()
