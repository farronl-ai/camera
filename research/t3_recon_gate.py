#!/usr/bin/env python3
"""16c-gate — outcome-trained gate for the RECONSTRUCTION specialist.

Same pattern that locked the veil gate (F44/F45): per-candidate (mask, owner)
from the objocc factory; label = the ACTUAL outcome of applying contour
reconstruction with that candidate's matte (contour-band error + global GT-SSIM
deltas); ridge regression predicts the gain; fire above margin. Reconstruction
is post-fusion, so composition = sequential application ordered by score.

Run:  python research/t3_recon_gate.py
"""
from __future__ import annotations
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from bband import bband  # noqa: E402
from reconstruct import reconstruct_band, contamination_band  # noqa: E402
from t2_confidence import scenes  # noqa: E402
from t2_candidates import candidates_with_features  # noqa: E402
from focusstack.fusion import fuse_perband  # noqa: E402

CACHE = "/home/farron/camera/research/t3_recon_cache.npz"
TOL = 5e-4          # global tolerance in labels
FIRE_MARGIN = 3e-4  # predicted-gain threshold


def gt_contour_band(alpha):
    m = (alpha > 0.5).astype(np.uint8)
    return (cv2.dilate(m, np.ones((3, 3), np.uint8)) != cv2.erode(m, np.ones((3, 3), np.uint8))).astype(np.uint8)


def expand(x):
    q = [x[a] * x[b] for a in range(len(x)) for b in range(a, len(x))]
    return np.concatenate([x, q]).astype(np.float32)


def main():
    cache = dict(np.load(CACHE, allow_pickle=True))["rows"].item() if os.path.exists(CACHE) else {}
    data, per_scene, dirty = [], [], False
    for i, sc in enumerate(scenes()):
        key = sc["sid"]
        gt_b = gt_contour_band(sc["alpha"])
        if key in cache:
            e0, g0, crows = cache[key]
            rows = [dict(i=i, feats=np.array(cr["feats"], np.float32), owner=int(cr["owner"]),
                         score=float(cr["score"]), dg=float(cr["dg"]), de=float(cr["de"]))
                    for cr in crows]
        else:
            base = fuse_perband(sc["frames"], harden=0.5)
            e0 = bband(base, sc["gt"], gt_b, 2)[0]
            g0 = M.ref_ssim(base, sc["gt"])
            rows, crows = [], []
            for c in candidates_with_features(sc):
                fr_o = [sc["frames"][c["owner"]], sc["frames"][1 - c["owner"]]]
                band = contamination_band(c["alpha"], sc["max_r"])
                rec = reconstruct_band(fr_o, c["alpha"], band, base, sc["max_r"])
                de = bband(rec, sc["gt"], gt_b, 2)[0] - e0
                dg = M.ref_ssim(rec, sc["gt"]) - g0
                rows.append(dict(i=i, feats=c["feats"], owner=c["owner"],
                                 score=float(c["feats"][0]), dg=dg, de=de))
                crows.append(dict(feats=c["feats"].tolist(), owner=c["owner"],
                                  score=float(c["feats"][0]), dg=dg, de=de))
            cache[key] = (e0, g0, crows)
            dirty = True
            np.savez(CACHE, rows=np.array({k: v for k, v in cache.items()}, dtype=object))
            print(f"  labeled {key} ({len(crows)} candidates)", flush=True)
        data.extend(rows)
        per_scene.append(dict(i=i, sc=sc, e0=e0, g0=g0, gt_b=gt_b, cands=rows))
    if dirty:
        np.savez(CACHE, rows=np.array({k: v for k, v in cache.items()}, dtype=object))
    print(f"  scenes={len(per_scene)} candidates={len(data)}  "
          f"helpful(de<0 & dg>-{TOL}): {sum(1 for r in data if r['de'] < 0 and r['dg'] > -TOL)}")

    n_tr = int(0.75 * len(per_scene))
    train = [r for r in data if r["i"] < n_tr]
    held = [r for r in data if r["i"] >= n_tr]
    X = np.stack([expand(r["feats"]) for r in train])
    # combined objective: predicted global delta, with contour gain as tiebreak —
    # regress dg (property currency); fire needs pred_dg >= margin
    yg = np.array([r["dg"] for r in train], np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = np.hstack([(X - mu) / sd, np.ones((len(X), 1), np.float32)])
    A = Xn.T @ Xn + 3.0 * np.eye(Xn.shape[1], dtype=np.float32)
    wgt = np.linalg.solve(A, Xn.T @ yg)

    def pred(r):
        xn = np.hstack([(expand(r["feats"]) - mu) / sd, [1.0]])
        return float(xn @ wgt)

    for name, split in (("train", train), ("held", held)):
        pp = np.array([pred(r) for r in split])
        yy = np.array([r["dg"] for r in split])
        f = pp >= FIRE_MARGIN
        print(f"  {name}: fired {int(f.sum())}/{len(split)}  actual-dg>=0: "
              f"{int((yy[f] >= 0).sum())}/{int(f.sum()) if f.any() else 0}  "
              f"mean dg={yy[f].mean() if f.any() else 0:+.4f}")

    print(f"\n  {'scene':9s} n_fired  e2 base->rec   glob base->rec")
    cov = {"0.02": [0, 0], "0.035": [0, 0]}
    worst = 0.0
    for s in per_scene:
        regime = "0.035" if s["sc"]["max_r"] > 40 else "0.02"
        cov[regime][1] += 1
        fired_idx = [j for j, c in enumerate(s["cands"]) if pred(c) >= FIRE_MARGIN]
        if not fired_idx:
            continue
        cov[regime][0] += 1
        cands_full = candidates_with_features(s["sc"])
        out = fuse_perband(s["sc"]["frames"], harden=0.5)
        for j in sorted(fired_idx, key=lambda j: -s["cands"][j]["score"]):
            if j >= len(cands_full):
                continue
            c = cands_full[j]
            fr_o = [s["sc"]["frames"][c["owner"]], s["sc"]["frames"][1 - c["owner"]]]
            band = contamination_band(c["alpha"], s["sc"]["max_r"])
            out = reconstruct_band(fr_o, c["alpha"], band, out, s["sc"]["max_r"])
        ec = bband(out, s["sc"]["gt"], s["gt_b"], 2)[0]
        gc = M.ref_ssim(out, s["sc"]["gt"])
        worst = min(worst, gc - s["g0"])
        print(f"  scene_{s['i']:02d}  {len(fired_idx)}      {s['e0']:6.1f}->{ec:6.1f}   "
              f"{s['g0']:.4f}->{gc:.4f}")
    for reg, (fc, tc) in cov.items():
        print(f"  coverage CoC {reg}: {fc}/{tc}")
    print(f"  worst delta={worst:+.4f} (>= -0.001)")
    np.savez("/home/farron/camera/research/t3_recon_gate.npz", w=wgt, mu=mu, sd=sd,
             margin=FIRE_MARGIN)


if __name__ == "__main__":
    main()
