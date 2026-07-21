#!/usr/bin/env python3
"""16c-gate v2 — reconstruction gate on its HOME regime: thin structures + C3 mattes.

Factory: occ_gen-style thin-structure scenes (curves + dots over real photos) at
scale (many seeds — no bridges needed; the C3 difference-matte is classical).
Per scene: C3 matte (estimate_alpha_v3) -> reconstruct_band -> outcome labels
(contour bband e2 + global GT-SSIM deltas) -> ridge gate on features drawn from
the C3 pipeline internals -> fire margin -> every-scene property.

Run:  python research/thinocc_gate.py [n_scenes]
"""
from __future__ import annotations
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from bband import bband  # noqa: E402
from occ_gen import near_layer, occ_defocus, LONG  # noqa: E402
from hires_gen import add_noise  # noqa: E402
from reconstruct import reconstruct_band, contamination_band, estimate_alpha_v3  # noqa: E402
from focusstack.fusion import fuse_perband  # noqa: E402
from focusstack.focus import content_aware_energies  # noqa: E402
from focusstack.io import to_gray_float  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "thinocc_cache.npz")
FIRE_MARGIN = 3e-4


def thin_scenes(n):
    photos = sorted(glob.glob(os.path.join(HERE, "data", "hires", "*", "gt.png")))
    out = []
    for i in range(n):
        gp = photos[i % len(photos)]
        far = cv2.imread(gp)
        h, w = far.shape[:2]
        s = LONG / max(h, w)
        far = cv2.resize(far, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        near, alpha = near_layer(far, seed=2000 + i)
        gt = (near * alpha[..., None] + far.astype(np.float32) * (1 - alpha[..., None])).astype(np.uint8)
        max_r = 0.012 * max(gt.shape[:2])
        frames = [add_noise(occ_defocus(far, near, alpha, f, 0.15, 0.85, max_r), 3.0, 7 * i + k)
                  for k, f in enumerate([0.15, 0.85])]
        out.append(dict(sid=f"thin_{i:02d}", gt=gt, alpha=alpha, frames=frames, max_r=max_r))
    return out


def c3_features(frames, a_est, max_r):
    """Features from the C3 matte + stack internals (thin-regime analogues)."""
    grays = [to_gray_float(f) for f in frames]
    E = np.stack(content_aware_energies(grays), 0)
    srt = np.sort(E, axis=0)
    dom = (srt[-1] - srt[-2]) / (srt[-1] + 1e-6)
    sup = a_est > 0.15
    if not sup.any():
        return None
    supf = float(sup.mean())
    dom_in = float(dom[sup].mean())
    a_hi = float((a_est > 0.6).sum()) / (float(sup.sum()) + 1e-6)   # matte hardness
    grad = cv2.magnitude(cv2.Scharr(grays[0], cv2.CV_32F, 1, 0), cv2.Scharr(grays[0], cv2.CV_32F, 0, 1))
    edge_in = float(grad[sup].mean()) / (float(grad.mean()) + 1e-6)  # owner-frame edge energy in support
    band = contamination_band(a_est, max_r)
    bandf = float(band.mean())
    # matte-edge quality (F46: THE discriminator for edge-stamping safety):
    # (a) transition-shell sharpness — crisp mattes have a narrow, high-gradient
    #     transition; (b) silhouette-on-edge alignment — the alpha contour should
    #     sit ON owner-frame edges
    shell = (a_est > 0.2) & (a_est < 0.8)
    ga = cv2.magnitude(cv2.Scharr(a_est, cv2.CV_32F, 1, 0), cv2.Scharr(a_est, cv2.CV_32F, 0, 1))
    sharpness = float(ga[shell].mean()) if shell.any() else 0.0
    align = float(grad[shell].mean()) / (float(grad.mean()) + 1e-6) if shell.any() else 0.0
    return np.array([supf, dom_in, a_hi, edge_in, bandf, sharpness, align], np.float32)


def expand(x):
    q = [x[a] * x[b] for a in range(len(x)) for b in range(a, len(x))]
    return np.concatenate([x, q]).astype(np.float32)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    cache = dict(np.load(CACHE, allow_pickle=True))["rows"].item() if os.path.exists(CACHE) else {}
    rows = []
    for sc in thin_scenes(n):
        key = sc["sid"]
        # cache holds LABELS only (feature-set independent); features recompute fresh
        if key not in cache:
            m = (sc["alpha"] > 0.5).astype(np.uint8)
            gt_b = (cv2.dilate(m, np.ones((3, 3), np.uint8)) != cv2.erode(m, np.ones((3, 3), np.uint8))).astype(np.uint8)
            base = fuse_perband(sc["frames"], harden=0.5)
            e0 = bband(base, sc["gt"], gt_b, 2)[0]
            g0 = M.ref_ssim(base, sc["gt"])
            a3, owner = estimate_alpha_v3(sc["frames"], sc["max_r"])
            if a3.max() <= 0:
                cache[key] = None
            else:
                fr_o = [sc["frames"][owner], sc["frames"][1 - owner]]
                rec = reconstruct_band(fr_o, a3, contamination_band(a3, sc["max_r"]), base, sc["max_r"])
                de = bband(rec, sc["gt"], gt_b, 2)[0] - e0
                dg = M.ref_ssim(rec, sc["gt"]) - g0
                cache[key] = dict(de=de, dg=dg, e0=e0, g0=g0)
            np.savez(CACHE, rows=np.array({k: v for k, v in cache.items()}, dtype=object))
            print(f"  labeled {key}: {cache[key] if cache[key] is None else (round(cache[key]['de'],2), round(cache[key]['dg'],4))}", flush=True)
        if cache[key] is not None:
            r = cache[key]
            a3, _ = estimate_alpha_v3(sc["frames"], sc["max_r"])
            feats = c3_features(sc["frames"], a3, sc["max_r"]) if a3.max() > 0 else None
            if feats is not None:
                rows.append(dict(sid=key, feats=feats, de=r["de"], dg=r["dg"]))
    print(f"\n  scenes with matte: {len(rows)}/{n}  "
          f"helpful(de<0 & dg>=-5e-4): {sum(1 for r in rows if r['de'] < 0 and r['dg'] >= -5e-4)}")

    n_tr = int(0.75 * len(rows))
    train, held = rows[:n_tr], rows[n_tr:]
    X = np.stack([expand(r["feats"]) for r in train])
    yg = np.array([r["dg"] for r in train], np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = np.hstack([(X - mu) / sd, np.ones((len(X), 1), np.float32)])
    wgt = np.linalg.solve(Xn.T @ Xn + 3.0 * np.eye(Xn.shape[1], dtype=np.float32), Xn.T @ yg)

    def pred(r):
        xn = np.hstack([(expand(r["feats"]) - mu) / sd, [1.0]])
        return float(xn @ wgt)

    # PROPERTY-DRIVEN margin: smallest train margin at which every train fire
    # has dg >= -0.001 (chosen on train only; verified on held)
    ptr = np.array([pred(r) for r in train])
    dtr = np.array([r["dg"] for r in train])
    margin = FIRE_MARGIN
    for m in sorted(ptr[dtr < -0.001], reverse=False):
        pass
    bad_preds = ptr[dtr < -0.001]
    if len(bad_preds):
        margin = max(FIRE_MARGIN, float(bad_preds.max()) + 1e-4)
    print(f"  property-driven margin: {margin:+.4f}")

    worst = 0.0
    for name, split in (("train", train), ("held", held)):
        pp = np.array([pred(r) for r in split])
        dgs = np.array([r["dg"] for r in split])
        f = pp >= margin
        if f.any():
            worst = min(worst, dgs[f].min()) if name == "held" else worst
        print(f"  {name}: fired {int(f.sum())}/{len(split)}  dg>=0: {int((dgs[f] >= 0).sum())}"
              f"/{int(f.sum())}  mean dg={dgs[f].mean() if f.any() else 0:+.4f}  "
              f"min dg={dgs[f].min() if f.any() else 0:+.4f}")
    np.savez(os.path.join(HERE, "thinocc_gate.npz"), w=wgt, mu=mu, sd=sd, margin=margin)


if __name__ == "__main__":
    main()
