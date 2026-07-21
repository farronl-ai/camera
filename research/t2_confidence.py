#!/usr/bin/env python3
"""16e T2 — per-component learned confidence: the routing layer.

Features per selected matte candidate (score, purity, ring_other, areafit,
depth margin, area fraction) -> numpy logistic regression -> fire/hold gate.
Labels from the objects-as-occluders factory (|alpha err| < 0.05).
Owner fix folded in: owner = majority decisive winner INSIDE the eroded mask
(physics: the owner is whoever is sharp inside), overriding the scoring's k.

Train scenes 00-19, held 20-27. Required property: gated chain >= baseline on
EVERY scene (gate eats bad mattes, keeps good ones).

Run:  python research/t2_confidence.py
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
from veilband import fringe_mask, fuse_perband_weighted_corr  # noqa: E402
from maskmatte import mask_matte  # noqa: E402
from focusstack.fusion import fuse_perband, guided_filter  # noqa: E402
from focusstack.focus import content_aware_energies  # noqa: E402
from focusstack.io import to_gray_float  # noqa: E402

ROOT = "/home/farron/camera/research/data/objocc"


def scenes():
    man = json.load(open(os.path.join(ROOT, "manifest.json")))
    for m in man:
        d = os.path.join(ROOT, m["id"])
        yield dict(sid=m["id"], max_r=m["max_r"],
                   gt=cv2.imread(os.path.join(d, "gt.png")),
                   alpha=cv2.imread(os.path.join(d, "alpha.png"), 0).astype(np.float32) / 255.0,
                   frames=[cv2.imread(os.path.join(d, f"frame_{k}.png")) for k in (0, 1)],
                   dir=d)


def matte_with_features(sc):
    """Run selection, extract features + owner-fixed matte."""
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

    best = None
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
            if best is None or score > best["score"]:
                best = dict(score=score, purity=purity, ring_other=ring_other,
                            areafit=min(1.0, areafit * 4), margin=din - dout,
                            areafrac=area / (h * w), mi=mi)
    if best is None:
        return None, np.zeros((h, w), np.float32), 0
    sel = masks[best["mi"]] > 0
    # OWNER FIX: majority decisive winner inside the eroded mask
    interior = cv2.erode(sel.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    dec_int = decisive & interior
    owner = int(np.bincount(winner[dec_int].ravel(), minlength=len(frames)).argmax()) \
        if dec_int.sum() > 20 else 0
    alpha = np.clip(guided_filter(grays[owner] / 255.0, sel.astype(np.float32), 2, 1e-4), 0.0, 1.0)
    # snap-consistency: a mask aligned with true image edges survives guided
    # snapping; a bad mask deforms (IoU drop)
    snapped = alpha > 0.5
    inter = float((snapped & sel).sum())
    union = float((snapped | sel).sum()) + 1e-6
    feats = [best["score"], best["purity"], best["ring_other"], best["areafit"],
             best["margin"], best["areafrac"], inter / union]
    return np.array(feats, np.float32), alpha, owner


def logistic_fit(X, y, iters=400, lr=0.5):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    Xb = np.hstack([Xn, np.ones((len(Xn), 1))])
    wgt = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ wgt))
        wgt -= lr * Xb.T @ (p - y) / len(y)
    return wgt, mu, sd


def logistic_p(wgt, mu, sd, x):
    xn = np.hstack([(x - mu) / sd, [1.0]])
    return float(1 / (1 + np.exp(-xn @ wgt)))


def main():
    data = []
    for i, sc in enumerate(scenes()):
        feats, a_est, owner = matte_with_features(sc)
        aerr = float(np.abs(a_est - sc["alpha"]).mean()) if a_est.max() > 0 else 1.0
        # LABEL = the OUTCOME (did the correction actually help), not a matte
        # proxy — a small mean alpha error can still hide a misplaced edge.
        base = fuse_perband(sc["frames"], harden=0.5)
        label = 0.0
        if feats is not None and a_est.max() > 0:
            D = build_D(sc["frames"], a_est, sc["max_r"], owner, 1 - owner)
            cor = fuse_perband_weighted_corr(sc["frames"], D, 0, far_idx=1 - owner)
            fr = fringe_mask(sc["alpha"], sc["max_r"])
            e0 = float(np.abs(base.astype(np.float32) - sc["gt"].astype(np.float32)).sum(2)[fr].mean())
            ec = float(np.abs(cor.astype(np.float32) - sc["gt"].astype(np.float32)).sum(2)[fr].mean())
            g0, gc = M.ref_ssim(base, sc["gt"]), M.ref_ssim(cor, sc["gt"])
            label = 1.0 if (ec < e0 and gc - g0 >= -0.0005) else 0.0
            sc["_cache"] = (base, cor, e0, ec, g0, gc)
        else:
            sc["_cache"] = (base, base, 0, 0, 0, 0)
        data.append(dict(i=i, sc=sc, feats=feats, a=a_est, owner=owner, aerr=aerr, label=label))
        print(f"  {sc['sid']:9s} aerr={aerr:.3f} owner={owner} label={int(label)}")

    n_tr = int(0.75 * len(data))
    train = [r for r in data if r["i"] < n_tr and r["feats"] is not None]
    held = [r for r in data if r["i"] >= n_tr and r["feats"] is not None]
    X = np.stack([r["feats"] for r in train])
    y = np.array([r["label"] for r in train])
    wgt, mu, sd = logistic_fit(X, y)
    # threshold: highest tau with precision 1.0 on train
    ps = np.array([logistic_p(wgt, mu, sd, r["feats"]) for r in train])
    # tau = worst bad-example probability + a SAFETY MARGIN (train-only choice;
    # bare precision-1.0 thresholds leak borderline candidates on held data)
    bad_p = ps[y == 0]
    tau = float(min(1.0, (bad_p.max() if len(bad_p) else 0.0) + 0.10))
    print(f"\n  train fired@tau={tau:.3f}: {int((ps>=tau).sum())}/{len(train)} "
          f"(all-good={bool((y[ps>=tau]==1).all())})")
    ph = np.array([logistic_p(wgt, mu, sd, r["feats"]) for r in held])
    yh = np.array([r["label"] for r in held])
    fired_h = ph >= tau
    print(f"  held fired: {int(fired_h.sum())}/{len(held)}, of which good: {int(yh[fired_h].sum())} "
          f"(labels: {yh.astype(int).tolist()})")

    # every-scene >= baseline property with the gate
    print(f"\n  {'scene':9s} fired  fringe base->gated   glob base->gated")
    worst = 0.0
    for r in data:
        sc = r["sc"]
        base, cor, e0, ec, g0, gc = sc["_cache"]
        p = logistic_p(wgt, mu, sd, r["feats"]) if r["feats"] is not None else 0.0
        fired = p >= tau and r["a"].max() > 0
        if not fired:
            ec, gc = e0, g0
        if fired:
            worst = min(worst, gc - g0)
            print(f"  {sc['sid']:9s} {str(fired):5s} {e0:6.1f}->{ec:6.1f}   {g0:.4f}->{gc:.4f}")
    print(f"\n  worst global delta = {worst:+.4f}  (property: >= -0.001)")
    np.savez("/home/farron/camera/research/t2_gate.npz", w=wgt, mu=mu, sd=sd, tau=tau)


if __name__ == "__main__":
    main()
