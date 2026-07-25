"""Composed specialist enhancement — conservative specialist routing.

Contour reconstruction remains live behind its locked outcome-trained gate.
The veil branch is retained as research machinery but safety-disabled after F54:

- **Veil correction** (wide occluders): disabled in `enhance="auto"` because
  native-resolution false-texture auditing overturned its never-harm claim.
- **Contour reconstruction** (thin occluders): classical C3 difference matte
  -> gated -> strong-veil ribbon re-rendered post-fusion
  (reconstruct.reconstruct_band). No bridge needed.

Identity by construction when the contour gate stays silent.
"""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np

from .bridge import run_bridge
from .focus import content_aware_energies
from .fusion import fuse_perband, guided_filter
from .gates import RECON_GATE, VEIL_GATE, predict_gain
from .io import to_gray_float
from .reconstruct import (contamination_band, estimate_thin_matte,
                          reconstruct_band, thin_matte_features, _disk_blur)


# F54 safety hold. The model and gate remain in-tree for reproducible research,
# but auto enhancement must not call them until a replacement passes the
# expanded native-resolution hallucination audit.
VEIL_AUTO_ENABLED = False


def _mask_candidates(images, masks, depth, topk=4):
    """(feats, alpha, owner) candidates from semantic masks (veil branch)."""
    h, w = images[0].shape[:2]
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h))
    d = (depth - depth.min()) / (np.ptp(depth) + 1e-9)
    grays = [to_gray_float(f) for f in images]
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
        for k in range(len(images)):
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
        owner = int(np.bincount(winner[dec_int].ravel(), minlength=len(images)).argmax()) \
            if dec_int.sum() > 20 else 0
        alpha = np.clip(guided_filter(grays[owner] / 255.0, sel.astype(np.float32), 2, 1e-4), 0.0, 1.0)
        snapped = alpha > 0.5
        iou = float((snapped & sel).sum()) / (float((snapped | sel).sum()) + 1e-6)
        feats = np.array([c["score"], c["purity"], c["ring_other"], c["areafit"],
                          c["margin"], c["areafrac"], iou], np.float32)
        out.append(dict(feats=feats, alpha=alpha, owner=owner))
    return out


def _build_veil_D(images, alpha, radius, owner, far_idx):
    """Forward-modeled haze field, fringe-clamped (F40/F41 + F44 hygiene)."""
    near_pm = images[owner].astype(np.float32) * alpha[..., None]
    far_f = images[far_idx].astype(np.float32)
    ab = _disk_blur(alpha, 0.7 * radius)
    pm_b = np.stack([_disk_blur(near_pm[..., c], 0.7 * radius) for c in range(3)], 2)
    D = (pm_b - near_pm) + far_f * (alpha - ab)[..., None]
    band = ((ab > 0.02) & (ab < 0.98) & (alpha < 0.5)).astype(np.float32)
    band = cv2.GaussianBlur(band, (0, 0), 2.0)
    return D * band[..., None]


def enhance(images, fused_pass1, radius=None, harden=0.5,
            bridge_python=None, log=None):
    """Run the composed gated specialists. Returns (out, report).

    `fused_pass1` is the generalist fusion of `images` (used for the bridge
    input and returned unchanged when nothing fires).
    """
    def say(msg):
        if log:
            log(msg)

    if radius is None:
        radius = 0.012 * max(fused_pass1.shape[:2])
    report = {
        "bridge": False,
        "veil_fired": 0,
        "veil_disabled_safety": not VEIL_AUTO_ENABLED,
        "recon_fired": 0,
    }
    out = fused_pass1

    # --- veil branch (needs the bridge) ---
    D_by_far = {}
    if VEIL_AUTO_ENABLED:
        with tempfile.TemporaryDirectory() as td:
            p1 = os.path.join(td, "pass1.png")
            cv2.imwrite(p1, fused_pass1)
            dp = run_bridge("depth", p1, python=bridge_python)
            mp = run_bridge("masks", p1, python=bridge_python)
            if dp and mp:
                report["bridge"] = True
                masks, depth = np.load(mp), np.load(dp)
                for c in _mask_candidates(images, masks, depth):
                    if predict_gain(VEIL_GATE, c["feats"]) >= VEIL_GATE["margin"]:
                        f = 1 - c["owner"] if len(images) == 2 else len(images) - 1
                        D = _build_veil_D(images, c["alpha"], radius, c["owner"], f)
                        D_by_far[f] = D_by_far.get(f, 0) + D
                        report["veil_fired"] += 1
    if D_by_far:
        say(f"veil correction firing on {report['veil_fired']} region(s); refusing ...")
        out = fuse_perband(images, harden=harden, veil_D=D_by_far)

    # --- reconstruction branch (classical) ---
    alpha, owner = estimate_thin_matte(images, radius)
    if alpha.max() > 0:
        feats = thin_matte_features(images, alpha, radius)
        if feats is not None and predict_gain(RECON_GATE, feats) >= RECON_GATE["margin"]:
            say("contour reconstruction firing ...")
            far_idx = 1 - owner if len(images) == 2 else len(images) - 1
            out = reconstruct_band([images[owner], images[far_idx]], alpha,
                                   contamination_band(alpha, radius), out, radius)
            report["recon_fired"] = 1
    return out, report
