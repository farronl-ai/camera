"""Composed specialist enhancement — conservative specialist routing.

- **Giant-veil recovery** (exactly two frames): semantic candidate bank ->
  observation-domain reranking -> frozen high-precision license -> joint
  two-layer inversion -> regularizer/PSF component consensus.
- **Contour reconstruction** (thin occluders): classical C3 difference matte
  -> gated -> strong-veil ribbon re-rendered post-fusion.

Every refusal path is byte-identical to the incoming generalist fusion.
"""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np

from .bridge import run_bridge, run_bridge_many
from .focus import content_aware_energies
from .fusion import guided_filter
from .gates import RECON_GATE, predict_gain
from .io import to_gray_float
from .reconstruct import (contamination_band, estimate_thin_matte,
                          reconstruct_band, thin_matte_features, _disk_blur)
from .veil_layers import recover_giant_veil


# Operational kill switch. F54's correction-after-fusion implementation remains
# retired; this enables only F55's separately validated joint-layer specialist.
VEIL_AUTO_ENABLED = True


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
    """Retired F40/F41 haze field, retained only for research reproduction.

    The auto path must never call this correction-after-fusion model; F54
    demonstrated that it can extend foreground texture into the background.
    """
    near_pm = images[owner].astype(np.float32) * alpha[..., None]
    far_f = images[far_idx].astype(np.float32)
    ab = _disk_blur(alpha, 0.7 * radius)
    pm_b = np.stack(
        [
            _disk_blur(near_pm[..., channel], 0.7 * radius)
            for channel in range(3)
        ],
        axis=2,
    )
    haze = (pm_b - near_pm) + far_f * (alpha - ab)[..., None]
    band = (
        (ab > 0.02)
        & (ab < 0.98)
        & (alpha < 0.5)
    ).astype(np.float32)
    band = cv2.GaussianBlur(band, (0, 0), 2.0)
    return haze * band[..., None]


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
        "veil_model": "joint_two_layer_giant",
        "veil_reason": (
            "kill_switch_disabled" if not VEIL_AUTO_ENABLED
            else "not_evaluated"
        ),
        "recon_fired": 0,
    }
    out = fused_pass1

    # --- giant-veil branch (bridge + exactly two focal frames) ---
    if VEIL_AUTO_ENABLED and len(images) == 2:
        with tempfile.TemporaryDirectory() as td:
            p1 = os.path.join(td, "pass1.png")
            cv2.imwrite(p1, fused_pass1)
            frame_paths = []
            for index, image in enumerate(images):
                path = os.path.join(td, f"frame_{index}.png")
                cv2.imwrite(path, image)
                frame_paths.append(path)
            dp = run_bridge("depth", p1, python=bridge_python)
            mask_paths = run_bridge_many(
                "masks",
                [p1, *frame_paths],
                python=bridge_python,
            )
            if mask_paths:
                mp = mask_paths[0]
                owner_mask_paths = mask_paths[1:]
            else:
                # Older/custom bridge shims may only expose the single-image
                # surface.  Preserve the prior recovery path, but omit the new
                # owner-silhouette repair when owner-frame masks are unavailable.
                mp = run_bridge("masks", p1, python=bridge_python)
                owner_mask_paths = None
            if dp and mp:
                try:
                    masks, depth = np.load(mp), np.load(dp)
                    owner_masks = (
                        [np.load(path) for path in owner_mask_paths]
                        if owner_mask_paths is not None
                        else None
                    )
                    candidates = _mask_candidates(images, masks, depth)
                    report["bridge"] = True
                    recovered, veil_report = recover_giant_veil(
                        images,
                        fused_pass1,
                        candidates,
                        owner_masks_by_frame=owner_masks,
                    )
                except (ValueError, OSError, cv2.error) as error:
                    report["veil_reason"] = (
                        f"invalid_bridge_or_model:{type(error).__name__}"
                    )
                else:
                    report["veil_reason"] = veil_report["reason"]
                    report["veil_evidence"] = veil_report
                    if veil_report["fired"]:
                        say(
                            "joint two-layer giant-veil recovery firing "
                            f"(candidate {veil_report['candidate_rank']}, "
                            f"forward ratio={veil_report['forward_ratio']:.3f}, "
                            f"stable={veil_report['stable_fraction']:.3f}) ..."
                        )
                        out = recovered
                        report["veil_fired"] = 1
            else:
                report["veil_reason"] = "bridge_unavailable"
    elif VEIL_AUTO_ENABLED:
        report["veil_reason"] = "requires_two_frames"

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
