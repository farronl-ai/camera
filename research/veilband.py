#!/usr/bin/env python3
"""16d P0+P1 — per-band haze profile, then the band-limited oracle ladder.

P0: WHERE does the wide-occluder fringe error live across pyramid bands, and how
big is it? (Prediction: peak at level ≈ log2 CoC; small at finest+deepest.)
P1: O1 oracle — exact error field D = noiseless obs_far − GT, subtracted from the
FUSED pyramid over a band window swept from coarsest downward, D masked to the
background side (α<0.5: where fusion actually used far-frame content).
NO division anywhere — forward-model subtraction only.

Run:  python research/veilband.py
"""
from __future__ import annotations
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from bband import bband  # noqa: E402
from hardbench import disk_blur  # noqa: E402
from wideocc_gen import scenes  # noqa: E402
from occ_gen import occ_defocus  # noqa: E402
from focusstack.fusion import fuse_perband, _laplacian_pyramid, _auto_levels  # noqa: E402


def fringe_mask(alpha, max_r):
    ab = disk_blur(alpha, 0.7 * max_r)
    return (ab > 0.05) & (ab < 0.95)


def band_profile(fused, gt, fringe, levels):
    pf = _laplacian_pyramid(fused.astype(np.float32), levels)
    pg = _laplacian_pyramid(gt.astype(np.float32), levels)
    prof = []
    for k in range(levels + 1):
        m = cv2.resize(fringe.astype(np.float32), (pf[k].shape[1], pf[k].shape[0])) > 0.5
        if not m.any():
            prof.append(0.0)
            continue
        prof.append(float(np.abs(pf[k] - pg[k]).sum(axis=2)[m].mean()))
    return prof


def correct_bands(fused, D, levels, k_from):
    """Subtract D's bands k>=k_from (toward coarse, incl. base) from fused pyramid."""
    pf = _laplacian_pyramid(fused.astype(np.float32), levels)
    pd = _laplacian_pyramid(D, levels)
    out = None
    for k in range(levels, -1, -1):
        band = pf[k] - (pd[k] if k >= k_from else 0.0)
        if out is None:
            out = band
        else:
            out = cv2.pyrUp(out, dstsize=(band.shape[1], band.shape[0])) + band
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    for coc in (0.012, 0.04):
        print(f"\n===== CoC regime {coc} =====")
        agg_prof = None
        rows = []
        for sc in scenes(coc):
            gt, alpha, max_r = sc["gt"], sc["alpha"], sc["max_r"]
            levels = _auto_levels(gt.shape, None)
            fr = fringe_mask(alpha, max_r)
            base = fuse_perband(sc["frames"], harden=0.5)

            # P0: per-band fringe error profile + magnitudes
            prof = band_profile(base, gt, fr, levels)
            agg_prof = np.array(prof) if agg_prof is None else agg_prof + np.array(prof)
            err = np.abs(base.astype(np.float32) - gt.astype(np.float32)).sum(2)
            fringe_err = float(err[fr].mean())
            share = float(err[fr].sum() / err.sum() * 100)

            # P1: exact D (noiseless far render − GT), background-side mask
            obs_far_clean = occ_defocus(sc["far"].astype(np.uint8), sc["near"], alpha,
                                        0.85, 0.15, 0.85, max_r).astype(np.float32)
            D = (obs_far_clean - gt.astype(np.float32)) * (alpha < 0.5)[..., None]
            m = (alpha > 0.5).astype(np.uint8)
            gt_b = (cv2.dilate(m, np.ones((3, 3), np.uint8)) != cv2.erode(m, np.ones((3, 3), np.uint8))).astype(np.uint8)
            row = {"sid": sc["sid"], "fringe": fringe_err, "share": share,
                   "glob0": M.ref_ssim(base, gt), "e2_0": bband(base, gt, gt_b, 2)[0], "win": {}}
            for k_from in (levels, levels - 1, levels - 2, levels - 3):
                cor = correct_bands(base, D, levels, k_from)
                row["win"][k_from] = (float(np.abs(cor.astype(np.float32) - gt.astype(np.float32)).sum(2)[fr].mean()),
                                      M.ref_ssim(cor, gt), bband(cor, gt, gt_b, 2)[0])
            rows.append(row)

        print(f"P0 mean per-band fringe |err| (fine->base): "
              + " ".join(f"{v/len(rows):.1f}" for v in agg_prof))
        print(f"{'scene':22s} fringe|err| share%  glob   e2   | O1 by window (fringe/glob/e2)")
        for r in rows:
            wins = "  ".join(f"k>={k}: {v[0]:5.1f}/{v[1]:.4f}/{v[2]:4.1f}" for k, v in r["win"].items())
            print(f"{r['sid']:22s} {r['fringe']:6.1f} {r['share']:5.1f}  {r['glob0']:.4f} {r['e2_0']:4.1f} | {wins}")


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


def fuse_perband_weighted_corr(images, D, k_from, radius=6, eps=1e-3,
                               energy_ksize=7, harden=0.5, far_idx=1):
    """O1b: perband with in-loop veil correction — subtract w_far * L_k(D) at
    bands k >= k_from. The haze enters the output only through the far frame's
    weights, so the subtraction must be scaled by them (P1 showed unweighted D
    over-corrects where fusion already avoided the haze)."""
    from focusstack.fusion import (_auto_levels, _laplacian_pyramid,
                                   _gaussian_pyramid, guided_filter)
    from focusstack.io import to_gray_float
    floats = [img.astype(np.float32) for img in images]
    n = len(floats)
    levels = _auto_levels(floats[0].shape, None)
    ip = [_laplacian_pyramid(im, levels) for im in floats]
    gp = [_gaussian_pyramid(to_gray_float(f), levels) for f in images]
    dp = _laplacian_pyramid(D.astype(np.float32), levels)
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
            if band >= k_from:
                fb = fb - w[far_idx][..., None] * dp[band]
            fused_bands.append(fb)
            w_last = w
        else:
            wb = np.stack([cv2.pyrDown(w_last[k]) for k in range(n)], axis=0)
            wb = np.clip(wb, 0.0, None)
            wb /= (wb.sum(axis=0, keepdims=True) + 1e-8)
            fb = sum(wb[k][..., None] * coeffs[k] for k in range(n))
            if band >= k_from:
                fb = fb - wb[far_idx][..., None] * dp[band]
            fused_bands.append(fb)
    result = fused_bands[-1]
    for band in range(levels - 1, -1, -1):
        result = cv2.pyrUp(result, dstsize=(fused_bands[band].shape[1], fused_bands[band].shape[0])) + fused_bands[band]
    return np.clip(result, 0, 255).astype(np.uint8)


def o1b():
    from focusstack.fusion import _auto_levels
    for coc in (0.04, 0.012):
        print(f"\n===== O1b weighted in-loop correction, CoC {coc} =====")
        for sc in scenes(coc):
            gt, alpha, max_r = sc["gt"], sc["alpha"], sc["max_r"]
            levels = _auto_levels(gt.shape, None)
            fr = fringe_mask(alpha, max_r)
            base = fuse_perband(sc["frames"], harden=0.5)
            obs_far_clean = occ_defocus(sc["far"].astype(np.uint8), sc["near"], alpha,
                                        0.85, 0.15, 0.85, max_r).astype(np.float32)
            D = (obs_far_clean - gt.astype(np.float32)) * (alpha < 0.5)[..., None]
            e0 = float(np.abs(base.astype(np.float32) - gt.astype(np.float32)).sum(2)[fr].mean())
            g0 = M.ref_ssim(base, gt)
            row = []
            for k_from in (levels - 1, levels - 3, 0):
                cor = fuse_perband_weighted_corr(sc["frames"], D, k_from)
                ec = float(np.abs(cor.astype(np.float32) - gt.astype(np.float32)).sum(2)[fr].mean())
                row.append(f"k>={k_from}: {ec:5.1f}/{M.ref_ssim(cor, gt):.4f}")
            print(f"  {sc['sid']:22s} base {e0:5.1f}/{g0:.4f} | " + "  ".join(row))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "o1b":
    o1b()


def estimate_D(frames, max_r, owner=0, far_idx=1):
    """P2: forward-simulated haze field from estimates only (no oracle).

    alpha-hat: the owner's coherent winner region at coarse scale (wide occluders
    are the EASY matting case), guided-snapped to the owner frame. near-hat premult
    = alpha * obs_owner. far-hat = obs_far. D-hat per the exact error identity.
    """
    from focusstack.focus import content_aware_energies
    from focusstack.io import to_gray_float
    from focusstack.fusion import guided_filter
    grays = [to_gray_float(f) for f in frames]
    E = np.stack(content_aware_energies(grays), 0)
    winner = np.argmax(E, 0)
    srt = np.sort(E, axis=0)
    top = srt[-1]
    # winner is only meaningful where focus is decisive (F26): propagate the
    # DECISIVE labels into textureless holes from the nearest decisive pixel
    decisive = ((srt[-1] - srt[-2]) / (srt[-1] + 1e-6) > 0.3) & (top > np.median(top))
    if decisive.sum() < 100:
        return np.zeros_like(top)[..., None] * np.zeros(3), np.zeros_like(top)
    inv = np.where(decisive, 0, 1).astype(np.uint8)
    _, lbl = cv2.distanceTransformWithLabels(inv, cv2.DIST_L2, 5,
                                             labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros(int(lbl.max()) + 1, np.uint8)
    ys, xs = np.where(decisive)
    lut[lbl[ys, xs]] = winner[ys, xs]
    winner_filled = lut[lbl]
    side = (winner_filled == owner).astype(np.float32)
    side = cv2.medianBlur((side * 255).astype(np.uint8), 15).astype(np.float32) / 255.0
    a = np.clip(guided_filter(grays[owner] / 255.0, side, 4, 1e-3), 0.0, 1.0)
    near_pm = frames[owner].astype(np.float32) * a[..., None]
    far_f = frames[far_idx].astype(np.float32)
    ab = disk_blur(a, 0.7 * max_r)
    pm_b = np.stack([disk_blur(near_pm[..., c], 0.7 * max_r) for c in range(3)], 2)
    D = (pm_b - near_pm) + far_f * (a - ab)[..., None]
    return D * (a < 0.5)[..., None], a


def p2():
    from focusstack.fusion import _auto_levels
    for coc in (0.04, 0.012):
        print(f"\n===== P2 estimator, CoC {coc} =====")
        for sc in scenes(coc):
            gt, alpha, max_r = sc["gt"], sc["alpha"], sc["max_r"]
            fr = fringe_mask(alpha, max_r)
            base = fuse_perband(sc["frames"], harden=0.5)
            Dh, a_est = estimate_D(sc["frames"], max_r)
            e0 = float(np.abs(base.astype(np.float32) - gt.astype(np.float32)).sum(2)[fr].mean())
            row = []
            for k_from in (0, 3):
                cor = fuse_perband_weighted_corr(sc["frames"], Dh, k_from)
                ec = float(np.abs(cor.astype(np.float32) - gt.astype(np.float32)).sum(2)[fr].mean())
                row.append(f"k>={k_from}: {ec:5.1f}/{M.ref_ssim(cor, gt):.4f}")
            a_err = float(np.abs(a_est - alpha).mean())
            print(f"  {sc['sid']:22s} base {e0:5.1f}/{M.ref_ssim(base, gt):.4f} | "
                  + "  ".join(row) + f"  |alpha err|={a_err:.3f}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "p2":
    p2()


def o2():
    """True alpha, ESTIMATED content (near premult from owner frame, far observed):
    if O2 ~= O1b, matte quality alone closes the gap -> semantic alpha is the path."""
    from focusstack.fusion import _auto_levels
    for coc in (0.04, 0.012):
        print(f"\n===== O2: true alpha + estimated content, CoC {coc} =====")
        for sc in scenes(coc):
            gt, alpha, max_r = sc["gt"], sc["alpha"], sc["max_r"]
            fr = fringe_mask(alpha, max_r)
            base = fuse_perband(sc["frames"], harden=0.5)
            near_pm = sc["frames"][0].astype(np.float32) * alpha[..., None]
            far_f = sc["frames"][1].astype(np.float32)
            ab = disk_blur(alpha, 0.7 * max_r)
            pm_b = np.stack([disk_blur(near_pm[..., c], 0.7 * max_r) for c in range(3)], 2)
            D = ((pm_b - near_pm) + far_f * (alpha - ab)[..., None]) * (alpha < 0.5)[..., None]
            e0 = float(np.abs(base.astype(np.float32) - gt.astype(np.float32)).sum(2)[fr].mean())
            row = []
            for k_from in (0, 3):
                cor = fuse_perband_weighted_corr(sc["frames"], D, k_from)
                ec = float(np.abs(cor.astype(np.float32) - gt.astype(np.float32)).sum(2)[fr].mean())
                row.append(f"k>={k_from}: {ec:5.1f}/{M.ref_ssim(cor, gt):.4f}")
            print(f"  {sc['sid']:22s} base {e0:5.1f}/{M.ref_ssim(base, gt):.4f} | " + "  ".join(row))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "o2":
    o2()
