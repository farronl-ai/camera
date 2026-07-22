#!/usr/bin/env python3
"""Generate the specialist-layer figures for docs/SHOWCASE.md (F43-F48 arc).

Three figures, same conventions as make_showcase.py (docs/img/*.jpg, JPEG q85):

  spec_recon.jpg — contour reconstruction on a canonical thin-occluder scene
      where the SHIPPED gate (focusstack.gates.RECON_GATE) actually fires:
      [base perband | reconstructed | ground truth] at the most-differing crop.
  spec_veil.jpg  — veil correction at giant CoC with the ORACLE matte
      (mechanism at its clearest): [base | corrected | ground truth] fringe crop.
  spec_fence.jpg — the real-data fire: the fence stack through the SHIPPED
      enhance path, [base | enhanced | amplified difference] at the wire edge.

Crops are disagreement-guided (eyetool discipline), never hand-picked.
Run:  python research/make_showcase_specialists.py [recon|veil|fence|all]
"""
from __future__ import annotations
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from eyetool import _disagreement, _top_regions, _amplify_diff  # noqa: E402
from focusstack.fusion import fuse_perband  # noqa: E402
from focusstack.gates import RECON_GATE, predict_gain  # noqa: E402
from focusstack.reconstruct import (contamination_band, reconstruct_band,  # noqa: E402
                                    thin_matte_features)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
IMG = os.path.join(REPO, "docs", "img")
os.makedirs(IMG, exist_ok=True)


def save(name, img, max_w=1620, q=85):
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, round(h * max_w / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(IMG, name), img, [cv2.IMWRITE_JPEG_QUALITY, q])
    print(f"  wrote {name}  ({img.shape[1]}x{img.shape[0]})")


def gap(h, w=6):
    return np.full((h, w, 3), 255, np.uint8)


def hstack(*imgs):
    h = min(i.shape[0] for i in imgs)
    row = []
    for k, im in enumerate(imgs):
        row.append(im[:h])
        if k < len(imgs) - 1:
            row.append(gap(h))
    return np.hstack(row)


def crop_at(imgs, center, half, zoom):
    """Same window from every image, zoomed with nearest (pixel-honest)."""
    h, w = imgs[0].shape[:2]
    y, x = center
    y0 = int(np.clip(y - half, 0, h - 2 * half))
    x0 = int(np.clip(x - half, 0, w - 2 * half))
    sl = (slice(y0, y0 + 2 * half), slice(x0, x0 + 2 * half))
    return [cv2.resize(im[sl], None, fx=zoom, fy=zoom,
                       interpolation=cv2.INTER_NEAREST) for im in imgs]


def fig_recon():
    """Thin-occluder contour reconstruction, gate-verified on the shipped model."""
    from thinocc_gate import thin_scenes
    from reconstruct import estimate_alpha_v3
    # dg-ranked candidates from the F48 label cache (canonical thin, idx<120,
    # restricted to low indices so the factory build stays cheap)
    candidates = [15, 5, 35, 25, 2, 13]
    scs = thin_scenes(max(candidates) + 1)
    for idx in candidates:
        sc = scs[idx]
        a3, owner = estimate_alpha_v3(sc["frames"], sc["max_r"])
        if a3.max() <= 0:
            print(f"  thin_{idx}: no matte, skip")
            continue
        feats = thin_matte_features(sc["frames"], a3, sc["max_r"])
        gain = predict_gain(RECON_GATE, feats) if feats is not None else -1
        fires = feats is not None and gain >= RECON_GATE["margin"]
        print(f"  thin_{idx}: predicted gain {gain:+.4f} (margin "
              f"{RECON_GATE['margin']:+.4f}) -> {'FIRE' if fires else 'refuse'}")
        if not fires:
            continue
        base = fuse_perband(sc["frames"], harden=0.5)
        rec = reconstruct_band([sc["frames"][owner], sc["frames"][1 - owner]], a3,
                               contamination_band(a3, sc["max_r"]), base, sc["max_r"])
        dg = M.ref_ssim(rec, sc["gt"]) - M.ref_ssim(base, sc["gt"])
        print(f"    actual GT-SSIM delta {dg:+.4f}")
        heat = _disagreement(base, rec)
        (cy, cx), = _top_regions(heat, 1, 110)
        cells = crop_at([base, rec, sc["gt"]], (cy, cx), 110, 3)
        save("spec_recon.jpg", hstack(*cells))
        # companion: what the specialist sees, same crop — the owner frame, the
        # C3 difference matte, and the contamination band it re-renders
        band = contamination_band(a3, sc["max_r"])
        matte = cv2.cvtColor((a3 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        tint = base.astype(np.float32).copy()
        bm = band.astype(np.float32)[..., None]
        tint = tint * (1 - 0.55 * bm) + np.array([0.0, 140.0, 255.0]) * 0.55 * bm
        mcells = crop_at([sc["frames"][owner], matte,
                          np.clip(tint, 0, 255).astype(np.uint8)], (cy, cx), 110, 3)
        save("spec_matte.jpg", hstack(*mcells))
        return
    print("  no candidate fired — no figure written")


def fig_veil():
    """Veil correction at giant CoC, oracle matte (mechanism figure)."""
    from wideocc_gen import scenes
    from semalpha import build_D
    from veilband import fuse_perband_weighted_corr, fringe_mask
    best = None
    for sc in scenes(0.04):
        gt, alpha, max_r = sc["gt"], sc["alpha"], sc["max_r"]
        base = fuse_perband(sc["frames"], harden=0.5)
        D = build_D(sc["frames"], alpha, max_r, owner=0, far_idx=1)
        cor = fuse_perband_weighted_corr(sc["frames"], D, 0)
        fr = fringe_mask(alpha, max_r)
        e0 = float(np.abs(base.astype(np.float32) - gt.astype(np.float32)).sum(2)[fr].mean())
        e1 = float(np.abs(cor.astype(np.float32) - gt.astype(np.float32)).sum(2)[fr].mean())
        g0, g1 = M.ref_ssim(base, gt), M.ref_ssim(cor, gt)
        print(f"  {sc['sid']}: fringe |err| {e0:.1f}->{e1:.1f}  global {g0:.4f}->{g1:.4f}")
        if best is None or (e0 - e1) > best["gain"]:
            best = dict(gain=e0 - e1, base=base, cor=cor, gt=gt, fr=fr, sid=sc["sid"])
    # crop where the correction moved the output most, restricted to the fringe
    heat = _disagreement(best["base"], best["cor"]) * best["fr"].astype(np.float32)
    (cy, cx), = _top_regions(heat, 1, 130)
    cells = crop_at([best["base"], best["cor"], best["gt"]], (cy, cx), 130, 2.5)
    print(f"  chose {best['sid']} crop ({cy},{cx})")
    save("spec_veil.jpg", hstack(*cells))


def fig_fence():
    """Real-data fire: fence stack through the shipped enhance path.

    The enhanced output is cached (the semantic bridge takes minutes) so the
    figure composition can iterate cheaply; delete the cache to force a re-run.
    """
    from focusstack.enhance import enhance
    a = cv2.imread(os.path.join(HERE, "data", "standard", "c_05_1.tif"))
    b = cv2.imread(os.path.join(HERE, "data", "standard", "c_05_2.tif"))
    base = fuse_perband([a, b], harden=0.5)
    cache = os.path.join(HERE, "inspect", "fence_enhanced.npz")
    if os.path.exists(cache):
        out = np.load(cache)["out"]
    else:
        out, rep = enhance([a, b], base, log=print)
        print(f"  fence: report={rep}")
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, out=out)
    d = float(np.abs(out.astype(np.int16) - base.astype(np.int16)).mean())
    print(f"  fence: mean diff={d:.3f}")
    if d == 0:
        print("  nothing fired — no figure written")
        return
    heat = _disagreement(base, out, win=31)
    (cy, cx), = _top_regions(heat, 1, 70)
    cells = crop_at([base, out], (cy, cx), 70, 4)
    amp = _amplify_diff(*crop_at([out, base], (cy, cx), 70, 1), gain=16.0)
    amp = cv2.resize(amp, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    print(f"  crop ({cy},{cx})")
    save("spec_fence.jpg", hstack(cells[0], cells[1], amp))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    print("generating specialist figures -> docs/img/")
    if which in ("recon", "all"):
        fig_recon()
    if which in ("veil", "all"):
        fig_veil()
    if which in ("fence", "all"):
        fig_fence()
    print("done")
