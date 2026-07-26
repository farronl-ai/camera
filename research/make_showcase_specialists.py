#!/usr/bin/env python3
"""Generate the specialist-layer figures for docs/SHOWCASE.md.

Figures use the same conventions as make_showcase.py (docs/img/*.jpg, JPEG q85):

  spec_recon.jpg — contour reconstruction on a canonical thin-occluder scene
      where the SHIPPED gate (focusstack.gates.RECON_GATE) actually fires:
      [base perband | reconstructed | ground truth] at the most-differing crop.
  spec_veil.jpg  — retired veil-gain failure on a realistic object scene
      (mechanism at its clearest): [base | corrected | ground truth] fringe crop.
  spec_joint.jpg — F55/F56's shipped owner-safe joint-layer recovery on a fresh
      holdout fire: [base | recovered physical scene | ground truth].
  spec_fence.jpg — the former real-data subtraction fire retained as an audit
      artifact: [base | former output | amplified difference] at the wire edge.

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
    """F54 failure panel: subtraction, rejected gain, and GT.

    Scene 31 is selected by the recorded realistic-object oracle audit because
    it is the worst low-matte-error counterexample—not for visual drama. The
    crop is located automatically where gain increases GT error inside the true
    fringe.
    """
    from t2_confidence import scenes
    from veilband import fringe_mask
    from veilgain import (
        WINNER,
        build_D_ca,
        calibrate_band_noise,
        fuse_perband_gain,
        glaw_sq,
    )
    from focusstack.fusion import _auto_levels

    sc = list(scenes())[31]
    D, ab, pm = build_D_ca(
        sc["frames"], sc["alpha"], sc["max_r"], owner=0, far_idx=1
    )
    levels = _auto_levels(sc["gt"].shape, None)
    ck, _ = calibrate_band_noise(sc["gt"].shape[:2], levels)
    subtraction = fuse_perband_gain(
        sc["frames"], {1: D}, ab, sc["alpha"], omega=0.0
    )
    hybrid = fuse_perband_gain(
        sc["frames"],
        {1: D},
        ab,
        sc["alpha"],
        g_law=glaw_sq,
        shrink_m=2.0,
        ck=ck,
        pm_by_far={1: pm},
        **WINNER,
    )
    fringe = fringe_mask(sc["alpha"], sc["max_r"])
    e_sub = np.abs(subtraction.astype(np.float32) - sc["gt"]).mean(axis=2)
    e_hybrid = np.abs(hybrid.astype(np.float32) - sc["gt"]).mean(axis=2)
    heat = np.maximum(e_hybrid - e_sub, 0.0) * fringe.astype(np.float32)
    (cy, cx), = _top_regions(heat, 1, 130)
    cells = crop_at(
        [subtraction, hybrid, sc["gt"]], (cy, cx), 130, 2.5
    )
    print(
        f"  {sc['sid']}: subtraction {M.ref_ssim(subtraction, sc['gt']):.4f}, "
        f"rejected gain {M.ref_ssim(hybrid, sc['gt']):.4f}; crop ({cy},{cx})"
    )
    save("spec_veil.jpg", hstack(*cells))


def fig_joint():
    """Shipped joint-layer recovery on an actually licensed holdout scene."""
    from t2_candidates import candidates_with_features
    from t2_confidence import scenes
    from veilship import false_texture_error
    from focusstack.veil_layers import recover_giant_veil

    sc = list(scenes())[114]
    base = fuse_perband(sc["frames"], harden=0.5)
    output, report = recover_giant_veil(
        sc["frames"],
        base,
        candidates_with_features(sc, topk=4),
    )
    if not report["fired"]:
        print(f"  {sc['sid']}: package refused unexpectedly — no figure written")
        return

    dg = M.ref_ssim(output, sc["gt"]) - M.ref_ssim(base, sc["gt"])
    base_error = np.abs(
        base.astype(np.float32) - sc["gt"].astype(np.float32)
    ).mean()
    output_error = np.abs(
        output.astype(np.float32) - sc["gt"].astype(np.float32)
    ).mean()
    ft0, _ = false_texture_error(
        base,
        sc["gt"],
        sc["alpha"],
        sc["max_r"],
    )
    ft, _ = false_texture_error(
        output,
        sc["gt"],
        sc["alpha"],
        sc["max_r"],
    )
    heat = _disagreement(base, output)
    (cy, cx), = _top_regions(heat, 1, 130)
    cells = crop_at(
        [base, output, sc["gt"]],
        (cy, cx),
        130,
        2.5,
    )
    amplified = _amplify_diff(cells[1], cells[0], gain=5.0)
    print(
        f"  {sc['sid']}: rank={report['candidate_rank']} "
        f"forward_ratio={report['forward_ratio']:.3f}, "
        f"GT-SSIM delta={dg:+.5f}, MAE={base_error:.3f}->{output_error:.3f}, "
        f"false-texture={ft0:.3f}->{ft:.3f}; crop ({cy},{cx})"
    )
    save(
        "spec_joint.jpg",
        hstack(*cells, amplified),
        max_w=2160,
        q=88,
    )


def fig_fence():
    """Former real-data subtraction fire retained as an audit artifact.

    The enhanced output is cached (the semantic bridge takes minutes) so the
    historical figure composition can iterate cheaply.  Do not delete the
    cache: current ``enhance`` runs the replacement model, not this old branch.
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
    if which in ("joint", "all"):
        fig_joint()
    if which in ("fence", "all"):
        fig_fence()
    print("done")
