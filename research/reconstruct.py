#!/usr/bin/env python3
"""16b — matte-aware boundary RECONSTRUCTION: oracle ceiling, then estimators.

The F33 insight: boundary-band error is coefficient contamination — the lens
already mixed both sides into the captured pixels; no weight map can unmix it.
The fix is to RE-RENDER the boundary band as a fresh composite

    out = alpha * (owner frame) + (1 - alpha) * (background extended inward)

where each ingredient is taken from where it is UNcontaminated: the owner (near)
frame holds the sharp silhouette and sharp matte; the background is clean a few
px away from the contour and is extended inward by inpainting (NOT recovered by
division — F27). Elsewhere the image stays pure perband.

Rungs (oracle ladder, per DEVSTYLE):
  B  — TRUE sharp alpha, owner frame content, inpainted far frame: the realizable
       ceiling from FRAMES (only alpha is oracle).
  C  — estimated alpha (from the owner frame itself) + same machinery: buildable.

Run:  python research/reconstruct.py            # occ benchmark, rungs B (and C when ready)
"""
from __future__ import annotations
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bband import bband  # noqa: E402
import metrics as M  # noqa: E402
from hardbench import disk_blur  # noqa: E402
from focusstack.fusion import fuse_perband  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def occ_scenes():
    """Rebuild the 4 occ scenes with ALL internals exposed (alpha, layers, frames)."""
    from occ_gen import near_layer, occ_defocus, LONG
    from hires_gen import add_noise
    out = []
    for gp in sorted(glob.glob(os.path.join(HERE, "data", "hires", "*", "gt.png")))[:4]:
        far = cv2.imread(gp)
        h, w = far.shape[:2]
        s = LONG / max(h, w)
        far = cv2.resize(far, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        near, alpha = near_layer(far, seed=hash(gp) % 1000)
        gt = (near * alpha[..., None] + far.astype(np.float32) * (1 - alpha[..., None])).astype(np.uint8)
        max_r = 0.012 * max(gt.shape[:2])
        frames = [add_noise(occ_defocus(far, near, alpha, f, 0.15, 0.85, max_r), 3.0, i)
                  for i, f in enumerate([0.15, 0.85])]
        out.append(dict(sid=os.path.basename(os.path.dirname(gp)), gt=gt, alpha=alpha,
                        frames=frames, max_r=max_r))
    return out


def contamination_band(alpha, max_r):
    """Where the far frame is veiled: blurred-alpha support minus deep-interior."""
    r = 0.7 * max_r
    ab = disk_blur(alpha, r)
    return ((ab > 0.02) & (alpha < 0.98)).astype(np.uint8)


def reconstruct_band(frames, alpha_sharp, band, base_fused, max_r, inpaint_r=5):
    """out = a*near_frame + (1-a)*far_est inside band; base_fused elsewhere.

    far_est blends the OBSERVED far frame with an inpainted extension BY VEIL
    STRENGTH: where the veil is faint (most of the band around thin structures,
    F27) the observation is nearly clean — keep it; only the narrow strong-veil
    ribbon hugging the contour is replaced by inward extension. Replacing the
    whole band with inpaint destroys good data (measured: -0.08 global).
    """
    near_f, far_f = frames[0].astype(np.float32), frames[1]
    ab = disk_blur(alpha_sharp, 0.7 * max_r)                 # veil strength field
    strong = (ab > 0.15).astype(np.uint8)                    # narrow ribbon only
    far_ext = cv2.inpaint(far_f, strong, inpaint_r, cv2.INPAINT_TELEA).astype(np.float32)
    v = np.clip((ab - 0.15) / 0.5, 0.0, 1.0)[..., None]      # veil-strength blend
    far_est = (1.0 - v) * far_f.astype(np.float32) + v * far_ext

    # Correct composite: obs_near ALREADY contains alpha*near + blur(bg)*(1-alpha);
    # replace only its blurred-background component with the sharp estimate:
    #   out = obs_near + (1-alpha) * (far_est - blur(far_est, r_bg_in_near_frame)).
    # (The naive a*obs_near form double-counts alpha — systematic error on thin
    # structures where most pixels are partial-alpha.)
    r_bg = 0.75 * max_r
    far_est_blur = np.stack([disk_blur(far_est[..., c], r_bg) for c in range(3)], axis=2)
    a = alpha_sharp[..., None]
    recon = near_f + (1.0 - a) * (far_est - far_est_blur)

    # Override ONLY the strong-veil ribbon (feathered): in the faint zone perband's
    # denoised output is better than any raw-frame reconstruction.
    m = cv2.GaussianBlur(strong.astype(np.float32), (0, 0), 1.5)[..., None]
    out = m * recon + (1.0 - m) * base_fused.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def estimate_alpha(frames):
    """Rung C matte estimate: owner-frame focus dominance, softened (no oracle)."""
    from focusstack.focus import content_aware_energies
    from focusstack.io import to_gray_float
    e0, e1 = content_aware_energies([to_gray_float(frames[0]), to_gray_float(frames[1])])
    raw = ((e0 > 2.0 * e1) & (e0 > np.percentile(e0, 60))).astype(np.float32)
    raw = cv2.dilate(raw, np.ones((3, 3), np.uint8))
    return np.clip(cv2.GaussianBlur(raw, (0, 0), 1.0), 0.0, 1.0)


def main():
    print(f"{'scene':16s} |  base e2/ssim2  | rungB e2/ssim2  | rungC e2/ssim2  | glob b/B/C")
    agg = {k: [] for k in ["be", "bs", "ce", "cs", "re", "rs", "bg", "rg", "cg"]}
    for sc in occ_scenes():
        gt_b = ((cv2.dilate((sc["alpha"] > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8))
                 != cv2.erode((sc["alpha"] > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8)))).astype(np.uint8)
        base = fuse_perband(sc["frames"], harden=0.5)
        band = contamination_band(sc["alpha"], sc["max_r"])
        rec_b = reconstruct_band(sc["frames"], sc["alpha"], band, base, sc["max_r"])
        a_est = estimate_alpha(sc["frames"])
        band_c = contamination_band(a_est, sc["max_r"])
        rec_c = reconstruct_band(sc["frames"], a_est, band_c, base, sc["max_r"])
        e2b, s2b = bband(base, sc["gt"], gt_b, 2)
        e2r, s2r = bband(rec_b, sc["gt"], gt_b, 2)
        e2c, s2c = bband(rec_c, sc["gt"], gt_b, 2)
        gb, gr, gc = (M.ref_ssim(x, sc["gt"]) for x in (base, rec_b, rec_c))
        for k, v in zip(["be", "bs", "re", "rs", "ce", "cs", "bg", "rg", "cg"],
                        [e2b, s2b, e2r, s2r, e2c, s2c, gb, gr, gc]):
            agg[k].append(v)
        print(f"{sc['sid']:16s} | {e2b:6.1f}/{s2b:.4f} | {e2r:6.1f}/{s2r:.4f} | "
              f"{e2c:6.1f}/{s2c:.4f} | {gb:.4f}/{gr:.4f}/{gc:.4f}")
    print(f"\nMEAN  base  e2={np.mean(agg['be']):.1f} ssim={np.mean(agg['bs']):.4f} glob={np.mean(agg['bg']):.4f}")
    print(f"MEAN  rungB e2={np.mean(agg['re']):.1f} ssim={np.mean(agg['rs']):.4f} glob={np.mean(agg['rg']):.4f}  (true alpha ceiling)")
    print(f"MEAN  rungC e2={np.mean(agg['ce']):.1f} ssim={np.mean(agg['cs']):.4f} glob={np.mean(agg['cg']):.4f}  (estimated alpha — buildable)")


if __name__ == "__main__":
    main()
