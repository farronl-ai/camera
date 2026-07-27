"""Analytic GT factory for depth-dependent parallax (F81).

Why this exists: alignment cannot be tested on frames that differ by a single
global transform, because a global aligner is exactly right on those. The
failure mode that matters is near and far content moving by DIFFERENT amounts,
which happens whenever the camera centre translates — as it does on every
handheld sweep, since a hand pivots the device rather than the lens entrance
pupil. So this renders two depth planes seen from laterally displaced camera
centres, with a focal sweep across them, and keeps the reference viewpoint's
all-sharp composite as ground truth.

Deliberately included, because each one is a way the alignment can look fine and
not be: a disk (not Gaussian) defocus PSF, an occluding foreground with a real
silhouette, hard thin structure that visibly doubles when misregistered, and a
near/far displacement ratio large enough that no single warp can fit both.

`+refusal` is the same output with parallax-uncovered pixels withheld from the
focus contest, which is where the second half of the gain lives.

Run directly to compare the global-only aligner against the depth-aware pass:

    .venv/bin/python research/parallax_gen.py
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from focusstack.align import align_stack  # noqa: E402
from focusstack.fusion import fuse_perband, selection_instability_score  # noqa: E402
import metrics  # noqa: E402

HEIGHT, WIDTH = 420, 560
FRAMES = 6
REFERENCE = 3
# The whole point: the near plane moves ~4.5x as far as the far plane.
NEAR_SHIFT_PER_FRAME = 3.2
FAR_SHIFT_PER_FRAME = 0.7
NEAR_FOCUS_FRAME = 1
FAR_FOCUS_FRAME = 4
BLUR_PER_STEP = 1.15
PAD = 60
# Focus breathing: refocusing changes magnification. Real and large — 14% across a
# 12-frame phone macro sweep — and depth-INDEPENDENT, which is exactly what makes it
# separable from parallax and impossible for a per-depth translation to express.
BREATHING_PER_FRAME = 0.0


def _disk(radius: float) -> np.ndarray | None:
    """Disk (bokeh) PSF — real defocus is a circle of confusion, not a Gaussian."""
    r = int(round(radius))
    if r < 1:
        return None
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    kernel = ((xx ** 2 + yy ** 2) <= r * r).astype(np.float32)
    return kernel / kernel.sum()


def _defocus(image: np.ndarray, radius: float) -> np.ndarray:
    kernel = _disk(radius)
    if kernel is None:
        return image
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REPLICATE)


def _texture(seed: int, h: int, w: int) -> np.ndarray:
    """Structured content: smooth blobs plus lines and bars that double visibly."""
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 210, size=(h // 8, w // 8, 3), dtype=np.uint8)
    image = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    for _ in range(28):
        x0, y0 = rng.integers(0, w), rng.integers(0, h)
        x1, y1 = x0 + rng.integers(-90, 90), y0 + rng.integers(-90, 90)
        colour = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.line(image, (x0, y0), (x1, y1), colour, int(rng.integers(1, 3)))
    for _ in range(14):
        x0, y0 = rng.integers(0, w - 40), rng.integers(0, h - 20)
        cv2.rectangle(image, (x0, y0),
                      (x0 + int(rng.integers(8, 38)), y0 + int(rng.integers(4, 16))),
                      (250, 250, 250), -1)
    # Surface detail. Without it almost every detected edge is a silhouette, the
    # material/limb test rejects nearly everything, and any instrument that fits
    # object motion from MATERIAL edges (F92) cannot be validated here at all —
    # the factory yielded 12-21 usable features before this was added.
    for _ in range(90):
        x0, y0 = rng.integers(6, w - 46), rng.integers(6, h - 26)
        shade = int(rng.integers(0, 90)) if rng.random() < 0.5 else int(rng.integers(170, 255))
        cv2.rectangle(image, (x0, y0),
                      (x0 + int(rng.integers(6, 40)), y0 + int(rng.integers(3, 18))),
                      (shade, shade, shade), -1)
    for _ in range(70):
        x0, y0 = rng.integers(4, w - 4), rng.integers(4, h - 4)
        cv2.line(image, (x0, y0), (x0 + int(rng.integers(-30, 30)), y0),
                 (int(rng.integers(0, 255)),) * 3, 1)
    return np.clip(image, 0, 255).astype(np.uint8)


def build_stack() -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Return (frames, all-in-focus GT at the reference viewpoint, near-plane mask)."""
    canvas = (HEIGHT + 2 * PAD, WIDTH + 2 * PAD)
    background = _texture(1, *canvas)
    foreground = _texture(7, *canvas)
    alpha = np.zeros(canvas, np.float32)
    cv2.rectangle(alpha, (PAD + 60, PAD + 70), (PAD + 300, PAD + 330), 1.0, -1)
    cv2.circle(alpha, (PAD + 400, PAD + 140), 70, 1.0, -1)

    def viewpoint(layer: np.ndarray, shift: float, scale: float = 1.0) -> np.ndarray:
        # Magnification about the frame centre (breathing), then the viewpoint shift.
        cx, cy = (canvas[1] - 1) / 2.0, (canvas[0] - 1) / 2.0
        matrix = np.float32([[scale, 0, cx * (1 - scale) - shift],
                             [0, scale, cy * (1 - scale)]])
        return cv2.warpAffine(layer, matrix, (canvas[1], canvas[0]),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def crop(x: np.ndarray) -> np.ndarray:
        return x[PAD:PAD + HEIGHT, PAD:PAD + WIDTH]

    frames = []
    for k in range(FRAMES):
        step = k - REFERENCE
        breathe = 1.0 + BREATHING_PER_FRAME * step
        near_alpha = viewpoint(alpha, step * NEAR_SHIFT_PER_FRAME, breathe)
        near_radius = abs(k - NEAR_FOCUS_FRAME) * BLUR_PER_STEP
        far_radius = abs(k - FAR_FOCUS_FRAME) * BLUR_PER_STEP
        # Blur the matte with its own layer, so the occlusion boundary defocuses
        # the way a real foreground edge does instead of staying razor sharp.
        blurred_alpha = _defocus(near_alpha, near_radius)[..., None]
        composited = (
            _defocus(viewpoint(foreground, step * NEAR_SHIFT_PER_FRAME,
                               breathe).astype(np.float32),
                     near_radius) * blurred_alpha
            + _defocus(viewpoint(background, step * FAR_SHIFT_PER_FRAME,
                                 breathe).astype(np.float32),
                       far_radius) * (1.0 - blurred_alpha)
        )
        frames.append(np.clip(crop(composited), 0, 255).astype(np.uint8))

    reference_alpha = viewpoint(alpha, 0.0)[..., None]
    truth = (
        viewpoint(foreground, 0.0).astype(np.float32) * reference_alpha
        + viewpoint(background, 0.0).astype(np.float32) * (1.0 - reference_alpha)
    )
    return frames, np.clip(crop(truth), 0, 255).astype(np.uint8), crop(viewpoint(alpha, 0.0)) > 0.5


def plane_residual(reference: np.ndarray, moved: np.ndarray, mask: np.ndarray) -> float:
    """Leftover translation within one depth plane — the mechanism measure.

    Independent of the aligner under test: it asks the plane directly whether it
    still moves relative to the reference frame.
    """
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6)
    warp = np.eye(2, 3, dtype=np.float32)
    grey = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            for i in (reference, moved)]
    try:
        _, warp = cv2.findTransformECC(grey[0], grey[1], warp, cv2.MOTION_TRANSLATION,
                                       criteria, mask.astype(np.uint8) * 255, 5)
    except cv2.error:
        return float("nan")
    return float(np.hypot(warp[0, 2], warp[1, 2]))


def evaluate(frames, truth, near_mask, depth_bins, depth_model="bins"):
    aligned, report = align_stack(frames, motion="affine", depth_bins=depth_bins,
                                  depth_model=depth_model, return_report=True)
    x0, y0, x1, y1 = report["crop"]
    usable = report["usable"] if depth_bins else None
    fused = fuse_perband(aligned)
    fused_masked = fuse_perband(aligned, usable=usable)
    truth_cropped = truth[y0:y1, x0:x1]
    near = near_mask[y0:y1, x0:x1]

    residuals = {"near": [], "far": []}
    for index, frame in enumerate(aligned):
        if index == REFERENCE:
            continue
        residuals["near"].append(plane_residual(aligned[REFERENCE], frame, near))
        residuals["far"].append(plane_residual(aligned[REFERENCE], frame, ~near))

    return {
        "gt_ssim": metrics.ref_ssim(fused, truth_cropped),
        "gt_ssim_masked": metrics.ref_ssim(fused_masked, truth_cropped),
        "gt_psnr": metrics.ref_psnr(fused_masked, truth_cropped),
        "withheld": float(np.mean([1.0 - m.mean() for m in usable])) if usable else 0.0,
        "instability": selection_instability_score(aligned),
        "near_px": float(np.nanmean(residuals["near"])),
        "far_px": float(np.nanmean(residuals["far"])),
        "stretch": max([f["stretch"] for f in report["frames"].values()], default=0.0),
        "bins": report["bins"],
    }, fused, truth_cropped


def main() -> None:
    frames, truth, near_mask = build_stack()
    print(f"{'variant':<22} {'GT-SSIM':>9} {'+refusal':>9} {'PSNR':>7} {'near px':>8} "
          f"{'far px':>7} {'withheld':>9} {'bins':>5}")
    for bins, model, label in ((0, "bins", "global affine only"),
                               (3, "bins", "depth-binned"),
                               (3, "joint", "joint (experimental)")):
        result, fused, truth_cropped = evaluate(frames, truth, near_mask, bins, model)
        print(f"{label:<22} {result['gt_ssim']:9.6f} {result['gt_ssim_masked']:9.6f} "
              f"{result['gt_psnr']:7.3f} {result['near_px']:8.3f} {result['far_px']:7.3f} "
              f"{result['withheld'] * 100:8.2f}% {result['bins']:5d}")


if __name__ == "__main__":
    main()
