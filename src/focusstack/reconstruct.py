"""Matte-aware boundary reconstruction (post-fusion stage).

Why this exists: at a depth boundary, the lens has ALREADY mixed both sides into
the captured pixels (a defocused foreground spreads over the background), so no
fusion weighting — however perfect — can produce a clean hard edge there; even an
oracle per-pixel decision measures WORSE than soft blending. The only fix is to
re-render the boundary band as a fresh composite:

    out = obs_owner + (1 - alpha) * (far_est - blur(far_est))

with a SHARP matte `alpha` taken from the frame that owns the contour (the frame
where the occluding structure is in focus — its silhouette is physically sharp
there), and the background estimated from its clean pixels nearby (extension, not
division). Everything outside the strong-veil ribbon keeps the base fusion.

The matte is estimated by difference matting: inpaint the owner frame over a
generous support to get a background plate; alpha = normalized |owner - plate| —
non-zero exactly ON the structure, however thin (support-based mattes are always
too fat). Support itself is gated by stack boundary evidence (defocus-robust
max-over-frames edges + focus-winner discontinuities).

Scope (v1): the strongest-single-occluder model per ribbon component — owner via
majority focus-winner; for N>2 frames the most-defocused frame at the support
serves as the background source. Default OFF in the pipeline; enable with
--reconstruct-boundaries.
"""

from __future__ import annotations

import cv2
import numpy as np

from .focus import content_aware_energies
from .fusion import depth_from_focus, guided_filter
from .io import to_gray_float


def _disk_blur(img: np.ndarray, radius: float) -> np.ndarray:
    """Disk (circle-of-confusion) blur; fast downscale approximation for large radii."""
    if radius < 0.6:
        return img.astype(np.float32).copy()
    if radius > 12:
        f = max(1, int(radius / 6))
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
        small = _disk_blur(small, radius / f)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    r = int(np.ceil(radius))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    k = (yy * yy + xx * xx <= radius * radius).astype(np.float32)
    k /= k.sum()
    return cv2.filter2D(img.astype(np.float32), -1, k)


def _grad_mag(gray: np.ndarray) -> np.ndarray:
    return cv2.magnitude(cv2.Scharr(gray, cv2.CV_32F, 1, 0), cv2.Scharr(gray, cv2.CV_32F, 0, 1))


def _robust_norm(x: np.ndarray, pct: float = 99.0) -> np.ndarray:
    return np.clip(x / (np.percentile(x, pct) + 1e-6), 0.0, 1.0)


def stack_boundary(images: list[np.ndarray]) -> np.ndarray:
    """Soft boundary map from stack evidence (defocus-robust; no appearance model).

    Channels: multi-scale gradient MAX over frames (a contour is sharpest in its
    own focal plane), focus-winner discontinuity density, and focus-depth gradient.
    """
    grays = [to_gray_float(im) for im in images]
    edges = None
    for sigma in (0.0, 2.0, 4.0):
        per = [_grad_mag(g if sigma == 0 else cv2.GaussianBlur(g, (0, 0), sigma)) for g in grays]
        e = _robust_norm(np.maximum.reduce(per))
        edges = e if edges is None else np.maximum(edges, e)
    energies = np.stack(content_aware_energies(grays), 0)
    winner = np.argmax(energies, 0).astype(np.float32)
    wd = (_grad_mag(winner) > 0).astype(np.float32)
    winner_b = _robust_norm(cv2.boxFilter(wd, cv2.CV_32F, (9, 9)))
    depth_b = _robust_norm(_grad_mag(depth_from_focus(images)))
    return np.clip(0.5 * edges + 0.25 * winner_b + 0.4 * depth_b, 0.0, 1.0)


def _estimate_matte(images, radius, ribbon_thresh=0.45):
    """Difference matte + owner index (see module docstring); alpha=0 -> inactive."""
    boundary = stack_boundary(images)
    rib_r = max(2, int(round(0.7 * radius)))
    ribbon = cv2.dilate((boundary >= ribbon_thresh).astype(np.uint8),
                        np.ones((2 * rib_r + 1, 2 * rib_r + 1), np.uint8))
    grays = [to_gray_float(im) for im in images]
    energies = np.stack(content_aware_energies(grays), 0)
    winner = np.argmax(energies, 0)
    e_sorted = np.sort(energies, axis=0)
    dominance = (e_sorted[-1] - e_sorted[-2]) / (e_sorted[-1] + 1e-6)
    raw = ((dominance > 0.35) & (ribbon > 0)).astype(np.float32)
    if raw.sum() < 10:
        return np.zeros_like(raw), 0
    owner = int(np.bincount(winner[raw > 0.5].ravel(), minlength=len(images)).argmax())
    side = ((raw > 0.5) & (winner == owner)).astype(np.float32)
    side = cv2.dilate(side, np.ones((3, 3), np.uint8))
    support_a = np.clip(guided_filter(grays[owner] / 255.0, side, 2, 1e-4), 0.0, 1.0) * (ribbon > 0)
    if support_a.max() <= 0:
        return np.zeros_like(raw), owner
    support = cv2.dilate((support_a > 0.15).astype(np.uint8), np.ones((5, 5), np.uint8))
    plate = cv2.inpaint(images[owner], support, 5, cv2.INPAINT_TELEA).astype(np.float32)
    diff = np.abs(images[owner].astype(np.float32) - plate).sum(axis=2) * (support > 0)
    inside = diff[support > 0]
    hi = np.percentile(inside[inside > 0], 90) if (inside > 0).any() else 1.0
    alpha = np.clip(diff / (hi + 1e-6), 0.0, 1.0)
    alpha = np.clip(guided_filter(grays[owner] / 255.0, alpha.astype(np.float32), 1, 1e-4), 0.0, 1.0)
    return alpha.astype(np.float32), owner


def reconstruct_boundaries(
    images: list[np.ndarray],
    fused: np.ndarray,
    radius: float | None = None,
    inpaint_radius: int = 5,
) -> np.ndarray:
    """Re-render the strong-veil boundary band of `fused`; identity elsewhere.

    `radius`: blur scale of the stack (defaults to 0.012 * max dimension). Safe
    no-op when no occluding structure is found.
    """
    if len(images) < 2:
        return fused
    if radius is None:
        radius = 0.012 * max(fused.shape[:2])

    alpha, owner = _estimate_matte(images, radius)
    if alpha.max() <= 0:
        return fused

    if len(images) == 2:
        far_idx = 1 - owner
    else:
        energies = np.stack(content_aware_energies([to_gray_float(im) for im in images]), 0)
        sup = alpha > 0.15
        far_idx = int(np.argmin([energies[k][sup].mean() for k in range(len(images))]))
        if far_idx == owner:
            return fused

    owner_f = images[owner].astype(np.float32)
    far_f = images[far_idx]

    veil = _disk_blur(alpha, 0.7 * radius)
    strong = (veil > 0.15).astype(np.uint8)
    if strong.sum() == 0:
        return fused
    far_ext = cv2.inpaint(far_f, strong, inpaint_radius, cv2.INPAINT_TELEA).astype(np.float32)
    v = np.clip((veil - 0.15) / 0.5, 0.0, 1.0)[..., None]
    far_est = (1.0 - v) * far_f.astype(np.float32) + v * far_ext

    far_est_blur = np.stack([_disk_blur(far_est[..., c], 0.75 * radius) for c in range(3)], axis=2)
    recon = owner_f + (1.0 - alpha[..., None]) * (far_est - far_est_blur)

    m = cv2.GaussianBlur(strong.astype(np.float32), (0, 0), 1.5)[..., None]
    out = m * recon + (1.0 - m) * fused.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)
