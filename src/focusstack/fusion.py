"""Stage 3 — fusion (combining the sharp content).

Three strategies live here:

`fuse_max` — the intuitive baseline. For every pixel, look across all frames,
find the one with the highest focus score, and copy that pixel. Simple and
correct in spirit, but it decides in the *spatial* domain, so noise can flip the
winner from pixel to pixel and object boundaries (where two focus regions meet)
can show visible seams.

`fuse_pyramid` — the classic multi-scale approach: Laplacian-pyramid fusion.
Instead of choosing whole pixels, we decompose each image into a stack of
frequency *bands* (a Laplacian pyramid) and choose, band by band, the content
with the most energy. Fine detail (high-frequency bands) is taken from whichever
frame is sharp there; smooth low-frequency content is blended. Seamless, but on
real photos it can *ring* — leave a bright halo around thin high-contrast objects
at a focus boundary, because a defocused edge's energy bleeds into lower bands.

`fuse_decision` — guided-filter decision-map fusion (the best all-rounder on real
data). We decide per pixel which frame is in focus (argmax of the focus measure),
then clean that decision with a *guided filter* so it snaps to real object edges
and drops the speckle that plagues `fuse_max`. Result: crisp like max, clean like
pyramid, no halo. This is the idea behind the well-known GFF method.

`fuse_blend` — guided multi-band blending: the synthesis of the two above. It
takes `fuse_decision`'s edge-aware weight map but applies it *per pyramid band*
(Burt & Adelson multiresolution blending), so the boundary correctness of the
decision map meets the multi-scale, seamless reconstruction of the pyramid. No
halo (one coherent mask governs every band) and no seam (the mask is blurred to
each band's scale).

The pyramid method is the same multi-scale idea behind exposure fusion.
"""

from __future__ import annotations

import cv2
import numpy as np

from .focus import content_aware_energies, focus_measure
from .io import to_gray_float


# --------------------------------------------------------------------------- #
# Baseline: per-pixel maximum-sharpness selection
# --------------------------------------------------------------------------- #
def fuse_max(
    images: list[np.ndarray], focus_maps: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Pick each output pixel from the frame with the highest focus score there.

    Returns (fused BGR uint8 image, index_map) where index_map[y, x] is the frame
    index that won at that pixel — useful for debugging/visualization.
    """
    focus_stack = np.stack(focus_maps, axis=0)          # (N, H, W)
    index_map = np.argmax(focus_stack, axis=0).astype(np.int32)  # (H, W)

    image_stack = np.stack(images, axis=0)              # (N, H, W, 3)
    h, w = index_map.shape
    yy, xx = np.indices((h, w))
    # Advanced indexing: for every (y, x) grab the winning frame's BGR triple.
    fused = image_stack[index_map, yy, xx]              # (H, W, 3)
    return fused.astype(np.uint8), index_map


# --------------------------------------------------------------------------- #
# Laplacian-pyramid fusion
# --------------------------------------------------------------------------- #
def _auto_levels(shape: tuple[int, ...], requested: int | None) -> int:
    """Pick a sane pyramid depth; each level halves the resolution."""
    h, w = shape[:2]
    hard_cap = int(np.floor(np.log2(max(1, min(h, w))))) - 1  # keep base >= ~2px
    hard_cap = max(1, hard_cap)
    if requested is None:
        return min(hard_cap, 6)
    return max(1, min(requested, hard_cap))


def _gaussian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    """Blur-and-halve `levels` times; returns `levels + 1` images, coarse-last.

    Used both to decompose an image (via the Laplacian pyramid below) and to
    blur a weight *mask* to each band's scale for multi-band blending. Because
    both go through the same `cv2.pyrDown`, a mask pyramid and an image pyramid
    built from the same starting size have matching sizes at every level.
    """
    gaussian = [img]
    for _ in range(levels):
        gaussian.append(cv2.pyrDown(gaussian[-1]))
    return gaussian


def _laplacian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    """Decompose a float32 image into `levels` detail bands + 1 base band.

    Level i (for i < levels) holds the detail lost when going from resolution i
    to i+1 (a band-pass image). The final entry is the smallest Gaussian image
    (the low-frequency residual). Reconstruction is exact up to rounding.
    """
    gaussian = _gaussian_pyramid(img, levels)

    pyramid = []
    for i in range(levels):
        size = (gaussian[i].shape[1], gaussian[i].shape[0])  # (w, h)
        upsampled = cv2.pyrUp(gaussian[i + 1], dstsize=size)  # dstsize handles odd dims
        pyramid.append(gaussian[i] - upsampled)               # band-pass detail
    pyramid.append(gaussian[-1])                              # low-frequency base
    return pyramid


def fuse_pyramid(
    images: list[np.ndarray], levels: int | None = None, energy_ksize: int = 7
) -> np.ndarray:
    """Fuse a stack via Laplacian-pyramid maximum-energy selection.

    Args:
        images: BGR frames (any dtype; treated as float32 internally).
        levels: pyramid depth; None picks automatically from image size.
        energy_ksize: window for pooling detail energy before selecting a winner.

    Returns:
        fused BGR uint8 image.
    """
    floats = [img.astype(np.float32) for img in images]
    n = len(floats)
    levels = _auto_levels(floats[0].shape, levels)

    pyramids = [_laplacian_pyramid(im, levels) for im in floats]
    n_bands = levels + 1  # detail bands + base

    fused_bands: list[np.ndarray] = []
    for band in range(n_bands):
        coeffs = np.stack([pyramids[k][band] for k in range(n)], axis=0)  # (N, h, w, 3)

        if band < levels:
            # Detail band: keep the frame with the most local energy at each pixel.
            # Crucially, we reduce energy to a single scalar per pixel by summing
            # over the color channels, so ALL three channels are taken from the
            # same source frame — otherwise channels could be picked from
            # different frames and produce color fringing.
            squared = (coeffs ** 2).sum(axis=3)  # (N, h, w)
            energy = np.stack(
                [cv2.boxFilter(squared[k], cv2.CV_32F, (energy_ksize, energy_ksize))
                 for k in range(n)],
                axis=0,
            )
            idx = np.argmax(energy, axis=0)      # (h, w)
            hh, ww = idx.shape
            yy, xx = np.indices((hh, ww))
            fused_bands.append(coeffs[idx, yy, xx])  # (h, w, 3)
        else:
            # Base band is low-frequency and nearly identical across a focus
            # stack, so a plain average is stable and avoids low-freq ghosting.
            fused_bands.append(coeffs.mean(axis=0))

    # Collapse: start from the base and add each detail band back in, upsampling.
    result = fused_bands[-1]
    for band in range(levels - 1, -1, -1):
        size = (fused_bands[band].shape[1], fused_bands[band].shape[0])
        result = cv2.pyrUp(result, dstsize=size) + fused_bands[band]

    return np.clip(result, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Guided-filter decision-map fusion
# --------------------------------------------------------------------------- #
def _box(x: np.ndarray, radius: int) -> np.ndarray:
    """Normalized box (mean) filter — the workhorse of the guided filter."""
    return cv2.boxFilter(x, cv2.CV_32F, (2 * radius + 1, 2 * radius + 1), normalize=True)


def guided_filter(
    guide: np.ndarray, src: np.ndarray, radius: int = 8, eps: float = 1e-3
) -> np.ndarray:
    """Edge-preserving filter (He et al. 2010).

    Smooths `src` while keeping edges that exist in `guide`. Intuition: within
    each local window it fits `src` as a *linear function of the guide*,
    ``out = a * guide + b``. Where the guide is flat, `a -> 0` and the window is
    just averaged (smoothing); where the guide has an edge, `a` is large so the
    output follows that edge. `eps` sets how much guide-variance counts as "flat".

    Both inputs are float32, same HxW. Returns float32.
    """
    mean_i = _box(guide, radius)
    mean_p = _box(src, radius)
    mean_ii = _box(guide * guide, radius)
    mean_ip = _box(guide * src, radius)

    var_i = mean_ii - mean_i * mean_i          # local variance of the guide
    cov_ip = mean_ip - mean_i * mean_p         # local covariance guide<->src

    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return _box(a, radius) * guide + _box(b, radius)


def _guided_weights(
    images: list[np.ndarray], focus_method: str, radius: int, eps: float,
    smooth_ksize: int = 9, harden: float = 0.0,
) -> np.ndarray:
    """Per-frame, edge-aware fusion weight maps summing to 1 at every pixel.

    Raw per-pixel focus argmax -> a one-hot-ish decision per frame -> refine each
    with a guided filter (guided by that frame's luminance) so the boundary snaps
    to real edges and speckle is smoothed -> normalize across frames.

    `harden` (0..1) enables defocus-spread rejection: where one frame is
    CONFIDENTLY the sharpest (focus-energy dominance high — thin/bright
    structures), the soft guided weight is pushed back toward a hard one-hot
    decision so the other frame's defocus spread (a dim, wide blob) can't bleed
    in. Soft blending is kept where focus is ambiguous (smooth regions). 0 = off
    (default; identical to before).

    Returns an array of shape (N, H, W), float32. Shared by `fuse_decision`
    (blend in image space) and `fuse_blend` (blend per pyramid band).
    """
    n = len(images)
    if focus_method == "content_aware":
        # Routes laplacian<->mod_laplacian per pixel by local contrast; needs all frames.
        focus = content_aware_energies([to_gray_float(img) for img in images], smooth_ksize=smooth_ksize)
    else:
        focus = [focus_measure(to_gray_float(img), method=focus_method, smooth_ksize=smooth_ksize)
                 for img in images]
    energy = np.stack(focus, axis=0)
    winner = np.argmax(energy, axis=0)  # (H, W)

    conf = None
    if harden > 0:
        srt = np.sort(energy, axis=0)
        conf = (srt[-1] - srt[-2]) / (srt[-1] + 1e-6)            # dominance of the sharpest frame
        conf = np.clip(cv2.boxFilter(conf.astype(np.float32), cv2.CV_32F, (15, 15)) * harden, 0.0, 1.0)

    weights = []
    for k in range(n):
        raw = (winner == k).astype(np.float32)           # this frame's raw decision
        guide = to_gray_float(images[k]) / 255.0
        wg = np.clip(guided_filter(guide, raw, radius, eps), 0.0, None)
        if conf is not None:
            wg = (1.0 - conf) * wg + conf * raw          # harden toward hard-select where confident
        weights.append(wg)

    w = np.stack(weights, axis=0)
    return w / (w.sum(axis=0, keepdims=True) + 1e-8)     # partition of unity


def fuse_decision(
    images: list[np.ndarray],
    focus_method: str = "laplacian",
    radius: int = 8,
    eps: float = 1e-3,
    smooth_ksize: int = 9,
    harden: float = 0.0,
    return_weights: bool = False,
):
    """Guided-filter decision-map fusion.

    Compute edge-aware weight maps (see `_guided_weights`) and take a per-pixel
    weighted average of the frames in image space. Crisp and halo-free, but the
    blend happens at a single scale. `harden` (0..1) enables defocus-spread
    rejection (see `_guided_weights`).

    Returns the fused BGR uint8 image, or (fused, weights) if `return_weights`.
    """
    floats = [img.astype(np.float32) for img in images]
    n = len(images)

    w = _guided_weights(images, focus_method, radius, eps, smooth_ksize, harden)

    fused = sum(w[k][..., None] * floats[k] for k in range(n))
    fused = np.clip(fused, 0, 255).astype(np.uint8)

    if return_weights:
        return fused, w
    return fused


# --------------------------------------------------------------------------- #
# Guided multi-band blending (pyramid + edge-aware weights)
# --------------------------------------------------------------------------- #
def fuse_blend(
    images: list[np.ndarray],
    focus_method: str = "laplacian",
    levels: int | None = None,
    radius: int = 8,
    eps: float = 1e-3,
    smooth_ksize: int = 9,
    harden: float = 0.0,
    return_weights: bool = False,
):
    """Guided multi-band (Laplacian-pyramid) blending.

    The synthesis of `fuse_pyramid` and `fuse_decision`: one edge-aware weight
    map drives the blend at *every* pyramid level (Burt & Adelson multiresolution
    blending with a guided-filter mask).

    Why it beats both:
      - No halo: because a single coherent mask governs all bands, no coarse band
        can independently select the defocused frame in a ring around an object
        (which is how `fuse_pyramid` gets its halo).
      - No seam: the mask itself is put through a Gaussian pyramid, so its
        transition width is scale-appropriate per band — fine bands switch
        sharply, coarse bands switch gradually. Blending per-band-then-collapsing
        is strictly smoother at boundaries than the single-scale blend in
        `fuse_decision`.

    Returns fused BGR uint8, or (fused, full-res weights) if `return_weights`.
    """
    weights = _guided_weights(images, focus_method, radius, eps, smooth_ksize, harden)  # (N, H, W)
    fused = multiband_blend(images, weights, levels)
    if return_weights:
        return fused, weights
    return fused


def multiband_blend(
    images: list[np.ndarray], weights: np.ndarray, levels: int | None = None
) -> np.ndarray:
    """Burt & Adelson multiresolution blend of `images` with per-frame `weights`.

    Blend each image's Laplacian band using the weight mask blurred (Gaussian
    pyramid) to that band's scale, then collapse. Weights: array (N, H, W); they
    are renormalized per band so any downsampling drift can't shift brightness.

    Reused by `fuse_blend` (weights = guided focus map) and by the region-adaptive
    engine (weights = which differently-tuned candidate wins each pixel).
    """
    floats = [img.astype(np.float32) for img in images]
    n = len(floats)
    levels = _auto_levels(floats[0].shape, levels)

    image_pyramids = [_laplacian_pyramid(im, levels) for im in floats]
    weight_pyramids = [_gaussian_pyramid(weights[k], levels) for k in range(n)]

    fused_bands: list[np.ndarray] = []
    for band in range(levels + 1):
        denom = sum(weight_pyramids[k][band] for k in range(n)) + 1e-8
        blended = sum(
            (weight_pyramids[k][band] / denom)[..., None] * image_pyramids[k][band]
            for k in range(n)
        )
        fused_bands.append(blended)

    result = fused_bands[-1]
    for band in range(levels - 1, -1, -1):
        size = (fused_bands[band].shape[1], fused_bands[band].shape[0])
        result = cv2.pyrUp(result, dstsize=size) + fused_bands[band]

    return np.clip(result, 0, 255).astype(np.uint8)
