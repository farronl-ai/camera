"""Stage 3 — fusion (combining the sharp content).

Two strategies live here:

`fuse_max` — the intuitive baseline. For every pixel, look across all frames,
find the one with the highest focus score, and copy that pixel. Simple and
correct in spirit, but it decides in the *spatial* domain, so noise can flip the
winner from pixel to pixel and object boundaries (where two focus regions meet)
can show visible seams.

`fuse_pyramid` — the robust, standard approach: Laplacian-pyramid fusion. Instead
of choosing whole pixels, we decompose each image into a stack of frequency
*bands* (a Laplacian pyramid) and choose, band by band, the content with the most
energy. Fine detail (high-frequency bands) is taken from whichever frame is sharp
there; smooth low-frequency content is blended. Collapsing the fused pyramid back
yields a seamless, halo-resistant result. This is the same multi-scale idea
behind exposure fusion.
"""

from __future__ import annotations

import cv2
import numpy as np


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


def _laplacian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    """Decompose a float32 image into `levels` detail bands + 1 base band.

    Level i (for i < levels) holds the detail lost when going from resolution i
    to i+1 (a band-pass image). The final entry is the smallest Gaussian image
    (the low-frequency residual). Reconstruction is exact up to rounding.
    """
    gaussian = [img]
    for _ in range(levels):
        gaussian.append(cv2.pyrDown(gaussian[-1]))

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
