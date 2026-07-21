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

`fuse_perband` — per-band edge-aware fusion: makes the focus decision AND an
edge-aware guided weight at EACH pyramid band (not one global weight like blend),
so the decision is multi-scale like pyramid AND halo-free like blend. A fixed small
guided radius per band => effective scale grows with resolution automatically (no
magic number). Best across resolutions; recommended for high-res.

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
    guide: np.ndarray, src: np.ndarray, radius: int = 8, eps=1e-3
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

    # `eps` may be a scalar OR a per-pixel map (boundary-aware smoothing: small
    # eps preserves structure at true boundaries, large eps smooths within objects).
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return _box(a, radius) * guide + _box(b, radius)


def _guided_weights(
    images: list[np.ndarray], focus_method: str, radius: int, eps: float,
    smooth_ksize: int = 9, harden: float = 0.0, guide_scale: float = 1.0,
) -> np.ndarray:
    """Per-frame, edge-aware fusion weight maps summing to 1 at every pixel.

    Raw per-pixel focus argmax -> a one-hot-ish decision per frame -> refine each
    with a guided filter (guided by that frame's luminance) so the boundary snaps
    to real edges and speckle is smoothed -> normalize across frames.

    `harden` (0..1) enables defocus-spread rejection: where one frame is
    CONFIDENTLY the sharpest (focus-energy dominance high — thin/bright
    structures), the soft guided weight is pushed back toward a hard one-hot
    decision so the other frame's defocus spread (a dim, wide blob) can't bleed
    in. Soft blending is kept where focus is ambiguous (smooth regions). 0 = off.

    `guide_scale` (<1) speeds high-res fusion: the guided-filter *smoothing* is the
    costly stage and its output is low-frequency, so it is computed on a downscaled
    guide/decision and upsampled — near-lossless. Crucially the focus energy,
    confidence, and raw one-hot decision stay at FULL resolution, so `harden` can
    still hard-select thin/bright structures (which a downscaled decision would
    lose). Full-res focus is cheap relative to the guided filter.

    `radius`/`smooth_ksize` = None → **resolution-adaptive**: scaled to ~0.012·max
    dimension (≈ the circle-of-confusion at typical defocus), floored to the classic
    8/9. At low res this floor makes it identical to the old fixed default (no
    regression); at high res it grows so the smooth weight matches the large-CoC
    structure instead of blending in blur (fixed 8px is far too small at 3–4K — this
    was why blend lost to pyramid at high res, F18).

    Returns an array of shape (N, H, W), float32. Shared by `fuse_decision`
    (blend in image space) and `fuse_blend` (blend per pyramid band).
    """
    n = len(images)
    if radius is None or smooth_ksize is None:
        auto = int(round(0.012 * max(images[0].shape[:2])))
        if radius is None:
            radius = max(8, auto)
        if smooth_ksize is None:
            smooth_ksize = max(9, auto | 1)  # odd
    if focus_method == "content_aware":
        # Routes laplacian<->mod_laplacian per pixel by local contrast; needs all frames.
        focus = content_aware_energies([to_gray_float(img) for img in images], smooth_ksize=smooth_ksize)
    else:
        focus = [focus_measure(to_gray_float(img), method=focus_method, smooth_ksize=smooth_ksize)
                 for img in images]
    energy = np.stack(focus, axis=0)
    winner = np.argmax(energy, axis=0)  # (H, W)  — full res

    conf = None
    if harden > 0:
        srt = np.sort(energy, axis=0)
        conf = (srt[-1] - srt[-2]) / (srt[-1] + 1e-6)            # dominance of the sharpest frame
        conf = np.clip(cv2.boxFilter(conf.astype(np.float32), cv2.CV_32F, (15, 15)) * harden, 0.0, 1.0)

    h, w0 = winner.shape
    sub = guide_scale < 0.999
    if sub:
        sw, sh = max(8, int(w0 * guide_scale)), max(8, int(h * guide_scale))

    weights = []
    for k in range(n):
        raw = (winner == k).astype(np.float32)           # full-res one-hot decision
        guide = to_gray_float(images[k]) / 255.0
        if sub:
            # Subscale ONLY the smooth guided-filter step, then upsample.
            g_s = cv2.resize(guide, (sw, sh), interpolation=cv2.INTER_AREA)
            r_s = cv2.resize(raw, (sw, sh), interpolation=cv2.INTER_AREA)
            wg = cv2.resize(guided_filter(g_s, r_s, radius, eps), (w0, h), interpolation=cv2.INTER_LINEAR)
        else:
            wg = guided_filter(guide, raw, radius, eps)
        wg = np.clip(wg, 0.0, None)
        if conf is not None:
            wg = (1.0 - conf) * wg + conf * raw          # full-res hard-select where confident
        weights.append(wg)

    w = np.stack(weights, axis=0)
    return w / (w.sum(axis=0, keepdims=True) + 1e-8)     # partition of unity


def _weights(images, focus_method, radius, eps, smooth_ksize, harden, weight_scale=1.0):
    """Fusion weights; `weight_scale` < 1 subscales the costly guided-filter step
    for a high-res speedup while keeping focus/confidence/decision full-res so thin
    structures (and `harden`) are preserved. See `_guided_weights` guide_scale."""
    return _guided_weights(images, focus_method, radius, eps, smooth_ksize, harden,
                           guide_scale=weight_scale)


def fuse_decision(
    images: list[np.ndarray],
    focus_method: str = "content_aware",
    radius: int | None = None,
    eps: float = 1e-3,
    smooth_ksize: int | None = None,
    harden: float = 0.0,
    weight_scale: float = 1.0,
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

    w = _weights(images, focus_method, radius, eps, smooth_ksize, harden, weight_scale)

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
    focus_method: str = "content_aware",
    levels: int | None = None,
    radius: int | None = None,
    eps: float = 1e-3,
    smooth_ksize: int | None = None,
    harden: float = 0.0,
    weight_scale: float = 1.0,
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
    weights = _weights(images, focus_method, radius, eps, smooth_ksize, harden, weight_scale)  # (N, H, W)
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


def depth_from_focus(
    images: list[np.ndarray],
    focus_method: str = "content_aware",
    smooth_ksize: int = 9,
    radius: int = 8,
    eps: float = 1e-3,
) -> np.ndarray:
    """Depth map as a free byproduct of the fusion decision (depth-from-focus).

    In a focus stack, WHERE each frame wins the sharpness contest encodes depth:
    frame k is sharpest where the scene sits near frame k's focal plane. So the
    per-pixel winner index, scaled to [0, 1] by frame order (near -> far if the
    stack is ordered), is a coarse depth map. A guided filter (guided by the
    locally-sharpest luminance) snaps it to object boundaries and smooths the
    speckle, exactly as it does for fusion weights.

    Returns float32 (H, W) in [0, 1]. More frames -> finer depth quantization.
    """
    n = len(images)
    if focus_method == "content_aware":
        focus = content_aware_energies([to_gray_float(img) for img in images],
                                       smooth_ksize=smooth_ksize)
    else:
        focus = [focus_measure(to_gray_float(img), method=focus_method,
                               smooth_ksize=smooth_ksize) for img in images]
    energy = np.stack(focus, axis=0)
    winner = np.argmax(energy, axis=0)
    depth = winner.astype(np.float32) / max(1, n - 1)

    # Edge-aware smoothing, guided by the sharpest-available luminance.
    hh, ww = winner.shape
    yy, xx = np.indices((hh, ww))
    sharp_lum = np.stack([to_gray_float(im) for im in images], 0)[winner, yy, xx] / 255.0
    return np.clip(guided_filter(sharp_lum.astype(np.float32), depth, radius, eps), 0.0, 1.0)


def fuse_perband(
    images: list[np.ndarray],
    radius: int = 6,
    eps: float = 1e-3,
    energy_ksize: int = 7,
    harden: float = 0.0,
    boundary: np.ndarray | None = None,
    b_lambda: float = 0.5,
    b_eps_gain: float = 4.0,
    veil_D: np.ndarray | None = None,
    veil_far_idx: int = -1,
) -> np.ndarray:
    """Per-band edge-aware fusion — pyramid's multi-scale DECISION + guided (halo-free).

    `fuse_blend` makes ONE guided weight map (a single-scale decision) and broadcasts
    it across all bands. `fuse_pyramid` decides per band (multi-scale) but by hard
    max-energy, which HALOS at high-contrast focus boundaries. This does both right:
    at EACH Laplacian band it makes the focus decision from that band's own energy AND
    refines it with a guided filter (guided by that band's Gaussian image) — so the
    decision is multi-scale AND edge-aware/halo-free.

    A FIXED small `radius` per band is the elegant part: because coarser bands are
    downsampled, the effective full-resolution radius grows automatically with scale
    ("start at the finest pixels, move up") — multi-scale by construction, no
    resolution-dependent magic number. Best-of-both: best at high-res AND low-res,
    nearly as halo-free as `fuse_blend`.

    Two correctness details (each was a measured defect before being fixed):
      - On COARSE bands the window must shrink with the band, or a radius bigger
        than the band degenerates the guided filter into a global mean (~50/50
        blending that imports defocused energy). Both `radius` and `energy_ksize`
        are capped relative to the band's size.
      - The BASE band is not averaged: defocus *spread* (a bright/dark smear from
        the out-of-focus frame) lives heavily in low frequencies, so averaging
        pulls it in. Instead the coarsest detail band's weights are propagated
        down (pyrDown) and used to blend the base.
    """
    floats = [img.astype(np.float32) for img in images]
    n = len(floats)
    levels = _auto_levels(floats[0].shape, None)
    image_pyramids = [_laplacian_pyramid(im, levels) for im in floats]
    guide_pyramids = [_gaussian_pyramid(to_gray_float(f), levels) for f in images]

    # Optional boundary map B in [0,1] (from the boundary engine): consumed as its
    # OWN guide component (F30: never filter B through luminance alone, or the
    # integration collapses back to appearance) and as an eps modulator (preserve
    # decisions at true boundaries, smooth harder within objects). boundary=None
    # -> byte-identical to the plain path.
    b_pyr = None
    if boundary is not None:
        b_pyr = _gaussian_pyramid(np.clip(boundary.astype(np.float32), 0.0, 1.0), levels)

    # Optional veil correction (F40/F41): `veil_D` is the forward-simulated haze
    # field of the frame at `veil_far_idx`; haze enters the output only through
    # that frame's per-band weights, so we subtract w_far * L_k(D) at EVERY band
    # (subtraction of a simulated field has no division -> no F27 amplification).
    # veil_D=None -> byte-identical to the plain path.
    d_pyr = None
    if veil_D is not None:
        d_pyr = _laplacian_pyramid(veil_D.astype(np.float32), levels)
        if veil_far_idx < 0:
            veil_far_idx = n - 1

    fused_bands: list[np.ndarray] = []
    w_last: np.ndarray | None = None
    for band in range(levels + 1):
        coeffs = [image_pyramids[k][band] for k in range(n)]
        bh, bw = coeffs[0].shape[:2]
        if band < levels:
            # Cap window sizes to the band: a window ~the whole band degenerates
            # the decision into a global mean.
            r_b = max(1, min(radius, min(bh, bw) // 6))
            k_b = max(3, min(energy_ksize, (min(bh, bw) // 4) | 1))
            Bb = None
            eps_b = eps
            if b_pyr is not None:
                Bb = b_pyr[band]
                Bb = Bb / (float(Bb.max()) + 1e-6)   # restore contrast lost to pyrDown
                eps_b = eps * (1.0 + b_eps_gain * (1.0 - np.minimum(1.0, 3.0 * Bb)))
            energy = np.stack(
                [cv2.boxFilter((coeffs[k] ** 2).sum(axis=2), cv2.CV_32F, (k_b, k_b))
                 for k in range(n)], axis=0)
            winner = np.argmax(energy, axis=0)
            conf = None
            if harden > 0:
                srt = np.sort(energy, axis=0)
                conf = np.clip((srt[-1] - srt[-2]) / (srt[-1] + 1e-6), 0.0, 1.0)
            weights = []
            for k in range(n):
                raw = (winner == k).astype(np.float32)
                g = guide_pyramids[k][band] / 255.0
                if Bb is not None:
                    g = (1.0 - b_lambda) * g + b_lambda * Bb   # B as its own guide component
                wg = np.clip(guided_filter(g, raw, r_b, eps_b), 0.0, None)
                if conf is not None:
                    wg = (1.0 - conf) * wg + conf * raw
                weights.append(wg)
            w = np.stack(weights, axis=0)
            w /= (w.sum(axis=0, keepdims=True) + 1e-8)
            fb = sum(w[k][..., None] * coeffs[k] for k in range(n))
            if d_pyr is not None:
                fb = fb - w[veil_far_idx][..., None] * d_pyr[band]
            fused_bands.append(fb)
            w_last = w
        else:
            # Base band: blend with the coarsest detail weights propagated down —
            # NOT a plain mean, which would import the defocused frame's
            # low-frequency spread.
            wb = np.stack([cv2.pyrDown(w_last[k]) for k in range(n)], axis=0)
            wb = np.clip(wb, 0.0, None)
            wb /= (wb.sum(axis=0, keepdims=True) + 1e-8)
            fb = sum(wb[k][..., None] * coeffs[k] for k in range(n))
            if d_pyr is not None:
                fb = fb - wb[veil_far_idx][..., None] * d_pyr[levels]
            fused_bands.append(fb)

    result = fused_bands[-1]
    for band in range(levels - 1, -1, -1):
        size = (fused_bands[band].shape[1], fused_bands[band].shape[0])
        result = cv2.pyrUp(result, dstsize=size) + fused_bands[band]
    return np.clip(result, 0, 255).astype(np.uint8)
