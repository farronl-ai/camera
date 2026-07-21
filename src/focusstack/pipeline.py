"""Orchestration: run the three stages end-to-end.

    load  ->  align  ->  (focus measure)  ->  fuse  ->  save

`run()` is the single entry point used by both the CLI and library callers. When
`debug_dir` is set it also writes intermediate visualizations so you can *see*
what each stage did (aligned frames, focus maps, and — for the max method — a
colorized map of which source frame won each pixel).
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from . import io as fio
from .align import align_stack
from .focus import content_aware_energies, focus_measures
from .fusion import (depth_from_focus, fuse_blend, fuse_decision, fuse_max,
                     fuse_perband, fuse_pyramid)


def _focus_maps(images: list[np.ndarray], method: str) -> list[np.ndarray]:
    """Per-frame focus maps; content_aware needs all frames (cross-frame contrast)."""
    if method == "content_aware":
        return content_aware_energies([fio.to_gray_float(im) for im in images])
    return focus_measures(images, method=method)


def _normalize_map(m: np.ndarray) -> np.ndarray:
    """Scale a float map to a 0-255 uint8 image for viewing."""
    lo, hi = float(m.min()), float(m.max())
    if hi - lo < 1e-9:
        return np.zeros(m.shape, dtype=np.uint8)
    return ((m - lo) / (hi - lo) * 255.0).astype(np.uint8)


def _colorize_selection(index_map: np.ndarray, n: int) -> np.ndarray:
    """Map frame-index -> color so each source region is a distinct hue."""
    scaled = (index_map.astype(np.float32) / max(1, n - 1) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_JET)


def run(
    inputs: list[str],
    output: str,
    method: str = "perband",
    align: bool = True,
    align_motion: str = "affine",
    focus_method: str = "content_aware",
    levels: int | None = None,
    harden: float = 0.0,
    weight_scale: float = 1.0,
    normalize_exposure: bool = True,
    reconstruct_boundaries: bool = False,
    depth_out: str | None = None,
    debug_dir: str | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Run focus stacking on `inputs` and write the result to `output`.

    Returns the fused BGR uint8 image.
    """

    def log(msg: str) -> None:
        if verbose:
            print(f"[focusstack] {msg}")

    named = fio.load_images(inputs)
    names = [name for name, _ in named]
    images = [img for _, img in named]
    log(f"loaded {len(images)} frames: {', '.join(names)}")

    if align:
        log(f"aligning frames (motion={align_motion}) ...")
        images = align_stack(images, motion=align_motion)

    if normalize_exposure:
        log("normalizing per-frame exposure/WB drift ...")
        images = fio.normalize_exposure(images)

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        for name, img in zip(names, images):
            stem = os.path.splitext(name)[0]
            fio.save_image(os.path.join(debug_dir, f"aligned_{stem}.png"), img)

    if method == "max":
        log(f"computing focus maps (measure={focus_method}) ...")
        fmaps = _focus_maps(images, focus_method)
        log("fusing (per-pixel maximum sharpness) ...")
        fused, index_map = fuse_max(images, fmaps)
        if debug_dir:
            for name, fm in zip(names, fmaps):
                stem = os.path.splitext(name)[0]
                fio.save_image(os.path.join(debug_dir, f"focus_{stem}.png"), _normalize_map(fm))
            fio.save_image(
                os.path.join(debug_dir, "selection.png"),
                _colorize_selection(index_map, len(images)),
            )
    elif method == "pyramid":
        log("fusing (Laplacian pyramid) ...")
        fused = fuse_pyramid(images, levels=levels)
        if debug_dir:
            # Focus maps aren't used by the pyramid method, but dump them anyway
            # so the sharpness of each frame is visible alongside the result.
            for name, fm in zip(names, _focus_maps(images, focus_method)):
                stem = os.path.splitext(name)[0]
                fio.save_image(os.path.join(debug_dir, f"focus_{stem}.png"), _normalize_map(fm))
    elif method == "decision":
        log("fusing (guided-filter decision map) ...")
        fused, weights = fuse_decision(images, focus_method=focus_method, harden=harden,
                                       weight_scale=weight_scale, return_weights=True)
        if debug_dir:
            # The refined per-frame weight maps are the heart of this method —
            # dump them so the (clean, edge-aligned) selection is visible.
            for name, wmap in zip(names, weights):
                stem = os.path.splitext(name)[0]
                fio.save_image(os.path.join(debug_dir, f"weight_{stem}.png"), _normalize_map(wmap))
    elif method == "blend":
        log("fusing (guided multi-band blend) ...")
        fused, weights = fuse_blend(
            images, focus_method=focus_method, levels=levels, harden=harden,
            weight_scale=weight_scale, return_weights=True
        )
        if debug_dir:
            for name, wmap in zip(names, weights):
                stem = os.path.splitext(name)[0]
                fio.save_image(os.path.join(debug_dir, f"weight_{stem}.png"), _normalize_map(wmap))
    elif method == "perband":
        log("fusing (per-band edge-aware) ...")
        fused = fuse_perband(images, harden=harden)
    else:
        raise ValueError(
            f"Unknown method {method!r}; use 'blend', 'perband', 'decision', 'pyramid', or 'max'."
        )

    if reconstruct_boundaries:
        from .reconstruct import reconstruct_boundaries as _recon
        log("reconstructing boundary bands (experimental) ...")
        fused = _recon(images, fused)

    if depth_out:
        log("computing depth-from-focus map ...")
        d = depth_from_focus(images, focus_method=focus_method)
        fio.save_image(depth_out, (d * 255.0).astype(np.uint8))
        log(f"wrote depth map {depth_out}")

    fio.save_image(output, fused)
    log(f"wrote {output}")
    return fused
