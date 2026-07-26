#!/usr/bin/env python3
"""Objects-as-occluders factory with explicit formation-model versions.

The first object factory used the correct premultiplied two-layer equation, but
its large-radius research blur silently changed from a disk to a downscaled box.
It also exposed only the sharp alpha map, which made a truly partial-occlusion
pixel look like an opaque interior in diagnostics.

Historical V2 splits preserve the aperture-average model that produced their
frozen evidence. New primary opaque splits use the user's one-sided ownership
contract:

    near-focus = alpha*N + (1-alpha)*disk(B, r_far)
    S          = pre-antialias binary foreground ownership
    C          = disk(alpha, r_near)
    N_spread   = disk(alpha*N, r_near) / C
    W          = max(S, alpha, C)
    far-focus  = W*N_spread + (1-W)*B

``W`` is a one-sided dilation: foreground blur may extend outward, but it never
opens the latent opaque support and admits rear detail inward. ``S`` is kept
separate because resize antialiasing and a display matte can otherwise turn a
real owned boundary into a partially transparent one before defocus is even
applied. The normalized foreground spread avoids dimming an opaque interior
merely because its PSF extends beyond the support. True transmission remains a
separate renderer and is not part of the current opaque goal.

Foreground RGB is copied from a real segmented object, but the unreliable
source-photo boundary ring is replaced by nearest eroded-interior radiance before
compositing.  Scene strata control how much complete-coverage core survives the
giant CoC instead of accidentally making nearly every object an all-veil case.

Run:
    python research/objocc_v2_gen.py 12 dev
    python research/objocc_v2_gen.py 12 holdout
    python research/objocc_v2_gen.py 12 extension
    python research/objocc_v2_gen.py 36 s12
    python research/objocc_v2_gen.py 36 s16
    python research/objocc_v2_gen.py 72 s19
    python research/objocc_v2_gen.py 72 s23
    python research/objocc_v2_gen.py 36 t24
    python research/objocc_v2_gen.py 12 s25
    python research/objocc_v2_gen.py 36 s26
    python research/objocc_v2_gen.py 36 s27
    python research/objocc_v2_gen.py 36 s28
    python research/objocc_v2_gen.py 12 s29
"""
from __future__ import annotations

import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hires_gen import add_noise  # noqa: E402
from objocc_gen import LONG, good_object_masks  # noqa: E402
from focusstack.reconstruct import _disk_blur as fast_disk_blur  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "hires")
OUT = os.path.join(HERE, "data", "objocc_v2")
COC_FRACTION = 0.035
DEFOCUS_DISTANCE = 0.70
STRATA = ("solid", "mixed", "thin")
OPTICAL_REGIME_BY_STRATUM = {
    "solid": "substantial_coverage_core",
    "mixed": "boundary_dominant_partial_coverage",
    "thin": "all_veil_geometry_stress",
    "primary": "substantial_coverage_core",
    "boundary": "boundary_dominant_partial_coverage",
    "all_veil": "all_veil_geometry_stress",
}
# The first post-audit opaque cohort represents the ordinary target more
# heavily while retaining boundary and all-veil stress. Earlier splits keep the
# equal-cycle schedule that generated their frozen evidence.
PRIMARY_OPAQUE_SCHEDULE = (
    "primary",
    "primary",
    "boundary",
    "primary",
    "boundary",
    "all_veil",
)
# Transmission is isolated from the most severe geometry at first. Varying
# opacity then tests material separation without making every scene also an
# all-veil opaque stress.
TRANSMISSIVE_SCHEDULE = ("primary", "primary", "boundary")
TRANSMISSIVE_OPACITIES = (0.35, 0.55, 0.75)
ONE_SIDED_OPAQUE_SPLITS = {"s25", "s26", "s27", "s28", "s29"}
HARD_OWNERSHIP_OPAQUE_SPLITS = {"s27", "s28", "s29"}
# A primary solid-opaque cohort must not turn a segmenter's internal omission
# into an optical aperture. Tiny enclosed speckles are repaired; a mask with a
# substantial enclosed region is topologically ambiguous and belongs in a
# later explicit aperture/transmission cohort instead.
SOLID_OPAQUE_MASK_SPLITS = {"s27", "s28", "s29"}
SOLID_OPAQUE_MAX_ENCLOSED_HOLE_FRACTION = 0.005


def exact_disk_blur(image: np.ndarray, radius: float) -> np.ndarray:
    """Offline circular-aperture convolution without the old box shortcut."""
    source = np.asarray(image, np.float32)
    if radius < 0.6:
        return source.copy()
    r = int(np.ceil(radius))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    kernel = (xx * xx + yy * yy <= radius * radius).astype(np.float32)
    kernel /= kernel.sum()
    return cv2.filter2D(
        source,
        cv2.CV_32F,
        kernel,
        borderType=cv2.BORDER_REFLECT_101,
    )


def _nearest_safe_radiance(
    image: np.ndarray,
    mask: np.ndarray,
    erosion_radius: int = 3,
) -> np.ndarray:
    """Replace uncertain cutout-edge RGB with nearest eroded-interior RGB."""
    binary = np.asarray(mask) > 0
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * erosion_radius + 1, 2 * erosion_radius + 1),
    )
    safe = cv2.erode(binary.astype(np.uint8), kernel) > 0
    if not np.any(safe):
        safe = binary

    # distanceTransformWithLabels labels each nonzero pixel with its nearest
    # zero pixel.  Encoding safe pixels as zero gives a direct nearest-source
    # lookup for the unreliable boundary and outside padding.
    search = (~safe).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(
        search,
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    safe_y, safe_x = np.where(search == 0)
    nearest = np.clip(labels.astype(np.int64) - 1, 0, len(safe_y) - 1)
    filled = image[safe_y[nearest], safe_x[nearest]].copy()
    filled[safe] = image[safe]
    return filled.astype(np.float32)


def _solid_opaque_source_mask(
    mask: np.ndarray,
) -> tuple[np.ndarray | None, dict]:
    """Repair tiny segmentation holes or reject ambiguous opaque topology."""
    binary = np.asarray(mask) > 0
    contours, _ = cv2.findContours(
        binary.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    filled = np.zeros(binary.shape, np.uint8)
    cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
    enclosed_holes = (filled > 0) & ~binary
    hole_pixels = int(enclosed_holes.sum())
    foreground_pixels = int(binary.sum())
    hole_fraction = hole_pixels / max(foreground_pixels, 1)
    report = {
        "source_mask_contract": "solid_opaque_no_ambiguous_holes_v1",
        "source_mask_enclosed_hole_pixels": hole_pixels,
        "source_mask_enclosed_hole_fraction": float(hole_fraction),
        "source_mask_small_holes_filled": False,
        "source_mask_rejected_ambiguous_holes": False,
    }
    if hole_fraction > SOLID_OPAQUE_MAX_ENCLOSED_HOLE_FRACTION:
        report["source_mask_rejected_ambiguous_holes"] = True
        return None, report
    report["source_mask_small_holes_filled"] = hole_pixels > 0
    return filled > 0, report


def render_focal_pair(
    background: np.ndarray,
    foreground: np.ndarray,
    alpha: np.ndarray,
    max_radius: float,
    *,
    hard_ownership: np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Render the primary one-sided opaque near/far focal pair."""
    rendered = render_one_sided_opaque_focal_pair(
        background,
        foreground,
        alpha,
        max_radius,
        hard_ownership=hard_ownership,
    )
    return rendered["frames"], rendered["geometry_coverage"]


def render_one_sided_opaque_focal_pair(
    background: np.ndarray,
    foreground: np.ndarray,
    geometry_alpha: np.ndarray,
    max_radius: float,
    *,
    hard_ownership: np.ndarray | None = None,
) -> dict:
    """Render opaque defocus without admitting rear detail inside its support.

    The blurred premultiplied foreground is normalized by its blurred geometry
    so defocus changes foreground radiance spatially without making an opaque
    object transparent. The pre-antialias binary ownership support remains
    distinct from the soft display matte: both the full owned support and the
    soft matte are preserved, and the PSF adds only an outward veil.
    """
    bg = np.asarray(background, np.float32)
    fg = np.asarray(foreground, np.float32)
    a = np.clip(np.asarray(geometry_alpha, np.float32), 0.0, 1.0)
    if hard_ownership is None:
        owned = a >= 0.5
    else:
        owned = np.asarray(hard_ownership, bool)
        if owned.shape != a.shape:
            raise ValueError(
                "hard ownership must match geometry alpha shape"
            )
    radius = DEFOCUS_DISTANCE * max_radius

    far_blurred = exact_disk_blur(bg, radius)
    near_frame = a[..., None] * fg + (1.0 - a[..., None]) * far_blurred
    near_coverage = a

    aperture_spread = np.clip(exact_disk_blur(a, radius), 0.0, 1.0)
    blurred_premult = exact_disk_blur(a[..., None] * fg, radius)
    conditional_foreground = np.zeros_like(blurred_premult)
    np.divide(
        blurred_premult,
        aperture_spread[..., None],
        out=conditional_foreground,
        where=aperture_spread[..., None] > 1e-6,
    )
    far_coverage = np.maximum.reduce(
        [a, aperture_spread, owned.astype(np.float32)]
    )
    far_frame = (
        far_coverage[..., None] * conditional_foreground
        + (1.0 - far_coverage[..., None]) * bg
    )
    gt = a[..., None] * fg + (1.0 - a[..., None]) * bg
    return {
        "frames": [near_frame, far_frame],
        "geometry_coverage": [near_coverage, far_coverage],
        "extinction": [near_coverage, far_coverage],
        "aperture_spread": [near_coverage, aperture_spread],
        "hard_ownership": owned,
        "gt": gt,
        "formation_model": "one_sided_opaque_v2",
    }


def _render_one_sided_opaque_focal_pair_v1(
    background: np.ndarray,
    foreground: np.ndarray,
    geometry_alpha: np.ndarray,
    max_radius: float,
) -> dict:
    """Retain the pre-F69 operator solely for frozen S25/S26 reproduction."""
    bg = np.asarray(background, np.float32)
    fg = np.asarray(foreground, np.float32)
    a = np.clip(np.asarray(geometry_alpha, np.float32), 0.0, 1.0)
    radius = DEFOCUS_DISTANCE * max_radius

    far_blurred = exact_disk_blur(bg, radius)
    near_frame = a[..., None] * fg + (1.0 - a[..., None]) * far_blurred
    aperture_spread = np.clip(exact_disk_blur(a, radius), 0.0, 1.0)
    blurred_premult = exact_disk_blur(a[..., None] * fg, radius)
    conditional_foreground = np.zeros_like(blurred_premult)
    np.divide(
        blurred_premult,
        aperture_spread[..., None],
        out=conditional_foreground,
        where=aperture_spread[..., None] > 1e-6,
    )
    far_coverage = np.maximum(a, aperture_spread)
    far_frame = (
        far_coverage[..., None] * conditional_foreground
        + (1.0 - far_coverage[..., None]) * bg
    )
    gt = a[..., None] * fg + (1.0 - a[..., None]) * bg
    return {
        "frames": [near_frame, far_frame],
        "geometry_coverage": [a, far_coverage],
        "extinction": [a, far_coverage],
        "aperture_spread": [a, aperture_spread],
        "gt": gt,
        "formation_model": "one_sided_opaque_v1",
    }


def render_layered_focal_pair(
    background: np.ndarray,
    foreground: np.ndarray,
    geometry_alpha: np.ndarray,
    max_radius: float,
    material_opacity: float | np.ndarray,
) -> dict:
    """Render the historical aperture-mixed / transmissive observations.

    This function is retained for frozen historical splits and later explicit
    transmission research. It is not the primary opaque renderer.

    Geometry answers whether a ray intersects the front object. Extinction
    answers how much radiance that intersection contributes/removes. They are
    identical only for an opaque material. Keeping both fields prevents a
    slender opaque object from being mislabeled as transparent and prevents a
    transmissive object from inheriting an opaque hard-ownership rule.
    """
    bg = np.asarray(background, np.float32)
    fg = np.asarray(foreground, np.float32)
    geometry = np.clip(np.asarray(geometry_alpha, np.float32), 0.0, 1.0)
    opacity = np.asarray(material_opacity, np.float32)
    if opacity.ndim == 0:
        opacity = np.full_like(geometry, float(opacity))
    if opacity.shape != geometry.shape:
        raise ValueError(
            "material opacity must be scalar or match geometry alpha"
        )
    if not np.isfinite(opacity).all() or np.any(
        (opacity < 0.0) | (opacity > 1.0)
    ):
        raise ValueError("material opacity must lie in [0, 1]")

    extinction_near = geometry * opacity
    radius = DEFOCUS_DISTANCE * max_radius
    far_blurred = exact_disk_blur(bg, radius)
    near_frame = (
        extinction_near[..., None] * fg
        + (1.0 - extinction_near[..., None]) * far_blurred
    )
    geometry_far = np.clip(
        exact_disk_blur(geometry, radius),
        0.0,
        1.0,
    )
    extinction_far = np.clip(
        exact_disk_blur(extinction_near, radius),
        0.0,
        1.0,
    )
    far_premult = exact_disk_blur(
        extinction_near[..., None] * fg,
        radius,
    )
    far_frame = far_premult + (1.0 - extinction_far[..., None]) * bg
    gt = (
        extinction_near[..., None] * fg
        + (1.0 - extinction_near[..., None]) * bg
    )
    return {
        "frames": [near_frame, far_frame],
        "geometry_coverage": [geometry, geometry_far],
        "extinction": [extinction_near, extinction_far],
        "aperture_spread": [geometry, geometry_far],
        "gt": gt,
        "formation_model": "aperture_mixed_extinction_v1",
    }


def coverage_stats(alpha: np.ndarray, coverage: np.ndarray) -> dict:
    """Frame-specific core/veil proportions relative to the sharp support."""
    sharp = alpha >= 0.95
    sharp_n = max(int(sharp.sum()), 1)
    core = sharp & (coverage >= 0.95)
    inner_veil = sharp & (coverage > 0.05) & (coverage < 0.95)
    outer_veil = (alpha < 0.05) & (coverage > 0.05)
    return {
        "sharp_pixels": int(sharp.sum()),
        "core_pixels": int(core.sum()),
        "inner_veil_pixels": int(inner_veil.sum()),
        "outer_veil_pixels": int(outer_veil.sum()),
        "core_fraction": float(core.sum() / sharp_n),
        "inner_veil_fraction": float(inner_veil.sum() / sharp_n),
    }


def coverage_classification(
    alpha: np.ndarray,
    coverage: np.ndarray,
) -> np.ndarray:
    """BGR diagnostic: green core, yellow inner veil, magenta outer veil."""
    sharp = alpha >= 0.95
    core = sharp & (coverage >= 0.95)
    inner = sharp & (coverage > 0.05) & (coverage < 0.95)
    outer = (alpha < 0.05) & (coverage > 0.05)
    out = np.zeros((*alpha.shape, 3), np.uint8)
    out[core] = (40, 190, 40)
    out[inner] = (0, 210, 255)
    out[outer] = (210, 50, 210)
    return out


def _stratum_matches(stratum: str, stats: dict) -> bool:
    core = stats["core_fraction"]
    if stratum == "primary":
        return core >= 0.75 and stats["core_pixels"] >= 5000
    if stratum == "boundary":
        return 0.25 <= core < 0.60 and stats["core_pixels"] >= 1500
    if stratum == "all_veil":
        return core < 0.15 and stats["inner_veil_pixels"] >= 2500
    if stratum == "solid":
        return core >= 0.55 and stats["core_pixels"] >= 2500
    if stratum == "mixed":
        return 0.15 <= core < 0.55 and stats["core_pixels"] >= 800
    if stratum == "thin":
        return core < 0.15 and stats["inner_veil_pixels"] >= 2500
    raise ValueError(stratum)


def _source_assets(*, solid_opaque: bool = False):
    assets = []
    for path in sorted(glob.glob(os.path.join(SRC, "*", "gt.png"))):
        mask_path = path + ".masks.npy"
        if not os.path.exists(mask_path):
            continue
        image = cv2.imread(path)
        for index, mask in enumerate(
            good_object_masks(np.load(mask_path), *image.shape[:2])
        ):
            if solid_opaque:
                prepared_mask, mask_report = _solid_opaque_source_mask(mask)
                if prepared_mask is None:
                    continue
                mask = prepared_mask
            else:
                mask_report = {
                    "source_mask_contract": "legacy_semantic_mask",
                    "source_mask_enclosed_hole_pixels": None,
                    "source_mask_enclosed_hole_fraction": None,
                    "source_mask_small_holes_filled": False,
                    "source_mask_rejected_ambiguous_holes": False,
                }
            ys, xs = np.where(mask > 0)
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            crop = image[y0:y1 + 1, x0:x1 + 1]
            binary = mask[y0:y1 + 1, x0:x1 + 1]
            radiance = _nearest_safe_radiance(crop, binary)
            assets.append((path, index, radiance, binary, mask_report))
    return assets


def _prepare_object(
    clean_radiance: np.ndarray,
    mask: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper returning radiance and the soft display matte."""
    radiance, matte, _ = _prepare_opaque_object(
        clean_radiance,
        mask,
        scale,
    )
    return radiance, matte


def _prepare_opaque_object(
    clean_radiance: np.ndarray,
    mask: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resize radiance while preserving binary ownership before antialiasing."""
    radiance = cv2.resize(
        clean_radiance,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )
    matte = cv2.resize(
        mask.astype(np.float32),
        (radiance.shape[1], radiance.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    matte = np.clip(cv2.GaussianBlur(matte, (0, 0), 0.5), 0.0, 1.0)
    hard_ownership = cv2.resize(
        (np.asarray(mask) > 0).astype(np.uint8),
        (radiance.shape[1], radiance.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ) > 0
    return radiance, matte, hard_ownership


def generate(count: int, split: str) -> None:
    supported_splits = {
        "dev",
        "holdout",
        "extension",
        "s12",
        "s16",
        "s19",
        "s23",
        "t24",
        "s25",
        "s26",
        "s27",
        "s28",
        "s29",
    }
    if split not in supported_splits:
        raise ValueError(
            f"split must be one of {sorted(supported_splits)}"
        )
    split_dir = os.path.join(OUT, split)
    os.makedirs(split_dir, exist_ok=True)
    is_transmissive = split == "t24"
    is_one_sided_opaque = split in ONE_SIDED_OPAQUE_SPLITS
    solid_opaque_sources = split in SOLID_OPAQUE_MASK_SPLITS
    assets = _source_assets(solid_opaque=solid_opaque_sources)
    if not assets:
        raise RuntimeError("no source objects with cached masks")
    photos = sorted(glob.glob(os.path.join(SRC, "*", "gt.png")))
    background_cache = {}
    seed = {
        "dev": 6001,
        "holdout": 9001,
        "extension": 12001,
        "s12": 15001,
        "s16": 18001,
        "s19": 21001,
        "s23": 24001,
        "t24": 27001,
        "s25": 30001,
        "s26": 33001,
        "s27": 36001,
        "s28": 39001,
        "s29": 42001,
    }[split]
    if is_transmissive:
        stratum_schedule = TRANSMISSIVE_SCHEDULE
        material_model = "scalar_transmissive_occluder"
        cohort_role = "transmissive_oracle_development"
    elif split == "s23" or is_one_sided_opaque:
        stratum_schedule = PRIMARY_OPAQUE_SCHEDULE
        material_model = (
            "one_sided_opaque_occluder"
            if is_one_sided_opaque
            else "opaque_occluder"
        )
        cohort_role = (
            "primary_one_sided_opaque_development"
            if split == "s25"
            else (
                "primary_one_sided_opaque_post_rule_validation"
                if split == "s29"
                else "primary_one_sided_opaque_validation"
                if split == "s28"
                else "primary_one_sided_opaque_post_freeze"
                if split in {"s26", "s27"}
                else "primary_opaque_post_audit"
            )
        )
    else:
        stratum_schedule = STRATA
        material_model = "opaque_occluder"
        cohort_role = "legacy_equal_optical_regime_cycle"
    rng = np.random.default_rng(seed)
    manifest = {
        "version": (
            4 if split in HARD_OWNERSHIP_OPAQUE_SPLITS else 3
        ),
        "split": split,
        "seed": seed,
        "renderer": (
            "one_sided_opaque_v2"
            if split in HARD_OWNERSHIP_OPAQUE_SPLITS
            else "one_sided_opaque_v1"
            if is_one_sided_opaque
            else "exact_disk_two_layer_extinction"
        ),
        "material_model": material_model,
        "cohort_role": cohort_role,
        "source_mask_contract": (
            "solid_opaque_no_ambiguous_holes_v1"
            if solid_opaque_sources
            else "legacy_semantic_mask"
        ),
        "source_mask_max_enclosed_hole_fraction": (
            SOLID_OPAQUE_MAX_ENCLOSED_HOLE_FRACTION
            if solid_opaque_sources
            else None
        ),
        "stratum_schedule": list(stratum_schedule),
        "coc_fraction": COC_FRACTION,
        "defocus_distance": DEFOCUS_DISTANCE,
        "scenes": [],
    }

    attempts = 0
    while len(manifest["scenes"]) < count and attempts < 4000:
        attempts += 1
        scene_index = len(manifest["scenes"])
        stratum = stratum_schedule[scene_index % len(stratum_schedule)]
        material_opacity = (
            TRANSMISSIVE_OPACITIES[
                scene_index % len(TRANSMISSIVE_OPACITIES)
            ]
            if is_transmissive
            else 1.0
        )
        (
            source_path,
            mask_index,
            source_radiance,
            source_mask,
            source_mask_report,
        ) = assets[int(rng.integers(len(assets)))]
        background_path = photos[int(rng.integers(len(photos)))]
        if os.path.dirname(background_path) == os.path.dirname(source_path):
            continue
        if background_path not in background_cache:
            original = cv2.imread(background_path)
            bh, bw = original.shape[:2]
            resize = LONG / max(bh, bw)
            background_cache[background_path] = cv2.resize(
                original,
                (int(round(bw * resize)), int(round(bh * resize))),
                interpolation=cv2.INTER_AREA,
            )
        background = background_cache[background_path]
        bh, bw = background.shape[:2]
        h, w = background.shape[:2]
        max_radius = COC_FRACTION * max(h, w)

        # Scale is part of the sampled scene, then the scene is admitted by an
        # explicit optical-core stratum rather than an object-area proxy.
        scale_range = {
            "solid": (0.22, 0.85),
            "mixed": (0.20, 0.65),
            "thin": (0.18, 0.48),
            "primary": (0.30, 0.95),
            "boundary": (0.20, 0.65),
            "all_veil": (0.18, 0.48),
        }[stratum]
        scale = float(rng.uniform(*scale_range))
        foreground_crop, alpha_crop, hard_ownership_crop = (
            _prepare_opaque_object(
            source_radiance,
            source_mask,
            scale,
            )
        )
        oh, ow = alpha_crop.shape
        if oh >= 0.82 * h or ow >= 0.82 * w:
            continue
        if oh >= h or ow >= w:
            continue

        margin = int(np.ceil(DEFOCUS_DISTANCE * max_radius)) + 3
        if h - oh <= 2 * margin or w - ow <= 2 * margin:
            continue
        py = int(rng.integers(margin, h - oh - margin))
        px = int(rng.integers(margin, w - ow - margin))
        alpha = np.zeros((h, w), np.float32)
        hard_ownership = np.zeros((h, w), bool)
        foreground = np.zeros((h, w, 3), np.float32)
        alpha[py:py + oh, px:px + ow] = alpha_crop
        hard_ownership[py:py + oh, px:px + ow] = hard_ownership_crop
        foreground[py:py + oh, px:px + ow] = foreground_crop
        area_fraction = float((alpha >= 0.5).mean())
        min_area_fraction = (
            0.007
            if stratum in {"thin", "all_veil"}
            else 0.025
        )
        if not min_area_fraction <= area_fraction <= 0.28:
            continue

        fast_coverage = np.clip(
            fast_disk_blur(alpha, DEFOCUS_DISTANCE * max_radius),
            0.0,
            1.0,
        )
        stats = coverage_stats(alpha, fast_coverage)
        if not _stratum_matches(stratum, stats):
            continue
        far_coverage = np.clip(
            exact_disk_blur(alpha, DEFOCUS_DISTANCE * max_radius),
            0.0,
            1.0,
        )
        stats = coverage_stats(alpha, far_coverage)
        if not _stratum_matches(stratum, stats):
            continue
        if split in HARD_OWNERSHIP_OPAQUE_SPLITS:
            rendered = render_one_sided_opaque_focal_pair(
                background,
                foreground,
                alpha,
                max_radius,
                hard_ownership=hard_ownership,
            )
        elif is_one_sided_opaque:
            rendered = _render_one_sided_opaque_focal_pair_v1(
                background,
                foreground,
                alpha,
                max_radius,
            )
        else:
            rendered = render_layered_focal_pair(
                background,
                foreground,
                alpha,
                max_radius,
                material_opacity,
            )
        clean_frames = rendered["frames"]
        coverages = rendered["geometry_coverage"]
        extinctions = rendered["extinction"]
        aperture_spreads = rendered["aperture_spread"]
        rendered_stats = coverage_stats(alpha, coverages[1])

        sid = f"{split}_{scene_index:03d}"
        scene_dir = os.path.join(split_dir, sid)
        os.makedirs(scene_dir, exist_ok=True)
        gt = rendered["gt"]
        frames = [
            add_noise(
                np.clip(frame, 0, 255).astype(np.uint8),
                3.0,
                seed + 10 * scene_index + frame_index,
            )
            for frame_index, frame in enumerate(clean_frames)
        ]
        cv2.imwrite(
            os.path.join(scene_dir, "gt.png"),
            np.clip(gt, 0, 255).astype(np.uint8),
        )
        cv2.imwrite(
            os.path.join(scene_dir, "alpha.png"),
            np.round(alpha * 255).astype(np.uint8),
        )
        if split in HARD_OWNERSHIP_OPAQUE_SPLITS:
            cv2.imwrite(
                os.path.join(scene_dir, "hard_ownership.png"),
                hard_ownership.astype(np.uint8) * 255,
            )
        cv2.imwrite(
            os.path.join(scene_dir, "foreground_layer.png"),
            np.clip(foreground, 0, 255).astype(np.uint8),
        )
        cv2.imwrite(
            os.path.join(scene_dir, "background_layer.png"),
            np.clip(background, 0, 255).astype(np.uint8),
        )
        cv2.imwrite(
            os.path.join(scene_dir, "opacity.png"),
            np.round(np.clip(extinctions[0], 0, 1) * 255).astype(np.uint8),
        )
        for frame_index, (
            frame,
            clean,
            coverage,
            extinction,
            aperture_spread,
        ) in enumerate(
            zip(
                frames,
                clean_frames,
                coverages,
                extinctions,
                aperture_spreads,
            )
        ):
            cv2.imwrite(os.path.join(scene_dir, f"frame_{frame_index}.png"), frame)
            cv2.imwrite(
                os.path.join(scene_dir, f"frame_{frame_index}_clean.png"),
                np.clip(clean, 0, 255).astype(np.uint8),
            )
            cv2.imwrite(
                os.path.join(scene_dir, f"coverage_{frame_index}.png"),
                np.round(np.clip(coverage, 0, 1) * 255).astype(np.uint8),
            )
            cv2.imwrite(
                os.path.join(scene_dir, f"extinction_{frame_index}.png"),
                np.round(np.clip(extinction, 0, 1) * 255).astype(np.uint8),
            )
            cv2.imwrite(
                os.path.join(
                    scene_dir,
                    f"aperture_spread_{frame_index}.png",
                ),
                np.round(
                    np.clip(aperture_spread, 0, 1) * 255
                ).astype(np.uint8),
            )
        cv2.imwrite(
            os.path.join(scene_dir, "coverage_classes.png"),
            coverage_classification(alpha, coverages[1]),
        )
        cv2.imwrite(
            os.path.join(scene_dir, "vis.png"),
            np.hstack(
                [
                    cv2.resize(image, (384, int(image.shape[0] * 384 / image.shape[1])))
                    for image in (
                        frames[0],
                        frames[1],
                        np.clip(gt, 0, 255).astype(np.uint8),
                        coverage_classification(alpha, coverages[1]),
                    )
                ]
            ),
        )
        row = {
            "id": sid,
            "stratum": stratum,
            "optical_regime": OPTICAL_REGIME_BY_STRATUM[stratum],
            "material_model": material_model,
            "material_opacity": float(material_opacity),
            "material_transmittance": float(1.0 - material_opacity),
            "formation_model": rendered["formation_model"],
            **(
                {
                    "hard_ownership_pixels": int(hard_ownership.sum()),
                    "hard_ownership_min_far_coverage": float(
                        coverages[1][hard_ownership].min(initial=1.0)
                    ),
                    "hard_ownership_nonopaque_pixels": int(
                        (coverages[1][hard_ownership] < 1.0).sum()
                    ),
                }
                if split in HARD_OWNERSHIP_OPAQUE_SPLITS
                else {}
            ),
            "source": os.path.basename(os.path.dirname(source_path)),
            "source_mask_index": int(mask_index),
            **source_mask_report,
            "background": os.path.basename(os.path.dirname(background_path)),
            "scale": scale,
            "placement_xy": [px, py],
            "max_radius": float(max_radius),
            "defocus_radius": float(DEFOCUS_DISTANCE * max_radius),
            "alpha_area_fraction": area_fraction,
            **rendered_stats,
            "aperture_core_fraction": stats["core_fraction"],
            "aperture_inner_fraction": stats["inner_veil_fraction"],
        }
        manifest["scenes"].append(row)
        if is_one_sided_opaque:
            formation_summary = (
                f"owned-core={rendered_stats['core_fraction']:.3f} "
                f"owned-inner={rendered_stats['inner_veil_fraction']:.3f} "
                f"aperture-core={stats['core_fraction']:.3f}"
            )
        else:
            formation_summary = (
                f"core={stats['core_fraction']:.3f} "
                f"inner-veil={stats['inner_veil_fraction']:.3f}"
            )
        print(
            f"{sid} {stratum:5s} opacity={material_opacity:.2f} "
            f"{formation_summary} area={area_fraction:.3f}",
            flush=True,
        )

    if len(manifest["scenes"]) != count:
        raise RuntimeError(
            f"generated only {len(manifest['scenes'])}/{count} scenes "
            f"after {attempts} attempts"
        )
    for manifest_path in (
        os.path.join(split_dir, "manifest.json"),
        os.path.join(HERE, f"objocc_v2_{split}_manifest.json"),
    ):
        with open(manifest_path, "w") as handle:
            json.dump(manifest, handle, indent=2)
    print(f"{count} physically audited {split} scenes -> {split_dir}")


def scenes(split: str):
    """Load generated V2 scenes with frame-specific optical coverage."""
    split_dir = os.path.join(OUT, split)
    with open(os.path.join(split_dir, "manifest.json")) as handle:
        manifest = json.load(handle)
    for row in manifest["scenes"]:
        scene_dir = os.path.join(split_dir, row["id"])
        material_model = row.get("material_model", "opaque_occluder")
        material_opacity = float(row.get("material_opacity", 1.0))
        opacity_path = os.path.join(scene_dir, "opacity.png")
        opacity = (
            cv2.imread(opacity_path, 0).astype(np.float32) / 255.0
            if os.path.exists(opacity_path)
            else (
                cv2.imread(
                    os.path.join(scene_dir, "alpha.png"),
                    0,
                ).astype(np.float32)
                / 255.0
            )
            * material_opacity
        )
        yield {
            "sid": row["id"],
            "stratum": row["stratum"],
            "optical_regime": row.get(
                "optical_regime",
                OPTICAL_REGIME_BY_STRATUM[row["stratum"]],
            ),
            "material_model": material_model,
            "material_opacity": material_opacity,
            "formation_model": row.get(
                "formation_model",
                manifest.get(
                    "renderer",
                    "exact_disk_two_layer_extinction",
                ),
            ),
            "dir": scene_dir,
            "gt": cv2.imread(os.path.join(scene_dir, "gt.png")),
            "alpha": (
                cv2.imread(os.path.join(scene_dir, "alpha.png"), 0).astype(
                    np.float32
                )
                / 255.0
            ),
            "hard_ownership": (
                cv2.imread(
                    os.path.join(scene_dir, "hard_ownership.png"),
                    0,
                )
                > 0
                if os.path.exists(
                    os.path.join(scene_dir, "hard_ownership.png")
                )
                else cv2.imread(
                    os.path.join(scene_dir, "alpha.png"),
                    0,
                )
                >= 128
            ),
            "coverage": [
                cv2.imread(
                    os.path.join(scene_dir, f"coverage_{index}.png"),
                    0,
                ).astype(np.float32)
                / 255.0
                for index in (0, 1)
            ],
            "opacity": opacity,
            "foreground_layer": (
                cv2.imread(
                    os.path.join(scene_dir, "foreground_layer.png")
                )
                if os.path.exists(
                    os.path.join(scene_dir, "foreground_layer.png")
                )
                else None
            ),
            "background_layer": (
                cv2.imread(
                    os.path.join(scene_dir, "background_layer.png")
                )
                if os.path.exists(
                    os.path.join(scene_dir, "background_layer.png")
                )
                else None
            ),
            "extinction": [
                (
                    cv2.imread(
                        os.path.join(
                            scene_dir,
                            f"extinction_{index}.png",
                        ),
                        0,
                    ).astype(np.float32)
                    / 255.0
                    if os.path.exists(
                        os.path.join(
                            scene_dir,
                            f"extinction_{index}.png",
                        )
                    )
                    else (
                        cv2.imread(
                            os.path.join(
                                scene_dir,
                                f"coverage_{index}.png",
                            ),
                            0,
                        ).astype(np.float32)
                        / 255.0
                    )
                    * material_opacity
                )
                for index in (0, 1)
            ],
            "aperture_spread": [
                (
                    cv2.imread(
                        os.path.join(
                            scene_dir,
                            f"aperture_spread_{index}.png",
                        ),
                        0,
                    ).astype(np.float32)
                    / 255.0
                    if os.path.exists(
                        os.path.join(
                            scene_dir,
                            f"aperture_spread_{index}.png",
                        )
                    )
                    else cv2.imread(
                        os.path.join(
                            scene_dir,
                            f"coverage_{index}.png",
                        ),
                        0,
                    ).astype(np.float32)
                    / 255.0
                )
                for index in (0, 1)
            ],
            "frames": [
                cv2.imread(os.path.join(scene_dir, f"frame_{index}.png"))
                for index in (0, 1)
            ],
            "max_r": float(row["max_radius"]),
            "factory": row,
        }


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: objocc_v2_gen.py COUNT "
            "{dev|holdout|extension|s12|s16|s19|s23|t24|s25|s26|s27|s28|s29}"
        )
    generate(int(sys.argv[1]), sys.argv[2])


if __name__ == "__main__":
    main()
