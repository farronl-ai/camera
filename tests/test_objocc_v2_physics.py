import sys
from pathlib import Path

import cv2
import numpy as np


RESEARCH = Path(__file__).resolve().parents[1] / "research"
sys.path.insert(0, str(RESEARCH))

from objocc_v2_gen import (  # noqa: E402
    _solid_opaque_source_mask,
    coverage_stats,
    exact_disk_blur,
    render_focal_pair,
    render_layered_focal_pair,
    render_one_sided_opaque_focal_pair,
)


def test_solid_opaque_source_mask_fills_only_small_enclosed_holes():
    mask = np.zeros((80, 90), np.uint8)
    mask[10:70, 15:75] = 1
    mask[31:33, 40:42] = 0

    repaired, report = _solid_opaque_source_mask(mask)

    assert repaired is not None
    assert np.all(repaired[31:33, 40:42])
    assert report["source_mask_small_holes_filled"]
    assert not report["source_mask_rejected_ambiguous_holes"]


def test_solid_opaque_source_mask_rejects_large_internal_aperture():
    mask = np.zeros((80, 90), np.uint8)
    mask[10:70, 15:75] = 1
    mask[25:50, 35:55] = 0

    repaired, report = _solid_opaque_source_mask(mask)

    assert repaired is None
    assert report["source_mask_rejected_ambiguous_holes"]
    assert (
        report["source_mask_enclosed_hole_fraction"]
        > 0.005
    )


def test_complete_coverage_core_never_reveals_background():
    background = np.zeros((81, 81, 3), np.float32)
    background[..., 0] = 220
    foreground = np.zeros_like(background)
    foreground[..., 2] = 200
    alpha = np.zeros((81, 81), np.float32)
    alpha[15:66, 15:66] = 1.0

    frames, coverage = render_focal_pair(
        background,
        foreground,
        alpha,
        max_radius=10.0,
    )

    y, x = 40, 40
    assert coverage[1][y, x] == 1.0
    np.testing.assert_allclose(frames[1][y, x], foreground[y, x], atol=1e-5)
    stats = coverage_stats(alpha, coverage[1])
    assert stats["core_pixels"] > 0
    assert stats["inner_veil_pixels"] == 0
    assert stats["outer_veil_pixels"] > 0


def test_far_focus_owned_support_is_foreground_only_with_outward_veil():
    background = np.full((65, 65, 3), (20, 40, 80), np.float32)
    foreground = np.full((65, 65, 3), (180, 120, 30), np.float32)
    alpha = np.zeros((65, 65), np.float32)
    alpha[16:49, 16:49] = 1.0
    frames, coverage = render_focal_pair(
        background,
        foreground,
        alpha,
        max_radius=8.0,
    )

    owned = alpha == 1.0
    np.testing.assert_allclose(
        frames[1][owned],
        foreground[owned],
        atol=2e-4,
    )
    assert np.any((~owned) & (coverage[1] > 0.0))


def test_opaque_far_focus_never_admits_rear_detail_inside_foreground():
    size = 81
    yy, xx = np.mgrid[:size, :size]
    checker = ((xx + yy) % 2).astype(np.float32) * 220.0
    background_a = np.repeat(checker[..., None], 3, axis=2)
    background_b = 255.0 - background_a
    foreground = np.full_like(background_a, (35, 90, 180))
    alpha = np.zeros((size, size), np.float32)
    # Deliberately thinner than the defocus diameter: the old symmetric
    # aperture model revealed rear checkerboard throughout this support.
    alpha[25:56, 34:47] = 1.0

    rendered_a = render_one_sided_opaque_focal_pair(
        background_a,
        foreground,
        alpha,
        max_radius=18.0,
    )
    rendered_b = render_one_sided_opaque_focal_pair(
        background_b,
        foreground,
        alpha,
        max_radius=18.0,
    )

    owned = alpha == 1.0
    np.testing.assert_array_equal(
        rendered_a["frames"][1][owned],
        rendered_b["frames"][1][owned],
    )
    np.testing.assert_array_equal(
        rendered_a["geometry_coverage"][1][owned],
        np.ones(int(owned.sum()), np.float32),
    )
    assert np.any(
        rendered_a["geometry_coverage"][1][~owned] > 0.0
    ), "foreground defocus must still spread outward"


def test_one_sided_coverage_is_never_smaller_than_sharp_support():
    rng = np.random.default_rng(19)
    background = rng.uniform(0, 255, (73, 89, 3)).astype(np.float32)
    foreground = rng.uniform(0, 255, (73, 89, 3)).astype(np.float32)
    alpha = np.zeros((73, 89), np.float32)
    alpha[17:61, 29:55] = 1.0
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.5)

    rendered = render_one_sided_opaque_focal_pair(
        background,
        foreground,
        alpha,
        max_radius=13.0,
    )

    assert np.all(rendered["geometry_coverage"][1] >= alpha)
    sharp = alpha >= 0.95
    assert np.all(
        1.0 - rendered["geometry_coverage"][1][sharp]
        <= 1.0 - alpha[sharp]
    )


def test_disk_renderer_matches_brute_aperture_average_away_from_border():
    rng = np.random.default_rng(5)
    image = rng.normal(size=(31, 31)).astype(np.float32)
    radius = 3.0
    filtered = exact_disk_blur(image, radius)

    r = int(radius)
    offsets = [
        (dy, dx)
        for dy in range(-r, r + 1)
        for dx in range(-r, r + 1)
        if dx * dx + dy * dy <= radius * radius
    ]
    brute = np.zeros((25, 25), np.float32)
    for dy, dx in offsets:
        brute += image[
            3 + dy:28 + dy,
            3 + dx:28 + dx,
        ]
    brute /= len(offsets)
    np.testing.assert_allclose(filtered[3:28, 3:28], brute, atol=2e-6)


def test_primary_opaque_pair_wrapper_matches_dict_renderer():
    rng = np.random.default_rng(11)
    background = rng.uniform(0, 255, (71, 83, 3)).astype(np.float32)
    foreground = rng.uniform(0, 255, (71, 83, 3)).astype(np.float32)
    alpha = np.zeros((71, 83), np.float32)
    alpha[13:59, 17:68] = 1.0
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.5)

    legacy_frames, legacy_coverage = render_focal_pair(
        background,
        foreground,
        alpha,
        max_radius=9.0,
    )
    layered = render_one_sided_opaque_focal_pair(
        background,
        foreground,
        alpha,
        max_radius=9.0,
    )

    for legacy, current in zip(legacy_frames, layered["frames"]):
        np.testing.assert_array_equal(legacy, current)
    for legacy, current in zip(
        legacy_coverage,
        layered["geometry_coverage"],
    ):
        np.testing.assert_array_equal(legacy, current)
    np.testing.assert_array_equal(
        layered["extinction"][0],
        layered["geometry_coverage"][0],
    )
    np.testing.assert_array_equal(
        layered["extinction"][1],
        layered["geometry_coverage"][1],
    )


def test_transmission_separates_geometry_from_extinction():
    background = np.full((81, 81, 3), (20, 60, 100), np.float32)
    foreground = np.full((81, 81, 3), (180, 120, 40), np.float32)
    alpha = np.zeros((81, 81), np.float32)
    alpha[10:71, 10:71] = 1.0
    opacity = 0.4

    rendered = render_layered_focal_pair(
        background,
        foreground,
        alpha,
        max_radius=10.0,
        material_opacity=opacity,
    )

    y, x = 40, 40
    assert rendered["geometry_coverage"][1][y, x] == 1.0
    np.testing.assert_allclose(
        rendered["extinction"][1][y, x],
        opacity,
        atol=1e-6,
    )
    expected = opacity * foreground[y, x] + (1.0 - opacity) * background[y, x]
    np.testing.assert_allclose(rendered["frames"][1][y, x], expected, atol=1e-5)
    np.testing.assert_allclose(rendered["gt"][y, x], expected, atol=1e-5)


def test_scalar_transmission_preserves_aperture_coverage_geometry():
    rng = np.random.default_rng(17)
    background = rng.uniform(0, 255, (61, 75, 3)).astype(np.float32)
    foreground = rng.uniform(0, 255, (61, 75, 3)).astype(np.float32)
    alpha = np.zeros((61, 75), np.float32)
    alpha[18:43, 14:61] = 1.0

    opaque = render_layered_focal_pair(
        background,
        foreground,
        alpha,
        max_radius=8.0,
        material_opacity=1.0,
    )
    transmissive = render_layered_focal_pair(
        background,
        foreground,
        alpha,
        max_radius=8.0,
        material_opacity=0.55,
    )

    for opaque_coverage, transmitted_coverage in zip(
        opaque["geometry_coverage"],
        transmissive["geometry_coverage"],
    ):
        np.testing.assert_array_equal(
            opaque_coverage,
            transmitted_coverage,
        )
    for coverage, extinction in zip(
        transmissive["geometry_coverage"],
        transmissive["extinction"],
    ):
        np.testing.assert_allclose(extinction, 0.55 * coverage, atol=2e-6)
