import sys
from pathlib import Path

import cv2
import numpy as np


RESEARCH = Path(__file__).resolve().parents[1] / "research"
sys.path.insert(0, str(RESEARCH))

from objocc_v2_gen import (  # noqa: E402
    coverage_stats,
    exact_disk_blur,
    render_focal_pair,
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
    assert stats["inner_veil_pixels"] > 0


def test_far_focus_mix_is_confined_to_partial_coverage():
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

    far = frames[1]
    mixed = (
        np.any(np.abs(far - background) > 1e-5, axis=2)
        & np.any(np.abs(far - foreground) > 1e-5, axis=2)
    )
    partial = (coverage[1] > 0.0) & (coverage[1] < 1.0)
    assert np.all(~mixed | partial)


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
