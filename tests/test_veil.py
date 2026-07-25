import cv2
import numpy as np

from focusstack.veil import (
    build_veil_model,
    estimate_noise_sigma,
    fit_chromatic_spread,
    forward_residual,
    unsupported_texture_score,
)


def _veil_stack(size=96):
    yy, xx = np.mgrid[:size, :size]
    background = np.stack(
        [
            70.0 + 0.25 * xx,
            90.0 + 0.15 * yy,
            110.0 + 0.1 * (xx + yy),
        ],
        axis=2,
    )
    alpha = (((xx - size / 2) ** 2 + (yy - size / 2) ** 2) < (size / 5) ** 2).astype(
        np.float32
    )
    owner = background.copy()
    owner[alpha > 0.5] = (35.0, 120.0, 210.0)
    far = cv2.GaussianBlur(owner, (0, 0), 5.0)
    return [np.uint8(np.clip(owner, 0, 255)), np.uint8(np.clip(far, 0, 255))], alpha


def test_noise_estimator_recovers_synthetic_sigma():
    rng = np.random.default_rng(4)
    base = np.full((384, 384, 3), 128.0, np.float32)
    images = [
        np.clip(base + rng.normal(0.0, 3.0, base.shape), 0, 255).astype(np.uint8)
        for _ in range(3)
    ]
    sigma = estimate_noise_sigma(images)
    assert 2.3 < sigma < 3.7


def test_veil_model_is_channel_specific_and_auditable():
    images, alpha = _veil_stack()
    model = build_veil_model(images, alpha, (4.5, 5.0, 5.5), owner=0, far_idx=1)

    assert model["D"].shape == images[0].shape
    assert model["ab"].shape == images[0].shape
    assert not np.allclose(model["ab"][..., 0], model["ab"][..., 2])
    assert model["mask"].sum() > 32
    assert np.isfinite(forward_residual(images[1], images, model))


def test_unsupported_texture_score_detects_invented_quiet_detail():
    shape = (96, 96, 3)
    images = [np.full(shape, 128, np.uint8), np.full(shape, 128, np.uint8)]
    base = images[0].copy()
    corrected = base.astype(np.float32)
    yy, xx = np.mgrid[:96, :96]
    checker = ((xx + yy) % 2) * 16.0 - 8.0
    corrected[..., 1] += checker
    model = {
        "mask": np.ones(shape[:2], np.float32),
        "sigma": 3.0,
    }

    identity = unsupported_texture_score(base, base, images, model)
    invented = unsupported_texture_score(base, corrected, images, model)
    assert identity == 0.0
    assert invented > 1.0


def test_chromatic_spread_fit_is_bounded_and_finite():
    images, alpha = _veil_stack()
    radii, evidence = fit_chromatic_spread(
        images, alpha, base_radius=5.0, owner=0, far_idx=1
    )

    assert radii.shape == (3,)
    assert np.all(radii > 0)
    assert evidence["spread"] in (0.0, 0.04, 0.08, 0.12)
    assert np.isfinite(evidence["fit_error"])
    assert evidence["fit_margin"] >= 0.0
