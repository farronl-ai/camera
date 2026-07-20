import cv2
import numpy as np

from focusstack.focus import focus_measure, focus_measures
from focusstack.fusion import (depth_from_focus, fuse_blend, fuse_decision, fuse_max,
                               fuse_perband, fuse_pyramid, guided_filter)


def _two_region_stack():
    """Two frames: A sharp on the left half, B sharp on the right half."""
    rng = np.random.default_rng(1)
    base = rng.integers(0, 256, (128, 128, 3)).astype(np.uint8)
    blur = lambda im: cv2.GaussianBlur(im, (15, 15), 6)

    a = base.copy()
    a[:, 64:] = blur(base)[:, 64:]  # A blurred on the right
    b = base.copy()
    b[:, :64] = blur(base)[:, :64]  # B blurred on the left
    return base, a, b


def _sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return focus_measure(gray)


def test_fuse_max_recovers_sharp_regions():
    _, a, b = _two_region_stack()
    fmaps = focus_measures([a, b])
    _, index_map = fuse_max([a, b], fmaps)

    # Left half should come mostly from frame 0 (A), right half from frame 1 (B).
    assert (index_map[:, :64] == 0).mean() > 0.8
    assert (index_map[:, 64:] == 1).mean() > 0.8


def test_fuse_pyramid_sharper_than_either_input():
    _, a, b = _two_region_stack()
    fused = fuse_pyramid([a, b])

    # Each input is blurred in one half, so the fused result should be sharper
    # overall than either single frame.
    assert fused.shape == a.shape
    assert _sharpness(fused).mean() > _sharpness(a).mean()
    assert _sharpness(fused).mean() > _sharpness(b).mean()


def test_fuse_decision_sharper_and_selects_correctly():
    _, a, b = _two_region_stack()
    fused, weights = fuse_decision([a, b], return_weights=True)

    assert fused.shape == a.shape
    # Sharper overall than either single (each is blurred in one half).
    assert _sharpness(fused).mean() > _sharpness(a).mean()
    assert _sharpness(fused).mean() > _sharpness(b).mean()

    # Weights are a partition of unity, and each half favors its sharp frame.
    assert np.allclose(weights.sum(axis=0), 1.0, atol=1e-4)
    assert weights[0][:, :64].mean() > 0.5   # frame A (sharp left) wins the left
    assert weights[1][:, 64:].mean() > 0.5   # frame B (sharp right) wins the right


def test_fuse_blend_sharper_and_partition():
    _, a, b = _two_region_stack()
    fused, weights = fuse_blend([a, b], return_weights=True)

    assert fused.shape == a.shape
    # Sharper overall than either single input.
    assert _sharpness(fused).mean() > _sharpness(a).mean()
    assert _sharpness(fused).mean() > _sharpness(b).mean()

    # Weights partition unity; each half favors its sharp frame.
    assert np.allclose(weights.sum(axis=0), 1.0, atol=1e-4)
    assert weights[0][:, :64].mean() > 0.5
    assert weights[1][:, 64:].mean() > 0.5


def test_depth_from_focus_orders_regions():
    # Frame 0 is sharp on the left half, frame 1 on the right -> depth should be
    # near 0 on the left and near 1 on the right (winner index scaled to [0,1]).
    _, a, b = _two_region_stack()
    d = depth_from_focus([a, b])
    assert d.shape == a.shape[:2]
    assert d.min() >= 0.0 and d.max() <= 1.0
    assert d[:, :64].mean() < 0.35
    assert d[:, 64:].mean() > 0.65


def test_perband_fuses_sharper_than_inputs():
    _, a, b = _two_region_stack()
    fused = fuse_perband([a, b], harden=0.5)
    assert fused.shape == a.shape
    assert _sharpness(fused).mean() > _sharpness(a).mean()
    assert _sharpness(fused).mean() > _sharpness(b).mean()


def test_harden_runs_and_preserves_sharpness():
    # Defocus-spread rejection (harden>0) must still fuse sharper than inputs,
    # and harden=0 must be identical to the default path (no regression).
    _, a, b = _two_region_stack()
    default = fuse_blend([a, b])
    off = fuse_blend([a, b], harden=0.0)
    assert np.array_equal(default, off)  # off == default path
    hardened = fuse_blend([a, b], harden=0.8)
    assert hardened.shape == a.shape
    assert _sharpness(hardened).mean() > _sharpness(a).mean()
    assert _sharpness(hardened).mean() > _sharpness(b).mean()


def test_weight_scale_is_identity_at_1_and_safe_below():
    # weight_scale=1.0 must be byte-identical to the default path (no regression);
    # a downscaled run must still fuse sharper than either input.
    _, a, b = _two_region_stack()
    assert np.array_equal(fuse_blend([a, b]), fuse_blend([a, b], weight_scale=1.0))
    fast = fuse_blend([a, b], weight_scale=0.5)
    assert fast.shape == a.shape
    assert _sharpness(fast).mean() > _sharpness(a).mean()
    assert _sharpness(fast).mean() > _sharpness(b).mean()


def test_content_aware_focus_fuses_without_regression():
    # content_aware routes operators per pixel; must still fuse a two-region
    # stack sharper than either input (and not error on the cross-frame path).
    _, a, b = _two_region_stack()
    fused = fuse_blend([a, b], focus_method="content_aware")
    assert fused.shape == a.shape
    assert _sharpness(fused).mean() > _sharpness(a).mean()
    assert _sharpness(fused).mean() > _sharpness(b).mean()


def test_fuse_blend_at_least_as_sharp_as_decision():
    # On this fixture the multi-band blend should not lose global sharpness
    # relative to the single-scale decision blend.
    _, a, b = _two_region_stack()
    blend_s = _sharpness(fuse_blend([a, b])).mean()
    decision_s = _sharpness(fuse_decision([a, b])).mean()
    assert blend_s >= 0.98 * decision_s


def test_guided_filter_preserves_flat_and_edges():
    # On a flat guide, the guided filter just averages the source (pure smoothing).
    flat = np.full((64, 64), 0.5, dtype=np.float32)
    noisy = flat + np.random.default_rng(0).normal(0, 0.05, flat.shape).astype(np.float32)
    smoothed = guided_filter(flat, noisy, radius=6, eps=1e-3)
    assert smoothed.std() < noisy.std()

    # With a step-edge guide, output follows the guide's edge instead of blurring it.
    guide = np.zeros((64, 64), dtype=np.float32)
    guide[:, 32:] = 1.0
    out = guided_filter(guide, guide.copy(), radius=6, eps=1e-6)
    step = abs(float(out[:, 40].mean()) - float(out[:, 24].mean()))
    assert step > 0.8  # the ~1.0 contrast across the edge is retained
