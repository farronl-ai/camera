import cv2
import numpy as np

from focusstack.focus import focus_measure, focus_measures
from focusstack.fusion import fuse_max, fuse_pyramid


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
