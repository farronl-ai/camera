import cv2
import numpy as np

from focusstack.focus import focus_measure


def test_sharp_scores_higher_than_blurred():
    """A sharp image should have more high-frequency energy than its blurred copy."""
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 256, (128, 128)).astype(np.float32)
    blurred = cv2.GaussianBlur(sharp, (11, 11), 4)

    assert focus_measure(sharp).mean() > focus_measure(blurred).mean()


def test_gradient_measure_also_ranks_sharpness():
    rng = np.random.default_rng(1)
    sharp = rng.integers(0, 256, (96, 96)).astype(np.float32)
    blurred = cv2.GaussianBlur(sharp, (9, 9), 3)

    fs = focus_measure(sharp, method="gradient")
    fb = focus_measure(blurred, method="gradient")
    assert fs.mean() > fb.mean()


def test_all_operators_rank_sharpness():
    rng = np.random.default_rng(3)
    sharp = rng.integers(0, 256, (96, 96)).astype(np.float32)
    blurred = cv2.GaussianBlur(sharp, (9, 9), 3)
    for method in ("laplacian", "gradient", "tenengrad", "mod_laplacian"):
        fs = focus_measure(sharp, method=method).mean()
        fb = focus_measure(blurred, method=method).mean()
        assert fs > fb, f"{method} failed to rank sharp > blurred"
