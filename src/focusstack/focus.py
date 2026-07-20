"""Stage 2 — focus measure (sharpness detection).

Why this stage exists: to combine frames we need a per-pixel answer to "how
in-focus is this spot in this frame?" A region that is in focus contains sharp
edges and fine texture — i.e. lots of *high spatial frequency* energy. A blurred
region is smooth (low frequency). So a focus measure is essentially a
high-frequency energy detector.

Two classic operators:
  - Laplacian: the second spatial derivative. It is ~zero on smooth gradients and
    spikes at edges/texture, so its magnitude is a direct sharpness signal.
  - Gradient (Sobel): the first derivative magnitude — strong at any intensity
    change. Slightly less selective for fine texture than the Laplacian.

We then blur the raw response, turning a noisy per-pixel value into a smoother
*regional* "sharpness energy" that is far more robust when we compare frames.
"""

from __future__ import annotations

import cv2
import numpy as np

from .io import to_gray_float


def focus_measure(
    gray: np.ndarray, method: str = "laplacian", smooth_ksize: int = 9
) -> np.ndarray:
    """Compute a per-pixel sharpness map (higher = more in focus).

    Args:
        gray: single-channel image (any numeric dtype; converted to float32).
        method: "laplacian" or "gradient".
        smooth_ksize: box-filter window used to pool the raw response into a
            regional energy. Set to 0/1 to disable pooling.

    Returns:
        float32 array, same HxW as the input.
    """
    if gray.dtype not in (np.float32, np.float64):
        gray = gray.astype(np.float32)

    if method == "laplacian":
        response = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        energy = np.abs(response)
    elif method == "gradient":
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        energy = cv2.magnitude(gx, gy)
    elif method == "tenengrad":
        # Squared gradient energy (Tenengrad): more selective for strong edges.
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        energy = gx * gx + gy * gy
    elif method == "mod_laplacian":
        # Modified Laplacian (Nayar): |I * [-1,2,-1]_x| + |I * [-1,2,-1]_y|.
        # Summing the abs of the two 1-D second derivatives (instead of the
        # signed 2-D Laplacian) stops opposite-sign x/y curvature from cancelling.
        kx = np.array([[-1.0, 2.0, -1.0]], dtype=np.float32)
        lx = cv2.filter2D(gray, cv2.CV_32F, kx)
        ly = cv2.filter2D(gray, cv2.CV_32F, kx.T)
        energy = np.abs(lx) + np.abs(ly)
    else:
        raise ValueError(
            f"Unknown focus measure {method!r}; use 'laplacian', 'gradient', "
            "'tenengrad', or 'mod_laplacian'."
        )

    if smooth_ksize and smooth_ksize > 1:
        # Pool into a regional score: a pixel is "in focus" if its neighborhood
        # is sharp, which suppresses single-pixel noise spikes.
        energy = cv2.boxFilter(energy, cv2.CV_32F, (smooth_ksize, smooth_ksize))

    return energy.astype(np.float32)


def focus_measures(
    images: list[np.ndarray], method: str = "laplacian", smooth_ksize: int = 9
) -> list[np.ndarray]:
    """Convenience: focus map for every (BGR) frame in a stack."""
    return [focus_measure(to_gray_float(img), method, smooth_ksize) for img in images]
