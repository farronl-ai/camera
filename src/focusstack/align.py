"""Stage 1 — registration (alignment).

Why this stage exists: when a lens refocuses, the image magnification changes
slightly (objects appear to grow/shrink) — this is called *focus breathing*. The
camera or subject may also shift a little between frames. If we fuse unaligned
frames, sharp detail from one frame lands on the wrong pixels of another and we
get ghosting/doubling. So first we warp every frame onto a common coordinate
frame (that of a chosen reference frame).

We use OpenCV's ECC (Enhanced Correlation Coefficient) algorithm, which directly
estimates the geometric warp that best aligns two images by maximizing their
correlation — no feature detection needed, and it is robust to the brightness
differences a focus change can introduce.
"""

from __future__ import annotations

import warnings

import cv2
import numpy as np

from .io import to_gray_float

# Motion models, from most constrained to most general. `affine` (translation +
# rotation + scale + shear) is a good default because it captures focus breathing
# (scale) plus small camera motion without the extra freedom of a full homography.
_MOTION_MODES = {
    "translation": cv2.MOTION_TRANSLATION,
    "euclidean": cv2.MOTION_EUCLIDEAN,  # translation + rotation only
    "affine": cv2.MOTION_AFFINE,
    "homography": cv2.MOTION_HOMOGRAPHY,  # full perspective (8 DOF)
}


def align_stack(
    images: list[np.ndarray],
    ref_index: int | None = None,
    motion: str = "affine",
    max_iterations: int = 500,
    eps: float = 1e-6,
) -> list[np.ndarray]:
    """Align every frame to a reference frame.

    Args:
        images: list of BGR uint8 frames, all the same size.
        ref_index: which frame to treat as the fixed reference. Defaults to the
            middle frame, which tends to be geometrically closest to all others.
        motion: one of `_MOTION_MODES`.
        max_iterations, eps: ECC convergence criteria.

    Returns:
        A new list of aligned BGR uint8 frames (the reference is unchanged).
        Frames for which ECC fails to converge are returned unaligned, with a
        warning, rather than aborting the whole run.
    """
    if motion not in _MOTION_MODES:
        raise ValueError(f"Unknown motion model {motion!r}; choose from {list(_MOTION_MODES)}")
    warp_mode = _MOTION_MODES[motion]

    n = len(images)
    if ref_index is None:
        ref_index = n // 2

    # ECC wants single-channel float in [0, 1].
    def norm_gray(img: np.ndarray) -> np.ndarray:
        return to_gray_float(img) / 255.0

    ref = norm_gray(images[ref_index])
    h, w = ref.shape
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, eps)

    aligned: list[np.ndarray] = []
    for i, img in enumerate(images):
        if i == ref_index:
            aligned.append(img)
            continue

        moving = norm_gray(img)
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        try:
            # findTransformECC estimates the warp mapping `moving` onto `ref`.
            _, warp_matrix = cv2.findTransformECC(
                ref, moving, warp_matrix, warp_mode, criteria, None, 5
            )
            # WARP_INVERSE_MAP applies that warp to resample `img` into ref's frame.
            common = dict(
                dsize=(w, h),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT,
            )
            if warp_mode == cv2.MOTION_HOMOGRAPHY:
                warped = cv2.warpPerspective(img, warp_matrix, **common)
            else:
                warped = cv2.warpAffine(img, warp_matrix, **common)
            aligned.append(warped)
        except cv2.error as e:
            warnings.warn(f"ECC alignment failed for frame {i}; using it unaligned. ({e})")
            aligned.append(img)

    return aligned
