"""Occlusion-edge ordering: which side of a boundary is in front (F83).

An occlusion boundary is the near object's own silhouette, so its sharpness
follows the FOREGROUND through a focal sweep — crisp in the frame that focuses
the occluder, blurred in the frame that focuses what lies behind. That makes the
focus index read ON a contour a statement about the occluder's depth, which
focus magnitude alone cannot provide (Marshall et al., JOSA A 1996).

This module is a validated instrument, not part of the runtime. It works: it
localizes contours to ~5 px where the depth map alone is ~32 px off, and its
ordering vote is correct on the analytic factory. What it does NOT support is
the conclusion it was built for — see F83. Discarding only the background side
of each boundary is worse than discarding both, because a defocused silhouette
blurs its own matte and the occluder's boundary pixels are themselves
foreground/background mixtures.

Kept because the contour localizer and the ordering bit are reusable, and
because the negative needs to stay reproducible.

    .venv/bin/python research/occlusion_order.py
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from focusstack.align import align_stack  # noqa: E402
from focusstack.focus import content_aware_energies  # noqa: E402
from focusstack.fusion import fuse_perband  # noqa: E402
from focusstack.io import to_gray_float  # noqa: E402
import metrics  # noqa: E402
from parallax_gen import build_stack  # noqa: E402

# Ordering the two sides of a boundary is one global bit for the whole sweep,
# so it is decided by vote and refused outright when the vote is close.
_MIN_ORDERING_VOTES = 200
_MIN_ORDERING_MARGIN = 0.05


def _occlusion_contours(
    grays: list[np.ndarray],
    raw_winner: np.ndarray,
    offset: int = 10,
    jump: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find the contours where one surface passes in front of another.

    A depth map alone cannot place these. Its transitions are soft and land tens
    of pixels from the real silhouette, while the evidence we need lives within
    about four pixels of it. An intensity edge, meanwhile, is sharply placed but
    says nothing about depth — a printed label is covered in edges that are all
    on one surface.

    So the two are combined: take intensity edges, and keep only those whose two
    sides sit at genuinely different focal depths. The sides are sampled well
    away from the edge along its own normal, because near the contour every
    estimate is contaminated by both surfaces at once.

    Returns the contour mask and, for each pixel, the focal indices sampled on
    either side plus the locally pooled index.
    """
    stack = np.stack(grays, axis=0)
    h, w = raw_winner.shape
    yy, xx = np.indices((h, w))
    sharp = stack[raw_winner, yy, xx].astype(np.uint8)
    smoothed = cv2.GaussianBlur(sharp, (5, 5), 0)

    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy) + 1e-6
    normal_x, normal_y = gx / magnitude, gy / magnitude

    regional = cv2.medianBlur(raw_winner.astype(np.uint8), 31).astype(np.float32)
    local = cv2.medianBlur(raw_winner.astype(np.uint8), 5).astype(np.float32)

    def sample(sign: int) -> np.ndarray:
        sx = np.clip(xx + sign * normal_x * offset, 0, w - 1).astype(np.float32)
        sy = np.clip(yy + sign * normal_y * offset, 0, h - 1).astype(np.float32)
        return cv2.remap(regional, sx, sy, cv2.INTER_NEAREST)

    side_a, side_b = sample(+1), sample(-1)
    edges = cv2.Canny(smoothed, 60, 180) > 0
    contour = edges & (np.abs(side_a - side_b) >= jump)
    return contour, side_a, side_b, local


def _near_is_low_index(
    contour: np.ndarray,
    side_a: np.ndarray,
    side_b: np.ndarray,
    local: np.ndarray,
) -> bool | None:
    """Decide whether a LOWER focal index means nearer, by asking the contours.

    An occlusion boundary is the near object's own silhouette, so its sharpness
    follows the foreground: it is crisp in the frame that focuses the occluder
    and blurred in the frame that focuses whatever lies behind. Reading the focus
    index on the contour therefore names the foreground's depth, and the side
    that matches is the one in front. (Marshall et al., JOSA A 1996.)

    Per contour this is only about three-quarters reliable, which is far too
    noisy to decide ownership pixel by pixel. But it does not have to: the sweep
    runs monotonically through focus, so the entire stack shares ONE bit — near
    is either the low index or the high one — and a noisy cue voting over
    thousands of contour pixels settles a single bit decisively. That bit then
    orders every boundary exactly, which the cue alone could never do.
    """
    if not contour.any():
        return None
    a = side_a[contour]
    b = side_b[contour]
    reading = local[contour]
    # Which side does the contour's own sharpness belong to?
    front_is_a = np.abs(a - reading) < np.abs(b - reading)
    front_index = np.where(front_is_a, a, b)
    back_index = np.where(front_is_a, b, a)

    decided = front_index != back_index
    if decided.sum() < _MIN_ORDERING_VOTES:
        return None
    share = float((front_index[decided] < back_index[decided]).mean())
    if abs(share - 0.5) < _MIN_ORDERING_MARGIN:
        # The votes are a coin flip; refuse to claim an ordering.
        return None
    return share > 0.5


def _front_side_mask(
    grays: list[np.ndarray],
    raw_winner: np.ndarray,
    frames: int,
    tolerance: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Mark the pixels that belong to the occluding side of a depth boundary.

    This is what makes refusal one-sided. The occluder is opaque and present in
    every frame — merely displaced — so it loses nothing to parallax and needs
    no refusal. Only the surface behind it has scene swung into and out of view.
    Discarding both sides, as a test without ordering must, throws away good
    data along every silhouette in the picture.
    """
    contour, side_a, side_b, local = _occlusion_contours(grays, raw_winner)
    near_is_low = _near_is_low_index(contour, side_a, side_b, local)
    report = {
        "contour_pixels": int(contour.sum()),
        "near_is_low_index": near_is_low,
    }
    if near_is_low is None:
        # No trustworthy ordering: treat every side as front, so the caller's
        # ribbon stays two-sided rather than being cut on a guess.
        return np.ones(raw_winner.shape, dtype=bool), report

    regional = cv2.medianBlur(raw_winner.astype(np.uint8), 31).astype(np.float32)
    nearness = -regional if near_is_low else regional
    front_nearness = np.maximum(
        -side_a if near_is_low else side_a,
        -side_b if near_is_low else side_b,
    )

    # Each pixel is judged against the nearest contour's foreground depth.
    interior = (~contour).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(
        interior, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
    )
    lookup = np.zeros(int(labels.max()) + 1, dtype=np.float32)
    lookup[labels[contour]] = front_nearness[contour]
    return (nearness >= lookup[labels] - tolerance), report



def main() -> None:
    frames, truth, near = build_stack()
    coarse = align_stack(frames, depth_bins=0, crop_valid=False)
    grays = [to_gray_float(image) for image in coarse]
    winner = np.argmax(np.stack(content_aware_energies(grays), axis=0), axis=0)

    contour, side_a, side_b, local = _occlusion_contours(grays, winner)
    true_contour = cv2.Canny(near.astype(np.uint8) * 255, 50, 150) > 0
    distance = cv2.distanceTransform((~true_contour).astype(np.uint8), cv2.DIST_L2, 3)
    print(f"contour localization: {int(contour.sum())} px, "
          f"median {np.median(distance[contour]):.1f} px from the true silhouette "
          f"({100 * float((distance[contour] < 8).mean()):.0f}% within 8 px)")

    ordering = _near_is_low_index(contour, side_a, side_b, local)
    print(f"ordering vote: near_is_low_index={ordering} (truth: True)")

    front, _ = _front_side_mask(grays, winner, len(coarse))
    print(f"front mask vs true near plane: {100 * float((front == near).mean()):.1f}% agreement")

    # The negative: ordering is right, but one-sided refusal is not.
    aligned, report = align_stack(frames, depth_bins=4, return_report=True)
    x0, y0, x1, y1 = report["crop"]
    target = truth[y0:y1, x0:x1]
    ribbon = [~m for m in report["usable"]]
    front_c = front[y0:y1, x0:x1]

    variants = {
        "no refusal": [np.ones_like(m) for m in ribbon],
        "background side only": [~(r & ~front_c) for r in ribbon],
        "foreground side only": [~(r & front_c) for r in ribbon],
        "both sides": [~r for r in ribbon],
    }
    for label, usable in variants.items():
        print(f"  {label:24s} GT-SSIM "
              f"{metrics.ref_ssim(fuse_perband(aligned, usable=usable), target):.6f}  "
              f"withheld {np.mean([1 - m.mean() for m in usable]) * 100:5.2f}%")


if __name__ == "__main__":
    main()
