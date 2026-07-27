"""Object motion from edges, so textureless interiors stop being unknowns (F87).

F85/F86 group regions by per-tile residual, which fails exactly where the tile
has nothing to correlate. The Lubriderm bottle is mostly flat white; its middle
carries no usable motion evidence at all, so tile clustering fragments it.

Its EDGES do not. A silhouette edge is high-contrast and trackable in every
frame, and three physical facts make it enough:

- an object's edges all move by the same amount, because the object is rigid;
- the object is not moving toward or away from the camera, so there is no scale
  change to solve for and the motion is a pure translation;
- so a flat interior bounded by co-moving edges belongs to those edges, and
  inherits their motion rather than needing evidence of its own.

The aperture problem is real and is handled by measuring only the component
along each edge's normal, then combining differently-oriented edges around the
same object: a vertical edge constrains horizontal motion, a horizontal one
constrains vertical, and an object's outline supplies both.

    .venv/bin/python research/edge_motion.py
"""
from __future__ import annotations

import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from focusstack.align import align_stack  # noqa: E402
from focusstack.io import to_gray_float  # noqa: E402

KITCHEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "data", "mobiledepth", "Figure3", "kitchen")
OUT = "/home/farron/camera/out/depth_align"
PATCH = 32
STRIDE = 8


def edge_samples(gray: np.ndarray, stride: int = STRIDE):
    """Sample points along strong edges, with each point's normal direction."""
    scaled = (gray * 255.0).astype(np.uint8)
    smoothed = cv2.GaussianBlur(scaled, (5, 5), 0)
    edges = cv2.Canny(smoothed, 60, 180) > 0
    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy) + 1e-6

    points = []
    h, w = gray.shape
    half = PATCH // 2
    for y in range(half, h - half, stride):
        for x in range(half, w - half, stride):
            if not edges[y, x]:
                continue
            points.append((x, y, float(gx[y, x] / magnitude[y, x]),
                           float(gy[y, x] / magnitude[y, x]),
                           float(magnitude[y, x])))
    return points


def normal_displacement(reference, moving, points):
    """Displacement of each edge along its own normal.

    Only the normal component is trusted. A straight edge carries no information
    about sliding along itself (the aperture problem), so the tangential part of
    any local match there is noise dressed as a measurement.
    """
    window = cv2.createHanningWindow((PATCH, PATCH), cv2.CV_64F)
    half = PATCH // 2
    out = []
    for x, y, nx, ny, strength in points:
        box = (slice(y - half, y + half), slice(x - half, x + half))
        patch = reference[box]
        if float(patch.std()) < 0.02:
            out.append(None)
            continue
        (dx, dy), response = cv2.phaseCorrelate(
            np.ascontiguousarray(patch.astype(np.float64)),
            np.ascontiguousarray(moving[box].astype(np.float64)),
            window,
        )
        out.append((float(dx * nx + dy * ny), float(response), nx, ny))
    return out


def main() -> None:
    src = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(KITCHEN, "*.jpg")))]
    coarse = align_stack(src, depth_bins=0, crop_valid=False)
    grays = [to_gray_float(image).astype(np.float32) / 255.0 for image in coarse]
    reference_index = len(coarse) // 2

    points = edge_samples(grays[reference_index])
    print(f"{len(points)} edge samples on the reference frame")

    # The bottle's two big vertical flats, and the background beside it.
    bands = {
        "bottle left edge  (x~487)": (470, 505, 160, 340),
        "bottle right edge (x~622)": (605, 640, 160, 340),
        "background right  (x~700)": (680, 730, 160, 340),
        "counter/foreground (y~430)": (300, 500, 400, 470),
    }

    for frame in (9, 11):
        measured = normal_displacement(grays[reference_index], grays[frame], points)
        print(f"\nframe {frame} vs reference, displacement along each edge's normal:")
        for label, (x0, x1, y0, y1) in bands.items():
            values, weights, verticals = [], [], []
            for (x, y, nx, ny, strength), result in zip(points, measured):
                if result is None or not (x0 <= x <= x1 and y0 <= y <= y1):
                    continue
                normal, response, rnx, rny = result
                if response < 0.15:
                    continue
                values.append(normal)
                weights.append(response)
                verticals.append(abs(rnx))
            if not values:
                print(f"  {label:28s}  no usable samples")
                continue
            values = np.array(values)
            print(f"  {label:28s}  n={len(values):3d}  "
                  f"normal displacement median {np.median(values):+6.2f} px  "
                  f"IQR {np.percentile(values, 75) - np.percentile(values, 25):5.2f}  "
                  f"mean |nx| {np.mean(verticals):.2f} (1.0 = vertical edge)")


if __name__ == "__main__":
    main()
