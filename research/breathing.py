"""Separate focus breathing from parallax by fitting them jointly (F90 work).

The physics, from PLAYBOOK §0: image displacement decomposes into a part that is
the same at every depth (camera rotation, and focus breathing's magnification) and
a part scaled by inverse depth (camera-centre translation). A single affine fitted
to the whole frame has to compromise between them, which is why the global stage
leaves 14% breathing on the kitchen sweep and why per-bin translation — which
cannot express a scale at all — then fragments a magnifying object.

Fitting them together is straightforward once stated, and linear:

    d(x, y) = s * (x - c)            <- breathing, depth-independent
            + theta * perp(x - c)    <- rotation, depth-independent
            + t_b                    <- parallax, one translation per depth bin

Unknowns are one scale, one rotation and two per bin, solved by weighted least
squares over tile observations that carry their own depth bin. The shared terms are
what the global stage should have removed; the per-bin terms are what the
depth-aware pass already handles.

    .venv/bin/python research/breathing.py
"""
from __future__ import annotations

import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import focusstack.align as align_mod  # noqa: E402
from focusstack.align import align_stack  # noqa: E402
from focusstack.fusion import depth_from_focus, fuse_perband  # noqa: E402
from focusstack.io import to_gray_float  # noqa: E402
import metrics  # noqa: E402
import parallax_gen as P  # noqa: E402

KITCHEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "data", "mobiledepth", "Figure3", "kitchen")
TILE = 48
MIN_RESPONSE = 0.12
IRLS_PASSES = 4


def tile_observations(reference, moving, bin_map, tile=TILE):
    """Per-tile displacement, each carrying the depth bin it sits in."""
    h, w = reference.shape
    window = cv2.createHanningWindow((tile, tile), cv2.CV_64F)
    rows = []
    for y in range(0, h - tile, tile // 2):
        for x in range(0, w - tile, tile // 2):
            box = (slice(y, y + tile), slice(x, x + tile))
            patch = reference[box]
            if float(patch.std()) < 0.02:
                continue
            labels = bin_map[box]
            # A tile straddling two depths belongs to neither; require dominance.
            counts = np.bincount(labels.ravel(), minlength=int(bin_map.max()) + 1)
            if counts.max() < 0.75 * labels.size:
                continue
            (dx, dy), response = cv2.phaseCorrelate(
                np.ascontiguousarray(patch.astype(np.float64)),
                np.ascontiguousarray(moving[box].astype(np.float64)),
                window,
            )
            if response < MIN_RESPONSE:
                continue
            rows.append((x + tile / 2.0, y + tile / 2.0, int(counts.argmax()),
                         float(dx), float(dy), float(response)))
    return rows


def fit_shared_scale(rows, shape, bins):
    """Solve one scale, one rotation, and a translation per depth bin.

    The shared terms are identifiable ONLY because different bins constrain the
    per-bin translations separately — with a single bin, a scale and a translation
    field are hopelessly confounded. Depth is what makes breathing separable.
    """
    if len(rows) < 4 + 2 * bins:
        return None
    h, w = shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    design, target, weights = [], [], []
    for px, py, b, dx, dy, response in rows:
        ux, uy = px - cx, py - cy
        row_x = [ux, -uy] + [0.0] * (2 * bins)
        row_y = [uy, ux] + [0.0] * (2 * bins)
        row_x[2 + 2 * b] = 1.0
        row_y[2 + 2 * b + 1] = 1.0
        design.append(row_x); target.append(dx); weights.append(response)
        design.append(row_y); target.append(dy); weights.append(response)

    A = np.asarray(design, float); y = np.asarray(target, float)
    w0 = np.asarray(weights, float)
    weight = w0.copy()
    solution = None
    for _ in range(IRLS_PASSES):
        root = np.sqrt(weight)[:, None]
        solution, *_ = np.linalg.lstsq(A * root, y * root.ravel(), rcond=None)
        residual = np.abs(A @ solution - y)
        cutoff = max(float(np.median(residual)) * 2.0, 1e-6)
        weight = w0 * np.minimum(1.0, cutoff / np.maximum(residual, 1e-9))
    return solution


def analyse(stack, label, truth=None, reference=None):
    coarse = align_stack(stack, depth_bins=0, crop_valid=False)
    grays = [to_gray_float(i).astype(np.float32) / 255.0 for i in coarse]
    ref = len(coarse) // 2 if reference is None else reference
    depth = depth_from_focus(coarse)
    edges = align_mod._valley_edges(depth.ravel(), 4)
    if edges is None:
        edges = np.unique(np.quantile(depth.ravel(), np.linspace(0, 1, 5)))
    bin_map = np.clip(np.digitize(depth, edges[1:-1]), 0, len(edges) - 2)
    bins = int(bin_map.max()) + 1

    print(f"\n{label}: {bins} depth bins")
    print(f"{'frame':>5} {'residual scale':>15} {'rotation (deg)':>15} "
          f"{'per-bin translations (x)':>30}")
    scales = []
    for k in range(len(grays)):
        if k == ref:
            continue
        rows = tile_observations(grays[ref], grays[k], bin_map)
        solution = fit_shared_scale(rows, grays[0].shape, bins)
        if solution is None:
            print(f"{k:5d}  too few observations")
            continue
        s, theta = solution[0], solution[1]
        tx = [solution[2 + 2 * b] for b in range(bins)]
        scales.append((k, s))
        print(f"{k:5d} {1 + s:15.4f} {np.degrees(theta):15.3f}   "
              + " ".join(f"{v:+7.2f}" for v in tx))
    return scales


if __name__ == "__main__":
    P.BREATHING_PER_FRAME = 0.025
    frames, truth, _ = P.build_stack()
    scales = analyse(frames, "factory with 2.5%/frame breathing applied", truth)
    print("\n  truth: the global affine should have absorbed the breathing entirely,")
    print("  so a residual scale far from 1.000 means it did not.")
    for k, s in scales:
        expected = 1.0 + 0.025 * (k - P.REFERENCE)
        print(f"    frame {k:2d}: true breathing {expected:.4f}, residual after affine {1+s:.4f}")

    src = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(KITCHEN, "*.jpg")))]
    analyse(src, "kitchen sweep")
