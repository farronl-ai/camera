"""Decompose camera motion into its components from all edges at once (F94 work).

The observation that motivates this: each component of camera motion leaves a
DIFFERENT SPATIAL SIGNATURE, and with a couple of hundred edges the problem is
massively over-determined, so the components can be read off directly instead of
guessed at.

| component            | spatial pattern                    | depth        |
|----------------------|------------------------------------|--------------|
| breathing            | radial (outward/inward from centre) | independent  |
| forward translation  | radial                              | scaled 1/Z   |
| pan / rotation       | near-uniform shift                  | independent  |
| lateral translation  | uniform direction                   | scaled 1/Z   |

Two consequences worth stating separately:

1. The quadrant SIGN PATTERN separates radial from uniform with no depth at all —
   under a radial component the left half moves left while the right half moves
   right, whereas a uniform component moves both the same way.
2. Depth then splits each pair, because only the translational components scale with
   inverse depth.

That hierarchy is why this session spent three findings confusing breathing with
forward translation: both are radial, and they are separable only through depth.

Fitted here as one weighted least-squares problem with per-bin translations as
nuisance parameters, and reported as variance explained per component so a stack can
be ASKED which effects it actually contains. Every edge is material (F92) and
confidence-gated (F89), because a defocusing edge moves for reasons that have nothing
to do with camera motion.

    .venv/bin/python research/motion_components.py
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
from focusstack.fusion import depth_from_focus  # noqa: E402
from focusstack.io import to_gray_float  # noqa: E402
import edge_similarity as ES  # noqa: E402
import object_segmentation as OS  # noqa: E402
import parallax_gen as P  # noqa: E402

KITCHEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "data", "mobiledepth", "Figure3", "kitchen")


def observations(grays, ref, frame, features):
    """Normal displacement per material edge, with its confidence."""
    rows = []
    for x, y, nx, ny in features:
        shift, peak = ES._match(ES._profile(grays[ref], x, y, nx, ny),
                                ES._profile(grays[frame], x, y, nx, ny))
        if peak >= 0.5 and abs(shift) < 40:
            rows.append((x, y, nx, ny, shift, peak))
    return rows


def decompose(rows, bin_map, shape, bins):
    """Solve radial + rotation + uniform shift + per-bin translation.

    The shared terms are identifiable because they have a spatial shape that no
    combination of per-region translations can imitate: a radial field reverses sign
    across the frame centre, and a rotational field is perpendicular to the radius.
    """
    h, w = shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    # NO uniform-translation term. A pan is indistinguishable from every depth
    # translating equally, so the two are confounded in image motion and only the
    # depth-VARYING part of translation is recoverable. Including both columns makes
    # the system singular and silently halves the answer (caught by known-answer
    # test: an applied +8 px read as +4.00). Radial is per-bin for the same reason
    # in reverse: breathing is the part its bins share, forward translation the part
    # that varies with depth.
    A, b, weight = [], [], []
    for x, y, nx, ny, shift, peak in rows:
        ux, uy = x - cx, y - cy
        depth_bin = int(bin_map[int(round(y)), int(round(x))])
        row = [-uy * nx + ux * ny] + [0.0] * (3 * bins)   # rotation is shared
        row[1 + 3 * depth_bin] = ux * nx + uy * ny        # radial, per depth
        row[1 + 3 * depth_bin + 1] = nx                   # translation, per depth
        row[1 + 3 * depth_bin + 2] = ny
        A.append(row); b.append(shift); weight.append(peak)
    if len(A) < 4 + 3 * bins:
        return None
    A = np.asarray(A, float); b = np.asarray(b, float)
    w0 = np.asarray(weight, float); wt = w0.copy()
    for _ in range(4):
        root = np.sqrt(wt)[:, None]
        solution, *_ = np.linalg.lstsq(A * root, b * root.ravel(), rcond=None)
        res = np.abs(A @ solution - b)
        cut = max(float(np.median(res)) * 2.5, 1e-6)
        wt = w0 * np.minimum(1.0, cut / np.maximum(res, 1e-9))

    radial = [solution[1 + 3 * j] for j in range(bins)]
    tx = [solution[1 + 3 * j + 1] for j in range(bins)]
    parts = {
        "rotation_deg": float(np.degrees(solution[0])),
        # Breathing is what every depth shares; forward translation is the spread.
        "breathing": float(np.median(radial)),
        "radial_spread": float(np.ptp(radial)) if bins > 1 else 0.0,
        "radial_per_bin": radial,
        "tx_per_bin": tx,
        "tx_spread": float(np.ptp(tx)) if bins > 1 else 0.0,
    }
    rms = float(np.sqrt(np.mean((A @ solution - b) ** 2)))
    return solution, parts, rms, len(A)


def quadrant_signature(rows, shape):
    """Mean horizontal displacement in the left vs right half.

    A radial component makes these OPPOSITE in sign; a uniform component makes them
    the same. This is the depth-free half of the decomposition, and it is a check on
    the fit rather than an input to it.
    """
    h, w = shape
    cx = (w - 1) / 2.0
    left = [s * nx for x, y, nx, ny, s, p in rows if x < cx and abs(nx) > 0.7]
    right = [s * nx for x, y, nx, ny, s, p in rows if x > cx and abs(nx) > 0.7]
    if not left or not right:
        return None
    return float(np.mean(left)), float(np.mean(right))


def report(stack, label, ref=None):
    coarse = align_stack(stack, depth_bins=0, crop_valid=False)
    grays = [to_gray_float(i).astype(np.float32) / 255.0 for i in coarse]
    ref = len(coarse) // 2 if ref is None else ref
    shape = grays[0].shape
    depth = depth_from_focus(coarse)
    edges = align_mod._valley_edges(depth.ravel(), 4)
    if edges is None:
        edges = np.unique(np.quantile(depth.ravel(), np.linspace(0, 1, 5)))
    bin_map = np.clip(np.digitize(depth, edges[1:-1]), 0, len(edges) - 2)
    bins = int(bin_map.max()) + 1
    features = OS.material_features(grays, ref, depth)

    print(f"\n{label}: {len(features)} material edges, {bins} depth bins")
    print(f"{'frame':>5} {'n':>4} | {'breathing':>10} {'radial spread':>14} "
          f"{'rot deg':>8} {'tx spread':>10} | {'rms':>5}")
    for k in range(len(grays)):
        if k == ref:
            continue
        rows = observations(grays, ref, k, features)
        result = decompose(rows, bin_map, shape, bins)
        if result is None:
            print(f"{k:5d}  too few observations")
            continue
        solution, parts, rms, n = result
        print(f"{k:5d} {n:4d} | {1 + parts['breathing']:10.4f} {parts['radial_spread']:14.4f} "
              f"{parts['rotation_deg']:+8.3f} {parts['tx_spread']:10.2f} | {rms:5.2f}")
    print("  breathing = magnification every depth shares; radial spread = its")
    print("  depth-VARYING part, i.e. forward translation; tx spread = lateral parallax.")
    print("  (a pan is confounded with all depths shifting equally and is not recoverable)")


if __name__ == "__main__":
    print("=== factory, where each component is synthesized to a known value ===")
    P.BREATHING_PER_FRAME = 0.0
    P.NEAR_SHIFT_PER_FRAME, P.FAR_SHIFT_PER_FRAME = 3.2, 0.7
    report(P.build_stack()[0], "parallax only (near 3.2, far 0.7 px/frame, no breathing)")

    P.BREATHING_PER_FRAME = 0.02
    P.NEAR_SHIFT_PER_FRAME, P.FAR_SHIFT_PER_FRAME = 0.0, 0.0
    report(P.build_stack()[0], "breathing only (2%/frame, no parallax)")

    P.BREATHING_PER_FRAME = 0.02
    P.NEAR_SHIFT_PER_FRAME, P.FAR_SHIFT_PER_FRAME = 3.2, 0.7
    report(P.build_stack()[0], "both")

    src = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(KITCHEN, "*.jpg")))]
    report(src, "kitchen sweep (real)")
