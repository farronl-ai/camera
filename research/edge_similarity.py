"""Object similarity (scale + translation) from edges alone (F91 work).

F90 established that a region needs a radial SCALE term, because the residual
magnification is depth-scaled forward camera translation and no global stage can
remove it. F89 established that interior edges over-determine exactly that fit and
make the one-object hypothesis falsifiable. This joins them, generalized to 2-D.

Each edge sample constrains only the component along its own normal — a straight
edge says nothing about sliding along itself — which is linear in the unknowns:

    d . n = s * ((x - c) . n)  +  t . n

Three unknowns (one scale, two translation) against many constraints, so the fit is
over-determined: its residual is a test, not a formality. Edges also work where tile
correlation cannot, on the flat interiors that fragment motion-clustered objects.

Two properties this is built to deliver:
  * a REGION MODEL that can represent what a near object actually does; and
  * a HETEROGENEITY SIGNAL — a region holding two objects cannot be fitted by one
    similarity, and says so through its residual, which is the physical replacement
    for the tile-confidence threshold that could not serve two scenes at once.

    .venv/bin/python research/edge_similarity.py
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

KITCHEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "data", "mobiledepth", "Figure3", "kitchen")
PROFILE_HALF = 28      # half-length of the 1-D profile sampled along the normal
PROFILE_SPAN = 24      # half-length of the averaging window along the tangent
MIN_PEAK = 0.5
STRIDE = 6


def edge_samples(gray, mask, stride=STRIDE):
    """Edge points inside `mask`, with unit normals, thinned to local maxima."""
    scaled = (gray * 255.0).astype(np.uint8)
    smoothed = cv2.GaussianBlur(scaled, (5, 5), 0)
    edges = cv2.Canny(smoothed, 60, 180) > 0
    edges &= cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy) + 1e-6

    h, w = gray.shape
    points = []
    margin = PROFILE_HALF + PROFILE_SPAN + 2
    ys, xs = np.nonzero(edges)
    for y, x in zip(ys, xs):
        if y % stride or x % stride:
            continue
        if not (margin <= x < w - margin and margin <= y < h - margin):
            continue
        points.append((float(x), float(y),
                       float(gx[y, x] / magnitude[y, x]),
                       float(gy[y, x] / magnitude[y, x])))
    return points


def _profile(gray, x, y, nx, ny):
    """1-D intensity profile across an edge, averaged along the edge direction.

    Averaging along the tangent is what turns a weak local match into a strong
    measurement; sampling along the normal is what keeps the aperture problem out.
    """
    tx, ty = -ny, nx
    offsets = np.arange(-PROFILE_HALF, PROFILE_HALF + 1, dtype=np.float32)
    spans = np.arange(-PROFILE_SPAN, PROFILE_SPAN + 1, dtype=np.float32)
    xs = x + np.outer(spans, tx) + np.outer(np.ones_like(spans), offsets) * nx
    ys = y + np.outer(spans, ty) + np.outer(np.ones_like(spans), offsets) * ny
    sampled = cv2.remap(gray, xs.astype(np.float32), ys.astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return sampled.mean(axis=0)


def _match(pa, pb):
    """Sub-pixel shift between two profiles, with a normalized peak as confidence."""
    pa = np.gradient(pa - pa.mean())
    pb = np.gradient(pb - pb.mean())
    denom = np.sqrt((pa ** 2).sum() * (pb ** 2).sum()) + 1e-12
    c = np.correlate(pb, pa, mode="full") / denom
    i = int(np.argmax(c))
    off = 0.0
    if 0 < i < len(c) - 1:
        d = c[i - 1] - 2 * c[i] + c[i + 1]
        off = 0.5 * (c[i - 1] - c[i + 1]) / d if abs(d) > 1e-12 else 0.0
    return (i - (len(pa) - 1)) + off, float(c[i])


def fit_region_similarity(reference, moving, mask, shape=None):
    """Scale + translation for one region from its edges, with a residual test."""
    shape = shape or reference.shape
    h, w = shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    points = edge_samples(reference, mask)
    if len(points) < 6:
        return None

    rows, target, weights = [], [], []
    for x, y, nx, ny in points:
        shift, peak = _match(_profile(reference, x, y, nx, ny),
                             _profile(moving, x, y, nx, ny))
        if peak < MIN_PEAK or abs(shift) > 40:
            continue
        ux, uy = x - cx, y - cy
        rows.append([ux * nx + uy * ny, nx, ny])
        target.append(shift)
        weights.append(peak)
    if len(rows) < 6:
        return None

    A = np.asarray(rows, float); b = np.asarray(target, float)
    w0 = np.asarray(weights, float); weight = w0.copy()
    solution = None
    for _ in range(4):
        root = np.sqrt(weight)[:, None]
        solution, *_ = np.linalg.lstsq(A * root, b * root.ravel(), rcond=None)
        residual = np.abs(A @ solution - b)
        cutoff = max(float(np.median(residual)) * 2.0, 1e-6)
        weight = w0 * np.minimum(1.0, cutoff / np.maximum(residual, 1e-9))
    residual = A @ solution - b
    return {
        "scale": 1.0 + float(solution[0]),
        "tx": float(solution[1]),
        "ty": float(solution[2]),
        "rms": float(np.sqrt((residual ** 2).mean())),
        "n": len(rows),
    }


def main() -> None:
    src = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(KITCHEN, "*.jpg")))]
    coarse = align_stack(src, depth_bins=0, crop_valid=False)
    grays = [to_gray_float(i).astype(np.float32) / 255.0 for i in coarse]
    ref = len(coarse) // 2
    shape = grays[0].shape

    depth = depth_from_focus(coarse)
    edges = align_mod._valley_edges(depth.ravel(), 4)
    bin_map = np.clip(np.digitize(depth, edges[1:-1]), 0, len(edges) - 2)

    bottle = np.zeros(shape, bool)
    bottle[128:392, 498:639] = True
    owner = int(np.bincount(bin_map[bottle].ravel()).argmax())
    whole_bin = bin_map == owner

    print("truth for frame 11: the bottle needs about +19.2 px and magnifies ~1.08\n")
    print(f"{'frame':>5} {'region':>22} {'n':>4} {'scale':>8} {'tx':>8} {'ty':>7} {'rms':>7}")
    for k in (8, 9, 10, 11):
        for label, mask in (("bottle (object-sized)", bottle),
                            (f"its whole bin ({100*whole_bin.mean():.0f}% of frame)", whole_bin)):
            fit = fit_region_similarity(grays[ref], grays[k], mask, shape)
            if fit is None:
                print(f"{k:5d} {label:>22}  too few edges")
                continue
            print(f"{k:5d} {label:>22} {fit['n']:4d} {fit['scale']:8.4f} "
                  f"{fit['tx']:+8.2f} {fit['ty']:+7.2f} {fit['rms']:7.2f}")
    print("\nrms is the heterogeneity signal: one object fits one similarity,")
    print("a region holding several cannot, and says so without ground truth.")


if __name__ == "__main__":
    main()
