"""Group features into whole objects by shared rigid motion (F93 work).

The failure this exists to fix: motion clustering fragments objects. A bottle is
mostly flat white, its tiles carry no motion evidence, and its silhouette is a limb
that slides with the viewpoint (F92) — so a naive clusterer splits one object into
pieces and then fits each piece separately, which is worse than not splitting at all.

The model. Every admissible feature i sits at x_i with unit normal n_i, and in frame
k contributes ONE scalar observation, the displacement along its own normal (the
aperture problem forbids more). If features i and j belong to one rigid object then a
single motion per frame explains both, in every frame:

    d_i,k . n_i  =  [ s_k (x_i - c) + t_k ] . n_i     for all i in the object, all k

An object is therefore a maximal set of features admitting one motion sequence within
measurement noise. That is a falsifiable definition rather than a similarity
heuristic, and the whole sweep is used at once: an object is an object in every
frame, so a feature's residual PROFILE across frames is its membership signature.

Two admissibility rules, both physical:
  * material edges only — a curved object's limb is view-dependent (F92);
  * a feature is used in a frame only where its match is confident, since detail
    blurs away off the focal plane (F89) and absence of evidence is not evidence.

Consensus is found greedily: fit from a small seed, collect every feature the model
explains across all frames, refit on the consensus, repeat on what is left.

    .venv/bin/python research/object_segmentation.py
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
from focusstack.fusion import depth_from_focus  # noqa: E402
from focusstack.io import to_gray_float  # noqa: E402
import edge_similarity as ES  # noqa: E402

KITCHEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "data", "mobiledepth", "Figure3", "kitchen")
MIN_PEAK = 0.5
MIN_FRAMES = 3          # a feature must be measurable in at least this many frames
INLIER_PX = 2.0         # matched to the ~1 px agreement material edges actually show
MIN_OBJECT = 6


def material_features(grays, ref, depth, stride=6):
    """Edge features that are material, not limb.

    A limb edge has a DEPTH STEP across it — that is what a silhouette is. A printed
    or textured edge does not. Depth-from-focus is too coarse to place a contour
    (F83) but is perfectly adequate to answer whether depth changes across one, so
    it is used only for that yes/no.
    """
    gray = grays[ref]
    scaled = (gray * 255.0).astype(np.uint8)
    smoothed = cv2.GaussianBlur(scaled, (5, 5), 0)
    edges = cv2.Canny(smoothed, 60, 180) > 0
    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy) + 1e-6

    probe = np.ones((9, 9), np.uint8)
    depth_step = cv2.dilate(depth, probe) - cv2.erode(depth, probe)

    h, w = gray.shape
    margin = ES.PROFILE_HALF + ES.PROFILE_SPAN + 2
    out = []
    ys, xs = np.nonzero(edges)
    for y, x in zip(ys, xs):
        if y % stride or x % stride:
            continue
        if not (margin <= x < w - margin and margin <= y < h - margin):
            continue
        if depth_step[y, x] > 0.15:      # depth changes across it -> limb, reject
            continue
        out.append((float(x), float(y),
                    float(gx[y, x] / magnitude[y, x]),
                    float(gy[y, x] / magnitude[y, x])))
    return out


def observe(grays, ref, features):
    """Normal displacement of every feature in every frame, with confidence."""
    table = np.full((len(features), len(grays)), np.nan)
    for i, (x, y, nx, ny) in enumerate(features):
        base = ES._profile(grays[ref], x, y, nx, ny)
        for k, gray in enumerate(grays):
            if k == ref:
                table[i, k] = 0.0
                continue
            shift, peak = ES._match(base, ES._profile(gray, x, y, nx, ny))
            if peak >= MIN_PEAK and abs(shift) < 40:
                table[i, k] = shift
    return table


def fit_motion(features, table, members, shape, frames):
    """One (scale, tx, ty) per frame, fitted to the given member features."""
    h, w = shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    motions = {}
    for k in range(frames):
        # TRANSLATION ONLY, deliberately. A radial term can imitate two
        # spatially-separated regions translating differently, so a similarity
        # model lets one consensus swallow a whole two-plane scene and report an
        # excellent residual for the wrong structure — which is exactly what it
        # did on the analytic factory. The grouping model must be the one that
        # CANNOT explain a depth difference (F93/F96).
        rows, target = [], []
        for i in members:
            if np.isnan(table[i, k]):
                continue
            x, y, nx, ny = features[i]
            rows.append([nx, ny])
            target.append(table[i, k])
        if len(rows) < 4:
            continue
        A = np.asarray(rows, float); b = np.asarray(target, float)
        solution, *_ = np.linalg.lstsq(A, b, rcond=None)
        motions[k] = solution
    return motions


def residuals(features, table, motions, shape):
    """Per-feature RMS residual against a motion sequence, over measured frames."""
    h, w = shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    out = np.full(len(features), np.inf)
    for i, (x, y, nx, ny) in enumerate(features):
        errs = []
        for k, solution in motions.items():
            if np.isnan(table[i, k]):
                continue
            predicted = solution[0] * nx + solution[1] * ny
            errs.append(predicted - table[i, k])
        if len(errs) >= MIN_FRAMES:
            out[i] = float(np.sqrt(np.mean(np.square(errs))))
    return out


def segment(features, table, shape, frames, max_objects=6):
    """Greedy consensus: repeatedly take the largest set one motion explains."""
    remaining = set(range(len(features)))
    objects = []
    for _ in range(max_objects):
        if len(remaining) < MIN_OBJECT:
            break
        best = None
        pool = sorted(remaining)
        # Seed from spatial neighbourhoods: an object is contiguous, so nearby
        # features are the cheapest place to look for a shared motion.
        for seed in pool[:: max(1, len(pool) // 24)]:
            sx, sy = features[seed][0], features[seed][1]
            near = [i for i in pool
                    if (features[i][0] - sx) ** 2 + (features[i][1] - sy) ** 2 < 120 ** 2]
            if len(near) < MIN_OBJECT:
                continue
            motions = fit_motion(features, table, near, shape, frames)
            if not motions:
                continue
            consensus = [i for i in pool
                         if residuals(features, table, motions, shape)[i] < INLIER_PX]
            if len(consensus) >= MIN_OBJECT and (best is None or len(consensus) > len(best)):
                best = consensus
        if best is None:
            break
        motions = fit_motion(features, table, best, shape, frames)
        objects.append((best, motions))
        remaining -= set(best)
    return objects, remaining


def main() -> None:
    src = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(KITCHEN, "*.jpg")))]
    coarse = align_stack(src, depth_bins=0, crop_valid=False)
    grays = [to_gray_float(i).astype(np.float32) / 255.0 for i in coarse]
    ref = len(coarse) // 2
    shape = grays[0].shape
    depth = depth_from_focus(coarse)

    features = material_features(grays, ref, depth)
    print(f"{len(features)} material edge features (limb edges rejected by depth step)")
    table = observe(grays, ref, features)
    measured = np.sum(~np.isnan(table), axis=1)
    keep = [i for i in range(len(features)) if measured[i] >= MIN_FRAMES]
    print(f"{len(keep)} measurable in >= {MIN_FRAMES} frames\n")

    features = [features[i] for i in keep]
    table = table[keep]
    objects, leftover = segment(features, table, shape, len(grays))

    bottle = (498, 639, 128, 392)
    print(f"{'object':>7} {'features':>9} {'in bottle box':>14} {'tx @frame 11':>13} {'scale':>8}")
    for j, (members, motions) in enumerate(objects):
        inside = sum(1 for i in members
                     if bottle[0] <= features[i][0] <= bottle[1]
                     and bottle[2] <= features[i][1] <= bottle[3])
        solution = motions.get(11)
        tx = f"{solution[1]:+.2f}" if solution is not None else "n/a"
        sc = f"{1 + solution[0]:.4f}" if solution is not None else "n/a"
        print(f"{j:7d} {len(members):9d} {inside:14d} {tx:>13} {sc:>8}")
    print(f"{'unassigned':>7} {len(leftover):9d}")
    print("\ntruth: the bottle needs about +19.2 px at frame 11, scale ~1.00")


if __name__ == "__main__":
    main()
