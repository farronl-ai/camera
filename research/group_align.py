"""Alignment keyed on motion groups, not depth (F100 work).

Everything needed for this was already measured and never connected (F99):

  * F92 — material edges only; a curved object's limb is view-dependent.
  * F93 — an object is a maximal feature set admitting one rigid motion. Grouping
    this way isolates the kitchen bottle at 92.9-100% purity, where depth does not.
  * F89 — measure an object near its own focal plane and PROPAGATE; off it, the
    object's features blur and their matches collapse toward zero shift.
  * F99 — do not key the correction on depth. Content at one depth VALUE can have
    different motion, so a depth-keyed fit averages the target away: the curve at the
    bottle's depth never exceeded 5 px in any frame while its own features reached 19.

So: group by motion, fit each group's own translation per frame weighted by focal
proximity, propagate across the sweep where a group is blurred, and paint each
group's correction only where its own features say it lives. Support is easy here
precisely because a motion group is compact — its features bound it — which is the
part F98 made hard by trying to segment the whole frame.

    .venv/bin/python research/group_align.py
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
from focusstack.fusion import depth_from_focus, fuse_perband, guided_filter  # noqa: E402
from focusstack.io import to_gray_float  # noqa: E402
import edge_similarity as ES  # noqa: E402
import metrics  # noqa: E402
import object_segmentation as OS  # noqa: E402
import parallax_gen as P  # noqa: E402

KITCHEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "data", "mobiledepth", "Figure3", "kitchen")
FOCAL_SIGMA = 2.5      # frames; how far from its focal plane a feature stays usable
MIN_WEIGHT = 4.0       # total focal-weighted support a group needs in a frame
MIN_GROUP = 8
SUPPORT_RADIUS = 26
_SHARPNESS = 6.0
_CLAIM_FULL = 0.45   # guided-support value at which a group fully owns a pixel


def focal_frames(grays, features, half=9):
    """The frame where each feature is sharpest — the only place it can be trusted."""
    out = []
    for (x, y, nx, ny) in features:
        xi, yi = int(round(x)), int(round(y))
        energy = [float(np.abs(cv2.Laplacian(
            g[yi - half:yi + half + 1, xi - half:xi + half + 1], cv2.CV_32F)).mean())
            for g in grays]
        out.append(float(np.argmax(energy)))
    return out


def measure(grays, ref, features):
    """Normal displacement of every feature in every frame, with a match score."""
    table = np.full((len(features), len(grays)), np.nan)
    score = np.zeros((len(features), len(grays)))
    for i, (x, y, nx, ny) in enumerate(features):
        base = ES._profile(grays[ref], x, y, nx, ny)
        for k, gray in enumerate(grays):
            if k == ref:
                table[i, k] = 0.0; score[i, k] = 1.0
                continue
            shift, peak = ES._match(base, ES._profile(gray, x, y, nx, ny))
            if peak >= 0.5 and abs(shift) < 40:
                table[i, k] = shift; score[i, k] = peak
    return table, score


def group_motion(features, table, score, focal, members, frames, ref):
    """Per-frame translation for one group, then propagated where it is blind.

    Weighting by focal proximity is not optional. A blurred profile matches a sharp
    one confidently at about zero shift, so an unweighted fit reports that the
    object stopped moving exactly when it left focus (F99).
    """
    raw, support = {}, {}
    for k in frames:
        A, b, w = [], [], []
        for i in members:
            if np.isnan(table[i, k]):
                continue
            x, y, nx, ny = features[i]
            weight = score[i, k] * float(np.exp(-0.5 * ((k - focal[i]) / FOCAL_SIGMA) ** 2))
            A.append([nx, ny]); b.append(table[i, k]); w.append(weight)
        if len(A) < 3:
            continue
        A = np.asarray(A); b = np.asarray(b); w = np.asarray(w)
        root = np.sqrt(w)[:, None]
        solution, *_ = np.linalg.lstsq(A * root, b * root.ravel(), rcond=None)
        raw[k] = solution
        support[k] = float(w.sum())

    trusted = sorted(k for k in raw if support[k] >= MIN_WEIGHT)
    motion = {}
    for k in frames:
        if k in raw and support.get(k, 0.0) >= MIN_WEIGHT:
            motion[k] = raw[k]
            continue
        if len(trusted) < 2:
            motion[k] = raw.get(k, np.zeros(2))
            continue
        # Propagate from the frames NEAREST this one: handheld drift is not linear
        # across a whole sweep (F93), so distant frames bias the local slope.
        near = sorted(trusted, key=lambda m: abs(m - k))[:4]
        t = np.array([m - ref for m in near], float)
        motion[k] = np.array([
            np.polyval(np.polyfit(t, [raw[m][0] for m in near], 1), k - ref),
            np.polyval(np.polyfit(t, [raw[m][1] for m in near], 1), k - ref),
        ])
    return motion


def group_support(features, members, shape, guide):
    """Where a motion group lives, grown from its own features and snapped to edges.

    The convex hull matters: a group's features cluster on whatever part of the
    object carries texture — on the kitchen bottle, its printed label — so circles
    around them leave the object's own top and bottom unclaimed and uncorrected. An
    object is connected, so the hull of its features is a far better first guess at
    its body than their immediate neighbourhoods.
    """
    seed = np.zeros(shape, np.float32)
    points = np.array([[int(round(features[i][0])), int(round(features[i][1]))]
                       for i in members], dtype=np.int32)
    if len(points) >= 3:
        cv2.fillConvexPoly(seed, cv2.convexHull(points), 1.0)
    for i in members:
        x, y, _, _ = features[i]
        cv2.circle(seed, (int(round(x)), int(round(y))), SUPPORT_RADIUS, 1.0, -1)
    return guided_filter(guide, seed, 24, 1e-3)


def focal_seeded_groups(features, table, focal, shape, frames):
    """Group by focal signature FIRST, then split each depth by motion.

    Motion consensus alone collapses the analytic factory's two planes into one
    group (F96): a greedy search that maximizes consensus SIZE rewards the
    compromise fit sitting midway between them, and no inlier threshold serves both
    that scene and the kitchen (F97). The focal signature does separate them, with
    no threshold, so it seeds the partition and motion only refines within a depth —
    which is the division of labour F97 recommended and this had not yet adopted.
    """
    peaks = np.asarray(focal, float)
    labels = np.zeros(len(features), int)
    for _ in range(4):
        best = None
        for g in range(labels.max() + 1):
            members = np.nonzero(labels == g)[0]
            if len(members) < 2 * MIN_GROUP:
                continue
            ordered = np.sort(peaks[members])
            total = float(np.var(ordered))
            if total < 1e-9:
                continue
            for i in range(MIN_GROUP, len(ordered) - MIN_GROUP):
                lo, hi = ordered[:i], ordered[i:]
                w = len(lo) / len(ordered)
                quality = w * (1 - w) * (lo.mean() - hi.mean()) ** 2 / total
                if quality >= 0.55 and (best is None or quality > best[2]):
                    best = (g, 0.5 * (ordered[i - 1] + ordered[i]), quality)
        if best is None:
            break
        g, threshold, _ = best
        members = np.nonzero(labels == g)[0]
        labels[members[peaks[members] > threshold]] = labels.max() + 1

    groups = []
    for g in range(labels.max() + 1):
        members = list(np.nonzero(labels == g)[0])
        if len(members) < MIN_GROUP:
            continue
        # Within one depth, motion consensus may still find two rigid bodies.
        sub, _ = OS.segment([features[i] for i in members],
                            table[members], shape, frames)
        if len(sub) > 1:
            for part, _ in sub:
                if len(part) >= MIN_GROUP:
                    groups.append([members[j] for j in part])
        else:
            groups.append(members)
    return groups


def align_by_groups(frames_in):
    coarse = align_stack(frames_in, depth_bins=0, crop_valid=False)
    grays = [to_gray_float(i).astype(np.float32) / 255.0 for i in coarse]
    ref = len(coarse) // 2
    shape = grays[0].shape
    depth = depth_from_focus(coarse)

    features = OS.material_features(grays, ref, depth)
    if len(features) < 3 * MIN_GROUP:
        return coarse, None
    table, score = measure(grays, ref, features)
    focal = focal_frames(grays, features)

    groups = focal_seeded_groups(features, table, focal, shape, len(grays))
    if len(groups) < 2:
        return coarse, None

    frames = [k for k in range(len(grays)) if k != ref]
    motions = [group_motion(features, table, score, focal, m, frames, ref) for m in groups]
    supports = [group_support(features, m, shape, grays[ref]) for m in groups]

    # Sharpen the memberships. Proportional blending lets a large background group,
    # whose support spreads over the whole frame, outvote a small compact one inside
    # its own object: the bottle's group measured +18.54 px and only about a quarter
    # of it reached the bottle. A group should own the territory its own features
    # claim, and blend only where two genuinely compete.
    stack = np.stack([np.clip(s, 0.0, None) for s in supports], axis=0)
    peak = stack.max(axis=0, keepdims=True)
    sharp = np.where(peak > 1e-6, (stack / np.maximum(peak, 1e-6)) ** _SHARPNESS, 0.0)
    total_sharp = sharp.sum(axis=0, keepdims=True)
    weights = list(np.where(total_sharp > 1e-6, sharp / np.maximum(total_sharp, 1e-6), 0.0))
    # Claim strength must be a GATE, not a scale factor. Derived from the raw guided
    # value it stayed below 1 even deep inside an object — the bottle's own label
    # centre was fully owned by its group and still had 30% of its correction shaved
    # off, and its edge 42%. A pixel a group clearly owns gets that group's motion in
    # full; only genuinely unclaimed pixels fall back to the global warp.
    claim = np.clip(peak[0] / _CLAIM_FULL, 0.0, 1.0)
    unclaimed = 1.0 - claim

    h, w = shape
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
    out, usable = [], []
    for k in range(len(grays)):
        if k == ref:
            out.append(coarse[k]); usable.append(np.ones(shape, bool)); continue
        dx = np.zeros(shape, np.float32); dy = np.zeros(shape, np.float32)
        for weight, motion in zip(weights, motions):
            m = motion.get(k)
            if m is None:
                continue
            dx += weight * (1.0 - unclaimed) * float(m[0])
            dy += weight * (1.0 - unclaimed) * float(m[1])
        out.append(cv2.remap(coarse[k], grid_x + dx, grid_y + dy, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE))
        # Parallax uncovers scene here exactly as it does for the shipped path
        # (F82): where this frame's displacement disagrees across a real depth
        # step, the observation behind the occluder does not exist.
        probe = np.ones((5, 5), np.uint8)
        step = ((cv2.dilate(depth, probe) - cv2.erode(depth, probe))
                > align_mod._OCCLUSION_MIN_DEPTH_STEP).astype(np.uint8)
        usable.append(~align_mod._occlusion_mask(dx, dy, step))
    return out, {"groups": groups, "features": features, "motions": motions,
                 "usable": usable}


def edge_shift(a, b, y0, y1, x0, x1):
    pa = np.gradient(a[y0:y1, x0:x1].mean(0)); pb = np.gradient(b[y0:y1, x0:x1].mean(0))
    pa = pa - pa.mean(); pb = pb - pb.mean()
    c = np.correlate(pb, pa, mode="full"); i = int(np.argmax(c)); off = 0.0
    if 0 < i < len(c) - 1:
        d = c[i - 1] - 2 * c[i] + c[i + 1]
        off = 0.5 * (c[i - 1] - c[i + 1]) / d if abs(d) > 1e-12 else 0.0
    return (i - (len(pa) - 1)) + off


def main() -> None:
    src = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(KITCHEN, "*.jpg")))]
    aligned, info = align_by_groups(src)
    coarse = align_stack(src, depth_bins=0, crop_valid=False)
    g = [to_gray_float(i).astype(np.float32) / 255.0 for i in coarse]
    ga = [to_gray_float(i).astype(np.float32) / 255.0 for i in aligned]
    ref = len(g) // 2

    if info:
        box = (498, 639, 128, 392)
        feats = info["features"]
        print("kitchen groups:")
        for j, members in enumerate(info["groups"]):
            inb = 100 * np.mean([box[0] <= feats[i][0] <= box[1]
                                 and box[2] <= feats[i][1] <= box[3] for i in members])
            m11 = info["motions"][j].get(11)
            print(f"  group {j}: {len(members):3d} features, {inb:5.1f}% in the bottle,"
                  f" frame-11 tx {m11[0]:+6.2f}" if m11 is not None else "")

    print("\nresidual misregistration of the bottle's right edge:")
    for k in (8, 9, 10, 11):
        before = edge_shift(g[ref], g[k], 160, 340, 600, 700)
        after = edge_shift(ga[ref], ga[k], 160, 340, 600, 700)
        print(f"  frame {k}: global-only {before:+6.2f} px -> motion groups {after:+6.2f} px")


if __name__ == "__main__":
    main()
