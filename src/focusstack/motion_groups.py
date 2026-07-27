"""Motion-group override for the depth-aware alignment (F102).

Depth bins and motion groups win on opposite scenes (F101). Where depth cleanly
separates a scene, bins are near-ideal. Where it does not — a bin covering half the
frame, fitted to its majority, containing an object that needs ten times its
correction — only grouping by measured motion isolates the object.

So this does not replace the depth path. It overrides it, and only where a group
demonstrably disagrees with the bin it sits in. A bin fitted to +2.3 px containing a
group measured at +18.5 px is not a marginal call; everything else is left exactly as
the depth path had it, which makes the override non-regressing by construction.

The pieces are all measured elsewhere and are load-bearing here:

* material edges only — a curved object's silhouette is a limb that slides with the
  viewpoint and is not a rigid feature (F92);
* trust a feature near its OWN focal plane — off it the feature blurs and its match
  collapses confidently toward zero shift, so match score cannot be the gate (F99);
* propagate a group's motion from the frames nearest the target rather than
  measuring it where the object is invisible (F89);
* support is the convex hull of a group's features, because features cluster on
  whatever part of an object carries texture and circles around them leave the rest
  of the body uncorrected (F100);
* claim strength is a GATE, not a scale factor — a pixel a group owns gets that
  group's motion in full (F100).
"""

from __future__ import annotations

import cv2
import numpy as np

from .io import to_gray_float

PROFILE_HALF = 28
PROFILE_SPAN = 24
STRIDE = 6
MIN_PEAK = 0.5
MIN_GROUP = 8
MIN_FRAMES = 3
# Consensus radius. Measured on the kitchen sweep: at 1.4 px the target object's
# group does not form at all, at 2.0 it forms at 93% purity. Material edges agree to
# about 1 px, so this is a little over two sigma.
INLIER_PX = 2.0
FOCAL_SIGMA = 2.5
MIN_SUPPORT = 4.0
SUPPORT_RADIUS = 26
CLAIM_FULL = 0.45
SHARPNESS = 6.0
# How far a group's own motion must diverge from its depth bin's before the bin is
# overruled. Well above measurement noise (~1 px) and far below the case this exists
# for, so a marginal disagreement never disturbs a working depth fit.
DISAGREEMENT_PX = 5.0


def _profile(gray, x, y, nx, ny):
    """Intensity across an edge, averaged along it."""
    tx, ty = -ny, nx
    offsets = np.arange(-PROFILE_HALF, PROFILE_HALF + 1, dtype=np.float32)
    spans = np.arange(-PROFILE_SPAN, PROFILE_SPAN + 1, dtype=np.float32)
    xs = x + np.outer(spans, tx) + np.outer(np.ones_like(spans), offsets) * nx
    ys = y + np.outer(spans, ty) + np.outer(np.ones_like(spans), offsets) * ny
    return cv2.remap(gray, xs.astype(np.float32), ys.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).mean(axis=0)


def _match(pa, pb):
    """Sub-pixel shift between two profiles, with a normalized peak."""
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


def _material_features(gray, depth, valid):
    """Edges that belong to a surface, not to a silhouette."""
    scaled = (gray * 255.0).astype(np.uint8)
    smoothed = cv2.GaussianBlur(scaled, (5, 5), 0)
    edges = cv2.Canny(smoothed, 60, 180) > 0
    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy) + 1e-6
    probe = np.ones((9, 9), np.uint8)
    step = cv2.dilate(depth, probe) - cv2.erode(depth, probe)

    h, w = gray.shape
    margin = PROFILE_HALF + PROFILE_SPAN + 2
    out = []
    ys, xs = np.nonzero(edges)
    for y, x in zip(ys, xs):
        if y % STRIDE or x % STRIDE:
            continue
        if not (margin <= x < w - margin and margin <= y < h - margin):
            continue
        if step[y, x] > 0.15 or not valid[y, x]:
            continue
        out.append((float(x), float(y), float(gx[y, x] / magnitude[y, x]),
                    float(gy[y, x] / magnitude[y, x])))
    return out


def _focal_frames(grays, features, half=9):
    out = []
    for (x, y, _nx, _ny) in features:
        xi, yi = int(round(x)), int(round(y))
        energy = [float(np.abs(cv2.Laplacian(
            g[yi - half:yi + half + 1, xi - half:xi + half + 1], cv2.CV_32F)).mean())
            for g in grays]
        out.append(float(np.argmax(energy)))
    return out


def _measure(grays, ref, features):
    table = np.full((len(features), len(grays)), np.nan)
    score = np.zeros((len(features), len(grays)))
    for i, (x, y, nx, ny) in enumerate(features):
        base = _profile(grays[ref], x, y, nx, ny)
        for k, gray in enumerate(grays):
            if k == ref:
                table[i, k] = 0.0
                score[i, k] = 1.0
                continue
            shift, peak = _match(base, _profile(gray, x, y, nx, ny))
            if peak >= MIN_PEAK and abs(shift) < 40:
                table[i, k] = shift
                score[i, k] = peak
    return table, score


def _consensus_groups(features, table, frames):
    """Greedy sets of features that one TRANSLATION explains in every frame.

    Translation-only on purpose: a model with a radial term can imitate two
    separated regions moving differently and will swallow a whole scene into one
    group with an excellent residual for the wrong structure (F96).
    """
    def fit(members):
        motions = {}
        for k in range(frames):
            rows, target = [], []
            for i in members:
                if np.isnan(table[i, k]):
                    continue
                _x, _y, nx, ny = features[i]
                rows.append([nx, ny])
                target.append(table[i, k])
            if len(rows) < 4:
                continue
            solution, *_ = np.linalg.lstsq(np.asarray(rows, float),
                                           np.asarray(target, float), rcond=None)
            motions[k] = solution
        return motions

    def residual(motions):
        out = np.full(len(features), np.inf)
        for i, (_x, _y, nx, ny) in enumerate(features):
            errs = [motions[k][0] * nx + motions[k][1] * ny - table[i, k]
                    for k in motions if not np.isnan(table[i, k])]
            if len(errs) >= MIN_FRAMES:
                out[i] = float(np.sqrt(np.mean(np.square(errs))))
        return out

    remaining = set(range(len(features)))
    groups = []
    for _ in range(6):
        if len(remaining) < MIN_GROUP:
            break
        pool = sorted(remaining)
        best = None
        for seed in pool[:: max(1, len(pool) // 24)]:
            sx, sy = features[seed][0], features[seed][1]
            near = [i for i in pool
                    if (features[i][0] - sx) ** 2 + (features[i][1] - sy) ** 2 < 120 ** 2]
            if len(near) < MIN_GROUP:
                continue
            motions = fit(near)
            if not motions:
                continue
            consensus = [i for i in pool if residual(motions)[i] < INLIER_PX]
            if len(consensus) >= MIN_GROUP and (best is None or len(consensus) > len(best)):
                best = consensus
        if best is None:
            break
        groups.append(best)
        remaining -= set(best)
    return groups


def _group_motion(features, table, score, focal, members, frames, ref):
    """Per-frame translation for one group, propagated where the group is blind."""
    raw, support = {}, {}
    for k in frames:
        rows, target, weight = [], [], []
        for i in members:
            if np.isnan(table[i, k]):
                continue
            _x, _y, nx, ny = features[i]
            w = score[i, k] * float(np.exp(-0.5 * ((k - focal[i]) / FOCAL_SIGMA) ** 2))
            rows.append([nx, ny])
            target.append(table[i, k])
            weight.append(w)
        if len(rows) < 3:
            continue
        A = np.asarray(rows, float)
        b = np.asarray(target, float)
        w = np.asarray(weight, float)
        root = np.sqrt(w)[:, None]
        solution, *_ = np.linalg.lstsq(A * root, b * root.ravel(), rcond=None)
        raw[k] = solution
        support[k] = float(w.sum())

    trusted = sorted(k for k in raw if support[k] >= MIN_SUPPORT)
    motion = {}
    for k in frames:
        if k in raw and support.get(k, 0.0) >= MIN_SUPPORT:
            motion[k] = raw[k]
        elif len(trusted) >= 2:
            near = sorted(trusted, key=lambda m: abs(m - k))[:4]
            t = np.array([m - ref for m in near], float)
            motion[k] = np.array([
                np.polyval(np.polyfit(t, [raw[m][0] for m in near], 1), k - ref),
                np.polyval(np.polyfit(t, [raw[m][1] for m in near], 1), k - ref)])
        elif k in raw:
            motion[k] = raw[k]
    return motion


def _support(features, members, shape, guide):
    """A group's body: the hull of its features, snapped to image structure."""
    seed = np.zeros(shape, np.float32)
    points = np.array([[int(round(features[i][0])), int(round(features[i][1]))]
                       for i in members], dtype=np.int32)
    if len(points) >= 3:
        cv2.fillConvexPoly(seed, cv2.convexHull(points), 1.0)
    for i in members:
        x, y, _nx, _ny = features[i]
        cv2.circle(seed, (int(round(x)), int(round(y))), SUPPORT_RADIUS, 1.0, -1)
    from .fusion import guided_filter
    return guided_filter(guide, seed, 24, 1e-3)


def overrides(images, coarse, valid, ref_index, depth, displacement_at):
    """Motion groups whose measured motion contradicts their depth bin's fit.

    `displacement_at(frame, x, y)` reports what the depth path already does at a
    point, so a group is compared against the correction it would actually replace,
    at its own location — not against some bin chosen frame-wide.

    Returns a list of (weight map, {frame: (dx, dy)}) to be composed into the
    sampling field alongside the depth bins, plus a small diagnostic. An empty list
    means every group agreed with its bin and the depth path is left untouched.
    """
    grays = [to_gray_float(image).astype(np.float32) / 255.0 for image in coarse]
    common = np.logical_and.reduce(valid)
    features = _material_features(grays[ref_index], depth, common)
    report = {"features": len(features), "groups": 0, "overridden": 0}
    if len(features) < 3 * MIN_GROUP:
        return [], report

    table, score = _measure(grays, ref_index, features)
    focal = _focal_frames(grays, features)
    groups = _consensus_groups(features, table, len(grays))
    report["groups"] = len(groups)
    if not groups:
        return [], report

    frames = [k for k in range(len(grays)) if k != ref_index]
    shape = grays[0].shape
    chosen = []
    for members in groups:
        motion = _group_motion(features, table, score, focal, members, frames, ref_index)
        if not motion:
            continue
        # Compare with what the depth path already does where this group lives.
        cx = float(np.mean([features[i][0] for i in members]))
        cy = float(np.mean([features[i][1] for i in members]))
        disagreement = 0.0
        for k, value in motion.items():
            fitted = displacement_at(k, cx, cy)
            if fitted is None:
                continue
            disagreement = max(disagreement, float(np.hypot(value[0] - fitted[0],
                                                            value[1] - fitted[1])))
        if disagreement < DISAGREEMENT_PX:
            continue
        chosen.append((_support(features, members, shape, grays[ref_index]), motion))

    if not chosen:
        return [], report
    report["overridden"] = len(chosen)

    # Sharpen so a compact group owns its own body rather than being outvoted by a
    # large one whose support spreads across the frame, and gate on claim strength
    # so an owned pixel receives the correction in full.
    stack = np.stack([np.clip(s, 0.0, None) for s, _ in chosen], axis=0)
    peak = stack.max(axis=0)
    sharp = np.where(peak[None] > 1e-6,
                     (stack / np.maximum(peak[None], 1e-6)) ** SHARPNESS, 0.0)
    total = sharp.sum(axis=0, keepdims=True)
    share = np.where(total > 1e-6, sharp / np.maximum(total, 1e-6), 0.0)
    claim = np.clip(peak / CLAIM_FULL, 0.0, 1.0)
    return [(share[j] * claim, motion) for j, (_s, motion) in enumerate(chosen)], report
