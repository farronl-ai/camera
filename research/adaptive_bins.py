"""Residual-driven bin splitting (F84 follow-up).

F84 located the blocker: a depth bin is a range, not an object. On the kitchen
sweep the bin holding the bottle covers 55% of the frame, so ECC fits the other
91% of its pixels and the bottle is left 19 px out of register. Raising the
acceptance cap does nothing, because the correction is never proposed.

The reframe: stop grouping by depth and group by MEASURED MOTION. Depth is only
a seed. "Wants the same correction" is the operational definition of an object
here, and the per-tile residual after a bin's own fit measures exactly that.

Evidence is pooled across frames on purpose. A single frame's residual is noisy
and could be a bad phase correlation, but an object is an object in every frame:
its residual grows with that frame's motion while staying consistent in
direction, so the residual PROFILE across the stack is a far stronger signature
than any one frame's number.

    .venv/bin/python research/adaptive_bins.py
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
from focusstack import io as fio  # noqa: E402
from focusstack.align import align_stack  # noqa: E402
from focusstack.fusion import (depth_from_focus, fuse_perband,  # noqa: E402
                               guided_filter)
from focusstack.io import to_gray_float  # noqa: E402

KITCHEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "data", "mobiledepth", "Figure3", "kitchen")
OUT = "/home/farron/camera/out/depth_align"
TILE = 40
MIN_TILES_PER_GROUP = 4
SPLIT_THRESHOLD_PX = 2.0
# A split must be justified by a residual big enough to matter, in absolute
# pixels, not merely by two groups differing from each other. Bins that are
# already homogeneous — as they are whenever depth genuinely separates the
# scene — must come back unsplit, or the extra regions invent structure and
# each carries its own fitting error.
MIN_STRANDED_RESIDUAL_PX = 4.0
# An object is one thing. A real stranded object is a coherent connected piece
# of the picture that moves rigidly; a bad phase correlation off a disoccluded
# tile is confetti. Requiring coherence is a physical constraint rather than a
# tuned threshold, and it is what separates a scene whose bins genuinely hold
# two objects from one whose bins are already right.
MAX_PIECES_PER_GROUP = 3
MIN_COHERENT_FRACTION = 0.6


def fit_similarity(reference, moving, mask, tile=40):
    """Scale AND translation for one region, from its own tile displacements.

    Translation alone cannot express what a near region actually does: the camera
    drifts forward as well as sideways, and forward motion magnifies content in
    proportion to inverse depth (measured on the kitchen bottle: near 1.085 versus
    far 1.032 in the same aligned frames). A region undergoing magnification that
    is fitted with a translation is fitted wrongly everywhere except its centre,
    which is what makes a motion-clustered object fragment.
    """
    h, w = reference.shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    window = cv2.createHanningWindow((tile, tile), cv2.CV_64F)
    A, y, wt = [], [], []
    for py in range(0, h - tile, tile // 2):
        for px in range(0, w - tile, tile // 2):
            box = (slice(py, py + tile), slice(px, px + tile))
            if mask[box].mean() < 0.75:
                continue
            patch = reference[box]
            if float(patch.std()) < 0.02:
                continue
            (dx, dy), response = cv2.phaseCorrelate(
                np.ascontiguousarray(patch.astype(np.float64)),
                np.ascontiguousarray(moving[box].astype(np.float64)), window)
            if response < 0.12:
                continue
            ux, uy = px + tile / 2.0 - cx, py + tile / 2.0 - cy
            A.append([ux, 1.0, 0.0]); y.append(dx); wt.append(response)
            A.append([uy, 0.0, 1.0]); y.append(dy); wt.append(response)
    if len(A) < 8:
        return None
    A = np.asarray(A, float); y = np.asarray(y, float)
    w0 = np.asarray(wt, float); weight = w0.copy()
    for _ in range(4):
        root = np.sqrt(weight)[:, None]
        sol, *_ = np.linalg.lstsq(A * root, y * root.ravel(), rcond=None)
        res = np.abs(A @ sol - y)
        cut = max(float(np.median(res)) * 2.0, 1e-6)
        weight = w0 * np.minimum(1.0, cut / np.maximum(res, 1e-9))
    return float(sol[0]), float(sol[1]), float(sol[2])   # scale, tx, ty


def residual_after_similarity(reference, moving, mask, model, tile=40):
    """Per-tile residual once the region's own scale+translation is removed."""
    h, w = reference.shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    scale, tx, ty = model
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
    map_x = (grid_x + scale * (grid_x - cx) + tx).astype(np.float32)
    map_y = (grid_y + scale * (grid_y - cy) + ty).astype(np.float32)
    warped = cv2.remap(moving, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    window = cv2.createHanningWindow((tile, tile), cv2.CV_64F)
    out = []
    for py in range(0, h - tile, tile):
        for px in range(0, w - tile, tile):
            box = (slice(py, py + tile), slice(px, px + tile))
            if mask[box].mean() < 0.6:
                continue
            patch = reference[box]
            if float(patch.std()) < 0.02:
                continue
            (dx, dy), response = cv2.phaseCorrelate(
                np.ascontiguousarray(patch.astype(np.float64)),
                np.ascontiguousarray(warped[box].astype(np.float64)), window)
            out.append((px + tile // 2, py + tile // 2, float(dx), float(dy), response))
    return out


def fit_translation(reference, moving, mask):
    """Best single translation for one region, or None if it will not converge."""
    if mask.sum() < 500:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(reference, moving, warp, cv2.MOTION_TRANSLATION,
                                       criteria, mask.astype(np.uint8) * 255, 5)
    except cv2.error:
        return None
    if not np.isfinite(warp).all():
        return None
    return float(warp[0, 2]), float(warp[1, 2])


def tile_grid(shape, mask):
    """Tiles whose support lies inside this region."""
    h, w = shape
    boxes = []
    for y in range(0, h - TILE, TILE):
        for x in range(0, w - TILE, TILE):
            box = (slice(y, y + TILE), slice(x, x + TILE))
            if mask[box].mean() > 0.6:
                boxes.append((x, y, box))
    return boxes


def tile_residuals(reference, moving, boxes, fitted):
    """Residual left in each tile after the region's own fit is applied."""
    h, w = reference.shape
    shifted = cv2.warpAffine(
        moving, np.float32([[1, 0, fitted[0]], [0, 1, fitted[1]]]), (w, h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE,
    )
    window = cv2.createHanningWindow((TILE, TILE), cv2.CV_64F)
    out = []
    for x, y, box in boxes:
        patch = reference[box]
        if float(patch.std()) < 0.02:
            out.append(None)
            continue
        (dx, dy), response = cv2.phaseCorrelate(
            np.ascontiguousarray(patch.astype(np.float64)),
            np.ascontiguousarray(shifted[box].astype(np.float64)),
            window,
        )
        out.append((float(dx), float(dy), float(response)))
    return out


def two_means(features, iterations=20):
    """Split tiles into two motion groups. Plain numpy; no sklearn on 3.14."""
    if len(features) < 2 * MIN_TILES_PER_GROUP:
        return None
    data = np.asarray(features, dtype=np.float64)
    # Seed at the extremes of the dominant direction, which is far more stable
    # than random seeding when one group is a small minority — and the minority
    # is exactly the object we are trying to rescue.
    spread = data - data.mean(axis=0)
    direction = np.linalg.svd(spread, full_matrices=False)[2][0]
    projection = spread @ direction
    centres = np.array([data[projection.argmin()], data[projection.argmax()]])
    labels = np.zeros(len(data), dtype=int)
    for _ in range(iterations):
        distances = np.stack([np.linalg.norm(data - c, axis=1) for c in centres])
        new_labels = distances.argmin(axis=0)
        if (new_labels == labels).all():
            break
        labels = new_labels
        for k in (0, 1):
            if (labels == k).any():
                centres[k] = data[labels == k].mean(axis=0)
    if min((labels == 0).sum(), (labels == 1).sum()) < MIN_TILES_PER_GROUP:
        return None
    separation = float(np.linalg.norm(centres[0] - centres[1]))
    if separation < SPLIT_THRESHOLD_PX:
        return None
    # Per-frame residual magnitude of the worse group: the feature vector
    # concatenates (dx, dy) per frame, so norms are taken pairwise.
    # Max across frames, not median: a sweep spends most of its frames near the
    # reference, where every region is well fitted, so a median hides an object
    # that is stranded badly in the few frames that actually moved. Being
    # stranded in ANY frame is enough to deserve a region.
    stranded = max(
        float(np.max(np.hypot(c[0::2], c[1::2]))) for c in centres
    )
    if stranded < MIN_STRANDED_RESIDUAL_PX:
        return None
    return labels, separation


def is_coherent(region: np.ndarray) -> bool:
    """Is this group one object, or scattered debris?

    A stranded object is contiguous: a few connected pieces at most, with the
    bulk of its area in them. Debris from unreliable correlations is spread over
    many small fragments, and splitting a bin along debris invents structure
    that then carries its own fitting error.
    """
    total = int(region.sum())
    if total == 0:
        return False
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        region.astype(np.uint8), connectivity=8
    )
    areas = np.sort(stats[1:, cv2.CC_STAT_AREA])[::-1]
    if areas.size == 0:
        return False
    biggest = areas[:MAX_PIECES_PER_GROUP].sum()
    return float(biggest) / total >= MIN_COHERENT_FRACTION


def split_region(grays, reference_index, mask, shifts):
    """Split one region into two if its tiles disagree about the correction.

    The feature is the tile's residual across EVERY frame, concatenated, so the
    grouping reflects a consistent motion signature rather than one frame's
    noise.
    """
    boxes = tile_grid(grays[0].shape, mask)
    if len(boxes) < 2 * MIN_TILES_PER_GROUP:
        return None

    columns = []
    for i, gray in enumerate(grays):
        if i == reference_index or shifts.get(i) is None:
            continue
        residuals = tile_residuals(grays[reference_index], gray, boxes, shifts[i])
        columns.append(residuals)
    if not columns:
        return None

    features, kept = [], []
    for t in range(len(boxes)):
        row = []
        ok = True
        for column in columns:
            if column[t] is None or column[t][2] < 0.05:
                ok = False
                break
            row.extend(column[t][:2])
        if ok:
            features.append(row)
            kept.append(boxes[t])
    if len(features) < 2 * MIN_TILES_PER_GROUP:
        return None

    result = two_means(features)
    if result is None:
        return None
    labels, separation = result

    # Tiles vote, but the region must not inherit the tile grid: a blocky
    # support puts staircase transitions into the sampling field, which is its
    # own artifact. The vote is snapped to image structure instead, so a region
    # ends where the object does.
    guide = grays[reference_index]
    parts = []
    for k in (0, 1):
        part = np.zeros(mask.shape, dtype=np.float32)
        for (x, y, box), label in zip(kept, labels):
            if label == k:
                part[box] = 1.0
        snapped = (guided_filter(guide, part, 12, 1e-4) > 0.5) & mask
        if not is_coherent(snapped):
            return None
        parts.append(snapped)
    return parts, separation


def merge_by_motion(grays, reference_index, regions, tolerance=1.0):
    """Rejoin regions whose motion agrees: they are one object.

    Splitting alone is unsafe. Subdividing a rigid surface gives each piece its
    own independent fit, and small regions fit more noisily, so one object ends
    up transported by slightly different amounts in different places — a
    discontinuity inside a surface that has none. Motion is the same evidence
    used to split, read the other way: pieces that move identically across the
    whole sweep belong together, however the split arrived at them.
    """
    signatures = []
    for region in regions:
        vector = []
        for i, gray in enumerate(grays):
            if i == reference_index:
                continue
            fitted = fit_translation(grays[reference_index], gray, region)
            vector.extend(fitted if fitted else (np.nan, np.nan))
        signatures.append(np.array(vector, dtype=np.float64))

    merged, used = [], set()
    for a in range(len(regions)):
        if a in used:
            continue
        group = regions[a].copy()
        used.add(a)
        for b in range(a + 1, len(regions)):
            if b in used:
                continue
            pair = np.stack([signatures[a], signatures[b]])
            valid = ~np.isnan(pair).any(axis=0)
            if valid.sum() < 4:
                continue
            difference = pair[0][valid] - pair[1][valid]
            if float(np.sqrt((difference ** 2).mean())) < tolerance:
                group |= regions[b]
                used.add(b)
        merged.append(group)
    return merged


def main() -> None:
    src = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(KITCHEN, "*.jpg")))]
    coarse = align_stack(src, depth_bins=0, crop_valid=False)
    grays = [to_gray_float(image).astype(np.float32) / 255.0 for image in coarse]
    reference_index = len(coarse) // 2

    depth = depth_from_focus(coarse)
    edges = align_mod._valley_edges(depth.ravel(), 4)
    bin_map = np.clip(np.digitize(depth, edges[1:-1]), 0, len(edges) - 2)
    regions = [bin_map == b for b in range(len(edges) - 1)]

    cx, cy = 612, 250
    bottle = np.zeros(depth.shape, bool)
    bottle[cy - 90:cy + 90, cx - 80:cx + 60] = True
    print("target: the bottle needs about +19.2 px on frame 11\n")

    for level in range(3):
        shifts_per_region = []
        for region in regions:
            shifts = {}
            for i, gray in enumerate(grays):
                if i == reference_index:
                    continue
                shifts[i] = fit_translation(grays[reference_index], gray, region)
            shifts_per_region.append(shifts)

        owner = max(range(len(regions)),
                    key=lambda r: (regions[r] & bottle).sum())
        share = 100 * regions[owner].mean()
        purity = 100 * float((regions[owner] & bottle).sum()) / max(1, regions[owner].sum())
        got = shifts_per_region[owner].get(11)
        print(f"level {level}: {len(regions)} regions | bottle's region covers "
              f"{share:4.1f}% of frame, bottle is {purity:4.1f}% of it, "
              f"frame-11 fit {('(%+.2f,%+.2f)' % got) if got else 'none'}")

        if level == 2:
            render(regions)
            break
        grown = []
        for region, shifts in zip(regions, shifts_per_region):
            outcome = split_region(grays, reference_index, region, shifts)
            if outcome is None:
                grown.append(region)
                continue
            parts, separation = outcome
            grown.extend(p for p in parts if p.sum() > 500)
        regions = grown


def render(regions) -> None:
    """Run the real pipeline with these regions in place of the depth bins."""
    src = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(KITCHEN, "*.jpg")))]
    shipped, shipped_report = align_stack(src, return_report=True)

    original = align_mod._depth_bin_masks
    align_mod._depth_bin_masks = lambda depth, valid, bins: [r & valid for r in regions]
    try:
        adaptive, adaptive_report = align_stack(src, return_report=True)
    finally:
        align_mod._depth_bin_masks = original

    sc, ac = shipped_report["crop"], adaptive_report["crop"]
    x0, y0 = max(sc[0], ac[0]), max(sc[1], ac[1])
    x1, y1 = min(sc[2], ac[2]), min(sc[3], ac[3])
    ss = (slice(y0 - sc[1], y1 - sc[1]), slice(x0 - sc[0], x1 - sc[0]))
    as_ = (slice(y0 - ac[1], y1 - ac[1]), slice(x0 - ac[0], x1 - ac[0]))

    shipped_src = [i[ss] for i in fio.normalize_exposure(shipped)]
    adaptive_src = [i[as_] for i in fio.normalize_exposure(adaptive)]
    panels = [
        fuse_perband(shipped_src, usable=[m[ss] for m in shipped_report["usable"]]),
        fuse_perband(adaptive_src, usable=[m[as_] for m in adaptive_report["usable"]]),
        adaptive_src[len(adaptive_src) // 2],
    ]
    cx, cy, r = 612, 250, 95
    box = (slice(cy - r, cy + r), slice(cx - r, cx + r))
    reference = panels[2][box].astype(np.float32)
    for label, image in zip(("shipped", "adaptive"), panels[:2]):
        print(f"  {label:9s} mean |diff| vs reference frame in bottle crop: "
              f"{np.abs(image[box].astype(np.float32) - reference).mean():.2f}")
    bar = np.full((2 * r, 4, 3), 255, np.uint8)
    tiles = []
    for image in panels:
        tiles += [image[box], bar]
    cv2.imwrite(f"{OUT}/ADAPTIVE_bottle_shipped_adaptive_ref.png",
                cv2.resize(np.hstack(tiles[:-1]), None, fx=2.4, fy=2.4,
                           interpolation=cv2.INTER_NEAREST))


if __name__ == "__main__":
    main()
