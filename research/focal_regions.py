"""Region masks from the per-feature focal signature (F98 work).

The shipped alignment cuts depth bins from the guided depth map at histogram
valleys. That map is smoothed, it cannot place a contour (F83), and on the kitchen
sweep it produces a bin covering 55% of the frame that fits its majority and leaves
the bottle 19 px out of register (F84).

F97 gives a better basis. Each material feature's own sharpness curve across the
sweep peaks at its depth, recovering known focal frames to within 0.1 frame with a
within-group spread of 0.3–0.6 against 2–3 frames of separation — and it does that on
both scenes with identical settings, where no motion threshold does. So build the
regions from clustered focal signatures and propagate them to pixels, rather than
slicing a smoothed depth map.

    .venv/bin/python research/focal_regions.py
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
import metrics  # noqa: E402
import object_segmentation as OS  # noqa: E402
import parallax_gen as P  # noqa: E402

KITCHEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "data", "mobiledepth", "Figure3", "kitchen")
FOCAL_HALF = 9
SEPARATION = 1.2      # frames; groups separate by 2-3, spread within is 0.3-0.6
MIN_FEATURES = 5


def focal_signature(grays, features, half=FOCAL_HALF):
    """Each feature's focal frame, plus how decisive its peak is."""
    out = []
    for (x, y, nx, ny) in features:
        xi, yi = int(round(x)), int(round(y))
        energy = []
        for gray in grays:
            patch = gray[yi - half:yi + half + 1, xi - half:xi + half + 1]
            energy.append(float(np.abs(cv2.Laplacian(patch, cv2.CV_32F)).mean()))
        e = np.asarray(energy)
        k = int(np.argmax(e))
        off = 0.0
        if 0 < k < len(e) - 1:
            d = e[k - 1] - 2 * e[k] + e[k + 1]
            off = 0.5 * (e[k - 1] - e[k + 1]) / d if abs(d) > 1e-9 else 0.0
        out.append((k + float(np.clip(off, -1, 1)), float(e.max() / (e.mean() + 1e-9))))
    return out


def _otsu_split(values):
    """Best bimodal split of a 1-D set, with the separation quality it achieves.

    Returns (threshold, quality) where quality is between-class variance over total
    variance — 1.0 for two tight well-separated clumps, near 0 for one continuum.
    """
    ordered = np.sort(values)
    if len(ordered) < 4:
        return None, 0.0
    total = float(np.var(ordered))
    if total < 1e-9:
        return None, 0.0
    best, best_quality = None, 0.0
    for i in range(2, len(ordered) - 2):
        lo, hi = ordered[:i], ordered[i:]
        w = len(lo) / len(ordered)
        between = w * (1 - w) * (lo.mean() - hi.mean()) ** 2
        if between > best_quality:
            best_quality = between
            best = 0.5 * (ordered[i - 1] + ordered[i])
    return best, best_quality / total


def cluster_focal(signature, min_quality=0.55, max_groups=5):
    """Recursively split the focal axis wherever it is genuinely bimodal.

    Gap-based linkage fails here: the focal frames are bimodal but their tails fill
    the gap, so no single consecutive difference is large even when the modes sit
    three frames apart. Otsu asks the right question instead — is this set better
    described as two clumps than as one — and answers it with no tuned distance.
    """
    peaks = np.array([p for p, _ in signature], float)
    labels = np.zeros(len(peaks), int)
    for _ in range(max_groups - 1):
        best = None
        for g in range(labels.max() + 1):
            members = np.nonzero(labels == g)[0]
            if len(members) < 2 * MIN_FEATURES:
                continue
            threshold, quality = _otsu_split(peaks[members])
            if threshold is not None and quality >= min_quality:
                if best is None or quality > best[2]:
                    best = (g, threshold, quality)
        if best is None:
            break
        g, threshold, _ = best
        new_label = labels.max() + 1
        members = np.nonzero(labels == g)[0]
        labels[members[peaks[members] > threshold]] = new_label
    return labels


def regions_from_features(features, labels, shape, guide):
    """Grow each focal group into a pixel region, snapped to image structure."""
    h, w = shape
    groups = [g for g in range(labels.max() + 1)
              if int(np.sum(labels == g)) >= MIN_FEATURES]
    if len(groups) < 2:
        return None
    stack = []
    for g in groups:
        seed = np.zeros((h, w), np.float32)
        for i, (x, y, nx, ny) in enumerate(features):
            if labels[i] == g:
                cv2.circle(seed, (int(round(x)), int(round(y))), 9, 1.0, -1)
        # A wide guided filter carries the label across untextured interior while
        # stopping at image structure, which is where a depth change would show.
        stack.append(guided_filter(guide, seed, 28, 1e-3))
    stack = np.stack(stack, 0)
    winner = np.argmax(stack, axis=0)
    return [winner == j for j in range(len(groups))]


def evaluate(frames, label, truth=None, box=None, near=None):
    coarse = align_stack(frames, depth_bins=0, crop_valid=False)
    grays = [to_gray_float(i).astype(np.float32) / 255.0 for i in coarse]
    ref = len(coarse) // 2
    shape = grays[0].shape
    depth = depth_from_focus(coarse)
    features = OS.material_features(grays, ref, depth)
    signature = focal_signature(grays, features)
    labels = cluster_focal(signature)
    regions = regions_from_features(features, labels, shape, grays[ref])
    print(f"\n{label}: {len(features)} features -> {labels.max() + 1} focal groups"
          f" -> {0 if regions is None else len(regions)} regions")
    if regions is None:
        return
    print("  region sizes (% of frame): "
          + " ".join(f"{100 * r.mean():.1f}" for r in regions))

    target = near if near is not None else None
    if box is not None:
        target = np.zeros(shape, bool)
        target[box[2]:box[3], box[0]:box[1]] = True
    if target is not None:
        best = max(regions, key=lambda r: (r & target).sum())
        inter = float((best & target).sum())
        print(f"  best region vs target: IoU {100 * inter / max(1, (best | target).sum()):.1f}%"
              f"  covers {100 * inter / max(1, target.sum()):.1f}%"
              f"  purity {100 * inter / max(1, best.sum()):.1f}%")

    original = align_mod._depth_bin_masks
    align_mod._depth_bin_masks = lambda d, v, b: [r & v for r in regions]
    try:
        aligned, report = align_stack(frames, return_report=True)
    finally:
        align_mod._depth_bin_masks = original
    shipped, shipped_report = align_stack(frames, return_report=True)
    if truth is not None:
        for name, (al, rep) in (("shipped depth bins", (shipped, shipped_report)),
                                ("focal regions", (aligned, report))):
            x0, y0, x1, y1 = rep["crop"]
            print(f"  {name:20s} GT-SSIM "
                  f"{metrics.ref_ssim(fuse_perband(al, usable=rep['usable']), truth[y0:y1, x0:x1]):.6f}"
                  f"   regions {rep['bins']}")
    if box is not None:
        shifts = report["frames"].get(11, {}).get("shifts")
        if shifts:
            print("  frame-11 per-region x-shift: "
                  + " ".join("none" if s is None else f"{s[0]:+.1f}" for s in shifts)
                  + "   (the bottle needs about +19.2)")


if __name__ == "__main__":
    P.BREATHING_PER_FRAME = 0.0
    P.NEAR_SHIFT_PER_FRAME, P.FAR_SHIFT_PER_FRAME = 3.2, 0.7
    frames, truth, near = P.build_stack()
    evaluate(frames, "factory (two planes)", truth=truth, near=near)

    src = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(KITCHEN, "*.jpg")))]
    evaluate(src, "kitchen sweep", box=(498, 639, 128, 392))
