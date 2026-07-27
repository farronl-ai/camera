"""Diagnostics for the per-region two-frame prototype.

Small, single-question probes (DEVSTYLE §13: the smallest experiment that can
falsify the claim is also the cheapest). Each function answers one thing about
the kitchen sweep and prints a table.

    .venv/bin/python research/twoframe_probe.py
"""
from __future__ import annotations

import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import focusstack.align as A  # noqa: E402
from focusstack.io import to_gray_float  # noqa: E402
import twoframe as TF  # noqa: E402

BOTTLE = (128, 392, 498, 639)      # y0, y1, x0, x1 — group_align's box
STREAK = (230, 400, 600, 700)      # F108's acceptance box


def main() -> None:
    src = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(TF.KITCHEN, "*.jpg")))]
    ref = len(src) // 2
    coarse, warps, valid = TF.global_stage(src, ref)
    common = np.logical_and.reduce(valid)
    peak, contrast, energies = TF.focal_field(coarse)

    print("1. Where is each region sharpest? (summed focus energy per frame)")
    for label, box in (("bottle", BOTTLE), ("streak box", STREAK),
                       ("wall right of bottle", (230, 400, 680, 760))):
        y0, y1, x0, x1 = box
        totals = energies[:, y0:y1, x0:x1].reshape(len(energies), -1).sum(axis=1)
        print(f"  {label:<22} argmax frame {int(np.argmax(totals)):2d}   "
              f"profile " + " ".join(f"{t / totals.max():.2f}" for t in totals))

    print("\n2. Focal-peak composition of the streak box")
    y0, y1, x0, x1 = STREAK
    values = peak[y0:y1, x0:x1].ravel()
    weights = contrast[y0:y1, x0:x1].ravel()
    threshold, quality = TF._otsu_split(values, weights, len(src))
    print(f"  Otsu threshold {threshold} quality {quality:.3f} "
          f"(tile gate is {TF.OTSU_MIN_QUALITY})")
    histogram, _ = np.histogram(values, bins=np.arange(-0.5, len(src) + 0.5),
                                weights=weights)
    print("  weighted peak histogram: "
          + " ".join(f"{i}:{v / histogram.sum() * 100:4.1f}%"
                     for i, v in enumerate(histogram)))

    print("\n3. Which pair does each tile touching the bottle choose?")
    tiles = TF.tile_pairs(peak, contrast, energies, common)
    for tile in tiles:
        ty0, ty1, tx0, tx1 = tile["box"]
        if ty1 <= BOTTLE[0] or ty0 >= BOTTLE[1] or tx1 <= BOTTLE[2] or tx0 >= BOTTLE[3]:
            continue
        print(f"  tile y{ty0:4d} x{tx0:4d}  pair {str(tile['pair']):>8} "
              f"quality {tile['quality']:.2f}")

    print("\n4. Can the estimator find the bottle's motion when handed the bottle?")
    ref_gray = to_gray_float(coarse[ref]).astype(np.float32) / 255.0
    grays = [to_gray_float(c).astype(np.float32) / 255.0 for c in coarse]
    gradient = cv2.magnitude(
        cv2.Sobel(ref_gray * 255.0, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(ref_gray * 255.0, cv2.CV_32F, 0, 1, ksize=3))
    box_mask = np.zeros(ref_gray.shape, bool)
    box_mask[BOTTLE[0]:BOTTLE[1], BOTTLE[2]:BOTTLE[3]] = True
    print(f"  {'frame':>5} {'bottle-box ECC dx':>18} {'truth (F89/F100)':>18}")
    truth = {8: -9.1, 9: -12.5, 10: -16.2, 11: -20.0}
    for frame in (7, 8, 9, 10, 11):
        warp = TF.masked_translation(ref_gray, grays[frame],
                                     box_mask & (gradient >= A._REFINE_MIN_GRADIENT), 60.0)
        got = "refused" if warp is None else f"{float(warp[0, 2]):18.2f}"
        print(f"  {frame:5d} {got:>18} {truth.get(frame, float('nan')):18.1f}")

    print("\n5. What does the pair's own layer mask contain at the bottle?")
    kept = TF.merge_pairs(tiles)
    owner, _weights = TF.ownership(tiles, kept, ref_gray.shape)
    for index, pair in enumerate(kept):
        owned = owner == index
        inside = owned & box_mask
        if inside.mean() * ref_gray.size < 500:
            continue
        support = cv2.dilate(owned.astype(np.uint8),
                             np.ones((TF.TILE, TF.TILE), np.uint8)) > 0
        masks, _dense = TF.layer_masks(energies, pair, support & common, gradient)
        share = 100.0 * inside.sum() / max(1, box_mask.sum())
        print(f"  pair {str(pair):>8} owns {share:5.1f}% of the bottle box")
        for frame, mask in zip((pair if pair[0] != pair[1] else (pair[0],)), masks):
            in_box = 100.0 * (mask & box_mask).sum() / max(1, mask.sum())
            print(f"      frame {frame:2d} layer: {int(mask.sum()):7d} px, "
                  f"{in_box:5.1f}% of them inside the bottle box")


def streak_provenance() -> None:
    """Is the two-frame streak zone actually reference-sourced, and is the
    bright band right of the bottle real scene or an artifact?

    Answered by looking at the SOURCE frames in the same window: a band that is
    present in the frame the region is composited from is de-veiling (the
    reference's defocused dark object was spreading over it); a band present in
    neither source is manufactured.
    """
    src = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(TF.KITCHEN, "*.jpg")))]
    ref = len(src) // 2
    coarse, _warps, _valid = TF.global_stage(src, ref)
    fused, info = TF.twoframe_stack(src, ref)
    tx0, ty0, _tx1, _ty1 = info["crop"]

    y0, y1, x0, x1 = STREAK
    panels = [TF._stamp(coarse[k], STREAK) for k in (6, 8, 10, 11)]
    panels.append(TF._stamp(fused, (y0 - ty0, y1 - ty0, x0 - tx0, x1 - tx0)))
    cv2.imwrite(os.path.join(TF.OUT, "TF_streak_sources.png"), np.hstack(panels))
    print("wrote TF_streak_sources.png  columns: frame 6 | 8 | 10 | 11 | two-frame")

    grey = to_gray_float(src[ref]).astype(np.float32)
    box = grey[y0:y1, x0:x1]
    mean = cv2.boxFilter(box, cv2.CV_32F, (9, 9))
    std = np.sqrt(np.maximum(cv2.boxFilter(box * box, cv2.CV_32F, (9, 9)) - mean * mean, 0))
    flank = (box > 170) & (std < 4.0)
    print(f"\nlow-contrast white flank inside the streak box: {int(flank.sum())} px")

    print("\nprovenance in the bottle box: which source does each pair's fusion use?")
    _f, detail = TF.twoframe_stack(src, ref, report=True)
    box = np.zeros(detail["owner"].shape, bool)
    box[BOTTLE[0]:BOTTLE[1], BOTTLE[2]:BOTTLE[3]] = True
    for diagnostic in detail["diagnostics"]:
        here = diagnostic["owned"] & box
        if here.sum() < 500:
            continue
        print(f"  pair {str(diagnostic['pair']):>8} covers {int(here.sum()):6d} "
              f"bottle-box px")
        for frame, render in zip(
                (diagnostic["pair"] if diagnostic["pair"][0] != diagnostic["pair"][1]
                 else (diagnostic["pair"][0],)), diagnostic["rendered"]):
            delta = np.abs(diagnostic["fused"].astype(np.float32)
                           - render.astype(np.float32)).max(axis=2)
            print(f"      frame {frame:2d}: median |fused − this source| here "
                  f"{np.median(delta[here]):5.1f}")

    from focusstack.align import align_stack
    from focusstack.fusion import fuse_perband
    shipped_aligned, report = align_stack(src, return_report=True)
    shipped = fuse_perband(shipped_aligned, usable=report["usable"])
    sx0, sy0, _a, _b = report["crop"]
    reference = src[ref][y0:y1, x0:x1].astype(np.float32)
    print(f"  {'output':<12} {'mean |Δ| on flank':>18} {'max |Δ|':>9} "
          f"{'px over 12':>11}")
    for label, image, (ox, oy) in (("shipped", shipped, (sx0, sy0)),
                                   ("two-frame", fused, (tx0, ty0))):
        crop = image[y0 - oy:y1 - oy, x0 - ox:x1 - ox].astype(np.float32)
        delta = np.abs(crop - reference).max(axis=2)
        print(f"  {label:<12} {delta[flank].mean():18.2f} {delta[flank].max():9.0f} "
              f"{100 * (delta[flank] > 12).mean():10.2f}%")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "streak":
        streak_provenance()
    else:
        main()
