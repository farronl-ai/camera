"""Two instruments for the boundary problem, measured side by side (F84 work).

Neither is promoted. Both exist to produce comparable data on the two distinct
defects visible at the kitchen bottle:

1. VEILING. A defocused occluder spreads its own material outward over the
   background, and never pulls background inward (the ordered-attenuation
   invariant in STATE.md). That asymmetry gives a mask the parallax ribbon
   cannot: a direction, and a width equal to the occluder's defocus radius IN
   THAT FRAME, which is readable from how far the contour's own edge profile has
   spread. Frames where the occluder is sharp should veil nothing.

2. BIN HOMOGENEITY. A depth bin is a range, not an object. On the kitchen sweep
   one bin covers 55% of the frame; the bottle is 8.7% of it and needs +19.2 px
   while the bin is fitted to +2.3 px by everything else in it. This measures how
   much of each bin disagrees with its own fit, and what one residual-driven
   split would recover.

    .venv/bin/python research/boundary_probe.py
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
from focusstack.focus import content_aware_energies  # noqa: E402
from focusstack.fusion import depth_from_focus, fuse_perband  # noqa: E402
from focusstack.io import to_gray_float  # noqa: E402
import metrics  # noqa: E402
from occlusion_order import (_front_side_mask, _near_is_low_index,  # noqa: E402
                             _occlusion_contours)
from parallax_gen import build_stack  # noqa: E402

KITCHEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "data", "mobiledepth", "Figure3", "kitchen")
OUT = "/home/farron/camera/out/depth_align"


# --------------------------------------------------------------------------- #
# Instrument 1 — veiling footprint
# --------------------------------------------------------------------------- #

def front_focal_index(grays: list[np.ndarray], winner: np.ndarray):
    """For every pixel, the focal frame of the occluder nearest to it.

    F83 established that a contour's own focus reading names the OCCLUDER's
    depth, and that one global bit orders the two sides. That is all this needs:
    the front side's focal index, propagated outward from each contour.
    """
    contour, side_a, side_b, local = _occlusion_contours(grays, winner)
    near_is_low = _near_is_low_index(contour, side_a, side_b, local)
    if near_is_low is None or not contour.any():
        return None, contour, None

    front_index = np.minimum(side_a, side_b) if near_is_low else np.maximum(side_a, side_b)
    interior = (~contour).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(
        interior, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
    )
    lookup = np.zeros(int(labels.max()) + 1, dtype=np.float32)
    lookup[labels[contour]] = front_index[contour]
    front, _ = _front_side_mask(grays, winner, len(grays))
    return lookup[labels], contour, front


def veiling_masks(
    images: list[np.ndarray], pixels_per_step: float = 2.0
) -> tuple[list[np.ndarray], dict]:
    """Per frame, the background strip this frame's occluders have veiled.

    Two facts do the work, and neither needs a blur estimator — which is just as
    well, since contrast-over-gradient saturates by 2 px of blur and is swamped
    by texture on real frames.

    Direction: the foreground spreads outward over the background and never
    pulls background inward, so the footprint grows from the front region into
    the rear, never the reverse.

    Width: an occluder's defocus in frame k is set by how far k sits from the
    occluder's OWN focal frame, which the contour reading already reports. So a
    frame that focuses a given occluder veils nothing around it, while the same
    occluder eleven steps away veils a wide ribbon — per occluder, not per frame.
    """
    grays = [to_gray_float(image) for image in images]
    winner = np.argmax(np.stack(content_aware_energies(grays), axis=0), axis=0)
    focal, contour, front = front_focal_index(grays, winner)
    if focal is None:
        empty = [np.zeros(winner.shape, dtype=bool) for _ in images]
        return empty, {"ordering": None, "contour_pixels": int(contour.sum()), "radii": []}

    masks, radii = [], []
    for k in range(len(images)):
        radius_map = pixels_per_step * np.abs(float(k) - focal)
        radii.append(float(np.median(radius_map[contour])))
        mask = np.zeros(winner.shape, dtype=bool)
        for reach in (2, 4, 8, 16, 32):
            seed = (radius_map >= reach) & front
            if not seed.any():
                continue
            window = np.ones((2 * reach + 1, 2 * reach + 1), np.uint8)
            mask |= cv2.dilate(seed.astype(np.uint8), window) > 0
        masks.append(mask & ~front)
    return masks, {"radii": radii, "ordering": True,
                   "contour_pixels": int(contour.sum())}


# --------------------------------------------------------------------------- #
# Instrument 2 — bin homogeneity
# --------------------------------------------------------------------------- #

def bin_disagreement(
    reference: np.ndarray,
    moving: np.ndarray,
    bin_mask: np.ndarray,
    fitted: tuple[float, float],
    tile: int = 48,
) -> list[tuple[int, int, float, float, int]]:
    """Per-tile residual inside one bin, after the bin's own fit is applied.

    A homogeneous bin leaves near-zero residual everywhere. A bin that is really
    several objects leaves the minority ones stranded, and they show up here as
    tiles whose residual is large and consistent among themselves.
    """
    h, w = reference.shape
    shifted = cv2.warpAffine(
        moving,
        np.float32([[1, 0, fitted[0]], [0, 1, fitted[1]]]),
        (w, h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )
    window = cv2.createHanningWindow((tile, tile), cv2.CV_64F)
    rows = []
    for y in range(0, h - tile, tile):
        for x in range(0, w - tile, tile):
            box = (slice(y, y + tile), slice(x, x + tile))
            support = int(bin_mask[box].sum())
            if support < 0.6 * tile * tile:
                continue
            patch = reference[box]
            if float(patch.std()) < 0.02:
                continue
            (dx, dy), response = cv2.phaseCorrelate(
                np.ascontiguousarray(patch.astype(np.float64)),
                np.ascontiguousarray(shifted[box].astype(np.float64)),
                window,
            )
            rows.append((x + tile // 2, y + tile // 2, float(dx), float(dy), support))
    return rows


def split_gain(rows, fitted) -> tuple[float, float, str]:
    """Does this bin hold two populations wanting different corrections?

    Clustering the tile residual VECTORS matters: averaging signed residuals
    across a heterogeneous bin cancels them and reports nothing, which is what
    a median over the whole bin does.
    """
    if len(rows) < 6:
        return 0.0, 0.0, "too few tiles"
    vectors = np.array([[r[2], r[3]] for r in rows], dtype=np.float64)
    magnitude = np.hypot(vectors[:, 0], vectors[:, 1])
    split = float(np.median(magnitude))
    calm, stranded = vectors[magnitude <= split], vectors[magnitude > split]
    if len(stranded) == 0:
        return float(np.median(magnitude)), float(np.percentile(magnitude, 90)), "uniform"
    centre = stranded.mean(axis=0)
    detail = (f"{len(stranded)} tiles want ({centre[0]:+.1f},{centre[1]:+.1f}) "
              f"vs {len(calm)} near zero")
    return float(np.median(magnitude)), float(np.percentile(magnitude, 90)), detail


# --------------------------------------------------------------------------- #

def probe_factory() -> None:
    print("=== analytic factory (true GT) ===")
    frames, truth, _ = build_stack()
    aligned, report = align_stack(frames, depth_bins=4, return_report=True)
    x0, y0, x1, y1 = report["crop"]
    target = truth[y0:y1, x0:x1]
    ribbon = report["usable"]

    veil, info = veiling_masks(aligned)
    print(f"  contour {info['contour_pixels']} px, ordering {info['ordering']}")
    print("  per-frame edge spread radius: "
          + " ".join(f"{r:.1f}" for r in info["radii"]))

    combined = [r & ~v for r, v in zip(ribbon, veil)]
    for label, usable in (("no refusal", None),
                          ("parallax ribbon (shipped)", ribbon),
                          ("veiling only", [~v for v in veil]),
                          ("ribbon + veiling", combined)):
        withheld = 0.0 if usable is None else np.mean([1 - m.mean() for m in usable]) * 100
        print(f"  {label:26s} GT-SSIM "
              f"{metrics.ref_ssim(fuse_perband(aligned, usable=usable), target):.6f}  "
              f"withheld {withheld:5.2f}%")


def probe_kitchen() -> None:
    print("\n=== kitchen sweep (no GT; the bottle is the target) ===")
    paths = sorted(glob.glob(os.path.join(KITCHEN, "*.jpg")))
    src = [cv2.imread(p) for p in paths]
    coarse = align_stack(src, depth_bins=0, crop_valid=False)
    grays = [to_gray_float(i).astype(np.float32) / 255.0 for i in coarse]

    depth = depth_from_focus(coarse)
    edges = align_mod._valley_edges(depth.ravel(), 4)
    bin_map = np.clip(np.digitize(depth, edges[1:-1]), 0, len(edges) - 2)

    aligned, report = align_stack(src, return_report=True)
    cx, cy = 612, 250
    bottle = np.zeros(depth.shape, bool)
    bottle[cy - 90:cy + 90, cx - 80:cx + 60] = True

    print("  bin homogeneity for frame 11 (bottle needs about +19.2 px):")
    for b in range(len(edges) - 1):
        mask = bin_map == b
        share = 100 * mask.mean()
        fitted = report["frames"][11]["shifts"][b] if b < len(report["frames"][11]["shifts"]) else None
        if fitted is None:
            print(f"    bin {b}: {share:4.1f}% of frame, no accepted fit")
            continue
        rows = bin_disagreement(grays[6], grays[11], mask, fitted)
        median, p90, detail = split_gain(rows, fitted)
        holds = 100 * float((mask & bottle).sum()) / max(1, mask.sum())
        print(f"    bin {b}: {share:4.1f}% of frame  fit ({fitted[0]:+6.2f},{fitted[1]:+5.2f})  "
              f"tiles {len(rows):3d}  residual median {median:5.2f} p90 {p90:5.2f} px  "
              f"| bottle {holds:4.1f}% of bin | split: {detail}")

    veil, info = veiling_masks(aligned)
    print(f"  veiling: contour {info['contour_pixels']} px, ordering {info['ordering']}")
    print("    per-frame edge spread radius: "
          + " ".join(f"{r:.1f}" for r in info["radii"]))
    print("    veil coverage per frame: "
          + " ".join(f"{100 * m.mean():.1f}%" for m in veil))

    ribbon = report["usable"]
    combined = [r & ~v for r, v in zip(ribbon, veil)]
    sources = fio.normalize_exposure(aligned)
    panels = {
        "shipped": fuse_perband(sources, usable=ribbon),
        "ribbon+veil": fuse_perband(sources, usable=combined),
        "reference": sources[len(sources) // 2],
    }
    r = 95
    box = (slice(cy - r, cy + r), slice(cx - r, cx + r))
    bar = np.full((2 * r, 4, 3), 255, np.uint8)
    tiles = []
    for image in panels.values():
        tiles += [image[box], bar]
    cv2.imwrite(f"{OUT}/PROBE_bottle_shipped_veil_ref.png",
                cv2.resize(np.hstack(tiles[:-1]), None, fx=2.4, fy=2.4,
                           interpolation=cv2.INTER_NEAREST))
    reference = panels["reference"][box].astype(np.float32)
    for label in ("shipped", "ribbon+veil"):
        print(f"    {label:12s} mean |diff| vs reference frame in bottle crop: "
              f"{np.abs(panels[label][box].astype(np.float32) - reference).mean():.2f}")


if __name__ == "__main__":
    probe_factory()
    probe_kitchen()
