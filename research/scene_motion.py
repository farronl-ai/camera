"""Joint scene motion: objects, their depths, and camera motion together (F96).

This joins the two halves the arc arrived at separately.

F93 defines an object as a maximal feature set admitting one rigid motion, which
keeps a bottle whole where tile clustering fragmented it. F95 makes inverse depth an
explicit parameter, shared across frames because it belongs to the scene, and gets
~69% of observed displacement — its residual blamed on depth quantized into four
bins and on motion varying WITHIN a bin.

Both complaints have the same answer: let each OBJECT be its own depth.

    d_i,k . n_i = omega_k (rot) + rho_o(i) [ (-tx_k + ux tz_k) nx + (-ty_k + uy tz_k) ny ]

Unknowns are four camera parameters per frame, one inverse depth per object, and the
assignment of features to objects. Each is linear or trivial given the others, so it
alternates: motion given depth, depth given motion, then re-assign features to
whichever object's depth explains them — the assignment step is now a physical test
rather than a clustering heuristic.

Gauge and identifiability, carried from F94/F95: a pan is indistinguishable from
every depth translating equally, so rho is fixed only up to an affine
reparameterization. Only its depth-VARYING part is meaningful, and monotonicity is
not a validity check.

    .venv/bin/python research/scene_motion.py
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
import motion_components as MC  # noqa: E402
import object_segmentation as OS  # noqa: E402

KITCHEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "data", "mobiledepth", "Figure3", "kitchen")
OUTER = 6          # outer alternations
MIN_MEMBERS = 8    # an object needs enough features to own a depth


def gather(grays, ref, features):
    """Every (frame, feature) observation with its confidence."""
    rows = []
    for i, (x, y, nx, ny) in enumerate(features):
        base = MC.ES._profile(grays[ref], x, y, nx, ny)
        for k, gray in enumerate(grays):
            if k == ref:
                continue
            shift, peak = MC.ES._match(base, MC.ES._profile(gray, x, y, nx, ny))
            if peak >= 0.5 and abs(shift) < 40:
                rows.append((k, i, float(shift), float(peak)))
    return rows


def solve_motion(rows, features, labels, rho, shape, frames):
    """Camera rotation and translation per frame, given each object's depth."""
    h, w = shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    motion = {}
    for k in frames:
        A, b, wt = [], [], []
        for (kk, i, shift, peak) in rows:
            if kk != k:
                continue
            x, y, nx, ny = features[i]
            ux, uy = x - cx, y - cy
            r = rho[labels[i]]
            A.append([-uy * nx + ux * ny, -r * nx, -r * ny, r * (ux * nx + uy * ny)])
            b.append(shift); wt.append(peak)
        if len(A) < 8:
            continue
        A = np.asarray(A); b = np.asarray(b)
        root = np.sqrt(np.asarray(wt))[:, None]
        motion[k], *_ = np.linalg.lstsq(A * root, b * root.ravel(), rcond=None)
    return motion


def solve_depth(rows, features, labels, motion, shape, objects):
    """One inverse depth per object, pooled over every frame it appears in."""
    h, w = shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    num = np.zeros(objects); den = np.zeros(objects)
    for (k, i, shift, peak) in rows:
        if k not in motion:
            continue
        omega, tx, ty, tz = motion[k]
        x, y, nx, ny = features[i]
        ux, uy = x - cx, y - cy
        rot = omega * (-uy * nx + ux * ny)
        coef = -tx * nx - ty * ny + tz * (ux * nx + uy * ny)
        o = labels[i]
        num[o] += peak * coef * (shift - rot)
        den[o] += peak * coef * coef
    return num, den


def reassign(rows, features, labels, rho, motion, shape, objects):
    """Give each feature to the object whose depth best explains its motion."""
    h, w = shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    cost = np.zeros((len(features), objects))
    seen = np.zeros(len(features))
    for (k, i, shift, peak) in rows:
        if k not in motion:
            continue
        omega, tx, ty, tz = motion[k]
        x, y, nx, ny = features[i]
        ux, uy = x - cx, y - cy
        rot = omega * (-uy * nx + ux * ny)
        coef = -tx * nx - ty * ny + tz * (ux * nx + uy * ny)
        for o in range(objects):
            cost[i, o] += peak * (rot + rho[o] * coef - shift) ** 2
        seen[i] += peak
    new = labels.copy()
    for i in range(len(features)):
        if seen[i] > 0:
            new[i] = int(np.argmin(cost[i]))
    return new


def residual(rows, features, labels, rho, motion, shape):
    h, w = shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    errs = []
    for (k, i, shift, peak) in rows:
        if k not in motion:
            continue
        omega, tx, ty, tz = motion[k]
        x, y, nx, ny = features[i]
        ux, uy = x - cx, y - cy
        pred = (omega * (-uy * nx + ux * ny)
                + rho[labels[i]] * (-tx * nx - ty * ny + tz * (ux * nx + uy * ny)))
        errs.append(pred - shift)
    return float(np.sqrt(np.mean(np.square(errs)))), len(errs)


def run(stack, label, box=None):
    coarse = align_stack(stack, depth_bins=0, crop_valid=False)
    grays = [to_gray_float(i).astype(np.float32) / 255.0 for i in coarse]
    ref = len(coarse) // 2
    shape = grays[0].shape
    depth = depth_from_focus(coarse)
    features = OS.material_features(grays, ref, depth)
    rows = gather(grays, ref, features)
    frames = sorted({k for k, _, _, _ in rows})
    print(f"\n{label}: {len(features)} material edges, {len(rows)} observations, "
          f"{len(frames)} frames")
    if len(rows) < 200:
        print("  too few observations for a joint fit")
        return

    # Seed the objects from F93's motion consensus, then let the physical model
    # take over the assignment.
    table = OS.observe(grays, ref, features)
    seeds, _ = OS.segment(features, table, shape, len(grays))
    labels = np.zeros(len(features), int)
    for o, (members, _) in enumerate(seeds):
        for i in members:
            labels[i] = o
    objects = max(1, len(seeds))
    rho = np.linspace(1.0, 0.2, objects) if objects > 1 else np.array([1.0])

    print(f"  seeded with {objects} objects")
    for it in range(OUTER):
        motion = solve_motion(rows, features, labels, rho, shape, frames)
        if not motion:
            break
        num, den = solve_depth(rows, features, labels, motion, shape, objects)
        rho = np.where(den > 1e-9, num / np.maximum(den, 1e-9), rho)
        if np.ptp(rho) > 1e-9:
            rho = (rho - rho.mean()) / np.abs(rho - rho.mean()).mean()   # gauge
        labels = reassign(rows, features, labels, rho, motion, shape, objects)
        counts = np.bincount(labels, minlength=objects)
        for o in range(objects):
            if counts[o] < MIN_MEMBERS:
                rho[o] = rho[counts.argmax()]
        rms, n = residual(rows, features, labels, rho, motion, shape)
        print(f"   iter {it}: rms {rms:5.2f} px   sizes {list(counts)}")

    if box is not None:
        inside = [i for i in range(len(features))
                  if box[0] <= features[i][0] <= box[1] and box[2] <= features[i][1] <= box[3]]
        if inside:
            owner = int(np.bincount(labels[inside], minlength=objects).argmax())
            purity = 100 * float(np.mean(labels[inside] == owner))
            share = 100 * float(np.mean(labels == owner))
            print(f"  target object: {purity:.0f}% of its features agree, "
                  f"it holds {share:.0f}% of all features, rho {rho[owner]:+.2f}")
            h, w = shape
            cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
            for k in (8, 11):
                if k not in motion:
                    continue
                omega, tx, ty, tz = motion[k]
                x, y = 568.0, 260.0     # centre of the target
                ux, uy = x - cx, y - cy
                dx = omega * (-uy) + rho[owner] * (-tx + ux * tz)
                print(f"    predicted x-shift at frame {k}: {dx:+.2f} px")


if __name__ == "__main__":
    src = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(KITCHEN, "*.jpg")))]
    run(src, "kitchen sweep", box=(498, 639, 128, 392))
    print("\n  truth: the bottle needs about +7.4 px at frame 8 and +19.2 px at frame 11")
