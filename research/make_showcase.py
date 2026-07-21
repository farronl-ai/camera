#!/usr/bin/env python3
"""Generate the figures for docs/SHOWCASE.md (the in-repo progress visual).

Each figure is guarded — if its dataset folder is missing the figure is skipped
and the script still completes (re-runnable on a fresh clone after regenerating
the research datasets).

Run:  python research/make_showcase.py
Writes docs/img/*.jpg (JPEG q85, capped width — repo-friendly sizes).
"""
from __future__ import annotations
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from focusstack.fusion import (fuse_blend, fuse_perband, fuse_pyramid,  # noqa: E402
                               _guided_weights, depth_from_focus)
from focusstack.focus import content_aware_energies  # noqa: E402
from focusstack.io import to_gray_float, normalize_exposure  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
IMG = os.path.join(REPO, "docs", "img")
os.makedirs(IMG, exist_ok=True)
DATA = os.path.join(HERE, "data")


def save(name, img, max_w=1400, q=85):
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, round(h * max_w / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(IMG, name), img, [cv2.IMWRITE_JPEG_QUALITY, q])
    print(f"  wrote {name}  ({img.shape[1]}x{img.shape[0]})")


def zoom(img, y0, y1, x0, x1, f=3):
    return cv2.resize(img[y0:y1, x0:x1], None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)


def norm_gray(m):
    m = m.astype(np.float32)
    lo, hi = float(m.min()), float(m.max())
    return ((m - lo) / (hi - lo + 1e-9) * 255).astype(np.uint8)


def gap(h, w=6):
    return np.full((h, w, 3), 255, np.uint8)


def hstack(*imgs):
    h = min(i.shape[0] for i in imgs)
    row = []
    for k, im in enumerate(imgs):
        if im.shape[0] != h:
            im = im[:h]
        row.append(im)
        if k < len(imgs) - 1:
            row.append(gap(h))
    return np.hstack(row)


def fig_fence():
    a = cv2.imread(os.path.join(DATA, "standard", "c_05_1.tif"))
    b = cv2.imread(os.path.join(DATA, "standard", "c_05_2.tif"))
    if a is None or b is None:
        return print("  skip fence (no data)")
    fused = fuse_perband([a, b], harden=0.5)
    save("hero_fence.jpg", hstack(a, b, fused), max_w=1560)

    # anatomy: focus energy | winner | guided weight (frame A) | fused
    e = content_aware_energies([to_gray_float(a), to_gray_float(b)])
    energy = cv2.applyColorMap(norm_gray(np.maximum(e[0], e[1])), cv2.COLORMAP_INFERNO)
    winner = cv2.applyColorMap(((np.argmax(np.stack(e), 0)) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    w = _guided_weights([a, b], "content_aware", None, 1e-3, None, harden=0.5)
    weight = cv2.cvtColor(norm_gray(w[0]), cv2.COLOR_GRAY2BGR)
    save("anatomy.jpg", hstack(energy, winner, weight, fused), max_w=1560)

    # halo zoom: pyramid | blend | perband
    py = fuse_pyramid([a, b])
    bl = fuse_blend([a, b], harden=0.5)
    y0, y1, x0, x1 = 170, 320, 60, 240
    save("zoom_halo.jpg", hstack(zoom(py, y0, y1, x0, x1), zoom(bl, y0, y1, x0, x1),
                                 zoom(fused, y0, y1, x0, x1)), max_w=1620)


def fig_micro():
    stacks = sorted(glob.glob(os.path.join(DATA, "bbbc006", "*", "")))
    if not stacks:
        return print("  skip micro (no data)")
    d = stacks[-1]  # a02_s2: dense field of nuclei (verified rich in texture)
    frames = [cv2.imread(p) for p in sorted(glob.glob(d + "frame_*.png"))]
    fused = fuse_perband(frames, harden=0.5)
    save("real_micro.jpg", hstack(frames[0], fused), max_w=1400)
    # nuclei crop: the SHARPEST fused region (where texture was recovered),
    # not the brightest (which can be an out-of-focus blob)
    g = to_gray_float(fused)
    dens = cv2.boxFilter(np.abs(cv2.Laplacian(g, cv2.CV_32F)), cv2.CV_32F, (120, 120))
    y, x = np.unravel_index(int(dens.argmax()), dens.shape)
    h = 130
    y0 = min(max(0, y - h), g.shape[0] - 2 * h)
    x0 = min(max(0, x - h), g.shape[1] - 2 * h)
    save("real_micro_zoom.jpg", hstack(zoom(frames[0], y0, y0 + 2 * h, x0, x0 + 2 * h),
                                       zoom(fused, y0, y0 + 2 * h, x0, x0 + 2 * h)), max_w=1580)


def fig_hires():
    d = os.path.join(DATA, "hires_mixed", "02_fine_detail")
    if not os.path.isdir(d):
        return print("  skip hires (no data)")
    frames = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(d, "frame_*.png")))]
    gt = cv2.imread(os.path.join(d, "gt.png"))
    depth = cv2.imread(os.path.join(d, "depth.png"), 0)
    bl = fuse_blend(frames, harden=0.5)
    pb = fuse_perband(frames, harden=0.5)
    near = (depth < 128).astype(np.float32)
    dens = cv2.boxFilter(near, cv2.CV_32F, (120, 120))
    y, x = np.unravel_index(int(dens.argmax()), dens.shape)
    h = 150
    y0, x0 = max(0, y - h), max(0, x - h)
    save("zoom_hires.jpg", hstack(zoom(gt, y0, y0 + 2 * h, x0, x0 + 2 * h),
                                  zoom(bl, y0, y0 + 2 * h, x0, x0 + 2 * h),
                                  zoom(pb, y0, y0 + 2 * h, x0, x0 + 2 * h)), max_w=1620)


def fig_harden():
    from spread import harsh_spread_scene
    from hardbench import defocus_disk, add_noise
    base, depth = harsh_spread_scene(0)
    frames = []
    for i, f in enumerate([0.1, 0.9]):
        frames.append(add_noise(defocus_disk(base, depth, f, 26.0), 3.0, i))
    off = fuse_perband(frames, harden=0.0)
    on = fuse_perband(frames, harden=0.5)
    err = cv2.boxFilter(np.abs(off.astype(np.int16) - base).sum(2).astype(np.float32), cv2.CV_32F, (31, 31))
    y, x = np.unravel_index(int(err.argmax()), err.shape)
    h = 110
    y0, x0 = max(0, y - h), max(0, x - h)
    save("harden.jpg", hstack(zoom(base, y0, y0 + 2 * h, x0, x0 + 2 * h),
                              zoom(off, y0, y0 + 2 * h, x0, x0 + 2 * h),
                              zoom(on, y0, y0 + 2 * h, x0, x0 + 2 * h)), max_w=1620)


def fig_drift_depth():
    from nframe import scenes, make_stack
    from drift import apply_drift
    name, gt, depth = scenes()[0]
    frames, _ = make_stack(gt, depth, 4)
    drifted = apply_drift(frames)
    strip = hstack(*[cv2.resize(f, (380, round(f.shape[0] * 380 / f.shape[1]))) for f in drifted])
    no = fuse_perband(drifted, harden=0.5)
    yes = fuse_perband(normalize_exposure(drifted), harden=0.5)
    big = 2 * 380 + 6
    row2 = hstack(cv2.resize(no, (big, round(no.shape[0] * big / no.shape[1]))),
                  cv2.resize(yes, (big, round(yes.shape[0] * big / yes.shape[1]))))
    w = min(strip.shape[1], row2.shape[1])
    save("drift.jpg", np.vstack([strip[:, :w], np.full((6, w, 3), 255, np.uint8), row2[:, :w]]),
         max_w=1560)

    frames8, _ = make_stack(gt, depth, 8)
    d = depth_from_focus(frames8)
    dvis = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    save("depth.jpg", hstack(gt, dvis), max_w=1400)


if __name__ == "__main__":
    print("generating showcase figures -> docs/img/")
    fig_fence()
    fig_micro()
    fig_hires()
    fig_harden()
    fig_drift_depth()
    print("done")
