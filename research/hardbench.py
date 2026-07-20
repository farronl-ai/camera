#!/usr/bin/env python3
"""Hard, realistic benchmark with ground truth — where methods actually differ.

Clean Gaussian-defocus data is near-ceiling. Real lens defocus is a DISK (circle
of confusion), which produces 'defocus spread': a bright near object, when out of
focus, bleeds as a large bright disk over the in-focus background. Fusion that
naively trusts local sharpness/energy pulls that spread in -> visible artifacts.
We also add thin bright 'wires/hairs' (the structures that break region
partitioning) and sensor noise, at higher resolution.

Two scenes (GT = the all-sharp base):
  defocus_spread — fine-texture far background + thin bright near wires/dots.
                   In the far-focused frame the near wires spread into disks.
  gradient_metal — smooth curved surface, continuous depth -> gradual focus.

Evaluates baseline / adaptive / per-operator with GLOBAL and TILED-LOCAL GT-SSIM
(local catches visually-important small-region wins the global mean hides).

Run:  python research/hardbench.py
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from regions import fuse_adaptive  # noqa: E402
from focusstack.fusion import fuse_blend  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "hard_out")
os.makedirs(OUT, exist_ok=True)
SZ = 1024


def disk_blur(img, radius):
    """Disk (circle-of-confusion) blur. Exact kernel for small radius; for large
    radius (high-res), approximate the disk's wide support with a downscale ->
    box-blur -> upscale, so cost stays ~O(pixels) instead of O(pixels*radius^2)."""
    if radius < 0.6:
        return img.copy()
    if radius <= 12:
        r = int(np.ceil(radius))
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        k = ((xx * xx + yy * yy) <= radius * radius).astype(np.float32)
        k /= k.sum()
        return cv2.filter2D(img, -1, k)
    # large radius: downscale so the disk becomes small, box-blur, upscale
    f = max(1, int(radius / 6))
    small = cv2.resize(img, (img.shape[1] // f, img.shape[0] // f), interpolation=cv2.INTER_AREA)
    kr = max(1, int(round(radius / f)))
    small = cv2.blur(small, (2 * kr + 1, 2 * kr + 1))
    return cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)


def defocus_disk(base, depth, focus, max_radius, n_levels=12):
    rad = max_radius * np.abs(depth - focus)
    levels = np.linspace(0.0, max_radius, n_levels)
    blurred = [disk_blur(base, s) for s in levels]
    idx = np.clip(np.round(rad / (max_radius + 1e-9) * (n_levels - 1)).astype(int), 0, n_levels - 1)
    out = np.empty_like(base)
    for l in range(n_levels):
        m = idx == l
        out[m] = blurred[l][m]
    return out


def scene_defocus_spread(seed=0):
    r = np.random.default_rng(seed)
    # far background: fine multicolor texture
    bg = r.integers(0, 256, (SZ, SZ, 3), np.uint8)
    bg = cv2.GaussianBlur(bg, (3, 3), 0.7)
    base = bg.copy()
    depth = np.full((SZ, SZ), 0.9, np.float32)  # background far
    # near foreground: thin bright wires + bright dots (defocus-spread sources)
    for _ in range(7):
        p1 = (int(r.integers(SZ)), int(r.integers(SZ)))
        p2 = (int(r.integers(SZ)), int(r.integers(SZ)))
        cv2.line(base, p1, p2, (255, 255, 255), 2, cv2.LINE_AA)
    dot_mask = np.zeros((SZ, SZ), np.uint8)
    for _ in range(40):
        c = (int(r.integers(SZ)), int(r.integers(SZ)))
        cv2.circle(base, c, int(r.integers(2, 5)), (255, 255, 255), -1)
        cv2.circle(dot_mask, c, 6, 255, -1)
    # mark near structures in depth map (wires+dots at near plane 0.1)
    near = np.zeros((SZ, SZ), np.uint8)
    # recompute near mask: where base differs strongly from bg (the drawn structures)
    diff = np.abs(base.astype(np.int16) - bg.astype(np.int16)).max(2)
    near[diff > 60] = 255
    depth[near > 0] = 0.1
    return base, depth


def scene_gradient_metal(seed=1):
    r = np.random.default_rng(seed)
    x = np.linspace(0, np.pi, SZ)
    shade = 0.5 + 0.45 * np.sin(x)[None, :] * np.ones((SZ, 1))
    base = (shade[..., None] * np.array([175, 185, 195])).astype(np.uint8)
    tex = cv2.GaussianBlur(r.normal(0, 7, (SZ, SZ)).astype(np.float32), (1, 41), 0)
    base = np.clip(base + tex[..., None], 0, 255).astype(np.uint8)
    for _ in range(10):
        p1 = (int(r.integers(SZ)), int(r.integers(SZ)))
        p2 = (p1[0] + int(r.integers(-200, 200)), p1[1] + int(r.integers(-60, 60)))
        cv2.line(base, p1, p2, (140, 140, 150), 1, cv2.LINE_AA)
    depth = (np.linspace(0, 1, SZ)[None, :] * np.ones((SZ, 1))).astype(np.float32)
    return base, depth


SCENES = {"defocus_spread": (scene_defocus_spread, 2, 14.0),
          "gradient_metal": (scene_gradient_metal, 5, 8.0)}


def add_noise(img, sigma, seed):
    r = np.random.default_rng(seed)
    return np.clip(img.astype(np.float32) + r.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def tiled_ssim(fused, gt, tiles=8):
    """Per-tile GT-SSIM -> (mean, worst-tile) to catch local failures."""
    from metrics import _ssim_map, _gray32
    s = _ssim_map(_gray32(fused), _gray32(gt))
    h, w = s.shape
    th, tw = h // tiles, w // tiles
    vals = [s[i * th:(i + 1) * th, j * tw:(j + 1) * tw].mean()
            for i in range(tiles) for j in range(tiles)]
    return float(np.mean(vals)), float(np.min(vals))


def main():
    global SZ
    if len(sys.argv) > 1:
        SZ = int(sys.argv[1])
    print(f"resolution SZ={SZ}")
    OPS = ["laplacian", "gradient", "tenengrad", "mod_laplacian"]
    result = {}
    for name, (gen, nframes, maxrad) in SCENES.items():
        base, depth = gen()
        gt = base
        planes = np.linspace(float(depth.min()), float(depth.max()), nframes)
        frames = [add_noise(defocus_disk(base, depth, f, maxrad), 3.0, i) for i, f in enumerate(planes)]

        methods = {}
        methods["baseline"] = fuse_blend(frames)
        methods["adaptive"] = fuse_adaptive(frames)
        for op in OPS:
            methods[f"op_{op}"] = fuse_blend(frames, focus_method=op)

        rows = {}
        for mname, fused in methods.items():
            g = M.ref_ssim(fused, gt)
            mean_t, worst_t = tiled_ssim(fused, gt)
            rows[mname] = {"gt_ssim": round(g, 4), "tile_mean": round(mean_t, 4),
                           "tile_worst": round(worst_t, 4), "composite": round(M.composite(frames, fused), 4)}
        result[name] = rows

        # montages
        cv2.imwrite(os.path.join(OUT, f"{name}_frames.png"),
                    np.hstack([gt] + frames)[:, :SZ * 3])
        cv2.imwrite(os.path.join(OUT, f"{name}_base_adapt.png"),
                    np.hstack([gt, methods["baseline"], methods["adaptive"]]))
        print(f"\n=== {name} (nframes={nframes}, max CoC radius={maxrad}px) ===")
        print(f"{'method':14s} {'GT-SSIM':>8s} {'tileμ':>7s} {'tile_worst':>11s} {'composite':>10s}")
        for mname, rr in sorted(rows.items(), key=lambda t: -t[1]["gt_ssim"]):
            print(f"{mname:14s} {rr['gt_ssim']:8.4f} {rr['tile_mean']:7.4f} {rr['tile_worst']:11.4f} {rr['composite']:10.4f}")

    json.dump(result, open(os.path.join(HERE, "hardbench_result.json"), "w"), indent=2)
    print(f"\nwrote montages to {OUT}, result to hardbench_result.json")


if __name__ == "__main__":
    main()
