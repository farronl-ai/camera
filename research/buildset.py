#!/usr/bin/env python3
"""M3 groundwork — build a per-tile content->tune labeled dataset.

Generate diverse hard scenes (collages of different content archetypes placed at
varied depths, with realistic disk defocus + noise + thin wires) that have GROUND
TRUTH. Tile each scene; per tile record:
  X       — content features computed from the INPUT frames only (deployable)
  y_gt    — index of the tune that maximizes local GT-SSIM (dev supervision)
  y_comp  — index of the tune that maximizes local composite (no-GT labeling)

The GT-vs-composite label agreement tells us how close the no-answer-key ideal
is. y_gt trains M3's feature->tune model; at inference only X is used (no GT).

Run:  python research/buildset.py [n_scenes] [tile]
Writes research/trainset.npz + prints label stats & GT/composite agreement.
"""
from __future__ import annotations
import concurrent.futures as cf
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from metrics import _ssim_map, _gray32  # noqa: E402
from focusstack.fusion import fuse_blend  # noqa: E402
from synth import arch_fine_detail, arch_smooth_metal, arch_hard_edges, arch_foliage  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = 768
ARCHS = [arch_fine_detail, arch_smooth_metal, arch_hard_edges, arch_foliage]

TUNES = [
    {"focus_method": "laplacian", "radius": 8, "levels": None},
    {"focus_method": "laplacian", "radius": 3, "levels": 6},
    {"focus_method": "mod_laplacian", "radius": 8, "levels": None},
    {"focus_method": "content_aware", "radius": 8, "levels": None},
    {"focus_method": "tenengrad", "radius": 8, "levels": None},
    {"focus_method": "gradient", "radius": 4, "levels": 6},
]
FEATS = ["contrast", "lap_e", "ten_e", "grad_e", "color_var", "entropy",
         "defocus_range", "edge_density"]


def _disk(img, r):
    if r < 0.6:
        return img.copy()
    if r <= 12:
        rr = int(np.ceil(r)); yy, xx = np.ogrid[-rr:rr + 1, -rr:rr + 1]
        k = ((xx * xx + yy * yy) <= r * r).astype(np.float32); k /= k.sum()
        return cv2.filter2D(img, -1, k)
    f = max(1, int(r / 6)); s = cv2.resize(img, (img.shape[1] // f, img.shape[0] // f))
    s = cv2.blur(s, (2 * max(1, int(r / f)) + 1,) * 2)
    return cv2.resize(s, (img.shape[1], img.shape[0]))


def make_scene(seed):
    r = np.random.default_rng(seed)
    # collage: paste random archetype crops into a grid of regions
    base = np.zeros((RES, RES, 3), np.uint8)
    depth = np.zeros((RES, RES), np.float32)
    g = 3
    step = RES // g
    for i in range(g):
        for j in range(g):
            arch = ARCHS[int(r.integers(len(ARCHS)))]
            tile_img, _ = arch(int(r.integers(10000)))
            oy, ox = int(r.integers(0, tile_img.shape[0] - step)), int(r.integers(0, tile_img.shape[1] - step))
            base[i * step:(i + 1) * step, j * step:(j + 1) * step] = tile_img[oy:oy + step, ox:ox + step]
            depth[i * step:(i + 1) * step, j * step:(j + 1) * step] = float(r.uniform(0, 1))
    depth = cv2.GaussianBlur(depth, (0, 0), step / 4)  # smooth depth transitions
    # thin wires (near)
    for _ in range(int(r.integers(2, 6))):
        p1 = (int(r.integers(RES)), int(r.integers(RES))); p2 = (int(r.integers(RES)), int(r.integers(RES)))
        cv2.line(base, p1, p2, tuple(int(v) for v in r.integers(180, 256, 3)), 2, cv2.LINE_AA)
    planes = np.linspace(0, 1, 3)
    frames = []
    for k, f in enumerate(planes):
        rad = 9.0 * np.abs(depth - f)
        levels = np.linspace(0, 9.0, 10)
        blurred = [_disk(base, s) for s in levels]
        idx = np.clip(np.round(rad / 9.0 * 9).astype(int), 0, 9)
        out = np.empty_like(base)
        for l in range(10):
            m = idx == l; out[m] = blurred[l][m]
        out = np.clip(out.astype(np.float32) + np.random.default_rng(seed + k).normal(0, 3, out.shape), 0, 255).astype(np.uint8)
        frames.append(out)
    return frames, base


def features(frames, tile):
    grays = [_gray32(f) for f in frames]
    def lstd(g, k=7):
        m = cv2.boxFilter(g, cv2.CV_32F, (k, k)); m2 = cv2.boxFilter(g * g, cv2.CV_32F, (k, k))
        return np.sqrt(np.maximum(m2 - m * m, 0))
    stds = [lstd(g) for g in grays]
    contrast = np.maximum.reduce(stds)
    defocus_range = contrast - np.minimum.reduce(stds)
    lap_e = np.maximum.reduce([np.abs(cv2.Laplacian(g, cv2.CV_32F)) for g in grays])
    gx = [cv2.Sobel(g, cv2.CV_32F, 1, 0) for g in grays]; gy = [cv2.Sobel(g, cv2.CV_32F, 0, 1) for g in grays]
    grad_e = np.maximum.reduce([cv2.magnitude(a, b) for a, b in zip(gx, gy)])
    ten_e = np.maximum.reduce([a * a + b * b for a, b in zip(gx, gy)])
    sharp_idx = np.argmax(np.stack(stds, 0), 0)
    bgr = np.stack(frames, 0); hh, ww = sharp_idx.shape; yy, xx = np.indices((hh, ww))
    sharp_bgr = bgr[sharp_idx, yy, xx]
    color_var = sharp_bgr.astype(np.float32).std(2)
    edge = (grad_e > 40).astype(np.float32)
    # local entropy proxy: std of gray in a window (cheap)
    entropy = lstd(_gray32(sharp_bgr), 9)
    maps = {"contrast": contrast, "lap_e": lap_e, "ten_e": ten_e, "grad_e": grad_e,
            "color_var": color_var, "entropy": entropy, "defocus_range": defocus_range,
            "edge_density": edge}
    # tile-average
    n = RES // tile
    X = np.zeros((n * n, len(FEATS)), np.float32)
    for ti in range(n):
        for tj in range(n):
            sl = (slice(ti * tile, (ti + 1) * tile), slice(tj * tile, (tj + 1) * tile))
            X[ti * n + tj] = [maps[f][sl].mean() for f in FEATS]
    return X


def _process(args):
    seed, tile = args
    frames, gt = make_scene(seed)
    fused = [fuse_blend(frames, **t) for t in TUNES]
    ssim = [_ssim_map(_gray32(f), _gray32(gt)) for f in fused]
    comp = [M.composite_map(frames, f) for f in fused]
    n = RES // tile
    y_gt = np.zeros(n * n, np.int64); y_comp = np.zeros(n * n, np.int64)
    for ti in range(n):
        for tj in range(n):
            sl = (slice(ti * tile, (ti + 1) * tile), slice(tj * tile, (tj + 1) * tile))
            y_gt[ti * n + tj] = int(np.argmax([s[sl].mean() for s in ssim]))
            y_comp[ti * n + tj] = int(np.argmax([c[sl].mean() for c in comp]))
    return features(frames, tile), y_gt, y_comp


def main():
    n_scenes = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    tile = int(sys.argv[2]) if len(sys.argv) > 2 else 96
    args = [(s, tile) for s in range(n_scenes)]
    Xs, ygs, ycs = [], [], []
    with cf.ProcessPoolExecutor(max_workers=min(16, os.cpu_count())) as ex:
        for X, yg, yc in ex.map(_process, args):
            Xs.append(X); ygs.append(yg); ycs.append(yc)
    X = np.vstack(Xs); y_gt = np.concatenate(ygs); y_comp = np.concatenate(ycs)
    np.savez(os.path.join(HERE, "trainset.npz"), X=X, y_gt=y_gt, y_comp=y_comp,
             feats=np.array(FEATS), tunes=np.array([str(t) for t in TUNES]))
    print(f"dataset: {X.shape[0]} tiles x {X.shape[1]} features, {len(TUNES)} tunes")
    print(f"GT label distribution:   {np.bincount(y_gt, minlength=len(TUNES))}")
    print(f"comp label distribution: {np.bincount(y_comp, minlength=len(TUNES))}")
    print(f"GT/composite label agreement: {(y_gt == y_comp).mean():.3f}")
    print(f"(agreement = how deployable pure no-GT labeling is right now)")
    print(f"wrote {os.path.join(HERE, 'trainset.npz')}")


if __name__ == "__main__":
    main()
