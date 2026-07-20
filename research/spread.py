#!/usr/bin/env python3
"""Defocus-spread rejection via confidence-hardened blending.

Problem: near a thin/bright sharp structure, guided+multiband blending pulls in
the OTHER frame's defocus-spread (a big bright disk), softening/graying the
result. Fix (theory-grounded): where one frame is CONFIDENTLY the sharpest
(focus-energy dominance high — thin structures, strong edges), push the weights
toward hard one-hot selection so no spread can bleed in; keep soft guided blending
only where focus is ambiguous (smooth regions, where hard selection would speckle).

    w_final[k] = (1-conf)*w_guided[k] + conf*onehot(argmax)[k]
    conf = (E_max - E_2nd) / (E_max + eps)   (pooled)

Tested on a HARSH spread scene (thick bright near bars/dots, large CoC).

Run:  python research/spread.py
"""
from __future__ import annotations
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from focusstack.fusion import fuse_blend, guided_filter, multiband_blend  # noqa: E402
from focusstack.focus import focus_measure  # noqa: E402
from focusstack.io import to_gray_float  # noqa: E402
from regions import fuse_adaptive  # noqa: E402
from hardbench import disk_blur  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SZ = 1024


def harsh_spread_scene(seed=0):
    r = np.random.default_rng(seed)
    bg = cv2.GaussianBlur(r.integers(0, 256, (SZ, SZ, 3), np.uint8), (3, 3), 0.7)
    base = bg.copy()
    depth = np.full((SZ, SZ), 0.9, np.float32)
    for _ in range(5):
        p1 = (int(r.integers(SZ)), int(r.integers(SZ))); p2 = (int(r.integers(SZ)), int(r.integers(SZ)))
        cv2.line(base, p1, p2, (255, 255, 255), int(r.integers(3, 8)), cv2.LINE_AA)
    for _ in range(25):
        c = (int(r.integers(SZ)), int(r.integers(SZ)))
        cv2.circle(base, c, int(r.integers(6, 16)), (255, 255, 255), -1)
    diff = np.abs(base.astype(np.int16) - bg.astype(np.int16)).max(2)
    depth[diff > 60] = 0.1
    return base, depth


def make_frames(base, depth, seed=0):
    frames = []
    for i, f in enumerate([0.1, 0.9]):
        rad = 26.0 * np.abs(depth - f)
        levels = np.linspace(0, 26.0, 12)
        blurred = [disk_blur(base, s) for s in levels]
        idx = np.clip(np.round(rad / 26.0 * 11).astype(int), 0, 11)
        out = np.empty_like(base)
        for l in range(12):
            m = idx == l; out[m] = blurred[l][m]
        out = np.clip(out.astype(np.float32) + np.random.default_rng(seed + i).normal(0, 3, out.shape), 0, 255).astype(np.uint8)
        frames.append(out)
    return frames


def fuse_spread_reject(frames, focus_method="laplacian", radius=8, eps=1e-3,
                       smooth_ksize=9, conf_ksize=15):
    grays = [to_gray_float(f) for f in frames]
    E = np.stack([focus_measure(g, method=focus_method, smooth_ksize=smooth_ksize) for g in grays], 0)
    winner = np.argmax(E, 0)
    srt = np.sort(E, axis=0)
    emax, e2 = srt[-1], srt[-2]
    conf = (emax - e2) / (emax + 1e-6)
    conf = cv2.boxFilter(conf, cv2.CV_32F, (conf_ksize, conf_ksize))
    conf = np.clip(conf, 0, 1)
    n = len(frames)
    W = []
    for k, f in enumerate(frames):
        raw = (winner == k).astype(np.float32)
        wg = np.clip(guided_filter(to_gray_float(f) / 255.0, raw, radius, eps), 0.0, None)
        onehot = (winner == k).astype(np.float32)
        W.append((1 - conf) * wg + conf * onehot)
    W = np.stack(W, 0)
    W = W / (W.sum(0, keepdims=True) + 1e-8)
    return multiband_blend(frames, W)


def main():
    base, depth = harsh_spread_scene(0)
    gt = base
    frames = make_frames(base, depth, 0)
    methods = {
        "baseline": fuse_blend(frames),
        "adaptive": fuse_adaptive(frames),
        "spread_reject": fuse_spread_reject(frames),
    }
    from metrics import _ssim_map, _gray32
    print(f"{'method':16s} {'GT-SSIM':>8s} {'tile_worst':>11s}")
    ssim_maps = {}
    for name, fu in methods.items():
        s = _ssim_map(_gray32(fu), _gray32(gt))
        ssim_maps[name] = s
        t = 8; h, w = s.shape
        worst = min(s[i * (h // t):(i + 1) * (h // t), j * (w // t):(j + 1) * (w // t)].mean()
                    for i in range(t) for j in range(t))
        print(f"{name:16s} {M.ref_ssim(fu, gt):8.4f} {worst:11.4f}")
    # worst-error crop (baseline) for visual comparison
    err = cv2.boxFilter(np.abs(methods["baseline"].astype(np.int16) - gt).sum(2).astype(np.float32), cv2.CV_32F, (31, 31))
    y, x = np.unravel_index(err.argmax(), err.shape)
    h = 120; y0, x0 = max(0, y - h), max(0, x - h)
    def z(im): return cv2.resize(im[y0:y0 + 2 * h, x0:x0 + 2 * h], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(HERE, "spread_zoom.png"),
                np.hstack([z(gt), z(frames[1]), z(methods["baseline"]), z(methods["spread_reject"])]))
    print("wrote research/spread_zoom.png [ GT | spread-frame | baseline | spread_reject ]")


if __name__ == "__main__":
    main()
