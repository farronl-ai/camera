#!/usr/bin/env python3
"""Profile the classical engine at high res — find the real bottleneck (don't assume).

Times the stages of fuse_blend (content_aware + harden) on a 2-frame stack at
several resolutions: focus energies, guided weights, pyramid build, multiband
collapse. Guides where to spend the speedup.

Run:  python research/profile_engine.py
"""
from __future__ import annotations
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from focusstack.focus import content_aware_energies  # noqa: E402
from focusstack.fusion import (_auto_levels, _gaussian_pyramid, _laplacian_pyramid,  # noqa: E402
                               guided_filter, multiband_blend)
from focusstack.io import to_gray_float  # noqa: E402


def make_pair(sz):
    rng = np.random.default_rng(0)
    a = cv2.GaussianBlur(rng.integers(0, 256, (sz, sz, 3), np.uint8), (0, 0), 1.0)
    b = cv2.GaussianBlur(a, (0, 0), 3.0)  # b = blurred version (crude stack)
    return a, b


def timeit(fn, reps=3):
    best = 1e9
    for _ in range(reps):
        t = time.time(); fn(); best = min(best, time.time() - t)
    return best * 1000  # ms


def profile(sz, harden=0.5):
    a, b = make_pair(sz)
    images = [a, b]
    grays = [to_gray_float(im) for im in images]

    t_focus = timeit(lambda: content_aware_energies(grays))
    energy = np.stack(content_aware_energies(grays), 0)
    winner = np.argmax(energy, 0)
    srt = np.sort(energy, 0)
    conf = np.clip(cv2.boxFilter(((srt[-1] - srt[-2]) / (srt[-1] + 1e-6)).astype(np.float32),
                                 cv2.CV_32F, (15, 15)) * harden, 0, 1)

    def guided_step():
        W = []
        for k, im in enumerate(images):
            raw = (winner == k).astype(np.float32)
            wg = np.clip(guided_filter(to_gray_float(im) / 255.0, raw, 8, 1e-3), 0, None)
            W.append((1 - conf) * wg + conf * raw)
        w = np.stack(W, 0); return w / (w.sum(0, keepdims=True) + 1e-8)
    t_guided = timeit(guided_step)
    weights = guided_step()

    levels = _auto_levels(a.shape, None)
    t_lap = timeit(lambda: [_laplacian_pyramid(im.astype(np.float32), levels) for im in images])
    t_wpyr = timeit(lambda: [_gaussian_pyramid(weights[k], levels) for k in range(len(images))])
    t_blend = timeit(lambda: multiband_blend(images, weights, levels))

    total = t_focus + t_guided + t_lap + t_wpyr + t_blend
    print(f"\n=== {sz}x{sz} (levels={levels}) ===")
    for name, t in [("focus energies", t_focus), ("guided weights", t_guided),
                    ("laplacian pyramids", t_lap), ("weight pyramids", t_wpyr),
                    ("multiband collapse", t_blend)]:
        print(f"  {name:20s} {t:8.1f} ms  ({100*t/total:4.1f}%)")
    print(f"  {'TOTAL (sum)':20s} {total:8.1f} ms")


if __name__ == "__main__":
    for sz in (1024, 2048, 4096):
        profile(sz)
