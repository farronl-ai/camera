#!/usr/bin/env python3
"""L2 — content analysis: real edges + local structure-scale map S(x,y).

The key quantity is S(x,y): the characteristic spatial scale of the sharpest
structure at each pixel — SMALL on fine detail, LARGE on smooth/large structure.
Measured by scale-space selection: per-pixel argmax over octave scales of the
scale-normalized DoG response (max across frames = sharpest available). This is
what the fusion window should track LOCALLY (not a global resolution number).

Also: real edge detection (Canny) on the max-focus composite, for boundary-aware
regularization / stitching.

Run:  python research/analyze.py [id]   (default: first mixed stack)
Writes research/analyze_out/<id>_maps.png and prints a content sanity check.
"""
from __future__ import annotations
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from focusstack.io import to_gray_float  # noqa: E402
from focusstack.focus import focus_measure  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MIX = os.path.join(HERE, "data", "hires_mixed")
OUT = os.path.join(HERE, "analyze_out")
os.makedirs(OUT, exist_ok=True)
SCALES = (2.0, 4.0, 8.0, 16.0, 32.0)


def load(sid):
    d = os.path.join(MIX, sid)
    frames = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(d, "frame_*.png")))]
    gt = cv2.imread(os.path.join(d, "gt.png"))
    depth = cv2.imread(os.path.join(d, "depth.png"), cv2.IMREAD_GRAYSCALE)
    return frames, gt, depth


def max_focus_composite(frames):
    """Rough all-in-focus image: per pixel take the locally-sharpest frame."""
    fm = np.stack([focus_measure(to_gray_float(f)) for f in frames], 0)
    idx = np.argmax(fm, 0)
    h, w = idx.shape
    yy, xx = np.indices((h, w))
    return np.stack(frames, 0)[idx, yy, xx]


def detail_energy(frames, scale=2.0, pool=9):
    """Fine-scale detail presence per pixel, max across frames (sharpest available).

    High where genuine fine detail exists (regardless of how big the object is);
    ~0 on smooth surfaces. This — not the *dominant* scale — is what tells the
    fusion whether it must use a SMALL window to preserve detail here.
    """
    e = []
    for f in frames:
        g = to_gray_float(f)
        lo = cv2.GaussianBlur(g, (0, 0), scale)
        hi = cv2.GaussianBlur(g, (0, 0), scale * 1.6)
        e.append(cv2.boxFilter(np.abs(lo - hi), cv2.CV_32F, (pool, pool)))
    return np.maximum.reduce(e)


def local_window_map(frames, w_min=6.0, w_max=None):
    """Per-pixel fusion window (px): SMALL where fine detail, LARGE where smooth.

    Measured from fine-detail energy (content-driven), not a global constant.
    w_max defaults to ~the CoC (0.012*max_dim) — the scale robustness needs on
    smooth large-CoC regions; w_min is small to preserve fine detail.
    """
    h, w = frames[0].shape[:2]
    if w_max is None:
        w_max = max(w_min + 2, 0.012 * max(h, w))
    fe = detail_energy(frames)
    fe = np.clip(fe / (np.percentile(fe, 98) + 1e-6), 0, 1)
    return (w_max - (w_max - w_min) * fe).astype(np.float32)  # fine->w_min, smooth->w_max


# back-compat name used below
def structure_scale(frames):
    win = local_window_map(frames)
    return win, None


def main():
    ids = [os.path.basename(os.path.dirname(p)) for p in sorted(glob.glob(os.path.join(MIX, "*", "gt.png")))]
    sid = sys.argv[1] if len(sys.argv) > 1 else ids[0]
    frames, gt, depth = load(sid)
    comp = max_focus_composite(frames)
    edges = cv2.Canny(cv2.cvtColor(comp, cv2.COLOR_BGR2GRAY), 60, 160)
    S, _ = structure_scale(frames)

    # sanity: S should be SMALL where content is detailed (depth<128 => near/detailed)
    detailed = depth < 128
    print(f"[{sid}] structure-scale S sanity (px):")
    print(f"  mean S on DETAILED regions = {S[detailed].mean():.1f}")
    print(f"  mean S on SMOOTH   regions = {S[~detailed].mean():.1f}")
    print(f"  (expect detailed << smooth)")

    # visualize: composite | edges | S heatmap
    Svis = cv2.applyColorMap(np.clip((S - S.min()) / (np.ptp(S) + 1e-6) * 255, 0, 255).astype(np.uint8),
                             cv2.COLORMAP_TURBO)
    ev = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    sc = lambda im, w=640: cv2.resize(im, (w, int(im.shape[0] * w / im.shape[1])))
    cv2.imwrite(os.path.join(OUT, f"{sid}_maps.png"), np.hstack([sc(comp), sc(ev), sc(Svis)]))
    print(f"wrote {OUT}/{sid}_maps.png  [ max-focus composite | Canny edges | S (blue=fine, red=coarse) ]")


if __name__ == "__main__":
    main()
