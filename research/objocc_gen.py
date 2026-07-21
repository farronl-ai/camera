#!/usr/bin/env python3
"""16e — OBJECTS-AS-OCCLUDERS benchmark (the honest judge for semantic mattes).

F43: pastiche blobs aren't objects (SAM fragments them) and photo bokeh spoofs
objectness. Fix: the occluder IS a real object — a FastSAM cutout from one photo
(its silhouette = TRUE GT alpha), composited over a DIFFERENT photo with the
established per-channel defocus physics. Semantic models now face exactly the
distribution they were trained on, and we still hold perfect GT.

Object-mask selection per source photo: area 2-20% of image, compact
(isoperimetric > 0.02), border contact < 40% of its perimeter box. One scene per
(source object, target background) pairing until n_scenes.

Run:  python research/objocc_gen.py [n_scenes] [coc_frac]
Writes research/data/objocc/scene_XX/{gt,alpha,frame_0,frame_1,vis}.png + manifest.
"""
from __future__ import annotations
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from occ_gen import occ_defocus, LONG  # noqa: E402
from hires_gen import add_noise  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "hires")
OUT = os.path.join(HERE, "data", "objocc")


def good_object_masks(masks, h, w):
    keep = []
    for m in masks:
        mm = (m > 0).astype(np.uint8)
        area = float(mm.sum())
        if not (0.02 * h * w < area < 0.20 * h * w):
            continue
        per = cv2.countNonZero(cv2.morphologyEx(mm, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)))
        if 4 * np.pi * area / (per ** 2 + 1e-6) < 0.02:
            continue
        border = mm[0, :].sum() + mm[-1, :].sum() + mm[:, 0].sum() + mm[:, -1].sum()
        if border > 0.4 * (h + w):
            continue
        keep.append(mm)
    return keep


def main():
    n_scenes = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    coc_frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    photos = sorted(glob.glob(os.path.join(SRC, "*", "gt.png")))
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(7 + start)
    manifest = json.load(open(os.path.join(OUT, "manifest.json"))) if start and \
        os.path.exists(os.path.join(OUT, "manifest.json")) else []
    made = start
    srcs = [p for r in range(12) for p in photos]     # multiple rounds for scale
    for i, src in enumerate(srcs):
        if made >= n_scenes:
            break
        mp = src + ".masks.npy"
        if not os.path.exists(mp):
            continue
        img = cv2.imread(src)
        masks = good_object_masks(np.load(mp), *img.shape[:2])
        if not masks:
            continue
        mm = masks[int(rng.integers(len(masks)))]
        # background = a DIFFERENT photo
        bg_path = photos[(i + 3 + 2 * (i // len(photos)) + start) % len(photos)]
        bg = cv2.imread(bg_path)
        bh, bw = bg.shape[:2]
        s = LONG / max(bh, bw)
        bg = cv2.resize(bg, (int(bw * s), int(bh * s)), interpolation=cv2.INTER_AREA)
        hh, ww = bg.shape[:2]

        # scale the cutout (object + alpha) into the background frame, random placement
        ys, xs = np.where(mm > 0)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        obj = img[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
        a = mm[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
        scale = min(0.55 * hh / obj.shape[0], 0.55 * ww / obj.shape[1], 1.5)
        obj = cv2.resize(obj, None, fx=scale, fy=scale)
        a = cv2.resize(a, None, fx=scale, fy=scale)
        a = cv2.GaussianBlur(a, (0, 0), 1.0)          # soft AA silhouette
        oh, ow = a.shape
        py = int(rng.integers(0, hh - oh)) if hh > oh else 0
        px = int(rng.integers(0, ww - ow)) if ww > ow else 0
        near = np.zeros((hh, ww, 3), np.float32)
        alpha = np.zeros((hh, ww), np.float32)
        near[py:py + oh, px:px + ow] = obj
        alpha[py:py + oh, px:px + ow] = a

        gt = (near * alpha[..., None] + bg.astype(np.float32) * (1 - alpha[..., None])).astype(np.uint8)
        max_r = coc_frac * max(hh, ww)
        frames = [add_noise(occ_defocus(bg, near, alpha, f, 0.15, 0.85, max_r), 3.0, 10 * made + k)
                  for k, f in enumerate([0.15, 0.85])]
        sdir = os.path.join(OUT, f"scene_{made:02d}")
        os.makedirs(sdir, exist_ok=True)
        cv2.imwrite(os.path.join(sdir, "gt.png"), gt)
        cv2.imwrite(os.path.join(sdir, "alpha.png"), (alpha * 255).astype(np.uint8))
        for k, fr in enumerate(frames):
            cv2.imwrite(os.path.join(sdir, f"frame_{k}.png"), fr)
        ov = gt.copy()
        edge = cv2.morphologyEx((alpha > 0.5).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        ov[edge > 0] = (0, 0, 255)
        sc = lambda x: cv2.resize(x, (500, int(x.shape[0] * 500 / x.shape[1])))
        cv2.imwrite(os.path.join(sdir, "vis.png"), np.hstack([sc(gt), sc(frames[1]), sc(ov)]))
        manifest.append({"id": f"scene_{made:02d}", "src": os.path.basename(os.path.dirname(src)),
                         "bg": os.path.basename(os.path.dirname(bg_path)), "max_r": round(float(max_r), 1)})
        print(f"  scene_{made:02d}: object from {manifest[-1]['src']} over {manifest[-1]['bg']}, "
              f"alpha area {float((alpha>0.5).mean())*100:.1f}%, max_r={max_r:.1f}px")
        made += 1
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)
    print(f"\n{made} object-occluder scenes -> {OUT}")


if __name__ == "__main__":
    main()
