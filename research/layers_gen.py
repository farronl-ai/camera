#!/usr/bin/env python3
"""E1 — layered/nested scene generator with TRUE boundary + near-side ground truth.

Scenes are K-layer alpha-matte composites (2-4 depth layers) built from REAL photo
textures, rendered with per-layer, per-channel (chromatic) disk defocus at N focus
planes — proper back-to-front compositing, so occlusion physics is honest.

Deliberate messiness (our data is usually too clean):
  - irregular blob/ribbon object silhouettes with HOLES (nesting: the layer behind
    shows through inside an object — objects within objects);
  - a WRAP object: one texture split into two masks at different layer orders
    around a middle object — genuinely half in front, half behind;
  - a CAMOUFLAGE object: texture cut from the SAME photo region as its background
    → appearance channels see (almost) nothing where depth differs. This is the
    E3 orthogonality probe: only depth-aware evidence can find that boundary;
  - sensor noise + slight per-frame exposure jitter (engine corrects it — realistic).

Ground truth emitted per scene: gt.png (sharp composite), depth.png (visible-layer
depth), boundary.png (visible-layer label discontinuities), nearside.png (127+
signed depth step across the boundary), frames_*.png, and a vis montage.

Run:  python research/layers_gen.py [n_scenes] [long_side]
"""
from __future__ import annotations
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hardbench import disk_blur  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "hires")
OUT = os.path.join(HERE, "data", "layers")


def blob_mask(h, w, rng, cx=None, cy=None, scale=0.22, holes=True):
    """Irregular smoothed random polygon; optional interior holes (nesting)."""
    cx = int(rng.uniform(0.2, 0.8) * w) if cx is None else cx
    cy = int(rng.uniform(0.2, 0.8) * h) if cy is None else cy
    r0 = scale * min(h, w)
    ang = np.sort(rng.uniform(0, 2 * np.pi, int(rng.integers(7, 14))))
    rad = r0 * rng.uniform(0.5, 1.4, len(ang))
    pts = np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], 1).astype(np.int32)
    m = np.zeros((h, w), np.float32)
    cv2.fillPoly(m, [pts], 1.0)
    if holes and rng.random() < 0.8:
        for _ in range(int(rng.integers(1, 3))):
            hx, hy = int(cx + rng.uniform(-0.4, 0.4) * r0), int(cy + rng.uniform(-0.4, 0.4) * r0)
            cv2.circle(m, (hx, hy), int(r0 * rng.uniform(0.12, 0.3)), 0.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), 1.2)  # soft AA edge, still a crisp contour
    return m, (cx, cy, r0)


def ribbon_mask(h, w, rng):
    """A thick curved ribbon crossing the frame (for the wrap object)."""
    m = np.zeros((h, w), np.float32)
    y = rng.uniform(0.3, 0.7) * h
    pts = []
    for t in np.linspace(0, 1, 24):
        pts.append([int(t * (w - 1)), int(y + 0.18 * h * np.sin(2 * np.pi * (t * rng.uniform(0.7, 1.4) + rng.random())))])
    cv2.polylines(m, [np.array(pts, np.int32)], False, 1.0, int(rng.integers(14, 30)))
    return cv2.GaussianBlur(m, (0, 0), 1.2)


def texture(photos, rng, h, w, same_as=None, offset_frac=0.25):
    """Crop a texture from a random photo; `same_as` (photo,y,x) makes camouflage."""
    if same_as is not None:
        img, y0, x0 = same_as
        dy, dx = int(offset_frac * h * rng.uniform(-1, 1)), int(offset_frac * w * rng.uniform(-1, 1))
        y0 = np.clip(y0 + dy, 0, img.shape[0] - h)
        x0 = np.clip(x0 + dx, 0, img.shape[1] - w)
        return img[y0:y0 + h, x0:x0 + w].copy()
    img = photos[int(rng.integers(len(photos)))]
    y0 = int(rng.integers(0, max(1, img.shape[0] - h)))
    x0 = int(rng.integers(0, max(1, img.shape[1] - w)))
    return img[y0:y0 + h, x0:x0 + w].copy()


def composite_sharp(layers):
    """Back-to-front sharp composite + visible-layer label + visible depth map."""
    h, w = layers[0]["tex"].shape[:2]
    out = layers[0]["tex"].astype(np.float32).copy()
    label = np.zeros((h, w), np.int32)
    depth = np.full((h, w), layers[0]["depth"], np.float32)
    for i, L in enumerate(layers[1:], start=1):
        a = L["alpha"][..., None]
        out = L["tex"].astype(np.float32) * a + out * (1 - a)
        vis = L["alpha"] > 0.5
        label[vis] = i
        depth[vis] = L["depth"]
    return np.clip(out, 0, 255).astype(np.uint8), label, depth


def render_frame(layers, focus, max_r, ca=0.04, rng=None, seed=0):
    """Back-to-front composite with per-layer per-channel disk defocus."""
    h, w = layers[0]["tex"].shape[:2]
    out = np.empty((h, w, 3), np.float32)
    offs = (-ca, 0.0, ca)
    for c in range(3):
        r0 = max_r * abs(layers[0]["depth"] - (focus + offs[c]))
        acc = disk_blur(layers[0]["tex"][..., c].astype(np.float32), r0)
        for L in layers[1:]:
            r = max_r * abs(L["depth"] - (focus + offs[c]))
            pm = disk_blur((L["tex"][..., c] * L["alpha"]).astype(np.float32), r)
            ab = disk_blur(L["alpha"], r)
            acc = pm + acc * (1 - ab)
        out[..., c] = acc
    r = np.random.default_rng(seed)
    gain = r.uniform(0.94, 1.06)                       # exposure jitter (realistic)
    out = out * gain + r.normal(0, 3.0, out.shape)     # sensor noise
    return np.clip(out, 0, 255).astype(np.uint8)


def build_scene(photos, seed, long_side):
    rng = np.random.default_rng(seed)
    base_img = photos[int(rng.integers(len(photos)))]
    h0, w0 = base_img.shape[:2]
    s = long_side / max(h0, w0)
    h, w = int(h0 * s), int(w0 * s)
    bg = cv2.resize(base_img, (w, h), interpolation=cv2.INTER_AREA)
    # crop source location for camouflage
    by0 = int(rng.integers(0, max(1, base_img.shape[0] - h)))
    bx0 = int(rng.integers(0, max(1, base_img.shape[1] - w)))

    layers = [{"tex": bg, "alpha": None, "depth": 0.9}]
    depths = [0.55, 0.3, 0.12]
    # middle blob object (with holes -> nesting)
    a1, _ = blob_mask(h, w, rng, scale=0.26)
    layers.append({"tex": texture(photos, rng, h, w), "alpha": a1, "depth": depths[0]})
    # CAMOUFLAGE object: same-photo texture, nearer depth, iso-appearance boundary
    a2, _ = blob_mask(h, w, rng, scale=0.18, holes=False)
    layers.append({"tex": texture(photos, rng, h, w, same_as=(base_img, by0, bx0)),
                   "alpha": a2, "depth": depths[1]})
    # WRAP ribbon: one texture, split left/right across the middle object's order:
    # left half rendered IN FRONT of everything (depth .12), right half BEHIND the
    # middle blob (depth .7, occluded by it) -> half in front, half behind.
    rib = ribbon_mask(h, w, rng)
    xsplit = int(w * rng.uniform(0.4, 0.6))
    left = rib.copy(); left[:, xsplit:] = 0
    right = rib.copy(); right[:, :xsplit] = 0
    rib_tex = texture(photos, rng, h, w)
    layers.insert(1, {"tex": rib_tex, "alpha": right, "depth": 0.7})   # behind blob
    layers.append({"tex": rib_tex, "alpha": left, "depth": depths[2]})  # front-most
    return layers, (h, w)


def boundary_gt(label, depth):
    k = np.ones((3, 3), np.uint8)
    lab = label.astype(np.float32)
    bmap = ((cv2.dilate(lab, k) != cv2.erode(lab, k))).astype(np.uint8)
    # near-side: signed depth step across the boundary (dilate-min vs dilate-max)
    dmin = cv2.erode(depth, np.ones((5, 5), np.uint8))
    step = depth - dmin
    nearside = np.clip(127 + 60 * np.sign(step) * (bmap > 0), 0, 255).astype(np.uint8)
    return bmap, nearside


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    long_side = int(sys.argv[2]) if len(sys.argv) > 2 else 1536
    photos = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(SRC, "*", "gt.png")))]
    photos = [p for p in photos if p is not None]
    os.makedirs(OUT, exist_ok=True)
    manifest = []
    for i in range(n):
        layers, (h, w) = build_scene(photos, seed=100 + i, long_side=long_side)
        gt, label, depth = composite_sharp(layers)
        bmap, nearside = boundary_gt(label, depth)
        max_r = 0.012 * max(h, w)
        planes = [0.12, 0.42, 0.9]
        sdir = os.path.join(OUT, f"scene_{i:02d}")
        os.makedirs(sdir, exist_ok=True)
        cv2.imwrite(os.path.join(sdir, "gt.png"), gt)
        cv2.imwrite(os.path.join(sdir, "depth.png"), (depth * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(sdir, "boundary.png"), bmap * 255)
        cv2.imwrite(os.path.join(sdir, "nearside.png"), nearside)
        for fi, fp in enumerate(planes):
            fr = render_frame(layers, fp, max_r, seed=1000 * i + fi)
            cv2.imwrite(os.path.join(sdir, f"frame_{fi}.png"), fr)
        # vis montage: gt | frame1 | boundary overlay
        ov = gt.copy(); ov[bmap > 0] = (0, 0, 255)
        sc = lambda im: cv2.resize(im, (520, int(im.shape[0] * 520 / im.shape[1])))
        f1 = cv2.imread(os.path.join(sdir, "frame_1.png"))
        cv2.imwrite(os.path.join(sdir, "vis.png"), np.hstack([sc(gt), sc(f1), sc(ov)]))
        manifest.append({"id": f"scene_{i:02d}", "dims": [h, w], "layers": len(layers),
                         "planes": planes, "max_coc_px": round(max_r, 1)})
        print(f"  scene_{i:02d}: {w}x{h}, {len(layers)} layers (bg+ribbon-behind+blob+camo+ribbon-front)")
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)
    print(f"\ngenerated {n} layered scenes -> {OUT}")


if __name__ == "__main__":
    main()
