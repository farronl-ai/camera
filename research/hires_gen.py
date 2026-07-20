#!/usr/bin/env python3
"""H1 — generate high-res multi-focus stacks WITH ground truth.

Real high-res photos (Wikimedia Commons, freely licensed) serve as the all-in-focus
GT; we synthesize realistic defocus: depth-dependent DISK (circle-of-confusion) PSF
+ longitudinal CHROMATIC ABERRATION (per-channel focus offset -> color fringing at
defocus boundaries) + sensor noise. This is the Real-MFF construction protocol, but
at a resolution and content-diversity we control.

Content types span the scene-dependence axes: fine multicolor detail, smooth+
specular metal, hard edges, organic texture.

Run:  python research/hires_gen.py [width] [per_query] [frames]
Writes research/data/hires/<id>/ {gt.png, frame_0..N.png, depth.png} + manifest.json
"""
from __future__ import annotations
import json
import os
import sys
import urllib.parse
import urllib.request

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hardbench import disk_blur, add_noise  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "hires")
API = "https://commons.wikimedia.org/w/api.php"
UA = "focusstack-research/1.0 (https://github.com/farronl-ai/camera; research use)"

# query -> content-type label (for per-content-type analysis in H2)
QUERIES = {
    "kingfisher bird": "fine_detail",
    "butterfly wing macro": "fine_detail",
    "fern frond green": "foliage",
    "polished metal surface reflection": "metal_specular",
    "coin macro detail": "metal_specular",
    "tree bark texture closeup": "rough_texture",
    "stone inscription carved letters": "hard_edges",
}


def _get(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=timeout).read()


def search(query, n, width, minw=2400):
    params = dict(action="query", format="json", generator="search", gsrsearch=query,
                  gsrnamespace=6, gsrlimit=n * 4, prop="imageinfo",
                  iiprop="url|size|mime|extmetadata", iiurlwidth=width)
    data = json.loads(_get(API + "?" + urllib.parse.urlencode(params), timeout=60))
    out = []
    for p in data.get("query", {}).get("pages", {}).values():
        ii = (p.get("imageinfo") or [{}])[0]
        if ii.get("mime") != "image/jpeg" or ii.get("width", 0) < minw:
            continue
        lic = ii.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "?")
        out.append((p["title"], ii.get("thumburl") or ii.get("url"),
                    ii.get("width"), ii.get("height"), lic))
        if len(out) >= n:
            break
    return out


def defocus_ca(base, depth, focus, max_r, ca=0.04, n_levels=12):
    """Depth-dependent disk defocus with longitudinal chromatic aberration.

    Each channel focuses at a slightly different depth (offs), so out-of-focus
    boundaries get color fringing — a real high-res failure mode to stress fusion.
    """
    offs = (-ca, 0.0, ca)  # B, G, R focus offsets (BGR order)
    levels = np.linspace(0.0, max_r, n_levels)
    out = np.empty_like(base)
    for c in range(3):
        rad = max_r * np.abs(depth - (focus + offs[c]))
        blurred = [disk_blur(base[..., c], s) for s in levels]
        idx = np.clip(np.round(rad / (max_r + 1e-9) * (n_levels - 1)).astype(int), 0, n_levels - 1)
        plane = np.empty(base.shape[:2], base.dtype)
        for lv in range(n_levels):
            m = idx == lv
            plane[m] = blurred[lv][m]
        out[..., c] = plane
    return out


def depth_map(kind, h, w, seed):
    if kind == "gradient":
        return np.tile(np.linspace(0, 1, w), (h, 1)).astype(np.float32)
    if kind == "radial":
        yy, xx = np.mgrid[0:h, 0:w]
        d = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
        return (d / d.max()).astype(np.float32)
    # twoplane: smooth random foreground blobs (near) over background (far)
    r = np.random.default_rng(seed)
    m = np.zeros((h, w), np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(3):
        cy, cx = r.integers(h), r.integers(w)
        ry, rx = r.integers(h // 5, h // 2), r.integers(w // 5, w // 2)
        m[((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 < 1] = 1
    m = cv2.GaussianBlur(m, (0, 0), min(h, w) / 40)
    return (0.1 + 0.8 * (1 - m)).astype(np.float32)


def main():
    width = int(sys.argv[1]) if len(sys.argv) > 1 else 3072
    per_q = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    frames = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    os.makedirs(OUT, exist_ok=True)
    depth_kinds = ["gradient", "radial", "twoplane"]
    manifest = []
    idx = 0
    for query, ctype in QUERIES.items():
        try:
            hits = search(query, per_q, width)
        except Exception as e:  # noqa: BLE001
            print(f"  search FAIL '{query}': {str(e)[:80]}")
            continue
        for title, url, ow, oh, lic in hits:
            sid = f"{idx:02d}_{ctype}"
            sdir = os.path.join(OUT, sid)
            os.makedirs(sdir, exist_ok=True)
            try:
                arr = np.frombuffer(_get(url), np.uint8)
                base = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except Exception as e:  # noqa: BLE001
                print(f"  dl FAIL {title[:40]}: {str(e)[:60]}")
                continue
            if base is None:
                continue
            # cap long side to `width` to bound compute
            h, w = base.shape[:2]
            if max(h, w) > width:
                s = width / max(h, w)
                base = cv2.resize(base, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            h, w = base.shape[:2]
            kind = depth_kinds[idx % len(depth_kinds)]
            depth = depth_map(kind, h, w, idx)
            max_r = 0.012 * max(h, w)  # CoC scales with resolution
            planes = np.linspace(0, 1, frames)
            cv2.imwrite(os.path.join(sdir, "gt.png"), base)
            cv2.imwrite(os.path.join(sdir, "depth.png"), (depth * 255).astype(np.uint8))
            for fi, fp in enumerate(planes):
                fr = add_noise(defocus_ca(base, depth, fp, max_r), 3.0, fi)
                cv2.imwrite(os.path.join(sdir, f"frame_{fi}.png"), fr)
            manifest.append({"id": sid, "content_type": ctype, "depth_type": kind,
                             "dims": [h, w], "orig_dims": [oh, ow], "frames": frames,
                             "max_coc_px": round(max_r, 1), "title": title, "license": lic})
            print(f"  {sid:18s} {w}x{h}  depth={kind:9s} CoC={max_r:.0f}px  {lic}  {title[:40]}")
            idx += 1
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)
    print(f"\ngenerated {len(manifest)} high-res stacks -> {OUT}")


if __name__ == "__main__":
    main()
