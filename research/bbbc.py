#!/usr/bin/env python3
"""B2 — real optical z-stacks from BBBC006 (microscopy, real defocus).

Each BBBC006 zip is one focal plane (z_00..z_33) across all wells (~800MB). We
download a few planes and extract, for a handful of wells, the matching w1
(nuclei) images across planes -> real optical focal stacks (16-bit TIF, 696x520,
focus at z=16). Best-effort: if download is too slow/large, take what we got.

Run:  python research/bbbc.py [comma-separated planes] [n_wells]
"""
from __future__ import annotations
import io
import os
import sys
import urllib.request
import zipfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "bbbc006")
BASE = "https://data.broadinstitute.org/bbbc/BBBC006"
UA = "focusstack-research/1.0 (research use)"


def fetch_plane(z):
    url = f"{BASE}/BBBC006_v1_images_z_{z:02d}.zip"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    print(f"  downloading z_{z:02d} ...", flush=True)
    data = urllib.request.urlopen(req, timeout=600).read()
    print(f"  z_{z:02d}: {len(data)//(1024*1024)} MB", flush=True)
    return zipfile.ZipFile(io.BytesIO(data))


def key(name):
    b = os.path.basename(name)
    return b.split("_w1")[0] if "_w1" in b else None


def main():
    planes = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [10, 16, 22]
    n_wells = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    os.makedirs(OUT, exist_ok=True)

    # Download each plane once (cache), reusing the seed plane for key selection.
    seed = 16 if 16 in planes else planes[0]
    cache = {seed: fetch_plane(seed)}
    for z in planes:
        if z not in cache:
            cache[z] = fetch_plane(z)

    # From the seed plane, pick n_wells well/site keys (w1 = nuclei).
    w1 = sorted(n for n in cache[seed].namelist()
                if "_w1" in os.path.basename(n) and n.lower().endswith(".tif"))
    keys = []
    for n in w1:
        k = key(n)
        if k and k not in keys:
            keys.append(k)
        if len(keys) >= n_wells:
            break
    print(f"  selected wells: {keys}")

    made = 0
    for k in keys:
        sdir = os.path.join(OUT, k)
        os.makedirs(sdir, exist_ok=True)
        for i, z in enumerate(sorted(planes)):
            entry = next((n for n in cache[z].namelist()
                          if key(n) == k and "_w1" in os.path.basename(n) and n.lower().endswith(".tif")), None)
            if entry is None:
                continue
            raw = np.frombuffer(cache[z].read(entry), np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)     # 16-bit
            if img is None:
                continue
            img8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            img8 = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(os.path.join(sdir, f"frame_{i}_z{z:02d}.png"), img8)
        made += 1
        print(f"  {k}: {len(sorted(planes))} planes")
    print(f"\nbuilt {made} real optical z-stacks ({len(planes)} frames each) -> {OUT}")


if __name__ == "__main__":
    main()
