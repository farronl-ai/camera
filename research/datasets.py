#!/usr/bin/env python3
"""Dataset acquisition + registry for the adaptive-fusion research.

Downloads multi-focus datasets into research/data/<name>/ and records a registry
(research/data/registry.json) of pairs with resolutions. Everything here is
gitignored — we never commit third-party data.

Datasets:
  standard  — 20 color + 10 grayscale classic pairs (yuliu316316/MFIF). No GT.
  realmff   — Real-MFF: 710 pairs WITH all-in-focus ground truth (via gdown).
              Used to VALIDATE our no-reference metrics against true quality.
Run:  python research/datasets.py standard
      python research/datasets.py realmff
      python research/datasets.py registry     # (re)build registry.json
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

STANDARD_BASE = "https://raw.githubusercontent.com/yuliu316316/MFIF/master/sourceimages"
IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _get(url: str, dest: str, timeout: int = 60) -> tuple[str, bool, str]:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest, True, "cached"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "focusstack-research"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        with open(dest, "wb") as f:
            f.write(data)
        return dest, True, f"{len(data)//1024}KB"
    except Exception as e:  # noqa: BLE001
        return dest, False, str(e)[:80]


def download_standard() -> None:
    out = os.path.join(DATA, "standard")
    jobs = []
    for i in range(1, 21):
        for h in (1, 2):
            jobs.append((f"{STANDARD_BASE}/color/c_{i:02d}_{h}.tif",
                         os.path.join(out, f"c_{i:02d}_{h}.tif")))
    for i in range(1, 11):
        for h in (1, 2):
            jobs.append((f"{STANDARD_BASE}/grayscale/g_{i:02d}_{h}.tif",
                         os.path.join(out, f"g_{i:02d}_{h}.tif")))
    ok = fail = 0
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for _dest, good, msg in ex.map(lambda j: _get(*j), jobs):
            if good:
                ok += 1
            else:
                fail += 1
                print("  FAIL", _dest, msg)
    print(f"standard: {ok} ok, {fail} fail -> {out}")


def download_realmff() -> None:
    """Real-MFF is a Google-Drive zip; use gdown (pip-installed on demand)."""
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("gdown not installed; run: .venv/bin/pip install gdown")
        return
    import gdown
    import zipfile

    out = os.path.join(DATA, "realmff")
    os.makedirs(out, exist_ok=True)
    zip_path = os.path.join(out, "realmff.zip")
    file_id = "1UgV_AFmAlzZunaXmyVvoskbhbudr_SQp"
    if not (os.path.exists(zip_path) and os.path.getsize(zip_path) > 0):
        gdown.download(id=file_id, output=zip_path, quiet=False)
    if os.path.exists(zip_path):
        try:
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(out)
            print(f"realmff: extracted -> {out}")
        except zipfile.BadZipFile:
            print("realmff: downloaded file is not a valid zip (Drive quota / HTML page?)")


def build_registry() -> None:
    import cv2

    reg: dict = {}
    for name in sorted(os.listdir(DATA)):
        d = os.path.join(DATA, name)
        if not os.path.isdir(d):
            continue
        files = []
        for root, _dirs, fnames in os.walk(d):
            for fn in sorted(fnames):
                if fn.lower().endswith(IMG_EXT):
                    files.append(os.path.join(root, fn))
        if not files:
            continue
        sample = cv2.imread(files[0])
        reg[name] = {
            "count": len(files),
            "sample_shape": None if sample is None else list(sample.shape),
            "dir": d,
        }
    with open(os.path.join(DATA, "registry.json"), "w") as f:
        json.dump(reg, f, indent=2)
    for name, info in reg.items():
        print(f"  {name:12s} {info['count']:5d} files  sample={info['sample_shape']}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "standard"
    {"standard": download_standard, "realmff": download_realmff,
     "registry": build_registry}[cmd]()
    if cmd != "registry":
        build_registry()
