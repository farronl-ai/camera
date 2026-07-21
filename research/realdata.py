#!/usr/bin/env python3
"""Real-optical focal-stack acquisition — the DEEP / HANDHELD / PHOTOGRAPHIC regimes.

Companion to datasets.py (which handles the classic 2-frame MFIF pairs + Real-MFF).
This module fetches REAL optical multi-frame focal stacks — the data bottleneck the
project kept hitting (FRONTIER 2b/7/13; PLAYBOOK §V). Everything lands in
research/data/<name>/ and is GITIGNORED — we commit this fetcher + REAL_DATA.md,
never the third-party bytes. Because research/data/ is gitignored, downloaded data
persists across branch checkouts in this working tree, so pulling it once here
unblocks every branch that shares this checkout; separate worktrees just re-run this.

Catalog + provenance + gap-fill notes: research/REAL_DATA.md.

Datasets (see REAL_DATA.md for full detail, licenses, and which ship all-in-focus GT):
  mobiledepth  — Suwajanakorn et al. CVPR 2015. 13 handheld phone sweeps, N=12-41
                 frames, real optical defocus. NO all-in-focus GT (depth-oriented).
                 ~285MB zip -> ~551MB. VERIFIED direct download; pulled by default.
  araujo       — araujoalexandre, arXiv 2311.17846. 94 raw focus-bracketed bursts,
                 real defocus, PSEUDO all-in-focus GT (commercial software). Google
                 Drive (needs gdown). Size ~several GB.
  learn2af     — Herrmann et al. CVPR 2020 "Learning to Autofocus". 510 handheld
                 stacks (5x Pixel-3 rig), N=49 focal slices, real optical. 870 GB
                 TOTAL — NOT auto-pulled; prints wget commands. Depth/dual-pixel
                 oriented (confirm AiF GT before fidelity use).
  iphone12     — "Learn2Refocus" SIGGRAPH Asia 2025. 1637 scenes, N=9, 4K, real
                 iPhone-12 sweeps WITH Helicon-Focus all-in-focus GT. Very large;
                 project-page download only (prints link).

Run:  python research/realdata.py mobiledepth     # verified, ~285MB
      python research/realdata.py araujo           # needs: pip install gdown
      python research/realdata.py learn2af         # prints 870GB wget commands
      python research/realdata.py iphone12         # prints project-page link
      python research/realdata.py registry         # rescan research/data/
"""

from __future__ import annotations

import os
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def _download(url: str, dest: str, timeout: int = 600) -> bool:
    """Stream a URL to disk (large files -> no read-into-memory). Cached if present."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  cached: {dest}")
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "focusstack-research"})
        with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            got = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if total:
                    print(f"\r  {got/1e6:7.1f} / {total/1e6:7.1f} MB", end="", flush=True)
        print()
        os.replace(tmp, dest)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL {url}: {str(e)[:120]}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False


def download_mobiledepth() -> None:
    out = os.path.join(DATA, "mobiledepth")
    zip_path = os.path.join(out, "depth_from_focus_data2.zip")
    url = "https://www.supasorn.com/data/depth_from_focus_data2.zip"
    if not _download(url, zip_path):
        return
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out)
        os.remove(zip_path)
        seqs = []
        for root, _dirs, files in os.walk(out):
            imgs = [f for f in files if f.lower().endswith((".jpg", ".png"))]
            if imgs:
                seqs.append((os.path.relpath(root, out), len(imgs)))
        print(f"mobiledepth: {len(seqs)} sequences -> {out}")
        for name, n in sorted(seqs):
            print(f"    {name:24s} {n:3d} frames")
    except zipfile.BadZipFile:
        print("mobiledepth: downloaded file is not a valid zip.")


def download_araujo() -> None:
    """94 real focus-bracketed raw bursts w/ pseudo AiF GT. Google Drive -> gdown."""
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("gdown not installed; run: .venv/bin/pip install gdown")
        return
    import gdown

    out = os.path.join(DATA, "araujo")
    os.makedirs(out, exist_ok=True)
    dest = os.path.join(out, "focus_stacking_dataset.zip")
    file_id = "1aCskAEDjDn2V9t4R6MMLFmNZgMemHdCN"
    if not (os.path.exists(dest) and os.path.getsize(dest) > 0):
        gdown.download(id=file_id, output=dest, quiet=False)
    if os.path.exists(dest):
        try:
            with zipfile.ZipFile(dest) as z:
                z.extractall(out)
            print(f"araujo: extracted -> {out}")
        except zipfile.BadZipFile:
            print("araujo: not a valid zip (Drive quota / HTML page?). Open the link "
                  "in REAL_DATA.md manually.")


def download_learn2af() -> None:
    """510 handheld N=49 stacks, real optical. 870 GB — do NOT auto-pull; print cmds."""
    base = "https://storage.googleapis.com/cvpr2020-af-data"
    archives = [("test.tar.gz", 89), ("train1.tar.gz", 95), ("train2.tar.gz", 100),
                ("train3.tar.gz", 99), ("train4.tar.gz", 102), ("train5.tar.gz", 99),
                ("train6.tar.gz", 99), ("train7.tar.gz", 87)]
    out = os.path.join(DATA, "learn2af")
    print("Learning to Autofocus (CVPR 2020) — 870 GB TOTAL, no registration wall.")
    print(f"Public Google Cloud Storage; download only what you need into {out}/ :")
    print(f"  mkdir -p {out}")
    for name, gb in archives:
        tag = "  <- start here (test only, 89 GB)" if name == "test.tar.gz" else ""
        print(f"  wget -c {base}/{name} -P {out}   # {gb} GB{tag}")
    print("Readme PDF is in the same bucket. Confirm all-in-focus GT presence before "
          "using for GT-referenced fidelity (dataset is autofocus/dual-pixel oriented).")


def download_iphone12() -> None:
    """Learn2Refocus iPhone-12 stacks: 1637 scenes, N=9, 4K, WITH Helicon AiF GT."""
    print("Learn2Refocus / iPhone-12 focal stacks (SIGGRAPH Asia 2025).")
    print("  Project + data: https://learn2refocus.github.io")
    print("  Paper:          https://arxiv.org/abs/2512.19823")
    print("  1637 scenes x 9 frames @ 4032x3024, real optical, all-in-focus GT via")
    print("  Helicon Focus (pseudo-GT). Very large; follow the project-page download")
    print(f"  instructions and extract into {os.path.join(DATA, 'iphone12')}/.")


def build_registry() -> None:
    from datasets import build_registry as _br  # reuse the canonical scanner
    _br()


COMMANDS = {
    "mobiledepth": download_mobiledepth,
    "araujo": download_araujo,
    "learn2af": download_learn2af,
    "iphone12": download_iphone12,
    "registry": build_registry,
}

if __name__ == "__main__":
    sys.path.insert(0, HERE)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "mobiledepth"
    if cmd not in COMMANDS:
        print(f"unknown: {cmd}\nchoices: {', '.join(COMMANDS)}")
        sys.exit(1)
    COMMANDS[cmd]()
    if cmd not in ("registry", "learn2af", "iphone12"):
        build_registry()
