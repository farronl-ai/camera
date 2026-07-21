# REAL_DATA — real-optical focal-stack datasets (the data-bottleneck catalog)

The recurring bottleneck (FRONTIER 2b/7/13, PLAYBOOK §V): almost everything was
2-frame and/or synthetic. This catalog is the vetted set of **real optical, multi-frame**
focal stacks, with exact provenance so any session/branch can pull what it needs.

**How data lives here.** `research/data/` is gitignored — we commit the fetcher
(`research/realdata.py`) + this catalog, never third-party bytes. Because the dir is
gitignored, downloaded data **persists across branch checkouts in this working tree**,
so pulling once unblocks every branch sharing this checkout. Separate worktrees re-run
the fetcher. Fetch: `python research/realdata.py <name>` (see each entry).

Sourced via the deep-research sweep (2026-07-20); dataset facts verified against the
primary papers/pages. Two adversarial-verify refutations were themselves wrong (cheap
verifier over-refuted precise numbers) and are corrected below.

## Status legend
✅ in-tree now · ⬇️ scripted, run to pull · 📄 documented (manual/huge) · ❌ unavailable

## The catalog (ranked by how well it fills the real-data gaps)

| Dataset | Real optical? | N/scene | Res | AiF GT? | Size | Access | Fills |
|---|---|---|---|---|---|---|---|
| **mobiledepth** ✅ | yes (phone sweep) | 12–41 | 1280×720 | **no** (depth) | 551MB | verified direct zip | handheld/deep/align |
| **iphone12** (Learn2Refocus) 📄 | yes (iPhone 12) | 9 | 4K 4032×3024 | yes (Helicon pseudo) | very large | project page | photographic/deep/GT |
| **learn2af** (Learning to Autofocus) 📄 | yes (5×Pixel-3 rig) | **49** | 1512×2016 | confirm | **870 GB** | public GCS, no wall | deep/handheld/align |
| **araujo** ⬇️ | yes (focus bracketing) | multi | high-res raw | yes (pseudo) | ~few GB | Google Drive (gdown) | photographic/macro-ish/GT |
| **ddff12** 📄 | yes (Lytro ILLUM LF) | 10 (of 9×9) | 383×552 | depth GT | h5 | TUM server | deep(low-res)/GT |
| MFI-WHU 📄 | **no** (synthetic Gaussian) | 2 | — | — | small | GitHub `.rar` | — (easy-synthetic; skip) |
| UHD-MFF ❌ | yes (4K, 1950 pairs) | 2 | 3840×2160 | — | — | no public DL confirmed | (opaque; author-request) |
| MFFW ❌ | yes | 2 | — | — | — | ResearchGate link unconfirmed | (unavailable) |

## Entries

### mobiledepth — ✅ IN-TREE (`research/data/mobiledepth/`)
Suwajanakorn, Hernandez, Seitz, "Depth from Focus with Your Mobile Phone," CVPR 2015.
13 handheld phone focal sweeps, **real optical defocus**, deep stacks:
keyboard 32 · bucket 32 · kitchen 12 · bottles 41 · fruits 30 · metals 33 · plants 30 ·
telephone 33 · window 27 · largemotion 14 · smallmotion 14 · zeromotion 14 · balls 25.
Per-frame focal depths in `calibrated.txt`; depth output in `depth_var.bin`.
**No all-in-focus fused GT** (depth-from-focus dataset) — use for align-robustness and
real-defocus fusion checks, not GT-referenced fidelity. The `Figure6/*motion` sequences
are a ready-made alignment stress set (gap 3/7 / FRONTIER 7).
Pull: `python research/realdata.py mobiledepth` (verified HTTP 200, 285MB zip).

### iphone12 (Learn2Refocus) — 📄 the best photographic+GT option
SIGGRAPH Asia 2025. 1637 scenes (1474 train / 163 test), **N=9** frames, real iPhone-12
focus sweeps at **4032×3024 (4K)** with all-in-focus GT via Helicon Focus (depth mode).
Real focus breathing + handheld misalignment. GT is *pseudo* (commercial software), not
optical truth. Very large — download from the project page:
- https://learn2refocus.github.io · paper https://arxiv.org/abs/2512.19823
Pull instructions: `python research/realdata.py iphone12`.

### learn2af (Learning to Autofocus) — 📄 deepest real stacks, but 870 GB
Herrmann et al., CVPR 2020. 510 stacks (460 train / 50 test) over 51 scenes, captured
with **five Pixel-3 devices in a cross pattern**, **N=49** focal slices sampled in
inverse depth 0.102–3.91 m; raw dual-pixel 1512×2016. Real optical. Public Google Cloud
Storage, **no registration**. **870 GB total** (test alone 89 GB) — never auto-pulled.
`python research/realdata.py learn2af` prints per-archive `wget -c` commands (start with
`test.tar.gz`). Autofocus/dual-pixel oriented — **confirm AiF GT presence** before using
for fidelity eval; otherwise pair with iphone12 for GT.
*(Correction: the deep-research verify stage refuted the N=49 / 5-phone / 510-stack
figures 0–3; re-fetch of the CVPR paper confirms all three are correct.)*

### araujo — ⬇️ scripted (needs gdown)
araujoalexandre, "Towards Real-World Focus Stacking with Deep Learning," arXiv 2311.17846.
94 high-resolution **real** focus-bracketed raw bursts; **pseudo** all-in-focus GT from
commercial software. Google Drive.
Pull: `.venv/bin/pip install gdown && python research/realdata.py araujo`
(Drive id `1aCskAEDjDn2V9t4R6MMLFmNZgMemHdCN`; if quota-blocked, open the repo link and
download manually: https://github.com/araujoalexandre/FocusStackingDataset).

### ddff12 — 📄 real Lytro light-field (low-res)
Hazirbas et al., "Deep Depth From Focus," arXiv 1704.01085. **Real Lytro ILLUM**
light-field capture, 720 lightfield images / 12 scenes, focal stacks digitally refocused
from the LF, 9×9×**383×552**, depth GT. Low-res and LF-derived — marginal given Lytro is
already covered, but a real-optical cross-check.
- https://cvg.cit.tum.de/data/datasets/ddff12scene (`.h5` files; >10MB, direct)
*(Correction: verify stage refuted "real optical Lytro ILLUM" 0–3; TUM page confirms it
IS real Lytro ILLUM capture.)*

### Not usable / unavailable
- **MFI-WHU** (github.com/HaoZhang1018/MFI-WHU): 120 pairs but **synthetic Gaussian
  blur** on public source images — the easy-synthetic regime PLAYBOOK warns against.
- **UHD-MFF** (arXiv 2606.31242): 4K, 1950 pairs — paper exists, but no public download
  confirmed (GitHub-hosting claim refuted 0–3). Author-request only; re-check later.
- **MFFW**: ResearchGate distribution unconfirmed (refuted 1–2). Treat as unavailable.
- **Macro/museum/product** photography: still no public dataset (CU-Museum imagery is
  institution-only, PMC9836466). Genuine gap → FRONTIER 13 (first-party capture).

## Honest caveats
- All available all-in-focus GT is **pseudo** (Helicon / commercial software), not
  optical truth — fidelity numbers are tool-dependent (per PLAYBOOK 9b, treat as GT with
  that asterisk).
- Frame counts for mobiledepth are exact (counted in-tree); iphone12/learn2af from
  papers; confirm on extract.
- The macro/photographic real-optical gap is **narrowed, not closed**: iphone12 covers
  everyday photographic content with GT; true macro/product remains open.
