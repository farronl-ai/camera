#!/usr/bin/env python3
"""16e — SEMANTIC matte for the veil correction (the input three arcs wait on).

Stage prep (main env):  python research/semalpha.py prep
    -> research/data/wideocc/{sid}_pass1.png  (perband pass-1 fusions)
Stage bridge (.venv312): .venv312/bin/python research/bridge_depth.py <pngs>
Stage eval (main env):  python research/semalpha.py eval
    -> semantic alpha from DA-V2 depth (Otsu split + owner-frame guided snap),
       D-hat from the exact error identity, weight-scaled in-loop correction;
       report vs base (O2 ceiling in F41 for reference), plus |alpha err|.
"""
from __future__ import annotations
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from hardbench import disk_blur  # noqa: E402
from wideocc_gen import scenes  # noqa: E402
from veilband import fringe_mask, fuse_perband_weighted_corr  # noqa: E402
from focusstack.fusion import fuse_perband, guided_filter  # noqa: E402
from focusstack.focus import content_aware_energies  # noqa: E402
from focusstack.io import to_gray_float  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "wideocc")


def cmd_prep():
    os.makedirs(OUT, exist_ok=True)
    for coc in (0.04, 0.012):
        for sc in scenes(coc):
            p = os.path.join(OUT, f"{sc['sid']}_pass1.png")
            cv2.imwrite(p, fuse_perband(sc["frames"], harden=0.5))
            print(f"  {p}")


def semantic_alpha(depth_npy, frames):
    """STACK-SEEDED semantic matte: focus evidence gives sparse metric seeds
    (which pixels decisively belong to which frame's plane); DA-V2 gives dense
    object-shaped depth. The matte = the semantic-depth region grown from the
    owner's seeds. A global Otsu fails here because DA-V2 correctly sees the
    background photo's INTERNAL depth (F32 pathology) — seeding fixes that.
    Returns (alpha, owner_idx)."""
    d = depth_npy.astype(np.float32)
    if d.shape != frames[0].shape[:2]:
        d = cv2.resize(d, (frames[0].shape[1], frames[0].shape[0]))
    d = (d - d.min()) / (np.ptp(d) + 1e-9)      # higher = nearer

    grays = [to_gray_float(f) for f in frames]
    E = np.stack(content_aware_energies(grays), 0)
    winner = np.argmax(E, 0)
    srt = np.sort(E, axis=0)
    decisive = ((srt[-1] - srt[-2]) / (srt[-1] + 1e-6) > 0.3) & (srt[-1] > np.median(srt[-1]))

    # owner = frame whose decisive seeds sit NEAREST in semantic depth
    n = len(frames)
    med = [float(np.median(d[decisive & (winner == k)])) if (decisive & (winner == k)).any()
           else -1 for k in range(n)]
    owner = int(np.argmax(med))
    others = [m for k, m in enumerate(med) if k != owner and m >= 0]
    if not others or med[owner] < 0:
        return np.zeros(d.shape, np.float32), 0
    thr = 0.5 * (med[owner] + max(others))      # midpoint owner-seeds vs rest

    near = (d >= thr).astype(np.uint8)
    # keep only components actually containing owner seeds (kills bg-internal
    # near-ish photo content unconnected to the occluder)
    n_comp, labels = cv2.connectedComponents(near)
    seeds = decisive & (winner == owner)
    keep = np.zeros_like(near)
    for c in range(1, n_comp):
        comp = labels == c
        if seeds[comp].mean() > 0.02:
            keep[comp] = 1
    alpha = np.clip(guided_filter(grays[owner] / 255.0, keep.astype(np.float32), 2, 1e-4), 0.0, 1.0)
    return alpha, owner


def build_D(frames, alpha, max_r, owner, far_idx):
    near_pm = frames[owner].astype(np.float32) * alpha[..., None]
    far_f = frames[far_idx].astype(np.float32)
    ab = disk_blur(alpha, 0.7 * max_r)
    pm_b = np.stack([disk_blur(near_pm[..., c], 0.7 * max_r) for c in range(3)], 2)
    return ((pm_b - near_pm) + far_f * (alpha - ab)[..., None]) * (alpha < 0.5)[..., None]


def cmd_eval():
    for coc in (0.04, 0.012):
        print(f"\n===== 16e semantic-alpha correction, CoC {coc} =====")
        rows = []
        for sc in scenes(coc):
            gt, tru_a, max_r = sc["gt"], sc["alpha"], sc["max_r"]
            npy = os.path.join(OUT, f"{sc['sid']}_pass1.png.depth.npy")
            if not os.path.exists(npy):
                print(f"  {sc['sid']}: no depth npy — run bridge")
                continue
            fr = fringe_mask(tru_a, max_r)
            base = fuse_perband(sc["frames"], harden=0.5)
            a_sem, owner = semantic_alpha(np.load(npy), sc["frames"])
            far_idx = 1 - owner
            D = build_D(sc["frames"], a_sem, max_r, owner, far_idx)
            cor = fuse_perband_weighted_corr(sc["frames"], D, 0, far_idx=far_idx)
            e0 = float(np.abs(base.astype(np.float32) - gt.astype(np.float32)).sum(2)[fr].mean())
            ec = float(np.abs(cor.astype(np.float32) - gt.astype(np.float32)).sum(2)[fr].mean())
            g0, gc = M.ref_ssim(base, gt), M.ref_ssim(cor, gt)
            aerr = float(np.abs(a_sem - tru_a).mean())
            rows.append((e0, ec, g0, gc))
            print(f"  {sc['sid']:22s} fringe {e0:5.1f}->{ec:5.1f}  glob {g0:.4f}->{gc:.4f}  |a err|={aerr:.3f} owner={owner}")
        if rows:
            a = np.array(rows)
            print(f"  MEAN fringe {a[:,0].mean():5.1f}->{a[:,1].mean():5.1f}  glob {a[:,2].mean():.4f}->{a[:,3].mean():.4f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "prep":
        cmd_prep()
    else:
        cmd_eval()
