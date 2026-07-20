#!/usr/bin/env python3
"""Controlled archetype study: which fusion tune suits which CONTENT type?

Real-MFF is near-ceiling and laplacian wins everywhere, so scene-dependence
can't manifest there. Here we build content archetypes with KNOWN ground truth
and REALISTIC depth-dependent defocus (per-pixel blur ~ |depth - focus_plane|,
so a smooth depth gradient gives a gradually-changing blur — the "bent metal"
case). Then we ask, per archetype: which focus operator / params maximize true
GT-SSIM, and does the winner CHANGE across archetypes?

If the best tune varies by content type, scene-dependence is real and content-
adaptive fusion is justified on principle (not just metric noise).

Run:  python research/synth.py
Writes research/synth_out/*.png montages + research/synth_result.json.
"""

from __future__ import annotations

import itertools
import json
import os

import cv2
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from focusstack.fusion import fuse_blend  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "synth_out")
os.makedirs(OUT, exist_ok=True)
SZ = 768


def _rng(seed):
    return np.random.default_rng(seed)


# ---- archetype base images (the all-in-focus ground truth) + depth maps ---- #
def arch_fine_detail(seed=0):
    """Dense multi-color high-frequency texture (hummingbird/foliage feathers)."""
    r = _rng(seed)
    base = r.integers(0, 256, (SZ, SZ, 3), np.uint8)
    base = cv2.GaussianBlur(base, (3, 3), 0.6)  # tiny to make it 'fine' not pure noise
    yy = np.linspace(0, 1, SZ)[:, None] * np.ones((1, SZ))  # planar tilt depth
    return base, yy.astype(np.float32)


def arch_smooth_metal(seed=1):
    """Smoothly-shaded curved surface (bent metal) with faint brushed texture +
    sparse scratches; depth is a smooth gradient -> gradual focus change."""
    r = _rng(seed)
    x = np.linspace(0, np.pi, SZ)
    shade = (0.5 + 0.45 * np.sin(x)[None, :] * np.ones((SZ, 1)))  # curved highlight
    base = (shade[..., None] * np.array([180, 190, 200])).astype(np.uint8)
    # brushed texture (fine horizontal lines) + a few scratches
    tex = (r.normal(0, 6, (SZ, SZ))).astype(np.float32)
    tex = cv2.GaussianBlur(tex, (1, 31), 0)  # anisotropic -> brushed look
    base = np.clip(base.astype(np.float32) + tex[..., None], 0, 255).astype(np.uint8)
    for _ in range(6):
        p1 = (int(r.integers(SZ)), int(r.integers(SZ)))
        p2 = (p1[0] + int(r.integers(-120, 120)), p1[1] + int(r.integers(-40, 40)))
        cv2.line(base, p1, p2, (150, 150, 160), 1, cv2.LINE_AA)
    depth = (np.linspace(0, 1, SZ)[None, :] * np.ones((SZ, 1))).astype(np.float32)  # smooth gradient
    return base, depth


def arch_hard_edges(seed=2):
    """High-contrast hard edges (text, bars) — the halo-prone regime."""
    base = np.full((SZ, SZ, 3), 235, np.uint8)
    r = _rng(seed)
    for _ in range(14):
        p1 = (int(r.integers(SZ)), int(r.integers(SZ)))
        p2 = (int(r.integers(SZ)), int(r.integers(SZ)))
        c = tuple(int(v) for v in r.integers(0, 80, 3))
        cv2.line(base, p1, p2, c, int(r.integers(1, 5)), cv2.LINE_AA)
    for i, txt in enumerate(["FOCUS", "STACK", "EDGE", "TEST"]):
        cv2.putText(base, txt, (40, 160 + i * 180), cv2.FONT_HERSHEY_SIMPLEX, 3.5,
                    (20, 20, 20), 6, cv2.LINE_AA)
    depth = np.where(np.arange(SZ)[None, :] < SZ // 2, 0.15, 0.85).astype(np.float32) * np.ones((SZ, 1))
    return base, depth


def arch_foliage(seed=3):
    """Multi-scale organic colored texture (medium-high freq)."""
    r = _rng(seed)
    base = np.zeros((SZ, SZ, 3), np.float32)
    for scale, amp in [(4, 1.0), (8, 0.6), (16, 0.35), (32, 0.2)]:
        n = r.normal(0, 1, (SZ // scale, SZ // scale, 3))
        n = cv2.resize(n, (SZ, SZ), interpolation=cv2.INTER_CUBIC)
        base += amp * n
    base = (255 * (base - base.min()) / (np.ptp(base) + 1e-9)).astype(np.uint8)
    base = cv2.applyColorMap(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_SUMMER)
    base = cv2.addWeighted(base, 0.7, arch_fine_detail(seed + 9)[0], 0.3, 0)
    yy = (np.linspace(0, 1, SZ)[:, None] * np.ones((1, SZ))).astype(np.float32)
    return base, yy


ARCHETYPES = {"fine_detail": arch_fine_detail, "smooth_metal": arch_smooth_metal,
              "hard_edges": arch_hard_edges, "foliage": arch_foliage}


def defocus(base, depth, focus, max_sigma, n_levels=10):
    sig = max_sigma * np.abs(depth - focus)
    levels = np.linspace(0.0, max_sigma, n_levels)
    blurred = [base if s < 0.3 else cv2.GaussianBlur(base, (0, 0), float(s)) for s in levels]
    idx = np.clip(np.round(sig / (max_sigma + 1e-9) * (n_levels - 1)).astype(int), 0, n_levels - 1)
    out = np.empty_like(base)
    for l in range(n_levels):
        m = idx == l
        out[m] = blurred[l][m]
    return out


def make_stack(base, depth, n_frames=4, max_sigma=5.0):
    planes = np.linspace(float(depth.min()), float(depth.max()), n_frames)
    return [defocus(base, depth, f, max_sigma) for f in planes]


OPERATORS = ["laplacian", "gradient", "tenengrad", "mod_laplacian"]
PARAM_GRID = list(itertools.product([2, 4, 8], [1e-3, 1e-2], [None, 6]))  # radius, eps, levels


def main():
    result = {}
    for name, gen in ARCHETYPES.items():
        base, depth = gen()
        frames = make_stack(base, depth)
        gt = base
        best = {}
        for op in OPERATORS:
            op_best = (-1, None, None)
            for radius, eps, levels in PARAM_GRID:
                fused = fuse_blend(frames, focus_method=op, radius=radius, eps=eps, levels=levels)
                s = M.ref_ssim(fused, gt)
                if s > op_best[0]:
                    op_best = (s, {"radius": radius, "eps": eps, "levels": levels}, fused)
            best[op] = op_best
        # rank operators by their own-best GT-SSIM
        ranked = sorted(((op, best[op][0]) for op in OPERATORS), key=lambda t: -t[1])
        winner_op, winner_ssim = ranked[0]
        result[name] = {"ranked": [(o, round(s, 4)) for o, s in ranked],
                        "winner": winner_op, "winner_params": best[winner_op][1]}
        # montage: GT | a mid frame | winner fusion
        mid = frames[len(frames) // 2]
        cv2.imwrite(os.path.join(OUT, f"{name}.png"), np.hstack([gt, mid, best[winner_op][2]]))
        print(f"{name:13s} winner={winner_op:13s} ({winner_ssim:.4f})  ranked={result[name]['ranked']}")

    json.dump(result, open(os.path.join(HERE, "synth_result.json"), "w"), indent=2)
    ops_that_win = {v["winner"] for v in result.values()}
    print(f"\ndistinct winning operators across archetypes: {ops_that_win}")
    print("=> scene-dependence is REAL" if len(ops_that_win) > 1 else
          "=> one operator wins all archetypes here (need harder content / real defocus)")
    print(f"wrote montages to {OUT}, result to synth_result.json")


if __name__ == "__main__":
    main()
