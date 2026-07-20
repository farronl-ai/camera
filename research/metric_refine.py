#!/usr/bin/env python3
"""Refine the no-reference metric: can new terms fix the per-tile blindspot?

Current composite (0.3 q_abf + 0.7 q_ssim) predicts GLOBAL quality well but per
TILE agrees with GT only ~19% (F6), and is blind on smooth content (F5). Try
extra no-reference terms and re-calibrate against GT per tile:

  gradcons — per-pixel gradient consistency: how well the fused gradient matches
             the strongest source gradient (structure transfer, no GT).
  lowfreq  — SSIM of blurred-fused vs blurred-sharpest-source (a low-frequency
             fidelity term that still has signal on SMOOTH content).

Report each metric's mean per-tile Spearman vs GT-SSIM, split into SMOOTH vs
TEXTURED tiles, and the best re-calibrated composite. Honest even if smooth
content stays hard.

Run:  python research/metric_refine.py [n_scenes]
"""
from __future__ import annotations
import concurrent.futures as cf
import itertools
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from metrics import _ssim_map, _gray32, q_abf_map, q_ssim_map  # noqa: E402
from focusstack.fusion import fuse_blend  # noqa: E402
from buildset import make_scene, TUNES, RES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TILE = 96
METRICS = ["q_abf", "q_ssim", "gradcons", "lowfreq"]


def gradcons_map(sources, fused):
    gf = cv2.magnitude(cv2.Sobel(_gray32(fused), cv2.CV_32F, 1, 0), cv2.Sobel(_gray32(fused), cv2.CV_32F, 0, 1))
    gmax = np.maximum.reduce([cv2.magnitude(cv2.Sobel(_gray32(s), cv2.CV_32F, 1, 0),
                                            cv2.Sobel(_gray32(s), cv2.CV_32F, 0, 1)) for s in sources])
    return np.minimum(gf, gmax) / (np.maximum(gf, gmax) + 1e-6)


def lowfreq_map(sources, fused):
    grays = [_gray32(s) for s in sources]
    def lstd(g, k=7):
        m = cv2.boxFilter(g, cv2.CV_32F, (k, k)); m2 = cv2.boxFilter(g * g, cv2.CV_32F, (k, k))
        return np.sqrt(np.maximum(m2 - m * m, 0))
    winner = np.argmax(np.stack([lstd(g) for g in grays], 0), 0)
    hh, ww = winner.shape; yy, xx = np.indices((hh, ww))
    sharp = np.stack(sources, 0)[winner, yy, xx]
    fb = cv2.GaussianBlur(_gray32(fused), (0, 0), 3)
    sb = cv2.GaussianBlur(_gray32(sharp), (0, 0), 3)
    return _ssim_map(fb, sb)


MAP_FNS = {"q_abf": q_abf_map, "q_ssim": q_ssim_map, "gradcons": gradcons_map, "lowfreq": lowfreq_map}


def spearman(x, y):
    def rank(v):
        o = v.argsort(); r = np.empty_like(o, dtype=np.float64); r[o] = np.arange(len(v)); return r
    rx, ry = rank(x), rank(y); rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 1e-9 else 0.0


def _process(seed):
    frames, gt = make_scene(seed)
    fused = [fuse_blend(frames, **t) for t in TUNES]
    metric_tiles = {m: [] for m in METRICS}   # per candidate: (ntiles,)
    ssim_tiles = []
    contrast_tiles = []
    n = RES // TILE
    # tile contrast (smooth vs textured) from frames
    from buildset import features
    feats = features(frames, TILE)  # (ntiles, 8); col 0 = contrast
    for fu in fused:
        maps = {m: MAP_FNS[m](frames, fu) for m in METRICS}
        s = _ssim_map(_gray32(fu), _gray32(gt))
        mt = {m: [] for m in METRICS}; st = []
        for ti in range(n):
            for tj in range(n):
                sl = (slice(ti * TILE, (ti + 1) * TILE), slice(tj * TILE, (tj + 1) * TILE))
                for m in METRICS:
                    mt[m].append(maps[m][sl].mean())
                st.append(s[sl].mean())
        for m in METRICS:
            metric_tiles[m].append(mt[m])
        ssim_tiles.append(st)
    # shape per metric: (ncand, ntiles); ssim (ncand, ntiles)
    return ({m: np.array(metric_tiles[m]) for m in METRICS}, np.array(ssim_tiles), feats[:, 0])


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    per = []
    with cf.ProcessPoolExecutor(max_workers=min(16, os.cpu_count())) as ex:
        for r in ex.map(_process, range(n)):
            per.append(r)

    ntiles = (RES // TILE) ** 2
    # For each scene+tile, rank candidates by each metric vs GT-SSIM ranking.
    def per_tile_spearman(metric_key_or_fn, smooth_mask=None):
        cors = []
        for metrics_d, ssim, contrast in per:
            for t in range(ntiles):
                if smooth_mask is not None and not smooth_mask(contrast[t]):
                    continue
                mv = (metric_key_or_fn(metrics_d, t) if callable(metric_key_or_fn)
                      else metrics_d[metric_key_or_fn][:, t])
                yv = ssim[:, t]
                cors.append(spearman(mv, yv))
        return float(np.mean(cors)) if cors else 0.0

    # global z-score per metric for fair combination
    allv = {m: np.concatenate([md[m].ravel() for md, _, _ in per]) for m in METRICS}
    mu = {m: allv[m].mean() for m in METRICS}; sd = {m: allv[m].std() + 1e-9 for m in METRICS}
    def zc(md, t, w):  # weighted z-combo at tile t across candidates
        return sum(wi * (md[m][:, t] - mu[m]) / sd[m] for wi, m in zip(w, METRICS))

    print(f"metric-refinement over {n} scenes x {ntiles} tiles x {len(TUNES)} candidates")
    smooth = lambda c: c < 6.0
    textured = lambda c: c >= 6.0
    print(f"\n{'metric':10s} {'all':>7s} {'smooth':>7s} {'textured':>9s}  (mean per-tile Spearman vs GT)")
    for m in METRICS:
        print(f"  {m:8s} {per_tile_spearman(m):7.3f} {per_tile_spearman(m, smooth):7.3f} {per_tile_spearman(m, textured):9.3f}")
    # current composite (0.3 q_abf + 0.7 q_ssim)
    cur = lambda md, t: 0.3 * md['q_abf'][:, t] + 0.7 * md['q_ssim'][:, t]
    print(f"  {'CURRENT':8s} {per_tile_spearman(cur):7.3f} {per_tile_spearman(cur, smooth):7.3f} {per_tile_spearman(cur, textured):9.3f}")

    # Content-routed metric: lowfreq on smooth tiles, q_ssim+gradcons on textured
    # (the metric-level analogue of the content_aware operator).
    routed_cors = []
    for md, ss, contrast in per:
        for t in range(ntiles):
            mv = (md['lowfreq'][:, t] if contrast[t] < 6.0
                  else 0.75 * md['q_ssim'][:, t] + 0.25 * md['gradcons'][:, t])
            routed_cors.append(spearman(mv, ss[:, t]))
    print(f"  {'ROUTED':8s} {np.mean(routed_cors):7.3f}  (lowfreq on smooth, q_ssim+gradcons on textured)")

    # calibrate best nonneg simplex over all 4 metrics
    best = (-1, None)
    for w in itertools.product(np.arange(0, 1.01, 0.25), repeat=len(METRICS)):
        if abs(sum(w) - 1) > 1e-6:
            continue
        sc = per_tile_spearman(lambda md, t, w=w: zc(md, t, w))
        if sc > best[0]:
            best = (sc, w)
    print(f"\nbest refined composite weights {dict(zip(METRICS, [round(x,2) for x in best[1]]))}")
    print(f"  all-tile per-tile Spearman = {best[0]:.3f}  (vs current {per_tile_spearman(cur):.3f})")


if __name__ == "__main__":
    main()
