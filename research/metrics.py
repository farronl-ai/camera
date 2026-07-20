#!/usr/bin/env python3
"""No-reference (ground-truth-free) fusion-quality metrics.

The whole autonomous program optimizes against these, so they must be trustworthy
*without* an answer key. Each takes the source frames + the fused image and scores
how well the fusion preserved the sources' information — no reference needed.

Implemented (numpy + opencv only):
  q_abf   — Xydeas & Petrovic gradient-transfer metric. The canonical MFIF score:
            how much of each source's EDGE information (strength + orientation)
            survived into the fused image. Higher = better. In [0, 1].
  q_mi    — normalized mutual information between fused and sources (shared info).
  q_ssim  — structural similarity of the fused image to whichever source is
            locally sharper (rewards taking the in-focus content).
  sharp   — plain no-reference sharpness (mean |Laplacian| energy). A sanity/《more
            is better》signal, but NOT trusted alone (it rewards halos/speckle).

`composite()` combines the trustworthy ones; weights are calibrated in M0 against
Real-MFF ground truth (see validate_metrics.py).

For GT-referenced validation only (dev-time, not used at inference):
  ref_ssim, ref_psnr — compare a fused image to a true all-in-focus reference.
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np


def _gray32(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)


# --------------------------------------------------------------------------- #
# Q^AB/F — gradient transfer (Xydeas & Petrovic, 2000)
# --------------------------------------------------------------------------- #
def _grad_strength_orientation(gray: np.ndarray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    g = np.sqrt(gx * gx + gy * gy)
    a = np.arctan2(gy, gx)
    return g, a


def _q_pair(g_a, a_a, g_f, a_f):
    """Edge preservation of source A in fused F (per-pixel, in [0,1])."""
    # Relative strength: how well the fused edge magnitude matches the source's.
    eps = 1e-10
    gg = np.minimum(g_a, g_f) / (np.maximum(g_a, g_f) + eps)
    # Orientation agreement.
    da = np.abs(np.abs(a_a - a_f) - np.pi / 2) / (np.pi / 2)
    # Sigmoids (Xydeas-Petrovic constants).
    Gamma_g, k_g, sigma_g = 0.9994, -15.0, 0.5
    Gamma_a, k_a, sigma_a = 0.9879, -22.0, 0.8
    Qg = Gamma_g / (1.0 + np.exp(k_g * (gg - sigma_g)))
    Qa = Gamma_a / (1.0 + np.exp(k_a * (da - sigma_a)))
    return Qg * Qa


def _q_abf_num_den(sources, fused):
    """Return (per-pixel Q*strength sum, per-pixel strength sum). Shared by map/scalar."""
    gf, af = _grad_strength_orientation(_gray32(fused))
    num = np.zeros_like(gf)
    den = np.zeros_like(gf)
    for src in sources:
        gs, as_ = _grad_strength_orientation(_gray32(src))
        num += _q_pair(gs, as_, gf, af) * gs
        den += gs
    return num, den


def q_abf_map(sources: list[np.ndarray], fused: np.ndarray) -> np.ndarray:
    """Per-pixel gradient-transfer quality in [0,1] (edge-strength weighted)."""
    num, den = _q_abf_num_den(sources, fused)
    return num / (den + 1e-10)


def q_abf(sources: list[np.ndarray], fused: np.ndarray) -> float:
    """Weighted gradient-transfer quality; weight by each source's edge strength."""
    num, den = _q_abf_num_den(sources, fused)
    return float(num.sum() / (den.sum() + 1e-10))


def q_abf_ms(sources: list[np.ndarray], fused: np.ndarray,
             levels: int = 4, pool: str = "mean") -> float:
    """Multi-scale Q_ABF — the perband lesson transplanted into the metric.

    Plain Q_ABF uses a fixed 3x3 Sobel, so it only sees the finest scale and
    collapses at high resolution (F17), where the meaningful structure lives at
    coarser scales. Here gradient transfer is evaluated at each pyramid level and
    combined: pool='sum' accumulates num/den across levels (weights levels by
    their total edge strength — fine levels dominate by pixel count); pool='mean'
    averages the per-level scores (equal say per scale). Which pooling is right is
    an empirical question — validate against GT per regime before trusting.
    """
    f = _gray32(fused)
    ss = [_gray32(s) for s in sources]
    nums, dens, scores = [], [], []
    for _ in range(max(1, levels)):
        num, den = _q_abf_num_den(ss, f)
        n, d = float(num.sum()), float(den.sum())
        nums.append(n); dens.append(d)
        scores.append(n / (d + 1e-10))
        if min(f.shape) < 32:
            break
        f = cv2.pyrDown(f)
        ss = [cv2.pyrDown(s) for s in ss]
    if pool == "sum":
        return float(sum(nums) / (sum(dens) + 1e-10))
    return float(np.mean(scores))


# --------------------------------------------------------------------------- #
# Normalized mutual information
# --------------------------------------------------------------------------- #
def _entropy(hist: np.ndarray) -> float:
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _mi(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    a = a.ravel()
    b = b.ravel()
    h2, _, _ = np.histogram2d(a, b, bins=bins)
    ha = _entropy(h2.sum(axis=1))
    hb = _entropy(h2.sum(axis=0))
    hab = _entropy(h2)
    return ha + hb - hab


def q_mi(sources: list[np.ndarray], fused: np.ndarray) -> float:
    """Normalized MI: shared information between fused and sources."""
    f = _gray32(fused)
    total = 0.0
    for src in sources:
        s = _gray32(src)
        mi = _mi(s, f)
        hs = _entropy(np.histogram(s.ravel(), bins=64)[0])
        hf = _entropy(np.histogram(f.ravel(), bins=64)[0])
        total += 2.0 * mi / (hs + hf + 1e-10)
    return total / len(sources)


# --------------------------------------------------------------------------- #
# SSIM-to-sharper-source
# --------------------------------------------------------------------------- #
def _ssim_map(x: np.ndarray, y: np.ndarray, win: int = 7):
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = (win, win)
    mu_x = cv2.boxFilter(x, cv2.CV_32F, k)
    mu_y = cv2.boxFilter(y, cv2.CV_32F, k)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sx = cv2.boxFilter(x * x, cv2.CV_32F, k) - mu_x2
    sy = cv2.boxFilter(y * y, cv2.CV_32F, k) - mu_y2
    sxy = cv2.boxFilter(x * y, cv2.CV_32F, k) - mu_xy
    return ((2 * mu_xy + C1) * (2 * sxy + C2)) / ((mu_x2 + mu_y2 + C1) * (sx + sy + C2) + 1e-12)


def q_ssim_map(sources: list[np.ndarray], fused: np.ndarray) -> np.ndarray:
    """Per-pixel SSIM of fused to the locally-sharpest source."""
    f = _gray32(fused)
    grays = [_gray32(s) for s in sources]
    sharp = [cv2.boxFilter(np.abs(cv2.Laplacian(g, cv2.CV_32F)), cv2.CV_32F, (9, 9)) for g in grays]
    winner = np.argmax(np.stack(sharp, 0), axis=0)
    ssim_stack = np.stack([_ssim_map(f, g) for g in grays], 0)
    hh, ww = winner.shape
    yy, xx = np.indices((hh, ww))
    return ssim_stack[winner, yy, xx]


def q_ssim(sources: list[np.ndarray], fused: np.ndarray) -> float:
    """SSIM of fused to the locally-sharpest source (rewards taking in-focus content)."""
    return float(q_ssim_map(sources, fused).mean())


# --------------------------------------------------------------------------- #
# Plain sharpness (sanity signal only — not trusted alone)
# --------------------------------------------------------------------------- #
def sharp(_sources, fused: np.ndarray) -> float:
    return float(np.abs(cv2.Laplacian(_gray32(fused), cv2.CV_32F)).mean())


# --------------------------------------------------------------------------- #
# Composite (weights calibrated in M0) + GT-referenced (dev validation only)
# --------------------------------------------------------------------------- #
ALL_METRICS = {"q_abf": q_abf, "q_mi": q_mi, "q_ssim": q_ssim, "sharp": sharp}

# Composite weights. Calibrated against Real-MFF ground truth by
# validate_metrics.py (mean per-pair Spearman vs GT-SSIM = +0.723): q_mi was
# zeroed (it anti-correlates with true quality), q_ssim dominates, q_abf adds
# halo/edge sensitivity. Loaded from metric_weights.json if present.
COMPOSITE_WEIGHTS = {"q_abf": 0.3, "q_ssim": 0.7}

_wpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metric_weights.json")
if os.path.exists(_wpath):
    try:
        with open(_wpath) as _f:
            _cal = json.load(_f).get("weights", {})
        _cal = {k: v for k, v in _cal.items() if v > 0}  # drop zeroed metrics
        if _cal:
            COMPOSITE_WEIGHTS = _cal
    except Exception:  # noqa: BLE001
        pass


def composite(sources: list[np.ndarray], fused: np.ndarray,
              weights: dict | None = None) -> float:
    w = weights or COMPOSITE_WEIGHTS
    return sum(coef * ALL_METRICS[name](sources, fused) for name, coef in w.items())


def all_scores(sources: list[np.ndarray], fused: np.ndarray) -> dict:
    return {name: fn(sources, fused) for name, fn in ALL_METRICS.items()}


_MAP_FNS = {"q_abf": q_abf_map, "q_ssim": q_ssim_map}


def composite_map(sources: list[np.ndarray], fused: np.ndarray,
                  weights: dict | None = None) -> np.ndarray:
    """Per-pixel calibrated composite quality (only map-able metrics; weights renormalized).

    Used by the region-adaptive engine to choose, per pixel, which differently-tuned
    candidate fusion is locally best — no ground truth needed.
    """
    w = {k: v for k, v in (weights or COMPOSITE_WEIGHTS).items() if k in _MAP_FNS}
    tot = sum(w.values()) + 1e-12
    out = None
    for name, coef in w.items():
        m = _MAP_FNS[name](sources, fused) * (coef / tot)
        out = m if out is None else out + m
    return out


def ref_ssim(fused: np.ndarray, reference: np.ndarray) -> float:
    return float(_ssim_map(_gray32(fused), _gray32(reference)).mean())


def ref_psnr(fused: np.ndarray, reference: np.ndarray) -> float:
    mse = float(np.mean((fused.astype(np.float32) - reference.astype(np.float32)) ** 2))
    return 100.0 if mse < 1e-9 else 10.0 * np.log10((255.0 ** 2) / mse)
