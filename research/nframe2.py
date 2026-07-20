#!/usr/bin/env python3
"""B1 follow-up — is the N-frame weight dilution HARMFUL, and does top-K fix it?

nframe.py showed weight mass on the true-sharpest frame drops with N, BUT GT-SSIM
rises with N. So the dilution may be benign (leakage onto ADJACENT near-equal
planes) rather than harmful (leakage onto DISTANT blurred planes). Discipline: don't
fix an effect whose harm isn't established. Here we:
  (1) split leakage into adjacent (|k-true|==1) vs distant (|k-true|>=2);
  (2) A/B a top-K energy gate (keep only the K highest-energy frames per pixel
      before weighting — distant blurred frames get zero) at N=8: does GT-SSIM rise?
Top-K with K>=N is identity (so N=2 is untouched by construction).

Run:  python research/nframe2.py
"""
from __future__ import annotations
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from nframe import scenes, make_stack  # noqa: E402
from focusstack.fusion import (_auto_levels, _laplacian_pyramid, _gaussian_pyramid,  # noqa: E402
                               guided_filter)
from focusstack.focus import content_aware_energies  # noqa: E402
from focusstack.io import to_gray_float  # noqa: E402


def _topk_mask(energy, k):
    """Boolean (N,H,W): True for the k highest-energy frames per pixel."""
    if k is None or k >= energy.shape[0]:
        return np.ones_like(energy, bool)
    kth = np.sort(energy, axis=0)[-k]                 # (H,W) k-th largest
    return energy >= kth[None]


def fuse_perband_topk(frames, k=None, radius=6, eps=1e-3, energy_ksize=7):
    """perband with an optional top-K energy gate per band (distant frames -> 0)."""
    floats = [f.astype(np.float32) for f in frames]
    n = len(floats)
    levels = _auto_levels(floats[0].shape, None)
    lps = [_laplacian_pyramid(im, levels) for im in floats]
    gps = [_gaussian_pyramid(to_gray_float(f), levels) for f in frames]
    bands = []
    w_last = None
    for b in range(levels + 1):
        coeffs = [lps[c][b] for c in range(n)]
        bh, bw = coeffs[0].shape[:2]
        if b < levels:
            r_b = max(1, min(radius, min(bh, bw) // 6))
            k_b = max(3, min(energy_ksize, (min(bh, bw) // 4) | 1))
            E = np.stack([cv2.boxFilter((coeffs[c] ** 2).sum(2), cv2.CV_32F, (k_b, k_b))
                          for c in range(n)], 0)
            gate = _topk_mask(E, k)
            winner = np.argmax(E, 0)
            W = []
            for c in range(n):
                raw = ((winner == c) & gate[c]).astype(np.float32)
                wg = np.clip(guided_filter(gps[c][b] / 255.0, raw, r_b, eps), 0, None) * gate[c]
                W.append(wg)
            W = np.stack(W, 0)
            W /= (W.sum(0, keepdims=True) + 1e-8)
            bands.append(sum(W[c][..., None] * coeffs[c] for c in range(n)))
            w_last = W
        else:
            wb = np.stack([cv2.pyrDown(w_last[c]) for c in range(n)], 0)
            wb = np.clip(wb, 0, None); wb /= (wb.sum(0, keepdims=True) + 1e-8)
            bands.append(sum(wb[c][..., None] * coeffs[c] for c in range(n)))
    result = bands[-1]
    for b in range(levels - 1, -1, -1):
        size = (bands[b].shape[1], bands[b].shape[0])
        result = cv2.pyrUp(result, dstsize=size) + bands[b]
    return np.clip(result, 0, 255).astype(np.uint8)


def leakage(frames, planes, depth):
    """Full-res guided-weight mass split into adjacent vs distant planes."""
    E = np.stack(content_aware_energies([to_gray_float(f) for f in frames]), 0)
    # replicate blend/decision soft weights (guided of one-hot), simplified full-res
    winner = np.argmax(E, 0)
    W = []
    for c in range(len(frames)):
        raw = (winner == c).astype(np.float32)
        W.append(np.clip(guided_filter(to_gray_float(frames[c]) / 255.0, raw, 8, 1e-3), 0, None))
    W = np.stack(W, 0); W /= (W.sum(0, keepdims=True) + 1e-8)
    true_idx = np.argmin(np.abs(planes[:, None, None] - depth[None]), 0)
    frame_ids = np.arange(len(frames))[:, None, None]
    dist = np.abs(frame_ids - true_idx[None])
    adj = float((W * (dist == 1)).sum(0).mean())
    far = float((W * (dist >= 2)).sum(0).mean())
    return adj, far


def main():
    print("N=8 leakage split + top-K gate A/B (perband GT-SSIM):")
    print(f"{'scene':15s} | {'adj-leak':>8s} {'far-leak':>8s} | {'K=2':>7s} {'K=3':>7s} {'K=all':>7s}")
    for name, gt, depth in scenes():
        frames, planes = make_stack(gt, depth, 8)
        adj, far = leakage(frames, planes, depth)
        s2 = M.ref_ssim(fuse_perband_topk(frames, 2), gt)
        s3 = M.ref_ssim(fuse_perband_topk(frames, 3), gt)
        sa = M.ref_ssim(fuse_perband_topk(frames, None), gt)
        print(f"{name:15s} | {adj:8.3f} {far:8.3f} | {s2:7.4f} {s3:7.4f} {sa:7.4f}")


if __name__ == "__main__":
    main()
