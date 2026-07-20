#!/usr/bin/env python3
"""Frontier 6 — exposure/WB drift: does the engine break, and does normalization fix it?

Real capture drifts brightness/white-balance across a stack (auto-exposure, flicker).
The engine assumes constant exposure. Theory: defocus blur PRESERVES the local mean,
so within a stack any frame-mean difference is exposure, not focus — per-frame
per-channel gain can be estimated cleanly (gain = target_mean / frame_mean) and is
~1.0 on undrifted stacks (auto-safe).

Probe: inject per-frame gain + WB tilt into a GT scene, measure degradation per
method, then A/B the normalization fix. Gate: normalization must be ~identity on
undrifted stacks.

Run:  python research/drift.py
"""
from __future__ import annotations
import sys
import os

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from nframe import scenes, make_stack  # noqa: E402
from focusstack.fusion import fuse_blend, fuse_perband  # noqa: E402

# per-frame (gain, per-channel WB gains BGR) — realistic auto-exposure wobble
DRIFT = [(0.88, (1.00, 1.00, 1.00)),
         (1.06, (1.05, 1.00, 0.96)),   # + slight WB tilt
         (0.95, (0.98, 1.00, 1.03)),
         (1.12, (1.00, 1.00, 1.00))]


def apply_drift(frames):
    out = []
    for f, (g, wb) in zip(frames, DRIFT):
        x = f.astype(np.float32) * g * np.array(wb, np.float32)[None, None, :]
        out.append(np.clip(x, 0, 255).astype(np.uint8))
    return out


def normalize_exposure(frames):
    """Per-frame per-channel scalar gain to the stack-median channel means."""
    means = np.array([f.reshape(-1, 3).mean(0) for f in frames])   # (N,3)
    target = np.median(means, axis=0)                              # (3,)
    out = []
    for f, m in zip(frames, means):
        gain = target / np.maximum(m, 1e-3)
        out.append(np.clip(f.astype(np.float32) * gain[None, None, :], 0, 255).astype(np.uint8))
    return out


def main():
    name, gt, depth = scenes()[0]
    frames, _ = make_stack(gt, depth, 4)
    drifted = apply_drift(frames)
    fixed = normalize_exposure(drifted)
    id_check = normalize_exposure(frames)   # gate: ~identity on undrifted

    print(f"scene={name} N=4   (GT-SSIM)")
    for label, stk in [("clean", frames), ("drifted", drifted), ("drift+norm", fixed),
                       ("clean+norm (gate)", id_check)]:
        pb = M.ref_ssim(fuse_perband(stk, harden=0.5), gt)
        bl = M.ref_ssim(fuse_blend(stk, harden=0.5), gt)
        print(f"  {label:18s} perband={pb:.4f}  blend={bl:.4f}")
    d = float(np.abs(np.stack(id_check).astype(np.int16) - np.stack(frames).astype(np.int16)).mean())
    print(f"  gate: normalize(clean) vs clean mean|diff| = {d:.3f} (want ~0)")


if __name__ == "__main__":
    main()
