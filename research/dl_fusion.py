#!/usr/bin/env python3
"""M4 — self-supervised multi-focus fusion CNN (Torch, no ground truth).

Runs on the provisioned Python 3.12 env (.venv312) since Torch has no 3.14 wheel.
A small fully-convolutional net predicts a per-pixel fusion weight w for a 2-frame
stack; fused = w*A + (1-w)*B. Trained purely self-supervised — the loss rewards
RETAINING the sharpest available detail (fused gradient >= max source gradient)
with a smooth weight map. No answer key. Rests on the project's conceptual base
(focus = high-frequency energy; fuse = take the locally sharpest content).

Evaluated against Real-MFF ground truth (dev only) and vs the classical engine.

Run:  .venv312/bin/python research/dl_fusion.py [iters]
"""
from __future__ import annotations
import glob
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
RMFF = os.path.join(HERE, "data", "realmff", "extracted", "RealMFF")
torch.manual_seed(0)
np.random.seed(0)

_SX = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32) / 4
_SY = _SX.transpose(2, 3).clone()


def gmag(x):  # x: (B,1,H,W) gray in [0,1]
    gx = F.conv2d(x, _SX, padding=1)
    gy = F.conv2d(x, _SY, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


class FusionNet(nn.Module):
    """Tiny FCN: [grayA, grayB, gA, gB] -> per-pixel weight w in [0,1]."""
    def __init__(self, ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=4, dilation=4), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, 1, 1),
        )

    def forward(self, a_gray, b_gray):
        ga, gb = gmag(a_gray), gmag(b_gray)
        return torch.sigmoid(self.net(torch.cat([a_gray, b_gray, ga, gb], 1)))


def to_gray(t):  # (B,3,H,W) -> (B,1,H,W)
    return (0.114 * t[:, 0:1] + 0.587 * t[:, 1:2] + 0.299 * t[:, 2:3])


def ss_loss(w, A, B):
    """Self-supervised: retain best gradient + smooth weight. No GT."""
    ga, gb = to_gray(A), to_gray(B)
    fused_gray = w * ga + (1 - w) * gb
    best = torch.maximum(gmag(ga), gmag(gb))
    l_grad = F.relu(best - gmag(fused_gray)).mean()               # keep sharpest detail
    l_tv = (w[:, :, 1:, :] - w[:, :, :-1, :]).abs().mean() + \
           (w[:, :, :, 1:] - w[:, :, :, :-1]).abs().mean()        # smooth map
    return l_grad + 0.02 * l_tv


def pairs(split):
    ids = sorted(os.path.basename(p)[:3] for p in glob.glob(os.path.join(RMFF, "imageA", "*_A.png")))
    tr = ids[:500]; te = ids[500:600]
    return tr if split == "train" else te


def load(idx):
    a = cv2.imread(os.path.join(RMFF, "imageA", f"{idx}_A.png"))
    b = cv2.imread(os.path.join(RMFF, "imageB", f"{idx}_B.png"))
    g = cv2.imread(os.path.join(RMFF, "Fusion", f"{idx}_F.png"))
    return a, b, g


def batch(ids, n=8, ps=128):
    A, B = [], []
    for _ in range(n):
        idx = ids[np.random.randint(len(ids))]
        a, b, _ = load(idx)
        h, w = a.shape[:2]
        y, x = np.random.randint(0, h - ps), np.random.randint(0, w - ps)
        A.append(a[y:y + ps, x:x + ps]); B.append(b[y:y + ps, x:x + ps])
    def t(imgs):
        return torch.from_numpy(np.stack(imgs).transpose(0, 3, 1, 2)).float() / 255.0
    return t(A), t(B)


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    net = FusionNet()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    tr = pairs("train")
    net.train()
    for it in range(iters):
        A, B = batch(tr)
        w = net(to_gray(A), to_gray(B))
        loss = ss_loss(w, A, B)
        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % 100 == 0:
            print(f"  iter {it+1}/{iters}  self-sup loss={loss.item():.4f}")

    # Evaluate vs GT + classical engine on held-out Real-MFF.
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(HERE, "..", "src"))  # focusstack (pure py on cv2+numpy)
    import metrics as M
    from focusstack.fusion import fuse_blend
    net.eval()
    dl_ss, cl_ss = [], []
    with torch.no_grad():
        for idx in pairs("test"):
            a, b, gt = load(idx)
            A = torch.from_numpy(a.transpose(2, 0, 1)[None]).float() / 255
            B = torch.from_numpy(b.transpose(2, 0, 1)[None]).float() / 255
            w = net(to_gray(A), to_gray(B))
            fused = (w * A + (1 - w) * B)[0].numpy().transpose(1, 2, 0) * 255
            fused = np.clip(fused, 0, 255).astype(np.uint8)
            dl_ss.append(M.ref_ssim(fused, gt))
            cl_ss.append(M.ref_ssim(fuse_blend([a, b], focus_method="content_aware", harden=0.5), gt))
    print(f"\nheld-out Real-MFF GT-SSIM ({len(dl_ss)} pairs):")
    print(f"  self-supervised CNN (no GT in training) {np.mean(dl_ss):.4f}")
    print(f"  classical (content_aware + harden)       {np.mean(cl_ss):.4f}")
    torch.save(net.state_dict(), os.path.join(HERE, "dl_fusion.pt"))
    print(f"wrote dl_fusion.pt")


if __name__ == "__main__":
    main()
