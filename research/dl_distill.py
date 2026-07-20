#!/usr/bin/env python3
"""M4 capstone — distill the classical engine into a fast CNN (speed w/o quality loss).

Self-supervised-from-scratch underperforms the mature classical engine. The way
AI-learning delivers "excellent results, faster" is DISTILLATION: train the CNN to
reproduce the classical engine's output (content_aware + harden) in a single
forward pass. The classical engine generates the training targets (not ground
truth, not a per-image answer key at deployment) — at inference the CNN runs alone.

Reports held-out GT-SSIM (CNN vs classical) and wall-clock speed.

Run:  .venv312/bin/python research/dl_distill.py [iters] [n_train_imgs]
"""
from __future__ import annotations
import glob
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
RMFF = os.path.join(HERE, "data", "realmff", "extracted", "RealMFF")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import metrics as M  # noqa: E402
from focusstack.fusion import fuse_blend  # noqa: E402

torch.manual_seed(0); np.random.seed(0)
from dl_fusion import FusionNet, to_gray, gmag  # reuse arch  # noqa: E402


def _ids(a, b):
    ids = sorted(os.path.basename(p)[:3] for p in glob.glob(os.path.join(RMFF, "imageA", "*_A.png")))
    return ids[a:b]


def _load(idx):
    return (cv2.imread(os.path.join(RMFF, "imageA", f"{idx}_A.png")),
            cv2.imread(os.path.join(RMFF, "imageB", f"{idx}_B.png")),
            cv2.imread(os.path.join(RMFF, "Fusion", f"{idx}_F.png")))


def ssim_loss(x, y):  # x,y: (B,1,H,W)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mx = F.avg_pool2d(x, 7, 1, 3); my = F.avg_pool2d(y, 7, 1, 3)
    vx = F.avg_pool2d(x * x, 7, 1, 3) - mx * mx
    vy = F.avg_pool2d(y * y, 7, 1, 3) - my * my
    vxy = F.avg_pool2d(x * y, 7, 1, 3) - mx * my
    s = ((2 * mx * my + C1) * (2 * vxy + C2)) / ((mx * mx + my * my + C1) * (vx + vy + C2) + 1e-8)
    return 1 - s.mean()


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 150

    # Precompute classical targets for the training images (once).
    print(f"precomputing classical targets for {n_train} images ...")
    cache = []
    for idx in _ids(0, n_train):
        a, b, _ = _load(idx)
        tgt = fuse_blend([a, b], focus_method="content_aware", harden=0.5)
        cache.append((a, b, tgt))

    def batch(n=8, ps=128):
        A, B, T = [], [], []
        for _ in range(n):
            a, b, t = cache[np.random.randint(len(cache))]
            h, w = a.shape[:2]
            y, x = np.random.randint(0, h - ps), np.random.randint(0, w - ps)
            A.append(a[y:y+ps, x:x+ps]); B.append(b[y:y+ps, x:x+ps]); T.append(t[y:y+ps, x:x+ps])
        f = lambda z: torch.from_numpy(np.stack(z).transpose(0, 3, 1, 2)).float() / 255
        return f(A), f(B), f(T)

    net = FusionNet(); opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    net.train()
    for it in range(iters):
        A, B, T = batch()
        w = net(to_gray(A), to_gray(B))
        fused = w * A + (1 - w) * B
        loss = F.l1_loss(fused, T) + 0.5 * ssim_loss(to_gray(fused), to_gray(T))
        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % 200 == 0:
            print(f"  iter {it+1}/{iters}  distill loss={loss.item():.4f}")

    # Eval on held-out: GT-SSIM (CNN vs classical) + speed.
    net.eval()
    dl, cl = [], []
    t_dl = t_cl = 0.0
    with torch.no_grad():
        for idx in _ids(500, 600):
            a, b, gt = _load(idx)
            A = torch.from_numpy(a.transpose(2, 0, 1)[None]).float() / 255
            B = torch.from_numpy(b.transpose(2, 0, 1)[None]).float() / 255
            t0 = time.time(); w = net(to_gray(A), to_gray(B))
            fused = np.clip((w * A + (1 - w) * B)[0].numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
            t_dl += time.time() - t0
            t1 = time.time(); clf = fuse_blend([a, b], focus_method="content_aware", harden=0.5); t_cl += time.time() - t1
            dl.append(M.ref_ssim(fused, gt)); cl.append(M.ref_ssim(clf, gt))
    print(f"\nheld-out Real-MFF GT-SSIM (100 pairs):")
    print(f"  distilled CNN (1 forward pass) {np.mean(dl):.4f}   avg {1000*t_dl/len(dl):.1f} ms/img")
    print(f"  classical engine (teacher)     {np.mean(cl):.4f}   avg {1000*t_cl/len(cl):.1f} ms/img")
    print(f"  speedup: {t_cl/t_dl:.1f}x")
    torch.save(net.state_dict(), os.path.join(HERE, "dl_distill.pt"))


if __name__ == "__main__":
    main()
