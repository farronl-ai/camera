#!/usr/bin/env python3
"""16e — mask-matte + blind veil chain on the objects-as-occluders benchmark."""
import sys, os, glob, json
sys.path.insert(0, "/home/farron/camera/research")
import cv2, numpy as np
import metrics as M
from maskmatte import mask_matte
from semalpha import build_D
from veilband import fringe_mask, fuse_perband_weighted_corr
from focusstack.fusion import fuse_perband

ROOT = "/home/farron/camera/research/data/objocc"

def scenes():
    man = json.load(open(os.path.join(ROOT, "manifest.json")))
    for m in man:
        d = os.path.join(ROOT, m["id"])
        yield dict(sid=m["id"], max_r=m["max_r"],
                   gt=cv2.imread(os.path.join(d, "gt.png")),
                   alpha=cv2.imread(os.path.join(d, "alpha.png"), 0).astype(np.float32) / 255.0,
                   frames=[cv2.imread(os.path.join(d, f"frame_{k}.png")) for k in (0, 1)],
                   dir=d)

if sys.argv[1] == "prep":
    for sc in scenes():
        cv2.imwrite(os.path.join(sc["dir"], "pass1.png"), fuse_perband(sc["frames"], harden=0.5))
        print(" ", sc["sid"], "pass1")
elif sys.argv[1] == "eval":
    rows = []
    for sc in scenes():
        p = os.path.join(sc["dir"], "pass1.png")
        masks, depth = np.load(p + ".masks.npy"), np.load(p + ".depth.npy")
        a_est, owner = mask_matte(masks, depth, sc["frames"])
        aerr = float(np.abs(a_est - sc["alpha"]).mean())
        base = fuse_perband(sc["frames"], harden=0.5)
        fr = fringe_mask(sc["alpha"], sc["max_r"])
        if a_est.max() > 0:
            D = build_D(sc["frames"], a_est, sc["max_r"], owner, 1 - owner)
            cor = fuse_perband_weighted_corr(sc["frames"], D, 0, far_idx=1 - owner)
        else:
            cor = base
        e0 = float(np.abs(base.astype(np.float32) - sc["gt"].astype(np.float32)).sum(2)[fr].mean())
        ec = float(np.abs(cor.astype(np.float32) - sc["gt"].astype(np.float32)).sum(2)[fr].mean())
        g0, gc = M.ref_ssim(base, sc["gt"]), M.ref_ssim(cor, sc["gt"])
        rows.append((e0, ec, g0, gc, aerr))
        print(f"  {sc['sid']:9s} fringe {e0:5.1f}->{ec:5.1f}  glob {g0:.4f}->{gc:.4f}  |a err|={aerr:.3f} owner={owner}")
    a = np.array(rows)
    lands = int((a[:, 4] < 0.05).sum())
    print(f"\n  MEAN fringe {a[:,0].mean():5.1f}->{a[:,1].mean():5.1f}  glob {a[:,2].mean():.4f}->{a[:,3].mean():.4f}")
    print(f"  matte lands (<0.05): {lands}/{len(rows)}")
