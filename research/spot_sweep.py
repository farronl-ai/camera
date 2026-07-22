#!/usr/bin/env python3
"""C3 spot sweep: shipped enhance path on real + GT data."""
import sys, os, glob
sys.path.insert(0, "/home/farron/camera/research")
import cv2, numpy as np
import metrics as M
from focusstack.fusion import fuse_perband
from focusstack.enhance import enhance
from focusstack.align import align_stack
from focusstack.io import normalize_exposure

def check(name, frames, gt=None):
    base = fuse_perband(frames, harden=0.5)
    out, rep = enhance(frames, base)
    d = float(np.abs(out.astype(np.int16) - base.astype(np.int16)).mean())
    line = f"  {name:16s} veil={rep['veil_fired']} recon={rep['recon_fired']} diff={d:.3f}"
    if gt is not None:
        line += f"  GT-SSIM {M.ref_ssim(base, gt):.4f}->{M.ref_ssim(out, gt):.4f}"
    else:
        line += f"  q_ssim {M.q_ssim(frames, base):.4f}->{M.q_ssim(frames, out):.4f}"
    print(line, flush=True)

root = "/home/farron/camera/research/data/realmff/extracted/RealMFF"
ids = sorted(os.path.basename(p)[:3] for p in glob.glob(os.path.join(root, "imageA", "*_A.png")))
rng = np.random.default_rng(11)
for i in rng.permutation(ids)[:10]:
    a, b = cv2.imread(f"{root}/imageA/{i}_A.png"), cv2.imread(f"{root}/imageB/{i}_B.png")
    gt = cv2.imread(f"{root}/Fusion/{i}_F.png")
    if a is None or b is None: continue
    check(f"realmff_{i}", [a, b], gt)
for seq in ["Figure3/kitchen", "Figure6/largemotion"]:
    fr = [cv2.imread(p) for p in sorted(glob.glob(f"/home/farron/camera/research/data/mobiledepth/{seq}/*.jpg"))]
    fr = normalize_exposure(align_stack([f for f in fr if f is not None], motion="affine"))
    check(seq.split("/")[-1], fr)
fence = [cv2.imread("/home/farron/camera/research/data/standard/c_05_1.tif"),
         cv2.imread("/home/farron/camera/research/data/standard/c_05_2.tif")]
check("fence", fence)
micro = [cv2.imread(p) for p in sorted(glob.glob("/home/farron/camera/research/data/bbbc006/mcf-z-stacks-03212011_a01_s1/frame_*.png"))]
check("microscopy", micro)
