#!/usr/bin/env python3
"""16c v0.5 (depth-evidence ribbon): both-directions check."""
import sys, os, glob
sys.path.insert(0, "/home/farron/camera/research")
import cv2, numpy as np
import metrics as M
from bband import bband
from reconstruct import occ_scenes
from focusstack.align import align_stack
from focusstack.io import normalize_exposure
from focusstack.fusion import fuse_perband, fuse_blend
from focusstack.reconstruct import reconstruct_boundaries, _estimate_matte

print("== ON-MODEL: occ benchmark (package recon, depth-evidence ribbon) ==")
for sc in occ_scenes():
    m=(sc["alpha"]>0.5).astype(np.uint8)
    gt_b=(cv2.dilate(m,np.ones((3,3),np.uint8))!=cv2.erode(m,np.ones((3,3),np.uint8))).astype(np.uint8)
    base=fuse_perband(sc["frames"],harden=0.5)
    rec=reconstruct_boundaries(sc["frames"],base,radius=sc["max_r"])
    eb,er=bband(base,sc["gt"],gt_b,2)[0],bband(rec,sc["gt"],gt_b,2)[0]
    print(f"  {sc['sid']:16s} e2 {eb:5.1f}->{er:5.1f}  glob {M.ref_ssim(base,sc['gt']):.4f}->{M.ref_ssim(rec,sc['gt']):.4f}")

print("== REAL: mobiledepth (q_ssim, alpha%, fire overlay) ==")
MD="/home/farron/camera/research/data/mobiledepth"; OUT="/home/farron/camera/research/analyze_out/mobiledepth"
for seq in ["Figure3/kitchen","Figure6/largemotion"]:
    frames=[cv2.imread(p) for p in sorted(glob.glob(os.path.join(MD,seq,"*.jpg")))]
    frames=normalize_exposure(align_stack([f for f in frames if f is not None],motion="affine"))
    radius=0.012*max(frames[0].shape[:2])
    base=fuse_perband(frames,harden=0.5)
    alpha,owner=_estimate_matte(frames,radius)
    rec=reconstruct_boundaries(frames,base,radius=radius)
    name=seq.split("/")[-1]
    ov=base.copy(); ov[alpha>0.15]=(0,0,255)
    cv2.imwrite(os.path.join(OUT,f"{name}_alphafire_v2.png"),ov)
    cv2.imwrite(os.path.join(OUT,f"{name}_perband_recon_v2.png"),rec)
    d=float(np.abs(rec.astype(np.int16)-base.astype(np.int16)).mean())
    print(f"  {name:14s} q_ssim {M.q_ssim(frames,base):.4f}->{M.q_ssim(frames,rec):.4f}  "
          f"alpha%={(alpha>0.15).mean()*100:.2f} owner={owner} diff={d:.3f}")
