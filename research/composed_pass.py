#!/usr/bin/env python3
"""16f C1 — composed two-specialist pass: both locked gates on the same stacks.

Order: veil correction in-fusion (gated mask candidates -> summed fringe-clamped
D-hat per far index) THEN reconstruction post-fusion (gated C3 matte). Sets:
  objocc held (wide regime; recon gate is OOD — watched), thin held (recon
  regime; veil naturally silent), mixed NEW scenes (thin curves + object cutout
  in one scene — the both-fire test).

Stages:  prep   -> generate mixed scenes + pass1s (then run bridges on them)
         eval   -> composed run + property/coverage/additivity report
"""
from __future__ import annotations
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from occ_gen import near_layer, occ_defocus, LONG  # noqa: E402
from hires_gen import add_noise  # noqa: E402
from objocc_gen import good_object_masks  # noqa: E402
from semalpha import build_D  # noqa: E402
from t2_candidates import candidates_with_features, corr_multi  # noqa: E402
from focusstack.gates import expand  # noqa: E402
from thinocc_gate import thin_scenes, c3_features  # noqa: E402
from t2_confidence import scenes as objocc_scenes  # noqa: E402
from reconstruct import reconstruct_band, contamination_band, estimate_alpha_v3  # noqa: E402
from focusstack.fusion import fuse_perband  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MIX = os.path.join(HERE, "data", "mixedboth")


def load_gate(path):
    z = np.load(path)
    return dict(w=z["w"], mu=z["mu"], sd=z["sd"], margin=float(z["margin"]))


def predict(gate, feats):
    xn = np.hstack([(expand(feats) - gate["mu"]) / gate["sd"], [1.0]])
    return float(xn @ gate["w"])


def mixed_scenes(n=20):
    photos = sorted(glob.glob(os.path.join(HERE, "data", "hires", "*", "gt.png")))
    rng = np.random.default_rng(31)
    out = []
    made = 0
    for i in range(n * 3):
        if made >= n:
            break
        gp = photos[i % len(photos)]
        mp = gp + ".masks.npy"
        if not os.path.exists(mp):
            continue
        src = cv2.imread(gp)
        objs = good_object_masks(np.load(mp), *src.shape[:2])
        if not objs:
            continue
        mm = objs[int(rng.integers(len(objs)))]
        bg = cv2.imread(photos[(i + 4) % len(photos)])
        bh, bw = bg.shape[:2]
        s = LONG / max(bh, bw)
        bg = cv2.resize(bg, (int(bw * s), int(bh * s)), interpolation=cv2.INTER_AREA)
        hh, ww = bg.shape[:2]
        # thin layer
        near_t, a_t = near_layer(bg, seed=5000 + i)
        # object layer
        ys, xs = np.where(mm > 0)
        obj = src[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
        a_o = mm[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
        sc_f = min(0.45 * hh / obj.shape[0], 0.45 * ww / obj.shape[1], 1.5)
        obj = cv2.resize(obj, None, fx=sc_f, fy=sc_f)
        a_o = cv2.GaussianBlur(cv2.resize(a_o, None, fx=sc_f, fy=sc_f), (0, 0), 1.0)
        oh, ow = a_o.shape
        py, px = int(rng.integers(0, hh - oh)), int(rng.integers(0, ww - ow))
        near = near_t.copy()
        alpha_o = np.zeros((hh, ww), np.float32)
        alpha_o[py:py + oh, px:px + ow] = a_o
        obj_full = np.zeros((hh, ww, 3), np.float32)
        obj_full[py:py + oh, px:px + ow] = obj
        blend_o = alpha_o[..., None]
        near = obj_full * blend_o + near * (1 - blend_o)
        alpha = np.maximum(a_t, alpha_o)
        gt = (near * alpha[..., None] + bg.astype(np.float32) * (1 - alpha[..., None])).astype(np.uint8)
        max_r = 0.018 * max(hh, ww)
        frames = [add_noise(occ_defocus(bg, near, alpha, f, 0.15, 0.85, max_r), 3.0, 3 * made + k)
                  for k, f in enumerate([0.15, 0.85])]
        out.append(dict(sid=f"mix_{made:02d}", gt=gt, alpha=alpha, frames=frames,
                        max_r=max_r, dir=os.path.join(MIX, f"mix_{made:02d}")))
        made += 1
    return out


def cmd_prep():
    os.makedirs(MIX, exist_ok=True)
    for sc in mixed_scenes():
        os.makedirs(sc["dir"], exist_ok=True)
        cv2.imwrite(os.path.join(sc["dir"], "pass1.png"), fuse_perband(sc["frames"], harden=0.5))
        print(f"  {sc['sid']} pass1", flush=True)


def composed(sc, veil_gate, recon_gate, has_bridge):
    """Returns (out, veil_n, recon_n). sc needs dir/ with bridge outputs for veil."""
    veil_n = 0
    D_by_far = {}
    if has_bridge:
        cands = candidates_with_features(sc)
        for c in cands:
            if predict(veil_gate, c["feats"]) >= veil_gate["margin"]:
                D = build_D(sc["frames"], c["alpha"], sc["max_r"], c["owner"], 1 - c["owner"])
                f = 1 - c["owner"]
                D_by_far[f] = D_by_far.get(f, 0) + D
                veil_n += 1
    out = corr_multi(sc["frames"], D_by_far) if D_by_far else fuse_perband(sc["frames"], harden=0.5)

    recon_n = 0
    a3, owner = estimate_alpha_v3(sc["frames"], sc["max_r"])
    if a3.max() > 0:
        feats = c3_features(sc["frames"], a3, sc["max_r"])
        if feats is not None and predict(recon_gate, feats) >= recon_gate["margin"]:
            fr_o = [sc["frames"][owner], sc["frames"][1 - owner]]
            out = reconstruct_band(fr_o, a3, contamination_band(a3, sc["max_r"]), out, sc["max_r"])
            recon_n = 1
    return out, veil_n, recon_n


def cmd_eval():
    veil_gate = load_gate(os.path.join(HERE, "t2_gate.npz"))
    recon_gate = load_gate(os.path.join(HERE, "thinocc_gate.npz"))
    sets = []
    oo = list(objocc_scenes())
    sets.append(("objocc-held", oo[75:], True))
    sets.append(("thin-held", thin_scenes(120)[90:], False))
    mx = mixed_scenes()
    sets.append(("mixed", mx, True))
    for name, scs, has_bridge in sets:
        print(f"\n== {name} ({len(scs)} scenes, bridge={has_bridge}) ==")
        worst, vtot, rtot, both = 0.0, 0, 0, 0
        for sc in scs:
            if has_bridge and not os.path.exists(os.path.join(sc["dir"], "pass1.png.masks.npy")):
                print(f"  {sc['sid']}: bridge outputs missing, veil branch off")
                has_b = False
            else:
                has_b = has_bridge
            base = fuse_perband(sc["frames"], harden=0.5)
            g0 = M.ref_ssim(base, sc["gt"])
            out, vn, rn = composed(sc, veil_gate, recon_gate, has_b)
            gc = M.ref_ssim(out, sc["gt"]) if (vn or rn) else g0
            vtot += vn; rtot += rn; both += int(vn > 0 and rn > 0)
            if vn or rn:
                worst = min(worst, gc - g0)
                print(f"  {sc['sid']:12s} veil={vn} recon={rn}  glob {g0:.4f}->{gc:.4f}", flush=True)
        print(f"  SET: veil fires={vtot} recon fires={rtot} both-fire scenes={both}  "
              f"worst dg={worst:+.4f} (>= -0.001)")


if __name__ == "__main__":
    cmd_prep() if (len(sys.argv) > 1 and sys.argv[1] == "prep") else cmd_eval()
