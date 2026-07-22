#!/usr/bin/env python3
"""FRONTIER 20 — stack-gap recovery: mild remnant-anchored deconvolution where
NO frame is sharp (focus gaps). The best mix is still blurred there (F33-class
structural limit for selection); scene recovery admits deconvolution with the
KNOWN disk PSF (depth-from-focus scale; here oracle radius from the factory).

GAP FACTORY: real-photo GT (sharp everywhere), 3 wavy depth bands
{0.15, 0.5, 0.85}, frames focused {0.15, 0.85} -> the middle band is blurred
by the SAME disk radius r_gap = 0.35*max_r in BOTH frames. r_gap kept <= 12 px
(exact-kernel regime of disk_blur) so the deconvolution PSF is the true PSF.
Gap-eval region eroded away from band boundaries (seams are boundary work,
16c's domain, not gap work).

Rungs (litscan R6-R8, one variable each; GT-credited):
  p0   factory sanity + baseline gap deficit + ORACLE rung (RL on noiseless
       blurred GT — the estimation-free ceiling; PLAYBOOK oracle-ladder idiom)
  r6   Richardson-Lucy on the fused gap, known PSF, k in {2..15} (early
       stopping IS the regularizer — Lucy 1974)
  r7   residual RL: anchor = edge-preserving base of the fused; deconvolve
       only the residual (ringing ~ signal magnitude — Yuan 2007)
  r8   Wiener one-shot (alpha=2 rung of Krishnan-Fergus), lambda sweep
  eye  crops: gap center + band boundary, GT alongside

Run: cd research && ../.venv/bin/python gapfill.py p0
Results: research/gapfill_<cmd>.json
"""
from __future__ import annotations
import glob
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metrics as M  # noqa: E402
from hardbench import disk_blur, add_noise  # noqa: E402
from focusstack.fusion import fuse_perband, guided_filter  # noqa: E402

LONG = 1536
SIGMA = 3.0
DEPTHS = (0.15, 0.5, 0.85)
FOCI = (0.15, 0.85)


def disk_kernel(radius):
    r = int(np.ceil(radius))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    k = ((xx * xx + yy * yy) <= radius * radius).astype(np.float32)
    return k / k.sum()


def scenes_gap(coc_frac=0.012, n_scenes=4, d_gap=0.5):
    """Dicts: sid, gt (sharp uint8), frames (2 noisy uint8), masks (3 feathered
    band masks), gap_eval (eroded bool), r_gap (true blur radius of the gap in
    each frame), max_r."""
    out = []
    for i, gp in enumerate(sorted(glob.glob(os.path.join(HERE, "data", "hires", "*", "gt.png")))[:n_scenes]):
        gt = cv2.imread(gp)
        h, w = gt.shape[:2]
        s = LONG / max(h, w)
        gt = cv2.resize(gt, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        hh, ww = gt.shape[:2]
        max_r = coc_frac * max(hh, ww)
        rng = np.random.default_rng(900 + i)
        # wavy horizontal band boundaries at ~1/3 and ~2/3
        yy = np.arange(hh, dtype=np.float32)[:, None] * np.ones((1, ww), np.float32)
        xs = np.arange(ww, dtype=np.float32)[None, :]
        b1 = hh / 3 + hh * 0.06 * np.sin(2 * np.pi * xs / ww * rng.uniform(1.5, 3.5) + rng.uniform(0, 6))
        b2 = 2 * hh / 3 + hh * 0.06 * np.sin(2 * np.pi * xs / ww * rng.uniform(1.5, 3.5) + rng.uniform(0, 6))
        band_depth = [DEPTHS[0], d_gap, DEPTHS[2]]
        masks = [(yy < b1).astype(np.float32),
                 ((yy >= b1) & (yy < b2)).astype(np.float32),
                 (yy >= b2).astype(np.float32)]
        masks = [cv2.GaussianBlur(m, (0, 0), 3.0) for m in masks]
        tot = masks[0] + masks[1] + masks[2]
        masks = [m / tot for m in masks]
        frames = []
        for k, f in enumerate(FOCI):
            acc = np.zeros_like(gt, np.float32)
            for m, d in zip(masks, band_depth):
                r = max_r * abs(d - f)
                acc += m[..., None] * disk_blur(gt.astype(np.float32), r)
            frames.append(add_noise(np.clip(acc, 0, 255).astype(np.uint8), SIGMA, 70 * i + k))
        gap_hard = ((yy >= b1) & (yy < b2)).astype(np.uint8)
        er = max(3, int(2 * max_r))
        gap_eval = cv2.erode(gap_hard, np.ones((er, er), np.uint8)).astype(bool)
        r_gap = max_r * abs(d_gap - FOCI[0])  # == other focus by symmetry at 0.5
        out.append(dict(sid=f"{os.path.basename(os.path.dirname(gp))}_g{d_gap:g}", gt=gt,
                        frames=frames, masks=masks, gap_eval=gap_eval,
                        r_gap=(max_r * abs(d_gap - FOCI[0]), max_r * abs(d_gap - FOCI[1])),
                        max_r=max_r))
    return out


# --------------------------------------------------------------------------- #
# deconvolution primitives (FFT RL + Wiener), numpy-only
# --------------------------------------------------------------------------- #
def _psf_otf(psf, shape):
    pad = np.zeros(shape, np.float32)
    ph, pw = psf.shape
    pad[:ph, :pw] = psf
    pad = np.roll(pad, (-(ph // 2), -(pw // 2)), axis=(0, 1))
    return np.fft.rfft2(pad)


def _conv(img, otf):
    return np.fft.irfft2(np.fft.rfft2(img) * otf, s=img.shape)


def _pad(img2d, p):
    return np.pad(img2d, ((p, p), (p, p)), mode="edge")


def rl_deconv(img, radius, iters, gain=None):
    """Richardson-Lucy, channel-wise, disk PSF (symmetric -> self-adjoint).
    Replicate-padded before the FFT (circular-wrap artifacts otherwise — the
    R8 litscan pitfall, CONFIRMED by eye before this fix). gain: optional
    per-pixel update damping (Yuan 2007)."""
    psf = disk_kernel(radius)
    p = 4 * int(np.ceil(radius))
    eps = 1e-3
    out = np.empty_like(img, np.float32)
    if gain is not None:
        gain = _pad(gain, p)
    for c in range(img.shape[2]):
        b = np.maximum(_pad(img[..., c].astype(np.float32), p), eps)
        otf = _psf_otf(psf, b.shape)
        est = b.copy()
        for _ in range(iters):
            conv = np.maximum(_conv(est, otf), eps)
            upd = _conv(b / conv, otf)
            if gain is not None:
                upd = 1.0 + gain * (upd - 1.0)
            est = est * upd
        out[..., c] = est[p:-p, p:-p]
    return out


def wiener_deconv(img, radius, lam):
    psf = disk_kernel(radius)
    p = 4 * int(np.ceil(radius))
    out = np.empty_like(img, np.float32)
    for c in range(img.shape[2]):
        b = _pad(img[..., c].astype(np.float32), p)
        K = _psf_otf(psf, b.shape)
        est = np.fft.irfft2(np.conj(K) * np.fft.rfft2(b) / (np.abs(K) ** 2 + lam), s=b.shape)
        out[..., c] = est[p:-p, p:-p]
    return out


def blend_gap(fused, est, masks):
    m = masks[1][..., None]
    return np.clip(fused.astype(np.float32) * (1 - m) + est * m, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _lstd(x, win=7):
    return np.sqrt(np.maximum(cv2.boxFilter(x * x, cv2.CV_32F, (win, win))
                              - cv2.boxFilter(x, cv2.CV_32F, (win, win)) ** 2, 0.0))


def gap_score(out, base, gt, gap_eval, masks, floor=2.0):
    go, gg = M._gray32(out), M._gray32(gt)
    sm = M._ssim_map(go, gg)
    smb = M._ssim_map(M._gray32(base), gg)
    lo, lg = _lstd(go), _lstd(gg)
    sel = gap_eval & (lg > floor)
    off = cv2.dilate(masks[1], np.ones((15, 15), np.uint8)) < 0.05
    return dict(
        gap_ssim=float(sm[gap_eval].mean()),
        dgap=float(sm[gap_eval].mean() - smb[gap_eval].mean()),
        cr=float(np.median(lo[sel] / lg[sel])) if sel.sum() > 100 else float("nan"),
        err=float(np.abs(out.astype(np.float32) - gt.astype(np.float32)).sum(2)[gap_eval].mean()),
        dg_off=float(sm[off].mean() - smb[off].mean()),
        g=float(M.ref_ssim(out, gt)),
    )


def dump(cmd, obj):
    path = os.path.join(HERE, f"gapfill_{cmd}.json")
    json.dump(obj, open(path, "w"), indent=2)
    print(f"-> {path}", flush=True)


# --------------------------------------------------------------------------- #
# rungs
# --------------------------------------------------------------------------- #
def cmd_p0():
    print("== GAP P0: factory sanity + baseline deficit + oracle RL ceiling ==", flush=True)
    rows = []
    for sc in scenes_gap():
        fused = fuse_perband(sc["frames"], harden=0.5)
        s_base = gap_score(fused, fused, sc["gt"], sc["gap_eval"], sc["masks"])
        # oracle rung: RL on the noiseless, exactly-blurred GT (estimation-free ceiling)
        r = sc["r_gap"][0]
        blurred_gt = disk_blur(sc["gt"].astype(np.float32), r)
        orc = rl_deconv(blurred_gt, r, 8)
        s_orc = gap_score(blend_gap(fused, orc, sc["masks"]), fused, sc["gt"],
                          sc["gap_eval"], sc["masks"])
        rows.append(dict(sid=sc["sid"], r_gap=r, base=s_base, oracle=s_orc))
        print(f"  {sc['sid']:22s} r_gap={r:.1f}px base: ssim={s_base['gap_ssim']:.4f} "
              f"cr={s_base['cr']:.3f} | oracle-RL8: dgap={s_orc['dgap']:+.4f} "
              f"cr={s_orc['cr']:.3f} off={s_orc['dg_off']:+.4f}", flush=True)
    heads = [r["base"]["cr"] for r in rows]
    orcs = [r["oracle"]["dgap"] for r in rows]
    verdict = ("DEFICIT REAL + ORACLE CEILING POSITIVE -> proceed"
               if np.nanmean(heads) < 0.9 and np.mean(orcs) > 0 else
               "no exploitable gap deficit or oracle fails -> STOP, log conditional negative")
    print(f"P0: mean base cr={np.nanmean(heads):.3f}, mean oracle dgap={np.mean(orcs):+.4f} -> {verdict}", flush=True)
    dump("p0", dict(rows=rows, verdict=verdict))


def cmd_r6():
    print("== GAP R6: RL on the fused gap, known PSF, iteration sweep ==", flush=True)
    pre = [(sc, fuse_perband(sc["frames"], harden=0.5)) for sc in scenes_gap()]
    results = []
    for k in (2, 4, 8, 15):
        agg = []
        for sc, fused in pre:
            est = rl_deconv(fused.astype(np.float32), sc["r_gap"][0], k)
            s = gap_score(blend_gap(fused, est, sc["masks"]), fused, sc["gt"],
                          sc["gap_eval"], sc["masks"])
            agg.append(s)
        row = dict(iters=k,
                   dgap=float(np.mean([a["dgap"] for a in agg])),
                   worst_dgap=float(np.min([a["dgap"] for a in agg])),
                   cr=float(np.nanmean([a["cr"] for a in agg])),
                   err=float(np.mean([a["err"] for a in agg])),
                   dg_off=float(np.min([a["dg_off"] for a in agg])))
        results.append(row)
        print(f"  k={k:2d}  dgap={row['dgap']:+.4f} (worst {row['worst_dgap']:+.4f}) "
              f"cr={row['cr']:.3f} err={row['err']:.2f} off={row['dg_off']:+.4f}", flush=True)
    dump("r6", dict(sweep=results))


def cmd_r7():
    print("== GAP R7: residual/gain-controlled RL (anchored deconvolution) ==", flush=True)
    pre = [(sc, fuse_perband(sc["frames"], harden=0.5)) for sc in scenes_gap()]
    # gain map from the ANCHOR's gradients (Yuan 2007): suppress updates on
    # smooth regions where RL only fits noise/ringing.
    results = []
    for k in (4, 8, 15):
        for a in (0.2, 0.5):
            agg = []
            for sc, fused in pre:
                g = M._gray32(fused)
                anchor = guided_filter(g / 255.0, g, 8, 1e-2)
                gx = cv2.Sobel(anchor, cv2.CV_32F, 1, 0)
                gy = cv2.Sobel(anchor, cv2.CV_32F, 0, 1)
                mag = np.sqrt(gx * gx + gy * gy)
                gain = (1 - a) + a * np.clip(mag / (np.percentile(mag, 90) + 1e-6), 0, 1)
                est = rl_deconv(fused.astype(np.float32), sc["r_gap"][0], k, gain=gain.astype(np.float32))
                s = gap_score(blend_gap(fused, est, sc["masks"]), fused, sc["gt"],
                              sc["gap_eval"], sc["masks"])
                agg.append(s)
            row = dict(iters=k, alpha=a,
                       dgap=float(np.mean([x["dgap"] for x in agg])),
                       worst_dgap=float(np.min([x["dgap"] for x in agg])),
                       cr=float(np.nanmean([x["cr"] for x in agg])),
                       dg_off=float(np.min([x["dg_off"] for x in agg])))
            results.append(row)
            print(f"  k={k:2d} a={a}  dgap={row['dgap']:+.4f} (worst {row['worst_dgap']:+.4f}) "
                  f"cr={row['cr']:.3f} off={row['dg_off']:+.4f}", flush=True)
    dump("r7", dict(sweep=results))


def cmd_r8():
    print("== GAP R8: Wiener one-shot (alpha=2 rung), lambda sweep ==", flush=True)
    pre = [(sc, fuse_perband(sc["frames"], harden=0.5)) for sc in scenes_gap()]
    results = []
    for lam in (1e-3, 3e-3, 1e-2, 3e-2):
        agg = []
        for sc, fused in pre:
            est = wiener_deconv(fused.astype(np.float32), sc["r_gap"][0], lam)
            s = gap_score(blend_gap(fused, est, sc["masks"]), fused, sc["gt"],
                          sc["gap_eval"], sc["masks"])
            agg.append(s)
        row = dict(lam=lam,
                   dgap=float(np.mean([a["dgap"] for a in agg])),
                   worst_dgap=float(np.min([a["dgap"] for a in agg])),
                   cr=float(np.nanmean([a["cr"] for a in agg])),
                   dg_off=float(np.min([a["dg_off"] for a in agg])))
        results.append(row)
        print(f"  lam={lam:g}  dgap={row['dgap']:+.4f} (worst {row['worst_dgap']:+.4f}) "
              f"cr={row['cr']:.3f} off={row['dg_off']:+.4f}", flush=True)
    dump("r8", dict(sweep=results))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "p0"
    fn = {"p0": cmd_p0, "r6": cmd_r6, "r7": cmd_r7, "r8": cmd_r8}.get(cmd)
    if fn is None:
        print(f"unknown subcommand: {cmd}")
        return
    fn()


if __name__ == "__main__":
    main()
