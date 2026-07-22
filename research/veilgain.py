#!/usr/bin/env python3
"""FRONTIER 19 — hybrid veil recovery: 16d subtraction + clamped multiplicative
contrast restoration (the F27 division revisit, inside the restoration system).

Subtraction removes the additive haze but leaves surviving background detail
amplitude-scaled (~(1-ab^2), measured in P0); only a gain restores it. The gain
is clamped (R1: t0, omega), analytically shrunk (R4: sigma=3 known, gain field
ours) or remnant-guided-denoised (R2/R3: guide = the far frame's own band),
fringe-clamped, and rides w_far inside the perband loop (F40 idiom).

Rungs (oracle alpha, owner=0/far=1 known, giant CoC first, 8-bit first):
  p0    headroom + noise calibration + sanity + byte-identity (kill switch)
  h1    clamped in-loop gain sweep (t0 x omega x g_law), no denoising
  h1a   full-image clamped correction control (pre-fusion placement test)
  h2    analytic shrink-after-gain sweep (T = m*(1+ab)*sigma*c_k*(G-1))
  h3    guided denoise of the correction (guide = far frame's gray band)
  final winning stack: 10 backgrounds x coc {0.04, 0.012} x {8bit, float}
  eye   max-disagreement + clamp-edge crops, GT alongside

Run: cd research && ../.venv/bin/python veilgain.py p0
Results: research/veilgain_<cmd>.json
"""
from __future__ import annotations
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metrics as M  # noqa: E402
from hardbench import disk_blur  # noqa: E402
from semalpha import build_D  # noqa: E402
from veilband import fringe_mask  # noqa: E402
from wideocc_gen import scenes  # noqa: E402
from focusstack.fusion import (fuse_perband, guided_filter, _auto_levels,  # noqa: E402
                               _laplacian_pyramid, _gaussian_pyramid)
from focusstack.io import to_gray_float  # noqa: E402

SIGMA = 3.0          # factory noise std — known, not estimated
OWNER, FAR = 0, 1    # focus planes [0.15, 0.85]: frame 0 owns the near occluder


# --------------------------------------------------------------------------- #
# gain laws: attenuation g(ab) that the residual gain 1/g must invert
# --------------------------------------------------------------------------- #
def glaw_lin(ab):
    return 1.0 - ab


def glaw_sq(ab):
    return 1.0 - ab ** 2


def glaw_empirical(curve):
    """Interpolator over P0's measured post-subtraction attenuation per ab-bin."""
    centers = np.asarray(curve["ab"], np.float32)
    ratios = np.asarray(curve["ratio"], np.float32)

    def g(ab):
        return np.interp(ab, centers, ratios).astype(np.float32)
    return g


G_LAWS = {"lin": glaw_lin, "sq": glaw_sq}  # "emp" added after p0 runs


# --------------------------------------------------------------------------- #
# noise calibration: per-band std of unit white noise through the pyramid
# --------------------------------------------------------------------------- #
def calibrate_band_noise(shape, levels, n_seeds=5):
    """c_k such that image noise sigma appears in band k with std sigma*c_k."""
    cks = []
    for s in range(n_seeds):
        x = np.random.default_rng(1000 + s).normal(0, 1, shape).astype(np.float32)
        lp = _laplacian_pyramid(x, levels)
        cks.append([float(b.std()) for b in lp])
    cks = np.array(cks)
    spread = float(np.max(np.ptp(cks, axis=0) / (cks.mean(axis=0) + 1e-9)))
    return cks.mean(axis=0).tolist(), spread


def _soft(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


# --------------------------------------------------------------------------- #
# the hybrid operator (fork of t2_candidates.corr_multi + gain slot)
# --------------------------------------------------------------------------- #
CA = 0.04  # factory chromatic offset (occ_gen.occ_defocus offs = (-ca, 0, ca))


def build_D_ca(frames, alpha, max_r, owner, far_idx):
    """Per-channel D matching the factory's chromatic render: the near layer's
    blur radius in the far frame is |near_d-(focus+off_c)|*max_r per channel
    ({0.66,0.70,0.74}*max_r). A channel-shared D leaves a purple/green mottle
    residual in the fringe (user-caught over-extension); per-channel D models
    it out. Returns (D, ab3, pm3) — ab3/pm3 per-channel for the gain."""
    near_pm = frames[owner].astype(np.float32) * alpha[..., None]
    far_f = frames[far_idx].astype(np.float32)
    D = np.empty_like(far_f)
    ab3 = np.empty_like(far_f)
    pm3 = np.empty_like(far_f)
    for c, off in enumerate((-CA, 0.0, CA)):
        r = abs(0.15 - (0.85 + off)) * max_r     # per-channel: {0.66,0.70,0.74}*max_r
        ab3[..., c] = disk_blur(alpha, r)
        pm3[..., c] = disk_blur(near_pm[..., c], r)
        D[..., c] = (pm3[..., c] - near_pm[..., c]) + far_f[..., c] * (alpha - ab3[..., c])
    ab_g = ab3[..., 1]
    band = ((ab_g > 0.02) & (ab_g < 0.98) & (alpha < 0.5)).astype(np.float32)
    band = cv2.GaussianBlur(band, (0, 0), 2.0)
    return D * band[..., None], ab3, pm3


def build_pm(frames, alpha, max_r, owner):
    """Blurred near-premult pm_b — the occluder's own texture bleed. The
    subtraction remnant is ab*pm_b + (1-ab^2)*far_true (+noise); without
    removing the first term the gain re-amplifies OCCLUDER texture into the
    background fringe (H6, user-caught over-extension artifact)."""
    near_pm = frames[owner].astype(np.float32) * alpha[..., None]
    return np.stack([disk_blur(near_pm[..., c], 0.7 * max_r) for c in range(3)], 2)


def fuse_perband_gain(images, D_by_far, ab, alpha, t0=0.25, omega=1.0,
                      g_law=glaw_lin, shrink_m=0.0, guide_radius=0, guide_beta=4.0,
                      ck=None, radius=6, eps=1e-3, energy_ksize=7, harden=0.5,
                      wmode="scaled", coherence=False, pm_by_far=None,
                      return_float=False):
    """perband + w_far-scaled in-loop D subtraction + clamped residual gain.

    omega=0 -> pure 16d subtraction (corr_multi-equivalent).
    D_by_far empty/None -> byte-identical to fuse_perband(harden=harden).
    Gain: G_k = 1 + omega*(1/max(g(ab_k), t0) - 1), applied to the far frame's
    post-subtraction band detail, masked to build_D's exact fringe-clamp band
    (identity off-band), never on the base band (DC shifts).
    shrink_m>0: soft-threshold the correction at m*(1+ab)*SIGMA*c_k*(G-1)
    (analytic — D carries the same far-frame noise scaled by ab, so the
    corrected term's noise std is (1+ab)*sigma*c_k before the gain).
    guide_radius>0: guided-filter the correction, guide = far frame's own gray
    Laplacian band (the remnant evidence), eps = beta*(sigma*c_k)^2.
    """
    floats = [img.astype(np.float32) for img in images]
    n = len(floats)
    levels = _auto_levels(floats[0].shape, None)
    ip = [_laplacian_pyramid(im, levels) for im in floats]
    gp = [_gaussian_pyramid(to_gray_float(f), levels) for f in images]
    dp = {f: _laplacian_pyramid(D.astype(np.float32), levels) for f, D in (D_by_far or {}).items()}

    pp = {f: _laplacian_pyramid(P.astype(np.float32), levels)
          for f, P in (pm_by_far or {}).items()}
    gain_on = bool(dp) and omega > 0.0
    if gain_on:
        # ab may be 2D (shared) or HxWx3 (per-channel, chromatic model);
        # normalize to 3-channel so the gain math is uniform.
        ab32 = ab.astype(np.float32)
        if ab32.ndim == 2:
            ab32 = np.stack([ab32] * 3, axis=-1)
        ab_g = ab32[..., 1]
        band = ((ab_g > 0.02) & (ab_g < 0.98) & (alpha < 0.5)).astype(np.float32)
        band = cv2.GaussianBlur(band, (0, 0), 2.0)
        ab_pyr = _gaussian_pyramid(ab32, levels)
        m_pyr = _gaussian_pyramid(band, levels)
        far_gray = [_laplacian_pyramid(to_gray_float(images[f]), levels) for f in dp]

    fused_bands, w_last = [], None
    for bandk in range(levels + 1):
        coeffs = [ip[k][bandk] for k in range(n)]
        bh, bw = coeffs[0].shape[:2]
        if bandk < levels:
            r_b = max(1, min(radius, min(bh, bw) // 6))
            k_b = max(3, min(energy_ksize, (min(bh, bw) // 4) | 1))
            energy = np.stack([cv2.boxFilter((coeffs[k] ** 2).sum(axis=2), cv2.CV_32F, (k_b, k_b))
                               for k in range(n)], axis=0)
            winner = np.argmax(energy, axis=0)
            srt = np.sort(energy, axis=0)
            conf = np.clip((srt[-1] - srt[-2]) / (srt[-1] + 1e-6), 0.0, 1.0) if harden > 0 else None
            weights = []
            for k in range(n):
                raw = (winner == k).astype(np.float32)
                wg = np.clip(guided_filter(gp[k][bandk] / 255.0, raw, r_b, eps), 0.0, None)
                if conf is not None:
                    wg = (1.0 - conf) * wg + conf * raw
                weights.append(wg)
            w = np.stack(weights, axis=0)
            w /= (w.sum(axis=0, keepdims=True) + 1e-8)
            fb = sum(w[k][..., None] * coeffs[k] for k in range(n))
            for gi, (fidx, pyr) in enumerate(dp.items()):
                fb = fb - w[fidx][..., None] * pyr[bandk]
                if gain_on:
                    g = np.maximum(g_law(ab_pyr[bandk]), t0)          # HxWx3
                    G = 1.0 + omega * (1.0 / g - 1.0)
                    if wmode == "deficit":
                        # output-deficit form: deficit vs GT is (1 - w_far*(1/G));
                        # estimate true detail as remnant*G -> coef = G - w_far.
                        coef = np.maximum(G - w[fidx][..., None], 0.0)
                    else:
                        # F40-style: restore only the far frame's own contribution
                        coef = w[fidx][..., None] * (G - 1.0)
                    remnant = ip[fidx][bandk] - pyr[bandk]
                    if fidx in pp:
                        remnant = remnant - ab_pyr[bandk] * pp[fidx][bandk]
                    corr = coef * remnant
                    if shrink_m > 0.0:
                        t = shrink_m * (1.0 + ab_pyr[bandk]) * SIGMA * ck[bandk] * coef
                        corr = _soft(corr, t)
                    if guide_radius > 0:
                        ge = guide_beta * (SIGMA * ck[bandk]) ** 2
                        guide = far_gray[gi][bandk]
                        for c in range(corr.shape[2]):
                            corr[..., c] = guided_filter(guide, corr[..., c], guide_radius, ge)
                    if coherence and bandk == 0:
                        # H4: real edges carry energy across scales; speckle is
                        # finest-band-isolated. Attenuate finest-band correction
                        # where the mid-band correction is below its own analytic
                        # noise floor (ratio gate, no tuned threshold).
                        g1 = np.maximum(g_law(ab_pyr[1]), t0)
                        G1 = 1.0 + omega * (1.0 / g1 - 1.0)
                        w1 = cv2.pyrDown(w[fidx])[..., None]
                        coef1 = (np.maximum(G1 - w1, 0.0)
                                 if wmode == "deficit" else w1 * (G1 - 1.0))
                        corr1 = coef1 * (ip[fidx][1] - pyr[1])
                        mag1 = cv2.boxFilter(np.abs(corr1).sum(axis=2), cv2.CV_32F, (5, 5))
                        floor1 = (3.0 * (1.0 + ab_pyr[1]) * SIGMA * ck[1] * coef1).mean(axis=2)
                        wc = mag1 / (mag1 + floor1 + 1e-6)
                        wc = cv2.resize(wc, (corr.shape[1], corr.shape[0]))
                        corr = corr * wc[..., None]
                    fb = fb + m_pyr[bandk][..., None] * corr
            fused_bands.append(fb)
            w_last = w
        else:
            wb = np.stack([cv2.pyrDown(w_last[k]) for k in range(n)], axis=0)
            wb = np.clip(wb, 0.0, None)
            wb /= (wb.sum(axis=0, keepdims=True) + 1e-8)
            fb = sum(wb[k][..., None] * coeffs[k] for k in range(n))
            for fidx, pyr in dp.items():
                fb = fb - wb[fidx][..., None] * pyr[levels]     # base: subtraction only
            fused_bands.append(fb)
    result = fused_bands[-1]
    for bandk in range(levels - 1, -1, -1):
        result = cv2.pyrUp(result, dstsize=(fused_bands[bandk].shape[1], fused_bands[bandk].shape[0])) + fused_bands[bandk]
    result = np.clip(result, 0, 255)
    return result.astype(np.float32) if return_float else result.astype(np.uint8)


def correct_full_image(images, D, ab, alpha, t0=0.25, omega=1.0, g_law=glaw_lin):
    """H1a control: same clamp math applied to the far FRAME pre-fusion (the
    deveil/F27 placement) — full-band gain (DC included) and the fusion
    weights then see the amplified noise. Expected worse than in-loop."""
    band = ((ab > 0.02) & (ab < 0.98) & (alpha < 0.5)).astype(np.float32)
    band = cv2.GaussianBlur(band, (0, 0), 2.0)
    g = np.maximum(g_law(ab.astype(np.float32)), t0)
    gm1 = omega * (1.0 / g - 1.0) * band
    far_corr = images[FAR].astype(np.float32) - D
    far_corr = np.clip(far_corr * (1.0 + gm1[..., None]), 0, 255).astype(images[FAR].dtype)
    frames = list(images)
    frames[FAR] = far_corr
    return fuse_perband(frames, harden=0.5)


# --------------------------------------------------------------------------- #
# metrics (new, GT-credited)
# --------------------------------------------------------------------------- #
def _lstd(band_img, win=7):
    e = cv2.boxFilter(band_img ** 2, cv2.CV_32F, (win, win))
    return np.sqrt(np.maximum(e, 0.0))


def contrast_ratio(out, gt, ab, alpha, levels, floor=1.0):
    """Per detail band: median lstd(out)/lstd(gt) over the STRONG veil band
    ((ab in (0.5,0.95)) & (alpha<0.5)). Amplitude-linear: 2x attenuation reads 0.5."""
    po = _laplacian_pyramid(M._gray32(out), levels)
    pg = _laplacian_pyramid(M._gray32(gt), levels)
    strong = ((ab > 0.5) & (ab < 0.95) & (alpha < 0.5)).astype(np.float32)
    rows = []
    for k in range(levels):
        m = cv2.resize(strong, (po[k].shape[1], po[k].shape[0])) > 0.5
        so, sg = _lstd(po[k]), _lstd(pg[k])
        sel = m & (sg > floor)
        rows.append(float(np.median(so[sel] / sg[sel])) if sel.sum() > 50 else float("nan"))
    return rows


AB_EDGES = np.arange(0.05, 1.0, 0.1)


def contrast_by_abbin(out, gt, ab, alpha, levels, bands=(1, 2, 3), floor=1.0, min_px=20):
    """Empirical attenuation per ab-bin (keyed by INTEGER bin index): median
    lstd(out)/lstd(gt) over mid bands + pixel count. Feeds glaw_empirical."""
    po = _laplacian_pyramid(M._gray32(out), levels)
    pg = _laplacian_pyramid(M._gray32(gt), levels)
    bins = {}
    for i in range(len(AB_EDGES) - 1):
        vals = []
        for k in bands:
            abk = cv2.resize(ab.astype(np.float32), (po[k].shape[1], po[k].shape[0]))
            alk = cv2.resize(alpha.astype(np.float32), (po[k].shape[1], po[k].shape[0]))
            so, sg = _lstd(po[k]), _lstd(pg[k])
            sel = (sg > floor) & (alk < 0.5) & (abk >= AB_EDGES[i]) & (abk < AB_EDGES[i + 1])
            if sel.sum() >= min_px:
                vals.append(so[sel] / sg[sel])
        if vals:
            v = np.concatenate(vals)
            bins[i] = dict(ab=float((AB_EDGES[i] + AB_EDGES[i + 1]) / 2),
                           med=float(np.median(v)), n=int(v.size))
    return bins


def offband_harm(out, base, gt, alpha, max_r):
    """SSIM delta vs baseline OUTSIDE the (dilated) fringe — leakage detector."""
    fr = fringe_mask(alpha, max_r)
    ksz = 2 * max(1, int(max_r / 4)) + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    off = cv2.dilate(fr.astype(np.uint8), ker) == 0
    g = M._gray32(gt)
    so = M._ssim_map(M._gray32(out), g)[off].mean()
    sb = M._ssim_map(M._gray32(base), g)[off].mean()
    return float(so - sb)


def score(out, base, gt, alpha, max_r, ab, levels, g_base=None):
    fr = fringe_mask(alpha, max_r)
    g = M.ref_ssim(out, gt)
    return dict(
        g=g,
        dg=g - (g_base if g_base is not None else M.ref_ssim(base, gt)),
        fringe=float(np.abs(out.astype(np.float32) - gt.astype(np.float32)).sum(2)[fr].mean()),
        dg_off=offband_harm(out, base, gt, alpha, max_r),
        cr=contrast_ratio(out, gt, ab, alpha, levels),
    )


def cr_mid(crows):
    v = [c for c in crows[1:4] if not np.isnan(c)]
    return float(np.mean(v)) if v else float("nan")


# --------------------------------------------------------------------------- #
# scene prep shared by all rungs
# --------------------------------------------------------------------------- #
def prep(sc):
    ab = disk_blur(sc["alpha"], 0.7 * sc["max_r"])
    D = build_D(sc["frames"], sc["alpha"], sc["max_r"], OWNER, FAR)
    levels = _auto_levels(sc["gt"].shape, None)
    base = fuse_perband(sc["frames"], harden=0.5)
    sub = fuse_perband_gain(sc["frames"], {FAR: D}, ab, sc["alpha"], omega=0.0)
    return ab, D, levels, base, sub


def dump(cmd, obj):
    path = os.path.join(HERE, f"veilgain_{cmd}.json")
    json.dump(obj, open(path, "w"), indent=2)
    print(f"-> {path}", flush=True)


def load_emp():
    p0 = json.load(open(os.path.join(HERE, "veilgain_p0.json")))
    return glaw_empirical(p0["emp_curve"])


# --------------------------------------------------------------------------- #
# rungs
# --------------------------------------------------------------------------- #
def cmd_p0():
    print("== P0: calibration + identity + headroom (coc=0.04, 8-bit, oracle alpha) ==", flush=True)
    out = {}
    scs = scenes(0.04)
    levels = _auto_levels(scs[0]["gt"].shape, None)
    ck, spread = calibrate_band_noise(scs[0]["gt"].shape[:2], levels)
    print(f"c_k (noise std per band, unit input): {['%.3f' % c for c in ck]}  seed-spread {spread * 100:.1f}%", flush=True)
    out["ck"] = ck
    out["ck_spread"] = spread

    from t2_candidates import corr_multi
    rows, curves = [], []
    for sc in scs:
        ab, D, lv, base, sub = prep(sc)
        # identity 1: no-D path == fuse_perband
        ident1 = np.array_equal(fuse_perband_gain(sc["frames"], {}, ab, sc["alpha"]), base)
        # identity 2: omega=0 == corr_multi (pure 16d subtraction)
        ident2 = np.array_equal(sub, corr_multi(sc["frames"], {FAR: D}))
        # metric sanity: GT vs GT
        gtgt = cr_mid(contrast_ratio(sc["gt"], sc["gt"], ab, sc["alpha"], lv))
        cr_b = contrast_ratio(base, sc["gt"], ab, sc["alpha"], lv)
        cr_s = contrast_ratio(sub, sc["gt"], ab, sc["alpha"], lv)
        curves.append(contrast_by_abbin(sub, sc["gt"], ab, sc["alpha"], lv))
        s_sub = score(sub, base, sc["gt"], sc["alpha"], sc["max_r"], ab, lv)
        rows.append(dict(sid=sc["sid"], ident_noD=bool(ident1), ident_sub=bool(ident2),
                         gtgt=gtgt, cr_base=cr_b, cr_sub=cr_s, dg_sub=s_sub["dg"],
                         fringe_sub=s_sub["fringe"]))
        print(f"  {sc['sid']:26s} idA={ident1} idB={ident2} gt/gt={gtgt:.3f} "
              f"cr_base(mid)={cr_mid(cr_b):.3f} cr_sub(mid)={cr_mid(cr_s):.3f} dg_sub={s_sub['dg']:+.4f}", flush=True)

    # aggregate empirical attenuation curve (post-subtraction) over scenes:
    # per bin index, pixel-count-weighted mean of scene medians
    emp = {"ab": [], "ratio": []}
    for i in sorted({i for cv in curves for i in cv}):
        entries = [cv[i] for cv in curves if i in cv]
        wsum = sum(e["n"] for e in entries)
        emp["ab"].append(entries[0]["ab"])
        emp["ratio"].append(float(sum(e["med"] * e["n"] for e in entries) / wsum))
    out["emp_curve"] = emp
    out["rows"] = rows
    print("empirical post-subtraction attenuation g(ab):", flush=True)
    for a, r in zip(emp["ab"], emp["ratio"]):
        print(f"   ab={a:.2f}  ratio={r:.3f}", flush=True)
    heads = [cr_mid(r["cr_sub"]) for r in rows]
    verdict = "HEADROOM EXISTS -> proceed to H1" if np.nanmean(heads) < 0.9 else \
        "NO HEADROOM (ratio >= 0.9): premise fails on this factory -> STOP, log conditional negative"
    out["verdict"] = verdict
    print(f"P0 verdict: mean cr_sub(mid) = {np.nanmean(heads):.3f} -> {verdict}", flush=True)
    dump("p0", out)


def cmd_h1():
    print("== H1: clamped in-loop gain sweep (coc=0.04, 8-bit, oracle alpha) ==", flush=True)
    laws = dict(G_LAWS)
    try:
        laws["emp"] = load_emp()
    except FileNotFoundError:
        print("  (no p0 json — run p0 first for the empirical law)", flush=True)
    pre = []
    for sc in scenes(0.04):
        ab, D, lv, base, sub = prep(sc)
        g_base = M.ref_ssim(base, sc["gt"])
        s_sub = score(sub, base, sc["gt"], sc["alpha"], sc["max_r"], ab, lv, g_base=g_base)
        pre.append((sc, ab, D, lv, base, sub, g_base, s_sub))
        print(f"  prep {sc['sid']:26s} dg_sub={s_sub['dg']:+.4f} cr_sub={cr_mid(s_sub['cr']):.3f}", flush=True)
    results = []
    for lname, law in laws.items():
        for t0 in (0.10, 0.15, 0.25, 0.40):
            for om in (0.9, 0.95, 1.0):
                agg = []
                for sc, ab, D, lv, base, sub, g_base, s_sub in pre:
                    outp = fuse_perband_gain(sc["frames"], {FAR: D}, ab, sc["alpha"],
                                             t0=t0, omega=om, g_law=law)
                    s = score(outp, base, sc["gt"], sc["alpha"], sc["max_r"], ab, lv, g_base=g_base)
                    s["dg_vs_sub"] = s["g"] - s_sub["g"]
                    s["fr_vs_sub"] = s["fringe"] - s_sub["fringe"]
                    agg.append(s)
                row = dict(law=lname, t0=t0, omega=om,
                           cr_mid=float(np.nanmean([cr_mid(a["cr"]) for a in agg])),
                           dg=float(np.mean([a["dg"] for a in agg])),
                           dg_vs_sub=float(np.mean([a["dg_vs_sub"] for a in agg])),
                           dg_off=float(np.mean([a["dg_off"] for a in agg])),
                           worst_dg_off=float(np.min([a["dg_off"] for a in agg])),
                           fringe=float(np.mean([a["fringe"] for a in agg])),
                           fr_vs_sub=float(np.mean([a["fr_vs_sub"] for a in agg])))
                results.append(row)
                print(f"  {lname:4s} t0={t0:.2f} om={om:.2f}  cr_mid={row['cr_mid']:.3f} "
                      f"dg_vs_sub={row['dg_vs_sub']:+.4f} dg_off={row['dg_off']:+.4f} "
                      f"(worst {row['worst_dg_off']:+.4f}) dfr={row['fr_vs_sub']:+.2f}", flush=True)
    # baseline reference row
    sub_cr = float(np.nanmean([cr_mid(s_sub["cr"]) for *_, s_sub in pre]))
    print(f"  reference: subtraction-alone cr_mid={sub_cr:.3f}", flush=True)
    dump("h1", dict(sub_cr_mid=sub_cr, sweep=results))


def cmd_h1b():
    """Deficit-mode sweep: coef = G - w_far (invert the OUTPUT deficit incl.
    w_far dilution) vs H1's w_far*(G-1) which under-corrects where w_far<1 —
    the mechanism behind H1's 0.90 contrast plateau."""
    print("== H1b: output-deficit gain (coef = G - w_far) ==", flush=True)
    pre = []
    for sc in scenes(0.04):
        ab, D, lv, base, sub = prep(sc)
        g_base = M.ref_ssim(base, sc["gt"])
        s_sub = score(sub, base, sc["gt"], sc["alpha"], sc["max_r"], ab, lv, g_base=g_base)
        pre.append((sc, ab, D, lv, base, sub, g_base, s_sub))
    results = []
    for lname in ("lin", "sq"):
        law = G_LAWS[lname]
        for t0 in (0.05, 0.10, 0.25):
            for om in (0.95, 1.0):
                agg = []
                for sc, ab, D, lv, base, sub, g_base, s_sub in pre:
                    outp = fuse_perband_gain(sc["frames"], {FAR: D}, ab, sc["alpha"],
                                             t0=t0, omega=om, g_law=law, wmode="deficit")
                    s = score(outp, base, sc["gt"], sc["alpha"], sc["max_r"], ab, lv, g_base=g_base)
                    s["dg_vs_sub"] = s["g"] - s_sub["g"]
                    s["fr_vs_sub"] = s["fringe"] - s_sub["fringe"]
                    agg.append(s)
                row = dict(law=lname, t0=t0, omega=om, wmode="deficit",
                           cr_mid=float(np.nanmean([cr_mid(a["cr"]) for a in agg])),
                           dg_vs_sub=float(np.mean([a["dg_vs_sub"] for a in agg])),
                           dg_off=float(np.mean([a["dg_off"] for a in agg])),
                           worst_dg_off=float(np.min([a["dg_off"] for a in agg])),
                           fringe=float(np.mean([a["fringe"] for a in agg])),
                           fr_vs_sub=float(np.mean([a["fr_vs_sub"] for a in agg])),
                           cr_bands=[float(np.nanmean([a["cr"][k] for a in agg]))
                                     for k in range(len(agg[0]["cr"]))])
                results.append(row)
                print(f"  {lname:4s} t0={t0:.2f} om={om:.2f}  cr_mid={row['cr_mid']:.3f} "
                      f"dg_vs_sub={row['dg_vs_sub']:+.4f} dg_off={row['dg_off']:+.4f} "
                      f"(worst {row['worst_dg_off']:+.4f}) dfr={row['fr_vs_sub']:+.2f} "
                      f"bands={['%.2f' % b for b in row['cr_bands']]}", flush=True)
    dump("h1b", dict(sweep=results))


def cmd_h1a():
    print("== H1a control: full-image clamped gain pre-fusion (F27 placement) ==", flush=True)
    h1 = json.load(open(os.path.join(HERE, "veilgain_h1.json")))
    ok = [r for r in h1["sweep"] if r["worst_dg_off"] > -5e-4 and r["dg_vs_sub"] >= -5e-4]
    win = max(ok if ok else h1["sweep"], key=lambda r: r["cr_mid"])
    law = load_emp() if win["law"] == "emp" else G_LAWS[win["law"]]
    print(f"  using H1 winner: {win['law']} t0={win['t0']} omega={win['omega']}", flush=True)
    rows = []
    for sc in scenes(0.04):
        ab, D, lv, base, sub = prep(sc)
        g_base = M.ref_ssim(base, sc["gt"])
        s_sub = score(sub, base, sc["gt"], sc["alpha"], sc["max_r"], ab, lv, g_base=g_base)
        inl = fuse_perband_gain(sc["frames"], {FAR: D}, ab, sc["alpha"],
                                t0=win["t0"], omega=win["omega"], g_law=law)
        ful = correct_full_image(sc["frames"], D, ab, sc["alpha"],
                                 t0=win["t0"], omega=win["omega"], g_law=law)
        s_in = score(inl, base, sc["gt"], sc["alpha"], sc["max_r"], ab, lv, g_base=g_base)
        s_fu = score(ful, base, sc["gt"], sc["alpha"], sc["max_r"], ab, lv, g_base=g_base)
        rows.append(dict(sid=sc["sid"],
                         inloop=dict(dg=s_in["dg"], cr=cr_mid(s_in["cr"]), fringe=s_in["fringe"]),
                         full=dict(dg=s_fu["dg"], cr=cr_mid(s_fu["cr"]), fringe=s_fu["fringe"]),
                         sub=dict(dg=s_sub["dg"], cr=cr_mid(s_sub["cr"]), fringe=s_sub["fringe"])))
        print(f"  {sc['sid']:26s} dg in-loop {s_in['dg']:+.4f} vs full {s_fu['dg']:+.4f} "
              f"(sub {s_sub['dg']:+.4f}) | cr {cr_mid(s_in['cr']):.3f} vs {cr_mid(s_fu['cr']):.3f}", flush=True)
    dump("h1a", dict(winner=win, rows=rows))


def cmd_h2h3(which):
    """Denoising rung on the best AMPLITUDE-CALIBRATED config: winner = closest
    cr_mid to 1.0 (overshoot is as wrong as undershoot) among no-off-band-harm
    rows, pooled over h1 (scaled) + h1b (deficit)."""
    print(f"== {which.upper()}: denoising rung on the amplitude-calibrated winner ==", flush=True)
    pool = []
    for name in ("h1", "h1b"):
        try:
            sw = json.load(open(os.path.join(HERE, f"veilgain_{name}.json")))["sweep"]
            for r in sw:
                r.setdefault("wmode", "scaled")
            pool += sw
        except FileNotFoundError:
            pass
    ok = [r for r in pool if r["worst_dg_off"] > -5e-4]
    win = min(ok if ok else pool, key=lambda r: abs(r["cr_mid"] - 1.0))
    print(f"  winner: {win['wmode']}/{win['law']} t0={win['t0']} omega={win['omega']} "
          f"cr={win['cr_mid']:.3f} dg_vs_sub={win['dg_vs_sub']:+.4f}", flush=True)
    law = load_emp() if win["law"] == "emp" else G_LAWS[win["law"]]
    scs = scenes(0.04)
    levels = _auto_levels(scs[0]["gt"].shape, None)
    ck, _ = calibrate_band_noise(scs[0]["gt"].shape[:2], levels)
    pre = [(sc, *prep(sc)) for sc in scs]
    grid = [dict(shrink_m=m) for m in (1.0, 2.0, 3.0)] if which == "h2" else \
        [dict(guide_radius=r, guide_beta=b) for r in (2, 4, 8) for b in (1.0, 4.0, 16.0)]
    results = []
    pre2 = []
    for sc, ab, D, lv, base, sub in pre:
        g_base = M.ref_ssim(base, sc["gt"])
        s_sub = score(sub, base, sc["gt"], sc["alpha"], sc["max_r"], ab, lv, g_base=g_base)
        pre2.append((sc, ab, D, lv, base, sub, g_base, s_sub))
    for cfg in grid:
        agg = []
        for sc, ab, D, lv, base, sub, g_base, s_sub in pre2:
            outp = fuse_perband_gain(sc["frames"], {FAR: D}, ab, sc["alpha"],
                                     t0=win["t0"], omega=win["omega"], g_law=law, ck=ck,
                                     wmode=win["wmode"], **cfg)
            s = score(outp, base, sc["gt"], sc["alpha"], sc["max_r"], ab, lv, g_base=g_base)
            s["dg_vs_sub"] = s["g"] - s_sub["g"]
            s["fr_vs_sub"] = s["fringe"] - s_sub["fringe"]
            agg.append(s)
        row = dict(cfg=cfg,
                   cr_mid=float(np.nanmean([cr_mid(a["cr"]) for a in agg])),
                   dg_vs_sub=float(np.mean([a["dg_vs_sub"] for a in agg])),
                   dg_off=float(np.mean([a["dg_off"] for a in agg])),
                   worst_dg_off=float(np.min([a["dg_off"] for a in agg])),
                   fringe=float(np.mean([a["fringe"] for a in agg])),
                   fr_vs_sub=float(np.mean([a["fr_vs_sub"] for a in agg])),
                   cr_bands=[float(np.nanmean([a["cr"][k] for a in agg]))
                             for k in range(len(agg[0]["cr"]))])
        results.append(row)
        print(f"  {cfg}  cr_mid={row['cr_mid']:.3f} dg_vs_sub={row['dg_vs_sub']:+.4f} "
              f"dg_off={row['dg_off']:+.4f} (worst {row['worst_dg_off']:+.4f}) "
              f"dfr={row['fr_vs_sub']:+.2f}", flush=True)
    dump(which, dict(winner=win, sweep=results))


WINNER = dict(t0=0.05, omega=1.0, wmode="deficit")  # law=sq; shrink_m from FINAL


def cmd_final():
    """Winning stack (deficit/sq + analytic shrink m in {1,2}) on ALL 10
    backgrounds x coc {0.04 primary, 0.012 off-regime} x {8bit, float}."""
    print("== FINAL: 10 backgrounds x coc {0.04,0.012} x {8bit,float} ==", flush=True)
    results = {"provenance": {"backgrounds": sorted(os.path.basename(os.path.dirname(p))
                                                    for p in __import__("glob").glob(os.path.join(HERE, "data", "hires", "*", "gt.png"))),
                              "factory": "wideocc_gen.scenes (blob occluder + real-photo texture, disk-PSF, noise sigma=3)",
                              "alpha": "oracle (factory GT)", "config": dict(WINNER, law="sq")},
               "rows": []}
    ck = None
    for coc in (0.04, 0.012):
        for fl in (False, True):
            tag = f"coc{coc:g}/{'float' if fl else '8bit'}"
            for sc in scenes(coc, n_scenes=10, float_frames=fl):
                ab, D, lv, base, sub = prep(sc)
                if ck is None:
                    ck, _ = calibrate_band_noise(sc["gt"].shape[:2], lv)
                g_base = M.ref_ssim(base, sc["gt"])
                s_sub = score(sub, base, sc["gt"], sc["alpha"], sc["max_r"], ab, lv, g_base=g_base)
                row = dict(cond=tag, sid=sc["sid"],
                           sub=dict(dg=s_sub["dg"], cr=cr_mid(s_sub["cr"]), fringe=s_sub["fringe"]))
                for m in (1.0, 2.0):
                    outp = fuse_perband_gain(sc["frames"], {FAR: D}, ab, sc["alpha"],
                                             g_law=glaw_sq, shrink_m=m, ck=ck,
                                             return_float=fl, **WINNER)
                    s = score(outp, base, sc["gt"], sc["alpha"], sc["max_r"], ab, lv, g_base=g_base)
                    row[f"m{m:g}"] = dict(dg=s["dg"], dg_vs_sub=s["g"] - s_sub["g"],
                                          cr=cr_mid(s["cr"]), fringe=s["fringe"],
                                          dfr=s["fringe"] - s_sub["fringe"], dg_off=s["dg_off"])
                results["rows"].append(row)
                print(f"  {tag:14s} {sc['sid']:22s} sub dg={s_sub['dg']:+.4f} cr={cr_mid(s_sub['cr']):.3f} | "
                      f"m1 dgs={row['m1']['dg_vs_sub']:+.4f} cr={row['m1']['cr']:.3f} off={row['m1']['dg_off']:+.4f} | "
                      f"m2 dgs={row['m2']['dg_vs_sub']:+.4f} cr={row['m2']['cr']:.3f} off={row['m2']['dg_off']:+.4f}", flush=True)
    # aggregates per condition
    print("\n-- aggregates --", flush=True)
    agg = {}
    for cond in sorted({r["cond"] for r in results["rows"]}):
        rows = [r for r in results["rows"] if r["cond"] == cond]
        a = {}
        for m in ("m1", "m2"):
            a[m] = dict(dg_vs_sub=float(np.mean([r[m]["dg_vs_sub"] for r in rows])),
                        worst_dg_vs_sub=float(np.min([r[m]["dg_vs_sub"] for r in rows])),
                        cr=float(np.nanmean([r[m]["cr"] for r in rows])),
                        worst_dg_off=float(np.min([r[m]["dg_off"] for r in rows])),
                        dfr=float(np.mean([r[m]["dfr"] for r in rows])))
            print(f"  {cond:14s} {m}: dg_vs_sub={a[m]['dg_vs_sub']:+.4f} "
                  f"(worst {a[m]['worst_dg_vs_sub']:+.4f}) cr={a[m]['cr']:.3f} "
                  f"worst_off={a[m]['worst_dg_off']:+.4f} dfr={a[m]['dfr']:+.2f}", flush=True)
        a["sub_cr"] = float(np.nanmean([r["sub"]["cr"] for r in rows]))
        agg[cond] = a
    results["agg"] = agg
    dump("final", results)


def cmd_h5():
    """H5 decomposition: is the 8-bit wall INPUT quantization (frames) or
    OUTPUT quantization (uint8 cast of a sub-step correction)? 8-bit inputs,
    float outputs for BOTH sub and hybrid -> if the losing scenes recover,
    the wall is output-side (ours to remove: emit 16-bit), and R5 bin
    projection on inputs is unnecessary."""
    print("== H5: 8-bit inputs, FLOAT outputs (output-quantization isolation) ==", flush=True)
    rows = []
    ck = None
    for coc in (0.04, 0.012):
        for sc in scenes(coc, n_scenes=10):
            ab = disk_blur(sc["alpha"], 0.7 * sc["max_r"])
            D = build_D(sc["frames"], sc["alpha"], sc["max_r"], OWNER, FAR)
            lv = _auto_levels(sc["gt"].shape, None)
            if ck is None:
                ck, _ = calibrate_band_noise(sc["gt"].shape[:2], lv)
            base = fuse_perband(sc["frames"], harden=0.5)
            sub_f = fuse_perband_gain(sc["frames"], {FAR: D}, ab, sc["alpha"], omega=0.0,
                                      return_float=True)
            g_sub = M.ref_ssim(sub_f, sc["gt"])
            row = dict(coc=coc, sid=sc["sid"])
            for m in (1.0, 2.0):
                hyb = fuse_perband_gain(sc["frames"], {FAR: D}, ab, sc["alpha"],
                                        g_law=glaw_sq, shrink_m=m, ck=ck,
                                        return_float=True, **WINNER)
                s = score(hyb, base, sc["gt"], sc["alpha"], sc["max_r"], ab, lv)
                row[f"m{m:g}"] = dict(dg_vs_sub=float(M.ref_ssim(hyb, sc["gt"]) - g_sub),
                                      dg_off=s["dg_off"], cr=cr_mid(s["cr"]))
            rows.append(row)
            print(f"  coc{coc:g} {sc['sid']:22s} m1 dgs={row['m1']['dg_vs_sub']:+.4f} "
                  f"m2 dgs={row['m2']['dg_vs_sub']:+.4f} off={row['m2']['dg_off']:+.4f}", flush=True)
    for coc in (0.04, 0.012):
        rr = [r for r in rows if r["coc"] == coc]
        for m in ("m1", "m2"):
            print(f"  agg coc{coc:g} {m}: mean={np.mean([r[m]['dg_vs_sub'] for r in rr]):+.4f} "
                  f"worst={np.min([r[m]['dg_vs_sub'] for r in rr]):+.4f}", flush=True)
    dump("h5", dict(rows=rows))


def cmd_h4():
    """H4 (eye-triggered): cross-scale coherence gate on the finest-band
    correction. Sparse speckle seen in recovered dark regions (eye pass,
    scene 02) = finest-band-isolated energy; real edges are coherent with
    the mid band. Ratio gate vs the mid band's analytic noise floor."""
    print("== H4: cross-scale coherence on finest-band correction (8-bit) ==", flush=True)
    rows = []
    ck = None
    for coc in (0.04, 0.012):
        for sc in scenes(coc, n_scenes=10):
            ab, D, lv, base, sub = prep(sc)
            if ck is None:
                ck, _ = calibrate_band_noise(sc["gt"].shape[:2], lv)
            g_base = M.ref_ssim(base, sc["gt"])
            g_sub = M.ref_ssim(sub, sc["gt"])
            row = dict(coc=coc, sid=sc["sid"])
            for m in (1.0, 2.0):
                for coh in (False, True):
                    outp = fuse_perband_gain(sc["frames"], {FAR: D}, ab, sc["alpha"],
                                             g_law=glaw_sq, shrink_m=m, ck=ck,
                                             coherence=coh, **WINNER)
                    s = score(outp, base, sc["gt"], sc["alpha"], sc["max_r"], ab, lv, g_base=g_base)
                    row[f"m{m:g}{'c' if coh else ''}"] = dict(
                        dg_vs_sub=float(s["g"] - g_sub), cr=cr_mid(s["cr"]),
                        dg_off=s["dg_off"], fringe=s["fringe"])
            rows.append(row)
            print(f"  coc{coc:g} {sc['sid']:22s} "
                  f"m1 {row['m1']['dg_vs_sub']:+.4f}->{row['m1c']['dg_vs_sub']:+.4f} "
                  f"m2 {row['m2']['dg_vs_sub']:+.4f}->{row['m2c']['dg_vs_sub']:+.4f} "
                  f"cr {row['m2']['cr']:.3f}->{row['m2c']['cr']:.3f}", flush=True)
    for coc in (0.04, 0.012):
        rr = [r for r in rows if r["coc"] == coc]
        for key in ("m1", "m1c", "m2", "m2c"):
            print(f"  agg coc{coc:g} {key:3s}: mean={np.mean([r[key]['dg_vs_sub'] for r in rr]):+.4f} "
                  f"worst={np.min([r[key]['dg_vs_sub'] for r in rr]):+.4f} "
                  f"cr={np.nanmean([r[key]['cr'] for r in rr]):.3f}", flush=True)
    dump("h4", dict(rows=rows))


def cmd_eye():
    """Eye pass (artifact detection only): max-disagreement crops sub vs hybrid
    + GT, plus a clamp-edge crop, per the 2 highest-fringe scenes."""
    from eyetool import compare
    outdir = os.path.join(HERE, "veilgain_eye")
    os.makedirs(outdir, exist_ok=True)
    ck = None
    for sc in scenes(0.04, n_scenes=10)[:4]:
        ab, D, lv, base, sub = prep(sc)
        if ck is None:
            ck, _ = calibrate_band_noise(sc["gt"].shape[:2], lv)
        hyb = fuse_perband_gain(sc["frames"], {FAR: D}, ab, sc["alpha"],
                                g_law=glaw_sq, shrink_m=1.0, ck=ck, **WINNER)
        p = os.path.join(outdir, f"{sc['sid']}.png")
        compare({"sub": sub, "hybrid": hyb}, gt=sc["gt"], out=p, k=3)
        print(f"  {p}", flush=True)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "p0"
    if cmd == "p0":
        cmd_p0()
    elif cmd == "h1":
        cmd_h1()
    elif cmd == "h1a":
        cmd_h1a()
    elif cmd == "h1b":
        cmd_h1b()
    elif cmd in ("h2", "h3"):
        cmd_h2h3(cmd)
    elif cmd == "final":
        cmd_final()
    elif cmd == "h5":
        cmd_h5()
    elif cmd == "h4":
        cmd_h4()
    elif cmd == "eye":
        cmd_eye()
    else:
        print(f"unknown/not-yet-built subcommand: {cmd}")


if __name__ == "__main__":
    main()
