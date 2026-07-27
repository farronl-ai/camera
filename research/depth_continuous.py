"""Motion as a continuous function of depth, with no bins at all (F99 work).

The shipped alignment splits depth into bins and fits one translation per bin. That
is why the kitchen bottle fails: its bin spans 55% of the frame, so ECC fits the
majority and hands the bottle +2.3 px where it needs +19.2 (F84).

Every attempt to fix that by making BETTER REGIONS failed (F98) — sparse propagation,
dense focal peaks, both worse than the bins they replaced. The reason they failed is
that they were all answering the wrong question. Depth is already known per pixel,
and it is right even inside a blank surface: on the kitchen sweep the bottle's blank
interior reads 0.509 against 0.612 at its own edges, while the counter reads 0.244
and the background 0.826. The depth map was never the problem. Quantizing it was.

So: fit displacement as a smooth function of depth,

    d(x) = t( depth(x) )

with t a piecewise-linear curve over a few knots, each knot fitted from the
observations at that depth. Between knots it interpolates rather than rounding to a
bin's majority, so a pixel is never handed the motion of content 0.3 of the depth
range away. No regions, no memberships, no segmentation, and the field stays smooth
by construction.

    .venv/bin/python research/depth_continuous.py
"""
from __future__ import annotations

import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from focusstack.align import align_stack  # noqa: E402
from focusstack.fusion import depth_from_focus, fuse_perband  # noqa: E402
from focusstack.io import to_gray_float  # noqa: E402
import metrics  # noqa: E402
import edge_similarity as ES  # noqa: E402
import object_segmentation as OS  # noqa: E402
import parallax_gen as P  # noqa: E402

KITCHEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "data", "mobiledepth", "Figure3", "kitchen")
KNOTS = 5
TILE = 40
MIN_RESPONSE = 0.15
# Confidence mass a knot needs before its own frame's fit is trusted there.
_KNOT_SUPPORT = 3.0


def focal_frames(grays, features, half=9):
    """The frame at which each feature is sharpest — where it can be trusted."""
    out = []
    for (x, y, nx, ny) in features:
        xi, yi = int(round(x)), int(round(y))
        e = np.array([float(np.abs(cv2.Laplacian(
            g[yi - half:yi + half + 1, xi - half:xi + half + 1], cv2.CV_32F)).mean())
            for g in grays])
        out.append(float(np.argmax(e)))
    return out


def edge_rows(grays, ref, frame, features, depth, focal=None, sigma=2.5):
    """Displacement and depth for every material edge.

    NOT tiles. Phase correlation resolves shifts only to about a quarter of the
    patch, so a 40 px tile silently saturates near 10 px and the kitchen bottle's
    19 px is simply unmeasurable that way — the surviving tiles are all
    low-displacement background, and the fitted curve says the whole scene barely
    moved. Edge-normal profile matching measures the same 19 px correctly (F87/F92),
    and each edge carries the depth at its own location.
    """
    rows = []
    for idx, (x, y, nx, ny) in enumerate(features):
        shift, peak = ES._match(ES._profile(grays[ref], x, y, nx, ny),
                                ES._profile(grays[frame], x, y, nx, ny))
        if peak < 0.5 or abs(shift) > 40:
            continue
        d = float(depth[int(round(y)), int(round(x))])
        weight = float(peak)
        if focal is not None:
            # Trust a feature only near ITS OWN focal plane. Match confidence cannot
            # detect this: a blurred profile correlates confidently against a sharp
            # one at about zero shift, a systematic bias toward "no motion" on
            # exactly the objects that moved most. The kitchen bottle's label
            # reports +2.6 px at frame 11 where it really moved +20, confidently.
            weight *= float(np.exp(-0.5 * ((frame - focal[idx]) / sigma) ** 2))
        rows.append((d, shift, nx, ny, weight))
    return rows


def fit_curve(rows, knots):
    """Piecewise-linear displacement-versus-depth from edge-normal observations.

    Each edge constrains only its normal component, so the two displacement
    curves are solved together: d.n = tx(depth) nx + ty(depth) ny, linear in the
    knot values of both curves.
    """
    if len(rows) < 3 * len(knots):
        return None
    K = len(knots)
    A, b, weight = [], [], []
    for d, shift, nx, ny, peak in rows:
        basis = np.zeros(K)
        if d <= knots[0]:
            basis[0] = 1.0
        elif d >= knots[-1]:
            basis[-1] = 1.0
        else:
            j = int(np.searchsorted(knots, d) - 1)
            span = knots[j + 1] - knots[j]
            u = (d - knots[j]) / max(span, 1e-6)
            basis[j], basis[j + 1] = 1.0 - u, u
        A.append(np.concatenate([basis * nx, basis * ny]))
        b.append(shift); weight.append(peak)
    A = np.asarray(A); b = np.asarray(b)
    w0 = np.asarray(weight); wt = w0.copy()
    for _ in range(3):
        root = np.sqrt(wt)[:, None]
        sol, *_ = np.linalg.lstsq(A * root, b * root.ravel(), rcond=None)
        res = np.abs(A @ sol - b)
        cut = max(float(np.median(res)) * 2.5, 1e-6)
        wt = w0 * np.minimum(1.0, cut / np.maximum(res, 1e-9))
    return sol[:K], sol[K:]


def evaluate_curve(curve, knots, depth):
    """Dense displacement field by interpolating the curve at each pixel's depth."""
    kx, ky = curve
    return (np.interp(depth, knots, kx).astype(np.float32),
            np.interp(depth, knots, ky).astype(np.float32))


def align_depth_continuous(frames, knots=KNOTS):
    """Global warp, then a continuous depth-driven correction. One resample."""
    coarse = align_stack(frames, depth_bins=0, crop_valid=False)
    grays = [to_gray_float(i).astype(np.float32) / 255.0 for i in coarse]
    ref = len(coarse) // 2
    depth = depth_from_focus(coarse)
    h, w = depth.shape
    knot_positions = np.quantile(depth, np.linspace(0.05, 0.95, knots))
    knot_positions = np.unique(knot_positions)
    if len(knot_positions) < 2:
        return coarse, None

    features = OS.material_features(grays, ref, depth)
    focal = focal_frames(grays, features)
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
    # Fit every frame, and record how well supported each fit is.
    raw, support = {}, {}
    for k in range(len(grays)):
        if k == ref:
            continue
        rows = edge_rows(grays, ref, k, features, depth, focal)
        curve = fit_curve(rows, knot_positions)
        if curve is None:
            continue
        raw[k] = curve
        # Support per knot: the confidence mass of observations near that depth.
        mass = np.zeros(len(knot_positions))
        for d, shift, nx, ny, peak in rows:
            mass[int(np.argmin(np.abs(knot_positions - d)))] += peak
        support[k] = mass

    # PROPAGATE rather than trust a degraded frame. Off an object's focal plane its
    # own features blur away, and a profile match against a near-featureless patch
    # collapses toward zero shift — a systematic bias, not noise. On the kitchen
    # sweep the bottle's label features report +2.6 px at frame 11 where the object
    # has really moved +20. So each knot is refitted across the sweep from the
    # frames that actually support it, using the frames NEAREST the target (F89/F93:
    # handheld drift is not linear across a whole sweep).
    curves = {}
    frames_sorted = sorted(raw)
    for k in frames_sorted:
        kx, ky = raw[k]
        kx, ky = kx.copy(), ky.copy()
        for j in range(len(knot_positions)):
            trusted = [(m, raw[m][0][j], raw[m][1][j]) for m in frames_sorted
                       if support[m][j] >= _KNOT_SUPPORT]
            if len(trusted) < 3 or support[k][j] >= _KNOT_SUPPORT:
                continue
            near = sorted(trusted, key=lambda t: abs(t[0] - k))[:4]
            t = np.array([m - ref for m, _, _ in near], float)
            kx[j] = np.polyval(np.polyfit(t, [v for _, v, _ in near], 1), k - ref)
            ky[j] = np.polyval(np.polyfit(t, [v for _, _, v in near], 1), k - ref)
        curves[k] = (knot_positions, (kx, ky))

    out = []
    for k in range(len(grays)):
        if k == ref or k not in curves:
            out.append(coarse[k]); continue
        dx, dy = evaluate_curve(curves[k][1], knot_positions, depth)
        out.append(cv2.remap(coarse[k], grid_x + dx, grid_y + dy, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE))
    return out, curves


def main() -> None:
    print("=== kitchen: does the bottle finally get its correction? ===")
    src = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(KITCHEN, "*.jpg")))]
    aligned, curves = align_depth_continuous(src)
    if curves and 11 in curves:
        knots, (kx, _) = curves[11]
        print("  frame 11 displacement vs depth:")
        for d, v in zip(knots, kx):
            print(f"    depth {d:.3f} -> {v:+6.2f} px")
        print("  the bottle sits at depth ~0.51-0.61 and needs about +19.2 px")

    coarse = align_stack(src, depth_bins=0, crop_valid=False)
    g = [to_gray_float(i).astype(np.float32) / 255.0 for i in coarse]
    ga = [to_gray_float(i).astype(np.float32) / 255.0 for i in aligned]
    ref = len(g) // 2

    def edge_shift(a, b, y0, y1, x0, x1):
        pa = np.gradient(a[y0:y1, x0:x1].mean(0)); pb = np.gradient(b[y0:y1, x0:x1].mean(0))
        pa = pa - pa.mean(); pb = pb - pb.mean()
        c = np.correlate(pb, pa, mode="full"); i = int(np.argmax(c)); off = 0.0
        if 0 < i < len(c) - 1:
            d = c[i - 1] - 2 * c[i] + c[i + 1]
            off = 0.5 * (c[i - 1] - c[i + 1]) / d if abs(d) > 1e-12 else 0.0
        return (i - (len(pa) - 1)) + off

    print("\n  residual misregistration of the bottle's right edge (should reach ~0):")
    for k in (8, 9, 10, 11):
        before = edge_shift(g[ref], g[k], 160, 340, 600, 700)
        after = edge_shift(ga[ref], ga[k], 160, 340, 600, 700)
        print(f"    frame {k}: global-only {before:+6.2f} px -> depth-continuous {after:+6.2f} px")


if __name__ == "__main__":
    main()
