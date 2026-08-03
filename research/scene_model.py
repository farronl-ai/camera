"""Scene-model reconstruction — round B2 of the scene-model second pass.

Where this sits. Round A (`forward_certify.py`, F113) built the ARBITER: a
forward renderer that scores a candidate composite plus pass-1's scene model
against the RAW frames in their own geometry. Round B1 (`layer_decompose.py`,
F114) replaced pass-1's focus-contest winner map with a focal-signature
DECOMPOSITION carrying trinary ownership. Neither round changed a pixel of any
composite. This round does: it assembles per-layer appearance from the frames
and writes a scene-model composite.

REFINEMENT, NOT REPLACEMENT. The input is the routed two-frame composite that
ships today (`focusstack.twoframe.twoframe_stack` on the exposure-normalized
stack). The second pass rewrites a pixel ONLY where B1's decomposition OWNS it
and the assembled appearance survives the certifier; boundary-band, unknown and
un-assembled pixels keep the input composite's value BYTE-IDENTICALLY. That is
F101's non-regression-by-construction and F106's trinary application, applied to
a reconstruction: rewrite / keep / never-degrade.

The hybrid model (F114). Discrete layers are ownership at OCCLUSION boundaries;
continuous-ramp content (the kitchen countertop recedes in 1/Z) is not a stack of
layers and is not this round's to rewrite — pass 1's winner map is already a good
BLUR map there. Rewrite authority is exactly B1's owned regions.

The four things assembly has to get right, and where each rule comes from:

  1. GEOMETRY — one rigid transform per (frame, layer): the global affine
     composed with the layer's own propagated shift, collapsed to a rigid
     translation where the layer's own material edges prefer that, verified by
     `twoframe.gate_shift`. Composed once, resampled once (PLAYBOOK §0: every
     resample softens).
  2. VISIBILITY — an observation may only be used where the frame actually saw
     the surface. F114 §9 measured that the occlusion ORDERING is a focal-peak
     proxy whose independent guard (F83's contour bit) REFUSES, so this module
     does not lean on ordering at all: a pixel is declined if ANY other layer's
     footprint, warped into that frame's geometry and pulled back through this
     layer's own map, covers it. Ordering-free by construction, and the price is
     measured (`orderguard`).
  3. ADMISSION — the PHYSICAL same-surface test (see `same_surface_physical`).
     This is the per-pixel version F112/R5 designed and did not build, and it
     retires `SURFACE_SIGMA`'s §12.3 two-scene split.
  4. AGGREGATION — never average across a focus disagreement (F79/F31): decide
     first, reconstruct second. Only the SHARPEST admitted observation and the
     others that share its blur to within a pixel are averaged, and each of those
     must agree with the sharpest one.

NO COMPLETION. Where assembly has nothing admissible the input composite's pixel
stands. F114 forbids placing content that is occluded in the frames that own it,
and nothing here does.

    .venv/bin/python research/scene_model.py kat        # the physical test, known answers
    .venv/bin/python research/scene_model.py slope      # the blur ladder's own KAT
    .venv/bin/python research/scene_model.py localkat   # the LOCAL veto's arbiter
    .venv/bin/python research/scene_model.py factory    # bar A: GT-SSIM + the ladder
    .venv/bin/python research/scene_model.py kitchen    # bars B/C/D + both ledgers
    .venv/bin/python research/scene_model.py orderguard # what the ordering-free rule costs
    .venv/bin/python research/scene_model.py boundary   # a REJECTED clause, priced (§23)
    .venv/bin/python research/scene_model.py render     # out/inspect/kitchen_scenemodel.png
"""
from __future__ import annotations

import glob
import os
import sys
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from focusstack import motion_groups as MG  # noqa: E402
from focusstack import twoframe as TF  # noqa: E402
from focusstack.align import _homogeneous  # noqa: E402
from focusstack.io import to_gray_float  # noqa: E402
import forward_certify as FC  # noqa: E402
import layer_decompose as LD  # noqa: E402

OUT = os.path.abspath(os.path.join(HERE, "..", "out", "certify"))
INSPECT = os.path.abspath(os.path.join(HERE, "..", "out", "inspect"))
KITCHEN = os.path.join(HERE, "data", "mobiledepth", "Figure3", "kitchen")

# --- constants, and where each one comes from --------------------------------
# The residual low-pass that survives after two observations have been matched to
# a common defocus. It is the SAMPLING SCALE: below one pixel the disk PSF, the
# bilinear kernel of the module's single resample, and the pixel grid itself are
# not distinguishable, so a disagreement at that scale is not evidence about
# which surface is being observed. `kat` measures the verdict's sensitivity to it.
SIGMA0 = 1.0
# A rewritten region needs at least this many certified pixels before the
# certifier is allowed to arbitrate it. `MIN_COVERAGE` frames is the certifier's
# own quorum; 200 px is `estimate_radii`'s own minimum weight for a fit to be
# scored at all. Borrowed, not chosen.
MIN_ARBITRABLE = 200
# The smallest window over which the certifier may return a LOCAL verdict: the
# quorum it already requires for a REGIONAL one, made square. Nothing new —
# `ceil(sqrt(MIN_ARBITRABLE))` is the finest scale at which the never-degrade
# rule can be asked the same question it is asked per region. See §15.
LOCAL_WINDOW = int(np.ceil(np.sqrt(MIN_ARBITRABLE)))     # 15 px
# How far a frontier revert carries past the pixel that earned it. `GATE_TOL` is
# the displacement every gate and every budget in this arc already declines to
# resolve, so it is also how far a frontier's true position is undetermined; a
# revert that stops short of it leaves the undetermined side standing. Rounded
# up to whole pixels. §15 prices 1 / 2 / 3 / 5 — the choice is not free.
FRONTIER_SLACK = int(np.ceil(TF.GATE_TOL))               # 2 px


# ---------------------------------------------------------------------------
# The physical same-surface test — F112/R5's unbuilt design
# ---------------------------------------------------------------------------
# F112 shipped `same_surface` with ONE global low-pass scale, `SURFACE_SIGMA`,
# and logged it as an unresolved §12.3 split: the analytic factory (which has
# ground truth) wants 2, the kitchen boxes want 8, and 4.0 was the smallest value
# clearing every bar. DEVSTYLE §12.3 says a threshold two scenes want opposite
# values for is the wrong instrument, and names the cure: find the physical
# invariant it is standing in for. R5 named it exactly:
#
#     "The low-pass must exceed the RESIDUAL DEFOCUS DIFFERENCE between the two
#      frames, and that is a per-pixel quantity the module already has the
#      ingredients for: `peak` is the focal frame per pixel and the arc's
#      validated blur proxy is distance from the object's focal frame."
#
# So do not exceed the difference — REMOVE it, exactly. Two observations of one
# latent surface are
#
#     m = L (x) disk(R_m)        r = L (x) disk(R_r)
#     R(k, p) = c * |k - peak(p)|      the disk radius, c measured per scene
#
# so blurring EACH by the OTHER's disk makes them identical:
#
#     m (x) disk(R_r)  ==  L (x) disk(R_m) (x) disk(R_r)  ==  r (x) disk(R_m)
#
# CROSS-CONVOLUTION, and it is exact — no PSF family mismatch, because the family
# used to match is the family the physics uses (PLAYBOOK §0: real defocus is a
# DISK). A Gaussian second-moment match was tried first and measured: it leaves a
# family residual that grows with radius (agreement 0.867 at 12 px against 1.000
# here). Disk radii are integers by construction, so the ladder is the exact
# parameter space and not a discretization of it.
#
# The BUDGET is F112's, unchanged and unretuned: a GATE_TOL displacement times
# the local gradient (F106's unexplained-motion rule asked per pixel),
# `normalize_exposure`'s measured multiplicative residual, and the sensor noise
# floor. Replacing that linearization with the exact statement it approximates —
# minimize the disagreement over a shift grid at +-GATE_TOL, charge only the
# sub-grid remainder to the gradient — was built and MEASURED WORSE on the
# committed fixture, on every row: the search buys a moved occluder a
# GATE_TOL-wide sliver at each edge of the strip it vacated (0.263 admitted at a
# 4 px move against 0.010 linearized) and simultaneously admits MORE of a 4 px
# misregistration (0.885 vs 0.691). The linearization is not merely cheaper here,
# it is the tighter bound. Recorded as a negative deliverable, not repeated.


def _disk_ladder(image: np.ndarray, radius_max: float):
    """`image` convolved with every disk radius up to `radius_max`, integer grid."""
    top = int(np.ceil(max(0.0, float(radius_max))))
    stack = [FC.defocus(image, r) for r in range(top + 1)]
    return np.stack(stack, 0)


def _select(stack: np.ndarray, radius_map: np.ndarray):
    """Per-pixel pick from a ladder — a spatially varying convolution."""
    index = np.clip(np.rint(radius_map).astype(np.int32), 0, len(stack) - 1)
    h, w = radius_map.shape
    yy, xx = np.indices((h, w))
    return stack[index, yy, xx]


def _gradient_magnitude(blurred: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return np.sqrt(gx * gx + gy * gy)


def match_blur(member: np.ndarray, reference: np.ndarray,
               radius_member: np.ndarray, radius_reference: np.ndarray,
               sigma0: float = SIGMA0):
    """Bring two observations to a COMMON defocus by cross-convolution.

    Each is convolved with the OTHER's disk, which is exact for two observations
    of one latent surface. `sigma0` is then applied to both identically — the
    sampling-scale residual that the disk model, the bilinear resample and the
    pixel grid share, and it cancels no structure because it is common.
    """
    r_m = np.maximum(radius_member, 0.0).astype(np.float32)
    r_r = np.maximum(radius_reference, 0.0).astype(np.float32)
    low_m = _select(_disk_ladder(member.astype(np.float32), r_r.max()), r_r)
    low_r = _select(_disk_ladder(reference.astype(np.float32), r_m.max()), r_m)
    if sigma0 > 0:
        low_m = cv2.GaussianBlur(low_m, (0, 0), sigma0)
        low_r = cv2.GaussianBlur(low_r, (0, 0), sigma0)
    return low_m, low_r


def agreement_budget(low_member: np.ndarray, low_reference: np.ndarray,
                     tol: float = TF.GATE_TOL) -> np.ndarray:
    """What a disagreement is ALLOWED to be, in levels. F112's terms, unchanged."""
    return (tol * np.maximum(_gradient_magnitude(low_member),
                             _gradient_magnitude(low_reference)).max(axis=2)
            + TF.SURFACE_GAIN * np.maximum(low_member, low_reference).max(axis=2)
            + TF.SURFACE_NOISE)


def same_surface_physical(member, reference, radius_member, radius_reference,
                          sigma0: float = SIGMA0, tol: float = TF.GATE_TOL):
    """`twoframe.same_surface`, with its one free scale replaced by physics.

    Returns a bool mask: True where the two observations, once brought to a
    COMMON defocus by cross-convolution, agree to within what a GATE_TOL
    displacement, the exposure residual and sensor noise can explain.

    Known-answer tested in `kat`, on the COMMITTED fixture and against the
    committed pass marks, so it and the global-sigma version are directly
    comparable on the four questions F112/R3 asked.
    """
    low_m, low_r = match_blur(member, reference, radius_member, radius_reference,
                              sigma0)
    disagreement = np.abs(low_m - low_r).max(axis=2)
    return disagreement <= agreement_budget(low_m, low_r, tol)


# ---------------------------------------------------------------------------
# The blur ladder's scale, measured rather than assumed
# ---------------------------------------------------------------------------
def blur_slope(model, appearances, supports, raw, peaks, verbose=False):
    """Disk radius per frame of focal distance, `c`, for one scene.

    PLAYBOOK §0c closes contrast-over-gradient blur estimation. The instrument
    used here is the certifier's own forward radius search, which KAT-2 measured
    recovering a KNOWN radius exactly (100%) on every rung including from a real
    imperfect composite. This regresses its per (frame, layer) answers against
    the focal distance the ladder already asserts:

        r(k, i) = c * |k - peak_i|     ->    c = sum(r*d) / sum(d*d)

    Through the origin, because a layer at its own focal frame is in focus by the
    definition of a focal peak. The factory's own constant is BLUR_PER_STEP =
    1.15, which is `slope`'s known answer.
    """
    report, _cost = FC.estimate_radii(model, appearances, supports, raw,
                                      update=False)
    numerator = denominator = 0.0
    rows = []
    for (k, i), (radius, _gain, _mae, count) in report.items():
        if count <= 0 or i >= len(peaks):
            continue
        distance = abs(k - peaks[i])
        numerator += radius * distance
        denominator += distance * distance
        rows.append((k, i, distance, radius))
    slope = numerator / denominator if denominator > 0 else 0.0
    if verbose:
        residual = np.array([r - slope * d for _k, _i, d, r in rows])
        print(f"    blur slope c = {slope:.3f} px per frame of focal distance "
              f"({len(rows)} (frame, layer) radii, residual rms "
              f"{float(np.sqrt((residual ** 2).mean())):.2f} px)")
    return float(slope), rows


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
@dataclass
class Assembly:
    composite: np.ndarray            # crop geometry, the scene-model composite
    base: np.ndarray                 # crop geometry, the input routed composite
    reference: np.ndarray            # crop geometry, the reference FRAME
    rewritten: np.ndarray            # crop geometry, bool
    layer_of: np.ndarray             # crop geometry, int32 (-1 where not rewritten)
    best_frame: np.ndarray           # crop geometry, int32 (-1 where none)
    scoped_null: np.ndarray          # crop geometry, per-region best SINGLE frame
    crop: tuple
    dec: object
    slope: float
    diag: dict = field(default_factory=dict)


def _geometry(images, dec, verbose=False):
    """One rigid transform per (frame, layer) — the two-frame discipline.

    The menu, the arbiter and the gate are all pass 1's own. `twoframe_stack`
    builds exactly this menu per (frame, layer) for the two frames it elects; the
    only thing added here is that it is built for EVERY frame, because a
    reconstruction that needs all N cannot get them from a stage that renders two
    (F113's closing note).

    Trinary per F106/F110: VERIFIED applies the layer shift, CONTRADICTED refuses
    the observation outright (a fit the evidence contradicts may not be softened
    into the global affine — that is the silent invention the gate exists to
    stop), UNVERIFIABLE declines the correction and keeps the global stage's
    geometry without throwing the observation away.
    """
    n = len(images)
    ref = dec.diag["ref"]
    h, w = images[0].shape[:2]
    coarse, warps, common = dec.diag["coarse"], dec.diag["warps"], dec.diag["common"]
    labels, peak = dec.labels, dec.diag["peak"]
    grays = [to_gray_float(c).astype(np.float32) / 255.0 for c in coarse]
    ref_gray = grays[ref]
    gradient = cv2.magnitude(
        cv2.Sobel(ref_gray * 255.0, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(ref_gray * 255.0, cv2.CV_32F, 0, 1, ksize=3))
    textured = gradient >= TF.A._REFINE_MIN_GRADIENT
    from focusstack.fusion import depth_from_focus
    evidence = TF.EdgeEvidence(grays, ref, depth_from_focus(coarse), common, peak)

    shifts = dec.diag["layer_shift"]
    matrices, verdicts = {}, {}
    tally = {"verified": 0, "contradicted": 0, "unverifiable": 0}
    for i in range(len(dec.masks)):
        support = (labels == i)
        indices = evidence.indices_in(support & textured & common)
        for k in range(n):
            base = np.eye(3) if warps[k] is None else _homogeneous(warps[k])
            if k == ref:
                matrices[(k, i)] = np.eye(3)
                verdicts[(k, i)] = ("verified", 0.0, "reference")
                tally["verified"] += 1
                continue
            dx, dy = shifts[(k, i)]
            composed = base @ np.array([[1.0, 0.0, dx], [0.0, 1.0, dy],
                                        [0.0, 0.0, 1.0]])
            rigid = TF._rigidify(composed, support, (h, w))
            inverse = np.linalg.inv(base)
            best = None
            for name, candidate in (("affine", composed), ("rigid", rigid)):
                check = evidence.residual_translation(k, indices, inverse @ candidate)
                if check is None:
                    continue
                if best is None or check["rms"] < best[0] - 1e-6:
                    best = (check["rms"], name, candidate)
            chosen = composed if best is None else best[2]
            status, statistic, reason = TF.gate_shift(evidence, k, indices,
                                                      inverse @ chosen)
            tally[status] += 1
            if status == "contradicted":
                matrices[(k, i)] = None
            elif status == "unverifiable":
                matrices[(k, i)] = base
            else:
                matrices[(k, i)] = chosen
            verdicts[(k, i)] = (status, statistic,
                                f"{'—' if best is None else best[1]}; {reason}")
    if verbose:
        total = sum(tally.values())
        print(f"    geometry: {tally['verified']} verified, "
              f"{tally['unverifiable']} unverifiable (global affine kept), "
              f"{tally['contradicted']} contradicted (observation refused) "
              f"of {total} (frame, layer) pairs")
    return matrices, verdicts, tally


def _visibility(images, dec, matrices, k, i, order_guard="any"):
    """Where frame k can legitimately be read for layer i, in reference geometry.

    Ordering-free by default, and that is deliberate. F114 §9: the layer order is
    a focal-peak proxy and F83's contour bit, the one independent cue that could
    guard its DIRECTION, refuses on today's factory (`near_is_low=None`). An
    assembly that only skipped NEARER layers would be trusting that direction. So
    every other layer's footprint is treated as a possible occluder: a pixel is
    declined if the footprint of any layer j != i, carried into frame k's
    geometry by ITS OWN transform and pulled back through layer i's, lands on it.
    In reference geometry the two footprints are disjoint, so what this actually
    refuses is the DIFFERENTIAL-MOTION band at each shared boundary — exactly the
    strip where an occluder swung — and `orderguard` measures the price against
    the ordered variant.
    """
    h, w = images[0].shape[:2]
    matrix = matrices[(k, i)]
    if matrix is None:
        return np.zeros((h, w), bool), np.zeros((h, w), bool)
    inside = FC.warp_back(np.ones((h, w), np.float32), matrix, (h, w), 0.0) > 0.999
    occupied = np.zeros((h, w), np.float32)
    for j in range(len(dec.masks)):
        if j == i:
            continue
        if order_guard == "nearer" and dec.order[j] >= dec.order[i]:
            continue
        other = matrices[(k, j)]
        if other is None:
            continue
        footprint = (dec.labels == j).astype(np.float32)
        occupied = np.maximum(occupied, FC.warp_forward(footprint, other, (h, w), 0.0))
    visible = inside
    if occupied.any():
        back = FC.warp_back(occupied, matrix, (h, w), 0.0)
        visible = inside & (back < 0.001)
    return inside, visible


def assemble(images, ref=None, order_guard="any", verbose=False, raw=None,
             dec=None, matrices=None):
    """Build the scene-model composite. Returns an `Assembly`.

    Nothing is written outside B1's OWNED pixels, and nothing is written where
    the frames have nothing admissible to say. `dec` and `matrices` override B1's
    decomposition and the fitted geometry, which is how the factory's oracle
    rungs substitute a known answer for one estimated quantity at a time.
    """
    n = len(images)
    ref = n // 2 if ref is None else ref
    raw = images if raw is None else raw
    h, w = images[0].shape[:2]
    t0 = time.time()

    base_composite, info = TF.twoframe_stack(images, ref)
    crop = tuple(info["crop"])
    x0, y0, x1, y1 = crop
    if verbose:
        print(f"    input routed composite {base_composite.shape}, crop {crop} "
              f"({time.time() - t0:.1f}s)")

    dec = LD.decompose(images, ref, verbose=verbose) if dec is None else dec
    if matrices is None:
        matrices, verdicts, tally = _geometry(images, dec, verbose=verbose)
    else:
        verdicts, tally = {}, {"oracle": len(matrices)}

    # The certifier's model, on B1's decomposition, with the geometry frozen on
    # the NULL appearance exactly as round A freezes it — so the same model that
    # will judge the rewrite also supplies the blur ladder that builds it, and
    # neither is fitted to the candidate.
    model, _diag = FC.model_from_pass1(images, ref,
                                       segmentation=dec.segmentation())
    null = images[ref][y0:y1, x0:x1]
    null_canvas, null_support = FC.place(null, crop, model.shape)
    null_app, null_sup = FC.layer_views(model, null_canvas, null_support)
    FC.select_geometry(model, null_app, null_sup, raw)
    canvas, support = FC.place(base_composite, crop, model.shape)
    appearances, supports = FC.layer_views(model, canvas, support)
    slope, _rows = blur_slope(model, appearances, supports, raw, dec.peaks,
                              verbose=verbose)

    # Per-pixel focal peak, with the LAYER's peak standing in where the two focus
    # operators did not agree (B1's `evidenced` channel). A pixel that was
    # labelled but not measured takes its layer's answer and is not pretended to
    # carry its own.
    peak = dec.diag["peak"].astype(np.float32)
    evidenced = dec.diag["evidenced"]
    peak_used = peak.copy()
    for i in range(len(dec.masks)):
        here = (dec.labels == i) & ~evidenced
        peak_used[here] = dec.peaks[i]

    reference = images[ref].astype(np.float32)
    radius_ref = slope * np.abs(ref - peak_used)

    owned = dec.state == LD.OWNED
    observations = np.zeros((n, h, w, 3), np.float32)
    admitted = np.zeros((n, h, w), bool)
    # What each refusal stage actually costs, in owned-pixel-frames. Reported
    # because an inactive gate is a finding, not a detail (§12.2).
    refusals = {"no geometry": 0, "outside the frame": 0, "occluded": 0,
                "different surface": 0, "admitted": 0, "owned frames": 0}
    for k in range(n):
        # ONE resample per (frame, layer): the composed transform is applied
        # directly to the ORIGINAL frame (PLAYBOOK §0 — compose, resample once).
        # Layers sharing a matrix share the resample.
        done = {}
        for i in range(len(dec.masks)):
            here = owned & (dec.labels == i)
            if not here.any():
                continue
            refusals["owned frames"] += int(here.sum())
            matrix = matrices[(k, i)]
            if matrix is None:
                refusals["no geometry"] += int(here.sum())
                continue
            key = matrix.tobytes()
            if key not in done:
                done[key] = FC.warp_back(images[k].astype(np.float32), matrix,
                                         (h, w), 0.0)
            observations[k][here] = done[key][here]
            inside, visible = _visibility(images, dec, matrices, k, i,
                                          order_guard=order_guard)
            refusals["outside the frame"] += int((here & ~inside).sum())
            refusals["occluded"] += int((here & inside & ~visible).sum())
            admitted[k] |= here & visible
        if not admitted[k].any():
            continue
        radius_k = slope * np.abs(k - peak_used)
        agree = same_surface_physical(observations[k], reference,
                                      radius_k, radius_ref)
        refusals["different surface"] += int((admitted[k] & ~agree).sum())
        admitted[k] &= agree
        refusals["admitted"] += int(admitted[k].sum())

    # --- aggregation: the sharpest MUTUALLY-AGREEING subset -------------------
    # F79/F31: decision, then reconstruction, never an average across a focus
    # disagreement. The decision picks the sharpest admitted observation; only
    # observations that share its blur EXACTLY may join it, and the set is
    # defined by the modelled circle of confusion, never by a measured sharpness
    # (PLAYBOOK §0c forbids the latter). Disk radii are integers by
    # construction, so "shares its blur" needs no tolerance: it is equality.
    #
    # TWO members may only be averaged if BOTH their geometries VERIFIED. This
    # is F106 at the reconstruction stage and it was learned here the expensive
    # way. A first build averaged over any admitted frame within a pixel of the
    # sharpest radius, including the 41-of-72 (frame, layer) pairs whose fit was
    # UNVERIFIABLE and therefore fell back to the global affine. Averaging two
    # observations placed by two unverified geometries is a photometric blend of
    # two geometries — the exact thing F106 forbids — and on the kitchen it
    # visibly softened the Lubriderm label, which the sweep moves ~3 px/frame.
    # An unverifiable observation is still USED (F110's trinary keeps it); it is
    # simply used alone, as a one-hot decision, never blended.
    radii = np.rint(np.stack([slope * np.abs(k - peak_used) for k in range(n)], 0))
    verified = np.zeros((n, h, w), bool)
    for (k, i), entry in verdicts.items():
        if entry[0] == "verified":
            verified[k] |= (dec.labels == i)
    if not verdicts:                       # oracle geometry: verified by fiat
        verified[:] = True
    big = np.float32(1e9)
    masked_radii = np.where(admitted, radii, big)
    best_frame = np.argmin(masked_radii, axis=0).astype(np.int32)
    any_admitted = admitted.any(axis=0)
    best_frame[~any_admitted] = -1
    yy, xx = np.indices((h, w))
    pick = np.clip(best_frame, 0, n - 1)
    best_radius = masked_radii[pick, yy, xx]
    best_verified = verified[pick, yy, xx]

    lows = np.stack([cv2.GaussianBlur(observations[k], (0, 0), SIGMA0)
                     for k in range(n)], 0)
    sharpest_low = lows[pick, yy, xx]
    budget = agreement_budget(sharpest_low, sharpest_low)

    total = np.zeros((h, w, 3), np.float32)
    weight = np.zeros((h, w), np.float32)
    members = np.zeros((h, w), np.int32)
    for k in range(n):
        equal_blur = admitted[k] & any_admitted & (radii[k] == best_radius)
        mutual = np.abs(lows[k] - sharpest_low).max(axis=2) <= budget
        keep = equal_blur & mutual & verified[k] & best_verified
        keep |= (best_frame == k) & any_admitted        # the sharpest is always in
        # Sharpness-weighted inside an equal-blur set: monotone in the modelled
        # circle of confusion, so the marginally sharper member leads. Inside the
        # set the radii are equal, so the weights are uniform by construction —
        # the expression is kept because it states the intent, not because it
        # discriminates.
        w_k = (1.0 / (1.0 + radii[k])) * keep
        total += w_k[..., None] * observations[k]
        weight += w_k
        members += keep.astype(np.int32)

    assembled = np.zeros((h, w, 3), np.float32)
    have = weight > 0
    assembled[have] = total[have] / weight[have][..., None]

    # --- the region-scoped null: per owned region, the best SINGLE frame ------
    # F114 §7: the global null (one defocused reference frame) is a carpet that
    # hides region-scale defects, and per-layer appearance is the material for a
    # scoped one. This is the same assembly with the aggregation removed —
    # exactly one admissible observation per region, the sharpest.
    scoped = np.zeros((h, w, 3), np.float32)
    scoped[have] = observations[np.clip(best_frame, 0, n - 1), yy, xx][have]

    full_base = np.zeros((h, w, 3), np.float32)
    full_base[y0:y1, x0:x1] = base_composite
    rewrite = owned & have
    rewrite[:y0] = rewrite[y1:] = False
    rewrite[:, :x0] = rewrite[:, x1:] = False
    out = full_base.copy()
    out[rewrite] = assembled[rewrite]
    scoped_full = full_base.copy()
    scoped_full[rewrite] = scoped[rewrite]

    layer_of = np.where(rewrite, dec.labels, -1).astype(np.int32)
    composite = np.clip(out[y0:y1, x0:x1], 0, 255).astype(np.uint8)
    scoped_null = np.clip(scoped_full[y0:y1, x0:x1], 0, 255).astype(np.uint8)
    # BYTE-IDENTITY, asserted rather than hoped for (bar D).
    keep_mask = ~rewrite[y0:y1, x0:x1]
    assert np.array_equal(composite[keep_mask], base_composite[keep_mask]), \
        "non-owned pixels are not byte-identical to the input composite"
    assert np.array_equal(scoped_null[keep_mask], base_composite[keep_mask])

    if verbose:
        print(f"    rewritten {rewrite.mean() * 100:5.2f}% of the frame "
              f"({rewrite.sum() / max(1, owned.sum()) * 100:.1f}% of OWNED); "
              f"mean members per rewritten pixel "
              f"{float(members[rewrite].mean()) if rewrite.any() else 0:.2f}; "
              f"{time.time() - t0:.1f}s")
    return Assembly(
        composite=composite, base=base_composite,
        reference=images[ref][y0:y1, x0:x1].copy(),
        rewritten=rewrite[y0:y1, x0:x1], layer_of=layer_of[y0:y1, x0:x1],
        best_frame=np.where(rewrite, best_frame, -1)[y0:y1, x0:x1],
        scoped_null=scoped_null, crop=crop, dec=dec, slope=slope,
        diag={"model": model, "matrices": matrices, "verdicts": verdicts,
              "tally": tally, "admitted": admitted, "members": members,
              "info": info, "rewrite_full": rewrite, "peak_used": peak_used,
              "any_admitted": any_admitted, "owned": owned, "ref": ref,
              "refusals": refusals})


# ---------------------------------------------------------------------------
# Certification of the rewrite, globally and per region
# ---------------------------------------------------------------------------
def regions_of(assembly: Assembly):
    """Connected components of the rewrite, per layer. Reference geometry.

    EVERY component, with no minimum area — and that is the correction. The
    first build filtered components below `twoframe.MIN_LAYER_PIXELS` out of the
    ledger, on the reasonable-sounding grounds that a 30 px blob is not worth a
    certifier verdict. But `apply_veto` only ever REMOVES pixels, so a component
    that never reaches the ledger is not skipped, it is ADMITTED — written into
    the composite with no verdict of any kind. A size filter in front of a veto
    is a silent grant of authority. On the kitchen that was 30 components and
    1304 px, and two of the manager's three defects lived in them.

    Small components are still not arbitrable, and `apply_veto` still says so:
    they fall under `MIN_ARBITRABLE` and are reverted as UNARBITRATED, which is
    the rule the module already had (F106 — an unexplained change is refused,
    not waved through). The rule did not need changing. It needed running.
    """
    rewrite = assembly.diag["rewrite_full"]
    labels = assembly.dec.labels
    found = []
    for i in range(len(assembly.dec.masks)):
        here = (rewrite & (labels == i)).astype(np.uint8)
        if here.sum() == 0:
            continue
        count, tagged, stats, _c = cv2.connectedComponentsWithStats(here, 8)
        for tag in range(1, count):
            found.append({"layer": i, "mask": tagged == tag,
                          "area": int(stats[tag][4]),
                          "box": (int(stats[tag][0]), int(stats[tag][1]),
                                  int(stats[tag][0] + stats[tag][2]),
                                  int(stats[tag][1] + stats[tag][3]))})
    found.sort(key=lambda r: -r["area"])
    return found


def certify_candidates(assembly: Assembly, raw, candidates):
    """Score several composites through ONE frozen model. Returns Certifications."""
    model = assembly.diag["model"]
    crop = assembly.crop
    region = np.zeros(model.shape, bool)
    region[crop[1]:crop[3], crop[0]:crop[2]] = True
    results = {}
    for label, composite in candidates:
        canvas, support = FC.place(composite, crop, model.shape)
        appearances, supports = FC.layer_views(model, canvas, support)
        model.radii, model.gains = {}, {}
        FC.estimate_radii(model, appearances, supports, raw)
        results[label] = FC.certify(model, composite, crop, raw, region=region)
    return results, region


def apply_veto(assembly: Assembly, scene_result, base_result, regions,
               verbose=False):
    """Never-degrade: revert every region the certifier does not prefer.

    Trinary, and the third state is the honest one. A region with too little
    certified evidence is UNARBITRATED — the instrument has no verdict there, and
    F106's rule is that an unexplained change must be refused, not waved through.
    So it is reverted too, and counted separately from the regions the certifier
    actively vetoed.
    """
    delta = scene_result.unexplained - base_result.unexplained
    scored = (np.minimum(scene_result.coverage, base_result.coverage)
              >= FC.MIN_COVERAGE)
    verdicts = []
    reverted = np.zeros(assembly.dec.labels.shape, bool)
    for entry in regions:
        here = entry["mask"] & scored
        count = int(here.sum())
        if count < MIN_ARBITRABLE:
            state, value = "unarbitrated", float("nan")
            reverted |= entry["mask"]
        else:
            value = float(delta[here].mean())
            if value > 0.0:
                state, _ = "vetoed", reverted.__ior__(entry["mask"])
            else:
                state = "kept"
        verdicts.append(dict(entry, state=state, differential=value,
                             certified=count))
    if verbose:
        counts = {}
        for entry in verdicts:
            counts[entry["state"]] = counts.get(entry["state"], 0) + 1
        print(f"    never-degrade veto over {len(verdicts)} regions: "
              + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    return verdicts, reverted


# ---------------------------------------------------------------------------
# The LOCAL never-degrade clause — the correction round's whole subject
# ---------------------------------------------------------------------------
# The region veto is a mean over a region, and a mean can buy an aggregate
# improvement by paying in localized new structure. On the kitchen it did
# exactly that: ONE kept region (layer 5, 79019 px, -1.9503 levels) contains
# three separate clusters the manager's eyes found, whose own local
# differentials are +0.98, +3.15 and +3.87. That is the same bargain F106
# outlawed for geometry — a soft average standing in for a decision — one level
# up, at the appearance layer.
#
# The cure is not a new arbiter. It is the SAME never-degrade rule, asked at
# every scale at which the rewrite is actually written:
#
#   1. per COMPONENT   `regions_of` no longer filters by area, so every
#                      component gets a verdict and the unarbitrable ones
#                      revert. (See its docstring — this was a hole, not a
#                      threshold.)
#   2. per CLUSTER     `local_veto` below: the certifier differential pooled
#                      over the smallest window that still meets the
#                      certifier's own quorum, thresholded at the same 0.0 the
#                      region rule uses.
#   3. per FRONTIER    `retreat_frontier` below: a rewrite may not END on a
#                      step the surface cannot hide.
#
# No new free number is introduced by any of the three. `LOCAL_WINDOW` is
# `MIN_ARBITRABLE` made square, the threshold is the region rule's own 0.0, and
# the frontier test is F112's own agreement budget.


def cluster_pool(assembly: Assembly, scene_result, base_result):
    """The cluster clause's own pooled differential and quorum, in CROP geometry.

    One source of truth: `local_veto` thresholds it, and `boundary` (§23) asks it
    the separate question of whether the certifier has spoken at all.
    """
    x0, y0, x1, y1 = assembly.crop
    delta = (scene_result.unexplained - base_result.unexplained)[y0:y1, x0:x1]
    scored = (np.minimum(scene_result.coverage, base_result.coverage)
              >= FC.MIN_COVERAGE)[y0:y1, x0:x1]
    box = (LOCAL_WINDOW, LOCAL_WINDOW)
    total = cv2.boxFilter((delta * scored).astype(np.float32), -1, box,
                          normalize=False)
    count = cv2.boxFilter(scored.astype(np.float32), -1, box, normalize=False)
    quorum = count >= MIN_ARBITRABLE
    return np.where(quorum, total / np.maximum(count, 1.0), 0.0), quorum


def local_veto(assembly: Assembly, scene_result, base_result, reverted,
               verbose=False):
    """Clause 2: revert every cluster the certifier prefers the input at.

    The region rule is "mean differential over >= MIN_ARBITRABLE certified
    pixels must not be positive". This asks the identical question over the
    smallest window that still holds that many certified pixels — a
    `LOCAL_WINDOW` box, i.e. the quorum made square. Where the window does NOT
    hold the quorum the clause ABSTAINS and the region verdict stands, because
    issuing a verdict on less evidence than the module already demands would be
    inventing sensitivity the certifier does not have (F113: real-scene
    differential sensitivity is a few levels, and KAT-4 measured the knob at ~7x
    under the localization floor).

    Returns the surviving rewrite mask in CROP geometry, plus a count.
    """
    x0, y0, x1, y1 = assembly.crop
    keep = assembly.rewritten & ~reverted[y0:y1, x0:x1]
    pooled, quorum = cluster_pool(assembly, scene_result, base_result)
    worse = keep & quorum & (pooled > 0.0)
    if verbose:
        print(f"    local clause: {int(worse.sum())} px reverted "
              f"({worse.sum() / max(1, keep.sum()) * 100:.1f}% of the kept "
              f"rewrite); {int((keep & ~quorum).sum())} px had no local quorum "
              f"and kept their region's verdict")
    return keep & ~worse, int(worse.sum())


def quiet_frontier(assembly: Assembly, keep, radius=FRONTIER_SLACK, verbose=False):
    """Clause 3: a rewrite may not END on a step the surface cannot hide.

    The certifier cannot arbitrate everything, and the KAT says exactly where it
    stops: a known +40-level square injected into the composite puts 65% of its
    differential mass inside itself at 9x9 and only 24% at 5x5, and the whole
    frame score moves +0.019 and +0.003 respectively. So a defect of a few dozen
    pixels is UNDER the arbiter, at any pooling scale, and clause 2 will never
    see one. Two of the manager's five defect clusters are that size.

    What is still available on a cluster that small is the frontier itself. A
    rewrite frontier is a place where the composite switches SOURCE, and the
    switch is invisible exactly where the two sources agree. This module already
    owns a per-pixel statement of how large a disagreement is explainable:
    `agreement_budget` — a GATE_TOL displacement against the local gradient,
    `normalize_exposure`'s residual, and the sensor noise floor (F112's terms,
    unchanged). Applied to the switch instead of to the admission it says: at a
    frontier pixel, `|rewrite - input|` must be within budget. Where it is not,
    that pixel and everything within `FRONTIER_SLACK` of it reverts.

    The budget's own physics puts the cost in the right place. On a smooth
    surface the gradient term vanishes and the budget collapses to a few levels,
    so a frontier crossing flat wall is cut back hard — which is precisely where
    a seam is visible, because there is no texture to hide it behind. Along a
    real edge the gradient term is tens of levels and the frontier is left
    alone.

    This is NOT feathering (F79; negative deliverable 6 of the first round):
    nothing is blended, nothing is softened, no depth boundary is crossed. The
    rewrite withdraws, hard, to where its own edge is quiet.

    THIS CLAUSE IS NOT MONOTONE IN ITS INPUT, and anything placed in front of it
    inherits that. `bad` is seeded by the loud pixels that are ON the frontier,
    so REMOVING a loud pixel upstream removes the seed and the 2 px disc it would
    have grown. Measured (§23): a clause that withdrew 4805 px before this one
    deleted 32 of the 653 loud frontier pixels, this clause's own withdrawal fell
    5976 -> 5349 px, and **438 pixels the shipped pipeline reverts survived** —
    7 of them inside the F112 knob, which took the knob from 0.95x to 1.58x and
    broke its bar. A new refusal is only safely composable AFTER this clause, where
    it can only remove. Any future clause must be priced in both orders, and the
    strict-subset assertion is what catches it.

    ONE application, deliberately. Reverting the loud frontier exposes a new
    frontier, and iterating to a fixed point is unbounded in principle and
    ruinous in measurement: at this radius three rounds take the F112 knob's
    certifier ratio from 0.95x to 2.32x, i.e. they eat the repair the round
    before this one bought. Each further round would be a fresh revert with no
    new evidence behind it. The residual loud frontier is REPORTED instead
    (kitchen: 653 px of 9489, 6.9%).
    """
    budget = agreement_budget(assembly.composite.astype(np.float32),
                              assembly.base.astype(np.float32))
    step = np.abs(assembly.composite.astype(np.float32)
                  - assembly.base.astype(np.float32)).max(axis=2)
    loud = step > budget
    inner = cv2.erode(keep.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    bad = keep & ~inner & loud
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    grown = cv2.dilate(bad.astype(np.uint8), disc).astype(bool) if bad.any() else bad
    mask = keep & ~grown
    after = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    residual = int((mask & ~after & loud).sum()), int((mask & ~after).sum())
    if verbose:
        print(f"    frontier clause: {int(bad.sum())} loud frontier px, "
              f"{int((keep & grown).sum())} reverted with them at "
              f"{radius} px slack "
              f"({(keep & grown).sum() / max(1, keep.sum()) * 100:.1f}% of the "
              f"kept rewrite); residual loud frontier {residual[0]}/{residual[1]}")
    return mask, int((keep & grown).sum()), residual


# ---------------------------------------------------------------------------
# CONTOUR CONTINUITY — the instrument F115 said was missing (round B3a)
# ---------------------------------------------------------------------------
# F115's closing diagnosis: the two residual defects are neither detail nor
# noise, they are DISPLACED CONTOURS, and no evidence the module already owns can
# see them.
#
#   * the certifier cannot: §16's KAT puts its floor at a few dozen pixels and
#     both defects are under it (6 px and ~104 px spread over 78 rows);
#   * `agreement_budget` cannot: at a high-contrast contour the budget is TENS of
#     levels (median 13.9 at the shelf junction) and a 1 px displacement of a
#     steep edge stays inside it;
#   * B1's boundary band cannot: §23b measured both defects 3.6-11.6 px away from
#     any edge the decomposition drew;
#   * and focus energy MUST NOT, because it is monotone in edge contrast and
#     blind to edge POSITION — both defects RAISE it while moving a contour
#     (§23c). That is the whole reason a new instrument is needed and not a new
#     threshold on an old one.
#
# The statement the module is missing is about POSITION, so the instrument has to
# measure position:
#
#     A rewrite may not MOVE a strong contour that the input composite and the
#     reference frame AGREE on.
#
# Both observations already live in reference geometry, so no motion is being
# fitted and F92's material/limb distinction does not apply — this is a test of
# STASIS, not of motion, and a limb that has not moved between two frames in the
# same geometry is exactly as much a fixed contour as a printed edge is.
#
# The measurement is the arc's most-validated one (PLAYBOOK: correlate GRADIENT
# profiles, not intensity; integrate along the edge; trust only the normal
# component), applied densely instead of at sparse features, and it is the right
# instrument here for the specific reason PLAYBOOK records as a HAZARD elsewhere:
# "a blurred profile correlates confidently against a sharp one at about zero
# shift". For motion estimation that is a defocus bias. For this question it is
# the required property — a legitimate sharpening must read ZERO.

# The instrument's geometry. Profiles run +-CONTOUR_HALF px along the contour
# normal and are averaged over +-CONTOUR_SPAN px along the contour. Both are
# small on purpose: the box-1 residual is 6 px in total, and the sparse fitter's
# own +-28/+-24 (`motion_groups.PROFILE_HALF/SPAN`) would average it away
# entirely. CHOSEN ON THE KAT, not on any bar: `contourkat` (b') sweeps
# half in {4, 6, 8} x span in {0, 1, 2} and prints all nine rows. The rule is
# "maximize the 1 px hit rate — the size of the defect class — with a clean 0 px
# control", and it picks (6, 1) uniquely; the table is in scenemodel_NOTES §24.
CONTOUR_HALF = 6
CONTOUR_SPAN = 1
# The displacement a contour is allowed. HALF A PIXEL is not a tuned number: it
# is the largest displacement that cannot carry a contour to a different pixel of
# the grid the composite is stored on, so below it "the contour moved" is not a
# statement about the image that was written. It is used TWICE and identically —
# to decide that the input and the reference AGREE about where a contour is, and
# to decide that the rewrite has MOVED it — so the clause says exactly one thing:
# *the rewrite may not disagree with the two observations by more than they
# disagree with each other.* `contourkat` measures the population on both sides
# of it rather than asserting it.
CONTOUR_TOL = 0.5


def _unit_normals(gray: np.ndarray):
    """Unit contour normals, from the same smoothed Sobel `_material_features` uses."""
    smoothed = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy) + 1e-6
    return (gx / magnitude).astype(np.float32), (gy / magnitude).astype(np.float32)


def _contour_profiles(gray, nx, ny, half=CONTOUR_HALF, span=CONTOUR_SPAN):
    """Intensity across the contour at EVERY pixel — `motion_groups._profile`, dense.

    One `remap` per (normal offset, tangent offset) pair over the whole frame,
    which is why a per-pixel version of a per-feature instrument is affordable at
    all. Returns `(2*half+1, H, W)`.
    """
    h, w = gray.shape
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    tx, ty = -ny, nx
    out = np.empty((2 * half + 1, h, w), np.float32)
    for j, t in enumerate(range(-half, half + 1)):
        acc = np.zeros((h, w), np.float32)
        for s in range(-span, span + 1):
            acc += cv2.remap(gray, xx + t * nx + s * tx, yy + t * ny + s * ty,
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        out[j] = acc / (2 * span + 1)
    return out


def _match_dense(pa: np.ndarray, pb: np.ndarray):
    """`motion_groups._match`, vectorized over pixels. Returns (shift, peak).

    Identical arithmetic — gradient of the mean-removed profile, a full
    normalized cross-correlation, a parabolic sub-pixel peak — so the sparse
    instrument's validation carries over, and `contourkat` (a) checks that it
    does against known sub-pixel displacements.
    """
    ga = np.gradient(pa - pa.mean(axis=0, keepdims=True), axis=0)
    gb = np.gradient(pb - pb.mean(axis=0, keepdims=True), axis=0)
    denominator = np.sqrt((ga ** 2).sum(0) * (gb ** 2).sum(0)) + 1e-12
    length = pa.shape[0]
    correlation = np.empty((2 * length - 1,) + pa.shape[1:], np.float32)
    for index, lag in enumerate(range(-(length - 1), length)):
        if lag >= 0:
            correlation[index] = (gb[lag:] * ga[:length - lag]).sum(0)
        else:
            correlation[index] = (gb[:length + lag] * ga[-lag:]).sum(0)
    correlation /= denominator
    best = np.argmax(correlation, axis=0)
    yy, xx = np.indices(pa.shape[1:])
    peak = correlation[best, yy, xx]
    left = correlation[np.maximum(best - 1, 0), yy, xx]
    right = correlation[np.minimum(best + 1, 2 * length - 2), yy, xx]
    curvature = left - 2 * peak + right
    offset = np.where(np.abs(curvature) > 1e-12,
                      0.5 * (left - right) / np.where(np.abs(curvature) > 1e-12,
                                                      curvature, 1.0), 0.0)
    interior = (best > 0) & (best < 2 * length - 2)
    shift = (best - (length - 1)).astype(np.float32) + np.where(interior, offset, 0.0)
    return shift.astype(np.float32), peak.astype(np.float32)


def _gray(image) -> np.ndarray:
    return cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8),
                        cv2.COLOR_BGR2GRAY).astype(np.float32)


def contour_continuity(candidate, base, reference, tol=CONTOUR_TOL,
                       half=CONTOUR_HALF, span=CONTOUR_SPAN):
    """Where a candidate MOVES a contour the input and the reference agree on.

    `base` is the input composite, `reference` the reference frame in the same
    crop, `candidate` the composite under test. All three are in the SAME
    geometry, which is the whole reason this question is answerable without
    fitting any motion.

    Contour sites are Canny's, on the same smoothed uint8 the repo's own edge
    finder uses (`motion_groups._material_features`), so "strong contour" is not
    a new definition either. At every site three profile matches are made along
    the site's normal:

        d_agree = shift(base -> reference)     do the two observations agree?
        d_move  = shift(base -> candidate)     did the rewrite move it?
        d_ref   = shift(reference -> candidate)

    AGREED requires a confident match (`motion_groups.MIN_PEAK`, borrowed) at
    |d_agree| <= tol. A VIOLATION requires the rewrite to have moved that contour
    past the same tol away from BOTH observations — a rewrite that moves a
    contour towards the reference is not moving a contour they agree on.
    """
    gray_b, gray_c, gray_r = _gray(base), _gray(candidate), _gray(reference)
    nx, ny = _unit_normals(gray_b)
    prof_b = _contour_profiles(gray_b, nx, ny, half, span)
    prof_c = _contour_profiles(gray_c, nx, ny, half, span)
    prof_r = _contour_profiles(gray_r, nx, ny, half, span)
    d_agree, p_agree = _match_dense(prof_b, prof_r)
    d_move, p_move = _match_dense(prof_b, prof_c)
    d_ref, p_ref = _match_dense(prof_r, prof_c)

    edges = cv2.Canny(cv2.GaussianBlur(gray_b.astype(np.uint8), (5, 5), 0),
                      60, 180) > 0
    margin = half + span + 2
    inside = np.zeros(edges.shape, bool)
    inside[margin:-margin, margin:-margin] = True
    sites = edges & inside
    agreed = sites & (p_agree >= MG.MIN_PEAK) & (np.abs(d_agree) <= tol)
    violation = (agreed & (p_move >= MG.MIN_PEAK) & (np.abs(d_move) > tol)
                 & (p_ref >= MG.MIN_PEAK) & (np.abs(d_ref) > tol))
    return {"sites": sites, "agreed": agreed, "violation": violation,
            "d_agree": d_agree, "d_move": d_move, "d_ref": d_ref,
            "p_agree": p_agree, "p_move": p_move, "p_ref": p_ref}


def steady_contours(assembly: Assembly, keep, radius=FRONTIER_SLACK, verbose=False):
    """Clause 4: a rewrite may not move a contour the two observations agree on.

    Composed LAST, and that placement is forced rather than chosen. §23a measured
    `quiet_frontier` to be NON-MONOTONE in its input — its `bad` set is seeded by
    loud pixels lying ON the frontier, so deleting a rewrite pixel upstream can
    delete a seed and make it withdraw LESS (438 pixels survived that the shipped
    pipeline reverts, 7 of them inside the F112 knob, taking it from 0.95x to
    1.58x). A clause placed AFTER it can only remove, and the strict-subset
    assertion is what proves it did.

    The candidate tested is the composite the pipeline would actually ship at
    this point — `keep` applied to the base — not the raw assembly, because a
    contour displacement the earlier clauses already reverted is not this
    clause's to punish.

    Reversion reuses `quiet_frontier`'s machinery unchanged: the violating site
    plus everything within `FRONTIER_SLACK = ceil(GATE_TOL)` px of it, once. A
    displaced contour is a statement about a 1-2 px neighbourhood and GATE_TOL is
    this arc's standing answer to "how far is a position undetermined"; the disc
    is what reaches the rewritten pixels that did the displacing when the
    violating site itself sits just outside the rewrite. ONE application, for
    `quiet_frontier`'s reason: reverting to the input restores the agreed
    contour, so a second pass would be a fresh revert with no new evidence.
    """
    candidate = assembly.base.copy()
    candidate[keep] = assembly.composite[keep]
    report = contour_continuity(candidate, assembly.base, assembly.reference)
    changed = cv2.dilate((np.abs(candidate.astype(np.int16)
                                 - assembly.base.astype(np.int16)).max(axis=2) > 0
                          ).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    violation = report["violation"] & changed
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    grown = (cv2.dilate(violation.astype(np.uint8), disc).astype(bool)
             if violation.any() else violation)
    mask = keep & ~grown
    tested = report["agreed"] & changed
    if verbose:
        print(f"    contour clause: {int(violation.sum())} displaced contour px of "
              f"{int(tested.sum())} agreed contour px in the rewrite "
              f"({violation.sum() / max(1, tested.sum()) * 100:.2f}%); "
              f"{int((keep & grown).sum())} px reverted at {radius} px slack "
              f"({(keep & grown).sum() / max(1, keep.sum()) * 100:.1f}% of the "
              f"kept rewrite)")
    return mask, int((keep & grown).sum()), report


def finalize(assembly: Assembly, reverted, rewrite=None):
    """Apply the veto and return the final composite (crop geometry).

    `rewrite` overrides the region veto's own mask with the one the local
    clauses left; without it this is the region-only behaviour, which is what
    the ledgers compare against.
    """
    x0, y0, x1, y1 = assembly.crop
    if rewrite is None:
        rewrite = assembly.rewritten & ~reverted[y0:y1, x0:x1]
    out = assembly.base.copy()
    out[rewrite] = assembly.composite[rewrite]
    assert np.array_equal(out[~rewrite], assembly.base[~rewrite])
    return out, rewrite


def veto_all(assembly: Assembly, scene_result, base_result, regions,
             verbose=False, contour=True):
    """The whole FOUR-scale veto, in order, and the composite it leaves.

    The order is not a preference. Clauses 1-3 are the correction round's
    (region, cluster, frontier); clause 4 is round B3a's contour-continuity
    rule, and §23a forces it to be LAST — `quiet_frontier` is non-monotone in its
    input, so anything in front of it can make it withdraw LESS. Placed after it,
    a clause can only remove, and the assertion below proves it did.
    """
    verdicts, reverted = apply_veto(assembly, scene_result, base_result,
                                    regions, verbose=verbose)
    keep, n_local = local_veto(assembly, scene_result, base_result, reverted,
                               verbose=verbose)
    keep, n_front, residual = quiet_frontier(assembly, keep, verbose=verbose)
    shipped_106e2f5 = keep.copy()
    n_contour, report = 0, None
    if contour:
        keep, n_contour, report = steady_contours(assembly, keep, verbose=verbose)
        # THE STRICT-SUBSET LEDGER (§17, §23a). The instrument that catches a
        # non-monotone composition is the assertion, not the bar table.
        assert not (keep & ~shipped_106e2f5).any(), \
            "the contour clause rescued pixels 106e2f5 reverts"
    final, final_rewrite = finalize(assembly, reverted, keep)
    stats = {"local": n_local, "frontier": n_front, "residual": residual,
             "contour": n_contour, "contour_report": report,
             "shipped_106e2f5": shipped_106e2f5}
    return verdicts, reverted, final, final_rewrite, stats


# ---------------------------------------------------------------------------
# KAT — the physical same-surface test, against the questions F112/R3 asked
# ---------------------------------------------------------------------------
def _kat_scene():
    """The COMMITTED fixture, unchanged — `tests/test_twoframe_route.py`'s own.

    Reusing the existing KAT's scene and its pass marks is the point: it makes
    the physical test and the global-sigma one directly comparable on the four
    questions F112/R3 asked, instead of on a fixture chosen after the fact.
    """
    rng = np.random.default_rng(4)
    texture = cv2.GaussianBlur(rng.integers(0, 255, (300, 400)).astype(np.uint8),
                               (0, 0), 2)
    scene = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR).astype(np.float32)
    inner = np.zeros(scene.shape[:2], bool)
    inner[40:-40, 40:-40] = True
    return scene, inner


def kat() -> None:
    """Known-answer test before belief (§12.1), and directly comparable to R3.

    R3's table for the GLOBAL sigma=4 version is quoted alongside, measured on
    the same fixture. The row that must differ is the defocus row: a global
    low-pass can only absorb a defocus difference up to its own scale, so R3
    measured it failing at 8 px (0.931) and 12 px (0.759). A CROSS-CONVOLVED
    test has no scale to run out of.
    """
    scene, inner = _kat_scene()
    zero = np.zeros(scene.shape[:2], np.float32)
    ones = np.ones(scene.shape[:2], np.float32)
    print("=" * 78)
    print("KAT — the PHYSICAL same-surface test, against known answers")
    print("=" * 78)
    print("  Fixture and pass marks are the committed ones\n"
          "  (`tests/test_twoframe_route.py::test_same_surface_is_blind_to_defocus`).\n"
          "  Agreement is measured on the inner region, away from the borders.\n")

    print("  (a) DISK DEFOCUS — the premise. Bar > 0.97.")
    print(f"  {'radius (px)':>12} {'agreement':>10} {'verdict':>8}   R3 sigma=4")
    r3 = {1: "1.000", 2: "1.000", 4: "0.999", 6: "0.983", 8: "0.931", 12: "0.759"}
    for radius in (1, 2, 4, 6, 8, 12):
        agree = same_surface_physical(FC.defocus(scene, radius), scene,
                                      ones * radius, zero)
        value = float(agree[inner].mean())
        print(f"  {radius:12d} {value:10.3f} {'PASS' if value > 0.97 else 'FAIL':>8}"
              f"   {r3[radius]:>10}")

    print(f"\n  (b) A DISPLACEMENT THE GEOMETRY TOLERATES must not trip it "
          f"(GATE_TOL = {TF.GATE_TOL} px). Bar > 0.97 up to GATE_TOL.")
    print(f"  {'shift (px)':>12} {'agreement':>10} {'verdict':>8}   R3 sigma=4")
    r3 = {0.5: "1.000", 1.0: "1.000", 1.5: "1.000", 2.0: "0.981", 4.0: "0.800"}
    for shift in (0.5, 1.0, TF.GATE_TOL, 2.0, 4.0):
        moved = cv2.warpAffine(scene, np.float32([[1, 0, shift], [0, 1, 0]]),
                               (scene.shape[1], scene.shape[0]),
                               borderMode=cv2.BORDER_REFLECT)
        agree = same_surface_physical(moved, scene, zero, zero)
        value = float(agree[inner].mean())
        mark = "PASS" if (shift > TF.GATE_TOL or value > 0.97) else "FAIL"
        print(f"  {shift:12.1f} {value:10.3f} {mark:>8}   {r3[shift]:>10}")

    print(f"\n  (c) THE EXPOSURE RESIDUAL must not trip it "
          f"(SURFACE_GAIN = {TF.SURFACE_GAIN}; measured max on the sweep 1.85%).")
    print(f"  {'gain':>12} {'agreement':>10} {'verdict':>8}   R3 sigma=4")
    r3 = {1.015: "1.000", 1.019: "1.000", 1.05: "0.455"}
    for gain in (1.015, 1.019, 1.05):
        agree = same_surface_physical(np.clip(scene * gain, 0, 255), scene,
                                      zero, zero)
        value = float(agree[inner].mean())
        mark = "PASS" if (gain > 1.02 or value > 0.97) else "FAIL"
        print(f"  {gain:12.3f} {value:10.3f} {mark:>8}   {r3[gain]:>10}")

    print("\n  (d) A MOVED OCCLUDER must trip it, over the strip it vacated and the\n"
          "      strip it now covers, and nowhere else. Bars < 0.05 and > 0.95.")
    print(f"  {'move (px)':>12} {'strip':>10} {'elsewhere':>10} {'verdict':>8}"
          f"   R3 sigma=4")
    for step in (4, 8, 20):
        here, there = scene.copy(), scene.copy()
        here[100:200, 120:220] = (240, 240, 240)
        there[100:200, 120 + step:220 + step] = (240, 240, 240)
        agree = same_surface_physical(there, here, zero, zero)
        strip = np.zeros(scene.shape[:2], bool)
        strip[100:200, 120:120 + step] = True
        strip[100:200, 220:220 + step] = True
        band = np.zeros(scene.shape[:2], bool)
        band[90:210, 110:230 + step] = True
        a, b = float(agree[strip].mean()), float(agree[inner & ~band].mean())
        mark = "PASS" if (a < 0.05 and b > 0.95) else "FAIL"
        print(f"  {step:12d} {a:10.3f} {b:10.3f} {mark:>8}   "
              f"{'0.000 / 0.98':>10}")

    print("\n  (e) THE INSTRUMENT'S REAL LIMIT, measured rather than left implicit:\n"
          "      an occluder whose replacement LOOKS like what it replaced. Same\n"
          "      construction, but the moved surface is the scene's own texture\n"
          "      rolled sideways instead of a flat bright square.")
    print(f"  {'move (px)':>12} {'strip':>10} {'elsewhere':>10}")
    rolled = np.roll(scene, 137, axis=1)
    for step in (4, 8, 20):
        here, there = scene.copy(), scene.copy()
        here[100:200, 120:220] = rolled[100:200, 120:220]
        there[100:200, 120 + step:220 + step] = rolled[100:200, 120:220]
        agree = same_surface_physical(there, here, zero, zero)
        strip = np.zeros(scene.shape[:2], bool)
        strip[100:200, 120:120 + step] = True
        strip[100:200, 220:220 + step] = True
        band = np.zeros(scene.shape[:2], bool)
        band[90:210, 110:230 + step] = True
        print(f"  {step:12d} {float(agree[strip].mean()):10.3f} "
              f"{float(agree[inner & ~band].mean()):10.3f}")
    print("      An appearance test cannot separate two surfaces that look the\n"
          "      same. That is a property of the question, not of this build, and\n"
          "      it is why VISIBILITY is checked geometrically before admission.")

    print("\n  (f) SIGMA0's own sensitivity — the one number left, against the two\n"
          "      verdicts that matter (defocus blindness at 12 px, occlusion\n"
          "      detection at 8 px).")
    print(f"  {'sigma0':>12} {'defocus-12':>11} {'occluder-8':>11} {'elsewhere':>10}")
    here, there = scene.copy(), scene.copy()
    here[100:200, 120:220] = (240, 240, 240)
    there[100:200, 128:228] = (240, 240, 240)
    strip = np.zeros(scene.shape[:2], bool)
    strip[100:200, 120:128] = True
    strip[100:200, 220:228] = True
    band = np.zeros(scene.shape[:2], bool)
    band[90:210, 110:238] = True
    for sigma0 in (0.5, 1.0, 2.0):
        a1 = same_surface_physical(FC.defocus(scene, 12), scene, ones * 12.0,
                                   zero, sigma0=sigma0)
        a2 = same_surface_physical(there, here, zero, zero, sigma0=sigma0)
        print(f"  {sigma0:12.1f} {float(a1[inner].mean()):11.3f} "
              f"{float(a2[strip].mean()):11.3f} "
              f"{float(a2[inner & ~band].mean()):10.3f}")


def slope_kat() -> None:
    """The blur ladder's known answer: the factory's own BLUR_PER_STEP = 1.15."""
    import parallax_gen as P

    frames, _truth, _near = P.build_stack()
    print("=" * 78)
    print("KAT — the per-scene blur slope, against the factory's own constant")
    print("=" * 78)
    dec = LD.decompose(frames, P.REFERENCE)
    model, _diag = FC.model_from_pass1(frames, P.REFERENCE,
                                       segmentation=dec.segmentation())
    composite, info = TF.twoframe_stack(frames, P.REFERENCE)
    crop = tuple(info["crop"])
    null = frames[P.REFERENCE][crop[1]:crop[3], crop[0]:crop[2]]
    null_canvas, null_support = FC.place(null, crop, model.shape)
    null_app, null_sup = FC.layer_views(model, null_canvas, null_support)
    FC.select_geometry(model, null_app, null_sup, frames)
    canvas, support = FC.place(composite, crop, model.shape)
    appearances, supports = FC.layer_views(model, canvas, support)
    slope, rows = blur_slope(model, appearances, supports, frames, dec.peaks,
                             verbose=True)
    print(f"  measured c = {slope:.3f} px/frame   TRUTH "
          f"BLUR_PER_STEP = {P.BLUR_PER_STEP}   error "
          f"{abs(slope - P.BLUR_PER_STEP) / P.BLUR_PER_STEP * 100:.1f}%")
    print(f"  {'frame':>6} {'layer':>6} {'|k-peak|':>9} {'radius':>7} {'c*d':>7}")
    for k, i, distance, radius in sorted(rows):
        print(f"  {k:6d} {i:6d} {distance:9.2f} {radius:7d} {slope * distance:7.2f}")


def local_kat() -> None:
    """KAT for the LOCAL clause's instrument: does the differential localize?

    §12.1 — a new instrument gets a known-answer test before it is believed, and
    `local_veto` is a new instrument: it asks the certifier a question the
    certifier has only ever been asked about whole regions. The known answer is
    injected: a square of known SIZE and known amplitude, pasted into the input
    composite at three well-certified sites. A perfect arbiter would put all of
    the extra unexplained residual inside the square. A forward renderer cannot,
    because it convolves the composite with each layer's defocus disk before
    comparing — so this measures the certifier's own point spread, and with it
    the size below which the local clause is guaranteed blind.
    """
    from focusstack.io import normalize_exposure

    src = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(KITCHEN, "*.jpg")))]
    norm = normalize_exposure(src)
    ref = len(src) // 2
    print("=" * 78)
    print("KAT — the LOCAL clause's arbiter, against injected defects of known size")
    print("=" * 78)
    result = assemble(norm, ref, verbose=True, raw=src)
    x0, y0, _x1, _y1 = result.crop
    sites = [(300, 200), (560, 300), (200, 380)]
    candidates = [("base", result.base)]
    for side in (5, 9, 15, 25):
        image = result.base.copy()
        for (cx, cy) in sites:
            half = side // 2
            patch = image[cy - half:cy + half + 1, cx - half:cx + half + 1]
            image[cy - half:cy + half + 1, cx - half:cx + half + 1] = np.clip(
                patch.astype(np.int32) + 40, 0, 255).astype(np.uint8)
        candidates.append((f"square{side}", image))
    scores, _region = certify_candidates(result, src, candidates)
    base = scores["base"]

    print(f"\n  three {'+40 level'} squares per candidate, at "
          f"{', '.join(str(s) for s in sites)} (composite coords)\n")
    print(f"  {'side':>5} {'frame score':>12} {'mass inside':>12} {'within 10 px':>13} "
          f"{'pooled peak K=7':>16} {'K=15':>7} {'K=25':>7}")
    yy, xx = np.indices(base.unexplained.shape)
    for side in (5, 9, 15, 25):
        entry = scores[f"square{side}"]
        delta = entry.unexplained - base.unexplained
        scored = (np.minimum(entry.coverage, base.coverage) >= FC.MIN_COVERAGE)
        positive = np.maximum(delta, 0) * scored
        total = max(float(positive.sum()), 1e-9)
        inside = np.zeros(delta.shape, bool)
        near = np.zeros(delta.shape, bool)
        half = side // 2
        for (cx, cy) in sites:
            inside[cy + y0 - half:cy + y0 + half + 1,
                   cx + x0 - half:cx + x0 + half + 1] = True
            near |= ((xx - (cx + x0)) ** 2 + (yy - (cy + y0)) ** 2) <= 100
        peaks = []
        for window in (7, LOCAL_WINDOW, 25):
            box = (window, window)
            mass = cv2.boxFilter((delta * scored).astype(np.float32), -1, box,
                                 normalize=False)
            count = cv2.boxFilter(scored.astype(np.float32), -1, box,
                                  normalize=False)
            peaks.append(float(np.where(count > 0,
                                        mass / np.maximum(count, 1), 0)[inside].max()))
        print(f"  {side:5d} {entry.score - base.score:+12.4f} "
              f"{float(positive[inside].sum()) / total:11.1%} "
              f"{float(positive[near].sum()) / total:12.1%} "
              f"{peaks[0]:16.2f} {peaks[1]:7.2f} {peaks[2]:7.2f}")
    print("\n  Read the 5 and 9 rows against each other. A 9x9 defect is arbitrable\n"
          "  — two thirds of its residual lands on itself and the frame score moves\n"
          "  by 0.019 levels. A 5x5 one is NOT: a quarter lands on itself and the\n"
          "  frame moves 0.003, which is under the certifier's own real-scene\n"
          "  differential sensitivity. So the local clause has a floor of a few\n"
          "  dozen pixels no matter what window it pools over, and the defects that\n"
          "  fall through it are exactly why `quiet_frontier` exists.")


def _displace_strip(image, box, shift):
    """Translate one rectangle of an image by `shift` px in x. A KNOWN answer."""
    bx0, by0, bx1, by1 = box
    out = image.copy()
    strip = image[by0:by1, bx0 - 8:bx1 + 8].astype(np.float32)
    matrix = np.float32([[1, 0, shift], [0, 1, 0]])
    moved = cv2.warpAffine(strip, matrix, (strip.shape[1], strip.shape[0]),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    out[by0:by1, bx0:bx1] = np.clip(moved[:, 8:-8], 0, 255).astype(np.uint8)
    return out


def contour_kat() -> None:
    """KAT for the CONTOUR CONTINUITY instrument, before it is believed (§12.1).

    Four questions, in the order that makes each one interpretable:

      (a) does the dense correlator recover a KNOWN sub-pixel displacement, and
          does it read ZERO when the only change is sharpness? (synthetic, so the
          answer is exact and any failure is arithmetic, not scene)
      (b) does it trip on a known 1 px and 2 px displacement injected into a real
          composite, and what is the hit rate?
      (c) what does it do to the factory's own legitimate sharpening — a
          sharper-but-unmoved contour, which the assembly produces in abundance?
          The factory has GROUND TRUTH, so every flag can be adjudicated: a flag
          is a FALSE ALARM only if the candidate's contour is closer to the truth
          than the input's is.
      (d) does it flag the two kitchen residuals from their COORDINATES ALONE?
          The instrument is not told the answer; it is run on the whole frame and
          the boxes are read off afterwards.
    """
    import parallax_gen as P
    from focusstack.io import normalize_exposure

    print("=" * 78)
    print("KAT — CONTOUR CONTINUITY, the instrument F115 said was missing")
    print("=" * 78)

    # (a) ------------------------------------------------------------------
    print("\n  (a) THE CORRELATOR, against exactly-known answers. A step edge in a\n"
          "      textured field, displaced by a known amount and/or blurred.")
    rng = np.random.default_rng(11)
    field = cv2.GaussianBlur(rng.integers(20, 90, (200, 200)).astype(np.uint8),
                             (0, 0), 3).astype(np.float32)
    edge = field.copy()
    edge[:, 100:] += 150.0
    edge = np.clip(edge, 0, 255)
    base_rgb = cv2.cvtColor(edge.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    probe = np.zeros(edge.shape, bool)
    probe[40:160, 96:104] = True
    print(f"  {'candidate':<34} {'measured shift':>15} {'truth':>8} {'peak':>7}")
    cases = [("identity", 0.0, 0.0), ("shift +0.25 px", 0.25, 0.25),
             ("shift +0.50 px", 0.5, 0.5), ("shift +1.00 px", 1.0, 1.0),
             ("shift +2.00 px", 2.0, 2.0)]
    for label, shift, truth in cases:
        moved = cv2.warpAffine(edge, np.float32([[1, 0, shift], [0, 1, 0]]),
                               (200, 200), borderMode=cv2.BORDER_REFLECT)
        rgb = cv2.cvtColor(np.clip(moved, 0, 255).astype(np.uint8),
                           cv2.COLOR_GRAY2BGR)
        out = contour_continuity(rgb, base_rgb, base_rgb)
        here = probe & out["sites"]
        print(f"  {label:<34} {float(np.median(out['d_move'][here])):15.3f} "
              f"{truth:8.2f} {float(np.median(out['p_move'][here])):7.3f}")
    for radius in (1, 2, 4):
        blurred = FC.defocus(cv2.cvtColor(edge.astype(np.uint8),
                                          cv2.COLOR_GRAY2BGR), radius)
        out = contour_continuity(base_rgb, blurred, base_rgb)
        here = probe & out["sites"]
        print(f"  {'SHARPENED (disk %d -> sharp)' % radius:<34} "
              f"{float(np.median(out['d_move'][here])):15.3f} {0.0:8.2f} "
              f"{float(np.median(out['p_move'][here])):7.3f}"
              f"   flagged {int((out['violation'] & probe).sum())}"
              f" of {int((out['agreed'] & probe).sum())}")
    print("      The sharpening rows are the load-bearing ones: a legitimate\n"
          "      sharpening must read ZERO, and PLAYBOOK's own defocus-bias\n"
          "      HAZARD ('a blurred profile correlates confidently against a sharp\n"
          "      one at about zero shift') is exactly the property that makes it.")

    # (b) / (c) -------------------------------------------------------------
    frames, truth, _near = P.build_stack()
    result = assemble(frames, P.REFERENCE, raw=frames)
    x0, y0, x1, y1 = result.crop
    reference_truth = truth[y0:y1, x0:x1]

    print("\n  (b) INJECTED DISPLACEMENTS in a real composite: one strip of the\n"
          "      factory's own TRUTH, translated by a known amount, pasted into\n"
          "      the input composite. Everything else is byte-identical.")
    strip_box = (120, 120, 420, 300)
    inside = np.zeros(result.base.shape[:2], bool)
    inside[strip_box[1]:strip_box[3], strip_box[0]:strip_box[2]] = True
    outside = ~cv2.dilate(inside.astype(np.uint8),
                          np.ones((2 * CONTOUR_HALF + 1,) * 2, np.uint8)).astype(bool)
    truth_base = result.base.copy()
    truth_base[inside] = reference_truth[inside]
    # A translation in x moves a contour by `shift * nx` along ITS OWN normal, so
    # a horizontal contour translated horizontally has NOT moved and must not be
    # flagged. That is the aperture problem, and it is the reason the raw hit
    # rate is the wrong denominator: the KNOWN answer is per site, not per strip.
    nx_map, _ny = _unit_normals(_gray(result.base))
    print(f"  {'injected':<12} {'agreed':>7} {'observable':>11} {'FLAGGED':>8} "
          f"{'raw':>7} {'HIT RATE':>9} {'outside':>9}")
    for shift in (0.0, 1.0, 2.0):
        candidate = (truth_base if shift == 0.0
                     else _displace_strip(truth_base, strip_box, shift))
        out = contour_continuity(candidate, result.base, reference_truth)
        agreed_in = out["agreed"] & inside
        observable = agreed_in & (np.abs(shift * nx_map) > CONTOUR_TOL)
        flagged = out["violation"] & inside
        false_out = (out["violation"] & outside).sum() / max(
            1, (out["agreed"] & outside).sum())
        print(f"  {shift:5.1f} px {'':<3} {int(agreed_in.sum()):7d} "
              f"{int(observable.sum()):11d} {int(flagged.sum()):8d} "
              f"{flagged.sum() / max(1, agreed_in.sum()):6.1%} "
              f"{(flagged & observable).sum() / max(1, observable.sum()):8.1%} "
              f"{false_out:8.2%}")
    print("      Row 0.0 is the control: the same paste with NO displacement.\n"
          "      'observable' counts only the sites whose own normal actually SEES\n"
          "      the injected translation (|shift*nx| > tol); HIT RATE is against\n"
          "      those, and the raw column is against every agreed site in the\n"
          "      strip including the ones the displacement slides ALONG.\n"
          "      'outside' is the same instrument on the untouched remainder.")

    print("\n  (b') THE PROFILE GEOMETRY, chosen HERE and not on any bar. Nine\n"
          "       (half, span) pairs against the 1 px and 2 px hit rates and the\n"
          "       0 px control. The rule is: maximize the 1 px hit rate — the size\n"
          "       of the defect class this exists for — with a clean control.")
    print(f"  {'half':>5} {'span':>5} {'hit @1 px':>11} {'hit @2 px':>11} "
          f"{'0 px control':>13}")
    for half in (4, 6, 8):
        for span in (0, 1, 2):
            row = []
            for shift in (1.0, 2.0, 0.0):
                candidate = (truth_base if shift == 0.0
                             else _displace_strip(truth_base, strip_box, shift))
                out = contour_continuity(candidate, result.base, reference_truth,
                                         half=half, span=span)
                flagged = out["violation"] & inside
                observable = (out["agreed"] & inside
                              & (np.abs(shift * nx_map) > CONTOUR_TOL))
                row.append((flagged & observable).sum() / max(1, observable.sum())
                           if shift else int(flagged.sum()))
            mark = "  <- shipped" if (half, span) == (CONTOUR_HALF,
                                                      CONTOUR_SPAN) else ""
            print(f"  {half:5d} {span:5d} {row[0]:10.1%} {row[1]:10.1%} "
                  f"{row[2]:13d}{mark}")

    print("\n  (c) THE FACTORY'S OWN SHARPENING, adjudicated against GROUND TRUTH.\n"
          "      The candidate is the assembled composite after the SHIPPED veto\n"
          "      stack. A flag is a FALSE ALARM only if the rewrite put the\n"
          "      contour CLOSER to the truth than the input had it.")
    regions = regions_of(result)
    scores, _r = certify_candidates(result, frames,
                                    [("scene-model", result.composite),
                                     ("input routed", result.base)])
    scene, base_score = scores["scene-model"], scores["input routed"]
    _v, reverted = apply_veto(result, scene, base_score, regions)
    keep, _n = local_veto(result, scene, base_score, reverted)
    keep, _n, _res = quiet_frontier(result, keep)
    shipped = result.base.copy()
    shipped[keep] = result.composite[keep]
    out = contour_continuity(shipped, result.base, result.reference)
    changed = cv2.dilate((np.abs(shipped.astype(np.int16)
                                 - result.base.astype(np.int16)).max(axis=2) > 0
                          ).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    tested = out["agreed"] & changed
    flagged = out["violation"] & changed
    # adjudicate: where does the TRUTH put the contour?
    gray_b, gray_c = _gray(result.base), _gray(shipped)
    gray_t = _gray(reference_truth)
    nx, ny = _unit_normals(gray_b)
    pb = _contour_profiles(gray_b, nx, ny)
    d_bt, _p = _match_dense(pb, _contour_profiles(gray_t, nx, ny))
    d_ct, _p = _match_dense(_contour_profiles(gray_c, nx, ny),
                            _contour_profiles(gray_t, nx, ny))
    closer = np.abs(d_ct) < np.abs(d_bt)
    n_flag = int(flagged.sum())
    print(f"      agreed contour px inside the rewrite : {int(tested.sum())}")
    print(f"      FLAGGED                              : {n_flag} "
          f"({n_flag / max(1, tested.sum()):.2%} of them)")
    if n_flag:
        print(f"      of the flagged, closer to TRUTH after the rewrite "
              f"(= FALSE ALARM): {int((flagged & closer).sum())} "
              f"({(flagged & closer).sum() / n_flag:.0%} of flags, "
              f"{(flagged & closer).sum() / max(1, tested.sum()):.3%} of all "
              f"agreed contour px in the rewrite)")
        print(f"      median |contour - truth|: input {float(np.median(np.abs(d_bt[flagged]))):.2f} px"
              f"  ->  rewrite {float(np.median(np.abs(d_ct[flagged]))):.2f} px")
    print(f"      the same numbers on the contours the clause KEEPS: "
          f"median |contour - truth| input "
          f"{float(np.median(np.abs(d_bt[tested & ~flagged]))):.2f} px -> rewrite "
          f"{float(np.median(np.abs(d_ct[tested & ~flagged]))):.2f} px")

    # (d) -------------------------------------------------------------------
    print("\n  (d) THE TWO KITCHEN RESIDUALS, from their COORDINATES ALONE. The\n"
          "      instrument is run on the whole frame with no knowledge of either\n"
          "      box; the boxes are read off the result afterwards.")
    src = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(KITCHEN, "*.jpg")))]
    norm = normalize_exposure(src)
    ref = len(src) // 2
    kitchen_result = assemble(norm, ref, raw=src)
    kregions = regions_of(kitchen_result)
    kscores, _r = certify_candidates(kitchen_result, src,
                                     [("scene-model", kitchen_result.composite),
                                      ("input routed", kitchen_result.base)])
    kscene, kbase = kscores["scene-model"], kscores["input routed"]
    _v, kreverted = apply_veto(kitchen_result, kscene, kbase, kregions)
    kkeep, _n = local_veto(kitchen_result, kscene, kbase, kreverted)
    kkeep, _n, _res = quiet_frontier(kitchen_result, kkeep)
    kshipped = kitchen_result.base.copy()
    kshipped[kkeep] = kitchen_result.composite[kkeep]
    kout = contour_continuity(kshipped, kitchen_result.base,
                              kitchen_result.reference)
    kchanged = cv2.dilate((np.abs(kshipped.astype(np.int16)
                                  - kitchen_result.base.astype(np.int16)
                                  ).max(axis=2) > 0).astype(np.uint8),
                          np.ones((3, 3), np.uint8)).astype(bool)
    kflag = kout["violation"] & kchanged
    ktest = kout["agreed"] & kchanged
    # the two residuals, in COMPOSITE coords (inspection + 1 in x, §15)
    residuals = {"box 1 fleck  (x482-484 y150-153, insp)": (483, 151, 486, 154),
                 "box 4 shelf junction (x431-492 y192-270)": (431, 192, 492, 270)}
    print(f"  {'residual':<44} {'agreed':>7} {'FLAGGED':>8} {'max |d|':>8}")
    for label, (bx0, by0, bx1, by1) in residuals.items():
        window = np.zeros(kflag.shape, bool)
        window[by0:by1, bx0:bx1] = True
        here = kflag & window
        agreed_here = ktest & window
        peak = (float(np.abs(kout["d_move"][here]).max()) if here.any() else 0.0)
        print(f"  {label:<44} {int(agreed_here.sum()):7d} {int(here.sum()):8d} "
              f"{peak:8.2f}")
    print(f"\n      whole kitchen rewrite: {int(kflag.sum())} flagged of "
          f"{int(ktest.sum())} agreed contour px "
          f"({kflag.sum() / max(1, ktest.sum()):.2%})")


# ---------------------------------------------------------------------------
# The canonical kitchen instruments (§12.2 — scope the metric to the thing)
# ---------------------------------------------------------------------------
# Every one of these was reproduced against its recorded value before being used
# on anything new. Reproductions, on the INPUT routed composite:
#   * the four F112/R3 user boxes: maxima 61 / 17 / 101 / 127, exactly R3's table;
#   * the canonical F108 flank box: mean 1.114, max 45, 0.57% > 12, exactly
#     F112's instrument note (1.11 / 45 / 0.57%), and 0.00% with the knob box
#     removed — which is what "the entire >12 tail is one knob" means, verified.
USER_BOXES = {1: (473, 135, 506, 156),    # background pot in front of the bottle
              2: (562, 48, 610, 124),     # second copy of the bottle lid
              3: (222, 125, 243, 195),    # Coca-Cola right edge aliased right
              4: (444, 216, 475, 257)}    # blurry alias of the yellow rag
KNOB = (659, 243, 670, 314)               # F112's dark background knob, ORIGINAL
FLANK_BOX = (560, 240, 670, 420)          # F108's canonical flank, ORIGINAL


def _delta_map(composite, reference, crop):
    """|composite - reference| in ORIGINAL coordinates, max over channels."""
    x0, y0, _x1, _y1 = crop
    h, w = composite.shape[:2]
    out = np.full(reference.shape[:2], np.nan, np.float32)
    out[y0:y0 + h, x0:x0 + w] = np.abs(
        composite.astype(np.float32)
        - reference[y0:y0 + h, x0:x0 + w].astype(np.float32)).max(axis=2)
    return out


def kitchen_boxes(composite, reference, crop, label="", energy=False):
    """The four user boxes, in COMPOSITE coordinates, against the reference.

    F112/R6.4 rejected this metric as an ARBITER and the reason matters here:
    it is scored against the reference frame, so "refuse everything" scores
    perfectly and any legitimate sharpening of a locally-defocused background
    scores WORSE. It is reported because it is the recorded bar, and it is
    reported next to the box's mean FOCUS ENERGY — F112/R5.2's own instrument —
    which moves the opposite way when the change is a sharpening rather than a
    defect. Neither number is a verdict alone; the certifier's region ledger is.
    """
    from focusstack.focus import focus_measure

    x0, y0, _x1, _y1 = crop
    row = []
    for i in (1, 2, 3, 4):
        bx0, by0, bx1, by1 = USER_BOXES[i]
        a = composite[by0:by1, bx0:bx1].astype(np.float32)
        b = reference[y0 + by0:y0 + by1, x0 + bx0:x0 + bx1].astype(np.float32)
        delta = np.abs(a - b)
        focus = float(focus_measure(
            to_gray_float(composite).astype(np.float32))[by0:by1, bx0:bx1].mean())
        row.append((float(delta.mean()), float(delta.max()), focus))
    if label:
        cells = ("  ".join(f"{f:10.1f}" for _m, _p, f in row) if energy
                 else "  ".join(f"{m:6.2f}/{p:3.0f}" for m, p, _f in row))
        print(f"  {label:<26} {cells}")
    return row


def kitchen_flank(composite, reference, crop, label=""):
    delta = _delta_map(composite, reference, crop)
    box = np.zeros(delta.shape, bool)
    box[FLANK_BOX[1]:FLANK_BOX[3], FLANK_BOX[0]:FLANK_BOX[2]] = True
    knob = np.zeros(delta.shape, bool)
    knob[KNOB[1]:KNOB[3], KNOB[0]:KNOB[2]] = True
    values = delta[box]
    outside = box & ~knob & (delta > 12)
    stats = (float(np.nanmean(values)), float(np.nanmax(values)),
             100.0 * float(np.nanmean(values > 12)),
             100.0 * float(np.nanmean(delta[box & ~knob] > 12)))
    if label:
        where = ""
        if outside.any():
            ys, xs = np.nonzero(outside)
            # WHERE the tail is, not just how big: the knob's own recorded box is
            # a 10x70 box for a 30x70 object (round A, §7.3), so a tail one or
            # two rows outside it is still the knob.
            where = (f"  [{int(outside.sum())} px at x{xs.min()}-{xs.max()} "
                     f"y{ys.min()}-{ys.max()}]")
        print(f"  {label:<26} mean {stats[0]:5.3f}  max {stats[1]:5.0f}  "
              f">12 {stats[2]:5.2f}%  (knob box removed {stats[3]:.2f}%){where}")
    return stats


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def factory() -> None:
    """Bar A: the factory GT-SSIM, and the remainder attributed on the ladder."""
    import metrics
    import parallax_gen as P

    os.makedirs(OUT, exist_ok=True)
    frames, truth, near = P.build_stack()
    print("=" * 78)
    print("FACTORY — bar A: GT-SSIM of the scene-model composite")
    print("=" * 78)
    result = assemble(frames, P.REFERENCE, verbose=True, raw=frames)
    x0, y0, x1, y1 = result.crop
    reference_truth = truth[y0:y1, x0:x1]

    regions = regions_of(result)
    scores, region_mask = certify_candidates(
        result, frames,
        [("scene-model", result.composite), ("input routed", result.base),
         ("region-scoped null", result.scoped_null),
         ("global null", frames[P.REFERENCE][y0:y1, x0:x1])])
    verdicts, reverted, final, final_rewrite, vstats = veto_all(
        result, scores["scene-model"], scores["input routed"], regions,
        verbose=True)
    region_only, region_rewrite = finalize(result, reverted)
    final_scores, _r = certify_candidates(
        result, frames, [("scene-model, vetoed", final),
                         ("region veto only", region_only)])
    scores.update(final_scores)

    print(f"\n  {'candidate':<26} {'GT-SSIM':>10} {'certifier':>10} {'p99':>7} "
          f"{'cert%':>7}")
    ssims = {}
    for label, image in (("input routed (the bar)", result.base),
                         ("scene-model, raw", result.composite),
                         ("region veto only (B2)", region_only),
                         ("scene-model, vetoed", final),
                         ("region-scoped null", result.scoped_null),
                         ("GROUND TRUTH", reference_truth)):
        ssims[label] = float(metrics.ref_ssim(image, reference_truth))
        key = {"input routed (the bar)": "input routed",
               "scene-model, raw": "scene-model",
               "region veto only (B2)": "region veto only",
               "scene-model, vetoed": "scene-model, vetoed",
               "region-scoped null": "region-scoped null"}.get(label)
        entry = scores.get(key)
        cert = (f"{entry.score:10.4f} {entry.p99:7.3f} {entry.certified * 100:6.2f}%"
                if entry else " " * 25)
        print(f"  {label:<26} {ssims[label]:10.6f} {cert}")

    bar = ssims["input routed (the bar)"]
    got = ssims["scene-model, vetoed"]
    print(f"\n  BAR A: {got:.6f} vs the routed bar {bar:.6f} "
          f"({got - bar:+.6f}) — {'PASS' if got >= bar - 1e-9 else 'MISS'}")
    print(f"  remainder to 1.0: {1.0 - got:.6f}; rewritten "
          f"{final_rewrite.mean() * 100:.2f}% of the crop "
          f"(region veto alone {region_rewrite.mean() * 100:.2f}%)")
    print(f"  what the LOCAL clauses cost bar A: "
          f"{got - ssims['region veto only (B2)']:+.6f} GT-SSIM against B2's "
          f"region-only composite ({ssims['region veto only (B2)']:.6f})")

    # THE ATTRIBUTED REMAINDER. The same ladder discipline as round A's `floor`:
    # hold everything else and replace one estimated quantity with its TRUE value.
    print(f"\n  --- the remainder, attributed (each rung swaps ONE estimate for "
          f"its TRUTH) ---")
    truth_model, tinfo = FC.factory_truth_model(canvas=False)
    ladder = []
    ladder.append(("as measured (assembly + motion + render)", got))
    oracle = assemble_with_oracle(frames, P.REFERENCE, truth_model, tinfo)
    ladder.append(("+ TRUE per-layer motion (assembly + render)",
                   float(metrics.ref_ssim(oracle["composite"], reference_truth))))
    ladder.append(("+ TRUE masks and motion (render only)",
                   float(metrics.ref_ssim(oracle["truth_masks"], reference_truth))))
    ladder.append(("the factory's own reference frame (null)",
                   float(metrics.ref_ssim(frames[P.REFERENCE][y0:y1, x0:x1],
                                          reference_truth))))
    print(f"  {'rung':<48} {'GT-SSIM':>10} {'remainder':>10}")
    for label, value in ladder:
        print(f"  {label:<48} {value:10.6f} {1.0 - value:10.6f}")
    print(f"\n  attribution: motion estimation "
          f"{ladder[1][1] - ladder[0][1]:+.6f}, layer segmentation "
          f"{ladder[2][1] - ladder[1][1]:+.6f}, "
          f"everything left at true masks and true motion "
          f"{1.0 - ladder[2][1]:.6f} (assembly + render + the crop's own content).")
    # And how much of THAT last term is simply the price of moving a frame. Every
    # non-reference observation is resampled exactly once (PLAYBOOK §0), and every
    # resample softens; this measures it with nothing else in the way.
    round_trip = []
    for group in (0, 1):
        for k in (0, 1, 4, 5):
            matrix = truth_model.matrices[(k, group)]
            there = FC.warp_back(truth.astype(np.float32), matrix, truth.shape[:2])
            back = FC.warp_forward(there, matrix, truth.shape[:2])
            round_trip.append(float(metrics.ref_ssim(
                np.clip(back, 0, 255).astype(np.uint8)[y0:y1, x0:x1],
                reference_truth)))
    print(f"  of which: ONE round-trip resample of the truth alone reads GT-SSIM "
          f"{float(np.mean(round_trip)):.6f} (remainder "
          f"{1.0 - float(np.mean(round_trip)):.6f}) — every resample softens, and "
          f"an assembly pays it on every non-reference observation.")

    print(f"\n  --- the region ledger (scene-model vs input, per owned region) ---")
    _print_region_ledger(verdicts, scores, result)
    _print_scoped_ledger(result, scores, regions)


def assemble_with_oracle(frames, ref, truth_model, tinfo):
    """The factory's ORACLE rungs: true per-layer geometry, then true masks too.

    §12.4's rule applied to a reconstruction — a ladder is only an attribution if
    each rung replaces exactly one estimated quantity with the known answer. The
    factory is the one scene where the answer is known.

    THE ORACLE'S OWN TRAP, and the KAT that caught it. The first build of this
    rung substituted the true per-layer SHIFT into the same slot B1's propagated
    residual occupies — where `_geometry` composes it onto the global affine. The
    true shift is the TOTAL reference-to-frame displacement, so composing it
    double-counted the affine and the oracle scored 0.9414 against the estimate's
    0.9811. An oracle that loses to the thing it is supposed to bound is a broken
    instrument, exactly as F110 found the last time this ladder was built. The
    rung replaces the whole MATRIX.
    """
    dec = LD.decompose(frames, ref)
    near = tinfo["near_mask"]
    out = {}
    out["composite"] = assemble(frames, ref, raw=frames, dec=dec,
                                matrices=_oracle_for(dec, near, truth_model,
                                                     len(frames))).composite
    true_dec = _truth_decomposition(dec, near)
    out["truth_masks"] = assemble(frames, ref, raw=frames, dec=true_dec,
                                  matrices=_oracle_for(true_dec, near, truth_model,
                                                       len(frames))).composite
    return out


def _oracle_for(dec, near, truth_model, n):
    """The factory's TRUE per-layer geometry, indexed by `dec`'s own layer order."""
    near_layer = max(range(len(dec.masks)),
                     key=lambda i: float((dec.masks[i] & near).sum()))
    return {(k, i): truth_model.matrices[(k, 0 if i == near_layer else 1)]
            for k in range(n) for i in range(len(dec.masks))}


def _truth_decomposition(dec, near):
    """B1's decomposition with its MASKS replaced by the factory's true planes."""
    import copy

    overlap = [float((m & near).sum()) for m in dec.masks]
    near_index = int(np.argmax(overlap))
    far_index = int(np.argmin(overlap))
    clone = copy.copy(dec)
    clone.labels = np.where(near, 0, 1).astype(np.int32)
    clone.state = np.zeros(near.shape, np.uint8)          # everything OWNED
    clone.masks = [near.copy(), ~near]
    # The far surface continues behind the near one; the near one does not exist
    # behind anything. That is the `extent` distinction round A's KAT-1b paid for.
    clone.extents = [near.copy(), np.ones(near.shape, bool)]
    # Only the MASKS are swapped for truth on this rung. The focal peaks stay as
    # B1 measured them, matched to whichever layer turned out to be the near one.
    clone.peaks = [dec.peaks[near_index], dec.peaks[far_index]]
    clone.order = list(clone.peaks)
    clone.diag = dict(dec.diag)
    return clone


def _print_region_ledger(verdicts, scores, result):
    print(f"  {'#':>3} {'layer':>5} {'area':>7} {'cert px':>8} {'differential':>13} "
          f"{'state':>13}  box")
    for index, entry in enumerate(verdicts[:16]):
        value = entry["differential"]
        shown = "        n/a" if not np.isfinite(value) else f"{value:+13.4f}"
        print(f"  {index:3d} {entry['layer']:5d} {entry['area']:7d} "
              f"{entry['certified']:8d} {shown} {entry['state']:>13}  "
              f"{entry['box']}")
    if len(verdicts) > 16:
        print(f"  ... {len(verdicts) - 16} more regions")


def _print_scoped_ledger(result, scores, regions):
    """The REGION-SCOPED null (F114 §7): every region against its own best frame."""
    scene = scores.get("scene-model")
    scoped = scores.get("region-scoped null")
    glob = scores.get("global null")
    if scene is None or scoped is None:
        return
    print(f"\n  --- global vs region-scoped ledger ---")
    if glob is not None:
        print(f"  global null (the reference frame)       "
              f"{glob.score:8.4f} levels   p99 {glob.p99:7.3f}")
    print(f"  region-scoped null (best single frame)  "
          f"{scoped.score:8.4f} levels   p99 {scoped.p99:7.3f}")
    print(f"  scene-model composite                   "
          f"{scene.score:8.4f} levels   p99 {scene.p99:7.3f}")
    if glob is not None:
        print(f"  differential vs GLOBAL null   "
              f"{scene.score - glob.score:+.4f} levels")
    print(f"  differential vs SCOPED null   "
          f"{scene.score - scoped.score:+.4f} levels   "
          f"(negative = aggregation beats the best single frame)")
    delta = scene.unexplained - scoped.unexplained
    ok = (np.minimum(scene.coverage, scoped.coverage) >= FC.MIN_COVERAGE)
    wins = losses = 0
    for entry in regions:
        here = entry["mask"] & ok
        if here.sum() < MIN_ARBITRABLE:
            continue
        if float(delta[here].mean()) <= 0:
            wins += 1
        else:
            losses += 1
    print(f"  per-region: aggregation beats its own best single frame in "
          f"{wins} of {wins + losses} arbitrable regions")


def kitchen() -> None:
    """Bars B, C, D on the kitchen, plus both ledgers."""
    from focusstack.io import normalize_exposure

    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    src = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(KITCHEN, "*.jpg")))]
    norm = normalize_exposure(src)
    ref = len(src) // 2
    print("=" * 78)
    print("KITCHEN — bars B (canonical instruments), C (never-degrade), D (identity)")
    print("=" * 78)
    result = assemble(norm, ref, verbose=True, raw=src)
    crop = result.crop
    reference = norm[ref]

    regions = regions_of(result)
    scores, _region = certify_candidates(
        result, src,
        [("scene-model", result.composite), ("input routed", result.base),
         ("region-scoped null", result.scoped_null),
         ("global null", norm[ref][crop[1]:crop[3], crop[0]:crop[2]])])
    verdicts, reverted, final, final_rewrite, vstats = veto_all(
        result, scores["scene-model"], scores["input routed"], regions,
        verbose=True)
    region_only, region_rewrite = finalize(result, reverted)
    previous, previous_rewrite = finalize(result, reverted,
                                          vstats["shipped_106e2f5"])
    final_scores, _r = certify_candidates(
        result, src, [("scene-model, vetoed", final),
                      ("region veto only", region_only),
                      ("shipped 106e2f5", previous)])
    scores.update(final_scores)

    print(f"\n  --- BAR D: byte-identity outside the rewrite ---")
    outside = ~final_rewrite
    identical = np.array_equal(final[outside], result.base[outside])
    print(f"  pixels rewritten: {final_rewrite.sum()} "
          f"({final_rewrite.mean() * 100:.2f}% of the crop); "
          f"outside the rewrite byte-identical: {identical}  "
          f"{'PASS' if identical else 'FAIL'}")
    print(f"  the region veto alone kept {region_rewrite.sum()} px "
          f"({region_rewrite.mean() * 100:.2f}%); the LOCAL clauses withdrew "
          f"{region_rewrite.sum() - final_rewrite.sum()} px "
          f"({(1 - final_rewrite.sum() / max(1, region_rewrite.sum())) * 100:.1f}% "
          f"of it) — cluster {vstats['local']}, frontier {vstats['frontier']}, "
          f"contour {vstats['contour']}, "
          f"residual loud frontier {vstats['residual'][0]} of "
          f"{vstats['residual'][1]}")
    print(f"  STRICT SUBSET LEDGER: 106e2f5 shipped {int(previous_rewrite.sum())} px; "
          f"this rewrite {int(final_rewrite.sum())} px; "
          f"rescued (must be 0) {int((final_rewrite & ~previous_rewrite).sum())}; "
          f"pixel-identical where both write "
          f"{np.array_equal(final[final_rewrite], previous[final_rewrite])}")

    print(f"\n  --- BAR B: the four F112 user boxes (mean/max |Δ| vs norm[{ref}]) ---")
    print(f"  {'candidate':<26} {'box 1':>10} {'box 2':>10} {'box 3':>10} "
          f"{'box 4':>10}")
    before = kitchen_boxes(result.base, reference, crop, "input routed (the bar)")
    kitchen_boxes(region_only, reference, crop, "region veto only (B2)")
    kitchen_boxes(previous, reference, crop, "shipped 106e2f5")
    after = kitchen_boxes(final, reference, crop, "+ contour clause (B3a)")
    worse = [i + 1 for i, (a, b) in enumerate(zip(before, after)) if b[0] > a[0] + 0.01]
    over = [i + 1 for i, (a, b) in enumerate(zip(before, after)) if b[1] > a[1]]
    print(f"  regressed on MEAN |Δ| vs the reference: {worse if worse else 'NONE'}")
    print(f"  RE-ACCEPTANCE BAR — max |Δ| over its routed value: "
          f"{over if over else 'NONE'}  {'MISS' if over else 'PASS'}")
    print(f"  the counter-instrument — mean FOCUS ENERGY in the same boxes "
          f"(higher = sharper):")
    kitchen_boxes(result.base, reference, crop, "input routed", energy=True)
    kitchen_boxes(previous, reference, crop, "shipped 106e2f5", energy=True)
    kitchen_boxes(final, reference, crop, "+ contour clause (B3a)", energy=True)

    print(f"\n  --- BAR B: the canonical F108 flank box (x560-670, y240-420) ---")
    kitchen_flank(result.base, reference, crop, "input routed (the bar)")
    kitchen_flank(previous, reference, crop, "shipped 106e2f5")
    kitchen_flank(final, reference, crop, "+ contour clause (B3a)")

    print(f"\n  --- BAR B: the F112 knob (x659-669, y243-313) ---")
    _knob_report(result, final, scores, reference, crop, norm, src)

    print(f"\n  --- BAR C: the never-degrade veto, per owned region ---")
    _print_region_ledger(verdicts, scores, result)
    _print_scoped_ledger(result, scores, regions)

    print(f"\n  --- the certifier ledger ---")
    for label in ("input routed", "scene-model", "region veto only",
                  "shipped 106e2f5",
                  "scene-model, vetoed", "region-scoped null", "global null"):
        entry = scores.get(label)
        if entry is None:
            continue
        print(f"  {label:<24} {entry.score:8.4f} levels   p99 {entry.p99:7.3f}   "
              f"certified {entry.certified * 100:5.1f}%  boundary "
              f"{entry.boundary * 100:5.1f}%  excluded {entry.excluded * 100:5.1f}%")
    base_delta = FC.differential(scores["scene-model, vetoed"], scores["input routed"])
    print(f"  scene-model - input, whole certified frame: "
          f"{base_delta.score:+.4f} levels (negative = better)")

    np.save(os.path.join(OUT, "kitchen_scenemodel.npy"), final)
    cv2.imwrite(os.path.join(OUT, "kitchen_scenemodel_crop.png"), final)
    _save_rewrite_map(result, final_rewrite, os.path.join(OUT,
                                                          "kitchen_rewrite.png"))
    _eyes_honest_sliver(result, final, reference, crop)
    _defect_crops(result, region_only, final, reference, crop)
    _b3_defect_crops(result, previous, final, reference, crop, vstats)
    print(f"\n  elapsed {time.time() - t0:.1f} s")


# The three defects the manager's EYES found in B2's composite, which the
# region-aggregate veto structurally could not see. COMPOSITE coordinates (the
# inspection layer is the same picture shifted one pixel in x); each window is a
# little larger than the reported defect so the surrounding content is visible.
DEFECT_CROPS = {
    1: ((540, 40, 617, 116),
        "box 2: pale diagonal streak on the wall + spur on the pump limb"),
    2: ((467, 128, 514, 164),
        "box 1: pale line at the Lubriderm left silhouette"),
    3: ((431, 192, 492, 270),
        "box 4: dark streaks at the shelf edge + bright dashes right of the rag"),
}


def _defect_crops(result, region_only, final, reference, crop):
    """ROUTED | B2 | corrected | REFERENCE, 6x nearest, for the three defects.

    Four panels and not three: the round is a correction, so the composite it
    corrects has to be in the picture. Without the B2 panel a reader cannot tell
    a defect that was fixed from a defect that was never there.
    """
    x0, y0, _x1, _y1 = crop
    reference_crop = reference[y0:y0 + result.base.shape[0],
                               x0:x0 + result.base.shape[1]]
    for index, ((bx0, by0, bx1, by1), caption) in DEFECT_CROPS.items():
        panels = []
        for image in (result.base, region_only, final, reference_crop):
            panels.append(cv2.resize(image[by0:by1, bx0:bx1], None, fx=6, fy=6,
                                     interpolation=cv2.INTER_NEAREST))
        strip = np.hstack(panels)
        header = np.zeros((44, strip.shape[1], 3), np.uint8)
        cv2.putText(header, f"defect {index} — {caption}", (6, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (231, 179, 91), 1, cv2.LINE_AA)
        cv2.putText(header, f"x{bx0}-{bx1} y{by0}-{by1}   ROUTED (input) | B2 "
                            f"region veto only | CORRECTED local veto | REFERENCE "
                            f"frame 6", (6, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        path = os.path.join(OUT, f"B2R2_defect{index}.png")
        cv2.imwrite(path, np.vstack([header, strip]))
        print(f"  defect {index} crop (6x) -> {path}")


# The two residuals round B3a is aimed at, in COMPOSITE coordinates. Both are
# §23c's, quoted from the record and not re-derived here: box 1's 6 px fleck
# (inspection x482-484 y150-153, so composite x483-485) translates the bottle's
# silhouette one pixel, and box 4's shelf junction displaces a contour ~2 rows.
B3_CROPS = {
    1: ((467, 128, 514, 164),
        "box 1: the 6 px fleck — the silhouette translated ONE pixel"),
    3: ((431, 192, 492, 270),
        "box 4: the shelf junction — a contour displaced ~2 rows and steepened"),
}


def _b3_defect_crops(result, previous, final, reference, crop, vstats):
    """ROUTED | 106e2f5 SHIPPED | CONTOUR-VETOED | REFERENCE, 6x nearest.

    Same four-panel discipline as §20: without the previous shipped panel a
    reader cannot tell a residual this round removed from one that was never
    there. The flagged contour sites are marked on the third panel, because a
    veto's evidence belongs in the same picture as its effect.
    """
    x0, y0, _x1, _y1 = crop
    reference_crop = reference[y0:y0 + result.base.shape[0],
                               x0:x0 + result.base.shape[1]]
    report = vstats.get("contour_report")
    for index, ((bx0, by0, bx1, by1), caption) in B3_CROPS.items():
        panels = []
        for name, image in (("routed", result.base), ("106e2f5", previous),
                            ("contour", final), ("reference", reference_crop)):
            panel = cv2.resize(image[by0:by1, bx0:bx1], None, fx=6, fy=6,
                               interpolation=cv2.INTER_NEAREST)
            if name == "contour" and report is not None:
                ys, xs = np.nonzero(report["violation"][by0:by1, bx0:bx1])
                for py, px in zip(ys, xs):
                    cv2.rectangle(panel, (px * 6, py * 6),
                                  (px * 6 + 5, py * 6 + 5), (0, 0, 255), 1)
            panels.append(panel)
        strip = np.hstack(panels)
        header = np.zeros((44, strip.shape[1], 3), np.uint8)
        cv2.putText(header, f"B3a defect {index} — {caption}", (6, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (231, 179, 91), 1, cv2.LINE_AA)
        cv2.putText(header, f"x{bx0}-{bx1} y{by0}-{by1}   ROUTED | SHIPPED "
                            f"(106e2f5) | CONTOUR-VETOED (red = displaced contour "
                            f"site) | REFERENCE frame 6", (6, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        path = os.path.join(OUT, f"B3_defect{index}.png")
        cv2.imwrite(path, np.vstack([header, strip]))
        print(f"  B3a defect {index} crop (6x) -> {path}")

    # The shelf junction at 12x and, next to it, §23c's own instrument: the
    # row means across the contour. That table is what "displaced ~2 rows and
    # steepened" was measured with, so it is what the repair has to be read on.
    band = (431, 190, 492, 216)
    strip = np.hstack([cv2.resize(image[band[1]:band[3], band[0]:band[2]], None,
                                  fx=12, fy=12, interpolation=cv2.INTER_NEAREST)
                       for image in (result.base, previous, final,
                                     reference_crop)])
    header = np.zeros((26, strip.shape[1], 3), np.uint8)
    cv2.putText(header, "box 4 shelf junction at 12x — ROUTED | SHIPPED "
                        "(106e2f5) | CONTOUR-VETOED | REFERENCE", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (231, 179, 91), 1, cv2.LINE_AA)
    path = os.path.join(OUT, "B3_defect3_band.png")
    cv2.imwrite(path, np.vstack([header, strip]))
    print(f"  B3a shelf junction band (12x) -> {path}")
    print(f"  §23c's instrument — row means across the junction contour "
          f"(x{band[0]}-{band[2]}):")
    print(f"  {'row':>5} {'routed':>8} {'106e2f5':>9} {'contour':>9} "
          f"{'reference':>10}")
    for y in range(200, 205):
        cells = [float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[
            y, band[0]:band[2]].mean())
            for image in (result.base, previous, final, reference_crop)]
        print(f"  {y:5d} {cells[0]:8.1f} {cells[1]:9.1f} {cells[2]:9.1f} "
              f"{cells[3]:10.1f}")


def _knob_report(result, final, scores, reference, crop, norm, src):
    """The knob, on both instruments that can see it."""
    knob = np.zeros(reference.shape[:2], bool)
    knob[KNOB[1]:KNOB[3], KNOB[0]:KNOB[2]] = True
    for label, image in (("input routed", result.base), ("scene-model", final)):
        delta = _delta_map(image, reference, crop)
        frame_mean = float(np.nanmean(delta))
        box_mean = float(np.nanmean(delta[knob]))
        print(f"  {label:<20} |Δ| box mean {box_mean:6.3f}  frame mean "
              f"{frame_mean:6.3f}  ratio {box_mean / frame_mean:5.2f}x  "
              f"max {float(np.nanmax(delta[knob])):4.0f}")
    scene = scores["scene-model, vetoed"]
    base = scores["input routed"]
    glob = scores["global null"]
    for label, entry in (("input routed", base), ("scene-model", scene)):
        delta = FC.differential(entry, glob)
        window = knob & (delta.coverage >= FC.MIN_COVERAGE)
        scored = delta.coverage >= FC.MIN_COVERAGE
        if not window.any():
            print(f"  {label:<20} certifier differential: no certified pixels "
                  f"in the box")
            continue
        box = float(delta.unexplained[window].mean())
        frame = float(delta.unexplained[scored].mean())
        print(f"  {label:<20} certifier differential box {box:6.3f}  frame "
              f"{frame:6.3f}  ratio {box / frame:5.2f}x  "
              f"(bar 1.50x)  certified px {int(window.sum())}")
    rewritten_here = result.diag["rewrite_full"][KNOB[1]:KNOB[3], KNOB[0]:KNOB[2]]
    state = result.dec.state[KNOB[1]:KNOB[3], KNOB[0]:KNOB[2]]
    print(f"  knob box ownership: owned {float((state == LD.OWNED).mean()) * 100:.1f}%"
          f"  boundary {float((state == LD.BOUNDARY).mean()) * 100:.1f}%"
          f"  unknown {float((state == LD.UNKNOWN).mean()) * 100:.1f}%"
          f"  -> rewritten {float(rewritten_here.mean()) * 100:.1f}%")


def _save_rewrite_map(result, final_rewrite, path):
    base = result.base.astype(np.float32)
    grey = cv2.cvtColor(cv2.cvtColor(base.astype(np.uint8), cv2.COLOR_BGR2GRAY),
                        cv2.COLOR_GRAY2BGR).astype(np.float32)
    colour = np.zeros_like(base)
    # No grey in the palette: the base is drawn greyscale, so a grey layer would
    # be invisible exactly where the map has something to say.
    palette = np.array([[60, 60, 220], [60, 200, 60], [220, 160, 60],
                        [200, 60, 200], [60, 220, 220], [40, 140, 255]], np.uint8)
    for i in range(len(result.dec.masks)):
        colour[final_rewrite & (result.layer_of == i)] = palette[i % len(palette)]
    reverted = result.rewritten & ~final_rewrite
    colour[reverted] = (255, 255, 255)
    blend = 0.55 * grey + 0.45 * colour
    cv2.imwrite(path, np.clip(blend, 0, 255).astype(np.uint8))
    print(f"  rewrite map (white = reverted by the veto) -> {path}")


def _eyes_honest_sliver(result, final, reference, crop):
    """The pale sliver at the bottle's left silhouette, before / after / reference."""
    x0, y0, _x1, _y1 = crop
    bx0, by0, bx1, by1 = USER_BOXES[1]
    pad = 26
    sx0, sy0 = max(0, bx0 - pad), max(0, by0 - pad)
    sx1, sy1 = min(result.base.shape[1], bx1 + pad), min(result.base.shape[0], by1 + pad)
    panels = []
    for image in (result.base, final,
                  reference[y0:y0 + result.base.shape[0],
                            x0:x0 + result.base.shape[1]]):
        panel = cv2.resize(image[sy0:sy1, sx0:sx1], None, fx=6, fy=6,
                           interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(panel, ((bx0 - sx0) * 6, (by0 - sy0) * 6),
                      ((bx1 - sx0) * 6, (by1 - sy0) * 6), (0, 0, 255), 1)
        panels.append(panel)
    strip = np.hstack(panels)
    header = np.zeros((26, strip.shape[1], 3), np.uint8)
    cv2.putText(header, "sliver  INPUT ROUTED | SCENE-MODEL | REFERENCE frame 6",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (231, 179, 91), 1,
                cv2.LINE_AA)
    path = os.path.join(OUT, "kitchen_sliver.png")
    cv2.imwrite(path, np.vstack([header, strip]))
    print(f"  sliver crop (6x, eyes-honest) -> {path}")


def orderguard() -> None:
    """What the ordering-FREE visibility rule costs, against the ordered one."""
    from focusstack.io import normalize_exposure
    import parallax_gen as P

    print("=" * 78)
    print("ORDERING GUARD — the price of not trusting the layer order (F114 §9)")
    print("=" * 78)
    print("  'any' declines a pixel wherever ANY other layer's footprint lands on it\n"
          "  in that frame; 'nearer' trusts the focal-peak order and only skips the\n"
          "  layers it calls nearer. The difference is the disocclusion band.\n")
    src = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(KITCHEN, "*.jpg")))]
    cases = [("factory", P.build_stack()[0], P.REFERENCE, None),
             ("kitchen", normalize_exposure(src), len(src) // 2, src)]
    print(f"  {'scene':<10} {'guard':<8} {'rewritten':>10} {'members/px':>11} "
          f"{'OCCLUDED':>9}   near_is_low")
    kept = {}
    for name, images, ref, raw in cases:
        for guard in ("any", "nearer"):
            result = assemble(images, ref, order_guard=guard,
                              raw=raw if raw is not None else images)
            kept[(name, guard)] = result
            members = result.diag["members"][result.diag["rewrite_full"]]
            counts = result.diag["refusals"]
            print(f"  {name:<10} {guard:<8} {result.rewritten.mean() * 100:9.2f}% "
                  f"{float(members.mean()) if members.size else 0:11.2f}"
                  f" {counts['occluded'] / max(1, counts['owned frames']) * 100:8.3f}%"
                  f"   near_is_low={result.dec.near_is_low}")
    print("\n  the same runs, by refusal stage (share of owned pixel-frames)")
    print(f"  {'scene':<10} {'guard':<8} {'no geom':>8} {'off-frame':>10} "
          f"{'occluded':>9} {'diff surface':>13} {'admitted':>9}")
    for key, result in kept.items():
        counts = result.diag["refusals"]
        total = max(1, counts["owned frames"])
        print(f"  {key[0]:<10} {key[1]:<8} "
              f"{counts['no geometry'] / total * 100:7.2f}% "
              f"{counts['outside the frame'] / total * 100:9.2f}% "
              f"{counts['occluded'] / total * 100:8.3f}% "
              f"{counts['different surface'] / total * 100:12.2f}% "
              f"{counts['admitted'] / total * 100:8.2f}%")
    print("\n  Read the OCCLUDED column first. B1's boundary band already declines a\n"
          "  5 px ribbon at every layer boundary, and on both scenes the differential\n"
          "  motion between adjacent layers is smaller than that ribbon — so the\n"
          "  visibility test has almost nothing left to refuse, and the two guards are\n"
          "  indistinguishable. That is decompose_NOTES §9's own prediction, measured:\n"
          "  ordering is non-load-bearing for a reconstruction that does NOT complete\n"
          "  occluded content, and it becomes load-bearing the moment one does.")


# ---------------------------------------------------------------------------
# A REJECTED clause, kept as its own measurement (§23)
# ---------------------------------------------------------------------------
# "Abstention near a geometric boundary is refusal": inside B1's own boundary
# band, dilated by this module's own `FRONTIER_SLACK`, a rewrite survives only
# with a POSITIVE certifier verdict at quorum. Licence: F92 (a curved object's
# limb is never trustworthy from a moved frame) and F106 (what cannot be
# arbitrated must not be applied). No new tuned number — both constants already
# exist above, and "positive" is the region rule's own zero.
#
# It is not wired into `veto_all`, because measurement says it does not do the
# job it was designed for and cannot: the two residual defects it was aimed at
# are 3.6-6.6 px and a median 10.1 px OUTSIDE the zone. It is kept runnable so
# the negative is reproducible rather than merely asserted (`boundary`).


def _boundary_clause(assembly, scene_result, base_result, keep):
    """Withdraw every zone pixel the certifier has not positively certified."""
    x0, y0, x1, y1 = assembly.crop
    band = assembly.dec.diag["band"][y0:y1, x0:x1]
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                     (2 * FRONTIER_SLACK + 1,) * 2)
    zone = cv2.dilate(band.astype(np.uint8), disc).astype(bool)
    pooled, quorum = cluster_pool(assembly, scene_result, base_result)
    return keep & ~(zone & ~(quorum & (pooled <= 0.0))), zone


def _price_boundary(images, ref, raw, name, truth=None):
    """A / B / C: shipped, clause-before-frontier, clause-after-frontier."""
    result = assemble(images, ref, raw=raw)
    if truth is not None:
        x0, y0, x1, y1 = result.crop
        truth = truth[y0:y1, x0:x1]
    regions = regions_of(result)
    scores, _r = certify_candidates(result, raw,
                                    [("scene-model", result.composite),
                                     ("input routed", result.base)])
    scene, base = scores["scene-model"], scores["input routed"]
    _v, reverted = apply_veto(result, scene, base, regions)
    keep2, _n = local_veto(result, scene, base, reverted)
    a, _f, _res = quiet_frontier(result, keep2)
    b, _f, _res = quiet_frontier(result, _boundary_clause(result, scene, base,
                                                          keep2)[0])
    c, _zone = _boundary_clause(result, scene, base, a)
    variants = {"A shipped (5ec37d7)": a, "B clause then frontier": b,
                "C frontier then clause": c}
    composites = {k: finalize(result, reverted, m)[0] for k, m in variants.items()}
    final_scores, _r = certify_candidates(result, raw, list(composites.items()))
    print(f"\n  --- {name} ---")
    print(f"  {'variant':<24} {'kept px':>8} {'certifier':>10} "
          f"{'subset of A':>12} {'withdrawn':>10}"
          + ("   GT-SSIM" if truth is not None else ""))
    for label, mask in variants.items():
        subset = "yes" if not (mask & ~a).any() else f"NO (+{int((mask & ~a).sum())})"
        extra = ""
        if truth is not None:
            import metrics
            extra = f"  {float(metrics.ref_ssim(composites[label], truth)):9.6f}"
        print(f"  {label:<24} {int(mask.sum()):8d} "
              f"{final_scores[label].score:10.4f} {subset:>12} "
              f"{int((a & ~mask).sum()):10d}{extra}")
    return result, reverted, variants, composites, scores


def boundary() -> None:
    """Price the REJECTED boundary-abstention clause, and show it changing nothing.

    Runs the clause in both composable orders on both scenes, and writes the
    four-panel defect crops with the clause's own composite in the third panel.
    §12.8: the picture is the deliverable, the table is the argument.
    """
    from focusstack.io import normalize_exposure
    import metrics
    import parallax_gen as P

    os.makedirs(OUT, exist_ok=True)
    print("=" * 78)
    print("BOUNDARY-ABSTENTION — a rejected clause, priced (§23)")
    print("=" * 78)
    src = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(KITCHEN, "*.jpg")))]
    norm = normalize_exposure(src)
    ref = len(src) // 2
    result, reverted, variants, composites, scores = _price_boundary(
        norm, ref, src, "kitchen")
    reference = norm[ref]
    crop = result.crop
    knob = np.zeros(reference.shape[:2], bool)
    knob[KNOB[1]:KNOB[3], KNOB[0]:KNOB[2]] = True
    glob_s, _r = certify_candidates(result, src,
                                    [("g", norm[ref][crop[1]:crop[3],
                                                     crop[0]:crop[2]])])
    final_scores, _r = certify_candidates(result, src, list(composites.items()))
    print(f"\n  {'variant':<24} {'knob':>7} {'box1':>5} {'box2':>5} {'box3':>5} "
          f"{'box4':>5} {'flank>12':>9}")
    for label, image in composites.items():
        differential = FC.differential(final_scores[label], glob_s["g"])
        window = knob & (differential.coverage >= FC.MIN_COVERAGE)
        scored = differential.coverage >= FC.MIN_COVERAGE
        ratio = (float(differential.unexplained[window].mean())
                 / float(differential.unexplained[scored].mean()))
        boxes = kitchen_boxes(image, reference, crop)
        flank = kitchen_flank(image, reference, crop)
        print(f"  {label:<24} {ratio:6.2f}x "
              + " ".join(f"{b[1]:5.0f}" for b in boxes) + f" {flank[2]:8.2f}%")
    print(f"  bars: knob <= 1.50x, box maxima <= 61 / 17 / 101 / 127, "
          f"flank not worse than 0.23%")

    frames, truth, _near = P.build_stack()
    _price_boundary(frames, P.REFERENCE, frames, "factory", truth=truth)

    # the pictures: ROUTED | SHIPPED | + clause | REFERENCE
    cx0, cy0 = crop[0], crop[1]
    reference_crop = reference[cy0:cy0 + result.base.shape[0],
                               cx0:cx0 + result.base.shape[1]]
    for index, ((bx0, by0, bx1, by1), caption) in DEFECT_CROPS.items():
        panels = [cv2.resize(image[by0:by1, bx0:bx1], None, fx=6, fy=6,
                             interpolation=cv2.INTER_NEAREST)
                  for image in (result.base, composites["A shipped (5ec37d7)"],
                                composites["C frontier then clause"],
                                reference_crop)]
        strip = np.hstack(panels)
        header = np.zeros((44, strip.shape[1], 3), np.uint8)
        cv2.putText(header, f"defect {index} — {caption}", (6, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (231, 179, 91), 1, cv2.LINE_AA)
        cv2.putText(header, f"x{bx0}-{bx1} y{by0}-{by1}   ROUTED | SHIPPED "
                            f"(5ec37d7) | + boundary-abstention clause | "
                            f"REFERENCE frame 6", (6, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        path = os.path.join(OUT, f"B2R3_defect{index}.png")
        cv2.imwrite(path, np.vstack([header, strip]))
        print(f"  defect {index} crop (6x) -> {path}")


def render() -> None:
    """Register the scene-model kitchen composite to the EXISTING inspection layer."""
    from focusstack.io import normalize_exposure

    os.makedirs(INSPECT, exist_ok=True)
    target = cv2.imread(os.path.join(INSPECT, "kitchen_reference.png"))
    if target is None:
        print("  out/inspect/kitchen_reference.png absent — nothing to register to")
        return
    path = os.path.join(OUT, "kitchen_scenemodel.npy")
    if not os.path.exists(path):
        print("  run `kitchen` first (it writes out/certify/kitchen_scenemodel.npy)")
        return
    composite = np.load(path)
    src = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(KITCHEN, "*.jpg")))]
    reference = normalize_exposure(src)[len(src) // 2]
    # The registration is a pure lookup, not a fit: the inspection layer is a crop
    # of the reference frame, so template-matching a central patch of it back into
    # the reference recovers the crop origin exactly, and the composite's own crop
    # origin is known. One integer translation, no resample.
    patch = target[100:300, 100:400]
    scored = cv2.matchTemplate(reference, patch, cv2.TM_CCOEFF_NORMED)
    _mn, score, _mnl, location = cv2.minMaxLoc(scored)
    ox, oy = location[0] - 100, location[1] - 100
    print(f"  registration score {score:.4f} "
          f"({'PASS' if score > 0.99 else 'FAIL'}), inspection crop origin "
          f"({ox}, {oy})")
    cx0, cy0 = 15, 8                      # the composite's own crop origin
    th, tw = target.shape[:2]
    out = reference[oy:oy + th, ox:ox + tw].copy()
    y0, x0 = oy - cy0, ox - cx0
    piece = composite[y0:y0 + th, x0:x0 + tw]
    out[:piece.shape[0], :piece.shape[1]] = piece
    destination = os.path.join(INSPECT, "kitchen_scenemodel.png")
    cv2.imwrite(destination, out)
    routed = cv2.imread(os.path.join(INSPECT, "kitchen_routed.png"))
    if routed is not None:
        delta = np.abs(out.astype(np.float32) - routed.astype(np.float32)).max(axis=2)
        print(f"  vs kitchen_routed.png: {float((delta > 0).mean()) * 100:.2f}% of "
              f"pixels differ, mean |Δ| where they do "
              f"{float(delta[delta > 0].mean()) if (delta > 0).any() else 0:.2f}")
    print(f"  -> {destination}  ({out.shape[1]}x{out.shape[0]}, "
          f"registered to kitchen_reference.png)")


COMMANDS = {"kat": kat, "slope": slope_kat, "localkat": local_kat,
            "contourkat": contour_kat,
            "factory": factory, "kitchen": kitchen, "orderguard": orderguard,
            "boundary": boundary, "render": render}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print("commands: " + ", ".join(COMMANDS))
        return
    COMMANDS[sys.argv[1]]()


if __name__ == "__main__":
    main()
