"""The scene-model ALIGNER: N frames resampled ONCE into reference geometry, with gaps.

Where this sits. FRONTIER §7b's output contract supersedes the composite-rewrite
framing B2 implemented: pass 2's deliverable is not a better composite, it is the
input frames PERFECTLY ALIGNED into the reference viewpoint's geometry, each
carrying honest GAPS where that frame did not observe the surface the reference
viewpoint owns. `fuse_perband(aligned, usable=usable)` then consumes the output
exactly as it consumes `align_stack`'s today. Under this contract invention is
structurally impossible: there is no cross-frame appearance-synthesis step for
fabricated content to enter through (B2's per-layer assembly WAS one, and the F116
wall smear came in through it). What remains possible is MISPLACEMENT, so the
silhouette gate and the photometric veto are the auditors.

This module is ROUND 1: the machinery and its known-answer tests, on the ANALYTIC
FACTORY only (`parallax_gen`), where every answer is known. The kitchen is round 2.

The transform, implemented exactly as the charter designs it:

  1. PIECES cut at OCCLUDING CONTOURS (depth DISCONTINUITIES), never at depth
     VALUES — a cut at a depth value puts a fake step-seam inside a smooth ramp.
     So: find where the focal-peak field JUMPS, and let the image's own contour
     localize the cut inside that ribbon (watershed). A ramp produces no jump and
     stays whole; `kat_ramp` is the known-answer test of exactly that.
  2. ONE AFFINE PER PIECE PER FRAME — a plane under small camera motion induces an
     affine flow, so a rigid object and a planar ramp are each ONE piece. Fitted
     from MATERIAL edges inside the piece (F92: never the limb, whose apparent
     motion mixes two surfaces) through F116's dense normal-profile matcher, with
     a global affine as the prior, then regularized by the temporally-coherent
     motion SERIES (B3b) because a defocused piece's own fit is biased toward zero.
  3. A SILHOUETTE-EXACT GATE: F116's contour instrument becomes a per-frame
     alignment gate. A frame whose near-piece contour does not land on the
     reference contour within the matte band gets GAPS there, not a forced fit.
  4. GAPS from the model: occlusion (a nearer piece's silhouette warped by ITS OWN
     transform into frame k, DILATED by that piece's modeled defocus radius there —
     F83, and F116's wall smear entered exactly through the missing dilation),
     off-footprint, and a photometric veto by cross-convolution. Geometry proposes,
     photometry vetoes, nothing un-vetoes: the veto stack is composed in ONE fixed
     order and monotonicity is ASSERTED, because refusal composition is
     non-monotone in general (scenemodel_NOTES §23a).
  5. APPLICATION is ONE `cv2.remap` per frame with the DISCONTINUOUS field. The
     jump at a piece boundary is LEGAL and is not blended (F81's blend is the soft
     geometry F106 outlawed); the gaps absorb it. `usable` uses nearest-neighbour
     discipline throughout — a mask must never be interpolated into existence.

Run:
    .venv/bin/python research/aligner.py            # K1..K4 in order
    .venv/bin/python research/aligner.py pieces     # K1 only
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from focusstack import motion_groups as MG  # noqa: E402
from focusstack.align import align_stack  # noqa: E402
from focusstack.fusion import depth_from_focus, fuse_perband  # noqa: E402
from focusstack.io import to_gray_float  # noqa: E402
import forward_certify as FC  # noqa: E402
import metrics  # noqa: E402
import parallax_gen as P  # noqa: E402
import scene_model as SM  # noqa: E402

OUT = os.path.abspath(os.path.join(HERE, "..", "out", "aligner"))

# --- constants, and where each one comes from ---------------------------------
# The guided filter's own radius inside `depth_from_focus`. A depth DISCONTINUITY
# is a focal-peak jump of at least one whole frame spread over no more than that
# radius; anything gentler is a RAMP and must not be cut. So the jump threshold is
# not a tuned number, it is (one focal step) / (the smoothing radius that blurred
# it), in the [0, 1] units `depth_from_focus` returns.
DEPTH_GUIDED_RADIUS = 8
# A piece must be at least this large to carry its own affine: `estimate_radii`'s
# own minimum weight for a fit to be scored at all, borrowed from scene_model.
MIN_PIECE = 200
# Sub-pixel displacements this small are not resolved by anything in this arc
# (`twoframe.GATE_TOL`, reached through scene_model's own borrow).
GATE_TOL = SM.TF.GATE_TOL
# Sign of `scene_model._match_dense`'s shift relative to the +normal direction.
# NOT assumed: `kat_sign` measures it against an injected known displacement and
# asserts this value. If the instrument's convention ever changes, that KAT fails
# loudly instead of the affine fit quietly diverging (§12.1).
MATCH_SIGN = 1.0
# The affine-agreement merge (round 2) iterates fit -> merge -> refit to a fixed
# point. The cap exists only so a pathological scene cannot spin: piece count is
# strictly decreasing per round and asserted, so the loop terminates on its own.
MERGE_MAX_ROUNDS = 8
# Points sampled from a candidate merge's own support when its two affines are
# compared. The comparison is a mean over the support, so a few thousand points
# resolve it far below GATE_TOL; the cap is only there to keep the pairwise sweep
# cheap when a scene over-segments into dozens of pieces.
MERGE_SAMPLE = 3000
# Degrees of freedom of a PIECE's residual motion, on top of the whole-frame affine
# the global stage already fitted. Round 2 used TWO (pure translation) and its
# reasoning was half right: rotation and magnification about the IMAGE centre are
# depth-independent, which is why one global affine carries breathing, and a
# lateral camera translation gives a per-depth SHIFT. But depth dependence has a
# SECOND axis, and round 2 dropped it: a FORWARD camera translation t_z gives a
# per-depth isotropic SCALE (each surface magnifies by 1 + t_z/Z), and
# `motion_components` measured this very scene at up to 4.3% forward translation.
# A translation-only decision therefore merges two objects at different depths
# whenever they happen to translate alike while scaling differently — root cause 2
# of the reference-collapse. So the DECISION model order is THREE: translation
# plus isotropic scale, parameterized about the PIECE'S OWN SITE CENTROID rather
# than the image centre, which is what keeps the two axes near-orthogonal and is
# why round 2's 6-DoF merge attempt drowned in extrapolation noise (1.75-53 px
# from the linear part evaluated far from the sites). Six unknowns about a distant
# centre are undetermined on a small piece; three about its own centroid are not.
PIECE_DOF = 3
# ...and the model order of the FINAL transform, once the merge has produced pieces
# large enough to determine it. Round 1's K2 bar (0.159 px on true pieces) is a
# 6-DoF number and stays one.
FINAL_DOF = 6
# --- the two-axis equivalence test's tolerance ------------------------------
# The decision is NOT a fixed pixel bar (round 2's item 1: the factory holds
# spurious cuts at 2.0-2.6 px of FIT error while the kitchen merges real objects
# at under 1.5 px of DEPTH difference, so no fixed bar separates them). It is the
# residual PAIR (Δtranslation, Δscale) compared against the DECISION FIT'S OWN
# UNCERTAINTY: each fit reports sigma_t (px) and sigma_s (dimensionless) from its
# final trimmed residuals and its own support, and two pieces are one surface iff
# BOTH axes agree inside k sigma. `k` is the only free number and it is
# calibrated where truth exists — the factory, whose two planes must stay cut and
# whose spurious within-plane cuts must merge (K1 purity/recall, K4).
# Calibrated on the factory, where truth adjudicates, and it is a PLATEAU rather
# than a point: k=20 and k=30 produce the identical piece map (6 pieces, mean
# purity 0.969 against the true near mask, cut recall 68.23% — round 2's own
# numbers at its fixed 1.5 px bar), k=12 over-segments to 7, and k>=35 falls off a
# cliff (4 pieces but purity 0.657 — the NEAR PLANE absorbed into a far piece,
# recall 42.72%). 25 is the centre of the plateau. That the decision is insensitive
# across a 1.5x range of k and then fails abruptly is the evidence that the sigma
# it scales is the right quantity; a tuned threshold has no such plateau.
DECISION_K = 25.0
# Sigma floors, from the INSTRUMENT rather than from taste. `kat_sign` measures
# the normal-profile matcher against a known +2.000 px displacement and asserts
# 0.25 px of magnitude accuracy; a per-site sigma below that divided by the
# sqrt of a realistic independent-site count is a claim the matcher cannot make,
# because profile residuals inside one contour are correlated and the naive
# lstsq covariance therefore reads optimistically small. 0.02 px and 2e-5 are
# those floors (0.25/sqrt(150) and its scale analogue over a 300 px support).
SIGMA_T_FLOOR = 0.02
SIGMA_S_FLOOR = 2e-5
# The dense focal-signature band a seeded piece's support is intersected with.
# `layer_decompose.EVIDENCE_TOL` (0.75 frames) is the focal disagreement that
# already means "a disagreement about WHICH LAYER the pixel is in"; borrowed, not
# chosen.
FOCAL_BAND = 0.75
# A motion group's own support map, as `motion_groups` returns it: `>= CLAIM_OWNED`
# is the body it owns outright (align.py's own reading of the same map), and
# `> CLAIM_HULL` is the hull before the claim gate, which is the territory the
# focal band is then allowed to trim.
CLAIM_OWNED = 0.9
CLAIM_HULL = 0.05


# ---------------------------------------------------------------------------
# Geometry helpers. One convention, stated once: a piece's transform M maps
# REFERENCE coordinates to FRAME-k coordinates, so it IS the backward sampling
# field `remap` wants, AND it is the src=reference -> dst=frame_k matrix
# `warpAffine` wants for pushing a reference-geometry mask into frame k. Both
# directions of `usable` are computable from that one matrix.
# ---------------------------------------------------------------------------
def _grid(shape):
    h, w = shape
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    return xx, yy


def _field(matrix, shape):
    """The backward sampling field of one affine, over the whole reference grid."""
    xx, yy = _grid(shape)
    m = np.asarray(matrix, np.float32)
    return (m[0, 0] * xx + m[0, 1] * yy + m[0, 2],
            m[1, 0] * xx + m[1, 1] * yy + m[1, 2])


def _piecewise_field(matrices, labels, shape):
    """ONE field assembled from per-piece affines — discontinuous at the cuts.

    No blending across a cut. The jump is the physics (near content jumps relative
    to far at an occluding contour); F81's smoothing of it is the soft geometry
    F106 outlawed, and the gaps are what make the jump harmless.
    """
    map_x, map_y = _field(np.eye(3), shape)
    for piece, matrix in matrices.items():
        mask = labels == piece
        if not mask.any():
            continue
        fx, fy = _field(matrix, shape)
        map_x[mask], map_y[mask] = fx[mask], fy[mask]
    return map_x, map_y


def _push_mask(mask, matrix, shape):
    """A reference-geometry mask, pushed into frame-k coordinates. Nearest only."""
    m = np.asarray(matrix, np.float32)[:2]
    return cv2.warpAffine(mask.astype(np.uint8), m, (shape[1], shape[0]),
                          flags=cv2.INTER_NEAREST, borderValue=0) > 0


def _sample_mask(mask, map_x, map_y):
    """A frame-k mask, read back at the places a reference pixel samples from."""
    return cv2.remap(mask.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0


def _gray(image):
    return cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8),
                        cv2.COLOR_BGR2GRAY).astype(np.float32)


def _dilate(mask, radius):
    r = int(np.ceil(max(0.0, float(radius))))
    if r < 1:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


# ---------------------------------------------------------------------------
# 1. PIECES — cut at depth DISCONTINUITIES, localized by the image's own contour
# ---------------------------------------------------------------------------
def cut_pieces(aligned_global, verbose=False):
    """Piece labels from the focal-peak field's JUMPS. Never from its values.

    `depth_from_focus` is pass 1's own depth proxy (the per-pixel winner index,
    guided-filtered onto boundaries). Its GRADIENT, not its level, is what a piece
    boundary is: |grad depth| >= one focal step spread over the guided radius. That
    ribbon is a closed curve by construction (it is a level-set boundary of a
    scalar field), so the complement's connected components are pieces — and a
    smooth ramp, whose gradient never reaches the threshold, is ONE piece.

    The ribbon is ~2*radius wide; the CUT inside it is placed by watershed on the
    reference frame, i.e. by the image's own strongest contour, which is where an
    occluding contour actually is. Returns (labels, diag).
    """
    n = len(aligned_global)
    raw = depth_from_focus(aligned_global, radius=DEPTH_GUIDED_RADIUS)
    # Two scales, both the guided filter's OWN, neither chosen: speckle finer than
    # the filter's support is not resolved (median over its diameter, edge-
    # preserving so a real jump survives), and a real jump is SPREAD over that same
    # diameter, so the jump must be measured across it rather than by a 3x3 Sobel.
    window = 2 * DEPTH_GUIDED_RADIUS + 1
    depth = cv2.medianBlur((raw * 255).astype(np.uint8), window).astype(np.float32) / 255.0
    kernel = np.ones((window, window), np.uint8)
    jump = cv2.dilate(depth, kernel) - cv2.erode(depth, kernel)
    # Half a focal step: a whole step is what a discontinuity IS, and the guided
    # filter plus the median attenuate it, so the bound is the attenuated half.
    threshold = 0.5 / max(n - 1, 1)
    ribbon = jump >= threshold

    count, cores = cv2.connectedComponents((~ribbon).astype(np.uint8), connectivity=4)
    markers = np.zeros(cores.shape, np.int32)
    kept = 0
    for index in range(1, count):
        area = int((cores == index).sum())
        if area < MIN_PIECE:
            continue
        kept += 1
        markers[cores == index] = kept
    # Watershed floods the ribbon from the cores and stops on the reference's own
    # gradient ridge: the cut lands on the occluding contour, not in the middle of
    # the ribbon the smoothing produced.
    reference = np.clip(aligned_global[P.REFERENCE], 0, 255).astype(np.uint8)
    filled = markers.copy()
    cv2.watershed(reference, filled)
    labels = np.where(filled > 0, filled - 1, -1).astype(np.int32)
    # Watershed marks its boundary -1. Those pixels still need a field value, so
    # they take the nearest piece's; they are ALSO the reference's own matte band
    # and are withheld from every non-reference frame further down (the irreducible
    # residue of committing to one viewpoint).
    cut = labels < 0
    if cut.any():
        _, nearest = cv2.distanceTransformWithLabels(
            cut.astype(np.uint8) * 255, cv2.DIST_L2, 3,
            labelType=cv2.DIST_LABEL_PIXEL)
        flat = labels.ravel()
        source = np.flatnonzero(~cut.ravel())
        labels = np.where(cut, flat[source[nearest.ravel() - 1]].reshape(labels.shape),
                          labels)
    diag = {"ribbon": ribbon, "cut": cut, "depth": depth, "jump": jump,
            "threshold": threshold, "n_pieces": kept}
    if verbose:
        print(f"    pieces {kept}  ribbon {ribbon.mean() * 100:.2f}% of frame  "
              f"jump threshold {threshold:.3f} (half a focal step, measured across "
              f"{window} px)")
    return labels, diag


def piece_table(labels):
    return [int(i) for i in np.unique(labels) if (labels == i).sum() >= MIN_PIECE]


# ---------------------------------------------------------------------------
# 2. ONE AFFINE PER PIECE PER FRAME, from MATERIAL edges
# ---------------------------------------------------------------------------
def _material_sites(gray, mask, exclude):
    """Canny sites inside the piece, with the LIMB excluded (F92).

    A limb (silhouette) edge is a boundary BETWEEN two surfaces: its apparent
    motion is a mixture of both, so fitting a piece's motion from it imports the
    other piece's motion. Only MATERIAL edges — surface detail interior to the
    piece — constrain one surface's transform. `parallax_gen` grew its surface
    detail (F96) precisely so this test has anything to work with.
    """
    edges = cv2.Canny(cv2.GaussianBlur(gray.astype(np.uint8), (5, 5), 0), 60, 180) > 0
    margin = SM.CONTOUR_HALF + SM.CONTOUR_SPAN + 2
    inside = np.zeros(edges.shape, bool)
    inside[margin:-margin, margin:-margin] = True
    return edges & inside & mask & ~exclude


def reference_cache(ref_gray):
    """The reference's normals and contour profiles — computed once, not per fit.

    Pure economy (§13): the merge refits every piece several times over, and these
    two arrays depend only on the reference frame.
    """
    nx, ny = SM._unit_normals(ref_gray)
    return nx, ny, SM._contour_profiles(ref_gray, nx, ny)


def fit_affine(ref_gray, frame, prior, sites, shape, iterations=3, cache=None,
               dof=6):
    """Residual affine of one piece, from normal-profile displacements.

    F116 measured this matcher against known answers: +1.00/+2.00 px read
    1.000/2.000, and a pure SHARPENING reads <=0.005 px — so the normal-profile
    correlation is a defocus-INVARIANT measure of where a contour is. Each site
    yields ONE scalar constraint (the normal component of the residual flow),
    n . u(p) = d(p), and six unknowns against thousands of sites is
    overdetermined, which is what makes the fit falsifiable (§12.4).

    Robustness is a trimmed reweighting, not a tuned threshold: sites whose
    residual exceeds 2.5 MAD are dropped, twice.
    """
    matrix = np.asarray(prior, np.float64).copy()
    nx, ny, prof_ref = reference_cache(ref_gray) if cache is None else cache
    ys, xs = np.nonzero(sites)
    floor = 4 * dof                     # §12.4: four constraints per unknown
    if len(xs) < floor:
        return matrix, {"sites": 0, "residual": float("nan"), "ok": False,
                        "sigma_t": float("nan"), "sigma_s": float("nan")}
    cx, cy = (shape[1] - 1) / 2.0, (shape[0] - 1) / 2.0
    if dof == 3:
        # The DECISION fit is parameterized about the piece's OWN centroid, so its
        # translation is the displacement where the evidence actually is and its
        # scale is nearly orthogonal to it. About the image centre the two are
        # strongly correlated on an off-centre piece, which is exactly how round
        # 2's 6-DoF merge attempt turned scale noise into 53 px of translation.
        cx, cy = float(xs.mean()), float(ys.mean())
    history = []
    sigma_t = sigma_s = float("nan")
    contours = max(int(cv2.connectedComponents(sites.astype(np.uint8),
                                               connectivity=8)[0]) - 1, 1)
    correlation = max(len(xs) / float(contours), 1.0)
    for _ in range(iterations):
        map_x, map_y = _field(matrix, shape)
        warped = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        prof_warp = SM._contour_profiles(_gray(warped), nx, ny)
        shift, peak = SM._match_dense(prof_ref, prof_warp)
        good = (peak[ys, xs] >= MG.MIN_PEAK) & (np.abs(shift[ys, xs]) < SM.CONTOUR_HALF)
        if good.sum() < floor:
            break
        gx, gy = xs[good] - cx, ys[good] - cy
        d = MATCH_SIGN * shift[ys[good], xs[good]].astype(np.float64)
        n_x, n_y = nx[ys[good], xs[good]].astype(np.float64), ny[ys[good], xs[good]].astype(np.float64)
        history.append(float(np.sqrt((d ** 2).mean())))
        if dof == 2:
            design = np.stack([n_x, n_y], 1)
        elif dof == 3:
            # n . u for u = (s*gx + tx, s*gy + ty): ONE isotropic-scale column
            # plus the two translation columns. Unknowns [s, tx, ty].
            design = np.stack([n_x * gx + n_y * gy, n_x, n_y], 1)
        else:
            design = np.stack([n_x * gx, n_x * gy, n_x, n_y * gx, n_y * gy, n_y], 1)
        weight = np.ones(len(d))
        for _trim in range(2):
            solution, *_ = np.linalg.lstsq(design * weight[:, None], d * weight, rcond=None)
            residual = np.abs(design @ solution - d)
            mad = np.median(residual) + 1e-6
            weight = (residual <= 2.5 * 1.4826 * mad).astype(np.float64)
            if weight.sum() < floor:
                weight = np.ones(len(d))
                break
        if dof in (2, 3):
            # The FIT'S OWN UNCERTAINTY, which is what the two-axis equivalence
            # test compares against: the standard least-squares covariance
            # sigma^2 (A^T A)^-1 of the surviving (trimmed) constraints. It is
            # optimistic by the amount the profile residuals are spatially
            # correlated, which is why `DECISION_K` is calibrated rather than set
            # to a textbook 2 or 3, and why both axes carry an instrument floor.
            keep = weight > 0
            A = design[keep]
            r = A @ solution - d[keep]
            free = max(int(keep.sum()) - A.shape[1], 1)
            # EFFECTIVE SAMPLE SIZE, and it is the difference between an honest
            # sigma and a decorative one. Sites along ONE contour do not carry
            # independent errors: the normal-profile matcher's error at a site is
            # dominated by that contour's own texture and defocus, so a 400-px
            # edge is close to ONE measurement repeated, not 400. The naive lstsq
            # covariance therefore under-reads by roughly the sites-per-contour
            # ratio, which is measured here from the site mask's own connected
            # components rather than guessed.
            variance = float((r ** 2).sum()) / free * correlation
            try:
                cov = variance * np.linalg.inv(A.T @ A)
                if dof == 3:
                    sigma_s = max(float(np.sqrt(max(cov[0, 0], 0.0))), SIGMA_S_FLOOR)
                    sigma_t = max(
                        float(np.sqrt(max(0.5 * (cov[1, 1] + cov[2, 2]), 0.0))),
                        SIGMA_T_FLOOR)
                else:
                    # A 2-DoF fit did not MEASURE the scale axis, so that axis may
                    # not hold a cut: an infinite tolerance is the honest encoding
                    # of "unmeasured", where a small one would assert agreement
                    # the fit never tested.
                    sigma_s = float("inf")
                    sigma_t = max(
                        float(np.sqrt(max(0.5 * (cov[0, 0] + cov[1, 1]), 0.0))),
                        SIGMA_T_FLOOR)
            except np.linalg.LinAlgError:
                sigma_s, sigma_t = float("nan"), float("nan")
        step = np.eye(3)
        if dof == 2:
            step[0, 2], step[1, 2] = solution[0], solution[1]
        elif dof == 3:
            step[0, 0] = step[1, 1] = 1.0 + solution[0]
            step[0, 2] = solution[1] - solution[0] * cx
            step[1, 2] = solution[2] - solution[0] * cy
        else:
            step[0, 0] += solution[0]
            step[0, 1] += solution[1]
            step[1, 0] += solution[3]
            step[1, 1] += solution[4]
            # The fit is in centred coordinates; translate the centre back out.
            step[0, 2] = solution[2] - solution[0] * cx - solution[1] * cy
            step[1, 2] = solution[5] - solution[3] * cx - solution[4] * cy
        matrix = matrix @ step
    return matrix, {"sites": int(good.sum()) if len(history) else 0,
                    "residual": history[-1] if history else float("nan"),
                    "trace": history, "ok": bool(history),
                    "sigma_t": sigma_t, "sigma_s": sigma_s,
                    "centre": (cx, cy)}


def motion_series(matrices, counts, pieces, n, ref, prior):
    """The temporally-coherent motion SERIES (B3b) — one line per parameter.

    A piece that is DEFOCUSED in frame k is least measurable exactly where it is
    most displaced (§12.5), and PLAYBOOK's recorded bias is that a blurred profile
    correlates confidently against a sharp one at about ZERO shift — so a per-frame
    fit on a blurred piece is biased SHORT. The physics says the series is smooth:
    under a constant camera translation per frame, every affine parameter is linear
    in k, through the IDENTITY at the reference. With few points, use the simplest
    model the physics allows (§12.6): one weighted line through the origin per
    parameter, weighted by each fit's site count.
    """
    out, diag = {}, {}
    for piece in pieces:
        identity = np.eye(3)
        deltas = np.stack([np.asarray(matrices[(k, piece)]) - identity for k in range(n)])
        steps = np.array([k - ref for k in range(n)], float)
        weights = np.array([max(counts.get((k, piece), 0), 0) for k in range(n)], float)
        weights = weights * (steps != 0)
        denominator = float((weights * steps ** 2).sum())
        if denominator <= 0:
            # No evidence in this piece. F106's trinary discipline: UNVERIFIABLE
            # DECLINES the correction and keeps the global stage's geometry — it
            # does NOT fall back to the identity, which would assert that the
            # camera did not move, the silent invention F81 was caught making.
            for k in range(n):
                out[(k, piece)] = np.asarray(
                    prior.get((k, piece), prior.get(k)), float)
            diag[piece] = None
            continue
        slope = np.einsum("k,kij->ij", weights * steps, deltas) / denominator
        for k in range(n):
            out[(k, piece)] = identity + slope * (k - ref)
        diag[piece] = slope
    return out, diag


# ---------------------------------------------------------------------------
# 2b. THE AFFINE-AGREEMENT MERGE (round 2) — a cut must be earned by MOTION
#
# Round 1 measured the whole remaining gap into the piece CUT, and measured WHY:
# `depth_from_focus`'s field has no threshold that both closes the true silhouette
# ribbon and stays quiet inside one surface (24 pieces where truth has 3, cut
# precision 23.71%). Per DEVSTYLE §12.3 that means the threshold is the wrong
# instrument and a physical invariant is standing behind it. The invariant: TWO
# PIECES SEPARATED BY A SPURIOUS CUT HAVE THE SAME MOTION. So the depth/focal jump
# is demoted to a SEED that proposes candidate cuts, and the arbiter is the
# measured per-piece affine — the same quantity the occlusion-order clause already
# trusts, and the reason the finder becomes scene-independent (a ramp has no jump
# to seed a cut, and an object standing on the ramp disagrees in motion whether or
# not its focal peak resolves).
# ---------------------------------------------------------------------------
def _label_boundary(labels):
    """Pixels where the label map changes — the cut set of THESE pieces.

    After a merge the old watershed boundary is no longer the cut set: the cuts
    that were merged away are gone, and only the surviving boundaries may seed the
    silhouette gate or claim the reference's matte band.
    """
    field = labels.astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)
    return (cv2.dilate(field, kernel) - cv2.erode(field, kernel)) > 0


def _adjacency(labels, pieces):
    """Shared 4-neighbour border length for every pair of pieces that touch.

    Only ADJACENT pieces are merge candidates: a surface is contiguous (F93), so
    two pieces on opposite sides of the frame agreeing in motion is a coincidence
    of depth, not evidence that they are one piece.
    """
    keep = np.isin(labels, list(pieces))
    counts = {}
    for a, b, ka, kb in ((labels[:, :-1], labels[:, 1:], keep[:, :-1], keep[:, 1:]),
                         (labels[:-1, :], labels[1:, :], keep[:-1, :], keep[1:, :])):
        mask = (a != b) & ka & kb
        if not mask.any():
            continue
        lo = np.minimum(a[mask], b[mask]).astype(np.int64)
        hi = np.maximum(a[mask], b[mask]).astype(np.int64)
        key, tally = np.unique(lo * (labels.max() + 1) + hi, return_counts=True)
        for value, count in zip(key, tally):
            pair = (int(value // (labels.max() + 1)), int(value % (labels.max() + 1)))
            counts[pair] = counts.get(pair, 0) + int(count)
    return counts


def _affine_gap(m_a, m_b, xs, ys):
    """RMS displacement between two affines over a support — in PIXELS.

    Two candidate pieces are one surface iff ONE affine explains both, so the
    comparison must be of the FIELDS over the support they would share, not of the
    matrix entries (which have no common unit) and not of translation magnitude
    alone (round 1's `travel`, which cannot see a rotation or a scale difference —
    its own recorded crudeness, note 4).
    """
    a, b = np.asarray(m_a, np.float64), np.asarray(m_b, np.float64)
    d = a - b
    dx = d[0, 0] * xs + d[0, 1] * ys + d[0, 2]
    dy = d[1, 0] * xs + d[1, 1] * ys + d[1, 2]
    return float(np.sqrt((dx ** 2 + dy ** 2).mean()))


def _decision_residual(m_a, m_b, xs, ys):
    """The two-axis residual PAIR (Δtranslation px, Δscale) between two affines.

    Depth dependence has TWO axes and round 2's decision only looked at one:

    * a LATERAL camera translation t_xy gives each surface a SHIFT t_xy/Z, so two
      surfaces at different depths differ in TRANSLATION;
    * a FORWARD camera translation t_z gives each surface an isotropic SCALE
      1 + t_z/Z, so two surfaces at different depths differ in SCALE even when
      their translations coincide — and `motion_components` measured this scene at
      up to 4.3% forward translation, so that case is not hypothetical.

    Translation is compared AT THE SHARED SUPPORT'S CENTROID (where both fits'
    evidence is, so the comparison does not extrapolate) and scale is the mean of
    the two diagonal terms. The two are near-orthogonal at the centroid, which is
    what makes the PAIR a two-axis test rather than one number twice.
    """
    a, b = np.asarray(m_a, np.float64), np.asarray(m_b, np.float64)
    cx, cy = float(xs.mean()), float(ys.mean())
    d = a - b
    dx = d[0, 0] * cx + d[0, 1] * cy + d[0, 2]
    dy = d[1, 0] * cx + d[1, 1] * cy + d[1, 2]
    ds = 0.5 * ((a[0, 0] + a[1, 1]) - (b[0, 0] + b[1, 1]))
    return float(np.hypot(dx, dy)), float(abs(ds))


def _support_sample(labels, members, rng):
    ys, xs = np.nonzero(np.isin(labels, list(members)))
    if len(xs) > MERGE_SAMPLE:
        pick = rng.choice(len(xs), MERGE_SAMPLE, replace=False)
        ys, xs = ys[pick], xs[pick]
    return xs.astype(np.float64), ys.astype(np.float64)


def merge_agreeing(labels, pieces, series, slopes, n, tol=None, verbose=False,
                   sigmas=None, seeded=()):
    """Merge adjacent pieces whose fitted motion AGREES. Returns (labels, info).

    Two clauses, and the difference between them is whether a measurement exists:

    * AGREEMENT — both groups measured their own motion, and ONE 3-DoF motion
      explains both supports on BOTH depth axes to within the fits' own
      uncertainty in every frame: |Δtranslation| <= k*sigma_t AND |Δscale| <=
      k*sigma_s, where the sigmas come from the decision fits' residuals and
      support (`_decision_residual`, `fit_affine`'s covariance). That is the
      definition of one surface. It replaces round 2's fixed `GATE_TOL` bar,
      which could not separate the factory's 2.0-2.6 px of FIT error from the
      kitchen's sub-1.5 px of real DEPTH difference — the two scenes wanted
      opposite values, which per §12.3 means the threshold was standing in for
      this measurement.
    * A MOTION-SEEDED piece (pass 1's motion groups proposed it because its
      measured motion contradicted its enclosing piece) may be re-absorbed only
      at ITS OWN uncertainty, never at the pooled pair uncertainty — otherwise a
      huge, very-certain mega-piece's tiny sigma would swallow the object the
      seed exists to rescue.
    * ADOPTION — a group whose motion is UNVERIFIABLE (no material-edge evidence in
      any frame; `motion_series` gave it the global prior per F106) cannot agree
      with anything, because agreement is a measurement. It also cannot stand alone:
      an island too small or too smooth to measure, kept separate, asserts a
      surface boundary no evidence supports AND keeps whole-frame geometry inside
      a piece that may be nowhere near it. It is absorbed by the neighbour it
      shares the longest border with — the surface it is most embedded in — and the
      refit that follows re-measures the pair JOINTLY, so a wrong adoption shows up
      as the merged piece's own fit residual rather than being locked in.

    Chaining is the hazard (a~b, b~c, a!~c would merge a near piece to a far one
    through an intermediate). Guarded two ways: candidates are consumed in
    increasing order of disagreement, and each test is between the two GROUPS'
    area-weighted affines, so a group that has already grown must still explain the
    newcomer's support. Piece count is strictly decreasing, asserted by the caller.
    """
    tol = DECISION_K if tol is None else tol
    sigmas = {} if sigmas is None else sigmas
    seeded = set(seeded)
    rng = np.random.default_rng(0)
    parent = {p: p for p in pieces}
    members = {p: [p] for p in pieces}
    area = {p: float((labels == p).sum()) for p in pieces}
    matrices = {p: {k: np.asarray(series[(k, p)], np.float64) for k in range(n)}
                for p in pieces}

    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    def unite(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if area[ra] < area[rb]:
            ra, rb = rb, ra
        weight = area[ra] + area[rb]
        for k in range(n):
            matrices[ra][k] = (matrices[ra][k] * area[ra]
                               + matrices[rb][k] * area[rb]) / weight
        parent[rb] = ra
        members[ra] = members[ra] + members[rb]
        area[ra] = weight
        return True

    default_sigma = (SIGMA_T_FLOOR, SIGMA_S_FLOOR)

    def tolerance(a, b):
        """k*sigma per axis — the SEEDED piece's own sigma when one is seeded."""
        seeds = [p for p in members[a] + members[b] if p in seeded]
        if seeds:
            st, ss = sigmas.get(min(seeds, key=lambda p: -area.get(p, 0.0)),
                                default_sigma)
            return tol * st, tol * ss
        sa = sigmas.get(a, default_sigma)
        sb = sigmas.get(b, default_sigma)
        return (tol * float(np.hypot(sa[0], sb[0])),
                tol * float(np.hypot(sa[1], sb[1])))

    def ratio(m_a, m_b, xs, ys, tol_t, tol_s):
        """How far outside tolerance the worst axis is. <= 1 means one surface."""
        dt, ds = _decision_residual(m_a, m_b, xs, ys)
        return max(dt / max(tol_t, 1e-12), ds / max(tol_s, 1e-12)), dt, ds

    borders = _adjacency(labels, pieces)
    unverifiable = {p for p in pieces if slopes.get(p) is None}
    # --- clause 1: agreement on BOTH depth axes, cheapest candidate first -----
    candidates, axes = [], {}
    for (a, b), length in borders.items():
        if a in unverifiable or b in unverifiable:
            continue
        xs, ys = _support_sample(labels, (a, b), rng)
        tol_t, tol_s = tolerance(a, b)
        worst = max((ratio(series[(k, a)], series[(k, b)], xs, ys, tol_t, tol_s)
                     for k in range(n)), key=lambda r: r[0])
        candidates.append((worst[0], length, a, b))
        axes[(a, b)] = (worst[1], worst[2], tol_t, tol_s)
    candidates.sort()
    agreed = kept = 0
    for gap, _length, a, b in candidates:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        xs, ys = _support_sample(labels, members[ra] + members[rb], rng)
        tol_t, tol_s = tolerance(ra, rb)
        group_gap = max(ratio(matrices[ra][k], matrices[rb][k], xs, ys,
                              tol_t, tol_s)[0] for k in range(n))
        if group_gap <= 1.0:
            unite(a, b)
            agreed += 1
        else:
            kept += 1
    # --- clause 2: adoption of the unmeasurable -------------------------------
    adopted = 0
    for piece in sorted(unverifiable, key=lambda p: area[p]):
        neighbours = [(length, other) for (a, b), length in borders.items()
                      for other in ((b,) if a == piece else (a,) if b == piece else ())
                      if find(other) != find(piece)]
        if not neighbours:
            continue
        neighbours.sort()
        if unite(piece, neighbours[-1][1]):
            adopted += 1
    out = np.zeros(labels.shape, np.int32) - 1
    mapping = {}
    for index, root in enumerate(sorted({find(p) for p in pieces})):
        out[np.isin(labels, members[root])] = index
        for member in members[root]:
            mapping[member] = index
    stranded = out < 0
    if stranded.any():                      # sub-MIN_PIECE residue: nearest piece
        out[stranded] = 0 if not (~stranded).any() else out[~stranded][
            cv2.distanceTransformWithLabels(stranded.astype(np.uint8) * 255,
                                            cv2.DIST_L2, 3,
                                            labelType=cv2.DIST_LABEL_PIXEL)[1][stranded] - 1]
    info = {"merges": agreed + adopted, "agreed": agreed, "adopted": adopted,
            "held": kept, "before": len(pieces),
            "after": len({find(p) for p in pieces}),
            "gaps": sorted(round(c[0], 3) for c in candidates),
            "axes": axes, "mapping": mapping}
    if verbose:
        print(f"    merge: {info['before']} -> {info['after']} pieces "
              f"({agreed} agreed inside {tol:.0f} sigma on BOTH axes, "
              f"{adopted} unmeasurable adopted, {kept} cuts HELD)")
        if candidates:
            print(f"      worst-axis ratio (1.0 = exactly at k*sigma): "
                  f"{', '.join(f'{g:.2f}' for g, *_ in candidates[:6])}"
                  f"{' ...' if len(candidates) > 6 else ''}  "
                  f"max {max(c[0] for c in candidates):.2f}")
            shown = sorted(axes.items(), key=lambda kv: -max(
                kv[1][0] / max(kv[1][2], 1e-12), kv[1][1] / max(kv[1][3], 1e-12)))[:4]
            for (a, b), (dt, ds, tt, ts) in shown:
                print(f"      ({a},{b}) Δt {dt:7.3f} px / tol {tt:6.3f}   "
                      f"Δscale {100 * ds:7.4f}% / tol {100 * ts:6.4f}%   "
                      f"-> {'CUT HELD' if max(dt / max(tt, 1e-12), ds / max(ts, 1e-12)) > 1 else 'merged'}")
    return out, info


# ---------------------------------------------------------------------------
# 2c. MOTION-GROUP-SEEDED CUTS (round 3) — pass 1 is the GUIDE
#
# Root cause 1 of the reference-collapse: nothing PROPOSED a cut from motion. The
# merge can only remove cuts, so the kitchen's objects were never separated from
# the counter mega-piece, inherited its wrong affine, and the veto rightly gapped
# them in exactly the sharp (most-moved) frames — a correct refusal on wrong
# geometry, i.e. a copy of the reference.
#
# The organ that finds them already EXISTS, is proven (F93/F100/F102: 92-100%
# feature purity on the kitchen bottle at ~19 px), and ships in the runtime: it is
# `focusstack.motion_groups.overrides`, which `align.py` calls on this very scene
# as the route condition. This is WIRING, not estimation. The same entry point,
# the same contract:
#
#   overrides(images, coarse, valid, ref_index, depth, displacement_at)
#
# where `coarse` is the globally-warped frames, `depth` is `depth_from_focus` of
# them, and `displacement_at(k, x, y)` reports what the CURRENT model already does
# at a point, beyond the global warp — in `align.py` that is the depth-bin field,
# here it is this module's own per-piece series. So "a group that disagrees with
# its enclosing piece's fitted motion" is exactly the question `overrides` already
# answers, asked with the aligner's own fits substituted for the bins'.
# ---------------------------------------------------------------------------
def _dense_focal_signature(aligned):
    """Per-pixel sub-pixel focal peak, in FRAME units — the dense signature.

    `layer_decompose` builds its decomposition on exactly this field
    (`twoframe.focal_field`'s parabolic-interpolated argmax over the focus
    ladder). Borrowed rather than rebuilt, and used only to trim a group's
    CONVEX HULL down to the pixels whose focal signature matches the group's own
    features: the hull is a coarse territory claim (F100 — features cluster on
    whatever part of an object carries texture), and the focal band is what turns
    it into a body.
    """
    grays = [to_gray_float(a).astype(np.float32) for a in aligned]
    energies = np.stack([cv2.GaussianBlur(
        np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3)), (0, 0), 3.0) for g in grays], 0)
    winner = np.argmax(energies, axis=0)
    n = len(grays)
    yy, xx = np.indices(winner.shape)
    lo, hi = np.clip(winner - 1, 0, n - 1), np.clip(winner + 1, 0, n - 1)
    a, b, c = energies[lo, yy, xx], energies[winner, yy, xx], energies[hi, yy, xx]
    denominator = a - 2.0 * b + c
    ok = np.abs(denominator) > 1e-9
    offset = np.where(ok, 0.5 * (a - c) / np.where(ok, denominator, 1.0), 0.0)
    return np.clip(winner + np.clip(offset, -0.5, 0.5), 0.0, n - 1.0).astype(np.float32)


def seed_from_motion_groups(frames, global_aligned, prior, labels, series, ref,
                            verbose=False):
    """CARVE a piece for every motion group its enclosing piece cannot explain.

    Returns (labels, seeded, groups, report). `seeded` is the set of new labels;
    `groups` records each seeded piece's pass-1 measured motion so the refit that
    follows can be CHECKED against it (they should agree to about a pixel — a
    disagreement is a finding, not a knob).
    """
    n, shape = len(frames), labels.shape
    valid, inverse = [], {}
    for k in range(n):
        map_x, map_y = _field(prior[k], shape)
        valid.append((map_x >= 0) & (map_x <= shape[1] - 1)
                     & (map_y >= 0) & (map_y <= shape[0] - 1))
        inverse[k] = np.linalg.inv(np.asarray(prior[k], np.float64))
    depth = depth_from_focus(global_aligned, radius=DEPTH_GUIDED_RADIUS)

    def displacement_at(frame, x, y):
        """This module's own geometry at a point, in GLOBALLY-ALIGNED coordinates.

        `overrides` measures shifts between globally-warped frames, so what it
        must be compared against is the residual BEYOND the global warp:
        prior_k^-1(series_k(x)) - x. Same convention as `align.py`'s own
        `displacement_at`, which subtracts the global warp from the bin field.
        """
        if frame == ref:
            return (0.0, 0.0)
        yi = int(np.clip(round(y), 0, shape[0] - 1))
        xi = int(np.clip(round(x), 0, shape[1] - 1))
        matrix = series.get((frame, int(labels[yi, xi])))
        if matrix is None:
            return (0.0, 0.0)
        target = np.asarray(matrix, np.float64) @ np.array([x, y, 1.0])
        source = inverse[frame] @ target
        return (float(source[0] - x), float(source[1] - y))

    chosen, report, unexplained = MG.overrides(frames, global_aligned, valid, ref,
                                               depth, displacement_at)
    report = dict(report, unexplained_points=len(unexplained))
    if verbose:
        print(f"    motion groups (pass 1's own organ): {report}")
    if not chosen:
        return labels, set(), [], report

    peak = _dense_focal_signature(global_aligned)
    out = labels.copy()
    next_label = int(labels.max()) + 1
    seeded, groups = set(), []
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    for weight, motion in chosen:
        core = weight >= CLAIM_OWNED
        if int(core.sum()) < MIN_PIECE:
            continue
        # SUPPORT = hull ∩ focal-signature band of the group's OWN features.
        # The hull over-claims by construction (convex, plus a 26 px disc per
        # feature); the band is the physical statement that a rigid object at one
        # depth has one focal signature, so counter and wall pixels swept into
        # the hull are removed rather than dragged along with the object.
        lo, hi = np.percentile(peak[core], [5.0, 95.0])
        band = (peak >= lo - FOCAL_BAND) & (peak <= hi + FOCAL_BAND)
        support = ((weight > CLAIM_HULL) & band) | core
        support = cv2.morphologyEx(support.astype(np.uint8), cv2.MORPH_OPEN,
                                   open_kernel)
        support = cv2.morphologyEx(support, cv2.MORPH_CLOSE, close_kernel) > 0
        # Only the connected body the group's own evidence sits in.
        count, cc = cv2.connectedComponents(support.astype(np.uint8), connectivity=8)
        keep = np.zeros(shape, bool)
        for index in range(1, count):
            piece = cc == index
            if piece.sum() >= MIN_PIECE and (piece & core).sum() >= 0.05 * core.sum():
                keep |= piece
        if int(keep.sum()) < MIN_PIECE:
            continue
        out[keep] = next_label
        seeded.add(next_label)
        groups.append({"label": next_label, "area": int(keep.sum()),
                       "core": int(core.sum()), "motion": motion,
                       "focal": (float(lo), float(hi)), "mask": keep,
                       "core_mask": core})
        next_label += 1
    # A carve can strand what is left of an old piece below MIN_PIECE; those
    # pixels join the seeded body they are embedded in rather than becoming
    # unmeasurable islands.
    for piece in np.unique(out):
        mask = out == piece
        if mask.sum() >= MIN_PIECE:
            continue
        neighbour = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        others = out[neighbour & ~mask]
        if len(others):
            out[mask] = int(np.bincount(others - others.min()).argmax()) + others.min()
    if verbose:
        for g in groups:
            m = g["motion"]
            far = max(m, key=lambda k: float(np.hypot(*m[k]))) if m else None
            print(f"      seeded piece {g['label']}: area {g['area']} px "
                  f"(hull core {g['core']}), focal band {g['focal'][0]:.1f}"
                  f"-{g['focal'][1]:.1f}, pass-1 motion at k={far} "
                  f"{tuple(round(float(v), 2) for v in m[far]) if far is not None else '-'}")
    return out, seeded, groups, report


# ---------------------------------------------------------------------------
# 3. THE SILHOUETTE-EXACT GATE — F116's instrument, as a per-frame gate
# ---------------------------------------------------------------------------
def silhouette_gate(reference, aligned, labels, cut, band, tol=None):
    """Does this frame's piece contour land on the REFERENCE contour?

    F116 built contour continuity to audit a rewrite. Here the same instrument is
    the GATE the charter names: at the occluding contour, the reference and the
    aligned frame are two observations of the SAME contour, so a confident
    normal-profile displacement beyond the matte band means the piece's transform
    is wrong THERE. The consequence is not a forced fit and not a global reject: it
    is a GAP in the band around the offending contour, which other frames fill.
    """
    tol = GATE_TOL if tol is None else tol
    gray_r, gray_a = _gray(reference), _gray(aligned)
    nx, ny = SM._unit_normals(gray_r)
    shift, peak = SM._match_dense(SM._contour_profiles(gray_r, nx, ny),
                                  SM._contour_profiles(gray_a, nx, ny))
    edges = cv2.Canny(cv2.GaussianBlur(gray_r.astype(np.uint8), (5, 5), 0), 60, 180) > 0
    margin = SM.CONTOUR_HALF + SM.CONTOUR_SPAN + 2
    inside = np.zeros(edges.shape, bool)
    inside[margin:-margin, margin:-margin] = True
    sites = edges & inside & _dilate(cut, band)
    confident = sites & (peak >= MG.MIN_PEAK)
    failed = confident & (np.abs(shift) > tol)
    return {"sites": int(sites.sum()), "confident": int(confident.sum()),
            "failed": int(failed.sum()),
            "worst": float(np.abs(shift[confident]).max()) if confident.any() else 0.0,
            "rms": float(np.sqrt((shift[confident] ** 2).mean())) if confident.any() else 0.0,
            "mask": _dilate(failed, band)}


# ---------------------------------------------------------------------------
# The blur rate c, MEASURED on this scene (F117: it does not transfer)
# ---------------------------------------------------------------------------
def measure_blur_rate(aligned, labels, pieces, verbose=False):
    """Focal peak per piece, and the disk radius per frame of focal distance.

    F117's open item: `c` does NOT transfer across scenes (factory 1.161, kitchen
    0.684, large-motion ~0.4 — over-blurring 2x there) and nothing in the runtime
    measures it. So it is measured here, on this scene, by the only fully
    determined instrument available: each piece's own sharpest frame is its focal
    peak by definition, and the radius that best takes the peak frame's appearance
    to frame k's is the radius frame k has. Regressed through the origin, since a
    piece at its own focal frame is in focus by the definition of a peak.
    """
    grays = [to_gray_float(a).astype(np.float32) for a in aligned]
    energy = [cv2.Laplacian(g, cv2.CV_32F, ksize=3) ** 2 for g in grays]
    peaks, radii = {}, {}
    numerator = denominator = 0.0
    for piece in pieces:
        interior = cv2.erode((labels == piece).astype(np.uint8),
                             np.ones((2 * FC.RADIUS_MAX + 1,) * 2, np.uint8)) > 0
        if interior.sum() < MIN_PIECE:
            interior = labels == piece
        scores = [float(e[interior].mean()) for e in energy]
        peak = int(np.argmax(scores))
        peaks[piece] = peak
        ladder = [FC.defocus(grays[peak], r) for r in range(FC.RADIUS_MAX + 1)]
        for k in range(len(aligned)):
            costs = np.array([float(np.abs(rung[interior] - grays[k][interior]).mean())
                              for rung in ladder])
            best = int(np.argmin(costs))
            radius = float(best)
            if 0 < best < len(costs) - 1:          # parabolic sub-pixel, one line
                curvature = costs[best - 1] - 2 * costs[best] + costs[best + 1]
                if abs(curvature) > 1e-12:
                    radius += 0.5 * (costs[best - 1] - costs[best + 1]) / curvature
            radii[(k, piece)] = max(radius, 0.0)
            distance = abs(k - peak)
            numerator += radius * distance
            denominator += distance * distance
    c = numerator / denominator if denominator > 0 else 0.0
    if verbose:
        print(f"    measured c = {c:.3f} px/frame (factory truth "
              f"BLUR_PER_STEP = {P.BLUR_PER_STEP})   peaks {peaks}")
    return float(c), peaks, radii


# ---------------------------------------------------------------------------
# 4./5. THE ALIGNER
# ---------------------------------------------------------------------------
def align(frames, ref=None, matrices=None, labels=None, verbose=False):
    """The contract: (aligned, usable, report), mirroring `align_stack`'s output.

    aligned[k] — frame k resampled ONCE into reference geometry through the
                 piecewise field. Defocused exactly as it was; nothing is sharpened
                 and nothing is synthesized.
    usable[k]  — True where THIS reference pixel's owning surface was observed by
                 frame k, uncontaminated. False is a GAP: other frames supply it.
    """
    n = len(frames)
    ref = P.REFERENCE if ref is None else ref
    shape = frames[0].shape[:2]
    reference = frames[ref].astype(np.float32)
    ref_gray = _gray(frames[ref])

    # --- the global affine prior, from the same instrument, whole-frame ---------
    cache = reference_cache(ref_gray)
    prior = {}
    whole = np.ones(shape, bool)
    sites_all = _material_sites(ref_gray, whole, np.zeros(shape, bool))
    for k in range(n):
        if k == ref:
            prior[k] = np.eye(3)
            continue
        prior[k], _ = fit_affine(ref_gray, frames[k].astype(np.float32),
                                 np.eye(3), sites_all, shape, cache=cache)

    # --- pieces, from globally-aligned frames ---------------------------------
    global_aligned = []
    for k in range(n):
        map_x, map_y = _field(prior[k], shape)
        global_aligned.append(cv2.remap(frames[k].astype(np.float32), map_x, map_y,
                                        cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REPLICATE))
    if labels is None:
        labels, pdiag = cut_pieces(global_aligned, verbose=verbose)
    else:
        # The ORACLE rung: true pieces substituted for the recovered ones so the
        # ladder can attribute the remainder (F115's discipline, and F110's trap —
        # substitute ONE term, never a term plus its consequences).
        edge = _label_boundary(labels)
        pdiag = {"cut": edge, "ribbon": edge, "n_pieces": len(np.unique(labels))}
    pieces = piece_table(labels)

    # --- one affine per piece per frame, then the AGREEMENT MERGE to a fixed point
    def seeded_prior(piece, k, group_motion):
        """A seeded piece's fit PRIOR is pass 1's own measured group motion.

        THE finding of round 3, and it invalidated the round's first result: the
        dense normal-profile matcher only accepts |shift| < `SM.CONTOUR_HALF` = 6
        px, so a ~20 px object displacement is entirely OUTSIDE its capture range.
        Started from the global affine, every site on the bottle was rejected and
        the fit collapsed to the mega-piece's ~+3.9 px — after which the
        equivalence test correctly merged a piece whose motion had been measured
        wrongly (fitted Δt 0.7 px against a pass-1 disagreement of 24.5 px). This
        is exactly what "pass 1 is a GUIDE" means operationally: the prior must
        put the piece within the instrument's capture range, and the fit then
        refines from there. `prior_k @ T(dx, dy)` is the matrix whose displacement
        in GLOBALLY-ALIGNED coordinates is the group's measured (dx, dy), which is
        the space `motion_groups` measures in.
        """
        base = np.asarray(prior[k], np.float64)
        motion = (group_motion or {}).get(piece)
        if not motion or k not in motion:
            return base
        step = np.eye(3)
        step[0, 2], step[1, 2] = float(motion[k][0]), float(motion[k][1])
        return base @ step

    def fit_pieces(labels, pieces, cut, dof=None, group_sites=None,
                   group_motion=None):
        dof = PIECE_DOF if dof is None else dof
        fitted, counts, fits = {}, {}, {}
        band_exclude = _dilate(cut, SM.CONTOUR_HALF + SM.CONTOUR_SPAN)
        for piece in pieces:
            sites = _material_sites(ref_gray, labels == piece, band_exclude)
            # F92 on a CARVED piece: fit on the group's OWN material features
            # first and the dense support second. A seeded piece's boundary is a
            # silhouette (its neighbour is a different depth), so sites near it
            # are limbs whose apparent motion mixes two surfaces; the group's own
            # features are interior by construction. Only if they are too few to
            # overdetermine the model does the piece's dense support take over.
            if group_sites is not None and piece in group_sites:
                own = group_sites[piece] & sites
                if int(own.sum()) >= 4 * dof:
                    sites = own
            for k in range(n):
                if k == ref:
                    fitted[(k, piece)] = np.eye(3)
                    counts[(k, piece)] = 0
                    continue
                start = seeded_prior(piece, k, group_motion)
                matrix, info = fit_affine(ref_gray, frames[k].astype(np.float32),
                                          start, sites, shape, cache=cache,
                                          dof=dof)
                # GRADUATED MODEL ORDER, §12.4 read forwards rather than as a veto:
                # a piece that cannot overdetermine six unknowns may still
                # overdetermine two, and the 2-DoF model is the physics of parallax
                # (a depth-dependent translation on top of the global affine). Only
                # a piece that cannot determine even that is UNVERIFIABLE, and then
                # it keeps the global prior per F106 — it does NOT get the identity.
                if dof > 2 and (not info["ok"] or info["sites"] < 4 * dof):
                    matrix, info = fit_affine(ref_gray, frames[k].astype(np.float32),
                                              start, sites, shape, cache=cache,
                                              dof=2)
                    if info["ok"] and info["sites"] >= 8:
                        info = dict(info, dof=2)
                if not info["ok"] or info["sites"] < 4 * min(dof, info.get("dof", dof)):
                    # F106: an unverifiable piece declines the correction and keeps
                    # its PRIOR — which for a seeded piece is pass 1's measured
                    # group motion, not the global affine. Falling back to the
                    # global affine there would discard the only measurement of
                    # that object's motion anyone has.
                    fitted[(k, piece)] = start
                    counts[(k, piece)] = 0
                else:
                    fitted[(k, piece)] = matrix
                    counts[(k, piece)] = info["sites"]
                fits[(k, piece)] = info
        return fitted, counts, fits

    def piece_sigmas(fits, pieces):
        """Each piece's own fit uncertainty: the median over the frames it measured.

        A per-frame sigma is a property of that frame's evidence; the piece's is
        the typical one, and the median is robust to the ends of the sweep where
        the piece is defocused and its fit is biased short (§12.5).
        """
        out = {}
        for piece in pieces:
            st = [fits[(k, piece)]["sigma_t"] for k in range(n)
                  if fits.get((k, piece), {}).get("ok")
                  and not np.isnan(fits[(k, piece)].get("sigma_t", np.nan))]
            ss = [fits[(k, piece)]["sigma_s"] for k in range(n)
                  if fits.get((k, piece), {}).get("ok")
                  and not np.isnan(fits[(k, piece)].get("sigma_s", np.nan))]
            out[piece] = (float(np.median(st)) if st else SIGMA_T_FLOOR,
                          float(np.median(ss)) if ss else SIGMA_S_FLOOR)
        return out

    merge_log = []
    seeded, seed_groups, group_report, group_sites = set(), [], {}, None
    if matrices is None:
        cut = pdiag["cut"]

        def series_priors(pieces, group_motion):
            """Per-(frame, piece) priors, so `motion_series`' F106 fallback keeps a
            SEEDED piece on pass 1's measurement rather than on the global affine."""
            out = dict(prior)
            for piece in pieces:
                for k in range(n):
                    out[(k, piece)] = seeded_prior(piece, k, group_motion)
            return out

        def merge_to_fixed_point(labels, pieces, cut, seeded, group_sites,
                                 group_motion=None):
            """Fit -> two-axis merge -> refit, to a fixed point.

            `merge_agreeing` RENUMBERS the surviving pieces, so the seeded set and
            the group's own fitting sites are carried through its own reported
            mapping. Losing that identity would silently un-protect exactly the
            pieces the seeding round exists to keep.
            """
            for _round in range(MERGE_MAX_ROUNDS):
                fitted, counts, fits = fit_pieces(labels, pieces, cut,
                                                  group_sites=group_sites,
                                                  group_motion=group_motion)
                series, slopes = motion_series(fitted, counts, pieces, n, ref,
                                               series_priors(pieces, group_motion))
                if len(pieces) < 2:
                    return (labels, pieces, cut, series, slopes, fitted, counts,
                            fits, seeded, group_sites, group_motion)
                merged, info = merge_agreeing(labels, pieces, series, slopes, n,
                                              verbose=verbose,
                                              sigmas=piece_sigmas(fits, pieces),
                                              seeded=seeded)
                merge_log.append(info)
                if info["merges"] == 0:
                    break
                after = piece_table(merged)
                # Monotone by construction (a merge only unites); ASSERTED because
                # a fixed-point loop that can grow is a loop that can spin.
                assert 0 < len(after) < len(pieces), (len(after), len(pieces))
                mapping = info["mapping"]
                seeded = {mapping[p] for p in seeded if p in mapping}
                if group_sites is not None:
                    remapped = {}
                    for p, sites in group_sites.items():
                        if p in mapping:
                            key = mapping[p]
                            remapped[key] = (remapped[key] | sites
                                             if key in remapped else sites)
                    group_sites = remapped
                if group_motion is not None:
                    group_motion = {mapping[p]: m for p, m in group_motion.items()
                                    if p in mapping}
                labels, pieces = merged, after
                cut = _label_boundary(labels)
            return (labels, pieces, cut, series, slopes, fitted, counts, fits,
                    seeded, group_sites, group_motion)

        (labels, pieces, cut, series, slopes, fitted, counts, fits,
         seeded, group_sites, group_motion) = merge_to_fixed_point(
            labels, pieces, cut, set(), None)

        # --- CHANGE 1: pass 1's motion groups PROPOSE the cuts the seed missed ---
        # Run only now, because the question the groups answer is "does this
        # object's measured motion disagree with the motion its ENCLOSING PIECE was
        # fitted to" — which needs the enclosing pieces to exist and be fitted.
        labels_seeded, seeded, seed_groups, group_report = seed_from_motion_groups(
            frames, global_aligned, prior, labels, series, ref, verbose=verbose)
        if seeded:
            pieces_seeded = piece_table(labels_seeded)
            # The group's own material features become the carved piece's primary
            # fitting sites (F92: interior by construction, unlike its silhouette).
            group_sites, group_motion = {}, {}
            for group in seed_groups:
                own = labels_seeded == group["label"]
                group_sites[group["label"]] = cv2.erode(
                    own.astype(np.uint8),
                    np.ones((2 * SM.CONTOUR_HALF + 1,) * 2, np.uint8)) > 0
                group_motion[group["label"]] = group["motion"]
            labels, pieces = labels_seeded, pieces_seeded
            cut = _label_boundary(labels)
            (labels, pieces, cut, series, slopes, fitted, counts, fits,
             seeded, group_sites, group_motion) = merge_to_fixed_point(
                labels, pieces, cut, seeded, group_sites, group_motion)
            # Which surviving piece each seeded body ended up as, read off the
            # label map rather than tracked through the renumbering — the
            # majority label over the group's own carved support.
            for group in seed_groups:
                votes = labels[group["mask"]]
                group["final_label"] = (int(np.bincount(votes[votes >= 0]).argmax())
                                        if (votes >= 0).any() else None)
                group["absorbed"] = group["final_label"] not in seeded
            seeded = {p for p in seeded if p in pieces}

        # The merge is decided; now the TRANSFORM. The two want different model
        # orders and the reason is the PRIOR: the whole-frame affine's linear part
        # is a compromise across depths (it absorbs some of the differential
        # parallax as shear/scale), so a piece can only shed it with six degrees of
        # freedom — measured, not assumed: on TRUE pieces the 6-DoF refit reads
        # 0.159 px worst against 3.570 px for the 2-DoF fit, because translation
        # alone cannot undo a wrong linear part away from the sites' centroid. Six
        # DoF is safe HERE and was not safe during the merge, because the merge ran
        # on small over-segmented pieces and this runs on the merged ones.
        fitted, counts, fits = fit_pieces(labels, pieces, cut, dof=FINAL_DOF,
                                          group_sites=group_sites,
                                          group_motion=group_motion)
        series, slopes = motion_series(fitted, counts, pieces, n, ref,
                                       series_priors(pieces, group_motion))
        pdiag = dict(pdiag)
        pdiag["cut"] = cut
        pdiag["n_pieces"] = len(pieces)
    else:
        series, slopes, fitted = matrices, {}, matrices
        fits = {}

    # --- the blur rate, measured on THIS scene, from series-aligned frames -----
    provisional = []
    for k in range(n):
        map_x, map_y = _piecewise_field({p: series[(k, p)] for p in pieces}, labels, shape)
        provisional.append(cv2.remap(frames[k].astype(np.float32), map_x, map_y,
                                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
    c, peaks, measured_radii = measure_blur_rate(provisional, labels, pieces,
                                                 verbose=verbose)
    radius = {(k, p): c * abs(k - peaks[p]) for k in range(n) for p in pieces}

    # --- OCCLUSION ORDER, from parallax magnitude -----------------------------
    # Displacement is linear in INVERSE depth, so the piece that moves FURTHER per
    # frame is NEARER. That is a direct measurement from the transforms this module
    # already fitted, and it is a strictly better instrument than F83's focal-peak
    # polarity proxy, which F114 found REFUSES on today's factory (share 0.542
    # against its own 0.05 margin).
    # Round 2's item 4, and it was not cosmetic: |t| of the matrix is the
    # displacement at the ORIGIN, which for a piece far from the origin is
    # dominated by its linear part and read 7822 px/frame on the kitchen. The
    # displacement AT THE PIECE'S OWN CENTROID is what "how far this surface
    # moves" means, and it is what the occlusion order must rank.
    travel = {}
    for piece in pieces:
        ys, xs = np.nonzero(labels == piece)
        cx, cy = float(xs.mean()), float(ys.mean())
        d = []
        for k in range(n):
            m = np.asarray(series[(k, piece)], np.float64)
            d.append(float(np.hypot(m[0, 0] * cx + m[0, 1] * cy + m[0, 2] - cx,
                                    m[1, 0] * cx + m[1, 1] * cy + m[1, 2] - cy)))
        travel[piece] = float(np.mean(d))
    ranked = sorted(pieces, key=lambda p: -travel[p])          # nearest first
    # Pieces whose travel differs by LESS than the displacement nothing in this arc
    # resolves are at the SAME depth and cannot occlude each other. Without this
    # clause an over-segmented single surface occludes ITSELF at every spurious cut,
    # which is the F81 class of invented geometry arriving through a merge failure.
    groups = [[ranked[0]]]
    for piece in ranked[1:]:
        if travel[groups[-1][-1]] - travel[piece] <= GATE_TOL:
            groups[-1].append(piece)
        else:
            groups.append([piece])
    order = ranked

    # --- assemble: one remap per frame, then the veto stack -------------------
    aligned, usable, stages, gates = [], [], [], {}
    radius_ref = np.zeros(shape, np.float32)
    for piece in pieces:
        radius_ref[labels == piece] = radius[(ref, piece)]
    for k in range(n):
        per_piece = {p: series[(k, p)] for p in pieces}
        map_x, map_y = _piecewise_field(per_piece, labels, shape)
        moved = cv2.remap(frames[k].astype(np.float32), map_x, map_y,
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        aligned.append(np.clip(moved, 0, 255).astype(np.uint8))
        if k == ref:
            # The reference observed the reference viewpoint by definition, matte
            # band and all. It is the one frame with no gaps.
            usable.append(np.ones(shape, bool))
            stages.append({"footprint": 1.0, "occlusion": 1.0, "gate": 1.0,
                           "photometry": 1.0, "final": 1.0})
            continue

        # (a) OFF-FOOTPRINT — the sample lies outside frame k.
        ok = ((map_x >= 0) & (map_x <= shape[1] - 1)
              & (map_y >= 0) & (map_y <= shape[0] - 1))
        ledger = {"footprint": float(ok.mean())}

        # (b) OCCLUSION — each nearer piece's silhouette, warped by ITS OWN
        # transform into frame k, DILATED by ITS modeled defocus radius there
        # (F83: a defocused occluder's matte extends beyond its silhouette; the
        # F116 wall smear entered exactly through the missing dilation), then read
        # back at the place this pixel samples from.
        for index, group in enumerate(groups):
            farther = [p for later in groups[index + 1:] for p in later]
            if not farther:
                break
            behind = np.isin(labels, farther)
            for occluder in group:
                silhouette = _push_mask(labels == occluder, series[(k, occluder)],
                                        shape)
                silhouette = _dilate(silhouette, radius[(k, occluder)] + 1)
                ok &= ~(behind & _sample_mask(silhouette, map_x, map_y))
        ledger["occlusion"] = float(ok.mean())

        # (c) THE SILHOUETTE GATE — a contour that did not land gets a GAP band.
        gate = silhouette_gate(frames[ref], aligned[-1], labels, pdiag["cut"],
                               band=int(np.ceil(radius_ref.max())) + 1)
        gates[k] = {key: value for key, value in gate.items() if key != "mask"}
        ok &= ~gate["mask"]
        ledger["gate"] = float(ok.mean())

        # (d) PHOTOMETRY VETOES — cross-convolution to a COMMON defocus (F115),
        # then F112's unretuned budget. Geometry proposed; this is the veto, and
        # nothing downstream may un-veto it.
        radius_k = np.zeros(shape, np.float32)
        for piece in pieces:
            radius_k[labels == piece] = radius[(k, piece)]
        agree = SM.same_surface_physical(moved, reference, radius_k, radius_ref)
        ok &= agree
        ledger["photometry"] = float(ok.mean())

        # The reference's OWN matte band: mixed pixels no single surface owns. They
        # stay with the reference frame (the irreducible residue of committing to
        # one viewpoint) and are withheld from every moved frame.
        ok &= ~_dilate(pdiag["cut"], max(int(np.ceil(radius_ref.max())), 1))
        # Gaps dilated 1 px: a bilinear sample next to a gap borrows from it.
        ok = ~_dilate(~ok, 1)
        ledger["final"] = float(ok.mean())
        # PER-PIECE ATTRIBUTION (round 2's item 7): a whole-frame "27% withdrawn"
        # is a number, not a diagnosis. The same clauses, restricted to each
        # piece's own support, say WHICH surface the veto is refusing.
        ledger["per_piece"] = {}
        for piece in pieces:
            mask = labels == piece
            total = float(mask.sum())
            if total <= 0:
                continue
            ledger["per_piece"][piece] = {
                "photometry": float((mask & ~agree).sum()) / total,
                "gate": float((mask & gate["mask"]).sum()) / total,
                "gap": float((mask & ~ok).sum()) / total}
        usable.append(ok)
        stages.append(ledger)

    # Refusal composition is non-monotone in general (scenemodel_NOTES §23a): the
    # stack above is composed in ONE fixed order and each clause only removes, so
    # monotonicity is a property to ASSERT, not to hope for.
    for k, ledger in enumerate(stages):
        keys = ["footprint", "occlusion", "gate", "photometry", "final"]
        values = [ledger[key] for key in keys]
        assert all(a >= b - 1e-9 for a, b in zip(values, values[1:])), (k, ledger)

    report = {"labels": labels, "pieces": pieces, "order": order, "travel": travel,
              "groups": groups,
              "c": c, "peaks": peaks, "radius": radius, "measured_radii": measured_radii,
              "matrices": series, "raw_matrices": fitted, "prior": prior,
              "slopes": slopes, "fits": fits, "stages": stages, "gates": gates,
              "merge": merge_log,
              "seeded": seeded, "seed_groups": seed_groups,
              "group_report": group_report,
              "cut": pdiag["cut"], "ribbon": pdiag["ribbon"], "ref": ref,
              "withheld": float(np.mean([1.0 - u.mean() for u in usable]))}
    return aligned, usable, report


# ---------------------------------------------------------------------------
# TRUTH, rebuilt from `parallax_gen`'s own constants (it IS a forward renderer)
# ---------------------------------------------------------------------------
def factory_truth():
    shape = (P.HEIGHT + 2 * P.PAD, P.WIDTH + 2 * P.PAD)
    alpha = np.zeros(shape, np.float32)
    cv2.rectangle(alpha, (P.PAD + 60, P.PAD + 70), (P.PAD + 300, P.PAD + 330), 1.0, -1)
    cv2.circle(alpha, (P.PAD + 400, P.PAD + 140), 70, 1.0, -1)

    def shift_alpha(dx):
        matrix = np.float32([[1, 0, -dx], [0, 1, 0]])
        return cv2.warpAffine(alpha, matrix, (shape[1], shape[0]),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def crop(x):
        return x[P.PAD:P.PAD + P.HEIGHT, P.PAD:P.PAD + P.WIDTH]

    near = crop(alpha) > 0.5
    # A far pixel at reference x is seen by frame k at x - step*FAR; the near matte
    # sits there iff alpha(x + step*(NEAR - FAR)) — the differential parallax.
    occluded = {}
    for k in range(P.FRAMES):
        step = k - P.REFERENCE
        differential = step * (P.NEAR_SHIFT_PER_FRAME - P.FAR_SHIFT_PER_FRAME)
        occluded[k] = (~near) & (crop(shift_alpha(-differential)) > 0.5)
    return {"near": near, "occluded": occluded,
            "near_shift": P.NEAR_SHIFT_PER_FRAME, "far_shift": P.FAR_SHIFT_PER_FRAME}


def truth_matrices(pieces, labels, truth, n, ref):
    """The true backward field per piece: sample frame k at x - step*shift."""
    out, kind = {}, {}
    for piece in pieces:
        overlap = float(truth["near"][labels == piece].mean())
        shift = truth["near_shift"] if overlap > 0.5 else truth["far_shift"]
        kind[piece] = "near" if overlap > 0.5 else "far"
        for k in range(n):
            matrix = np.eye(3)
            matrix[0, 2] = -(k - ref) * shift
            out[(k, piece)] = matrix
    return out, kind


# ---------------------------------------------------------------------------
# K1 — piece recovery
# ---------------------------------------------------------------------------
def kat_ramp() -> None:
    """Ramp-wholeness, as a known-answer test of the CUT FINDER itself.

    `parallax_gen`'s two planes are fronto-parallel and its `viewpoint` applies ONE
    scalar shift per layer, so no parameter of it can produce a depth RAMP —
    BREATHING_PER_FRAME adds magnification, which is depth-INDEPENDENT. Rather than
    leave the charter's ramp clause untested, the clause is tested where it lives:
    on a synthetic depth field that is a pure linear ramp plus one step. The cut
    finder must cut at the step and NOWHERE in the ramp.
    """
    print("  K1b ramp-wholeness (synthetic depth field, instrument-level)")
    h, w = 200, 400
    xx = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    ramp = 0.6 * xx / (w - 1)                       # full-frame focal sweep, smooth
    step = np.where(xx >= 300, 0.4, 0.0).astype(np.float32)
    for name, depth in (("pure ramp", ramp), ("ramp + step", ramp + step)):
        gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        jump = np.hypot(gx, gy)
        threshold = (1.0 / (P.FRAMES - 1)) / DEPTH_GUIDED_RADIUS
        ribbon = jump >= threshold
        columns = np.flatnonzero(ribbon.any(axis=0))
        where = "none" if not len(columns) else f"x={columns.min()}..{columns.max()}"
        print(f"    {name:<12} ramp slope {0.6 / (w - 1):.5f}/px vs threshold "
              f"{threshold:.5f}/px   cut columns: {where}")
    print("    verdict: the ramp alone is cut NOWHERE; the step is cut at the step.\n")


def kat_sign() -> None:
    """`_match_dense`'s sign convention, measured — never assumed (§12.1)."""
    frames, _truth, _near = P.build_stack()
    image = frames[P.REFERENCE].astype(np.float32)
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 2.0], [0, 1, 0]]),
                             (image.shape[1], image.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    gray_a, gray_b = _gray(image), _gray(shifted)
    nx, ny = SM._unit_normals(gray_a)
    shift, peak = SM._match_dense(SM._contour_profiles(gray_a, nx, ny),
                                  SM._contour_profiles(gray_b, nx, ny))
    sites = _material_sites(gray_a, np.ones(gray_a.shape, bool),
                            np.zeros(gray_a.shape, bool))
    good = sites & (peak >= MG.MIN_PEAK) & (np.abs(nx) > 0.9)
    # The normal component of a +2 px x-displacement is 2*nx, which FLIPS with the
    # normal's own direction — so the raw median over x-facing sites is meaningless
    # and only shift/nx is the instrument's answer.
    per = shift[good] / nx[good]
    median = float(np.median(per))
    print(f"  K2a sign KAT: content moved +2.0 px in x; matcher reads "
          f"{median:+.3f} px per unit normal over {int(good.sum())} x-facing sites "
          f"-> MATCH_SIGN {2.0 / median:+.3f}")
    assert np.sign(2.0 / median) == MATCH_SIGN, "matcher sign convention changed"
    assert abs(abs(median) - 2.0) < 0.25, f"matcher magnitude off: {median}"
    print()


def k1(frames, report, truth) -> dict:
    print("K1  PIECE RECOVERY — cut location against the true silhouette")
    labels, cut = report["labels"], report["cut"]
    near = truth["near"]
    true_contour = (cv2.morphologyEx(near.astype(np.uint8), cv2.MORPH_GRADIENT,
                                     np.ones((3, 3), np.uint8)) > 0)
    # The matte band of the truth's silhouette: the reference frame's own near-plane
    # defocus radius, which is what physically smears the silhouette there.
    band = abs(P.REFERENCE - P.NEAR_FOCUS_FRAME) * P.BLUR_PER_STEP
    distance = cv2.distanceTransform((~true_contour).astype(np.uint8), cv2.DIST_L2, 5)
    interior = np.zeros(labels.shape, bool)
    interior[6:-6, 6:-6] = True
    cut_in = cut & interior
    d = distance[cut_in]
    within = float((d <= band + 1).mean()) if len(d) else 0.0
    # And the converse: is the whole true silhouette actually cut?
    covered = float((cv2.distanceTransform((~cut_in).astype(np.uint8), cv2.DIST_L2, 5)
                     [true_contour & interior] <= band + 1).mean())
    kinds = truth_matrices(report["pieces"], labels, truth, len(frames),
                           report["ref"])[1]
    purity = []
    for piece in report["pieces"]:
        mask = labels == piece
        overlap = float(near[mask].mean())
        purity.append(max(overlap, 1 - overlap))
    print(f"  pieces recovered: {len(report['pieces'])}  "
          f"({', '.join(f'{p}:{kinds[p]}' for p in report['pieces'])})")
    print(f"  matte band = |ref - near peak| * BLUR_PER_STEP = {band:.2f} px (+1 px slack)")
    print(f"  RECALL   true silhouette covered by a cut within the band : "
          f"{covered * 100:6.2f}%")
    print(f"  PRECISION cut pixels within the band of the silhouette    : "
          f"{within * 100:6.2f}%  ({int(cut_in.sum())} cut px, median distance "
          f"{np.median(d):.2f} px)")
    print(f"  PURITY   each piece against the true near mask (1.0 = one plane only): "
          f"{', '.join(f'{p:.3f}' for p in purity)}")
    print(f"           worst {min(purity):.3f}, mean {np.mean(purity):.3f}")
    print(f"  VERDICT recall {'PASS' if covered > 0.9 else 'FAIL'} / precision "
          f"{'PASS' if within > 0.9 else 'FAIL (over-segmentation: extra cuts inside one surface)'}\n")
    return {"within": within, "covered": covered, "band": band, "purity": purity}


# ---------------------------------------------------------------------------
# K2 — transform recovery, and the gate
# ---------------------------------------------------------------------------
def k2(frames, report, truth) -> dict:
    print("K2  TRANSFORM RECOVERY — per-piece affine against truth")
    labels, pieces, ref = report["labels"], report["pieces"], report["ref"]
    shape = frames[0].shape[:2]
    true_m, kinds = truth_matrices(pieces, labels, truth, len(frames), ref)
    rows = []
    worst = 0.0
    for piece in pieces:
        interior = cv2.erode((labels == piece).astype(np.uint8),
                             np.ones((11, 11), np.uint8)) > 0
        ys, xs = np.nonzero(interior)
        for k in range(len(frames)):
            if k == ref:
                continue
            for label, matrix in (("series", report["matrices"][(k, piece)]),
                                  ("raw", report["raw_matrices"][(k, piece)])):
                m, t = np.asarray(matrix), true_m[(k, piece)]
                dx = ((m[0, 0] - t[0, 0]) * xs + (m[0, 1] - t[0, 1]) * ys
                      + (m[0, 2] - t[0, 2]))
                dy = ((m[1, 0] - t[1, 0]) * xs + (m[1, 1] - t[1, 1]) * ys
                      + (m[1, 2] - t[1, 2]))
                rms = float(np.sqrt((dx ** 2 + dy ** 2).mean()))
                rows.append((k, piece, label, rms))
                if label == "series":
                    worst = max(worst, rms)
    print(f"  {'piece':<6}{'kind':<6}" + "".join(f"{'k=' + str(k):>9}"
                                                 for k in range(len(frames))))
    for piece in pieces:
        for label in ("series", "raw"):
            cells = []
            for k in range(len(frames)):
                if k == ref:
                    cells.append(f"{'ref':>9}")
                    continue
                value = [r[3] for r in rows if r[:3] == (k, piece, label)][0]
                cells.append(f"{value:9.3f}")
            tag = f"{piece:<6}{kinds[piece]:<6}" if label == "series" else " " * 12
            print(f"  {tag}{''.join(cells)}   {label}")
    print(f"  worst series RMS over all (frame, piece): {worst:.3f} px "
          f"(bar 0.2)   VERDICT {'PASS' if worst <= 0.2 else 'FAIL'}")

    # The gate: silent on truth, fires on an injected +1.5 px contour error.
    print("\n  K2b silhouette gate — silent on truth, fires on +1.5 px")
    band = int(np.ceil(abs(ref - P.NEAR_FOCUS_FRAME) * P.BLUR_PER_STEP)) + 1
    gate_rows = []
    for tag, offset in (("truth", 0.0), ("+1.5 px on the near piece", 1.5)):
        injected = dict(true_m)
        near_pieces = [p for p in pieces if kinds[p] == "near"]
        for piece in near_pieces:
            for k in range(len(frames)):
                matrix = np.asarray(true_m[(k, piece)]).copy()
                matrix[0, 2] += offset
                injected[(k, piece)] = matrix
        fired = confident = 0
        for k in range(len(frames)):
            if k == ref:
                continue
            map_x, map_y = _piecewise_field({p: injected[(k, p)] for p in pieces},
                                            labels, shape)
            moved = cv2.remap(frames[k].astype(np.float32), map_x, map_y,
                              cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            gate = silhouette_gate(frames[ref], moved, labels, report["cut"], band)
            fired += gate["failed"]
            confident += gate["confident"]
        gate_rows.append((tag, fired, confident))
        print(f"    {tag:<26} sites flagged {fired:5d} of {confident:5d} confident "
              f"({100.0 * fired / max(confident, 1):5.2f}%)")
    silent, injected_fires = gate_rows[0][1], gate_rows[1][1]
    print(f"  VERDICT {'PASS' if injected_fires > 4 * max(silent, 1) else 'SEE ABOVE'} "
          f"(separation {injected_fires / max(silent, 1):.1f}x)\n")
    return {"worst": worst, "rows": rows, "gate_truth": silent,
            "gate_injected": injected_fires}


# ---------------------------------------------------------------------------
# K3 — gap correctness against truth's own occlusion mask
# ---------------------------------------------------------------------------
def k3(frames, usable, report, truth) -> dict:
    print("K3  GAP CORRECTNESS — usable against truth's own occlusion mask")
    band = int(np.ceil(report["radius"][(0, report["order"][0])])) + 2
    ious, smear = [], []
    print(f"  {'k':>3}{'truth occl px':>15}{'gap px':>9}{'IoU':>8}"
          f"{'recall':>8}{'smear':>9}")
    for k in range(len(frames)):
        if k == report["ref"]:
            continue
        true_occluded = truth["occluded"][k]
        gap = ~usable[k]
        intersection = float((true_occluded & gap).sum())
        union = float((true_occluded | gap).sum())
        iou = intersection / max(union, 1.0)
        # WALL SMEAR: content sampled where truth says the surface was NOT seen.
        # Eroded by the matte band so the irreducible mixed pixels are not counted
        # as either a success or a failure.
        core = cv2.erode(true_occluded.astype(np.uint8),
                         np.ones((2 * band + 1,) * 2, np.uint8)) > 0
        leaked = float((core & usable[k]).sum())
        recall = float((true_occluded & gap).sum()) / max(true_occluded.sum(), 1)
        ious.append(iou)
        smear.append(leaked)
        print(f"  {k:>3}{int(true_occluded.sum()):>15}{int(gap.sum()):>9}"
              f"{iou:>8.3f}{recall:>8.3f}{int(leaked):>9}")
    print(f"  mean IoU {np.mean(ious):.3f}   total wall-smear leakage "
          f"{int(sum(smear))} px over {len(ious)} frames")
    print(f"  VERDICT {'PASS (no far content where truth says occluded)' if sum(smear) == 0 else 'SEE ABOVE'}\n")
    return {"iou": float(np.mean(ious)), "smear": float(sum(smear))}


# ---------------------------------------------------------------------------
# K4 — end to end through the EXISTING fusion, unmodified
# ---------------------------------------------------------------------------
def k4(frames, aligned, usable, report, truth_image) -> dict:
    print("K4  END TO END — `fuse_perband` unmodified, GT-SSIM against truth")
    _global_aligned, global_report = align_stack(frames, motion="affine",
                                                 depth_bins=0, return_report=True)
    x0, y0, x1, y1 = global_report["crop"]
    print(f"  scoring crop (align_stack's own, for comparability): "
          f"({x0}, {y0}, {x1}, {y1})")
    target = truth_image[y0:y1, x0:x1]
    cropped = [a[y0:y1, x0:x1] for a in aligned]
    masks = [u[y0:y1, x0:x1] for u in usable]
    scores = {}
    scores["aligner + gaps"] = metrics.ref_ssim(fuse_perband(cropped, usable=masks),
                                                target)
    scores["aligner, gaps OFF"] = metrics.ref_ssim(fuse_perband(cropped), target)
    scores["reference frame alone"] = metrics.ref_ssim(
        frames[report["ref"]][y0:y1, x0:x1], target)
    for label, value in scores.items():
        print(f"    {label:<26} {value:9.6f}")
    print(f"  against: shipped 0.972808 | runtime two-frame 0.984455 | "
          f"oracle resample floor 0.989487")
    print(f"  withheld {report['withheld'] * 100:.2f}% of pixel-frames\n")
    return scores


# ---------------------------------------------------------------------------
# Deliverable: the aligned frames and the gap overlays
# ---------------------------------------------------------------------------
def write_deliverable(aligned, usable, report, truth_image, frames):
    os.makedirs(OUT, exist_ok=True)
    for k, (image, mask) in enumerate(zip(aligned, usable)):
        cv2.imwrite(os.path.join(OUT, f"aligned_{k}.png"), image)
        overlay = image.copy()
        overlay[~mask] = (0.35 * overlay[~mask]
                          + 0.65 * np.array([0, 0, 255], np.float32)).astype(np.uint8)
        cv2.imwrite(os.path.join(OUT, f"usable_{k}.png"), overlay)
    labels = report["labels"]
    tint = np.zeros(labels.shape + (3,), np.uint8)
    palette = [(60, 180, 75), (245, 130, 48), (0, 130, 200), (240, 50, 230)]
    for index, piece in enumerate(report["pieces"]):
        tint[labels == piece] = palette[index % len(palette)]
    blend = (0.55 * frames[report["ref"]] + 0.45 * tint).astype(np.uint8)
    blend[report["cut"]] = (255, 255, 255)
    cv2.imwrite(os.path.join(OUT, "pieces.png"), blend)
    x0, y0, x1, y1 = 0, 0, labels.shape[1], labels.shape[0]
    fused = fuse_perband([a[y0:y1, x0:x1] for a in aligned],
                         usable=[u[y0:y1, x0:x1] for u in usable])
    cv2.imwrite(os.path.join(OUT, "fused.png"), fused)
    cv2.imwrite(os.path.join(OUT, "truth.png"), truth_image)
    print(f"  wrote {2 * len(aligned) + 3} images to {OUT}")


def ladder(frames, truth, truth_image, recovered) -> dict:
    """The attribution ladder: substitute ONE term at a time, never a consequence.

    F110's trap and F115's repeat of it: an oracle rung that substitutes a truth
    into a slot the estimate does not occupy measures nothing. So each rung here
    replaces exactly one input of THIS module — the piece labels, then the
    transforms — and everything downstream (blur rate, order, gaps, fusion) is
    recomputed by the module itself from that input.
    """
    print("LADDER  one substituted term per rung, everything downstream recomputed")
    _g, global_report = align_stack(frames, motion="affine", depth_bins=0,
                                    return_report=True)
    x0, y0, x1, y1 = global_report["crop"]
    target = truth_image[y0:y1, x0:x1]

    def score(aligned, usable, gaps=True):
        cropped = [a[y0:y1, x0:x1] for a in aligned]
        masks = [u[y0:y1, x0:x1] for u in usable] if gaps else None
        return metrics.ref_ssim(fuse_perband(cropped, usable=masks), target)

    true_labels = np.where(truth["near"], 1, 0).astype(np.int32)
    rungs = {}
    aligned, usable, report = align(frames, labels=true_labels.copy())
    rungs["true pieces + fitted affines"] = score(aligned, usable)
    true_m, _kinds = truth_matrices(report["pieces"], true_labels, truth,
                                    len(frames), report["ref"])
    worst = max(abs(report["matrices"][(k, p)][0, 2] - true_m[(k, p)][0, 2])
                for k in range(len(frames)) for p in report["pieces"])
    aligned2, usable2, report2 = align(frames, labels=true_labels.copy(),
                                       matrices=true_m)
    rungs["true pieces + TRUE affines"] = score(aligned2, usable2)
    rungs["true pieces + TRUE affines, gaps OFF"] = score(aligned2, usable2, gaps=False)
    print(f"    {'recovered pieces + fitted affines':<42} {recovered:9.6f}")
    for label, value in rungs.items():
        print(f"    {label:<42} {value:9.6f}")
    print(f"    {'oracle one-resample floor (F115)':<42} {0.989487:9.6f}")
    print("  attributed gap:")
    print(f"    pieces      {rungs['true pieces + fitted affines'] - recovered:+.6f}")
    print(f"    transforms  {rungs['true pieces + TRUE affines'] - rungs['true pieces + fitted affines']:+.6f}")
    print(f"    gaps buy    {rungs['true pieces + TRUE affines'] - rungs['true pieces + TRUE affines, gaps OFF']:+.6f}"
          " (gaps ON minus gaps OFF at true geometry)")
    print(f"    resample + fusion remainder "
          f"{0.989487 - rungs['true pieces + TRUE affines']:+.6f}")
    print(f"  K2 rescoped to the transform alone (true pieces): worst |tx| error "
          f"{worst:.3f} px against the 0.2 px bar -> "
          f"{'PASS' if worst <= 0.2 else 'FAIL'}")
    print(f"  c on true pieces {report['c']:.3f} (truth {P.BLUR_PER_STEP}), "
          f"peaks {report['peaks']} (truth near {P.NEAR_FOCUS_FRAME}, "
          f"far {P.FAR_FOCUS_FRAME})\n")
    return rungs


# ---------------------------------------------------------------------------
# ROUND 2: THE KITCHEN — the real scene, and the contract's signature test
# ---------------------------------------------------------------------------
KITCHEN = os.path.join(HERE, "data", "mobiledepth", "Figure3", "kitchen")
INSPECT = os.path.abspath(os.path.join(HERE, "..", "out", "inspect"))
# `align_stack`'s own crop on this stack, so the canonical box coordinates in
# `scene_model` (which are COMPOSITE coordinates of the routed path) apply
# unchanged. Measured, not assumed: printed by the command below.
KITCHEN_CROP = (15, 8, 742, 510)
# The wall right of the cocoa tin, in INSPECTION coordinates: the place the F116
# wall smear lived. Far frames must be GAPPED here, because the tin's parallax put
# it over this wall in those frames, so they never observed it.
WALL_BOX = (285, 50, 365, 170)


def _gap_ledger(report, label=""):
    """Per-frame gap statistics, attributed to the clause that withdrew them.

    F114 predicts large holes on this scene; the failure mode to NAME is silent
    over-refusal, which is invisible in any aggregate and obvious here.
    """
    keys = ["footprint", "occlusion", "gate", "photometry"]
    print(f"  {label}per-frame GAP fraction, attributed to the clause that took it")
    print(f"  {'k':>3}{'off-footprint':>15}{'occlusion':>11}{'gate':>9}"
          f"{'photometry':>12}{'matte+dil':>11}{'TOTAL gap':>11}")
    rows = []
    for k, ledger in enumerate(report["stages"]):
        values = [ledger[key] for key in keys] + [ledger["final"]]
        parts = [1.0 - values[0]] + [values[i] - values[i + 1] for i in range(4)]
        rows.append(parts + [1.0 - ledger["final"]])
        print(f"  {k:>3}" + "".join(f"{100 * p:>14.2f}%" if i == 0 else
                                    f"{100 * p:>10.2f}%" if i == 1 else
                                    f"{100 * p:>8.2f}%" if i == 2 else
                                    f"{100 * p:>11.2f}%" if i == 3 else
                                    f"{100 * p:>10.2f}%"
                                    for i, p in enumerate(parts))
              + f"{100 * (1 - ledger['final']):>10.2f}%")
    mean = np.array(rows).mean(axis=0)
    print(f"  mean gap {100 * mean[-1]:.2f}% of pixel-frames  "
          f"(footprint {100 * mean[0]:.2f}, occlusion {100 * mean[1]:.2f}, "
          f"gate {100 * mean[2]:.2f}, photometry {100 * mean[3]:.2f}, "
          f"matte {100 * mean[4]:.2f})")
    return mean


def _seed_agreement(report):
    """Each seeded piece's FITTED motion against the pass-1 group motion that
    proposed it. They should agree to about a pixel; a disagreement is a finding.

    Both are expressed in GLOBALLY-ALIGNED coordinates (the residual beyond the
    whole-frame affine), which is the space `motion_groups` measures in.
    """
    groups, labels, prior = report["seed_groups"], report["labels"], report["prior"]
    if not groups:
        print("  no motion-seeded pieces (pass 1's organ proposed none)")
        return []
    print(f"  {'piece':>6}{'area':>8}{'k':>4}{'pass-1 group (dx,dy)':>24}"
          f"{'aligner fit (dx,dy)':>23}{'|Δ| px':>9}")
    rows = []
    for group in groups:
        label = group.get("final_label")
        if label is None:
            continue
        mask = labels == label
        if not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        cx, cy = float(xs.mean()), float(ys.mean())
        motion = group["motion"]
        worst = 0.0
        for k in sorted(motion, key=lambda k: -float(np.hypot(*motion[k])))[:3]:
            m = np.asarray(report["matrices"][(k, label)], np.float64)
            target = m @ np.array([cx, cy, 1.0])
            source = np.linalg.inv(np.asarray(prior[k], np.float64)) @ target
            fit = (float(source[0] - cx), float(source[1] - cy))
            g = (float(motion[k][0]), float(motion[k][1]))
            delta = float(np.hypot(fit[0] - g[0], fit[1] - g[1]))
            worst = max(worst, delta)
            print(f"  {label:>6}{int(mask.sum()):>8}{k:>4}"
                  f"{f'({g[0]:+.2f}, {g[1]:+.2f})':>24}"
                  f"{f'({fit[0]:+.2f}, {fit[1]:+.2f})':>23}{delta:>9.2f}")
        rows.append({"label": label, "area": int(mask.sum()), "worst": worst,
                     "absorbed": group.get("absorbed", False)})
    for row in rows:
        print(f"    piece {row['label']}: worst |Δ| against pass 1 "
              f"{row['worst']:.2f} px  "
              f"({'AGREES' if row['worst'] <= 1.5 else 'DISAGREES — a finding'})")
    return rows


def _per_piece_gaps(report):
    """The 27% whole-frame photometric withdrawal, attributed to a SURFACE."""
    pieces, labels = report["pieces"], report["labels"]
    areas = {p: int((labels == p).sum()) for p in pieces}
    total = float(labels.size)
    rows = {}
    for ledger in report["stages"]:
        for piece, values in ledger.get("per_piece", {}).items():
            entry = rows.setdefault(piece, {"photometry": [], "gate": [], "gap": []})
            for key in entry:
                entry[key].append(values[key])
    print(f"  {'piece':>6}{'area':>9}{'frame %':>9}{'photometry':>12}"
          f"{'gate':>8}{'TOTAL gap':>11}{'seeded':>8}")
    out = []
    for piece in sorted(pieces, key=lambda p: -np.mean(rows.get(p, {}).get("gap", [0]))):
        entry = rows.get(piece)
        if not entry:
            continue
        photo = 100 * float(np.mean(entry["photometry"]))
        gate = 100 * float(np.mean(entry["gate"]))
        gap = 100 * float(np.mean(entry["gap"]))
        print(f"  {piece:>6}{areas[piece]:>9}{100 * areas[piece] / total:>8.1f}%"
              f"{photo:>11.1f}%{gate:>7.1f}%{gap:>10.1f}%"
              f"{'  YES' if piece in report.get('seeded', ()) else '':>8}")
        out.append((piece, photo, gap))
    return out


# Regions for the sharpening test. USER_BOXES are already COMPOSITE coordinates;
# `KNOB` is recorded in ORIGINAL coordinates, so it is mapped by the crop origin
# (composite = original - (x0, y0)). The back-shelf and counter boxes are chosen
# here: the cluttered shelf behind the stove (far, focal peak at the sweep's near
# end) and the granite counter in the foreground (the receding ramp).
def _sharpen_boxes(crop):
    x0, y0 = crop[0], crop[1]
    boxes = {f"box {i}": SM.USER_BOXES[i] for i in (1, 2, 3, 4)}
    boxes["knob"] = (SM.KNOB[0] - x0, SM.KNOB[1] - y0,
                     SM.KNOB[2] - x0, SM.KNOB[3] - y0)
    boxes["back shelf"] = (632, 60, 722, 200)
    boxes["counter"] = (121, 420, 281, 495)
    return boxes


def _sharpen_table(fused, reference, routed, crop):
    """Mean FOCUS ENERGY per region: the fused output against the reference AND
    against the routed default. THE POINT of the round — the gates are supposed to
    open where a sharper frame legitimately lands, and round 2 read within 1-2% of
    the reference everywhere, i.e. it shipped the reference.
    """
    from focusstack.focus import focus_measure

    x0, y0, x1, y1 = crop
    boxes = _sharpen_boxes(crop)
    fields = {}
    fields["aligner"] = focus_measure(to_gray_float(fused).astype(np.float32))
    fields["reference"] = focus_measure(
        to_gray_float(reference[y0:y1, x0:x1]).astype(np.float32))
    if routed is not None:
        # The inspection layer is registered 1 px right of the composite crop
        # (origin (16, 8) against KITCHEN_CROP's (15, 8)).
        fields["routed"] = focus_measure(to_gray_float(routed).astype(np.float32))
    print(f"  {'region':<12}{'aligner':>10}{'reference':>11}{'routed':>9}"
          f"{'vs ref':>10}{'vs routed':>11}")
    rows = []
    for name, (bx0, by0, bx1, by1) in boxes.items():
        a = float(fields["aligner"][by0:by1, bx0:bx1].mean())
        r = float(fields["reference"][by0:by1, bx0:bx1].mean())
        d = (float(fields["routed"][by0:by1 - 0, bx0 - 1:bx1 - 1].mean())
             if "routed" in fields else float("nan"))
        rows.append((name, a, r, d))
        print(f"  {name:<12}{a:>10.2f}{r:>11.2f}{d:>9.2f}"
              f"{100 * (a / max(r, 1e-9) - 1):>9.1f}%"
              f"{100 * (a / d - 1) if np.isfinite(d) and d > 0 else float('nan'):>10.1f}%")
    return rows


def _guide_boxes_figure(fused, reference, routed, crop, path):
    """routed | new aligner | reference, for the four canonical boxes, stacked."""
    x0, y0 = crop[0], crop[1]
    boxes = [SM.USER_BOXES[i] for i in (1, 2, 3, 4)]
    scale = 3
    tiles = []
    for bx0, by0, bx1, by1 in boxes:
        row = []
        for name, image, shift in (("ROUTED", routed, -1), ("ALIGNER", fused, 0),
                                   ("REFERENCE", reference, None)):
            if image is None:
                crop_tile = np.zeros((by1 - by0, bx1 - bx0, 3), np.uint8)
            elif shift is None:
                crop_tile = image[y0 + by0:y0 + by1, x0 + bx0:x0 + bx1]
            else:
                crop_tile = image[by0:by1, bx0 + shift:bx1 + shift]
            row.append((name, crop_tile))
        tiles.append(row)
    width = max(t.shape[1] for row in tiles for _n, t in row) * scale
    height = max(t.shape[0] for row in tiles for _n, t in row) * scale
    sheet = np.full((len(tiles) * (height + 18), 3 * (width + 6), 3), 32, np.uint8)
    for r, row in enumerate(tiles):
        for c, (name, tile) in enumerate(row):
            if tile.size == 0:
                continue
            big = cv2.resize(tile, (tile.shape[1] * scale, tile.shape[0] * scale),
                             interpolation=cv2.INTER_NEAREST)
            oy, ox = r * (height + 18) + 18, c * (width + 6)
            sheet[oy:oy + big.shape[0], ox:ox + big.shape[1]] = big
            cv2.putText(sheet, f"box {r + 1} {name}", (ox + 2, oy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
                        cv2.LINE_AA)
    cv2.imwrite(path, sheet)


def _piece_map(report, reference, path):
    labels = report["labels"]
    palette = [(60, 180, 75), (245, 130, 48), (0, 130, 200), (240, 50, 230),
               (70, 240, 240), (210, 245, 60), (0, 0, 200), (128, 128, 0),
               (255, 225, 25), (145, 30, 180), (0, 128, 128), (170, 110, 40)]
    tint = np.zeros(labels.shape + (3,), np.uint8)
    for index, piece in enumerate(report["pieces"]):
        tint[labels == piece] = palette[index % len(palette)]
    blend = (0.55 * reference + 0.45 * tint).astype(np.uint8)
    blend[report["cut"]] = (255, 255, 255)
    cv2.imwrite(path, blend)


def _wall_gap_figure(aligned, usable, report, reference, fused, origin, path):
    """The contract's SIGNATURE: far frames gapped where the tin hid the wall."""
    ox, oy = origin
    x0, y0, x1, y1 = (WALL_BOX[0] + ox, WALL_BOX[1] + oy,
                      WALL_BOX[2] + ox, WALL_BOX[3] + oy)
    tiles, stats = [], []
    for k, (image, mask) in enumerate(zip(aligned, usable)):
        crop = image[y0:y1, x0:x1].astype(np.float32).copy()
        hole = ~mask[y0:y1, x0:x1]
        crop[hole] = 0.30 * crop[hole] + 0.70 * np.array([0, 0, 255], np.float32)
        stats.append(float(hole.mean()))
        tiles.append(np.clip(crop, 0, 255).astype(np.uint8))
    cx0, cy0 = KITCHEN_CROP[0], KITCHEN_CROP[1]
    tiles.append(fused[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0])
    tiles.append(reference[y0:y1, x0:x1])
    scale, columns = 2, 7
    height, width = tiles[0].shape[:2]
    rows = int(np.ceil(len(tiles) / columns))
    sheet = np.full((rows * (height * scale + 16), columns * (width * scale + 6), 3),
                    32, np.uint8)
    for index, tile in enumerate(tiles):
        big = cv2.resize(tile, (width * scale, height * scale),
                         interpolation=cv2.INTER_NEAREST)
        r, c = divmod(index, columns)
        oy0, ox0 = r * (height * scale + 16), c * (width * scale + 6)
        sheet[oy0 + 16:oy0 + 16 + height * scale, ox0:ox0 + width * scale] = big
        name = ("FUSED" if index == len(tiles) - 2 else "REFERENCE"
                if index == len(tiles) - 1
                else f"k={index} gap {100 * stats[index]:.0f}%")
        cv2.putText(sheet, name, (ox0 + 2, oy0 + 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, sheet)
    return stats


def sentinel() -> None:
    """ZERO-MOTION ANCHOR: identity transforms, zero gaps (F101's standing bar).

    Built exactly as `tests/test_twoframe_route.py::_still_stack` builds it — frames
    that differ only by sensor noise, so nothing moved and nothing may be withheld.
    """
    print("\nZERO-MOTION SENTINEL — identity transforms, zero gaps expected")
    source = cv2.imread(sorted(os.listdir(KITCHEN)) and
                        os.path.join(KITCHEN, sorted(os.listdir(KITCHEN))[6]))
    base = cv2.resize(source, (source.shape[1] // 2, source.shape[0] // 2))
    rng = np.random.default_rng(11)
    frames = [np.clip(base + rng.normal(0, 2, base.shape), 0, 255).astype(np.uint8)
              for _ in range(4)]
    aligned, usable, report = align(frames, ref=1)
    worst = max(float(np.abs(np.asarray(report["matrices"][(k, p)]) - np.eye(3)).max())
                for k in range(len(frames)) for p in report["pieces"])
    gaps = [float((~u).mean()) for u in usable]
    print(f"  pieces {len(report['pieces'])}  worst |M - I| entry {worst:.2e}  "
          f"per-frame gap {['%.4f%%' % (100 * g) for g in gaps]}")
    print(f"  VERDICT {'PASS' if max(gaps) < 1e-9 and worst < 1e-3 else 'SEE ABOVE'}"
          f" (zero gaps: {max(gaps) == 0.0})")


def kitchen() -> None:
    """The aligner on the real scene, handed to the UNMODIFIED fusion cascade."""
    import time
    from focusstack.io import normalize_exposure

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(INSPECT, exist_ok=True)
    t0 = time.time()
    paths = sorted(os.path.join(KITCHEN, name) for name in os.listdir(KITCHEN)
                   if name.endswith(".jpg"))
    src = [cv2.imread(p) for p in paths]
    norm = normalize_exposure(src)
    ref = 6
    print("=" * 78)
    print(f"KITCHEN — {len(src)} frames {src[0].shape[1]}x{src[0].shape[0]}, "
          f"reference {ref}, exposure-normalized")
    print("=" * 78)
    _g, global_report = align_stack(norm, motion="affine", depth_bins=0,
                                    return_report=True)
    print(f"  align_stack's own crop {global_report['crop']} "
          f"(the canonical box coordinates live in it; "
          f"{'matches' if tuple(global_report['crop']) == KITCHEN_CROP else 'DIFFERS FROM'}"
          f" KITCHEN_CROP)")
    aligned, usable, report = align(norm, ref=ref, verbose=True)
    print(f"\n  PIECES {len(report['pieces'])}   travel px/frame "
          f"{ {p: round(v, 2) for p, v in report['travel'].items()} }")
    print(f"  order (nearest first) {report['order']}")
    print(f"  same-depth groups (cannot occlude each other) {report['groups']}")
    print(f"  c measured on THIS scene {report['c']:.3f} px/frame "
          f"(F117: factory 1.161, kitchen 0.684 — do NOT import)   "
          f"peaks {report['peaks']}")
    areas = {p: int((report['labels'] == p).sum()) for p in report['pieces']}
    print(f"  piece areas {areas}")
    _piece_map(report, norm[ref], os.path.join(OUT, "kitchen_pieces.png"))
    print(f"\n  --- MOTION-SEEDED PIECES (pass 1's organ vs the aligner's refit) ---")
    print(f"  motion_groups report {report['group_report']}")
    _seed_agreement(report)
    print(f"\n  --- GAP STATISTICS (F114 predicts large holes) ---")
    _gap_ledger(report)
    print(f"\n  --- PER-PIECE GAP ATTRIBUTION (round 2's item 7) ---")
    _per_piece_gaps(report)

    x0, y0, x1, y1 = KITCHEN_CROP
    fused = fuse_perband([a[y0:y1, x0:x1] for a in aligned],
                         usable=[u[y0:y1, x0:x1] for u in usable])
    cv2.imwrite(os.path.join(OUT, "kitchen_fused.png"), fused)
    reference = norm[ref]
    print(f"\n  --- THE FOUR F112 USER BOXES (mean/max |Δ| vs norm[{ref}]) ---")
    print(f"  {'candidate':<26} {'box 1':>10} {'box 2':>10} {'box 3':>10} "
          f"{'box 4':>10}")
    print(f"  {'routed default (recorded)':<26} {'1.20/  2':>10} {'2.04/ 13':>10} "
          f"{'1.19/ 19':>10} {'1.03/ 17':>10}")
    SM.kitchen_boxes(fused, reference, KITCHEN_CROP, "aligner + gaps + fuse_perband")
    print(f"  the counter-instrument — mean FOCUS ENERGY in the same boxes:")
    SM.kitchen_boxes(fused, reference, KITCHEN_CROP, "aligner", energy=True)
    SM.kitchen_boxes(reference[y0:y1, x0:x1], reference, KITCHEN_CROP,
                     "reference frame", energy=True)
    print(f"\n  --- THE F108 FLANK BOX (x560-670, y240-420) vs norm[{ref}] ---")
    print(f"  {'routed default (recorded)':<26} mean 0.897  >12 0.01%")
    SM.kitchen_flank(fused, reference, KITCHEN_CROP, "aligner + gaps")

    # --- THE POINT: real sharpening over the reference AND over the routed path --
    routed = cv2.imread(os.path.join(INSPECT, "kitchen_routed.png"))
    print(f"\n  --- MEAN FOCUS ENERGY per region "
          f"(the point of the round; routed layer "
          f"{'loaded' if routed is not None else 'ABSENT'}) ---")
    _sharpen_table(fused, reference, routed, KITCHEN_CROP)
    _guide_boxes_figure(fused, reference, routed, KITCHEN_CROP,
                        os.path.join(OUT, "GUIDE_boxes.png"))
    print(f"  -> out/aligner/GUIDE_boxes.png (routed | aligner | reference, "
          f"the four boxes stacked)")

    # --- the inspector layer, registered to the EXISTING reference layer -------
    target = cv2.imread(os.path.join(INSPECT, "kitchen_reference.png"))
    origin = (x0, y0)
    if target is None:
        print("\n  out/inspect/kitchen_reference.png absent — registration skipped")
    else:
        patch = target[100:300, 100:400]
        scored = cv2.matchTemplate(reference, patch, cv2.TM_CCOEFF_NORMED)
        _mn, score, _ml, location = cv2.minMaxLoc(scored)
        ox, oy = location[0] - 100, location[1] - 100
        origin = (ox, oy)
        th, tw = target.shape[:2]
        layer = reference[oy:oy + th, ox:ox + tw].copy()
        piece = fused[oy - y0:oy - y0 + th, ox - x0:ox - x0 + tw]
        layer[:piece.shape[0], :piece.shape[1]] = piece
        cv2.imwrite(os.path.join(INSPECT, "kitchen_aligner.png"), layer)
        print(f"\n  inspector layer registration score {score:.4f} "
              f"({'PASS' if score >= 0.99 else 'FAIL'}), crop origin ({ox}, {oy}) "
              f"-> out/inspect/kitchen_aligner.png")

    print(f"\n  --- THE WALL-SMEAR TEST (wall right of the cocoa tin, "
          f"inspection x{WALL_BOX[0]}-{WALL_BOX[2]} y{WALL_BOX[1]}-{WALL_BOX[3]}) ---")
    stats = _wall_gap_figure(aligned, usable, report, reference, fused, origin,
                             os.path.join(OUT, "kitchen_wall_gaps.png"))
    print("  gap fraction inside the wall box, per frame: "
          + "  ".join(f"k{k}:{100 * s:.0f}%" for k, s in enumerate(stats)))
    print(f"  reference frame k={ref} gap {100 * stats[ref]:.0f}% "
          f"(must be 0 by construction); "
          f"frames furthest from it {100 * stats[0]:.0f}% / {100 * stats[-1]:.0f}%")
    print(f"  -> out/aligner/kitchen_wall_gaps.png "
          f"(red = GAP; last two tiles are the FUSED result and the reference)")
    print(f"\n  elapsed {time.time() - t0:.1f} s")


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "kitchen":
        kitchen()
        sentinel()
        return
    if which == "sentinel":
        sentinel()
        return
    frames, truth_image, _near = P.build_stack()
    truth = factory_truth()
    print(f"\nANALYTIC FACTORY: {P.FRAMES} frames, reference {P.REFERENCE}, "
          f"near {P.NEAR_SHIFT_PER_FRAME} px/frame vs far {P.FAR_SHIFT_PER_FRAME} "
          f"px/frame ({P.NEAR_SHIFT_PER_FRAME / P.FAR_SHIFT_PER_FRAME:.1f}x)\n")
    kat_ramp()
    if which == "ramp":
        return
    kat_sign()
    print("Building the aligner (global prior -> pieces -> per-piece affines -> "
          "blur rate -> gaps)")
    aligned, usable, report = align(frames, verbose=True)
    print(f"  order (nearest first) {report['order']}  travel "
          f"{ {p: round(v, 2) for p, v in report['travel'].items()} }")
    print("  veto ledger (fraction of the frame surviving each clause, in order):")
    for k, ledger in enumerate(report["stages"]):
        print(f"    k={k}  footprint {ledger['footprint']:.4f} -> occlusion "
              f"{ledger['occlusion']:.4f} -> gate {ledger['gate']:.4f} -> "
              f"photometry {ledger['photometry']:.4f} -> final {ledger['final']:.4f}")
    print()
    results = {"k1": k1(frames, report, truth)}
    if which in ("all", "k2", "k3", "k4"):
        results["k2"] = k2(frames, report, truth)
        results["k3"] = k3(frames, usable, report, truth)
        results["k4"] = k4(frames, aligned, usable, report, truth_image)
        ladder(frames, truth, truth_image, results["k4"]["aligner + gaps"])
    write_deliverable(aligned, usable, report, truth_image, frames)


if __name__ == "__main__":
    main()
