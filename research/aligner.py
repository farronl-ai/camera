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
# the global stage already fitted. Two, not six, and this is physics rather than
# thrift: rotation and magnification (the camera's breathing) are DEPTH-INDEPENDENT
# — exactly the property that makes breathing separable from parallax and lets ONE
# global affine carry both — while PARALLAX is a translation whose magnitude is
# linear in inverse depth. So a piece's residual relative to the global affine is,
# to first order, a pure TRANSLATION: two unknowns against thousands of
# normal-profile constraints, which stays determined on a small piece where six
# unknowns do not (§12.4/§12.6). Round 1 fitted six per piece and paid for it in the
# merge: over-segmented pieces of ONE surface disagreed by 1.75-53 px purely from
# the linear part's extrapolation noise, so motion agreement was unmeasurable and
# the merge could not fire at all. The bar this must clear is unchanged (K2, 0.2 px).
PIECE_DOF = 2
# ...and the model order of the FINAL transform, once the merge has produced pieces
# large enough to determine it. Round 1's K2 bar (0.159 px on true pieces) is a
# 6-DoF number and stays one.
FINAL_DOF = 6


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
        return matrix, {"sites": 0, "residual": float("nan"), "ok": False}
    cx, cy = (shape[1] - 1) / 2.0, (shape[0] - 1) / 2.0
    history = []
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
        step = np.eye(3)
        if dof == 2:
            step[0, 2], step[1, 2] = solution[0], solution[1]
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
                    "trace": history, "ok": bool(history)}


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
                out[(k, piece)] = np.asarray(prior[k], float)
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


def _support_sample(labels, members, rng):
    ys, xs = np.nonzero(np.isin(labels, list(members)))
    if len(xs) > MERGE_SAMPLE:
        pick = rng.choice(len(xs), MERGE_SAMPLE, replace=False)
        ys, xs = ys[pick], xs[pick]
    return xs.astype(np.float64), ys.astype(np.float64)


def merge_agreeing(labels, pieces, series, slopes, n, tol=None, verbose=False):
    """Merge adjacent pieces whose fitted motion AGREES. Returns (labels, info).

    Two clauses, and the difference between them is whether a measurement exists:

    * AGREEMENT — both groups measured their own motion, and one affine explains
      both supports to within `GATE_TOL` in every frame. That is the definition of
      one surface, and it is a measurement, not a threshold on a proxy field.
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
    tol = GATE_TOL if tol is None else tol
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

    borders = _adjacency(labels, pieces)
    unverifiable = {p for p in pieces if slopes.get(p) is None}
    # --- clause 1: agreement, cheapest candidate first ------------------------
    candidates = []
    for (a, b), length in borders.items():
        if a in unverifiable or b in unverifiable:
            continue
        xs, ys = _support_sample(labels, (a, b), rng)
        gap = max(_affine_gap(series[(k, a)], series[(k, b)], xs, ys)
                  for k in range(n))
        candidates.append((gap, length, a, b))
    candidates.sort()
    agreed = kept = 0
    for gap, _length, a, b in candidates:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        xs, ys = _support_sample(labels, members[ra] + members[rb], rng)
        group_gap = max(_affine_gap(matrices[ra][k], matrices[rb][k], xs, ys)
                        for k in range(n))
        if group_gap <= tol:
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
    for index, root in enumerate(sorted({find(p) for p in pieces})):
        out[np.isin(labels, members[root])] = index
    stranded = out < 0
    if stranded.any():                      # sub-MIN_PIECE residue: nearest piece
        out[stranded] = 0 if not (~stranded).any() else out[~stranded][
            cv2.distanceTransformWithLabels(stranded.astype(np.uint8) * 255,
                                            cv2.DIST_L2, 3,
                                            labelType=cv2.DIST_LABEL_PIXEL)[1][stranded] - 1]
    info = {"merges": agreed + adopted, "agreed": agreed, "adopted": adopted,
            "held": kept, "before": len(pieces),
            "after": len({find(p) for p in pieces}),
            "gaps": sorted(round(c[0], 3) for c in candidates)}
    if verbose:
        print(f"    merge: {info['before']} -> {info['after']} pieces "
              f"({agreed} agreed within {tol} px, {adopted} unmeasurable adopted, "
              f"{kept} cuts HELD by motion disagreement)")
        if candidates:
            print(f"      pairwise disagreement px: "
                  f"{', '.join(f'{g:.2f}' for g, *_ in candidates[:6])}"
                  f"{' ...' if len(candidates) > 6 else ''}  "
                  f"max {max(c[0] for c in candidates):.2f}")
    return out, info


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
    def fit_pieces(labels, pieces, cut, dof=None):
        dof = PIECE_DOF if dof is None else dof
        fitted, counts, fits = {}, {}, {}
        band_exclude = _dilate(cut, SM.CONTOUR_HALF + SM.CONTOUR_SPAN)
        for piece in pieces:
            sites = _material_sites(ref_gray, labels == piece, band_exclude)
            for k in range(n):
                if k == ref:
                    fitted[(k, piece)] = np.eye(3)
                    counts[(k, piece)] = 0
                    continue
                matrix, info = fit_affine(ref_gray, frames[k].astype(np.float32),
                                          prior[k], sites, shape, cache=cache,
                                          dof=dof)
                # GRADUATED MODEL ORDER, §12.4 read forwards rather than as a veto:
                # a piece that cannot overdetermine six unknowns may still
                # overdetermine two, and the 2-DoF model is the physics of parallax
                # (a depth-dependent translation on top of the global affine). Only
                # a piece that cannot determine even that is UNVERIFIABLE, and then
                # it keeps the global prior per F106 — it does NOT get the identity.
                if dof > 2 and (not info["ok"] or info["sites"] < 4 * dof):
                    matrix, info = fit_affine(ref_gray, frames[k].astype(np.float32),
                                              prior[k], sites, shape, cache=cache,
                                              dof=2)
                    if info["ok"] and info["sites"] >= 8:
                        info = dict(info, dof=2)
                if not info["ok"] or info["sites"] < 4 * min(dof, info.get("dof", dof)):
                    fitted[(k, piece)] = np.asarray(prior[k], float)
                    counts[(k, piece)] = 0
                else:
                    fitted[(k, piece)] = matrix
                    counts[(k, piece)] = info["sites"]
                fits[(k, piece)] = info
        return fitted, counts, fits

    merge_log = []
    if matrices is None:
        cut = pdiag["cut"]
        for _round in range(MERGE_MAX_ROUNDS):
            fitted, counts, fits = fit_pieces(labels, pieces, cut)
            series, slopes = motion_series(fitted, counts, pieces, n, ref, prior)
            if len(pieces) < 2:
                break
            merged, info = merge_agreeing(labels, pieces, series, slopes, n,
                                          verbose=verbose)
            merge_log.append(info)
            if info["merges"] == 0:
                break
            after = piece_table(merged)
            # Monotone by construction (a merge only unites); ASSERTED because a
            # fixed-point loop that can grow is a loop that can spin.
            assert 0 < len(after) < len(pieces), (len(after), len(pieces))
            labels, pieces = merged, after
            cut = _label_boundary(labels)
        # The merge is decided; now the TRANSFORM. The two want different model
        # orders and the reason is the PRIOR: the whole-frame affine's linear part
        # is a compromise across depths (it absorbs some of the differential
        # parallax as shear/scale), so a piece can only shed it with six degrees of
        # freedom — measured, not assumed: on TRUE pieces the 6-DoF refit reads
        # 0.159 px worst against 3.570 px for the 2-DoF fit, because translation
        # alone cannot undo a wrong linear part away from the sites' centroid. Six
        # DoF is safe HERE and was not safe during the merge, because the merge ran
        # on small over-segmented pieces and this runs on the merged ones.
        fitted, counts, fits = fit_pieces(labels, pieces, cut, dof=FINAL_DOF)
        series, slopes = motion_series(fitted, counts, pieces, n, ref, prior)
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
    travel = {}
    for piece in pieces:
        d = [np.hypot(series[(k, piece)][0, 2], series[(k, piece)][1, 2])
             for k in range(n)]
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
    print(f"\n  --- GAP STATISTICS (F114 predicts large holes) ---")
    _gap_ledger(report)

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
