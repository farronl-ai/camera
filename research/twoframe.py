"""Per-region TWO-FRAME architecture — prototype of the user's F86-era proposal.

The proposal, in one line: for each region of the scene pick the frame where that
region's FOREGROUND is sharpest and the frame where its BACKGROUND is sharpest,
align that pair, fuse the pair, and stitch the per-region results into one output.

Why it might close F108's zone-coverage gap, stated as a mechanism before any
measurement:

  * Every gate the refusal net owns is LOCAL-EVIDENCE driven, and F108 proved all
    of them blind in a smooth low-contrast stretch (no edges -> no evidence;
    guided depth ramps -> both depth gates fail). A pair choice needs no
    per-pixel evidence at all: it comes from a REGION's focal statistics, which
    are pooled over ~10^4 pixels and survive where any one pixel votes nothing.
  * The streak's proven mechanism is the MIXING of near-reference frames whose
    edges are fanned by an uncorrected ±3-4 px. A two-frame composite cannot fan:
    there are two sources, and in the streak zone exactly one of them is sharp.
  * Inside a pair the geometry problem is ONE relative displacement per layer,
    not N. And — the part that makes it more than a convenience — each frame of a
    pair is responsible for exactly ONE depth layer, so it can be warped by ONE
    RIGID translation. The layer it gets wrong is, by construction, the layer it
    is defocused in, so the focus contest discards it. Geometry and focus are
    co-diagnostic here; no depth-dependent field is needed inside a region, and
    with no field there is no field discontinuity, no stretch limiter, and no
    soft geometric blend (F106).

How the hard-won rules bind this design, and where each shows up in the code:

  * "Resample pixels exactly ONCE" (PLAYBOOK §0). Analysis runs on the globally
    aligned frames; RENDERING composes the global affine and the region's rigid
    translation into one homogeneous matrix and remaps the ORIGINAL frame once.
  * "A geometric decision cannot be soft" (F106). Each candidate is rendered from
    two rigid warps. The only soft thing in the pipeline is the STITCH, which
    blends finished IMAGES (photometric) through `multiband_blend`, never
    sampling fields.
  * "The stitch is the risk" (F79). Ownership is one-hot per pixel and the
    reconstruction is multiband — the exact construction `fuse_coherent` uses,
    not a pixel-space paste.
  * "Never key a correction on depth" (F99). The layer masks that support each
    ECC fit are keyed on WHICH FRAME OF THE PAIR WINS THE FOCUS CONTEST — the
    operational selector, i.e. exactly the pixels that frame will supply — not on
    a depth value. And the fit is region-local, so the F99 failure (frame-wide
    depth-keying averages the target away against other content sharing its depth
    VALUE) has no room to occur: the averaging pool is one tile.
  * "Measure where the evidence is" (F89, DEVSTYLE §12.5). Free here: the frame
    chosen for a layer is the frame that layer is SHARPEST in, so every fit runs
    on the best-conditioned observation that exists.
  * "Known-answer-test every new instrument" (§12.1). `kat()` does this for the
    two new instruments — the coarse-to-fine masked translation estimator and the
    pair chooser — before either is believed on real data.

Run:

    .venv/bin/python research/twoframe.py kat        # instrument known-answer tests
    .venv/bin/python research/twoframe.py factory    # analytic GT factory
    .venv/bin/python research/twoframe.py oracle     # the architecture's ceiling
    .venv/bin/python research/twoframe.py variants   # the two free choices, A/B'd
    .venv/bin/python research/twoframe.py kitchen    # kitchen sweep + F108 crops

Findings and the rejected list: `research/twoframe_NOTES.md`.
"""
from __future__ import annotations

import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import focusstack.align as A  # noqa: E402
from focusstack.focus import content_aware_energies  # noqa: E402
from focusstack.fusion import (  # noqa: E402
    depth_from_focus,
    fuse_perband,
    multiband_blend,
)
from focusstack.io import to_gray_float  # noqa: E402
import focusstack.motion_groups as MG  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
KITCHEN = os.path.join(HERE, "data", "mobiledepth", "Figure3", "kitchen")
OUT = os.path.join(HERE, "..", "out", "depth_align")

# Region grid. A "region" here is deliberately NOT an object: F98 established
# that turning feature groups into pixel regions is the open problem and that
# whatever fills a region mask in is doing the real work while looking like
# plumbing. A tile claims nothing — it is only a locality over which focal
# statistics are pooled — so no segmentation is smuggled in. Tiles overlap so a
# structure straddling a tile edge votes in both.
TILE = 96
STRIDE = 48

# A tile is called two-layer only if splitting its focal-peak distribution
# explains this fraction of its variance (Otsu, never a gap — PLAYBOOK §0), both
# sides carry real weight, and the modes sit at least MIN_SEPARATION frames
# apart. Below that the tile is one depth and takes ONE frame.
OTSU_MIN_QUALITY = 0.45
MIN_SIDE_WEIGHT = 0.12
MIN_SEPARATION = 1.5

# Distinct pairs kept for the whole frame. Every extra pair is another full
# two-frame render; more importantly, a pair boundary is a stitch seam.
MAX_PAIRS = 6
MERGE_TOL = 1.0

# The layer fit is allowed a much larger shift than the shipped per-bin cap
# (_REFINE_MAX_FRACTION = 1.5% of the diagonal = 14 px on the kitchen), because
# the kitchen bottle alone needs ~19 px and that cap is one of the reasons the
# bins never even propose its correction.
MAX_SHIFT_FRACTION = 0.045
MIN_LAYER_PIXELS = 250
PYRAMID_LEVELS = 3

# --- hardening constants (see the dated section of twoframe_NOTES.md) --------
# Focal weighting is MANDATORY, not a refinement: a blurred profile correlates
# CONFIDENTLY against a sharp one at about zero shift, so match confidence cannot
# detect defocus bias and an unweighted fit reports that a layer stopped moving
# exactly where it left focus (F99). 2.5 frames is group_align's validated sigma.
FOCAL_SIGMA = 2.5
MIN_EDGE_FEATURES = 6        # features a focal-weighted edge fit needs
MIN_EDGE_WEIGHT = 2.5        # total focal-weighted support it needs
EDGE_MAX_STEP = 6.0          # px the edge stage may move the coarse ECC estimate

# The validity gate. A fit is applied only if, AFTER applying it, the layer's own
# material edges say the layer has stopped moving. GATE_TOL is in px of residual
# translation; GATE_MIN_OBSERVABLE is the share of the focal-weighted support an
# axis needs before the gate is allowed to judge that axis at all (F103: a
# feature whose normal is perpendicular to a motion agrees with it vacuously).
GATE_TOL = 1.5
GATE_MIN_WEIGHT = 2.0
GATE_MIN_OBSERVABLE = 0.10
GATE_REPAIRS = 4             # corrective iterations allowed before refusing
GATE_CONVERGED = 0.05        # px; a repair iterates to here, not merely to GATE_TOL


# --------------------------------------------------------------------------
# Stage 1 — the global affine, and the matrices needed to compose with it.
# --------------------------------------------------------------------------
def global_stage(images: list[np.ndarray], ref: int):
    """ECC affine per frame: aligned frames for ANALYSIS plus the warp matrices.

    `align_stack(depth_bins=0)` produces the same aligned frames but does not
    return its matrices, and a field cannot be composed without them. The KAT
    asserts the two agree pixel-for-pixel so this stays a view of the shipped
    stage rather than a second implementation of it.
    """
    h, w = images[0].shape[:2]
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-6)
    ref_gray = to_gray_float(images[ref]) / 255.0

    warps: list[np.ndarray | None] = []
    coarse: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    for i, image in enumerate(images):
        if i == ref:
            warps.append(None)
            coarse.append(image)
            valid.append(np.ones((h, w), bool))
            continue
        matrix = np.eye(2, 3, dtype=np.float32)
        try:
            _, matrix = cv2.findTransformECC(
                ref_gray, to_gray_float(image) / 255.0, matrix,
                cv2.MOTION_AFFINE, criteria, None, 5,
            )
        except cv2.error:
            warps.append(None)
            coarse.append(image)
            valid.append(np.ones((h, w), bool))
            continue
        common = dict(dsize=(w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warps.append(matrix)
        coarse.append(cv2.warpAffine(image, matrix, **common))
        valid.append(cv2.warpAffine(np.full((h, w), 255, np.uint8), matrix, **common) == 255)
    return coarse, warps, valid


# --------------------------------------------------------------------------
# Stage 2 — the focal field, and the per-region pair choice.
# --------------------------------------------------------------------------
def focal_field(coarse: list[np.ndarray]):
    """Per-pixel focal peak (subpixel) and focus contrast, plus the energy stack.

    The peak index is a depth PROXY and is used only to decide how many layers a
    tile contains and which pixels belong to which — never to key a correction
    (F99). Contrast (max - min energy across the sweep) is the vote weight: a
    blank wall pixel has no focal peak worth the name and must not vote on which
    frames the tile is composited from.
    """
    energies = np.stack(content_aware_energies([to_gray_float(i) for i in coarse]), 0)
    n = len(coarse)
    winner = np.argmax(energies, axis=0)
    yy, xx = np.indices(winner.shape)
    lo = np.clip(winner - 1, 0, n - 1)
    hi = np.clip(winner + 1, 0, n - 1)
    a, b, c = energies[lo, yy, xx], energies[winner, yy, xx], energies[hi, yy, xx]
    denominator = a - 2.0 * b + c
    offset = np.where(np.abs(denominator) > 1e-9, 0.5 * (a - c) / np.where(
        np.abs(denominator) > 1e-9, denominator, 1.0), 0.0)
    peak = np.clip(winner + np.clip(offset, -0.5, 0.5), 0.0, n - 1.0).astype(np.float32)
    contrast = (energies.max(axis=0) - energies.min(axis=0)).astype(np.float32)
    return peak, contrast, energies


def _otsu_split(values: np.ndarray, weights: np.ndarray, n: int):
    """Best two-clump description of a weighted 1-D focal distribution.

    Otsu, not a gap: a bimodal distribution whose tails meet has no large
    consecutive gap, so single-linkage chains through it into one clump every
    time (F98). Returns (threshold, quality) where quality is the fraction of
    total variance the split explains.
    """
    edges = np.arange(-0.25, n + 0.25, 0.25)
    histogram, _ = np.histogram(values, bins=edges, weights=weights)
    total = histogram.sum()
    if total <= 0:
        return None, 0.0
    centres = 0.5 * (edges[:-1] + edges[1:])
    p = histogram / total
    mean = float((p * centres).sum())
    variance = float((p * (centres - mean) ** 2).sum())
    if variance < 1e-9:
        return None, 0.0
    cumulative = np.cumsum(p)
    cumulative_mean = np.cumsum(p * centres)
    best, best_quality = None, 0.0
    for i in range(len(centres) - 1):
        w0 = cumulative[i]
        w1 = 1.0 - w0
        if w0 < MIN_SIDE_WEIGHT or w1 < MIN_SIDE_WEIGHT:
            continue
        mu0 = cumulative_mean[i] / w0
        mu1 = (mean - cumulative_mean[i]) / w1
        quality = w0 * w1 * (mu0 - mu1) ** 2 / variance
        if quality > best_quality:
            best, best_quality = 0.5 * (centres[i] + centres[i + 1]), quality
    return best, best_quality


def _sharpest_frame(energies: np.ndarray, box, member: np.ndarray) -> int:
    """The frame in which a set of pixels is sharpest — the user's own criterion."""
    y0, y1, x0, x1 = box
    block = energies[:, y0:y1, x0:x1]
    totals = (block * member[None]).reshape(len(block), -1).sum(axis=1)
    return int(np.argmax(totals))


def tile_pairs(peak, contrast, energies, valid):
    """One (foreground frame, background frame) pair per tile.

    A tile whose focal distribution is one clump gets a DEGENERATE pair (a, a) —
    one frame. That is not a failure mode to be engineered away: a region at one
    depth needs one frame, and it is also the exact construction F108 measured to
    be clean in the streak zone ("compositing that region from the reference
    alone is clean — that is the bar").
    """
    h, w = peak.shape
    n = energies.shape[0]
    tiles = []
    for y0 in range(0, max(1, h - TILE // 2), STRIDE):
        for x0 in range(0, max(1, w - TILE // 2), STRIDE):
            y1, x1 = min(y0 + TILE, h), min(x0 + TILE, w)
            if y1 - y0 < TILE // 2 or x1 - x0 < TILE // 2:
                continue
            box = (y0, y1, x0, x1)
            ok = valid[y0:y1, x0:x1]
            weight = contrast[y0:y1, x0:x1] * ok
            if weight.sum() <= 1e-6:
                continue
            values = peak[y0:y1, x0:x1]
            threshold, quality = _otsu_split(values.ravel(), weight.ravel(), n)
            pair = None
            if threshold is not None and quality >= OTSU_MIN_QUALITY:
                near = (values <= threshold) & ok
                far = (values > threshold) & ok
                a = _sharpest_frame(energies, box, near.astype(np.float32))
                b = _sharpest_frame(energies, box, far.astype(np.float32))
                if abs(a - b) >= MIN_SEPARATION:
                    pair = (min(a, b), max(a, b))
            if pair is None:
                single = _sharpest_frame(energies, box, ok.astype(np.float32))
                pair = (single, single)
            tiles.append({"box": box, "pair": pair, "quality": float(quality),
                          "weight": float(weight.sum())})
    return tiles


def merge_pairs(tiles):
    """Collapse the per-tile pairs into at most MAX_PAIRS distinct renders.

    Greedy by total tile weight: the heaviest pair absorbs every pair within
    MERGE_TOL frames of it on both members. Every extra pair is another render
    AND another stitch boundary, so fewer is better as long as no tile is forced
    onto a pair that is more than a frame from what it asked for.
    """
    counts: dict[tuple[int, int], float] = {}
    for tile in tiles:
        counts[tile["pair"]] = counts.get(tile["pair"], 0.0) + tile["weight"]
    kept: list[tuple[int, int]] = []
    remaining = dict(counts)
    while remaining and len(kept) < MAX_PAIRS:
        best = max(remaining, key=lambda p: remaining[p])
        kept.append(best)
        for pair in list(remaining):
            if (abs(pair[0] - best[0]) <= MERGE_TOL and abs(pair[1] - best[1]) <= MERGE_TOL):
                del remaining[pair]

    def nearest(pair):
        return min(kept, key=lambda k: abs(k[0] - pair[0]) + abs(k[1] - pair[1]))

    for tile in tiles:
        tile["kept"] = nearest(tile["pair"])
    return kept


def ownership(tiles, kept, shape):
    """One-hot pair ownership per pixel, from the overlapping tile votes.

    Hard, because the alternative would be blending two independently registered
    renders in a wide band — which is a geometric blend wearing a photometric
    costume. The multiband stitch does the feathering, at band scale, exactly as
    `fuse_coherent` does for frame ownership.
    """
    h, w = shape
    votes = np.zeros((len(kept), h, w), np.float32)
    index = {pair: i for i, pair in enumerate(kept)}
    for tile in tiles:
        y0, y1, x0, x1 = tile["box"]
        votes[index[tile["kept"]], y0:y1, x0:x1] += tile["weight"] + 1e-3
    votes = np.stack([cv2.GaussianBlur(v, (0, 0), STRIDE / 2.0) for v in votes], 0)
    owner = np.argmax(votes, axis=0)
    weights = np.zeros_like(votes)
    yy, xx = np.indices(owner.shape)
    weights[owner, yy, xx] = 1.0
    return owner, weights


# --------------------------------------------------------------------------
# Stage 3 — one rigid translation per (frame, layer).
# --------------------------------------------------------------------------
def masked_translation(ref_gray, moving_gray, mask, max_shift):
    """Coarse-to-fine masked ECC translation, or None if unsupportable.

    A single-scale ECC from identity does not find a 19 px shift, and phase
    correlation saturates at a quarter of the patch (PLAYBOOK §0b), so the
    estimate is built on a 3-level Gaussian pyramid: solve at 1/4 scale where 19
    px is 5, then refine. Known-answer tested in `kat()` at +5/+12/+20 px before
    being believed on real data (§12.1).
    """
    if mask.sum() < MIN_LAYER_PIXELS:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5)
    pyramid = []
    r, m, k = ref_gray, moving_gray, mask.astype(np.uint8) * 255
    for _ in range(PYRAMID_LEVELS):
        pyramid.append((r, m, k))
        if min(r.shape) < 64:
            break
        r, m = cv2.pyrDown(r), cv2.pyrDown(m)
        k = cv2.resize(k, (r.shape[1], r.shape[0]), interpolation=cv2.INTER_NEAREST)

    warp = np.eye(2, 3, dtype=np.float32)
    for level in range(len(pyramid) - 1, -1, -1):
        r, m, k = pyramid[level]
        if k.sum() // 255 < 40:
            warp[0, 2] *= 2.0
            warp[1, 2] *= 2.0
            continue
        try:
            _, warp = cv2.findTransformECC(r, m, warp, cv2.MOTION_TRANSLATION,
                                           criteria, k, 5)
        except cv2.error:
            pass
        if level > 0:
            warp = warp.copy()
            warp[0, 2] *= 2.0
            warp[1, 2] *= 2.0
    if not np.isfinite(warp).all():
        return None
    if float(np.hypot(warp[0, 2], warp[1, 2])) > max_shift:
        return None
    return warp


def layer_masks(energies, pair, owned, gradient):
    """Which pixels of the region each frame of the pair is responsible for.

    Keyed on the FOCUS CONTEST between the two frames — i.e. on which frame will
    actually supply the pixel — not on a depth value (F99). The gradient gate is
    the standing ECC guard: a textureless mask cannot vote (PLAYBOOK §0b).
    """
    a, b = pair
    textured = gradient >= A._REFINE_MIN_GRADIENT
    if a == b:
        dense = [np.ones(textured.shape, bool)]
    else:
        wins_a = energies[a] >= energies[b]
        dense = [wins_a, ~wins_a]
    return [owned & textured & d for d in dense], dense


# --------------------------------------------------------------------------
# Stage 3b — focal-weighted edge-profile layer fits, and the validity gate.
# --------------------------------------------------------------------------
class EdgeEvidence:
    """The frame's material edges, their focal frames, and cached profiles.

    Built once per stack and shared by every layer fit and every gate call. The
    features are MATERIAL only — `_material_features` routes silhouette edges to
    a separate list, and a curved object's limb slides across its own surface as
    the viewpoint moves, so it may decide coverage but never a rigid fit (F92).

    The focal frame of a feature is read out of the focal field the architecture
    already computes (`peak`), which is the same quantity `motion_groups.
    _focal_frames` measures with its own Laplacian window — subpixel, and free.
    """

    def __init__(self, grays, ref, depth, valid, peak):
        self.grays = grays
        self.ref = ref
        self.features = MG._material_features(grays[ref], depth, valid)
        self.focal = [float(peak[int(round(y)), int(round(x))])
                      for (x, y, _nx, _ny) in self.features]
        self._cache: dict = {}

    def profile(self, frame, index, dx=0.0, dy=0.0):
        key = (frame, index, round(dx, 3), round(dy, 3))
        hit = self._cache.get(key)
        if hit is None:
            x, y, nx, ny = self.features[index]
            hit = MG._profile(self.grays[frame], x + dx, y + dy, nx, ny)
            self._cache[key] = hit
        return hit

    def _offset_for(self, index, offset):
        """Where a candidate geometry sends this feature, as a local displacement.

        A geometry is either a translation (a 2-tuple) or a full 3x3 map. Sampling
        the moving frame at the mapped point IS sampling the warped image at the
        feature, so a candidate geometry is verified with no image warp at all.
        The feature's normal is not rotated with the map: every transform here is
        within a degree of the identity, so the second-order term is far below the
        instrument's own precision.
        """
        if offset is None:
            return 0.0, 0.0
        if not isinstance(offset, np.ndarray):
            return float(offset[0]), float(offset[1])
        x, y, _nx, _ny = self.features[index]
        mapped = offset @ np.array([x, y, 1.0])
        return float(mapped[0] - x), float(mapped[1] - y)

    def indices_in(self, mask):
        return [i for i, (x, y, _nx, _ny) in enumerate(self.features)
                if mask[int(round(y)), int(round(x))]]

    def residual_translation(self, frame, indices, offset=(0.0, 0.0),
                             symmetric=False):
        """Focal-weighted least-squares translation still separating ref and frame.

        Each material edge constrains only the component along its own normal, so
        the fit is over-determined in general and singular when every normal is
        parallel — which is why the observability of each axis is returned with
        the answer rather than assumed (F103).

        `offset` is a shift already applied to `frame`. Sampling that frame's
        profile at (x+offset) IS the profile of the shifted image at x, so the
        forward verification costs one profile per feature and no image warp.

        `symmetric` adds the focal weight of the REFERENCE side. group_align
        weights only the moving side because it measures an object near its own
        focal plane and propagates; here the pairing is the other way round — a
        layer supplied by frame 11 is one the reference is deeply defocused in, so
        every profile match on it is blurred-against-sharp. Measured consequence
        on the kitchen: three of four layer fits moved TOWARD ZERO against the
        ECC (frame 8's own layer −1.57 → −0.98), which is F99's defocus bias
        arriving from the side focal weighting was not watching. Requiring a
        feature to be sharp in BOTH frames makes the estimator DECLINE where it
        cannot see instead of returning a confident under-read.
        """
        rows, target, weights = [], [], []
        for i in indices:
            base = self.profile(self.ref, i)
            dx, dy = self._offset_for(i, offset)
            moved = self.profile(frame, i, dx, dy)
            shift, peak = MG._match(base, moved)
            if peak < MG.MIN_PEAK or abs(shift) > 40:
                continue
            focal_weight = float(np.exp(-0.5 * ((frame - self.focal[i]) / FOCAL_SIGMA) ** 2))
            if symmetric:
                focal_weight *= float(np.exp(
                    -0.5 * ((self.ref - self.focal[i]) / FOCAL_SIGMA) ** 2))
            weight = peak * focal_weight
            if weight <= 1e-3:
                continue
            _x, _y, nx, ny = self.features[i]
            rows.append([nx, ny])
            target.append(shift)
            weights.append(weight)
        if len(rows) < MIN_EDGE_FEATURES:
            return None
        A = np.asarray(rows, float)
        b = np.asarray(target, float)
        w = np.asarray(weights, float)
        total = float(w.sum())
        if total < MIN_EDGE_WEIGHT:
            return None
        # Two IRLS passes: a single outlier edge (a disoccluded one, or one whose
        # profile crosses the layer boundary) otherwise drags a 2-DOF fit freely.
        solution = np.zeros(2)
        active = w.copy()
        for _ in range(3):
            root = np.sqrt(active)[:, None]
            solution, *_ = np.linalg.lstsq(A * root, b * root.ravel(), rcond=None)
            residual = np.abs(A @ solution - b)
            cutoff = max(float(np.median(residual)) * 2.5, 0.5)
            active = w * np.minimum(1.0, cutoff / np.maximum(residual, 1e-9))
        # Observability per axis, in the frame's own eigenbasis: an axis no normal
        # points along is not measured, and must not be gated on.
        moment = (A * active[:, None]).T @ A / max(total, 1e-9)
        eigenvalues, eigenvectors = np.linalg.eigh(moment)
        # The RAW weighted RMS, before any translation is solved out. This is the
        # number that can see a geometry that is right in the middle of a layer
        # and wrong at its ends — a solved translation absorbs the constant part
        # and would report a spatially varying error as zero.
        rms = float(np.sqrt(float((w * b * b).sum()) / total))
        return {"shift": (float(solution[0]), float(solution[1])),
                "weight": total, "n": len(rows), "rms": rms,
                "eigenvalues": eigenvalues, "eigenvectors": eigenvectors}


def _observable_norm(fit):
    """Magnitude of a residual translation, counting only observable directions."""
    vector = np.asarray(fit["shift"], float)
    keep = np.zeros(2)
    for value, axis in zip(fit["eigenvalues"], fit["eigenvectors"].T):
        if value >= GATE_MIN_OBSERVABLE:
            keep += float(vector @ axis) * axis
    return float(np.hypot(keep[0], keep[1])), float(fit["eigenvalues"].min())


def edge_refined_shift(evidence, frame, indices, coarse):
    """Coarse ECC translation, refined by the focal-weighted edge-profile fit.

    Why both. The ECC pyramid is the only instrument here with the RANGE to find a
    19 px layer shift (PLAYBOOK §0b: phase correlation saturates at a quarter of
    the patch, and a single-scale ECC does not converge from identity). The
    edge-profile fit is the arc's most-validated motion instrument but its profile
    is ±28 px long, so it measures a residual well and a gross displacement badly.
    Chaining them uses each where it is valid: ECC brings the layer to within a
    pixel or two, the edge fit closes that residual on MATERIAL edges with focal
    weighting, and the step is capped so a bad edge fit cannot run away.
    """
    if coarse is None:
        return None, None
    fit = evidence.residual_translation(frame, indices, coarse, symmetric=True)
    if fit is None:
        return coarse, None
    step = np.asarray(fit["shift"], float)
    # Only the observable part of the residual may be applied — an unobservable
    # axis has no measurement behind it and moving along it is invention.
    applied = np.zeros(2)
    for value, axis in zip(fit["eigenvalues"], fit["eigenvectors"].T):
        if value >= GATE_MIN_OBSERVABLE:
            applied += float(step @ axis) * axis
    length = float(np.hypot(applied[0], applied[1]))
    if length > EDGE_MAX_STEP:
        return coarse, fit
    return (coarse[0] + applied[0], coarse[1] + applied[1]), fit


def gate_shift(evidence, frame, indices, shift):
    """Forward-verify a fitted layer shift against the layer's own edges.

    The promotion blocker this exists for: the estimator once returned a
    wrong-but-in-range shift and the factory fell to 0.668 with nothing objecting.
    `max_shift` is a plausibility bound, not evidence — it cannot tell a correct
    fit from a wrong one of the same size. So the fit must be VERIFIED: apply it,
    and ask the layer's own material edges whether the layer has stopped moving.
    Independent of the fitter (profile correlation along edge normals vs. masked
    ECC over a dense region), focal-weighted so defocus cannot bias it toward the
    reassuring answer of zero (F99), and material-only so a limb's view-dependent
    slide is not mistaken for error (F92).

    Note the deliberate asymmetry with the ESTIMATOR above, which weights both
    focal sides and declines when the layer is defocused in the reference. The
    gate does not, and it is the same split F104 draws for limb edges: a
    defocused profile match carries a few TENTHS of a pixel of bias, which
    forbids it from producing a rigid fit and is irrelevant to a question whose
    wrong answer is several pixels out. Measured both ways: on real correct
    kitchen fits this statistic reads 0.01–0.27 px; on a deliberate +8 px error
    it reads 7.97 px. It may verify what it may not estimate.

    Returns (status, statistic, reason) with status in {"verified",
    "contradicted", "unverifiable"} — trinary, because F106's rule is that a
    geometric decision is made in full, refused, or left alone, never ramped.
    """
    if shift is None:
        return "unverifiable", float("nan"), "no fit"
    fit = evidence.residual_translation(frame, indices, shift)
    if fit is None:
        return "unverifiable", float("nan"), "no edge support"
    if fit["weight"] < GATE_MIN_WEIGHT:
        return "unverifiable", float("nan"), "thin support"
    residual, weakest = _observable_norm(fit)
    if weakest < GATE_MIN_OBSERVABLE:
        # Every normal points the same way: one axis carries no measurement. The
        # gate judges the axis it can see and says so rather than pretending.
        reason = f"1 axis only, residual {residual:.2f} px"
    else:
        reason = f"residual {residual:.2f} px"
    status = "verified" if residual <= GATE_TOL else "contradicted"
    return status, residual, reason


def _repair(evidence, frame, indices, probe, shift, max_shift):
    """One corrective iteration of a contradicted fit, from the gate's own reading.

    Preferring the doubly-focal weighting (sharp in BOTH frames) keeps the repair
    unbiased where that evidence exists; where it does not, the moving-side
    weighting is used anyway, on F104's logic — a few tenths of a pixel of
    defocus bias is irrelevant to undoing an error of several pixels.
    """
    fit = evidence.residual_translation(frame, indices, probe, symmetric=True)
    if fit is None:
        fit = evidence.residual_translation(frame, indices, probe)
    if fit is None:
        return None
    step = np.zeros(2)
    for value, axis in zip(fit["eigenvalues"], fit["eigenvectors"].T):
        if value >= GATE_MIN_OBSERVABLE:
            step += float(np.asarray(fit["shift"]) @ axis) * axis
    if float(np.hypot(step[0], step[1])) > max_shift:
        return None
    moved = np.array([[1.0, 0.0, step[0]], [0.0, 1.0, step[1]], [0.0, 0.0, 1.0]]) @ probe
    if shift is not None:
        shift = (shift[0] + float(step[0]), shift[1] + float(step[1]))
    return shift, moved


def _centroid(mask, shape):
    h, w = shape
    if mask.sum() < 1:
        return np.array([(w - 1) / 2.0, (h - 1) / 2.0])
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()])


def _rigidify(matrix, mask, shape):
    """Reduce a composed layer transform to the pure translation it should be.

    The architecture's own premise is that each member of a pair owns ONE depth
    layer and can therefore be warped by ONE RIGID translation (that is what
    removes the field discontinuity, the stretch limiter and the soft geometric
    blend). The prototype did not honour it: it composed the GLOBAL AFFINE into
    every render, and that affine was fitted across both layers at once, so it
    absorbs differential parallax as a spurious scale — exactly F96's failure
    ("a radial term imitates two separated regions translating differently").

    Measured on the analytic factory, which has zero breathing so every non-
    translational term in its global affine is over-fit: frame 1's affine reads
    scale 0.9954, which spreads ±1.3 px of sampling error across the frame even
    when the layer's own shift is analytically exact. Since 1 px of near-layer
    error costs 0.0268 GT-SSIM here, that term alone was the factory gap.

    So the composed transform is evaluated at the layer's own centre of support
    and collapsed to the translation that reproduces it there. Nothing is
    invented: it is the same displacement, applied rigidly as claimed.
    """
    anchor = _centroid(mask, shape)
    centre = np.array([anchor[0], anchor[1], 1.0])
    mapped = np.asarray(matrix, float) @ centre
    return np.array([[1.0, 0.0, mapped[0] - centre[0]],
                     [0.0, 1.0, mapped[1] - centre[1]],
                     [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------
# The architecture.
# --------------------------------------------------------------------------
def twoframe_stack(images, ref=None, refusal=True, stitch="multiband",
                   step_evidence="both", fallback=True, erode=0, report=False,
                   fit="edge", gate=True, inject=None, oracle_shift=None,
                   layer_geometry="affine", select="geometry"):
    """Per-region two-frame fusion of a focus stack. Returns (fused, info).

    `fit`          "ecc" (the prototype's masked coarse-to-fine ECC) or "edge"
                   (that ECC plus a focal-weighted edge-profile refinement).
    `select`       "geometry" (default) — take the fit from `fit`, but propose
                   BOTH the composed global affine and its rigid collapse and
                   apply whichever forward-verifies better at the layer's own
                   material edges. "verify" also selects between the ecc and edge
                   fits the same way; measured slightly worse, see the notes.
                   None — take `fit` and `layer_geometry` as given, which is how
                   every ablation below is run.
    `gate`         forward-verify the chosen geometry before applying it. A
                   CONTRADICTED fit refuses its member; an UNVERIFIABLE one
                   declines the correction and keeps the global stage.
    `inject`       {(pair_index, layer): (dx, dy)} — a deliberate error added to a
                   fitted shift, so the gate can be tested against a known answer.
    `oracle_shift` callable(frame, layer, warp, frames, anchor) -> (dx, dy),
                   bypassing the estimator entirely; the in-architecture ceiling.
    """
    n = len(images)
    if ref is None:
        ref = n // 2
    h, w = images[0].shape[:2]
    diagonal = float(np.hypot(h, w))
    max_shift = MAX_SHIFT_FRACTION * diagonal

    coarse, warps, valid = global_stage(images, ref)
    common = np.logical_and.reduce(valid)
    peak, contrast, energies = focal_field(coarse)
    depth = depth_from_focus(coarse)
    probe = np.ones((5, 5), np.uint8)
    depth_step = ((cv2.dilate(depth, probe) - cv2.erode(depth, probe))
                  > A._OCCLUSION_MIN_DEPTH_STEP).astype(np.uint8)

    tiles = tile_pairs(peak, contrast, energies, common)
    kept = merge_pairs(tiles)
    owner, stitch_weights = ownership(tiles, kept, (h, w))

    ref_gray = to_gray_float(coarse[ref]).astype(np.float32) / 255.0
    grays = [to_gray_float(c).astype(np.float32) / 255.0 for c in coarse]
    gradient = cv2.magnitude(
        cv2.Sobel(ref_gray * 255.0, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(ref_gray * 255.0, cv2.CV_32F, 0, 1, ksize=3),
    )

    evidence = None
    if fit == "edge" or gate or select == "verify":
        evidence = EdgeEvidence(grays, ref, depth, common, peak)

    candidates, used, diagnostics = [], set(), []
    for index, pair in enumerate(kept):
        owned = owner == index
        # The fit is supported by a dilated footprint: the region decides which
        # PAIR to use, but the object whose motion is being measured usually
        # extends past the tiles that elected it, and more support is strictly
        # better conditioning for one rigid translation.
        support = cv2.dilate(owned.astype(np.uint8), np.ones((TILE, TILE), np.uint8)) > 0
        fit_masks, dense = layer_masks(energies, pair, support & common, gradient)
        frames = [pair[0]] if pair[0] == pair[1] else [pair[0], pair[1]]

        # EVERY frame of the pair is measured against EVERY layer, not just the
        # one it will supply. The extra measurement is what makes disocclusion
        # visible: a frame's own layer motion says where to put its content, and
        # the DIFFERENCE between its two layer motions says how far its near
        # layer swung across its far layer — which is exactly the width of the
        # strip it cannot legitimately supply (F82).
        table: dict[tuple[int, int], tuple[float, float] | None] = {}
        geometry: dict[tuple[int, int], np.ndarray] = {}
        verdicts: dict[tuple[int, int], tuple[str, float, str]] = {}
        for frame in frames:
            for layer, mask in enumerate(fit_masks):
                if frame == ref:
                    table[(frame, layer)] = (0.0, 0.0)
                    geometry[(frame, layer)] = np.eye(3)
                    verdicts[(frame, layer)] = ("verified", 0.0, "reference")
                    continue
                if erode > 0:
                    # Both sides of a defocused silhouette are compromised
                    # (PLAYBOOK §0): the layer's boundary ring mixes the other
                    # surface into the fit. Erode the DENSE layer and re-apply the
                    # texture gate — eroding the gated mask instead decimates a
                    # speckle pattern rather than shrinking a region, which is a
                    # measured disaster (see the notes' rejected list).
                    shrunk = cv2.erode(dense[layer].astype(np.uint8),
                                       np.ones((2 * erode + 1, 2 * erode + 1),
                                               np.uint8)) > 0
                    mask = mask & shrunk
                base = (np.eye(3) if warps[frame] is None
                        else A._homogeneous(warps[frame]))
                indices = (evidence.indices_in(dense[layer] & support & common)
                           if evidence is not None else [])
                fitted = masked_translation(ref_gray, grays[frame], mask, max_shift)
                ecc = (None if fitted is None
                       else (float(fitted[0, 2]), float(fitted[1, 2])))

                # CROSS-LAYER entries are not geometries and are never rendered.
                # They exist only to give F82 its disocclusion width — the
                # DIFFERENTIAL between a frame's two layer motions, which is the
                # one quantity here with a closed form and the one KAT 4
                # validated the masked ECC against (87–100% of the analytic
                # parallax). Routing them through the edge fit and the gate
                # collapsed that differential and with it the refusal that closes
                # the streak: measured, pair (6,8) refused 20.4% -> 0.0%, pair
                # (4,6) 11.8% -> 0.0%, and the kitchen flank went 1.96 -> 3.3.
                # Different question, different instrument, each used where it was
                # known-answer tested.
                if frames[layer] != frame:
                    table[(frame, layer)] = ecc
                    verdicts[(frame, layer)] = ("verified", 0.0, "cross-layer (ecc)")
                    continue

                # PROPOSALS. Nothing here is chosen yet; each is a hypothesis the
                # verification below has to prefer before it is applied.
                proposals = []
                if oracle_shift is not None:
                    proposals = [("oracle", oracle_shift(
                        frame, layer, warps[frame], frames,
                        _centroid(fit_masks[layer], (h, w))))]
                else:
                    proposals.append(("ecc", ecc))
                    if fit == "edge" and ecc is not None and evidence is not None:
                        refined, _f = edge_refined_shift(evidence, frame, indices, ecc)
                        if refined is not None and refined != ecc:
                            proposals.append(("edge", refined))
                    if select != "verify":
                        proposals = proposals[:1] if fit == "ecc" else proposals[-1:]
                if inject and (index, layer) in inject and layer < len(frames) \
                        and frames[layer] == frame:
                    bump = inject[(index, layer)]
                    proposals = [(label, None if s is None
                                  else (s[0] + bump[0], s[1] + bump[1]))
                                 for label, s in proposals]

                # A rendering matrix maps reference coordinates onto the ORIGINAL
                # frame; the evidence lives on the GLOBALLY ALIGNED grays. So each
                # candidate carries both: `matrix` for the single resample, and
                # `base⁻¹ @ matrix` for verification. For the plain composed
                # affine that check is exactly T(shift), which is what the gate
                # measured before it learned about geometries.
                inverse = np.linalg.inv(base)
                geometries = []
                for label, s in proposals:
                    if s is None:
                        continue
                    composed = base @ np.array([[1.0, 0.0, s[0]], [0.0, 1.0, s[1]],
                                                [0.0, 0.0, 1.0]])
                    rigid = _rigidify(composed, fit_masks[layer], (h, w))
                    options = ([("affine", composed), ("rigid", rigid)] if select
                               else [(layer_geometry,
                                      rigid if layer_geometry == "rigid" else composed)])
                    for name, candidate in options:
                        geometries.append((f"{label}/{name}", s, candidate,
                                           inverse @ candidate))

                # VERIFICATION-DRIVEN CHOICE. The estimator and the geometry are
                # the same kind of question — "which map puts this layer where the
                # evidence says it is" — so they are answered by the same measured
                # quantity: the focal-weighted RMS of what is STILL displaced at
                # the layer's own material edges once the map is applied. The raw
                # RMS, not a solved translation, because a composed global affine
                # is typically right in the middle of a layer and wrong at its
                # ends, and a solved translation would absorb that away.
                #
                # Both scene-splits this arc hit are decided here, in opposite
                # directions and with no threshold: the analytic factory has zero
                # breathing, so its global affine's scale is pure parallax
                # contamination (F96) and the rigid collapse verifies better; the
                # kitchen's affine carries a real y-scale of 1.03 by frame 11, and
                # there the composed affine verifies better and stands.
                chosen, chosen_label = None, "no fit"
                if geometries:
                    if select and evidence is not None and len(geometries) > 1:
                        best = None
                        for label, s, matrix, probe in geometries:
                            check = evidence.residual_translation(frame, indices, probe)
                            if check is None:
                                continue
                            if best is None or check["rms"] < best[0] - 1e-6:
                                best = (check["rms"], label, s, matrix, probe)
                        if best is not None:
                            chosen_label = f"{best[1]} rms {best[0]:.2f}"
                            chosen = best[2:]
                    if chosen is None:
                        label, s, matrix, probe = geometries[0]
                        chosen, chosen_label = (s, matrix, probe), label

                shift, matrix = (None, base) if chosen is None else chosen[:2]
                verdict = ("verified", 0.0, "ungated")
                if gate and chosen is not None:
                    probe = chosen[2]
                    verdict = gate_shift(evidence, frame, indices, probe)
                    repairing = verdict[0] == "contradicted"
                    for _attempt in range(GATE_REPAIRS):
                        # Once a fit is known to need repair, iterate the
                        # corrective measurement to convergence rather than
                        # stopping the moment it scrapes past the gate: GATE_TOL
                        # is the line between usable and refused, not a target.
                        if not repairing or (verdict[0] == "verified"
                                             and verdict[1] <= GATE_CONVERGED):
                            break
                        # The verification did not merely say NO — it measured HOW
                        # WRONG, and that measurement is the correction. Refusal is
                        # the answer when nothing can be recovered; here something
                        # can, so the fit is repaired once and re-verified, and only
                        # a fit that still fails is refused. Measured: refusal alone
                        # took the eroded-mask disaster from 0.668 to 0.916, because
                        # refusing the near layer over 82% of the frame hands it to
                        # a reference that is defocused there; one repair takes it
                        # to 0.971. Refusal remains the floor, not the plan.
                        repaired = _repair(evidence, frame, indices, probe,
                                           shift, max_shift)
                        if repaired is None:
                            break
                        shift, probe = repaired
                        matrix = base @ probe
                        status, statistic, reason = gate_shift(
                            evidence, frame, indices, probe)
                        verdict = (status, statistic, f"repaired -> {reason}")
                    if verdict[0] != "verified":
                        # CONTRADICTED means the evidence says the layer is still
                        # displaced after the correction — the observation is
                        # unusable and the member is refused below. UNVERIFIABLE
                        # means there is no evidence either way; the correction is
                        # then simply not made (the frame keeps the global stage's
                        # geometry, which is the baseline everything already
                        # accepts) but the observation itself is not thrown away.
                        # This is PLAYBOOK §0's own question applied to a fit:
                        # is the evidence ABSENT, or is it against you?
                        shift, matrix = None, base
                table[(frame, layer)] = shift
                geometry[(frame, layer)] = matrix
                verdicts[(frame, layer)] = (verdict[0], verdict[1],
                                            f"{chosen_label}; {verdict[2]}")

        rendered, valids, shifts = [], [], []
        refused_member = []
        for layer, frame in enumerate(frames):
            own = table[(frame, layer)]
            matrix = geometry[(frame, layer)]
            shifts.append(own)
            # A member whose OWN layer fit failed verification may not supply that
            # layer at all. The prototype's `None` quietly fell back to the global
            # affine, which is the same silent invention the gate exists to stop:
            # it is a geometry nothing measured. Refuse the member instead and let
            # the reference cover it (F82 — refused zones become reference-quality).
            refused_member.append(frame != ref
                                  and verdicts[(frame, layer)][0] == "contradicted")
            # `geometry` already holds the ONE homogeneous matrix chosen for this
            # (frame, layer): global stage composed with the verified layer shift,
            # collapsed to a rigid translation where the evidence preferred that.
            # It samples the ORIGINAL frame exactly once (PLAYBOOK §0).
            map_x, map_y = A._blended_coordinate_maps(
                [matrix], [np.ones((h, w), np.float32)], (h, w))
            rendered.append(cv2.remap(images[frame], map_x, map_y, cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0))
            valids.append(cv2.remap(np.full((h, w), 255, np.uint8), map_x, map_y,
                                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=0) == 255)
            used.add(frame)

        usable = None
        if len(rendered) == 1:
            fused = rendered[0] if not refused_member[0] else images[ref]
        else:
            if refusal or any(refused_member):
                usable = (_pair_refusal(frames, ref, table, dense, depth_step,
                                        (h, w), step_evidence) if refusal
                          else [np.ones((h, w), bool) for _ in frames])
                for layer, is_refused in enumerate(refused_member):
                    if is_refused:
                        usable[layer] = np.zeros((h, w), bool)
                # A gate failure leaves a hole no member can fill, so the
                # reference is not optional there even under `fallback=False`.
                admit_reference = fallback or any(refused_member)
                if admit_reference and ref not in frames:
                    # With only two sources, refusal has nowhere to fall back to:
                    # blocking one member forces the other, which at a
                    # disocclusion ribbon is the frame whose near layer is
                    # displaced. The REFERENCE is the geometry of last resort —
                    # unwarped, so its content is correctly placed by definition,
                    # merely defocused. It is admitted ONLY where both members
                    # are refused, which keeps the architecture two-frame
                    # everywhere it is confident (F82's own answer: refused zones
                    # become reference-quality).
                    rendered.append(images[ref])
                    usable.append(~usable[0] & ~usable[1])
            fused = fuse_perband(rendered, usable=usable)
        candidates.append(fused)
        valid_here = np.logical_and.reduce(valids)
        refused = 0.0
        if refusal and len(rendered) > 1 and usable is not None:
            refused = float(np.mean([1.0 - m[owned].mean() for m in usable[:2]]))
        diagnostics.append({"pair": pair, "shifts": shifts, "refused": refused,
                            "verdicts": verdicts, "gated": list(refused_member),
                            "frames": list(frames), "usable": usable,
                            "matrices": [geometry[(f, l)]
                                         for l, f in enumerate(frames)],
                            "area": float(owned.mean()), "valid": valid_here,
                            "rendered": rendered if report else None,
                            "fused": fused if report else None,
                            "owned": owned if report else None})

    for diagnostic in diagnostics:
        common &= diagnostic["valid"]

    if len(candidates) == 1:
        fused = candidates[0]
    elif stitch == "paste":
        # The F79 control: hard region copy, no multiband. Kept as an ablation
        # because "the stitch is the risk" is a claim that has to be re-measured
        # here, not inherited.
        stack = np.stack(candidates, 0)
        yy, xx = np.indices(owner.shape)
        fused = stack[owner, yy, xx]
    else:
        fused = multiband_blend(candidates, stitch_weights)

    x0, y0, x1, y1 = A._largest_valid_rectangle(common)
    info = {"tiles": tiles, "pairs": kept, "owner": owner, "peak": peak,
            "crop": (x0, y0, x1, y1), "diagnostics": diagnostics,
            "frames_used": sorted(used), "coarse": coarse if report else None}
    return fused[y0:y1, x0:x1].copy(), info


def twoframe_fullres(natives, working_width=1080, ref=None, **kwargs):
    """Native-resolution two-frame fusion, estimated at working scale (F107).

    Every parameter in this file is pixel-scaled (`TILE`, `STRIDE`,
    `MAX_SHIFT_FRACTION`, `MIN_LAYER_PIXELS`, the profile lengths inside
    `motion_groups`), so running the analysis at 24 MP puts every measurement
    outside the regime it was validated in. `fullres_apply.py`'s rule transfers
    it unchanged:

        field_native(X) = s * field_small(X / s)

    and it is easier here than anywhere else in the pipeline, because each
    candidate's geometry is ONE homogeneous matrix rather than a sampled field.
    Scaling a matrix is exact — no resize of a coordinate map, no interpolation
    of a field, no second resample:

        M_native = S @ M_small @ S^-1,   S = diag(s, s, 1)

    Ownership and usable masks are per-pixel decisions, not geometry, so they
    scale nearest-neighbour exactly as `fullres_apply` scales its occlusion
    masks. The NATIVE pixels are resampled exactly once.
    """
    nh, nw = natives[0].shape[:2]
    scale = nw / float(working_width)
    small = [cv2.resize(f, (working_width, int(round(nh / scale))),
                        interpolation=cv2.INTER_AREA) for f in natives]
    if ref is None:
        ref = len(small) // 2
    working_fused, info = twoframe_stack(small, ref, **kwargs)

    S = np.diag([scale, scale, 1.0])
    S_inv = np.diag([1.0 / scale, 1.0 / scale, 1.0])
    sh, sw = small[0].shape[:2]

    def blow_up(mask, order=cv2.INTER_NEAREST):
        return cv2.resize(mask.astype(np.uint8), (nw, nh), interpolation=order) > 0

    candidates, valids = [], []
    for diagnostic in info["diagnostics"]:
        rendered = []
        for matrix, frame in zip(diagnostic["matrices"], diagnostic["frames"]):
            native_matrix = S @ np.asarray(matrix, float) @ S_inv
            map_x, map_y = A._blended_coordinate_maps(
                [native_matrix], [np.ones((nh, nw), np.float32)], (nh, nw))
            rendered.append(cv2.remap(natives[frame], map_x, map_y, cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0))
            valids.append(cv2.remap(np.full((nh, nw), 255, np.uint8), map_x, map_y,
                                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=0) == 255)
        usable = diagnostic["usable"]
        if usable is not None:
            usable = [blow_up(m) for m in usable]
            if len(usable) > len(rendered):
                rendered.append(natives[ref])
        candidates.append(rendered[0] if len(rendered) == 1
                          else fuse_perband(rendered, usable=usable))

    if len(candidates) == 1:
        fused = candidates[0]
    else:
        owner = cv2.resize(info["owner"].astype(np.uint8), (nw, nh),
                           interpolation=cv2.INTER_NEAREST)
        weights = np.zeros((len(candidates), nh, nw), np.float32)
        yy, xx = np.indices(owner.shape)
        weights[owner, yy, xx] = 1.0
        fused = multiband_blend(candidates, list(weights))

    common = np.logical_and.reduce(valids)
    x0, y0, x1, y1 = A._largest_valid_rectangle(common)
    info["native_crop"] = (x0, y0, x1, y1)
    info["scale"] = scale
    info["working"] = (sw, sh)
    info["working_fused"] = working_fused
    return fused[y0:y1, x0:x1].copy(), info


def _pair_refusal(frames, ref, table, dense, depth_step, shape, evidence="both"):
    """Per-frame usable masks for one pair, by F82's rule.

    Each frame of the pair is rendered with ONE rigid translation, so it has no
    internal disagreement to detect and its own layer is placed exactly. The
    disocclusion is still real, and it is the failure this architecture is most
    exposed to: the background frame's NEAR layer sits ~19 px away from where the
    composite puts the near layer, so the strip the near object vacated in the
    background frame is still covered by that frame's own copy of the object.
    Measured on the kitchen bottle it renders a white ribbon of displaced bottle
    beside the correctly-placed one.

    The step evidence is taken from the PAIR ITSELF — the boundary of the focus
    contest between the two frames — unioned with the usual depth-step probe. A
    layer boundary IS a depth discontinuity by construction, and unlike the
    guided depth map it cannot ramp smoothly across the junction (F104/F108).
    """
    h, w = shape
    boundary = np.zeros((h, w), np.uint8)
    if len(dense) > 1:
        probe = np.ones((5, 5), np.uint8)
        first = dense[0].astype(np.uint8)
        boundary = (cv2.dilate(first, probe) - cv2.erode(first, probe)).astype(np.uint8)
    if evidence == "depth":
        step = np.asarray(depth_step, np.uint8)
    elif evidence == "pair":
        step = boundary
    else:
        step = np.maximum(np.asarray(depth_step, np.uint8), boundary)

    usable = []
    for layer, frame in enumerate(frames):
        own = table[(frame, layer)]
        if frame == ref or own is None:
            # The reference is an unwarped observation: it can always supply.
            usable.append(np.ones((h, w), bool))
            continue
        hard_x = np.zeros((h, w), np.float32)
        hard_y = np.zeros((h, w), np.float32)
        for other_layer, mask in enumerate(dense):
            other = table[(frame, other_layer)]
            if other is None:
                continue
            hard_x[mask] = other[0] - own[0]
            hard_y[mask] = other[1] - own[1]
        usable.append(~A._occlusion_mask(hard_x, hard_y, step))
    return usable


# --------------------------------------------------------------------------
# Instruments: known-answer tests (§12.1) before anything is believed.
# --------------------------------------------------------------------------
def kat() -> None:
    import parallax_gen as P
    from focusstack.align import align_stack

    print("KAT 1 — global stage matches the shipped align_stack(depth_bins=0)")
    frames, _truth, _mask = P.build_stack()
    coarse, _warps, _valid = global_stage(frames, P.REFERENCE)
    shipped = align_stack(frames, depth_bins=0, crop_valid=False)
    worst = max(float(np.abs(a.astype(int) - b.astype(int)).max())
                for a, b in zip(coarse, shipped))
    print(f"  worst per-pixel difference: {worst:.0f} (0 = identical)")

    print("\nKAT 2 — masked_translation on synthetic shifts of a real frame")
    source = cv2.imread(sorted(glob.glob(os.path.join(KITCHEN, "*.jpg")))[6])
    grey = to_gray_float(source).astype(np.float32) / 255.0
    h, w = grey.shape
    mask = np.zeros((h, w), bool)
    mask[120:400, 480:660] = True
    # `warpAffine` moves content by -truth, and the returned warp maps REFERENCE
    # coordinates onto moving ones, so the correct answer is dx = -truth.
    print(f"  {'truth dx':>9} {'measured dx':>12} {'measured dy':>12} {'error':>8}")
    for truth in (2.0, 5.0, 12.0, 20.0, 30.0):
        matrix = np.float32([[1, 0, -truth], [0, 1, 0]])
        moved = cv2.warpAffine(grey, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        warp = masked_translation(grey, moved, mask, 60.0)
        if warp is None:
            print(f"  {truth:9.1f} {'refused':>12}")
            continue
        dx, dy = float(warp[0, 2]), float(warp[1, 2])
        print(f"  {-truth:9.1f} {dx:12.3f} {dy:12.3f} {abs(dx + truth):8.3f}")

    print("\nKAT 3 — pair choice on the factory, where the answer is known")
    print(f"  truth: near plane sharpest at frame {P.NEAR_FOCUS_FRAME}, "
          f"far plane at frame {P.FAR_FOCUS_FRAME}")
    coarse, _warps, valid = global_stage(frames, P.REFERENCE)
    peak, contrast, energies = focal_field(coarse)
    tiles = tile_pairs(peak, contrast, energies, np.logical_and.reduce(valid))
    counts: dict[tuple[int, int], int] = {}
    for tile in tiles:
        counts[tile["pair"]] = counts.get(tile["pair"], 0) + 1
    for pair, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  pair {pair}: {count:3d} tiles")

    print("\nKAT 4 — layer translations on the factory against analytic truth\n"
          "  The global affine has already absorbed an unknown share of the scene's\n"
          "  motion, so the absolute per-layer residual has no closed form. The\n"
          "  DIFFERENTIAL does: whatever the affine took, it took from both layers\n"
          "  equally, so (near − far) must equal −(k−ref)·(3.2 − 0.7).")
    ref = P.REFERENCE
    ref_gray = to_gray_float(coarse[ref]).astype(np.float32) / 255.0
    grays = [to_gray_float(c).astype(np.float32) / 255.0 for c in coarse]
    near_mask = _mask
    gradient = cv2.magnitude(
        cv2.Sobel(ref_gray * 255.0, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(ref_gray * 255.0, cv2.CV_32F, 0, 1, ksize=3))
    textured = gradient >= A._REFINE_MIN_GRADIENT
    print(f"  {'frame':>5} {'near':>8} {'far':>8} {'near−far':>9} {'truth':>8} "
          f"{'ratio':>7}")
    for frame in range(len(frames)):
        if frame == ref:
            continue
        got = []
        for mask in (near_mask, ~near_mask):
            warp = masked_translation(ref_gray, grays[frame], mask & textured, 60.0)
            got.append(None if warp is None else float(warp[0, 2]))
        if None in got:
            print(f"  {frame:5d} {'refused':>8}")
            continue
        expected = -(frame - ref) * (P.NEAR_SHIFT_PER_FRAME - P.FAR_SHIFT_PER_FRAME)
        print(f"  {frame:5d} {got[0]:8.3f} {got[1]:8.3f} {got[0] - got[1]:9.3f} "
              f"{expected:8.3f} {(got[0] - got[1]) / expected:7.2f}")


def _synthetic_pair(truth_dx, truth_dy=0.0):
    """A real kitchen frame and a rigidly shifted copy of it, plus its evidence.

    `warpAffine` moves content by −truth, and every estimator here returns the
    warp that maps REFERENCE coordinates onto MOVING ones, so the known answer is
    (−truth_dx, −truth_dy). Depth is flat because the displacement IS rigid, which
    is the whole point of a known-answer test: nothing here is view-dependent, so
    every detected edge is legitimately a material edge (F92).
    """
    source = cv2.imread(sorted(glob.glob(os.path.join(KITCHEN, "*.jpg")))[6])
    grey = to_gray_float(source).astype(np.float32) / 255.0
    h, w = grey.shape
    matrix = np.float32([[1, 0, -truth_dx], [0, 1, -truth_dy]])
    moved = cv2.warpAffine(grey, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    mask = np.zeros((h, w), bool)
    mask[120:400, 480:660] = True
    evidence = EdgeEvidence([grey, moved], 0, np.zeros((h, w), np.float32),
                            np.ones((h, w), bool), np.ones((h, w), np.float32))
    return grey, moved, mask, evidence


def kat_hardening() -> None:
    """§12.1 for the two instruments this hardening adds, before either is used."""
    print("KAT 5 — focal-weighted edge-profile residual, on synthetic rigid shifts\n"
          "  Same convention as the ECC: the answer is the warp taking REFERENCE\n"
          "  coordinates to MOVING ones, so a content move of +t reads as −t.")
    print(f"  {'truth':>12} {'ecc dx':>8} {'edge dx':>8} {'edge dy':>8} "
          f"{'n':>4} {'|err| ecc':>10} {'|err| edge':>11}")
    for tdx, tdy in ((2.0, 0.0), (5.0, 0.0), (12.0, 0.0), (20.0, 0.0),
                     (7.0, 3.0), (-9.0, -4.0)):
        grey, moved, mask, evidence = _synthetic_pair(tdx, tdy)
        indices = evidence.indices_in(mask)
        warp = masked_translation(grey, moved, mask, 60.0)
        coarse = (float(warp[0, 2]), float(warp[1, 2])) if warp is not None else None
        refined, _f = edge_refined_shift(evidence, 1, indices, coarse)
        ecc_error = np.hypot(coarse[0] + tdx, coarse[1] + tdy)
        edge_error = np.hypot(refined[0] + tdx, refined[1] + tdy)
        print(f"  ({-tdx:+5.1f},{-tdy:+5.1f}) {coarse[0]:8.3f} {refined[0]:8.3f} "
              f"{refined[1]:8.3f} {len(indices):4d} {ecc_error:10.3f} {edge_error:11.3f}")

    print("\nKAT 6 — the validity gate, against known-good and known-wrong shifts\n"
          f"  A fit is applied only if forward verification finds ≤ {GATE_TOL} px of\n"
          "  residual translation left at the layer's own material edges.")
    grey, moved, mask, evidence = _synthetic_pair(12.0)
    indices = evidence.indices_in(mask)
    truth = (-12.0, 0.0)
    print(f"  {'candidate shift':>18} {'error':>7} {'residual':>9} {'verdict':>13}")
    for error in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, -8.0, 20.0):
        shift = (truth[0] + error, truth[1])
        status, statistic, _reason = gate_shift(evidence, 1, indices, shift)
        print(f"  ({shift[0]:+7.2f},{shift[1]:+5.2f}) {error:+7.1f} {statistic:9.3f} "
              f"{status:>13}")

    print("\n  and the same gate where it must NOT fire — a correct fit with only\n"
          "  ~0.3 px of estimation error, which is what the real layer fits carry:")
    for tdx in (3.0, 19.0):
        grey, moved, mask, evidence = _synthetic_pair(tdx)
        indices = evidence.indices_in(mask)
        warp = masked_translation(grey, moved, mask, 60.0)
        coarse = (float(warp[0, 2]), float(warp[1, 2]))
        status, statistic, _r = gate_shift(evidence, 1, indices, coarse)
        print(f"  measured {coarse[0]:+7.3f} for truth {-tdx:+6.1f}: residual "
              f"{statistic:.3f} px -> {status}")


def hardening() -> None:
    """The two acceptance tests this hardening pass had to pass, on both scenes."""
    import metrics
    import parallax_gen as P

    frames, truth, _near = P.build_stack()
    src = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(KITCHEN, "*.jpg")))]
    ref = len(src) // 2
    flank = flank_mask(src[ref])
    target = src[ref][STREAK[0]:STREAK[1], STREAK[2]:STREAK[3]].astype(np.float32)

    def score(**kwargs):
        fused, info = twoframe_stack(frames, P.REFERENCE, **kwargs)
        a, b, c, d = info["crop"]
        return metrics.ref_ssim(fused, truth[b:d, a:c]), info

    def kitchen_flank(**kwargs):
        fused, info = twoframe_stack(src, ref, **kwargs)
        x0, y0, _a, _b = info["crop"]
        crop = fused[STREAK[0] - y0:STREAK[1] - y0,
                     STREAK[2] - x0:STREAK[3] - x0].astype(np.float32)
        delta = np.abs(crop - target).max(axis=2)[flank]
        return delta, info

    print("=== TASK 1 — the validity gate, against a known-wrong layer fit ===")
    print("A wrong-but-in-range shift is invisible to `max_shift`: it is the right\n"
          "SIZE. Only verification can tell it from a correct fit of the same size.")
    clean, _ = score()
    print(f"\n{'scenario':<40} {'gate off':>9} {'gate on':>9} {'vs clean':>9}")
    print(f"{'(clean, no error injected)':<40} {clean:9.6f} {clean:9.6f} {0.0:9.6f}")
    for label, kwargs in (
        ("+2 px into pair 0 layer 0", dict(inject={(0, 0): (2.0, 0.0)})),
        ("+8 px into pair 0 layer 0", dict(inject={(0, 0): (8.0, 0.0)})),
        ("+20 px into pair 0 layer 0", dict(inject={(0, 0): (20.0, 0.0)})),
        ("both layers, x and y", dict(inject={(0, 0): (8.0, -3.0),
                                              (0, 1): (-6.0, 4.0)})),
        ("the notes' own case: fit masks eroded 7 px", dict(erode=7)),
        ("  eroded 13 px", dict(erode=13)),
    ):
        off, _ = score(gate=False, **kwargs)
        on, info = score(gate=True, **kwargs)
        print(f"{label:<40} {off:9.6f} {on:9.6f} {on - clean:+9.6f}")
        for diagnostic in info["diagnostics"]:
            for key, verdict in diagnostic["verdicts"].items():
                if "repair" in verdict[2] or verdict[0] != "verified":
                    print(f"      frame {key[0]} layer {key[1]}: {verdict[2]}")

    print("\n=== and the other half: the gate must be SILENT on correct fits ===")
    for label, kwargs in (("analytic factory", {}), ("kitchen sweep", {})):
        if label.startswith("analytic"):
            off, _ = score(gate=False)
            on, info = score(gate=True)
            print(f"  {label:<18} GT-SSIM gate off {off:.6f}  gate on {on:.6f}"
                  f"   refusals {sum(sum(d['gated']) for d in info['diagnostics'])}")
        else:
            off, _i = kitchen_flank(gate=False)
            on, info = kitchen_flank(gate=True)
            print(f"  {label:<18} flank mean gate off {off.mean():.2f}  "
                  f"gate on {on.mean():.2f}"
                  f"   refusals {sum(sum(d['gated']) for d in info['diagnostics'])}")

    print("\n=== TASK 2 — closing the estimation gap, one change at a time ===")
    print(f"{'configuration':<44} {'factory':>9} | {'flank mean':>10} {'>12':>7}")
    print(f"{'shipped depth-bin path':<44} {0.972808:9.6f} | "
          f"{5.98:10.2f} {16.34:6.2f}%")
    for label, kwargs in (
        ("two-frame PROTOTYPE (ecc, affine, no gate)",
         dict(fit="ecc", select=None, layer_geometry="affine", gate=False)),
        ("  ecc fit, rigid layer geometry",
         dict(fit="ecc", select=None, layer_geometry="rigid", gate=False)),
        ("  edge fit, affine layer geometry",
         dict(fit="edge", select=None, layer_geometry="affine", gate=False)),
        ("  edge fit, rigid layer geometry",
         dict(fit="edge", select=None, layer_geometry="rigid", gate=False)),
        ("  edge fit, VERIFIED geometry, no gate",
         dict(select="geometry", gate=False)),
        ("  + also verifying the fit choice", dict(select="verify")),
        ("HARDENED DEFAULT (edge, verified geometry, gate)", dict()),
    ):
        value, _ = score(**kwargs)
        delta, _i = kitchen_flank(**kwargs)
        print(f"{label:<44} {value:9.6f} | {delta.mean():10.2f} "
              f"{100 * (delta > 12).mean():6.2f}%")


def fullres() -> None:
    """KAT for the full-resolution transfer: 2x upscale in, downscale out."""
    src = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(KITCHEN, "*.jpg")))]
    h, w = src[0].shape[:2]
    print(f"kitchen native {w}x{h}; the KAT feeds a 2x upscale and asks whether the\n"
          "native result, brought back down, matches the working-resolution one.")
    natives = [cv2.resize(f, (2 * w, 2 * h), interpolation=cv2.INTER_CUBIC)
               for f in src]

    native, info = twoframe_fullres(natives, working_width=w)
    working = info["working_fused"]
    scale = info["scale"]
    print(f"  working  {info['working'][0]}x{info['working'][1]}  crop {info['crop']}")
    print(f"  native   {native.shape[1]}x{native.shape[0]}  crop "
          f"{info['native_crop']}  scale {scale:.2f}")
    print(f"  pairs {info['pairs']}")

    # Both outputs recompute their own common-footprint crop — natively for the
    # native one, exactly as `fullres_apply` does — so they do not begin at the
    # same pixel. Compare on the overlap, or the test measures the crop rather
    # than the transfer (measured: comparing the two crops naively reports mean
    # 13.2 and 28% over 12, all of it a ~5 px registration of the comparison).
    wx0, wy0, wx1, wy1 = info["crop"]
    nx0, ny0, nx1, ny1 = info["native_crop"]
    ox0 = int(np.ceil(max(wx0, nx0 / scale))) + 4
    oy0 = int(np.ceil(max(wy0, ny0 / scale))) + 4
    ox1 = int(min(wx1, nx1 / scale)) - 4
    oy1 = int(min(wy1, ny1 / scale)) - 4
    left = working[oy0 - wy0:oy1 - wy0, ox0 - wx0:ox1 - wx0]
    right = native[int(oy0 * scale) - ny0:int(oy1 * scale) - ny0,
                   int(ox0 * scale) - nx0:int(ox1 * scale) - nx0]
    down = cv2.resize(right, (left.shape[1], left.shape[0]),
                      interpolation=cv2.INTER_AREA)
    delta = np.abs(down.astype(np.float32) - left.astype(np.float32)).max(axis=2)
    print(f"  overlap {left.shape[1]}x{left.shape[0]}")
    print(f"  |native(downscaled) - working|: mean {delta.mean():.2f}, "
          f"median {np.median(delta):.2f}, 99th pct {np.percentile(delta, 99):.0f}, "
          f"max {delta.max():.0f}")
    print(f"  pixels over 12 levels: {100 * (delta > 12).mean():.2f}%")
    # A residual is only small relative to something. The control is the same
    # comparison against the reference FRAME, i.e. against an image that differs
    # from the output by exactly the work the pipeline did.
    reference = cv2.resize(natives[len(natives) // 2],
                           (info["working"][0], info["working"][1]),
                           interpolation=cv2.INTER_AREA)[oy0:oy1, ox0:ox1]
    control = np.abs(reference.astype(np.float32)
                     - left.astype(np.float32)).max(axis=2)
    print(f"  control (reference frame vs the working output): "
          f"mean {control.mean():.2f}, over 12 {100 * (control > 12).mean():.2f}%")
    # And the FLOOR: this KAT synthesises its native frames by upscaling, so the
    # 2x-up / area-down round trip is not the identity and no transfer, however
    # exact, can score below it.
    floor = np.abs(cv2.resize(natives[len(natives) // 2], (w, h),
                              interpolation=cv2.INTER_AREA)[oy0:oy1, ox0:ox1]
                   .astype(np.float32)
                   - src[len(src) // 2][oy0:oy1, ox0:ox1].astype(np.float32)
                   ).max(axis=2)
    print(f"  floor (the KAT's own resample round trip): mean {floor.mean():.2f}, "
          f"over 12 {100 * (floor > 12).mean():.2f}%")
    os.makedirs(OUT, exist_ok=True)
    cv2.imwrite(os.path.join(OUT, "TF_fullres_kat.png"), np.hstack([left, down]))
    print(f"  wrote {OUT}/TF_fullres_kat.png (working | native-downscaled)")


def modes() -> None:
    """TASK 4, timeboxed: do kitchen tiles actually hold THREE focal modes?

    Only a measurement, and deliberately not a build. Two frames can express two
    layers; the model has no way to REPORT a three-mode region, it just picks the
    best two. So: split each tile's weighted focal distribution with the same
    Otsu the architecture uses, then try to split each SIDE again with the same
    quality bar. A tile counts as three-mode only if a second split clears the
    same threshold on real weight — no new instrument, no new number.
    """
    src = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(KITCHEN, "*.jpg")))]
    ref = len(src) // 2
    coarse, _warps, valid = global_stage(src, ref)
    common = np.logical_and.reduce(valid)
    peak, contrast, energies = focal_field(coarse)
    n = energies.shape[0]

    two, three, total = 0, 0, 0
    for tile in tile_pairs(peak, contrast, energies, common):
        y0, y1, x0, x1 = tile["box"]
        weight = (contrast[y0:y1, x0:x1] * common[y0:y1, x0:x1]).ravel()
        values = peak[y0:y1, x0:x1].ravel()
        total += 1
        threshold, quality = _otsu_split(values, weight, n)
        if threshold is None or quality < OTSU_MIN_QUALITY:
            continue
        two += 1
        for side in (values <= threshold, values > threshold):
            if weight[side].sum() <= 1e-6:
                continue
            _t, q = _otsu_split(values[side], weight[side], n)
            if q >= OTSU_MIN_QUALITY:
                three += 1
                break
    print(f"kitchen tiles: {total}")
    print(f"  two-mode by the architecture's own Otsu bar : {two} ({100*two/total:.1f}%)")
    print(f"  a THIRD mode clearing the same bar          : {three} "
          f"({100*three/total:.1f}%)")
    print("\nThe brief's rule was: build the third-frame path only if >= 5% of tiles\n"
          "demand it. " + ("They do — record it and stop here."
                           if three / total >= 0.05 else
                           "They do not, so nothing was built."))


def _stamp(image, box, scale=3):
    y0, y1, x0, x1 = box
    crop = image[y0:y1, x0:x1]
    return cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale),
                      interpolation=cv2.INTER_NEAREST)


def factory() -> None:
    import metrics
    import parallax_gen as P
    from focusstack.align import align_stack

    frames, truth, near_mask = P.build_stack()

    aligned, report = align_stack(frames, motion="affine", depth_bins=3,
                                  return_report=True)
    x0, y0, x1, y1 = report["crop"]
    shipped = fuse_perband(aligned, usable=report["usable"])
    shipped_plain = fuse_perband(aligned)
    print(f"{'variant':<34} {'GT-SSIM':>9} {'crop':>18}")
    print(f"{'shipped bins (no refusal)':<34} "
          f"{metrics.ref_ssim(shipped_plain, truth[y0:y1, x0:x1]):9.6f} "
          f"{str((x0, y0, x1, y1)):>18}")
    print(f"{'shipped bins + refusal':<34} "
          f"{metrics.ref_ssim(shipped, truth[y0:y1, x0:x1]):9.6f} "
          f"{str((x0, y0, x1, y1)):>18}")

    for label, kwargs in (
        ("two-frame PROTOTYPE (ecc/affine)",
         dict(fit="ecc", select=None, layer_geometry="affine", gate=False)),
        ("  + rigid layer geometry",
         dict(fit="ecc", select=None, layer_geometry="rigid", gate=False)),
        ("  + edge-refined fit",
         dict(fit="edge", select=None, layer_geometry="rigid", gate=False)),
        ("two-frame HARDENED (verified + gate)", dict()),
    ):
        fused, info = twoframe_stack(frames, P.REFERENCE, **kwargs)
        a, b, c, d = info["crop"]
        print(f"{label:<34} {metrics.ref_ssim(fused, truth[b:d, a:c]):9.6f} "
              f"{str(info['crop']):>18}   pairs={info['pairs']} "
              f"frames={info['frames_used']}")
        for diagnostic in info["diagnostics"]:
            print(f"      pair {diagnostic['pair']} area {diagnostic['area'] * 100:5.1f}% "
                  f"shifts {diagnostic['shifts']}")


def oracle() -> None:
    """Oracle ladder for the factory: what is the ARCHITECTURE's ceiling?

    PLAYBOOK §0b: run the oracle ladder before building estimators, because the
    ceiling decides whether estimators are worth building. Rung 1 gives the
    two-frame composite the analytically exact per-layer displacement, so any
    remaining gap to the shipped path is a property of using two frames, not of
    measuring badly. Rung 2 swaps in the measured shifts, isolating estimation.
    """
    import metrics
    import parallax_gen as P

    frames, truth, near_mask = P.build_stack()
    h, w = frames[0].shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))

    def shifted(frame, dx):
        return cv2.remap(frames[frame], (grid_x + dx).astype(np.float32), grid_y,
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # The factory renders frame k's layer as L(x + (k-ref)*per_frame), so undoing
    # it samples at x - (k-ref)*per_frame. Sign checked against the "unaligned"
    # rung below: getting it backwards scores WORSE than doing nothing.
    near, far = P.NEAR_FOCUS_FRAME, P.FAR_FOCUS_FRAME
    print(f"{'rung':<44} {'GT-SSIM':>9}")
    for label, dn, df in (
        ("oracle 2-frame (exact per-layer shifts)",
         -(near - P.REFERENCE) * P.NEAR_SHIFT_PER_FRAME,
         -(far - P.REFERENCE) * P.FAR_SHIFT_PER_FRAME),
        ("oracle 2-frame, near layer 1 px wrong",
         -(near - P.REFERENCE) * P.NEAR_SHIFT_PER_FRAME + 1.0,
         -(far - P.REFERENCE) * P.FAR_SHIFT_PER_FRAME),
    ):
        fused = fuse_perband([shifted(near, dn), shifted(far, df)])
        print(f"{label:<44} {metrics.ref_ssim(fused, truth):9.6f}")

    # And the same two frames with NO per-layer correction at all — the ceiling
    # a two-frame architecture reaches if its alignment does nothing.
    fused = fuse_perband([frames[near], frames[far]])
    print(f"{'2-frame, unaligned':<44} {metrics.ref_ssim(fused, truth):9.6f}")
    # All six frames, oracle-aligned per layer: the N-frame ceiling on the same
    # scene, which says whether two frames lose information the others carry.
    rendered = []
    for k in range(len(frames)):
        step = k - P.REFERENCE
        near_view = shifted(k, -step * P.NEAR_SHIFT_PER_FRAME)
        far_view = shifted(k, -step * P.FAR_SHIFT_PER_FRAME)
        rendered.append(np.where(near_mask[..., None], near_view, far_view))
    print(f"{'oracle N-frame (per-layer, N=6)':<44} "
          f"{metrics.ref_ssim(fuse_perband(rendered), truth):9.6f}")

    # The rung the prototype was missing. The rungs above bypass twoframe_stack
    # entirely — two frames, two analytic shifts, one fuse — so the gap between
    # them and the measured run is estimation PLUS everything the architecture
    # itself does (tiling, degenerate single-frame regions, the stitch, refusal,
    # resampling through the global affine). Feeding the exact shifts THROUGH
    # twoframe_stack separates those two, and it is the only number that says how
    # much a better estimator can actually buy.
    print(f"\n{'in-architecture rungs (through twoframe_stack)':<44} {'GT-SSIM':>9}")
    for label, kwargs in (("exact shifts, gate off", dict(gate=False)),
                          ("exact shifts, gate on", dict(gate=True))):
        fused, info = twoframe_stack(frames, P.REFERENCE,
                                     oracle_shift=exact_layer_shift(P.REFERENCE),
                                     **kwargs)
        a, b, c, d = info["crop"]
        fired = sum(sum(x["gated"]) for x in info["diagnostics"])
        print(f"{label:<44} {metrics.ref_ssim(fused, truth[b:d, a:c]):9.6f}"
              f"   gate refusals {fired}")


def exact_layer_shift(ref):
    """Analytic per-(frame, layer) residual for the factory, AFTER the global affine.

    The rendered matrix is `base @ T(t)`, i.e. x -> A(x+t)+b, and the exact answer
    is x -> x + d with d = −(k−ref)·S_layer (the factory renders layer L of frame k
    displaced by −step·S_L, so undoing it samples there). A is not exactly the
    identity, so t is solved at the frame centre:  t = A⁻¹(d + (I−A)c − b).
    """
    import parallax_gen as P

    def shift(frame, layer, warp, frames, anchor):
        # The frame that OWNS a layer is the frame that layer is sharpest in, so
        # its distance to each focus frame names the plane.
        owner = frames[layer] if layer < len(frames) else frames[0]
        near = abs(owner - P.NEAR_FOCUS_FRAME) <= abs(owner - P.FAR_FOCUS_FRAME)
        per_frame = P.NEAR_SHIFT_PER_FRAME if near else P.FAR_SHIFT_PER_FRAME
        d = np.array([-(frame - ref) * per_frame, 0.0])
        if warp is None:
            return (float(d[0]), float(d[1]))
        matrix = np.asarray(warp, float)
        a_part, b_part = matrix[:, :2], matrix[:, 2]
        # Anchored at the LAYER's own centre of support, which is where the rigid
        # collapse evaluates the composed transform — anywhere else and the
        # affine's scale makes "exact" mean something different in each stage.
        t = np.linalg.solve(a_part, d + (np.eye(2) - a_part) @ anchor - b_part)
        return (float(t[0]), float(t[1]))

    return shift


STREAK = (230, 400, 600, 700)    # F108's acceptance box, uncropped coordinates
BOTTLE_EDGE = (160, 340, 600, 700)   # group_align's residual window


def _shift(box, x0, y0):
    return (box[0] - y0, box[1] - y0, box[2] - x0, box[3] - x0)


def kitchen() -> None:
    import eyetool
    import group_align as GA
    from focusstack.align import align_stack

    os.makedirs(OUT, exist_ok=True)
    src = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(KITCHEN, "*.jpg")))]
    ref = len(src) // 2
    reference = src[ref]

    fused, info = twoframe_stack(src, ref)
    pasted, _ = twoframe_stack(src, ref, stitch="paste")
    unrefused, _ = twoframe_stack(src, ref, refusal=False)
    print(f"pairs: {info['pairs']}")
    print(f"frames actually used: {info['frames_used']} of {len(src)}")
    for diagnostic in info["diagnostics"]:
        shifts = ", ".join("ref" if s == (0.0, 0.0) else
                           ("unmeasurable" if s is None else f"({s[0]:+.2f},{s[1]:+.2f})")
                           for s in diagnostic["shifts"])
        print(f"  pair {diagnostic['pair']} area {diagnostic['area'] * 100:5.1f}% "
              f"refused {diagnostic['refused'] * 100:4.1f}%  shifts [{shifts}]")
        for key, verdict in diagnostic["verdicts"].items():
            if verdict[2] not in ("reference", "cross-layer (ecc)"):
                print(f"        frame {key[0]} layer {key[1]}: {verdict[0]:<13} "
                      f"{verdict[2]}")
    single = sum(1 for t in info["tiles"] if t["pair"][0] == t["pair"][1])
    print(f"  tiles judged single-layer: {single}/{len(info['tiles'])}")

    shipped_aligned, report = align_stack(src, return_report=True)
    shipped = fuse_perband(shipped_aligned, usable=report["usable"])
    sx0, sy0, sx1, sy1 = report["crop"]
    tx0, ty0, tx1, ty1 = info["crop"]

    top = [_stamp(reference, STREAK),
           _stamp(shipped, _shift(STREAK, sx0, sy0)),
           _stamp(fused, _shift(STREAK, tx0, ty0))]
    bottom = [np.full_like(top[0], 127),
              eyetool._amplify_diff(top[1], top[0]),
              eyetool._amplify_diff(top[2], top[0])]
    cv2.imwrite(os.path.join(OUT, "TF_streak_ref_shipped_twoframe.png"),
                np.vstack([np.hstack(top), np.hstack(bottom)]))
    cv2.imwrite(os.path.join(OUT, "TF_kitchen_full.png"), fused)
    cv2.imwrite(os.path.join(OUT, "TF_kitchen_shipped.png"), shipped)
    cv2.imwrite(os.path.join(OUT, "TF_kitchen_paste_stitch.png"), pasted)

    # F108's own bar: "compositing that region from the reference alone is
    # clean". So the reference frame IS the local truth in the streak zone, and
    # the artifact is measurable as deviation from it. This is only legitimate
    # on the FLANK (a smooth surface at the reference's own focal plane, where
    # no other frame has anything better to offer); over the whole box the
    # background legitimately gets sharper and deviation is not error.
    flank = flank_mask(reference)
    target = reference[STREAK[0]:STREAK[1], STREAK[2]:STREAK[3]].astype(np.float32)
    print(f"\nF108 acceptance test — deviation from the reference frame on the "
          f"{int(flank.sum())} px low-contrast flank:")
    print(f"  {'output':<12} {'mean |Δ|':>9} {'max |Δ|':>8} {'px over 12':>11}")
    for label, image, (ox, oy) in (("shipped", shipped, (sx0, sy0)),
                                   ("two-frame", fused, (tx0, ty0))):
        crop = image[STREAK[0] - oy:STREAK[1] - oy,
                     STREAK[2] - ox:STREAK[3] - ox].astype(np.float32)
        delta = np.abs(crop - target).max(axis=2)[flank]
        print(f"  {label:<12} {delta.mean():9.2f} {delta.max():8.0f} "
              f"{100 * (delta > 12).mean():10.2f}%")

    # Where does each output's bottle edge sit relative to the reference frame's?
    # Measured on the OUTPUT, not on aligned sources: a fused edge that is fanned
    # or displaced is what the user sees.
    grey_ref = to_gray_float(reference).astype(np.float32) / 255.0
    print("\nbottle right-edge offset of the OUTPUT vs the reference frame:")
    for label, image, (ox, oy) in (("shipped", shipped, (sx0, sy0)),
                                   ("two-frame", fused, (tx0, ty0))):
        grey = to_gray_float(image).astype(np.float32) / 255.0
        y0, y1, x0, x1 = _shift(BOTTLE_EDGE, ox, oy)
        offset = _edge_offset(grey_ref, BOTTLE_EDGE, grey, (y0, y1, x0, x1))
        print(f"  {label:<12} {offset:+6.2f} px")

    coarse = align_stack(src, depth_bins=0, crop_valid=False)
    grey_coarse = [to_gray_float(i).astype(np.float32) / 255.0 for i in coarse]
    grey_shipped = [to_gray_float(i).astype(np.float32) / 255.0
                    for i in shipped_aligned]
    print("\nper-frame bottle residual (group_align.edge_shift):")
    print(f"  {'frame':>5} {'global only':>12} {'shipped path':>13} {'two-frame':>28}")
    for k in (8, 9, 10, 11):
        before = GA.edge_shift(grey_coarse[ref], grey_coarse[k], *BOTTLE_EDGE)
        after = GA.edge_shift(grey_shipped[ref], grey_shipped[k],
                              BOTTLE_EDGE[0] - sy0, BOTTLE_EDGE[1] - sy0,
                              BOTTLE_EDGE[2] - sx0, BOTTLE_EDGE[3] - sx0)
        print(f"  {k:5d} {before:+12.2f} {after:+13.2f} "
              f"{'not used at the bottle':>28}")

    # Disagreement-guided crops: point the eye at the informative pixels rather
    # than at a hand-picked window (PLAYBOOK I.2).
    ax0, ay0 = max(sx0, tx0), max(sy0, ty0)
    ax1, ay1 = min(sx1, tx1), min(sy1, ty1)
    eyetool.compare(
        {"reference": reference[ay0:ay1, ax0:ax1],
         "shipped": shipped[ay0 - sy0:ay1 - sy0, ax0 - sx0:ax1 - sx0],
         "twoframe": fused[ay0 - ty0:ay1 - ty0, ax0 - tx0:ax1 - tx0]},
        out=os.path.join(OUT, "TF_disagreement.png"), k=3, half=90, zoom=2.0)

    # Where is the stitch worst? Point the eye at the multiband/paste
    # disagreement, which is by construction where the region boundaries are.
    eyetool.compare(
        {"paste": pasted[ay0 - ty0:ay1 - ty0, ax0 - tx0:ax1 - tx0],
         "multiband": fused[ay0 - ty0:ay1 - ty0, ax0 - tx0:ax1 - tx0]},
        gt=reference[ay0:ay1, ax0:ax1],
        out=os.path.join(OUT, "TF_stitch_ablation.png"), k=2, half=90, zoom=2.0)

    # One supplementary window, recorded exactly: the cat figurine, where the
    # shipped output shows visible doubling. Hand-picked, so it diagnoses rather
    # than promotes (DEVSTYLE §1.3).
    cat = (365, 480, 320, 470)
    cv2.imwrite(os.path.join(OUT, "TF_cat_ref_shipped_twoframe.png"), np.hstack([
        _stamp(reference, cat, 2),
        _stamp(shipped, _shift(cat, sx0, sy0), 2),
        _stamp(fused, _shift(cat, tx0, ty0), 2)]))

    # Stitch ablation (F79's claim, re-measured here rather than inherited).
    delta = np.abs(fused.astype(np.float32) - pasted.astype(np.float32))
    print(f"\nstitch ablation: multiband vs hard paste — mean |Δ| {delta.mean():.3f}, "
          f"max {delta.max():.0f}, pixels over 8: {100 * (delta.max(2) > 8).mean():.2f}%")
    delta = np.abs(fused.astype(np.float32) - unrefused.astype(np.float32))
    print(f"pair-level disocclusion refusal: mean |Δ| {delta.mean():.3f}, "
          f"changed pixels {100 * (delta.max(2) > 0).mean():.2f}%")
    print(f"\nwrote {OUT}/TF_streak_ref_shipped_twoframe.png, TF_kitchen_full.png, "
          f"TF_kitchen_shipped.png, TF_kitchen_paste_stitch.png, TF_disagreement.png")


def flank_mask(reference, box=STREAK):
    """F108's low-contrast white flank inside the acceptance box.

    Scoped deliberately (DEVSTYLE §12.2): the whole streak box also contains
    background the fusion is SUPPOSED to change, so deviation measured over it
    is not error. The flank is the part where the reference frame is the local
    truth — a smooth surface at the reference's own focal plane, where no other
    frame has anything better to offer.
    """
    grey = to_gray_float(reference).astype(np.float32)[box[0]:box[1], box[2]:box[3]]
    mean = cv2.boxFilter(grey, cv2.CV_32F, (9, 9))
    variance = cv2.boxFilter(grey * grey, cv2.CV_32F, (9, 9)) - mean * mean
    return (grey > 170) & (np.sqrt(np.maximum(variance, 0.0)) < 4.0)


def variants() -> None:
    """A/B the two free choices — refusal and its step evidence — on both scenes."""
    import metrics
    import parallax_gen as P
    from focusstack.align import align_stack

    frames, truth, _near = P.build_stack()
    src = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(KITCHEN, "*.jpg")))]
    ref = len(src) // 2
    reference = src[ref]
    flank = flank_mask(reference)
    target = reference[STREAK[0]:STREAK[1], STREAK[2]:STREAK[3]].astype(np.float32)

    shipped_aligned, report = align_stack(src, return_report=True)
    shipped = fuse_perband(shipped_aligned, usable=report["usable"])
    sx0, sy0, _a, _b = report["crop"]
    crop = shipped[STREAK[0] - sy0:STREAK[1] - sy0,
                   STREAK[2] - sx0:STREAK[3] - sx0].astype(np.float32)
    delta = np.abs(crop - target).max(axis=2)[flank]
    print(f"{'variant':<34} {'factory GT-SSIM':>16} {'flank mean|Δ|':>14} "
          f"{'flank max':>10} {'>12':>7}")
    print(f"{'shipped depth-bin path':<34} {'0.972808':>16} {delta.mean():14.2f} "
          f"{delta.max():10.0f} {100 * (delta > 12).mean():6.2f}%")

    for label, kwargs in (
        ("two-frame, no refusal", dict(refusal=False)),
        ("refusal (depth step), no fallback", dict(step_evidence="depth", fallback=False)),
        ("refusal (pair edge), no fallback", dict(step_evidence="pair", fallback=False)),
        ("refusal (both), no fallback", dict(step_evidence="both", fallback=False)),
        ("refusal (depth step) + ref fallback", dict(step_evidence="depth")),
        ("refusal (pair edge) + ref fallback", dict(step_evidence="pair")),
        ("refusal (both) + ref fallback", dict(step_evidence="both")),
        ("  + fit masks eroded 3 px", dict(step_evidence="both", erode=3)),
        ("  + fit masks eroded 7 px", dict(step_evidence="both", erode=7)),
        ("  + fit masks eroded 13 px", dict(step_evidence="both", erode=13)),
    ):
        fused, info = twoframe_stack(frames, P.REFERENCE, **kwargs)
        a, b, c, d = info["crop"]
        score = metrics.ref_ssim(fused, truth[b:d, a:c])
        fused, info = twoframe_stack(src, ref, **kwargs)
        tx0, ty0, _c, _d = info["crop"]
        crop = fused[STREAK[0] - ty0:STREAK[1] - ty0,
                     STREAK[2] - tx0:STREAK[3] - tx0].astype(np.float32)
        delta = np.abs(crop - target).max(axis=2)[flank]
        print(f"{label:<34} {score:16.6f} {delta.mean():14.2f} {delta.max():10.0f} "
              f"{100 * (delta > 12).mean():6.2f}%")


def _edge_offset(grey_ref, ref_box, grey, box):
    """Where a vertical edge sits in `grey` relative to `grey_ref`'s own copy."""
    a = np.gradient(grey_ref[ref_box[0]:ref_box[1], ref_box[2]:ref_box[3]].mean(0))
    b = np.gradient(grey[box[0]:box[1], box[2]:box[3]].mean(0))
    a, b = a - a.mean(), b - b.mean()
    c = np.correlate(b, a, mode="full")
    i = int(np.argmax(c))
    off = 0.0
    if 0 < i < len(c) - 1:
        d = c[i - 1] - 2 * c[i] + c[i + 1]
        off = 0.5 * (c[i - 1] - c[i + 1]) / d if abs(d) > 1e-12 else 0.0
    return (i - (len(a) - 1)) + off


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "kat"
    {"kat": kat, "kat2": kat_hardening, "factory": factory, "oracle": oracle,
     "variants": variants, "kitchen": kitchen, "hardening": hardening,
     "fullres": fullres, "modes": modes}[mode]()
