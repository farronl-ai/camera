"""Per-region TWO-FRAME fusion — the routed alternative to the depth-bin path.

Where this sits. The shipped path (`align.py` + `fusion.fuse_perband`) registers
every frame to the reference with one global affine plus a depth-binned parallax
correction, then fuses all N frames. This module is a different architecture for
the same job, and it is NOT a replacement: `pipeline.run` ROUTES to it only on
scenes whose measured signature says the shipped path is in trouble (F101 — two
methods that win on opposite scenes are routed, never merged).

The architecture, in one line: for each region of the frame pick the frame in
which that region's FOREGROUND is sharpest and the frame in which its BACKGROUND
is sharpest, warp each member of that pair by ONE rigid transform, fuse the pair,
and stitch the per-region results into one output.

Why it exists (F108/F109). Every refusal gate the alignment arc built is driven by
LOCAL evidence, and F108 proved all of them simultaneously blind in a smooth
low-contrast stretch: no edges, and a guided depth map that ramps across the very
junction that matters. The pair choice needs no per-pixel evidence — it is a
REGIONAL statistic pooled over ~10^4 focal-peak votes, so a textureless pixel
simply abstains and the region still decides. And because each member of a pair
owns exactly one depth layer, the layer it places wrongly is by construction the
layer it is defocused in, so the focus contest discards it: no depth-dependent
field inside a region, hence no ramp, no stretch limiter, and no soft geometric
blend (F106 satisfied structurally rather than by a rule).

Provenance and status. Prototyped and hardened in `research/twoframe.py` over two
rounds (F109, F110; write-up in `research/twoframe_NOTES.md`). Measured there:
analytic factory GT-SSIM 0.9713 against the shipped path's 0.9728 — a real
regression, which is why this is routed and not shipped by default — and a
decisive win on the kitchen sweep, where F108's tan streak is eliminated (flank
mean |Δ| 5.98 -> 2.16, 16.34% -> 0.00% of pixels over 12) and the cat figurine
stops rendering doubled.

What the hard-won rules bind here, and where each shows up below:

  * "Resample pixels exactly ONCE" (PLAYBOOK §0). Analysis runs on globally
    aligned frames; RENDERING composes the global affine and the layer's rigid
    translation into one homogeneous matrix and remaps the ORIGINAL frame once.
  * "A geometric decision cannot be soft" (F106). Every candidate is rendered
    from rigid warps and ownership is one-hot; the only soft thing is the STITCH,
    which blends finished IMAGES through `multiband_blend`, never sampling fields.
  * "Never key a correction on depth" (F99). Layer masks are keyed on which frame
    of the pair WINS THE FOCUS CONTEST — the operational selector — not on a
    depth value.
  * "Validate an instrument against a known answer" (DEVSTYLE §12.1). This module
    is a port, and a port is a new instrument: `tests/test_twoframe_route.py`
    re-tests the global stage against `align_stack(depth_bins=0)` and the validity
    gate against an injected +8 px layer error.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import align as A
from . import motion_groups as MG
from .focus import content_aware_energies
from .fusion import depth_from_focus, fuse_coherent, multiband_blend
from .io import to_gray_float

# --- geometry of the analysis, all pixel-scaled (see WORKING_WIDTH) ----------
# A "region" is deliberately NOT an object: F98 established that turning feature
# groups into pixel regions is the open problem, and that whatever fills a region
# mask in is doing the real work while looking like plumbing. A tile claims
# nothing — it is only a locality over which focal statistics are pooled. Tiles
# overlap so a structure straddling a tile edge votes in both.
TILE = 96
STRIDE = 48

# A tile is called two-layer only if splitting its focal-peak distribution
# explains this fraction of its variance (Otsu, never a gap — F98), both sides
# carry real weight, and the modes sit at least MIN_SEPARATION frames apart.
# Below that the tile is one depth and takes ONE frame.
OTSU_MIN_QUALITY = 0.45
MIN_SIDE_WEIGHT = 0.12
MIN_SEPARATION = 1.5

# Distinct pairs kept for the whole frame. Every extra pair is another full
# two-frame render, and more importantly another stitch seam.
MAX_PAIRS = 6
MERGE_TOL = 1.0

# The layer fit is allowed a much larger shift than the shipped per-bin cap
# (`_REFINE_MAX_FRACTION` = 1.5% of the diagonal), because the kitchen bottle
# alone needs ~19 px and that cap is one reason the bins never propose it.
MAX_SHIFT_FRACTION = 0.045
MIN_LAYER_PIXELS = 250
PYRAMID_LEVELS = 3

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
# translation and was fixed between two measured populations (real correct fits
# read 0.01-0.32 px; the smallest error the gate must catch reads 1.97).
# GATE_MIN_OBSERVABLE is the share of the focal-weighted support an axis needs
# before the gate may judge that axis at all (F103: a feature whose normal is
# perpendicular to a motion agrees with it vacuously).
GATE_TOL = 1.5
GATE_MIN_WEIGHT = 2.0
GATE_MIN_OBSERVABLE = 0.10
GATE_REPAIRS = 4             # corrective iterations allowed before refusing
GATE_CONVERGED = 0.05        # px; a repair iterates to here, not merely to GATE_TOL

# Every constant above is in pixels and was exercised at 560x420 and 774x518
# (F110). Above this width the analysis runs downscaled and the resulting
# geometry is carried to native resolution by exact matrix conjugation (F107).
WORKING_WIDTH = 1100

# The architecture's LICENCE, and the second half of the routing rule.
#
# F109 stated this regime as a falsifiable claim before anything was measured:
# the two-frame architecture holds "only where each region's layers each have a
# frame that is both sharp and NEAR-REFERENCE enough to place. It would fail on a
# sweep whose near object is sharpest at an extreme." The kitchen is the good
# case — the bottle is sharpest AT the reference, so the +19.97 px correction the
# whole F99-F108 arc fought for is simply not needed. The large-motion sweep is
# the predicted bad case, and it fails exactly as predicted: the playing-card box
# is sharpest at frame 0 of 14 and needs +18.9 px, the pair elects frame 0 and
# fits it correctly (verified at 0.51 px) — and then F82's disocclusion refusal
# withdraws that member over 91% of the pair's area, because the ribbon it must
# refuse is as wide as the correction. The box returns reference-defocused, and
# the shipped override's sharp, correctly-placed "LAS VEGAS" is lost.
#
# So the licence is a DISPLACEMENT scale, and it is borrowed rather than invented:
# `align._REFINE_MAX_FRACTION` (1.5% of the frame diagonal) is the arc's existing
# statement of how large a per-region displacement can be before it stops being a
# refinement of the global warp and becomes re-registration. Two frames are the
# wrong instrument for re-registration — they carry 2.7x the shift sensitivity of
# an N-frame fusion (F110) and their refusal has only one other source to fall
# back to. Measured max layer shift: kitchen 2.1 px (limit 14.0), IMG-46 6.9 px
# (limit 20.2), large-motion 19.2 px (limit 14.0, declined).
SHIFT_LICENCE_FRACTION = A._REFINE_MAX_FRACTION


def shift_licence(shape) -> float:
    """Largest layer displacement the two-frame path may be trusted with, in px."""
    h, w = shape[:2]
    return SHIFT_LICENCE_FRACTION * float(np.hypot(h, w))


# --------------------------------------------------------------------------
# Stage 1 — the global affine, and the matrices needed to compose with it.
# --------------------------------------------------------------------------
def global_stage(images: list[np.ndarray], ref: int):
    """ECC affine per frame: aligned frames for ANALYSIS plus the warp matrices.

    `align_stack(depth_bins=0)` produces the same aligned frames but does not
    return its matrices, and a geometry cannot be composed without them. The port
    test asserts the two agree pixel-for-pixel, so this stays a VIEW of the
    shipped global stage rather than a second implementation of it.
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

    The peak index is a depth PROXY used only to decide how many layers a tile
    contains and which pixels belong to which — never to key a correction (F99).
    Contrast (max - min energy across the sweep) is the vote weight: a blank wall
    pixel has no focal peak worth the name and must not vote on which frames its
    tile is composited from.
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
    time (F98). Returns (threshold, quality), quality being the fraction of total
    variance the split explains.
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
    """The frame in which a set of pixels is sharpest — the architecture's criterion."""
    y0, y1, x0, x1 = box
    block = energies[:, y0:y1, x0:x1]
    totals = (block * member[None]).reshape(len(block), -1).sum(axis=1)
    return int(np.argmax(totals))


def tile_pairs(peak, contrast, energies, valid):
    """One (foreground frame, background frame) pair per tile.

    A tile whose focal distribution is one clump gets a DEGENERATE pair (a, a) —
    one frame. That is not a failure mode: a region at one depth needs one frame,
    and it is the exact construction F108 measured to be clean in the streak zone
    ("compositing that region from the reference alone is clean — that is the bar").
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
    MERGE_TOL frames of it on both members. Fewer pairs is better as long as no
    tile is forced onto a pair more than a frame from what it asked for.
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
            if abs(pair[0] - best[0]) <= MERGE_TOL and abs(pair[1] - best[1]) <= MERGE_TOL:
                del remaining[pair]

    def nearest(pair):
        return min(kept, key=lambda k: abs(k[0] - pair[0]) + abs(k[1] - pair[1]))

    for tile in tiles:
        tile["kept"] = nearest(tile["pair"])
    return kept


def ownership(tiles, kept, shape):
    """One-hot pair ownership per pixel, from the overlapping tile votes.

    Hard, because the alternative — blending two independently registered renders
    across a wide band — is a geometric blend wearing a photometric costume
    (F106). The multiband stitch does the feathering at band scale, exactly as
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
    px is 5, then refine. Known-answer tested at -2/-5/-12/-20/-30 px (F109 KAT 2).
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
        profile at (x+offset) IS the profile of the shifted image at x, so forward
        verification costs one profile per feature and no image warp.

        `symmetric` adds the focal weight of the REFERENCE side. `group_align`
        weights only the moving side because it measures an object near its own
        focal plane and propagates; here the pairing is inverted — a layer
        supplied by frame 11 is one the reference is deeply defocused in, so every
        profile match on it is blurred-against-sharp, and F99's defocus bias
        arrives from the side focal weighting was not watching (measured: three of
        four kitchen layer fits moved TOWARD ZERO against the ECC). Requiring a
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
        design = np.asarray(rows, float)
        observed = np.asarray(target, float)
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
            solution, *_ = np.linalg.lstsq(design * root, observed * root.ravel(),
                                           rcond=None)
            residual = np.abs(design @ solution - observed)
            cutoff = max(float(np.median(residual)) * 2.5, 0.5)
            active = w * np.minimum(1.0, cutoff / np.maximum(residual, 1e-9))
        # Observability per axis, in the fit's own eigenbasis: an axis no normal
        # points along is not measured, and must not be gated on.
        moment = (design * active[:, None]).T @ design / max(total, 1e-9)
        eigenvalues, eigenvectors = np.linalg.eigh(moment)
        # The RAW weighted RMS, before any translation is solved out. This is the
        # number that can see a geometry which is right in the middle of a layer
        # and wrong at its ends — a solved translation absorbs the constant part
        # and would report a spatially varying error as zero.
        rms = float(np.sqrt(float((w * observed * observed).sum()) / total))
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
    19 px layer shift. The edge-profile fit is the arc's most-validated motion
    instrument but its profile is +-28 px long, so it measures a residual well and
    a gross displacement badly. Chaining them uses each where it is valid, and the
    step is capped so a bad edge fit cannot run away.
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
    reassuring answer of zero (F99), material-only so a limb's view-dependent
    slide is not mistaken for error (F92).

    Note the deliberate asymmetry with the ESTIMATOR above, which weights both
    focal sides and declines when the layer is defocused in the reference. The
    gate does not, and it is F104's split transplanted: a defocused profile match
    carries a few TENTHS of a pixel of bias, which forbids it from producing a
    rigid fit and is irrelevant to a question whose wrong answer is several pixels
    out. It may verify what it may not estimate.

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
    weighting is used anyway, on F104's logic.
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

    The architecture's premise is that each member of a pair owns ONE depth layer
    and can therefore be warped by ONE RIGID translation. Composing the GLOBAL
    AFFINE into the render breaks it: that affine was fitted across both layers at
    once, so it absorbs differential parallax as a spurious scale (F96 — "a radial
    term imitates two separated regions translating differently"). Measured on the
    analytic factory, whose breathing is zero so every non-translational term is
    over-fit: frame 1's affine reads scale 0.9954, spreading +-1.3 px of sampling
    error across the frame even when the layer's own shift is exact.

    So the composed transform is evaluated at the layer's own centre of support
    and collapsed to the translation that reproduces it there. Nothing is
    invented: it is the same displacement, applied rigidly as claimed. It is a
    CANDIDATE, never a rule — the kitchen's affine carries a real y-scale of
    1.0285 by frame 11, and there the composed affine verifies better and stands.
    """
    anchor = _centroid(mask, shape)
    centre = np.array([anchor[0], anchor[1], 1.0])
    mapped = np.asarray(matrix, float) @ centre
    return np.array([[1.0, 0.0, mapped[0] - centre[0]],
                     [0.0, 1.0, mapped[1] - centre[1]],
                     [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------
# Stage 3c — the precondition of the focus contest: are the two members even
# looking at the same surface?
# --------------------------------------------------------------------------
# ROUND 3. The user marked four defects in the routed kitchen output and all
# four are one mechanism, which the architecture's own §2 claim did not survive:
#
#   "Geometry and focus are co-diagnostic: the layer a frame gets wrong is, by
#    construction, the layer it is defocused in, so the focus contest discards
#    it without being told to."
#
# That holds only if BOTH members observe the SAME SURFACE at the pixel. Where
# parallax has swung an occluder, they do not — one member sees the foreground
# and the other sees the background it uncovered — and then the focus contest
# is not comparing two renderings of one surface, it is comparing two different
# objects. It reliably picks the more TEXTURED one, which is not the nearer one.
# Measured on the kitchen: in the box where the background pot renders in front
# of the Lubriderm bottle, the pair's near-layer mask covers 0.0% of the box,
# because the bottle's own surface there is smooth white and loses the focus
# contest to sharp print on a pot 2 m behind it. Every evidence-driven gate in
# the module is downstream of that mask, so all of them were blind together —
# F108's wall, recurring one level in.
#
# The precondition is testable without texture, and PLAYBOOK §0 supplies the
# test in one line: DEFOCUS IS A LOW-PASS, ALWAYS. Two observations of one
# surface must therefore agree once both are low-passed past their own defocus,
# no matter how textureless the surface is; two different surfaces do not, and
# their disagreement is the contrast between them. So the member is admitted
# only where its low-passed appearance agrees with the REFERENCE's — the
# reference being the one frame that is unwarped and therefore the authority on
# what is visible in the composite's own geometry.
#
# Trinary, never ramped (F106): a pixel is served by a member, or it is not.
#
# What "agree" is allowed to mean is set by measurement, not by taste, and every
# term is something already measured elsewhere in the arc:
#   * a residual displacement of up to GATE_TOL px is the module's own statement
#     of a fit that verified, so a disagreement a GATE_TOL shift explains is not
#     evidence of a different surface — it is subtracted as `tol * |grad|`. This
#     is F106's unexplained-motion rule asked per pixel: refuse what no motion
#     the geometry admits can account for.
#   * `normalize_exposure` leaves a per-frame multiplicative residual; measured
#     on this sweep the largest is 1.85%, so 2% of the local level is allowed.
#   * sensor noise survives the low-pass at well under one level (measured p99
#     0.6-0.9 for sigma 3-8), so one level is allowed for it.
SURFACE_SIGMA = 4.0          # px; must exceed the residual defocus difference
SURFACE_GAIN = 0.02          # share of level, from normalize_exposure's residual
SURFACE_NOISE = 1.0          # levels, from the blurred noise floor


def _lowpass(image, sigma=SURFACE_SIGMA):
    return cv2.GaussianBlur(image.astype(np.float32), (0, 0), sigma)


def _gradient_magnitude(blurred):
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return np.sqrt(gx * gx + gy * gy)


def same_surface(member, reference, sigma=SURFACE_SIGMA, tol=GATE_TOL):
    """Where does `member` observe the same surface as the unwarped `reference`?

    Both are already in the composite's (reference's) geometry. Returns a bool
    mask: True where the low-passed appearances agree to within what a GATE_TOL
    displacement, the exposure residual and sensor noise can explain; False
    where they do not, which means the member is showing different content —
    an occluder that moved, or its own copy of one, standing where the
    composite's geometry says something else is visible.

    Known-answer tested in `tests/test_twoframe_route.py`: pure defocus (a disk
    blur, which is what real defocus is) must NOT trip it, a sub-pixel shift
    must not trip it, and a displaced occluder must.
    """
    member_low = _lowpass(member, sigma)
    reference_low = _lowpass(reference, sigma)
    disagreement = np.abs(member_low - reference_low).max(axis=2)
    explained = (tol * np.maximum(_gradient_magnitude(member_low),
                                  _gradient_magnitude(reference_low)).max(axis=2)
                 + SURFACE_GAIN * np.maximum(member_low, reference_low).max(axis=2)
                 + SURFACE_NOISE)
    return disagreement <= explained


def _pair_refusal(frames, ref, table, dense, depth_step, shape):
    """Per-frame usable masks for one pair, by F82's rule.

    Each frame of the pair is rendered with one rigid translation, so it has no
    internal disagreement to detect and its own layer is placed exactly. The
    disocclusion is still real, and it is the failure this architecture is most
    exposed to: the background frame's NEAR layer sits ~19 px from where the
    composite puts the near layer, so the strip the near object vacated in that
    frame is still covered by its own copy of the object. Measured on the kitchen
    bottle it renders a white ribbon of displaced bottle beside the correct one.

    The step evidence is taken from the PAIR ITSELF — the boundary of the focus
    contest between the two frames — unioned with the usual depth-step probe. A
    layer boundary IS a depth discontinuity by construction, and unlike the guided
    depth map it cannot ramp smoothly across the junction (F104/F108).
    """
    h, w = shape
    boundary = np.zeros((h, w), np.uint8)
    if len(dense) > 1:
        probe = np.ones((5, 5), np.uint8)
        first = dense[0].astype(np.uint8)
        boundary = (cv2.dilate(first, probe) - cv2.erode(first, probe)).astype(np.uint8)
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
# The architecture.
# --------------------------------------------------------------------------
def twoframe_stack(images, ref=None, harden=0.5, refusal=True, gate=True,
                   surface=True, inject=None):
    """Per-region two-frame fusion of a focus stack. Returns (fused, info).

    `harden`   passed to `fuse_coherent` for each pair, where it smooths the
               guided weights BEFORE the one-hot member decision. Pair members
               are misregistered by design outside their own layer, so the
               member choice is geometric and must be hard (F106) — per-band
               soft fusion is the mechanism that produced F112's aliases.
    `refusal`  withhold each member where the pair's own layer boundary says its
               observation does not exist (F82 disocclusion), falling back to the
               other member and, where both are refused, to the unwarped
               reference. Refusal with no fallback is measurably worse than none.
    `surface`  admit each member only where its low-passed appearance agrees
               with the unwarped reference's, i.e. only where the two are
               observing the SAME SURFACE. This is the precondition the focus
               contest needs and never had; see `same_surface`.
    `gate`     forward-verify every layer geometry before applying it. A
               CONTRADICTED fit refuses its member; an UNVERIFIABLE one declines
               the correction and keeps the global stage's geometry.
    `inject`   {(pair_index, layer): (dx, dy)} — a deliberate error added to a
               fitted shift so the gate can be tested against a known answer. Test
               hook only; the runtime never passes it.

    The ablation knobs the research prototype carried (fit=ecc, blanket rigid or
    affine layer geometry, hard-paste stitch, mask erosion, oracle shifts) are NOT
    ported. Each was measured and settled in F110; the configuration below is the
    one that was accepted, and re-opening any of them means re-running the
    research harness, not adding a parameter here.
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
        # the DIFFERENCE between its two layer motions says how far its near layer
        # swung across its far layer — exactly the width of the strip it cannot
        # legitimately supply (F82).
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
                base = (np.eye(3) if warps[frame] is None
                        else A._homogeneous(warps[frame]))
                indices = evidence.indices_in(dense[layer] & support & common)
                fitted = masked_translation(ref_gray, grays[frame], mask, max_shift)
                ecc = (None if fitted is None
                       else (float(fitted[0, 2]), float(fitted[1, 2])))

                # CROSS-LAYER entries are not geometries and are never rendered.
                # They exist only to give F82 its disocclusion width — the
                # DIFFERENTIAL between a frame's two layer motions, the one
                # quantity here with a closed form and the one the masked ECC was
                # known-answer tested against (87-100% of the analytic parallax).
                # Routing them through the edge fit and the gate collapsed that
                # differential and with it the refusal that closes the streak
                # (measured: kitchen flank 1.96 -> 3.3). Different question,
                # different instrument, each used where it was validated.
                if frames[layer] != frame:
                    table[(frame, layer)] = ecc
                    verdicts[(frame, layer)] = ("verified", 0.0, "cross-layer (ecc)")
                    continue

                # PROPOSALS. Nothing is chosen yet; each is a hypothesis the
                # verification below has to prefer before it is applied.
                shift_proposal = ecc
                if ecc is not None:
                    refined, _fit = edge_refined_shift(evidence, frame, indices, ecc)
                    if refined is not None:
                        shift_proposal = refined
                if inject and (index, layer) in inject and shift_proposal is not None:
                    bump = inject[(index, layer)]
                    shift_proposal = (shift_proposal[0] + bump[0],
                                      shift_proposal[1] + bump[1])

                # A rendering matrix maps reference coordinates onto the ORIGINAL
                # frame; the evidence lives on the GLOBALLY ALIGNED grays. So each
                # candidate carries both: `matrix` for the single resample, and
                # `base^-1 @ matrix` for verification.
                inverse = np.linalg.inv(base)
                geometries = []
                if shift_proposal is not None:
                    composed = base @ np.array([[1.0, 0.0, shift_proposal[0]],
                                                [0.0, 1.0, shift_proposal[1]],
                                                [0.0, 0.0, 1.0]])
                    rigid = _rigidify(composed, fit_masks[layer], (h, w))
                    for name, candidate in (("affine", composed), ("rigid", rigid)):
                        geometries.append((name, shift_proposal, candidate,
                                           inverse @ candidate))

                # VERIFICATION-DRIVEN CHOICE between the composed global affine
                # and its rigid collapse, by the focal-weighted RMS of what is
                # STILL displaced at the layer's own material edges once the map
                # is applied. The RAW rms, not a solved translation, because a
                # composed affine is typically right in the middle of a layer and
                # wrong at its ends, and a solved translation absorbs that away.
                # Both scene-splits this arc hit are decided here in opposite
                # directions with no threshold: the factory picks rigid, the
                # kitchen picks affine, and it picks per layer.
                chosen, chosen_label = None, "no fit"
                if geometries:
                    best = None
                    for label, s, matrix, candidate_probe in geometries:
                        check = evidence.residual_translation(frame, indices,
                                                              candidate_probe)
                        if check is None:
                            continue
                        if best is None or check["rms"] < best[0] - 1e-6:
                            best = (check["rms"], label, s, matrix, candidate_probe)
                    if best is not None:
                        chosen_label = f"{best[1]} rms {best[0]:.2f}"
                        chosen = best[2:]
                    else:
                        label, s, matrix, candidate_probe = geometries[0]
                        chosen, chosen_label = (s, matrix, candidate_probe), label

                shift, matrix = (None, base) if chosen is None else chosen[:2]
                verdict = ("verified", 0.0, "ungated")
                if gate and chosen is not None:
                    candidate_probe = chosen[2]
                    verdict = gate_shift(evidence, frame, indices, candidate_probe)
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
                        # the floor, not the plan: refusal alone took the eroded-
                        # mask disaster from 0.668 to 0.916, one repair to 0.957.
                        repaired = _repair(evidence, frame, indices, candidate_probe,
                                           shift, max_shift)
                        if repaired is None:
                            break
                        shift, candidate_probe = repaired
                        matrix = base @ candidate_probe
                        status, statistic, reason = gate_shift(
                            evidence, frame, indices, candidate_probe)
                        verdict = (status, statistic, f"repaired -> {reason}")
                    if verdict[0] != "verified":
                        # CONTRADICTED: the evidence says the layer is still
                        # displaced after the correction — the observation is
                        # unusable and the member is refused below. UNVERIFIABLE:
                        # there is no evidence either way, so the correction is
                        # simply not made (the frame keeps the global stage's
                        # geometry, the baseline everything already accepts) but
                        # the observation itself is not thrown away. PLAYBOOK §0's
                        # own question, asked of a fit: is the evidence ABSENT, or
                        # is it against you?
                        shift, matrix = None, base
                table[(frame, layer)] = shift
                geometry[(frame, layer)] = matrix
                verdicts[(frame, layer)] = (verdict[0], verdict[1],
                                            f"{chosen_label}; {verdict[2]}")

        rendered, valids, shifts = [], [], []
        refused_member = []
        for layer, frame in enumerate(frames):
            shifts.append(table[(frame, layer)])
            matrix = geometry[(frame, layer)]
            # A member whose OWN layer fit failed verification may not supply that
            # layer at all: falling back to the global affine would apply a
            # geometry nothing measured, which is the silent invention the gate
            # exists to stop. Refuse the member and let the reference cover it
            # (F82 — refused zones become reference-quality).
            refused_member.append(frame != ref
                                  and verdicts[(frame, layer)][0] == "contradicted")
            # `geometry` holds the ONE homogeneous matrix chosen for this
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

        # THE PRECONDITION, per member: a member may only be consulted where it
        # observes the same surface the reference does. The reference member
        # (rendered through the identity) agrees with itself everywhere, so this
        # never leaves a pair with nothing — and where it does, the reference
        # fallback below is what covers it.
        agreement = None
        if surface:
            agreement = [np.ones((h, w), bool) if frame == ref
                         else same_surface(image, images[ref])
                         for image, frame in zip(rendered, frames)]

        usable = None
        if len(rendered) == 1:
            if refused_member[0]:
                fused = images[ref]
            elif agreement is not None and not agreement[0].all():
                # A degenerate one-frame region still has a second observation
                # available: the unwarped reference. Admitted exactly where the
                # elected frame is showing something else, and nowhere else.
                # The fallback is carried in `usable` as one entry MORE than the
                # pair has frames, which is the convention `twoframe_fullres`
                # already reads to re-append the reference at native scale.
                usable = [agreement[0], ~agreement[0]]
                rendered = [rendered[0], images[ref]]
                fused = fuse_coherent(rendered, harden=harden, usable=usable)
            else:
                fused = rendered[0]
        else:
            if refusal or surface or any(refused_member):
                usable = (_pair_refusal(frames, ref, table, dense, depth_step, (h, w))
                          if refusal else [np.ones((h, w), bool) for _ in frames])
                if agreement is not None:
                    usable = [m & a for m, a in zip(usable, agreement)]
                for layer, is_refused in enumerate(refused_member):
                    if is_refused:
                        usable[layer] = np.zeros((h, w), bool)
                if ref not in frames:
                    # With only two sources, refusal has nowhere to fall back to:
                    # blocking one member forces the other, which at a disocclusion
                    # ribbon is the frame whose near layer is displaced. The
                    # REFERENCE is the geometry of last resort — unwarped, so its
                    # content is correctly placed by definition, merely defocused.
                    # Admitted ONLY where both members are refused, so the
                    # architecture stays two-frame everywhere it is confident.
                    rendered.append(images[ref])
                    usable.append(~usable[0] & ~usable[1])
            fused = fuse_coherent(rendered, harden=harden, usable=usable)
        candidates.append(fused)
        valid_here = np.logical_and.reduce(valids)
        refused = 0.0
        if usable is not None:
            refused = float(np.mean([1.0 - m[owned].mean()
                                     for m in usable[:len(frames)]]))
        diagnostics.append({"pair": pair, "shifts": shifts, "refused": refused,
                            "verdicts": verdicts, "gated": list(refused_member),
                            "frames": list(frames), "usable": usable,
                            "matrices": [geometry[(f, l)] for l, f in enumerate(frames)],
                            "area": float(owned.mean()), "valid": valid_here})

    for diagnostic in diagnostics:
        common &= diagnostic["valid"]

    fused = (candidates[0] if len(candidates) == 1
             else multiband_blend(candidates, stitch_weights))

    x0, y0, x1, y1 = A._largest_valid_rectangle(common)
    # The largest displacement any elected member had to be moved by. This is the
    # architecture's own report on whether it is refining or re-registering, and
    # it is the second half of the routing rule (see SHIFT_LICENCE_FRACTION).
    applied = [float(np.hypot(s[0], s[1])) for d in diagnostics for s in d["shifts"]
               if s is not None]
    biggest = max(applied) if applied else 0.0
    # Measured at the ANALYSIS resolution, and compared there, so the full-res
    # transfer inherits the verdict rather than re-deriving it in native pixels.
    licence = shift_licence((h, w))
    info = {"tiles": tiles, "pairs": kept, "owner": owner, "peak": peak,
            "crop": (x0, y0, x1, y1), "diagnostics": diagnostics,
            "frames_used": sorted(used),
            "refusals": sum(sum(d["gated"]) for d in diagnostics),
            "max_layer_shift": biggest, "shift_licence": licence,
            "within_licence": biggest <= licence}
    return fused[y0:y1, x0:x1].copy(), info


def twoframe_fullres(natives, working_width=WORKING_WIDTH, ref=None, **kwargs):
    """Native-resolution two-frame fusion, estimated at working scale (F107).

    Every constant in this module is in pixels and was validated at ~800-1100 px,
    so running the ANALYSIS at 24 MP puts every measurement outside its regime.
    F107's rule — estimate small, apply native — transfers here more cleanly than
    anywhere else in the pipeline, because each candidate's geometry is ONE
    homogeneous matrix rather than a sampled field, and scaling a matrix is exact:

        M_native = S @ M_small @ S^-1,   S = diag(s, s, 1)

    no resize of a coordinate map, no interpolation of a field, no second
    resample. Ownership and usable masks are per-pixel decisions, not geometry, so
    they scale nearest-neighbour exactly as `fullres_apply` scales its occlusion
    masks. NATIVE pixels are resampled exactly once.
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
    harden = kwargs.get("harden", 0.5)

    def blow_up(mask):
        return cv2.resize(mask.astype(np.uint8), (nw, nh),
                          interpolation=cv2.INTER_NEAREST) > 0

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
                          else fuse_coherent(rendered, harden=harden, usable=usable))

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
    info["scale"] = scale
    info["working_crop"] = info["crop"]
    info["working_fused"] = working_fused
    info["crop"] = (x0, y0, x1, y1)
    return fused[y0:y1, x0:x1].copy(), info


def fuse_twoframe(images, ref=None, harden=0.5, working_width=WORKING_WIDTH):
    """Entry point for the runtime route: two-frame fusion at any resolution.

    Below `working_width` the analysis and the render are the same resolution;
    above it the analysis runs downscaled and its per-candidate matrices are
    carried to native pixels by conjugation (F107), because every constant in this
    module is pixel-scaled and only validated at ~800-1100 px.
    """
    if images[0].shape[1] > working_width:
        return twoframe_fullres(images, working_width=working_width, ref=ref,
                                harden=harden)
    return twoframe_stack(images, ref, harden=harden)
