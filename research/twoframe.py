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
# The architecture.
# --------------------------------------------------------------------------
def twoframe_stack(images, ref=None, refusal=True, stitch="multiband",
                   step_evidence="both", fallback=True, erode=0, report=False):
    """Per-region two-frame fusion of a focus stack. Returns (fused, info)."""
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
        for frame in frames:
            for layer, mask in enumerate(fit_masks):
                if frame == ref:
                    table[(frame, layer)] = (0.0, 0.0)
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
                residual = masked_translation(ref_gray, grays[frame], mask, max_shift)
                table[(frame, layer)] = (None if residual is None
                                         else (float(residual[0, 2]), float(residual[1, 2])))

        rendered, valids, shifts = [], [], []
        for layer, frame in enumerate(frames):
            base = (np.eye(3) if warps[frame] is None else A._homogeneous(warps[frame]))
            own = table[(frame, layer)]
            shifts.append(own)
            if own is None:
                matrix = base
            else:
                # coarse(x) = original(global @ x) and the residual maps
                # reference coordinates onto coarse ones, so the composed matrix
                # samples the ORIGINAL frame exactly once (PLAYBOOK §0).
                residual = np.array([[1.0, 0.0, own[0]], [0.0, 1.0, own[1]],
                                     [0.0, 0.0, 1.0]])
                matrix = base @ residual
            map_x, map_y = A._blended_coordinate_maps(
                [matrix], [np.ones((h, w), np.float32)], (h, w))
            rendered.append(cv2.remap(images[frame], map_x, map_y, cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0))
            valids.append(cv2.remap(np.full((h, w), 255, np.uint8), map_x, map_y,
                                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=0) == 255)
            used.add(frame)

        if len(rendered) == 1:
            fused = rendered[0]
        else:
            usable = None
            if refusal:
                usable = _pair_refusal(frames, ref, table, dense, depth_step,
                                       (h, w), step_evidence)
                if fallback and ref not in frames:
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

    for refuse in (False, True):
        fused, info = twoframe_stack(frames, P.REFERENCE, refusal=refuse)
        a, b, c, d = info["crop"]
        label = "two-frame + refusal" if refuse else "two-frame (no refusal)"
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
    {"kat": kat, "factory": factory, "oracle": oracle, "variants": variants,
     "kitchen": kitchen}[mode]()
