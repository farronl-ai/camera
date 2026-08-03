"""Physical layer decomposition — round B1 of the scene-model second pass.

Where this sits. Pass 1 (`twoframe.twoframe_stack`) fuses a focus stack by
holding a per-pixel focus CONTEST: for each region it elects two frames and asks,
pixel by pixel, which of them is sharper. `twoframe.layer_masks` returns exactly
that answer (`energies[a] >= energies[b]`), and round A's certifier consumed it as
if it were a scene decomposition. F113 measured the price on the analytic factory:
of a 3.92-level model-error floor, **+2.93 levels is the layer segmentation** —
3.6x every other term combined, and localized (p99 47.6) rather than diffuse.

F112 says why in one sentence: the focus contest picks the more TEXTURED content,
not the NEARER content. Where parallax swung an occluder, the two members see
different objects and the smooth one loses every pixel — the kitchen bottle's
white silhouette lost 100% of its own pixels to the sharp printed pot behind it.
A selector shaped like a decomposition is still a selector.

What this module builds instead. A per-pixel statement, in the reference frame's
geometry, of which DEPTH LAYER owns this pixel's content — with occlusion
ordering at boundaries and an honest trinary state:

    OWNED     one layer's focal signature explains this pixel, and the
              instrument can localize it: certifiable.
    BOUNDARY  the pixel is within the focus operator's own localization limit of
              a layer boundary, or its content changes layer across the sweep.
              A defocused occluder's matte MIXES background onto its boundary
              (F83), so this is not a refusal, it is a statement that ownership
              there is not single-valued.
    UNKNOWN   no focal evidence within reach. Not guessed at.

Nothing here is a new estimator. Every channel has a validated home:

  * the per-pixel focal signature (`twoframe.focal_field`) — F97's doctrine that
    depth grouping comes from the focal signature and not from motion, because
    features at one depth blur together whatever they are doing;
  * Otsu on the focal distribution (`twoframe._otsu_split`) with the repo's own
    acceptance quality and side-weight floors — F98's finding that a bimodal
    distribution whose tails meet has no gap for single-linkage to find;
  * the guided filter (`fusion.guided_filter`) at `depth_from_focus`'s own
    radius, as the edge-aware vote pooling that already serves depth-from-focus;
  * `twoframe.same_surface` (F112) as a CONTEST channel: one surface agrees after
    both observations are low-passed, two do not;
  * `occlusion_order._near_is_low_index` (F83) as the independent check on the
    focal-peak ordering the certifier currently assumes without one.

THE REOPENED QUESTION. PLAYBOOK §0c and F98 record that turning feature groups
into pixel regions was tried twice and lost to valley depth bins. That question is
reopened HERE, for this round, on two conditions the record did not have:
(a) an arbiter now exists — the certifier scores a segmentation directly through
physics, against raw frames in their own geometry, which is what F81a forbade the
old metrics from doing; and (b) the REQUIREMENT changed — pass 1 needed
correction-field SUPPORT (coverage of an object, to hang a motion on), while a
second pass needs content OWNERSHIP (which layer's appearance a pixel is, so the
layer can be assembled). F98's own closing paragraph anticipates the split.

    .venv/bin/python research/layer_decompose.py kat      # factory, vs TRUE masks
    .venv/bin/python research/layer_decompose.py ladder   # the attributed floor
    .venv/bin/python research/layer_decompose.py kat4     # kitchen, KAT-4 re-run
    .venv/bin/python research/layer_decompose.py stats    # trinary statistics
    .venv/bin/python research/layer_decompose.py ablate   # what each channel buys
"""
from __future__ import annotations

import glob
import os
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from focusstack import twoframe as TF  # noqa: E402
from focusstack.focus import focus_measure  # noqa: E402
from focusstack.fusion import guided_filter  # noqa: E402
from focusstack.io import to_gray_float  # noqa: E402
import forward_certify as FC  # noqa: E402
import occlusion_order as OO  # noqa: E402

OUT = os.path.join(HERE, "..", "out", "certify")
KITCHEN = os.path.join(HERE, "data", "mobiledepth", "Figure3", "kitchen")

# --- constants, and where each one comes from -------------------------------
# The focus energy is pooled over a 9x9 box (`focus.focus_measure`'s default,
# used by `content_aware_energies` and therefore by `focal_field`). An
# instrument that pools over 9 px cannot localize a boundary better than its own
# pooling radius, so that radius IS the boundary band's half-width. Not a tuning
# knob: raise the pooling window and this must rise with it.
POOL = 9
BAND = POOL // 2 + 1                     # 5 px half-width
# Vote pooling radius. `fusion.depth_from_focus` already smooths the per-pixel
# focus winner into a depth map with `guided_filter(radius=8, eps=1e-3)`; the
# same construction, on the same signal, is reused here rather than reinvented.
GUIDE_RADIUS, GUIDE_EPS = 8, 1e-3
# Two INDEPENDENT focus operators must agree on a pixel's focal peak before that
# pixel is allowed to vote. The tolerance is half `twoframe.MIN_SEPARATION` (the
# minimum focal separation the pair stage will accept as two layers): a
# disagreement larger than half of that is a disagreement about WHICH LAYER the
# pixel is in, which is exactly the claim being made. F97 measured within-group
# focal spread at 0.30-0.56 frames, so this is ~1.5x the honest noise.
EVIDENCE_TOL = TF.MIN_SEPARATION / 2.0   # 0.75 frames
# At most this many depth layers. Borrowed from `twoframe.MAX_PAIRS`, which caps
# the number of distinct renders pass 1 will maintain for the same reason.
MAX_LAYERS = TF.MAX_PAIRS

OWNED, BOUNDARY, UNKNOWN = 0, 1, 2


# ---------------------------------------------------------------------------
# The decomposition
# ---------------------------------------------------------------------------
@dataclass
class Decomposition:
    labels: np.ndarray              # int32, layer index per pixel (-1 = none)
    state: np.ndarray               # uint8, OWNED / BOUNDARY / UNKNOWN
    masks: list                     # bool, OWNED pixels per layer
    extents: list                   # bool, where each surface EXISTS
    peaks: list                     # focal frame per layer
    order: list                     # nearness proxy, SMALLER is NEARER
    near_is_low: object = None      # the ordering bit, or None if refused
    diag: dict = field(default_factory=dict)

    def fractions(self):
        total = self.state.size
        return tuple(float((self.state == s).sum()) / total
                     for s in (OWNED, BOUNDARY, UNKNOWN))

    def segmentation(self):
        """The `forward_certify.SEGMENTER` payload."""
        return [{"mask": m, "extent": e, "peak": p, "order": o,
                 "name": f"depth{i}@{p:.1f}"}
                for i, (m, e, p, o) in enumerate(
                    zip(self.masks, self.extents, self.peaks, self.order))]


def _subpixel_peak(energies: np.ndarray) -> np.ndarray:
    """Parabolic-interpolated argmax over the frame axis — `focal_field`'s own."""
    n = energies.shape[0]
    winner = np.argmax(energies, axis=0)
    yy, xx = np.indices(winner.shape)
    lo, hi = np.clip(winner - 1, 0, n - 1), np.clip(winner + 1, 0, n - 1)
    a, b, c = energies[lo, yy, xx], energies[winner, yy, xx], energies[hi, yy, xx]
    denominator = a - 2.0 * b + c
    ok = np.abs(denominator) > 1e-9
    offset = np.where(ok, 0.5 * (a - c) / np.where(ok, denominator, 1.0), 0.0)
    return np.clip(winner + np.clip(offset, -0.5, 0.5), 0.0, n - 1.0).astype(np.float32)


def focal_ladder(peak, weight, n, verbose=False):
    """Cut the focal-peak distribution into depth layers by RECURSIVE Otsu.

    Otsu and not gaps (F98): a bimodal distribution whose tails meet has no large
    consecutive gap, so single-linkage — absolute or median-relative — chains
    through it into one clump every time. Recursion, because a real scene has
    more than two depths and the number is not known in advance; the recursion
    stops on the repo's own two conditions rather than on a depth budget:

      * `twoframe.OTSU_MIN_QUALITY` — the split must explain 45% of the subset's
        focal variance, which is pass 1's own bar for "this tile holds two
        layers, not one";
      * `twoframe.MIN_SEPARATION` — the two sides' focal frames must differ by
        1.5 frames, pass 1's own bar for "these are two depths and not one
        depth measured twice".

    `_otsu_split` additionally enforces `MIN_SIDE_WEIGHT` internally, so a split
    that would carve off a sliver is rejected before it is scored.
    """
    thresholds = []

    def recurse(lo, hi, depth):
        if len(thresholds) >= MAX_LAYERS - 1 or depth > MAX_LAYERS:
            return
        inside = (peak >= lo) & (peak < hi) & (weight > 0)
        if inside.sum() < TF.MIN_LAYER_PIXELS:
            return
        values, weights = peak[inside], weight[inside]
        threshold, quality = TF._otsu_split(values, weights, n)
        if threshold is None or quality < TF.OTSU_MIN_QUALITY:
            return
        left, right = values <= threshold, values > threshold
        if left.sum() == 0 or right.sum() == 0:
            return
        mean_left = float(np.average(values[left], weights=weights[left]))
        mean_right = float(np.average(values[right], weights=weights[right]))
        if mean_right - mean_left < TF.MIN_SEPARATION:
            return
        if verbose:
            print(f"      split at {threshold:5.2f} (quality {quality:.3f}, "
                  f"means {mean_left:.2f} | {mean_right:.2f})")
        thresholds.append(float(threshold))
        recurse(lo, threshold, depth + 1)
        recurse(threshold, hi, depth + 1)

    recurse(-1e9, 1e9, 0)
    return sorted(thresholds)


def _mirror_frames(peak, n):
    """Frame pairs equidistant from a layer's focal plane, so blur MATCHES.

    `same_surface` compares two observations after a low-pass, and the low-pass
    only removes a defocus DIFFERENCE up to its own sigma. Comparing a frame
    against the reference therefore asks the test to absorb `|k - ref|` steps of
    blur, which on a 12-frame sweep it cannot. Comparing frame k against its
    mirror `2p - k` about the layer's own focal frame asks it to absorb NOTHING:
    the layer's disk radius is proportional to |k - p| (this is the focal ladder
    the whole instrument is built on), so the two frames blur the layer by the
    same amount by construction, and any residual disagreement is content.
    """
    pairs = []
    for k in range(n):
        mirror = int(round(2.0 * peak - k))
        if mirror <= k or not (0 <= mirror < n):
            continue
        if abs(k - mirror) < 1:
            continue
        pairs.append((k, mirror))
    return pairs


def decompose(images, ref=None, contest=True, band_on=True, verbose=False):
    """Per-pixel depth-layer ownership in the reference frame's geometry."""
    n = len(images)
    ref = n // 2 if ref is None else ref
    h, w = images[0].shape[:2]

    coarse, warps, valid = TF.global_stage(images, ref)
    common = np.logical_and.reduce(valid)
    peak, contrast, energies = TF.focal_field(coarse)

    # --- evidence: two independent operators must name the same focal frame ---
    # `focal_field` routes between the Laplacian and the modified Laplacian by
    # local contrast. Tenengrad (squared gradient) is a different operator with a
    # different texture response and a different noise weighting, so where the
    # two agree on the peak the peak is a property of the SCENE; where they
    # disagree it is a property of the noise. This is §12.1 applied to a signal
    # instead of an instrument: know the answer two ways before believing it.
    greys = [to_gray_float(c) for c in coarse]
    second = np.stack([focus_measure(g, method="tenengrad", smooth_ksize=POOL)
                       for g in greys], 0)
    peak_second = _subpixel_peak(second)
    evidenced = (np.abs(peak - peak_second) <= EVIDENCE_TOL) & common & (contrast > 0)

    # --- the layer ladder, from the evidenced focal distribution --------------
    weight = contrast * evidenced
    thresholds = focal_ladder(peak, weight, n, verbose=verbose)
    edges = [-1e9] + thresholds + [1e9]
    k_layers = len(edges) - 1
    raw = np.zeros((k_layers, h, w), np.float32)
    for i in range(k_layers):
        band = (peak > edges[i]) & (peak <= edges[i + 1]) & evidenced
        raw[i] = weight * band

    # --- edge-aware vote pooling ---------------------------------------------
    # A focus contest is per-pixel and speckles; a depth layer is not speckle.
    # The votes are pooled by the guided filter, guided by the locally-sharpest
    # luminance, which is exactly how `depth_from_focus` turns the same winner
    # index into a depth map: edge-awareness without an edge detector, so the
    # pooled label snaps to the object boundary instead of ramping across it.
    winner = np.argmax(energies, axis=0)
    yy, xx = np.indices((h, w))
    guide = (np.stack(greys, 0)[winner, yy, xx] / 255.0).astype(np.float32)
    pooled = np.stack([guided_filter(guide, raw[i], GUIDE_RADIUS, GUIDE_EPS)
                       for i in range(k_layers)], 0)
    labels = np.argmax(pooled, axis=0).astype(np.int32)

    # --- reach: is there ANY evidenced pixel within the pooling window? -------
    # Threshold-free by construction. A pixel with no evidenced neighbour inside
    # the filter's own support has not been measured, it has been reached for,
    # and F98's warning is that the reaching rule then does the real work.
    span = 2 * GUIDE_RADIUS + 1
    reach = cv2.boxFilter(evidenced.astype(np.float32), cv2.CV_32F,
                          (span, span), normalize=False) > 0.5

    # --- de-speckle, with PASS 1's own rule and PASS 1's own kernels ---------
    # "A focus contest is per-pixel and speckles; a LAYER is not speckle." That
    # sentence and the open-5 / close-9 pair it justifies are already in
    # `forward_certify.model_from_pass1`, applied to the winner map. The rule is
    # right; round A applied it to the wrong signal. A pixel that exactly one
    # cleaned layer claims takes that layer; orphans and contested pixels keep
    # the pooled vote, so the morphology may only REMOVE speckle, never invent a
    # region. Its cost is audited directly on the factory ("misassigned OWNED
    # pixels" in `kat`).
    if k_layers > 1:
        cleaned = []
        for i in range(k_layers):
            m = cv2.morphologyEx((labels == i).astype(np.uint8), cv2.MORPH_OPEN,
                                 np.ones((5, 5), np.uint8))
            cleaned.append(cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                                            np.ones((9, 9), np.uint8)) > 0)
        stack_clean = np.stack(cleaned, 0)
        claims = stack_clean.sum(axis=0)
        unique = claims == 1
        labels = np.where(unique, np.argmax(stack_clean, axis=0), labels).astype(np.int32)

    state = np.full((h, w), OWNED, np.uint8)
    state[~(reach & common)] = UNKNOWN
    labels[state == UNKNOWN] = -1

    # --- the boundary band ----------------------------------------------------
    # Only edges between two REAL layers count. The frame border, where the
    # global warp has no data, is already UNKNOWN; ringing it with a boundary
    # band would be the instrument charging itself twice for one honesty.
    edge = np.zeros((h, w), bool)
    if k_layers > 1:
        probe = np.ones((3, 3), np.uint8)
        for i in range(k_layers):
            m = (labels == i).astype(np.uint8)
            edge |= (cv2.dilate(m, probe) > 0) & (labels >= 0) & (labels != i)
    band = cv2.dilate(edge.astype(np.uint8),
                      np.ones((2 * BAND + 1, 2 * BAND + 1), np.uint8)) > 0
    if band_on:
        state[band & (state == OWNED)] = BOUNDARY

    # --- the contest channel (F112): does the content STAY on this layer? -----
    contested = np.zeros((h, w), bool)
    layer_shift = {}
    if contest:
        contested, layer_shift = _contest(coarse, ref, labels, k_layers, peak,
                                          common, n, verbose=verbose)
        state[contested & (state == OWNED)] = BOUNDARY

    # --- ordering -------------------------------------------------------------
    # The certifier's layer ordering is "lower focal peak = nearer", named in
    # round A as an unguarded assumption. F83's contour cue guards it: an
    # occlusion boundary is the near object's own silhouette, so the focal index
    # READ ON the contour names the occluder's depth, and thousands of contour
    # pixels settle the sweep's single near-is-low bit.
    contour, side_a, side_b, local = OO._occlusion_contours(greys, winner)
    near_is_low = OO._near_is_low_index(contour, side_a, side_b, local)
    peaks, orders, masks = [], [], []
    for i in range(k_layers):
        owned = (labels == i) & (state == OWNED)
        pool_here = (labels == i)
        focal = (float(np.average(peak[pool_here & evidenced],
                                  weights=weight[pool_here & evidenced]))
                 if (pool_here & evidenced).any() else float(ref))
        peaks.append(focal)
        masks.append(owned)
    sign = 1.0 if (near_is_low is None or near_is_low) else -1.0
    orders = [sign * p for p in peaks]

    # --- extents: where the SURFACE exists, visible or not --------------------
    # A backdrop exists behind everything; a foreground does not exist behind the
    # things it hides. Round A's KAT-1b measured what collapsing these costs:
    # the renderer certifies disocclusion it invented by nearest-fill (factory
    # p99 34.4 -> 1.08 once separated). Pass-1's masks carried no extents at all,
    # so every layer's matte was its visible mask; a decomposition can say more.
    rank = np.argsort(np.argsort(orders))          # 0 = nearest
    ambiguous = state != OWNED
    extents = []
    for i in range(k_layers):
        behind_of_nearer = np.zeros((h, w), bool)
        for j in range(k_layers):
            if rank[j] < rank[i]:
                behind_of_nearer |= masks[j]
        extents.append(masks[i] | behind_of_nearer | ambiguous)

    diag = {"coarse": coarse, "warps": warps, "common": common, "peak": peak,
            "contrast": contrast, "energies": energies, "evidenced": evidenced,
            "thresholds": thresholds, "band": band, "contested": contested,
            "layer_shift": layer_shift, "guide": guide, "ref": ref}
    result = Decomposition(labels=labels, state=state, masks=masks,
                           extents=extents, peaks=peaks, order=orders,
                           near_is_low=near_is_low, diag=diag)
    if verbose:
        owned_f, band_f, unknown_f = result.fractions()
        print(f"    {k_layers} depth layers at focal frames "
              f"{[round(p, 2) for p in peaks]}")
        print(f"    evidence: {evidenced.mean() * 100:.1f}% of pixels have a "
              f"two-operator focal peak")
        print(f"    trinary: owned {owned_f * 100:.1f}%  boundary "
              f"{band_f * 100:.1f}%  unknown {unknown_f * 100:.1f}%"
              f"   (contested {contested.mean() * 100:.1f}%)")
        print(f"    ordering bit (F83 occlusion contours): near_is_low="
              f"{near_is_low}; focal order {[round(o, 2) for o in orders]}")
    return result


def _contest(coarse, ref, labels, k_layers, peak, common, n, verbose=False):
    """Demote pixels whose content does not STAY on the layer claiming them.

    F112's mechanism, asked of a decomposition instead of a fusion pair. Fit each
    candidate layer's own translation with the pipeline's validated masked ECC
    (`twoframe.masked_translation`, known-answer tested at -2..-30 px), then take
    the layer's blur-MATCHED mirror frame pairs and ask `same_surface` whether the
    layer's motion transports the pixel's content between them. Where it does not,
    the pixel's content changes layer across the sweep — a parallax swing — and it
    is not owned by anything single.

    The channel ABSTAINS where it cannot see: if two layers' fitted motions differ
    by less than `GATE_TOL`, their transports are indistinguishable and a
    disagreement cannot be attributed (F103's vacuous-consistency rule, which is
    why a feature whose normal is perpendicular to a motion must not join it).
    """
    h, w = labels.shape
    max_shift = TF.MAX_SHIFT_FRACTION * float(np.hypot(h, w))
    ref_grey = to_gray_float(coarse[ref]).astype(np.float32) / 255.0
    greys = [to_gray_float(c).astype(np.float32) / 255.0 for c in coarse]
    gradient = cv2.magnitude(
        cv2.Sobel(ref_grey * 255.0, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(ref_grey * 255.0, cv2.CV_32F, 0, 1, ksize=3))
    textured = gradient >= TF.A._REFINE_MIN_GRADIENT

    shifts = {}
    for i in range(k_layers):
        support = (labels == i) & textured & common
        for k in range(n):
            if k == ref:
                shifts[(k, i)] = (0.0, 0.0)
                continue
            fitted = TF.masked_translation(ref_grey, greys[k], support, max_shift)
            shifts[(k, i)] = ((0.0, 0.0) if fitted is None
                              else (float(fitted[0, 2]), float(fitted[1, 2])))
        # Propagate: a layer is measurable only near its own focal plane (F99),
        # and PLAYBOOK §0's recipe is a focal-weighted slope along the sweep.
        weights = np.exp(-0.5 * ((np.arange(n) - peak_of(labels, peak, i))
                                 / TF.FOCAL_SIGMA) ** 2)
        slope = []
        for axis in (0, 1):
            samples = [shifts[(k, i)][axis] / (k - ref) for k in range(n) if k != ref]
            sw = [weights[k] * abs(k - ref) for k in range(n) if k != ref]
            slope.append(FC._weighted_median(np.array(samples), np.array(sw)))
        for k in range(n):
            shifts[(k, i)] = (slope[0] * (k - ref), slope[1] * (k - ref))

    contested = np.zeros((h, w), bool)
    for i in range(k_layers):
        focal = peak_of(labels, peak, i)
        pairs = _mirror_frames(focal, n)
        if not pairs:
            continue
        # Observability: is this layer's transport distinguishable from another's?
        others = [j for j in range(k_layers) if j != i]
        votes = np.zeros((h, w), np.float32)
        seen = np.zeros((h, w), np.float32)
        for a, b in pairs:
            separated = any(
                max(abs(shifts[(a, i)][0] - shifts[(a, j)][0]),
                    abs(shifts[(a, i)][1] - shifts[(a, j)][1])) > TF.GATE_TOL
                for j in others)
            if not separated:
                continue
            warped = [_translate(coarse[k], -shifts[(k, i)][0], -shifts[(k, i)][1])
                      for k in (a, b)]
            agree = TF.same_surface(warped[0].astype(np.float32),
                                    warped[1].astype(np.float32))
            votes += (~agree).astype(np.float32)
            seen += 1.0
        if seen.max() > 0:
            # Refuse where the layer's own motion fails to transport the content
            # in the MAJORITY of the blur-matched pairs that could see it.
            fraction = votes / np.maximum(seen, 1.0)
            contested |= (labels == i) & (seen > 0) & (fraction > 0.5)
        if verbose:
            print(f"      layer {i} focal {focal:.2f}: {len(pairs)} mirror pairs, "
                  f"shift/frame ({shifts[(0, i)][0] / max(1, ref):+.2f}, "
                  f"{shifts[(0, i)][1] / max(1, ref):+.2f}) px")
    return contested, shifts


def peak_of(labels, peak, i):
    here = labels == i
    return float(np.median(peak[here])) if here.any() else 0.0


def _translate(image, dx, dy):
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
# The SEGMENTER hook, with a cache (the certifier builds its model 3x per ladder)
# ---------------------------------------------------------------------------
_CACHE = {}


def segmenter(verbose=False, **kwargs):
    def build(images, ref):
        key = (id(images), len(images), ref, images[0].shape,
               tuple(sorted(kwargs.items())))
        if key not in _CACHE:
            _CACHE[key] = decompose(images, ref, verbose=verbose, **kwargs)
        return _CACHE[key].segmentation()
    return build


def engage(verbose=False, **kwargs):
    FC.SEGMENTER = segmenter(verbose=verbose, **kwargs)


def disengage():
    FC.SEGMENTER = None
    _CACHE.clear()


# ---------------------------------------------------------------------------
# KAT — the factory, where the TRUE plane masks exist
# ---------------------------------------------------------------------------
def _iou(a, b):
    union = (a | b).sum()
    return float((a & b).sum()) / float(union) if union else float("nan")


def _describe(name, masks, true_near):
    """IoU of each candidate mask against the two TRUE planes."""
    print(f"\n  --- {name}: masks vs the factory's TRUE plane masks ---")
    print(f"  {'layer':<22} {'px':>8} {'IoU near':>9} {'IoU far':>8} "
          f"{'purity':>8}  best")
    true = {"near": true_near, "far": ~true_near}
    for i, mask in enumerate(masks):
        ious = {k: _iou(mask, v) for k, v in true.items()}
        best = max(ious, key=lambda k: ious[k])
        purity = (float((mask & true[best]).sum()) / max(1, int(mask.sum()))
                  if mask.any() else float("nan"))
        print(f"  {i:<22} {int(mask.sum()):8d} {ious['near']:9.3f} "
              f"{ious['far']:8.3f} {purity:8.3f}  {best}")
    # The union view: what fraction of each true plane is OWNED by a layer whose
    # majority is that plane, and how much is owned by the WRONG plane.
    for side, truth in true.items():
        claimed_right = np.zeros_like(truth)
        claimed_wrong = np.zeros_like(truth)
        for mask in masks:
            if mask.sum() == 0:
                continue
            majority_near = float((mask & true_near).sum()) / int(mask.sum()) > 0.5
            target = true_near if majority_near else ~true_near
            same = (target == truth).all()
            (claimed_right if same else claimed_wrong)[:] |= mask
        print(f"  true {side:<5}: {float((claimed_right & truth).sum()) / truth.sum() * 100:5.1f}% "
              f"owned by a {side}-majority layer, "
              f"{float((claimed_wrong & truth).sum()) / truth.sum() * 100:5.1f}% "
              f"owned by the OTHER plane's layer")


def kat() -> None:
    """Known-answer test on the analytic factory, where the true masks exist."""
    import parallax_gen as P

    os.makedirs(OUT, exist_ok=True)
    frames, _truth, true_near = P.build_stack()
    print("=" * 78)
    print("KAT — the decomposition against the factory's TRUE plane masks")
    print("=" * 78)
    print(f"  {len(frames)} frames {frames[0].shape}, ref {P.REFERENCE}; the near "
          f"plane covers {true_near.mean() * 100:.1f}% of the frame\n")

    result = decompose(frames, P.REFERENCE, verbose=True)
    owned_f, band_f, unknown_f = result.fractions()

    # The competing construction, unchanged, on the same frames.
    pass1, _diag = FC.model_from_pass1(frames, P.REFERENCE)
    _describe("PASS-1 masks (the focus contest's winner map)",
              [layer.mask for layer in pass1.layers], true_near)
    _describe("THIS decomposition (owned pixels only)", result.masks, true_near)

    # --- boundary band: does the TRUE boundary lie inside it? ----------------
    probe = np.ones((3, 3), np.uint8)
    m = true_near.astype(np.uint8)
    true_edge = (cv2.dilate(m, probe) - cv2.erode(m, probe)) > 0
    band = result.diag["band"]
    inside = float((true_edge & band).sum()) / max(1, int(true_edge.sum()))
    # Localization: distance from each of OUR boundary pixels to the TRUE one.
    dist_to_true = cv2.distanceTransform((~true_edge).astype(np.uint8), cv2.DIST_L2, 5)
    lab = result.labels
    ours = np.zeros(lab.shape, bool)
    for i in range(len(result.masks)):
        mi = (lab == i).astype(np.uint8)
        ours |= (cv2.dilate(mi, probe) > 0) & (lab >= 0) & (lab != i)
    contour_error = (float(np.median(dist_to_true[ours])) if ours.any()
                     else float("nan"))
    # The band's job is to CONTAIN the true silhouette, so the number that
    # matters is measured from the truth outward, not from us inward.
    dist_to_ours = cv2.distanceTransform((~ours).astype(np.uint8), cv2.DIST_L2, 5)
    need = dist_to_ours[true_edge]
    print(f"\n  --- boundary band (half-width {BAND} px, = the focus operator's "
          f"own pooling radius) ---")
    print(f"  true silhouette inside the band     : {inside * 100:5.1f}%")
    print(f"  true contour -> our contour         : median "
          f"{float(np.median(need)):.2f} px, p90 {float(np.percentile(need, 90)):.2f}, "
          f"p95 {float(np.percentile(need, 95)):.2f}, max {float(need.max()):.2f}")
    print(f"  half-width that would contain 95%   : "
          f"{float(np.percentile(need, 95)):.1f} px  (MEASURED, not adopted — the "
          f"band stays at the operator's own {BAND} px)")
    if ours.any():
        print(f"  our contour -> true contour         : median {contour_error:5.2f} px "
              f"(p90 {float(np.percentile(dist_to_true[ours], 90)):.2f})")
    print(f"  band covers {band.mean() * 100:5.1f}% of the frame")

    # --- error localization: where do the wrong pixels sit? ------------------
    near_layers = [i for i, mask in enumerate(result.masks)
                   if mask.any() and float((mask & true_near).sum()) / mask.sum() > 0.5]
    claimed_near = np.logical_or.reduce(
        [result.masks[i] for i in near_layers]) if near_layers else np.zeros_like(true_near)
    claimed_far = np.logical_or.reduce(
        [result.masks[i] for i in range(len(result.masks)) if i not in near_layers]) \
        if len(near_layers) < len(result.masks) else np.zeros_like(true_near)
    wrong = (claimed_near & ~true_near) | (claimed_far & true_near)
    print(f"\n  --- error localization ---")
    print(f"  misassigned OWNED pixels: {int(wrong.sum())} "
          f"({wrong.mean() * 100:.2f}% of frame)")
    if wrong.any():
        d = dist_to_true[wrong]
        print(f"  their distance to the true silhouette: median {np.median(d):.1f} px, "
              f"p90 {np.percentile(d, 90):.1f} px, max {d.max():.1f} px")
        print(f"  share within one band-width ({BAND} px) of it: "
              f"{float((d <= BAND).mean()) * 100:.1f}%")

    # --- ordering, checked two independent ways ------------------------------
    print(f"\n  --- occlusion ordering ---")
    print(f"  focal-peak proxy (round A's unguarded assumption): "
          f"{[round(p, 2) for p in result.peaks]}")
    print(f"  F83 contour ordering bit near_is_low = {result.near_is_low} "
          f"(truth: near plane focuses at frame {P.NEAR_FOCUS_FRAME}, far at "
          f"{P.FAR_FOCUS_FRAME} -> near IS low)")
    shifts = result.diag["layer_shift"]
    if shifts:
        print(f"  parallax magnitude per layer (independent channel; nearer moves "
              f"more, PLAYBOOK: displacement ~ 1/Z):")
        for i in range(len(result.masks)):
            per_frame = abs(shifts[(0, i)][0]) / max(1, P.REFERENCE)
            print(f"    layer {i} focal {result.peaks[i]:5.2f}: "
                  f"|dx| {per_frame:5.2f} px/frame   "
                  f"(truth: near {P.NEAR_SHIFT_PER_FRAME}, far {P.FAR_SHIFT_PER_FRAME})")

    _save_overlays("factory", frames[P.REFERENCE], result, true_near)
    print(f"\n  trinary: owned {owned_f * 100:.1f}%  boundary {band_f * 100:.1f}%  "
          f"unknown {unknown_f * 100:.1f}%")


def _save_overlays(tag, base, result, true_near=None):
    palette = np.array([[60, 60, 220], [60, 200, 60], [220, 160, 60],
                        [200, 60, 200], [60, 220, 220], [180, 180, 180]], np.uint8)
    colour = np.zeros(base.shape, np.uint8)
    for i in range(len(result.masks)):
        colour[result.labels == i] = palette[i % len(palette)]
    colour[result.state == BOUNDARY] = (255, 255, 255)
    colour[result.state == UNKNOWN] = (0, 0, 0)
    grey = cv2.cvtColor(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    blend = (0.45 * colour.astype(np.float32) + 0.55 * grey.astype(np.float32))
    if true_near is not None:
        probe = np.ones((3, 3), np.uint8)
        m = true_near.astype(np.uint8)
        edge = (cv2.dilate(m, probe) - cv2.erode(m, probe)) > 0
        blend[edge] = (0, 255, 255)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"decompose_{tag}.png")
    cv2.imwrite(path, np.clip(blend, 0, 255).astype(np.uint8))
    print(f"  overlay -> {path}")


# ---------------------------------------------------------------------------
# The attributed floor, re-run through the new decomposition
# ---------------------------------------------------------------------------
def ladder() -> None:
    print("=" * 78)
    print("ROUND A's LADDER, unchanged, with pass-1's masks replaced")
    print("=" * 78)
    engage(verbose=True)
    try:
        FC.floor_factory()
    finally:
        disengage()


def ladder_control() -> None:
    """The same ladder with the boundary band OFF — the coverage control.

    A segmentation can always lower an average residual by declining to be
    scored where it is weak, and the honest ladder above certifies 72% of the
    frame where pass-1's masks certified 81%. So the claim has to be made twice:
    once with the trinary honesty on, and once with EVERY labelled pixel owned,
    which restores the coverage and removes the escape route. If the term only
    falls in the first version, the decomposition is not better, it is quieter.
    """
    print("=" * 78)
    print("COVERAGE CONTROL — same labels, boundary band OFF, every pixel owned")
    print("=" * 78)
    engage(verbose=True, band_on=False, contest=False)
    try:
        FC.floor_factory()
    finally:
        disengage()


def kat4() -> None:
    print("=" * 78)
    print("KAT-4 re-run with the physical decomposition feeding the certifier")
    print("=" * 78)
    engage(verbose=True)
    try:
        FC.kat4()
    finally:
        disengage()


def stats() -> None:
    """Trinary ownership statistics on both scenes — round B2's input contract."""
    import parallax_gen as P
    from focusstack.io import normalize_exposure

    print("=" * 78)
    print("TRINARY OWNERSHIP STATISTICS")
    print("=" * 78)
    print(f"  {'scene':<12} {'layers':>7} {'owned':>8} {'boundary':>9} "
          f"{'unknown':>8} {'evidenced':>10} {'contested':>10}  ordering")
    rows = []
    frames, _t, near = P.build_stack()
    rows.append(("factory", frames, P.REFERENCE))
    paths = sorted(glob.glob(os.path.join(KITCHEN, "*.jpg")))
    src = [cv2.imread(p) for p in paths]
    rows.append(("kitchen", normalize_exposure(src), len(src) // 2))
    results = {}
    for name, images, ref in rows:
        result = decompose(images, ref)
        results[name] = result
        owned_f, band_f, unknown_f = result.fractions()
        print(f"  {name:<12} {len(result.masks):7d} {owned_f * 100:7.1f}% "
              f"{band_f * 100:8.1f}% {unknown_f * 100:7.1f}% "
              f"{result.diag['evidenced'].mean() * 100:9.1f}% "
              f"{result.diag['contested'].mean() * 100:9.1f}%  "
              f"near_is_low={result.near_is_low}")
        _save_overlays(name, images[ref], result,
                       near if name == "factory" else None)
    print("\n  per-layer detail")
    for name, result in results.items():
        print(f"  {name}:")
        for i, mask in enumerate(result.masks):
            total = int((result.labels == i).sum())
            print(f"    layer {i}: focal {result.peaks[i]:5.2f}  order "
                  f"{result.order[i]:+6.2f}  labelled {total:7d} px  owned "
                  f"{int(mask.sum()):7d} px "
                  f"({100.0 * mask.sum() / max(1, total):5.1f}% of its label)")


def ablate() -> None:
    """What each channel buys, on the factory, against the true masks."""
    import parallax_gen as P

    frames, _truth, true_near = P.build_stack()
    print("=" * 78)
    print("ABLATION — factory IoU against the true planes")
    print("=" * 78)
    print(f"  {'configuration':<40} {'layers':>7} {'IoU near':>9} {'IoU far':>8} "
          f"{'owned':>7} {'bnd':>6} {'unk':>6}")
    for label, kwargs in (("full", {}),
                          ("no same-surface contest", {"contest": False}),
                          ("no boundary band (coverage control)",
                           {"band_on": False, "contest": False})):
        result = decompose(frames, P.REFERENCE, **kwargs)
        near_layers = [i for i, m in enumerate(result.masks)
                       if m.any() and float((m & true_near).sum()) / m.sum() > 0.5]
        cn = (np.logical_or.reduce([result.masks[i] for i in near_layers])
              if near_layers else np.zeros_like(true_near))
        cf = (np.logical_or.reduce([result.masks[i] for i in range(len(result.masks))
                                    if i not in near_layers])
              if len(near_layers) < len(result.masks) else np.zeros_like(true_near))
        o, b, u = result.fractions()
        print(f"  {label:<40} {len(result.masks):7d} {_iou(cn, true_near):9.3f} "
              f"{_iou(cf, ~true_near):8.3f} {o * 100:6.1f}% {b * 100:5.1f}% "
              f"{u * 100:5.1f}%")


COMMANDS = {"kat": kat, "ladder": ladder, "control": ladder_control,
            "kat4": kat4, "stats": stats, "ablate": ablate}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print("commands: " + ", ".join(COMMANDS))
        return
    COMMANDS[sys.argv[1]]()


if __name__ == "__main__":
    main()
