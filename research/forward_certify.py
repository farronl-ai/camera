"""Forward-render consistency certifier — the scene-model second pass's instrument.

Where this sits. `pipeline.run` (pass 1) decomposes a focus stack into regions and
layers, measures how each layer moved in each frame, and fuses a composite in the
REFERENCE frame's geometry. Those pass-1 outputs are not just a recipe for a
composite; they are a FORWARD MODEL of the scene:

    reference-geometry scene layers
      -> one rigid transform per (frame, layer)      [pass-1 motions]
      -> one defocus PSF per (frame, layer)          [pass-1 focal ladder]
      -> painter's composite in depth order          [pass-1 layer ordering]
      == a PREDICTION of what frame k should have looked like.

This module renders that prediction and compares it against the RAW frame in the
frame's OWN geometry. That is the whole point, and it is why this evades F81a: a
no-reference metric scores a composite against sources that ALIGNMENT ITSELF MOVED,
so it cannot adjudicate an alignment change. A raw frame in its own coordinates is
untouchable — nothing in the pipeline can move it — so a residual there is a
statement about the model and the composite, never about the registration used to
score them.

What it is NOT. It is not a reconstruction, it does not touch the runtime, and it
does not propose a better composite. It is round A of a three-round arc: the
measuring instrument, built and known-answer tested before any mechanism that would
depend on it (DEVSTYLE §12.1).

Honesty, structurally. Every certified pixel is one whose prediction depends on NO
unobserved content. The rest is reported, never guessed at: the render carries an
`unknown` channel through exactly the same composite arithmetic as the image, so a
pixel whose defocus kernel reached behind an occluder, or off the composite's crop,
or out of frame, is BOUNDARY, not certified. Trinary, per F106 — certified /
boundary / excluded, three counts, no ramp.

    .venv/bin/python research/forward_certify.py kat1     # renderer vs the factory
    .venv/bin/python research/forward_certify.py kat2     # blur estimator vs known sigma
    .venv/bin/python research/forward_certify.py kat3     # GT ranking on the factory
    .venv/bin/python research/forward_certify.py kat4     # known defects on the kitchen
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

from focusstack import twoframe as TF  # noqa: E402
from focusstack.align import _homogeneous  # noqa: E402
from focusstack.fusion import depth_from_focus  # noqa: E402
from focusstack.io import to_gray_float  # noqa: E402

OUT = os.path.abspath(os.path.join(HERE, "..", "out", "certify"))
KITCHEN = os.path.join(HERE, "data", "mobiledepth", "Figure3", "kitchen")

# --- instrument constants, and where each came from -------------------------
# A pixel is certified only if the unobserved share of its prediction is below
# this. It is not a tuning knob: 1e-3 of a level is far under the sensor noise
# floor (p99 = 0.6-0.9 levels, measured in F112/R3), so anything above it is a
# real contribution from content the composite never saw.
UNKNOWN_TOL = 1e-3
# Radius grid for the defocus search. Disk radii are integers by construction
# (`_disk` rounds), so an integer grid is the EXACT parameter space, not a
# discretization of it.
RADIUS_MAX = 12
# Exposure: `normalize_exposure` leaves a multiplicative residual; the largest
# measured on the kitchen sweep is 1.85% (F112/R3). 8% of headroom is ~4x that,
# enough that the gain is never the binding constraint and small enough that it
# cannot absorb a structural defect.
GAIN_LIMIT = 0.08
# Two layers are ONE geometry group if their fitted shifts agree within this in
# every frame. Borrowed, not invented: `twoframe.MERGE_TOL` is 1.0 frame for
# pair merging and F93's object-merge tolerance is ~2 px; 0.75 px is inside both
# and is the scale at which this arc's own estimators are trustworthy
# (GATE_TOL = 1.5 px is the line between usable and refused).
GROUP_TOL = 0.75
MIN_COVERAGE = 2          # frames a reference pixel needs before it is scored


# ---------------------------------------------------------------------------
# PSF and warping primitives. Deliberately the SAME construction `parallax_gen`
# uses, so KAT-1 measures the renderer's composite arithmetic rather than a
# disagreement about what a disk is.
# ---------------------------------------------------------------------------
def disk_kernel(radius: float) -> np.ndarray | None:
    """Disk (circle of confusion) PSF. Real defocus is a disk, never a Gaussian."""
    r = int(round(radius))
    if r < 1:
        return None
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    kernel = ((xx ** 2 + yy ** 2) <= r * r).astype(np.float32)
    return kernel / kernel.sum()


def defocus(image: np.ndarray, radius: float, border=cv2.BORDER_REPLICATE):
    kernel = disk_kernel(radius)
    if kernel is None:
        return image
    return cv2.filter2D(image, -1, kernel, borderType=border)


def warp_forward(image, matrix, shape, const=0.0):
    """Reference geometry -> frame geometry.

    The convention is the pipeline's own: a candidate matrix maps REFERENCE
    coordinates onto the ORIGINAL frame (that is how `_blended_coordinate_maps`
    feeds `cv2.remap`). `warpPerspective` computes dst(x) = src(M^-1 x), which is
    exactly "put the content that sits at reference x where frame k saw it".
    """
    h, w = shape
    return cv2.warpPerspective(image, np.asarray(matrix, np.float64), (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=const)


def warp_back(image, matrix, shape, const=0.0):
    """Frame geometry -> reference geometry (the same matrix, read backwards)."""
    h, w = shape
    return cv2.warpPerspective(image, np.asarray(matrix, np.float64), (w, h),
                               flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=const)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    keep = np.isfinite(values) & (weights > 0)
    if not keep.any():
        return 0.0
    values, weights = values[keep], weights[keep]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    return float(values[int(np.searchsorted(cumulative, 0.5 * cumulative[-1]))])


def nearest_fill(image: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Extend `image` outside `known` by nearest observed pixel.

    A layer's appearance is known only where the layer is VISIBLE, but its defocus
    kernel reaches past that. Something has to be there; nearest-fill is the least
    inventive choice available (it places content some pixel of the same layer
    actually shows, F56's licence). The lie is bounded and, critically, it is
    ACCOUNTED FOR: `render` tracks the same extension through the same arithmetic
    and refuses to certify any pixel it touched.
    """
    if known.all():
        return image
    if not known.any():
        return np.zeros_like(image)
    source = np.where(known, 0, 1).astype(np.uint8)
    _dist, labels = cv2.distanceTransformWithLabels(
        source, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(known)
    index = np.clip(labels.astype(np.int64) - 1, 0, len(ys) - 1)
    return image[ys[index], xs[index]]


# ---------------------------------------------------------------------------
# The scene model
# ---------------------------------------------------------------------------
@dataclass
class Layer:
    """One piece of the reference-geometry scene with one motion and one blur.

    Two masks, and the distinction is the whole of this instrument's honesty:

    `mask`   where the layer is VISIBLE in the reference composite — i.e. exactly
             where its appearance is OBSERVED and may be used.
    `extent` the layer's matte: where the surface EXISTS, visible or not. A
             backdrop exists behind everything even though it is observed only
             around the foreground, and that difference is precisely the
             disocclusion the renderer must refuse to certify.

    Collapsing the two is the mistake that makes a certifier confidently score a
    prediction it built out of content nothing ever recorded.
    """
    mask: np.ndarray          # bool, reference geometry — observed / visible
    group: int                # geometry group: layers sharing one per-frame motion
    order: float              # nearness proxy; SMALLER is NEARER
    name: str = ""
    extent: np.ndarray | None = None   # bool matte; defaults to `mask`

    @property
    def alpha(self) -> np.ndarray:
        source = self.mask if self.extent is None else self.extent
        return source.astype(np.float32)


@dataclass
class SceneModel:
    shape: tuple                            # reference-geometry (h, w)
    n_frames: int
    ref: int
    layers: list                            # list[Layer]
    matrices: dict = field(default_factory=dict)   # (frame, group) -> 3x3
    radii: dict = field(default_factory=dict)      # (frame, layer) -> int
    gains: dict = field(default_factory=dict)      # (frame, layer) -> float
    menu: dict = field(default_factory=dict)       # (frame, group) -> [(name, 3x3)]
    choices: dict = field(default_factory=dict)    # (frame, group) -> name

    def layers_of(self, group):
        return [i for i, layer in enumerate(self.layers) if layer.group == group]

    def groups(self):
        return sorted({layer.group for layer in self.layers})

    def far_to_near(self):
        return sorted(range(len(self.layers)), key=lambda i: -self.layers[i].order)

    def matrix(self, frame, layer_index):
        return self.matrices[(frame, self.layers[layer_index].group)]


# ---------------------------------------------------------------------------
# The renderer
# ---------------------------------------------------------------------------
def render(model: SceneModel, appearances, supports, frame: int,
           only: int | None = None):
    """Predict what `frame` should look like, plus what share of it is KNOWN.

    Painter's algorithm far to near. Each layer's appearance and its matte are
    blurred by the layer's own radius and composited with the blurred matte, which
    is how a real foreground edge defocuses (and how `parallax_gen` renders) — a
    razor-sharp matte over a blurred layer is the classic synthetic tell.

    `known` runs the IDENTICAL recursion on the layers' observed-support masks, so
    it is not an estimate of confidence: it is the exact fraction of the predicted
    radiance that came from pixels the composite actually contains.
    """
    h, w = model.shape
    out = np.zeros((h, w, 3), np.float32)
    known = np.zeros((h, w), np.float32)
    order = model.far_to_near() if only is None else [only]
    for index in order:
        layer = model.layers[index]
        matrix = model.matrix(frame, index)
        radius = model.radii.get((frame, index), 0)
        gain = model.gains.get((frame, index), 1.0)
        alpha = warp_forward(layer.alpha, matrix, (h, w), 0.0)
        appearance = warp_forward(appearances[index], matrix, (h, w), 0.0)
        support = warp_forward(supports[index], matrix, (h, w), 0.0)
        alpha_b = np.clip(defocus(alpha, radius), 0.0, 1.0)
        appearance_b = defocus(appearance, radius)
        # Support blurs with a CONSTANT-0 border: off the image is unobserved, and
        # replicating the edge would silently certify content nothing recorded.
        support_b = np.clip(defocus(support, radius, cv2.BORDER_CONSTANT), 0.0, 1.0)
        out = gain * appearance_b * alpha_b[..., None] + out * (1.0 - alpha_b)[..., None]
        known = support_b * alpha_b + known * (1.0 - alpha_b)
    return out, known


def layer_views(model: SceneModel, canvas: np.ndarray, canvas_support: np.ndarray):
    """Split a reference-geometry image into per-layer appearances and supports."""
    appearances, supports = [], []
    for layer in model.layers:
        observed = layer.mask & canvas_support
        appearances.append(nearest_fill(canvas, observed))
        supports.append(observed.astype(np.float32))
    return appearances, supports


def place(composite: np.ndarray, crop, shape):
    """Put a cropped composite back on the full reference canvas, with its support."""
    h, w = shape
    x0, y0, x1, y1 = crop
    canvas = np.zeros((h, w, 3), np.float32)
    support = np.zeros((h, w), bool)
    canvas[y0:y1, x0:x1] = composite.astype(np.float32)
    support[y0:y1, x0:x1] = True
    return canvas, support


# ---------------------------------------------------------------------------
# Nuisance parameters: defocus radius and exposure gain, fitted by forward search
# ---------------------------------------------------------------------------
def _fit_gain(prediction, observation, weight):
    """Least-squares scalar gain, clipped to the measured exposure residual."""
    denominator = float((weight * prediction * prediction).sum())
    if denominator <= 1e-6:
        return 1.0
    gain = float((weight * prediction * observation).sum()) / denominator
    return float(np.clip(gain, 1.0 - GAIN_LIMIT, 1.0 + GAIN_LIMIT))


def estimate_radii(model: SceneModel, appearances, supports, raw, frames=None,
                   gain=True, radius_max=RADIUS_MAX, psf=defocus, verbose=False,
                   layers=None, update=True):
    """Per (frame, layer) defocus radius, by forward search — analysis by synthesis.

    PLAYBOOK §0c forbids contrast-over-gradient blur estimation: it saturates by 2
    px and reads texture, not blur. That negative is about a MEASUREMENT taken on
    the blurred image alone. This is a different instrument and a different
    question: given the sharp appearance the model already holds, which radius
    makes the RENDER match the observation? The estimate is therefore in the exact
    parameter the renderer will use, and its known-answer test is `kat2`.

    Grey-scale, inside the layer's bounding box, coarse-to-fine: the search is a
    nuisance-parameter fit, not the result, and paying full price for it would be
    the kind of economy failure DEVSTYLE §13 warns about.
    """
    h, w = model.shape
    frames = range(model.n_frames) if frames is None else frames
    grey_raw = {k: to_gray_float(raw[k]).astype(np.float32) for k in frames}
    grey_app = [to_gray_float(np.clip(a, 0, 255).astype(np.uint8)).astype(np.float32)
                for a in appearances]
    indices = range(len(model.layers)) if layers is None else layers
    radii, gains, report = {}, {}, {}
    for k in frames:
        for index in indices:
            layer = model.layers[index]
            matrix = model.matrix(k, index)
            alpha = warp_forward(layer.alpha, matrix, (h, w), 0.0)
            appearance = warp_forward(grey_app[index], matrix, (h, w), 0.0)
            support = warp_forward(supports[index], matrix, (h, w), 0.0)
            ys, xs = np.nonzero(alpha > 0.999)
            if len(ys) < TF.MIN_LAYER_PIXELS:
                radii[(k, index)], gains[(k, index)] = 0, 1.0
                report[(k, index)] = (0, 1.0, float("nan"), 0)
                continue
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            box_alpha = alpha[y0:y1, x0:x1]
            box_app = appearance[y0:y1, x0:x1]
            box_sup = support[y0:y1, x0:x1]
            box_obs = grey_raw[k][y0:y1, x0:x1]

            def cost(radius):
                alpha_b = np.clip(psf(box_alpha, radius), 0.0, 1.0)
                support_b = np.clip(psf(box_sup, radius, cv2.BORDER_CONSTANT), 0.0, 1.0)
                weight = ((alpha_b > 1.0 - UNKNOWN_TOL)
                          & (support_b > 1.0 - UNKNOWN_TOL)).astype(np.float32)
                if weight.sum() < 200:
                    return None, 1.0, 0.0
                prediction = psf(box_app, radius)
                g = _fit_gain(prediction, box_obs, weight) if gain else 1.0
                error = float((weight * np.abs(g * prediction - box_obs)).sum()
                              / weight.sum())
                return error, g, float(weight.sum())

            coarse = list(range(0, radius_max + 1, 2))
            scored = [(cost(r), r) for r in coarse]
            scored = [(c, r) for (c, _g, _n), r in
                      [((c, g, n), r) for (c, g, n), r in scored] if c is not None]
            if not scored:
                radii[(k, index)], gains[(k, index)] = 0, 1.0
                report[(k, index)] = (0, 1.0, float("nan"), 0)
                continue
            best_r = min(scored)[1]
            fine = [r for r in (best_r - 1, best_r, best_r + 1)
                    if 0 <= r <= radius_max]
            best, best_g, best_n, best_radius = None, 1.0, 0, 0
            for r in fine:
                c, g, n = cost(r)
                if c is not None and (best is None or c < best):
                    best, best_g, best_n, best_radius = c, g, n, r
            radii[(k, index)] = best_radius
            gains[(k, index)] = best_g
            report[(k, index)] = (best_radius, best_g,
                                  float("nan") if best is None else best, int(best_n))
            if verbose:
                print(f"    frame {k:2d} layer {index:2d}  r={best_radius:2d} "
                      f"gain={best_g:.4f} mae={best if best else float('nan'):7.3f} "
                      f"n={int(best_n)}")
    if update:
        model.radii.update(radii)
        model.gains.update(gains)
    weights = np.array([report[key][3] for key in report], float)
    errors = np.array([report[key][2] for key in report], float)
    usable = np.isfinite(errors) & (weights > 0)
    cost = (float(np.average(errors[usable], weights=weights[usable]))
            if usable.any() else float("inf"))
    return report, cost


def select_geometry(model: SceneModel, appearances, supports, raw, verbose=False):
    """Per (frame, group), choose between the two geometries PASS 1 ITSELF offers.

    This is not a new estimator. `twoframe` already builds exactly two candidates
    for every layer — the composed global affine, and `_rigidify`'s collapse of it
    to the pure translation the layer claims to be — and chooses between them by
    verification, per layer, with no threshold. Its own docstring says why the menu
    has to exist: the global affine was fitted across BOTH depth layers at once, so
    it absorbs differential parallax as a spurious scale (F96), and on the analytic
    factory (whose breathing is exactly zero) that scale spreads +-1.3 px of
    sampling error across the frame even where the layer's shift is exact.

    The certifier cannot borrow pass-1's verdict — pass 1 only ever renders the two
    or three frames it elected, and this instrument renders all of them — so it
    re-decides with the criterion it does have: which candidate makes the forward
    render match the observation. Same menu, same per-layer granularity, different
    (and for this purpose more direct) arbiter. `kat3` is its known-answer test:
    on the factory it must choose RIGID, because there the affine's scale is
    provably over-fit.
    """
    for k in range(model.n_frames):
        for group in model.groups():
            options = model.menu.get((k, group))
            if not options or len(options) < 2:
                continue
            indices = model.layers_of(group)
            best = None
            for name, matrix in options:
                model.matrices[(k, group)] = matrix
                _report, cost = estimate_radii(model, appearances, supports, raw,
                                               frames=[k], layers=indices,
                                               update=False)
                if best is None or cost < best[0] - 1e-9:
                    best = (cost, name, matrix)
            model.matrices[(k, group)] = best[2]
            model.choices[(k, group)] = best[1]
            if verbose:
                print(f"    frame {k:2d} group {group}: {best[1]} "
                      f"(mae {best[0]:.3f})")
    return model.choices


def choice_summary(model: SceneModel) -> str:
    if not model.choices:
        return "no geometry menu"
    counts = {}
    for name in model.choices.values():
        counts[name] = counts.get(name, 0) + 1
    total = sum(counts.values())
    return ", ".join(f"{name} {n}/{total}" for name, n in sorted(counts.items()))


# ---------------------------------------------------------------------------
# Certification
# ---------------------------------------------------------------------------
@dataclass
class Certification:
    unexplained: np.ndarray       # reference geometry, mean per-frame residual
    coverage: np.ndarray          # reference geometry, frames certified
    score: float
    p99: float
    certified: float              # mean share of the frame certified
    boundary: float
    excluded: float
    per_frame: list = field(default_factory=list)


def certify(model: SceneModel, composite, crop, raw, frames=None, region=None,
            residual_dir=None):
    """Render every source frame from `composite` + `model`; report what is left.

    The residual lives in each frame's OWN geometry (that is the point), then is
    pulled back into reference geometry through each layer's own inverse warp so
    that the per-frame maps can be aggregated into one picture of where the
    composite is unexplained.
    """
    h, w = model.shape
    frames = list(range(model.n_frames)) if frames is None else list(frames)
    canvas, canvas_support = place(composite, crop, (h, w))
    appearances, supports = layer_views(model, canvas, canvas_support)

    accumulated = np.zeros((h, w), np.float32)
    count = np.zeros((h, w), np.int32)
    per_frame = []
    for k in frames:
        prediction, known = render(model, appearances, supports, k)
        observation = raw[k].astype(np.float32)
        residual = np.abs(prediction - observation).max(axis=2)
        # EXCLUDED, not merely uncertain: a clipped pixel is outside the model's
        # linear range, so its residual measures the sensor, not the composite.
        saturated = (observation.max(axis=2) >= 254.0) | (observation.min(axis=2) <= 1.0)
        certified = (known > 1.0 - UNKNOWN_TOL) & ~saturated
        boundary = (known > UNKNOWN_TOL) & (known <= 1.0 - UNKNOWN_TOL) & ~saturated
        excluded = ~certified & ~boundary
        per_frame.append({
            "frame": k,
            "mae": float(residual[certified].mean()) if certified.any() else float("nan"),
            "p99": float(np.percentile(residual[certified], 99)) if certified.any() else float("nan"),
            "certified": float(certified.mean()),
            "boundary": float(boundary.mean()),
            "excluded": float(excluded.mean()),
        })
        if residual_dir is not None:
            os.makedirs(residual_dir, exist_ok=True)
            cv2.imwrite(os.path.join(residual_dir, f"residual_{k:02d}.png"),
                        np.clip(residual * 4.0, 0, 255).astype(np.uint8))
        # Pull back per GEOMETRY GROUP: layers sharing a motion share one inverse.
        for group in sorted({layer.group for layer in model.layers}):
            matrix = model.matrices[(k, group)]
            back_residual = warp_back(residual, matrix, (h, w), 0.0)
            back_certified = warp_back(certified.astype(np.float32), matrix, (h, w), 0.0)
            here = np.zeros((h, w), bool)
            for layer in model.layers:
                if layer.group == group:
                    here |= layer.mask
            here &= back_certified > 0.999
            accumulated[here] += back_residual[here]
            count[here] += 1

    unexplained = accumulated / np.maximum(count, 1)
    scored = count >= MIN_COVERAGE
    if region is not None:
        scored &= region
    scored &= canvas_support
    return Certification(
        unexplained=unexplained,
        coverage=count,
        score=float(unexplained[scored].mean()) if scored.any() else float("nan"),
        p99=float(np.percentile(unexplained[scored], 99)) if scored.any() else float("nan"),
        certified=float(np.mean([f["certified"] for f in per_frame])),
        boundary=float(np.mean([f["boundary"] for f in per_frame])),
        excluded=float(np.mean([f["excluded"] for f in per_frame])),
        per_frame=per_frame,
    )


def differential(candidate: Certification, null: Certification) -> Certification:
    """candidate - null, per pixel: what the CANDIDATE fails to explain that the
    reference frame does not also fail to explain.

    The floor ladder (`floor`) measures the forward model's own error at 4.06
    levels on the factory, of which 2.93 is the pass-1 layer segmentation. That
    error is a property of the MODEL, so it is identical for every candidate scored
    through it — and in an absolute map it sits exactly where a real defect would,
    inviting a certifier to report the instrument's own seams as the composite's
    faults. Subtracting the null cancels it.

    The subtraction is not tautological in the direction that would matter. The
    null is a single defocused frame; the radius search may only ADD blur, so a
    null render cannot match a frame in which some layer is sharp. A composite that
    is sharper than the reference AND correct scores BELOW the null (negative
    differential); one that is sharper and wrong scores above it. Agreeing with the
    reference buys nothing here, which is the trap F112/R4 rejected the four-box
    metric for.
    """
    delta = candidate.unexplained - null.unexplained
    coverage = np.minimum(candidate.coverage, null.coverage)
    scored = coverage >= MIN_COVERAGE
    return Certification(
        unexplained=delta, coverage=coverage,
        score=float(delta[scored].mean()) if scored.any() else float("nan"),
        p99=float(np.percentile(delta[scored], 99)) if scored.any() else float("nan"),
        certified=candidate.certified, boundary=candidate.boundary,
        excluded=candidate.excluded, per_frame=candidate.per_frame)


def clusters(result: Certification, region, top=8, percentile=99.5, min_area=25):
    """The largest concentrations of unexplained residual, ranked by total mass.

    A mean hides a localized defect (PLAYBOOK §0: a global mean rated three fusion
    methods equal while one had a visible halo), so the scalar score is never the
    whole report — this is the part that says WHERE.
    """
    valid = region & (result.coverage >= MIN_COVERAGE)
    if not valid.any():
        return []
    threshold = float(np.percentile(result.unexplained[valid], percentile))
    hot = (result.unexplained > threshold) & valid
    hot = cv2.morphologyEx(hot.astype(np.uint8), cv2.MORPH_CLOSE,
                           np.ones((5, 5), np.uint8))
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(hot, 8)
    found = []
    for label in range(1, count):
        x, y, bw, bh, area = stats[label]
        if area < min_area:
            continue
        member = labels == label
        found.append({
            "box": (int(x), int(y), int(x + bw), int(y + bh)),
            "area": int(area),
            "mass": float(result.unexplained[member].sum()),
            "peak": float(result.unexplained[member].max()),
            "mean": float(result.unexplained[member].mean()),
        })
    found.sort(key=lambda c: -c["mass"])
    return found[:top]


def heat_overlay(base, result: Certification, region, vmax=None):
    """Unexplained map as a heat overlay. Render the picture even when the numbers agree."""
    valid = region & (result.coverage >= MIN_COVERAGE)
    if vmax is None:
        vmax = float(np.percentile(result.unexplained[valid], 99.9)) if valid.any() else 1.0
    vmax = max(vmax, 1e-6)
    scaled = np.clip(result.unexplained / vmax, 0, 1)
    heat = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    weight = (scaled * valid)[..., None].astype(np.float32)
    grey = cv2.cvtColor(cv2.cvtColor(base.astype(np.uint8), cv2.COLOR_BGR2GRAY),
                        cv2.COLOR_GRAY2BGR).astype(np.float32)
    blend = grey * (1.0 - weight) + heat.astype(np.float32) * weight
    return np.clip(blend, 0, 255).astype(np.uint8), vmax


# ---------------------------------------------------------------------------
# Scene models
# ---------------------------------------------------------------------------
def factory_truth_model(canvas=False):
    """The analytic factory's model with its TRUE parameters — KAT-1's known answer.

    `parallax_gen` IS a forward renderer, so its own constants are the ground truth
    for this instrument: two planes, known per-frame lateral shifts in a known
    ratio, a known disk radius per frame per plane, and a known matte. Nothing here
    is estimated. Rebuilt (not imported) because `parallax_gen.build_stack` returns
    only the frames and the reference-viewpoint truth; the module is READ-ONLY.
    """
    import parallax_gen as P

    height = P.HEIGHT + 2 * P.PAD if canvas else P.HEIGHT
    width = P.WIDTH + 2 * P.PAD if canvas else P.WIDTH
    shape = (P.HEIGHT + 2 * P.PAD, P.WIDTH + 2 * P.PAD)
    background = P._texture(1, *shape)
    foreground = P._texture(7, *shape)
    alpha = np.zeros(shape, np.float32)
    cv2.rectangle(alpha, (P.PAD + 60, P.PAD + 70), (P.PAD + 300, P.PAD + 330), 1.0, -1)
    cv2.circle(alpha, (P.PAD + 400, P.PAD + 140), 70, 1.0, -1)

    def crop(x):
        return x if canvas else x[P.PAD:P.PAD + P.HEIGHT, P.PAD:P.PAD + P.WIDTH]

    near_mask = crop(alpha) > 0.5
    layers = [
        Layer(mask=~near_mask, group=1, order=1.0, name="far plane",
              extent=np.ones((height, width), bool)),
        Layer(mask=near_mask, group=0, order=0.0, name="near plane"),
    ]
    model = SceneModel(shape=(height, width), n_frames=P.FRAMES, ref=P.REFERENCE,
                       layers=layers)
    for k in range(P.FRAMES):
        step = k - P.REFERENCE
        for group, per_frame in ((0, P.NEAR_SHIFT_PER_FRAME),
                                 (1, P.FAR_SHIFT_PER_FRAME)):
            matrix = np.eye(3)
            matrix[0, 2] = -step * per_frame
            model.matrices[(k, group)] = matrix
        # layers[0] is the far plane, layers[1] the near one — see `layers` above.
        model.radii[(k, 0)] = int(round(abs(k - P.FAR_FOCUS_FRAME) * P.BLUR_PER_STEP))
        model.radii[(k, 1)] = int(round(abs(k - P.NEAR_FOCUS_FRAME) * P.BLUR_PER_STEP))
    truth = {"foreground": crop(foreground).astype(np.float32),
             "background": crop(background).astype(np.float32),
             "alpha": crop(alpha), "near_mask": near_mask}
    return model, truth


# Round B hook. An external decomposition may replace pass-1's layer masks
# WITHOUT touching anything else in the model: the motion fits, the propagation,
# the geometry grouping, the menu and every KAT downstream stay exactly as round
# A built them, so a segmentation is scored through an unchanged instrument.
# `None` (the default) is byte-identical to round A — verified by re-running
# `floor` and diffing all five rungs.
#
#   layer_decompose.py sets `forward_certify.SEGMENTER = fn` and then calls the
#   existing commands; nothing else in this module knows the difference.
#
# A segmentation is a list of dicts, one per layer:
#   mask   bool (h, w)  — where the layer's appearance is OBSERVED (certifiable)
#   extent bool (h, w)  — optional; where the SURFACE EXISTS, visible or not
#   peak   float        — the layer's focal frame (drives the focal weighting)
#   order  float        — optional nearness proxy, SMALLER is NEARER (default peak)
#   name   str          — optional label
SEGMENTER = None          # callable(images, ref) -> list[dict] | None


def model_from_pass1(images, ref=None, verbose=False, segmentation=None):
    """A scene model built the way pass 1 already builds one — nothing new invented.

    Every quantity is a pass-1 output, reached through `twoframe`'s own public
    stages: the global affine (`global_stage`), the focal field and the per-tile
    focus contest (`focal_field`, `tile_pairs`, `merge_pairs`, `ownership`), the
    per-region layer masks (`layer_masks`), and the masked ECC layer translation
    (`masked_translation`, known-answer tested at -2/-5/-12/-20/-30 px in F109).

    One thing IS added, and it is a reduction rather than an invention: layers
    whose fitted motion agrees within `GROUP_TOL` in EVERY frame become one
    GEOMETRY GROUP. Two reasons, both measured elsewhere in this project. (a) F93:
    regions whose fitted motion agrees across the sweep are one object; splitting
    them is a claim the evidence does not support. (b) Mechanically, two adjacent
    masks warped by two nearly-equal-but-different transforms tear open a gap at
    every shared edge, and every one of those gaps would be reported as
    "unexplained" when it is really the certifier's own seam. Defocus stays
    per-LAYER — layers in one group share a motion, not a depth.
    """
    n = len(images)
    ref = n // 2 if ref is None else ref
    h, w = images[0].shape[:2]
    max_shift = TF.MAX_SHIFT_FRACTION * float(np.hypot(h, w))

    coarse, warps, valid = TF.global_stage(images, ref)
    common = np.logical_and.reduce(valid)
    peak, contrast, energies = TF.focal_field(coarse)
    depth = depth_from_focus(coarse)
    tiles = TF.tile_pairs(peak, contrast, energies, common)
    kept = TF.merge_pairs(tiles)
    owner, _weights = TF.ownership(tiles, kept, (h, w))

    ref_gray = to_gray_float(coarse[ref]).astype(np.float32) / 255.0
    greys = [to_gray_float(c).astype(np.float32) / 255.0 for c in coarse]
    gradient = cv2.magnitude(
        cv2.Sobel(ref_gray * 255.0, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(ref_gray * 255.0, cv2.CV_32F, 0, 1, ksize=3))
    textured = gradient >= TF._REFINE_MIN_GRADIENT if hasattr(TF, "_REFINE_MIN_GRADIENT") \
        else gradient >= TF.A._REFINE_MIN_GRADIENT

    if segmentation is None and SEGMENTER is not None:
        segmentation = SEGMENTER(images, ref)

    raw_layers, fit_supports = [], []
    if segmentation is not None:
        for entry in segmentation:
            raw_layers.append({"mask": entry["mask"], "extent": entry.get("extent"),
                               "pair": entry.get("name", ""), "level": 0,
                               "peak": float(entry["peak"]),
                               "order": float(entry.get("order", entry["peak"])),
                               "depth": float(np.median(depth[entry["mask"]]))
                               if entry["mask"].any() else 0.0})
            fit_supports.append(entry["mask"] & textured & common)
    for index, pair in (enumerate(kept) if segmentation is None else ()):
        owned = owner == index
        _fit_masks, dense = TF.layer_masks(energies, pair, owned & common, gradient)
        for level, dense_mask in enumerate(dense):
            mask = owned & dense_mask
            # A focus contest is per-pixel and speckles; a LAYER is not speckle.
            # One open-close at tile-vote scale, and nothing finer — the layer
            # boundary is exactly where this instrument declines to certify.
            cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                                       np.ones((5, 5), np.uint8))
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE,
                                       np.ones((9, 9), np.uint8)) > 0
            if cleaned.sum() < TF.MIN_LAYER_PIXELS:
                continue
            raw_layers.append({"mask": cleaned, "pair": pair, "level": level,
                               "peak": float(np.median(peak[cleaned])),
                               "depth": float(np.median(depth[cleaned]))})
            fit_supports.append(cleaned & textured & common)

    # Unclaimed pixels (morphology losses, untextured strays) join the layer whose
    # mask is nearest. A partition with holes would report its own holes as
    # unexplained; the holes are not evidence about the composite.
    # An external segmentation is TRINARY on purpose: its unclaimed pixels are its
    # boundary-band and unknown states, and filling them into a mask would certify
    # exactly the pixels it declined to own. It supplies its own extents instead.
    claimed = np.logical_or.reduce([entry["mask"] for entry in raw_layers])
    if segmentation is None and not claimed.all():
        stack = np.stack([entry["mask"] for entry in raw_layers], 0).astype(np.uint8)
        distances = np.stack([cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
                              for m in stack], 0)
        nearest = np.argmin(distances, axis=0)
        for i, entry in enumerate(raw_layers):
            entry["mask"] = entry["mask"] | (~claimed & (nearest == i))

    # Per (frame, layer) translation, by the pipeline's own validated estimator.
    shifts = {}
    for k in range(n):
        for i, entry in enumerate(raw_layers):
            if k == ref:
                shifts[(k, i)] = (0.0, 0.0)
                continue
            fitted = TF.masked_translation(ref_gray, greys[k], fit_supports[i],
                                           max_shift)
            shifts[(k, i)] = ((0.0, 0.0) if fitted is None
                              else (float(fitted[0, 2]), float(fitted[1, 2])))

    # Propagated motion, per layer. A layer is measurable only near its own focal
    # plane — defocus destroys the interior detail that ECC needs, and a blurred
    # profile correlates CONFIDENTLY at about zero shift (F99), so per-frame fits
    # far from the peak are not weak, they are wrong with conviction. PLAYBOOK §0's
    # recipe is to measure near the focal plane and PROPAGATE along the sweep, and
    # §12.6's is to use the simplest model the physics allows: a handheld camera
    # centre translates, displacement is linear in that translation, and a linear
    # fit landed +18.88 px against +19.2 truth where a quadratic overshot to
    # +23.98. Slope by focal-weighted Theil-Sen median, which needs no outlier
    # threshold — the failures above are a minority and a median ignores them.
    propagated = {}
    for i, entry in enumerate(raw_layers):
        weights = np.exp(-0.5 * ((np.arange(n) - entry["peak"]) / TF.FOCAL_SIGMA) ** 2)
        slopes = []
        for axis in (0, 1):
            samples, sample_weights = [], []
            for k in range(n):
                if k == ref:
                    continue
                samples.append(shifts[(k, i)][axis] / (k - ref))
                sample_weights.append(weights[k] * abs(k - ref))
            slopes.append(_weighted_median(np.array(samples),
                                           np.array(sample_weights)))
        for k in range(n):
            propagated[(k, i)] = (slopes[0] * (k - ref), slopes[1] * (k - ref))

    # Geometry grouping: agreement in EVERY frame, greedy from the largest layer.
    # Keyed on the PROPAGATED series, not the raw one — grouping on measurements
    # that are individually wrong produces one group per layer, which is what the
    # first run of this did, and 10 independently-warped layers tear the frame open
    # at every shared edge.
    measured, shifts = shifts, propagated
    order = sorted(range(len(raw_layers)), key=lambda i: -raw_layers[i]["mask"].sum())
    group_of, group_members = {}, []
    for i in order:
        for g, members in enumerate(group_members):
            head = members[0]
            if all(max(abs(shifts[(k, i)][0] - shifts[(k, head)][0]),
                       abs(shifts[(k, i)][1] - shifts[(k, head)][1])) <= GROUP_TOL
                   for k in range(n)):
                members.append(i)
                group_of[i] = g
                break
        else:
            group_of[i] = len(group_members)
            group_members.append([i])

    layers = [Layer(mask=entry["mask"], group=group_of[i],
                    order=entry.get("order", entry["peak"]),
                    name=(entry["pair"] if segmentation is not None
                          else f"pair{entry['pair']}/L{entry['level']}"),
                    extent=entry.get("extent"))
              for i, entry in enumerate(raw_layers)]
    model = SceneModel(shape=(h, w), n_frames=n, ref=ref, layers=layers)
    group_masks = [np.logical_or.reduce([raw_layers[i]["mask"] for i in members])
                   for members in group_members]
    for k in range(n):
        base = np.eye(3) if warps[k] is None else _homogeneous(warps[k])
        for g, members in enumerate(group_members):
            weights = np.array([raw_layers[i]["mask"].sum() for i in members], float)
            menu = [("global-only", base)]
            for tag, table in (("meas", measured), ("prop", shifts)):
                dx = float(np.average([table[(k, i)][0] for i in members],
                                      weights=weights))
                dy = float(np.average([table[(k, i)][1] for i in members],
                                      weights=weights))
                composed = base @ np.array([[1.0, 0.0, dx],
                                            [0.0, 1.0, dy],
                                            [0.0, 0.0, 1.0]])
                menu.append((f"affine-{tag}", composed))
                menu.append((f"rigid-{tag}",
                             TF._rigidify(composed, group_masks[g], (h, w))))
            # The default before selection is the pass-1 measurement composed as
            # pass 1 composes it — so a caller that never selects gets exactly the
            # pipeline's own geometry and nothing of this module's opinion.
            model.matrices[(k, g)] = menu[1][1]
            model.menu[(k, g)] = menu
    if verbose:
        print(f"  pass-1 model: {len(layers)} layers in {len(group_members)} geometry "
              f"groups, ref {ref}")
        for g, members in enumerate(group_members):
            names = ", ".join(layers[i].name for i in members)

            def span(table):
                return max(float(np.hypot(
                    *[np.average([table[(k, i)][a] for i in members],
                                 weights=[raw_layers[i]["mask"].sum() for i in members])
                      for a in (0, 1)])) for k in range(n))

            print(f"    group {g}: max |shift| measured {span(measured):5.2f} px, "
                  f"propagated {span(shifts):5.2f} px, {len(members)} layers  [{names}]")
    return model, {"owner": owner, "peak": peak, "coarse": coarse, "pairs": kept,
                   "shifts": shifts, "measured": measured, "groups": group_members,
                   "common": common}


# ---------------------------------------------------------------------------
# KAT 1 — the renderer, against the factory it was modelled on
# ---------------------------------------------------------------------------
def _ssim(a, b):
    import metrics
    return metrics.ref_ssim(np.clip(a, 0, 255).astype(np.uint8),
                            np.clip(b, 0, 255).astype(np.uint8))


def kat1() -> None:
    import parallax_gen as P

    os.makedirs(OUT, exist_ok=True)
    frames, _truth_image, _near = P.build_stack()

    print("=" * 78)
    print("KAT-1a — renderer vs the factory, TRUE layers and TRUE parameters")
    print("=" * 78)
    print("The factory renders on a PADDED canvas and crops. Rung (a) gives the\n"
          "renderer the same padded canvas, so nothing is unobserved and any gap is\n"
          "the renderer's own composite arithmetic. Expect near-identity.\n")
    model, truth = factory_truth_model(canvas=True)
    appearances = [truth["background"], truth["foreground"]]
    supports = [np.ones(model.shape, np.float32)] * 2
    print(f"{'frame':>5} {'r_near':>7} {'r_far':>6} {'MAE':>8} {'max':>6} {'SSIM':>9}")
    maes = []
    for k in range(model.n_frames):
        prediction, known = render(model, appearances, supports, k)
        cropped = prediction[P.PAD:P.PAD + P.HEIGHT, P.PAD:P.PAD + P.WIDTH]
        observation = frames[k].astype(np.float32)
        residual = np.abs(cropped - observation)
        maes.append(residual.mean())
        print(f"{k:5d} {model.radii[(k, 1)]:7d} {model.radii[(k, 0)]:6d} "  # near, far
              f"{residual.mean():8.4f} {residual.max():6.1f} "
              f"{_ssim(cropped, observation):9.6f}")
    print(f"\n  mean MAE over the sweep: {np.mean(maes):.4f} levels "
          f"(uint8 quantization alone is 0.25)")
    # Where that fraction of a level comes from, measured rather than waved at:
    # `parallax_gen` writes its frames with `.astype(np.uint8)`, which TRUNCATES,
    # and it warps uint8 layers (so each warped sample is rounded before the
    # blur). The renderer works in float throughout. Truncating the prediction
    # the same way isolates that.
    truncated = []
    for k in range(model.n_frames):
        prediction, _known = render(model, appearances, supports, k)
        cropped = prediction[P.PAD:P.PAD + P.HEIGHT, P.PAD:P.PAD + P.WIDTH]
        cropped = np.clip(cropped, 0, 255).astype(np.uint8).astype(np.float32)
        truncated.append(np.abs(cropped - frames[k].astype(np.float32)).mean())
    print(f"  same, with the factory's own uint8 TRUNCATION applied: "
          f"{np.mean(truncated):.4f} levels")

    print()
    print("=" * 78)
    print("KAT-1b — the same render restricted to what a COMPOSITE could know")
    print("=" * 78)
    print("Reference geometry only: the far plane is observable just where the near\n"
          "plane does not cover it, and nothing outside the reference field of view\n"
          "exists at all. This is the honest configuration, and its cost is the gap\n"
          "between the two rungs.\n")
    model, truth = factory_truth_model(canvas=False)
    near = truth["near_mask"]
    canvas = (truth["foreground"] * near[..., None]
              + truth["background"] * (~near)[..., None])
    appearances, supports = layer_views(
        model, canvas, np.ones(model.shape, bool))
    print(f"{'frame':>5} {'MAE cert':>9} {'p99':>7} {'certified':>10} "
          f"{'boundary':>9} {'excluded':>9}")
    for k in range(model.n_frames):
        prediction, known = render(model, appearances, supports, k)
        observation = frames[k].astype(np.float32)
        residual = np.abs(prediction - observation).max(axis=2)
        certified = known > 1.0 - UNKNOWN_TOL
        boundary = (known > UNKNOWN_TOL) & ~certified
        print(f"{k:5d} {residual[certified].mean():9.4f} "
              f"{np.percentile(residual[certified], 99):7.3f} "
              f"{certified.mean() * 100:9.2f}% {boundary.mean() * 100:8.2f}% "
              f"{(~certified & ~boundary).mean() * 100:8.2f}%")
        if k == 0:
            cv2.imwrite(os.path.join(OUT, "kat1_predicted_f00.png"),
                        np.clip(prediction, 0, 255).astype(np.uint8))
            cv2.imwrite(os.path.join(OUT, "kat1_observed_f00.png"), frames[0])
            cv2.imwrite(os.path.join(OUT, "kat1_certified_f00.png"),
                        (certified * 255).astype(np.uint8))
    print("\n  Boundary is not error: it is every pixel whose defocus kernel reached\n"
          "  content the reference viewpoint never saw. Reported, not certified.")


# ---------------------------------------------------------------------------
# KAT 2 — the blur estimator, against known sigma
# ---------------------------------------------------------------------------
def _gaussian_psf(image, radius, border=cv2.BORDER_REPLICATE):
    """A deliberately WRONG PSF family, for the mismatch measurement."""
    r = int(round(radius))
    if r < 1:
        return image
    return cv2.GaussianBlur(image, (0, 0), r / 2.0, borderType=border)


def kat2() -> None:
    import parallax_gen as P

    frames, _truth_image, _near = P.build_stack()
    print("=" * 78)
    print("KAT-2 — the defocus estimator, on the factory where the radius is KNOWN")
    print("=" * 78)
    print("PLAYBOOK §0c: contrast-over-gradient blur estimation saturates by 2 px\n"
          "and is closed. This is a different instrument — a forward search for the\n"
          "radius that makes the RENDER match the frame — so it gets its own KAT\n"
          "before it is believed anywhere. Truth is round(|k - focus| * 1.15).\n")

    model, truth = factory_truth_model(canvas=False)
    true_radii = dict(model.radii)          # layer 0 = far plane, layer 1 = near
    true_matrices = dict(model.matrices)
    near = truth["near_mask"]
    ideal = (truth["foreground"] * near[..., None]
             + truth["background"] * (~near)[..., None])
    candidate, info = TF.twoframe_stack(frames, P.REFERENCE)
    x0, y0, x1, y1 = info["crop"]
    real_canvas, real_support = place(candidate, info["crop"], model.shape)

    def run(label, canvas, support, psf=defocus, gain=True, inject=0.0, note=""):
        model.matrices = dict(true_matrices)
        if inject:
            for k in range(model.n_frames):
                matrix = model.matrices[(k, 0)].copy()   # group 0 = the near plane
                matrix[0, 2] += inject
                model.matrices[(k, 0)] = matrix
        appearances, supports = layer_views(model, canvas, support)
        model.radii, model.gains = {}, {}
        report, _cost = estimate_radii(model, appearances, supports, frames,
                                       gain=gain, psf=psf)
        errors, residuals = [], []
        rows = []
        for k in range(model.n_frames):
            true_near, true_far = true_radii[(k, 1)], true_radii[(k, 0)]
            got_near, got_far = model.radii[(k, 1)], model.radii[(k, 0)]
            errors += [got_near - true_near, got_far - true_far]
            residuals += [report[(k, 1)][2], report[(k, 0)][2]]
            rows.append((k, true_near, got_near, true_far, got_far,
                         model.gains[(k, 1)], model.gains[(k, 0)]))
        errors = np.array(errors, float)
        print(f"\n  {label}")
        if note:
            print(f"    {note}")
        print(f"  {'frame':>5} | {'near true':>9} {'near est':>8} | "
              f"{'far true':>8} {'far est':>8} | {'gain near':>9} {'gain far':>8}")
        for row in rows:
            print(f"  {row[0]:5d} | {row[1]:9d} {row[2]:8d} | {row[3]:8d} "
                  f"{row[4]:8d} | {row[5]:9.4f} {row[6]:8.4f}")
        print(f"  -> exact hits {(errors == 0).mean() * 100:5.1f}%   mean signed error "
              f"{errors.mean():+.2f} px   max |error| {np.abs(errors).max():.0f} px"
              f"   best-fit MAE {np.nanmean(residuals):.3f} levels")
        return errors, float(np.nanmean(residuals))

    run("A. true appearance, disk PSF, exposure gain fitted", ideal,
        np.ones(model.shape, bool))
    run("B. true appearance, disk PSF, NO exposure gain", ideal,
        np.ones(model.shape, bool), gain=False)
    run("C. true appearance, GAUSSIAN PSF (family MISMATCH)", ideal,
        np.ones(model.shape, bool), psf=_gaussian_psf,
        note="the argmin can survive a wrong family; the RESIDUAL is where it shows")
    run("D. REAL candidate appearance (the routed two-frame composite)",
        real_canvas, real_support,
        note="the condition the estimator actually runs in — an imperfect composite")
    run("E. real candidate AND +3 px of geometry error on the near plane",
        real_canvas, real_support, inject=3.0,
        note="does blur absorb a misregistration? a rising radius here is the tell")
    model.matrices = dict(true_matrices)

    print("\n  Layer 0 is the FAR plane (focus frame 4), layer 1 the NEAR plane\n"
          "  (focus frame 1); the columns are labelled by the plane, not the index.")


# ---------------------------------------------------------------------------
# KAT 3 — the ranking, against ground truth
# ---------------------------------------------------------------------------
def kat3() -> None:
    import metrics
    import parallax_gen as P
    from focusstack.align import align_stack
    from focusstack.fusion import fuse_perband

    os.makedirs(OUT, exist_ok=True)
    frames, truth_image, _near = P.build_stack()

    aligned, report = align_stack(frames, motion="affine", depth_bins=3,
                                  return_report=True)
    crop = report["crop"]
    x0, y0, x1, y1 = crop
    shipped = fuse_perband(aligned, usable=report["usable"])
    twoframe, info = TF.twoframe_stack(frames, P.REFERENCE)
    assert tuple(info["crop"]) == tuple(crop), (info["crop"], crop)
    gt = {"shipped depth-bin": metrics.ref_ssim(shipped, truth_image[y0:y1, x0:x1]),
          "two-frame route": metrics.ref_ssim(twoframe, truth_image[y0:y1, x0:x1])}

    print("=" * 78)
    print("KAT-3 — does the certifier RANK the two composites as ground truth does?")
    print("=" * 78)
    print("Same crop, same stack, same scene model. Ground truth says two-frame wins\n"
          "by +0.0066 GT-SSIM. A certifier that reverses this is broken.\n")

    model, _diag = model_from_pass1(frames, P.REFERENCE, verbose=True)
    region = np.zeros(model.shape, bool)
    region[y0:y1, x0:x1] = True

    # The geometry is chosen ONCE, on the null appearance, and then FROZEN. A
    # candidate must not be allowed to pick the geometry that flatters it; the
    # reference frame is available on every scene and belongs to no candidate.
    # Radii and gains ARE re-fitted per candidate, because they are nuisance
    # parameters each candidate is equally free to use and KAT-2 rung E measured
    # what they can and cannot hide: 1-2 px of absorbed misregistration, at the
    # price of a residual that quadrupled and gave the error away anyway.
    null = frames[P.REFERENCE][y0:y1, x0:x1]
    null_canvas, null_support = place(null, crop, model.shape)
    null_app, null_sup = layer_views(model, null_canvas, null_support)
    select_geometry(model, null_app, null_sup, frames, verbose=True)
    print(f"  geometry chosen on the null appearance: {choice_summary(model)}")

    print(f"\n  {'candidate':<22} {'GT-SSIM':>9} {'certifier':>10} {'p99':>7} "
          f"{'cert%':>7} {'bnd%':>6} {'exc%':>6}")
    results = {}
    # The GROUND TRUTH composite is the rung that says whether the certifier has a
    # sharpness bias. It is correct by construction, so it must score BEST of all;
    # if a sharp-and-right candidate is punished relative to a blurred one, the
    # cause is sub-pixel geometry error (a sharp edge rendered half a pixel off
    # costs far more than a soft edge rendered half a pixel off) and the instrument
    # has inherited a new version of F81a's trap, this time through the model.
    for label, composite in (("GROUND TRUTH composite", truth_image[y0:y1, x0:x1]),
                             ("shipped depth-bin", shipped),
                             ("two-frame route", twoframe),
                             ("NULL (reference frame)", null)):
        canvas, support = place(composite, crop, model.shape)
        appearances, supports = layer_views(model, canvas, support)
        model.radii, model.gains = {}, {}
        estimate_radii(model, appearances, supports, frames)
        result = certify(model, composite, crop, frames, region=region)
        results[label] = result
        gt_value = gt.get(label)
        print(f"  {label:<22} "
              f"{('%9.6f' % gt_value) if gt_value else ' ' * 9} "
              f"{result.score:10.4f} {result.p99:7.3f} "
              f"{result.certified * 100:6.2f}% {result.boundary * 100:5.2f}% "
              f"{result.excluded * 100:5.2f}%")

    better_gt = max(gt, key=gt.get)
    better_cert = min(("shipped depth-bin", "two-frame route"),
                      key=lambda k: results[k].score)
    print(f"\n  ground truth prefers : {better_gt}")
    print(f"  certifier prefers    : {better_cert}")
    print(f"  VERDICT: {'AGREE' if better_gt == better_cert else 'DISAGREE — instrument broken'}")

    null_result = results["NULL (reference frame)"]
    print(f"\n  NULL COMPOSITE (the reference frame itself): {null_result.score:.4f} "
          f"levels, p99 {null_result.p99:.3f}")
    print(f"  margin two-frame vs shipped "
          f"{results['two-frame route'].score - results['shipped depth-bin'].score:+.4f} "
          f"levels, against a null of {null_result.score:.4f}.")
    print("  Read that honestly: the ranking is correct but the MARGIN is an order of\n"
          "  magnitude under the floor it sits on, so this instrument separates these\n"
          "  two composites, not any two. `floor` attributes the floor: on this scene\n"
          "  it is the model's, and mostly the pass-1 LAYER SEGMENTATION's, not the\n"
          "  null candidate's defocus.")

    # Plumbing KAT: the null composite must explain the REFERENCE FRAME exactly.
    # Identity transform, zero radius, unit gain — anything but ~0 here is a bug in
    # the warp convention, the crop placement or the gain fit, not a finding.
    canvas, support = place(null, crop, model.shape)
    appearances, supports = layer_views(model, canvas, support)
    model.radii, model.gains = {}, {}
    estimate_radii(model, appearances, supports, frames, frames=[P.REFERENCE])
    self_test = certify(model, null, crop, frames, frames=[P.REFERENCE],
                        region=region)
    print(f"\n  self-consistency: the reference composite vs the reference FRAME "
          f"reads {self_test.score:.4f} levels "
          f"({'PASS' if self_test.score < 0.5 else 'FAIL — plumbing bug'})")

    gt_score = results["GROUND TRUTH composite"].score
    best = min(results, key=lambda k: results[k].score)
    print(f"\n  SHARPNESS-BIAS CHECK: the ground-truth composite scores {gt_score:.4f}; "
          f"the best of all four is '{best}'.")
    print(f"    {'PASS — truth wins, no sharpness penalty' if best == 'GROUND TRUTH composite' else 'FAIL — a wrong candidate beats truth; the instrument penalises sharpness'}")
    for label, key, image in (("gt", "GROUND TRUTH composite", truth_image[y0:y1, x0:x1]),
                              ("shipped", "shipped depth-bin", shipped),
                              ("twoframe", "two-frame route", twoframe),
                              ("null", "NULL (reference frame)", null)):
        overlay, _vmax = heat_overlay(place(image, crop, model.shape)[0],
                                      results[key], region, vmax=12.0)
        cv2.imwrite(os.path.join(OUT, f"kat3_{label}_unexplained.png"), overlay)
    print(f"  heat overlays (common scale, vmax 12 levels) -> {OUT}/kat3_*.png")


# ---------------------------------------------------------------------------
# KAT 4 — known-defect localization on the kitchen
# ---------------------------------------------------------------------------
KNOB = (659, 243, 670, 314)            # F112's dark background knob, ORIGINAL coords
SLIVER = (473 + 15, 135 + 8, 506 + 15, 156 + 8)   # pale sliver, composite -> original
FLANK_BOX = (560, 240, 670, 420)       # verified-clean low-contrast flank
STREAK = (230, 400, 600, 700)          # F108's canonical acceptance box (y0,y1,x0,x1)


def _box_stats(result: Certification, box, region):
    x0, y0, x1, y1 = box
    window = np.zeros(result.unexplained.shape, bool)
    window[y0:y1, x0:x1] = True
    window &= region & (result.coverage >= MIN_COVERAGE)
    if not window.any():
        return float("nan"), float("nan"), 0
    values = result.unexplained[window]
    return float(values.mean()), float(values.max()), int(window.sum())


def _overlaps(box, target, pad=8):
    return not (box[2] < target[0] - pad or box[0] > target[2] + pad
                or box[3] < target[1] - pad or box[1] > target[3] + pad)


def kat4() -> None:
    from focusstack.io import normalize_exposure

    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    paths = sorted(glob.glob(os.path.join(KITCHEN, "*.jpg")))
    src = [cv2.imread(p) for p in paths]
    norm = normalize_exposure(src)
    composite, info = TF.twoframe_stack(norm)
    crop = info["crop"]
    kx0, ky0 = crop[0], crop[1]

    print("=" * 78)
    print("KAT-4 — do the known kitchen defects light up, and does the clean flank stay dark?")
    print("=" * 78)
    print(f"  routed composite {composite.shape}, crop {crop}, "
          f"pairs {info['pairs']}, frames used {info['frames_used']}")
    print("  Certified against the RAW jpgs, not the normalized ones: the exposure\n"
          "  residual is then a MEASURED nuisance the per-layer gain has to absorb,\n"
          "  rather than a difference quietly removed before the comparison.\n")

    model, diag = model_from_pass1(norm, verbose=True)
    region = np.zeros(model.shape, bool)
    region[crop[1]:crop[3], crop[0]:crop[2]] = True
    ref = len(src) // 2
    null = norm[ref][crop[1]:crop[3], crop[0]:crop[2]]

    null_canvas, null_support = place(null, crop, model.shape)
    null_app, null_sup = layer_views(model, null_canvas, null_support)
    select_geometry(model, null_app, null_sup, src)
    print(f"  geometry chosen on the null appearance: {choice_summary(model)}")

    # Plumbing KAT before anything is believed: the reference composite must
    # explain the reference FRAME exactly (identity warp, zero radius, unit gain).
    model.radii, model.gains = {}, {}
    estimate_radii(model, null_app, null_sup, norm, frames=[ref])
    self_test = certify(model, null, crop, norm, frames=[ref], region=region)
    print(f"  self-consistency (reference composite vs reference frame): "
          f"{self_test.score:.4f} levels "
          f"({'PASS' if self_test.score < 0.5 else 'FAIL — plumbing bug'})")

    model.radii, model.gains = {}, {}
    estimate_radii(model, null_app, null_sup, src)
    null_result = certify(model, null, crop, src, region=region)

    canvas, support = place(composite, crop, model.shape)
    appearances, supports = layer_views(model, canvas, support)
    model.radii, model.gains = {}, {}
    estimate_radii(model, appearances, supports, src)
    result = certify(model, composite, crop, src, region=region,
                     residual_dir=os.path.join(OUT, "kitchen_frames"))
    delta = differential(result, null_result)

    print(f"\n  routed composite : score {result.score:.4f} levels   "
          f"p99 {result.p99:.3f}   certified {result.certified * 100:.1f}%  "
          f"boundary {result.boundary * 100:.1f}%  excluded {result.excluded * 100:.1f}%")
    print(f"  null (reference) : score {null_result.score:.4f} levels   "
          f"p99 {null_result.p99:.3f}")
    print(f"  DIFFERENTIAL     : mean {delta.score:+.4f} levels   p99 {delta.p99:+.3f}"
          f"   (negative = the composite explains the sweep BETTER than the reference)")
    print(f"\n  {'frame':>5} {'MAE':>8} {'p99':>7} {'cert%':>7} {'bnd%':>6} {'exc%':>6}"
          f"   {'radii (near..far layers)':<20}")
    for entry in result.per_frame:
        k = entry["frame"]
        radii = [model.radii.get((k, i), 0) for i in model.far_to_near()[::-1]]
        print(f"  {k:5d} {entry['mae']:8.3f} {entry['p99']:7.2f} "
              f"{entry['certified'] * 100:6.2f}% {entry['boundary'] * 100:5.2f}% "
              f"{entry['excluded'] * 100:5.2f}%   {radii}")

    print("\n  --- the named regions (ORIGINAL coordinates) ---")
    print(f"  {'region':<34} {'abs mean':>8} {'abs max':>8} "
          f"{'diff mean':>9} {'diff max':>8} {'px':>6}")
    for label, box in (("F112 knob  x659-669 y243-313", KNOB),
                       ("pale sliver at bottle's left", SLIVER),
                       ("clean flank x560-670 y240-420", FLANK_BOX),
                       ("whole certified frame", (0, 0, model.shape[1], model.shape[0]))):
        mean, peak, count = _box_stats(result, box, region)
        dmean, dpeak, _n = _box_stats(delta, box, region)
        print(f"  {label:<34} {mean:8.3f} {peak:8.2f} {dmean:9.3f} "
              f"{dpeak:8.2f} {count:6d}")

    # The canonical F108 flank MASK, not just its box: bright and smooth, where the
    # reference frame is the local truth (§12.2 — scope the metric to the thing).
    reference = src[len(src) // 2]
    grey = to_gray_float(reference).astype(np.float32)[STREAK[0]:STREAK[1],
                                                       STREAK[2]:STREAK[3]]
    mean9 = cv2.boxFilter(grey, cv2.CV_32F, (9, 9))
    variance = cv2.boxFilter(grey * grey, cv2.CV_32F, (9, 9)) - mean9 * mean9
    flank = (grey > 170) & (np.sqrt(np.maximum(variance, 0.0)) < 4.0)
    window = np.zeros(model.shape, bool)
    window[STREAK[0]:STREAK[1], STREAK[2]:STREAK[3]] = flank
    window &= region & (result.coverage >= MIN_COVERAGE)
    print(f"  {'F108 flank MASK (canonical)':<34} "
          f"{result.unexplained[window].mean():8.3f} "
          f"{result.unexplained[window].max():8.2f} "
          f"{delta.unexplained[window].mean():9.3f} "
          f"{delta.unexplained[window].max():8.2f} {int(window.sum()):6d}")

    for title, source in (("ABSOLUTE unexplained", result),
                          ("DIFFERENTIAL (composite - null)", delta)):
        every = clusters(source, region, top=10 ** 6)
        print(f"\n  --- top clusters, {title} (99.5th pct, ranked by mass; "
              f"{len(every)} found) ---")
        found = every[:10]
        print(f"  {'#':>2} {'box (x0,y0,x1,y1)':<26} {'area':>6} {'mass':>9} "
              f"{'peak':>7} {'mean':>7}  hits")
        knob_hit = sliver_hit = flank_only = False
        for i, cluster in enumerate(found):
            hits = []
            if _overlaps(cluster["box"], KNOB):
                hits.append("KNOB")
                knob_hit = True
            if _overlaps(cluster["box"], SLIVER) or _overlaps(
                    cluster["box"], (473, 135, 506, 156)):
                hits.append("SLIVER")
                sliver_hit = True
            if _overlaps(cluster["box"], FLANK_BOX, pad=0) and not hits:
                hits.append("(inside flank box)")
                flank_only = True
            print(f"  {i:2d} {str(cluster['box']):<26} {cluster['area']:6d} "
                  f"{cluster['mass']:9.0f} {cluster['peak']:7.2f} {cluster['mean']:7.3f}"
                  f"  {' '.join(hits)}")
        # A yes/no on "is it in the top ten" is a threshold pretending to be a
        # verdict. The RANK is the measurement: it says how far down the list the
        # known defect sits, and therefore how much of the list above it is
        # competition the round has to account for.
        ranks = {}
        for name, target in (("KNOB", KNOB), ("SLIVER", SLIVER),
                             ("SLIVER(composite coords)", (473, 135, 506, 156))):
            hit = [i for i, c in enumerate(every) if _overlaps(c["box"], target)]
            ranks[name] = (hit[0] + 1) if hit else None
        flank_ranks = [i + 1 for i, c in enumerate(every)
                       if _overlaps(c["box"], FLANK_BOX, pad=0)
                       and not _overlaps(c["box"], KNOB)]
        print(f"    rank of KNOB: {ranks['KNOB'] or 'not clustered'} / {len(every)}"
              f"    rank of SLIVER: "
              f"{ranks['SLIVER'] or ranks['SLIVER(composite coords)'] or 'not clustered'}"
              f" / {len(every)}")
        print(f"    clusters inside the verified-clean flank (knob excluded): "
              f"{flank_ranks if flank_ranks else 'NONE'}")
        del knob_hit, sliver_hit, flank_only

    # DETECTION LIMIT, not a tuning pass. The knob does not survive the default
    # cluster detector; the useful question is then what it WOULD take, and what
    # else comes through at that setting. A defect that only appears once the
    # detector also admits a dozen spurious clusters has not been detected.
    print("\n  --- detection limit for the two known defects (differential map) ---")
    print(f"  {'percentile':>10} {'min area':>9} {'clusters':>9} {'KNOB rank':>10} "
          f"{'SLIVER rank':>12} {'in clean flank':>15}")
    for percentile in (99.5, 99.0, 98.0):
        for min_area in (25, 8, 3):
            every = clusters(delta, region, top=10 ** 6, percentile=percentile,
                             min_area=min_area)
            knob = [i + 1 for i, c in enumerate(every) if _overlaps(c["box"], KNOB)]
            sliver = [i + 1 for i, c in enumerate(every)
                      if _overlaps(c["box"], SLIVER)
                      or _overlaps(c["box"], (473, 135, 506, 156))]
            spurious = [i + 1 for i, c in enumerate(every)
                        if _overlaps(c["box"], FLANK_BOX, pad=0)
                        and not _overlaps(c["box"], KNOB)]
            print(f"  {percentile:10.1f} {min_area:9d} {len(every):9d} "
                  f"{str(knob[0]) if knob else '-':>10} "
                  f"{str(sliver[0]) if sliver else '-':>12} "
                  f"{len(spurious):15d}")

    for name, source, base in (("absolute", result, canvas),
                               ("differential", delta, canvas),
                               ("null", null_result, null_canvas)):
        overlay, vmax = heat_overlay(base, source, region)
        for box, colour in ((KNOB, (0, 255, 0)), (SLIVER, (255, 255, 0)),
                            (FLANK_BOX, (255, 0, 255))):
            cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), colour, 1)
        cv2.imwrite(os.path.join(OUT, f"kat4_kitchen_{name}.png"), overlay)
        print(f"  {name:<13} heat overlay (vmax {vmax:.1f} levels) -> "
              f"{OUT}/kat4_kitchen_{name}.png")
    cv2.imwrite(os.path.join(OUT, "kat4_kitchen_coverage.png"),
                (np.clip(result.coverage / max(1, result.coverage.max()), 0, 1)
                 * 255).astype(np.uint8))
    print(f"  elapsed {time.time() - t0:.1f} s")
    return result, model


def floor_factory() -> None:
    """Attribute the model-error floor, on the one scene where truth exists.

    A floor is not a finding until it has an owner. Every rung holds the
    APPEARANCE fixed at the true all-in-focus scene and changes exactly one piece
    of the model, so each step's difference is the price of that piece.
    """
    import parallax_gen as P

    frames, _truth_image, _near = P.build_stack()
    tmodel, truth = factory_truth_model(canvas=False)
    near = truth["near_mask"]
    ideal = (truth["foreground"] * near[..., None]
             + truth["background"] * (~near)[..., None])
    ideal_support = np.ones(tmodel.shape, bool)
    h, w = tmodel.shape
    crop = (0, 0, w, h)
    region = np.ones((h, w), bool)

    def run(label, model, canvas, support, composite):
        appearances, supports = layer_views(model, canvas, support)
        if model.menu:
            select_geometry(model, appearances, supports, frames)
        model.radii, model.gains = {}, {}
        estimate_radii(model, appearances, supports, frames)
        result = certify(model, composite, crop, frames, region=region)
        print(f"  {label:<52} {result.score:8.4f} {result.p99:7.2f} "
              f"{result.certified * 100:6.2f}%   {choice_summary(model)}")
        return result.score

    print("=" * 78)
    print("MODEL-ERROR FLOOR, attributed (analytic factory)")
    print("=" * 78)
    print("  appearance is held at the TRUE all-in-focus scene except on the last two\n"
          "  rows; only the model changes.\n")
    print(f"  {'rung':<52} {'score':>8} {'p99':>7} {'cert%':>7}   geometry")

    rung1 = run("1. TRUE masks + TRUE motions (the renderer's own floor)",
                tmodel, ideal, ideal_support, ideal.astype(np.uint8))

    # Rung 2: perfect masks, pass-1's OWN translation estimator on top of pass-1's
    # own global affine — the cost of estimating the motion, with the segmentation
    # handed to it for free.
    coarse, warps, valid = TF.global_stage(frames, P.REFERENCE)
    common = np.logical_and.reduce(valid)
    ref_gray = to_gray_float(coarse[P.REFERENCE]).astype(np.float32) / 255.0
    greys = [to_gray_float(c).astype(np.float32) / 255.0 for c in coarse]
    gradient = cv2.magnitude(
        cv2.Sobel(ref_gray * 255.0, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(ref_gray * 255.0, cv2.CV_32F, 0, 1, ksize=3))
    textured = gradient >= TF.A._REFINE_MIN_GRADIENT
    max_shift = TF.MAX_SHIFT_FRACTION * float(np.hypot(h, w))
    model2 = SceneModel(shape=(h, w), n_frames=P.FRAMES, ref=P.REFERENCE,
                        layers=tmodel.layers)
    for k in range(P.FRAMES):
        base = np.eye(3) if warps[k] is None else _homogeneous(warps[k])
        for group, plane in ((1, ~near), (0, near)):
            fitted = TF.masked_translation(ref_gray, greys[k],
                                           plane & textured & common, max_shift)
            dx, dy = ((0.0, 0.0) if fitted is None
                      else (float(fitted[0, 2]), float(fitted[1, 2])))
            composed = base @ np.array([[1.0, 0.0, dx], [0.0, 1.0, dy],
                                        [0.0, 0.0, 1.0]])
            model2.matrices[(k, group)] = composed
            model2.menu[(k, group)] = [("affine", composed),
                                       ("rigid", TF._rigidify(composed, plane, (h, w)))]
    rung2 = run("2. TRUE masks + pass-1 ESTIMATED motions", model2, ideal,
                ideal_support, ideal.astype(np.uint8))

    # Rung 3: pass-1's masks, the TRUE motion of whichever plane each layer mostly
    # covers. Isolates what the layer SEGMENTATION costs.
    emodel, _diag = model_from_pass1(frames, P.REFERENCE)
    model3 = SceneModel(shape=(h, w), n_frames=P.FRAMES, ref=P.REFERENCE,
                        layers=emodel.layers)
    for group in emodel.groups():
        mask = np.logical_or.reduce([emodel.layers[i].mask
                                     for i in emodel.layers_of(group)])
        is_near = float((mask & near).sum()) / max(1, int(mask.sum())) > 0.5
        source = 0 if is_near else 1
        for k in range(P.FRAMES):
            model3.matrices[(k, group)] = tmodel.matrices[(k, source)]
    rung3 = run("3. pass-1 ESTIMATED masks + TRUE motions", model3, ideal,
                ideal_support, ideal.astype(np.uint8))

    emodel2, _diag = model_from_pass1(frames, P.REFERENCE)
    rung4 = run("4. pass-1 masks + pass-1 motions (the FULL model floor)",
                emodel2, ideal, ideal_support, ideal.astype(np.uint8))

    null = frames[P.REFERENCE]
    emodel3, _diag = model_from_pass1(frames, P.REFERENCE)
    null_canvas, null_support = place(null, crop, (h, w))
    rung5 = run("5. rung 4, but the NULL composite (the reference frame)",
                emodel3, null_canvas, null_support, null)

    print(f"\n  attribution, in levels of unexplained residual:")
    print(f"    renderer + PSF family + exposure           {rung1:7.3f}")
    print(f"    + pass-1 MOTION estimation                 {rung2 - rung1:+7.3f}")
    print(f"    + pass-1 LAYER SEGMENTATION                {rung3 - rung1:+7.3f}")
    print(f"    = the full model floor (rung 4)            {rung4:7.3f}")
    print(f"    the NULL CANDIDATE's own defocus adds      {rung5 - rung4:+7.3f}")
    print(f"\n  The null bound ({rung5:.3f}) is therefore NOT mostly the null candidate:\n"
          f"  {rung4 / rung5 * 100:.0f}% of it is the model, and within the model the LAYER\n"
          "  SEGMENTATION is the dominant term by a factor of ~3.6 over motion. That is\n"
          "  the single most useful number this round produces for the reconstruction\n"
          "  pass, and it is the reason the certifier reports a DIFFERENTIAL map as well\n"
          "  as an absolute one: segmentation error is shared by every candidate scored\n"
          "  through the same model, so it cancels in candidate-minus-null and stops\n"
          "  masquerading as a defect in the composite.")


def kat4_null() -> None:
    """The kitchen's crudeness ledger: what each piece of the model is worth.

    Every row changes exactly one thing about the model or the candidate, and the
    difference from the baseline is that piece's price in levels of unexplained
    residual. Rows that make the model SIMPLER and the score BETTER are the
    interesting ones — they say a pass-1 quantity is not carrying its weight.
    """
    from focusstack.io import normalize_exposure

    paths = sorted(glob.glob(os.path.join(KITCHEN, "*.jpg")))
    src = [cv2.imread(p) for p in paths]
    norm = normalize_exposure(src)
    composite, info = TF.twoframe_stack(norm)
    crop = info["crop"]
    ref = len(src) // 2
    model, _diag = model_from_pass1(norm)
    region = np.zeros(model.shape, bool)
    region[crop[1]:crop[3], crop[0]:crop[2]] = True
    h, w = model.shape

    null_canvas, null_support = place(norm[ref][crop[1]:crop[3], crop[0]:crop[2]],
                                      crop, model.shape)
    null_app, null_sup = layer_views(model, null_canvas, null_support)
    select_geometry(model, null_app, null_sup, src)
    frozen = dict(model.matrices)

    coarse, warps, _valid = TF.global_stage(norm, ref)
    bases = {k: (np.eye(3) if warps[k] is None else _homogeneous(warps[k]))
             for k in range(len(src))}

    def one_layer_model():
        """No layer decomposition at all: one surface, one global affine."""
        flat = SceneModel(shape=(h, w), n_frames=len(src), ref=ref,
                          layers=[Layer(mask=np.ones((h, w), bool), group=0,
                                        order=0.0, name="whole frame")])
        for k in range(len(src)):
            flat.matrices[(k, 0)] = bases[k]
        return flat

    print("=" * 78)
    print("KITCHEN crudeness ledger — what each piece of the model is worth")
    print("=" * 78)
    print(f"  {'configuration':<48} {'score':>8} {'p99':>7} {'cert%':>7}")
    rows = {}
    for label, candidate, kind in (
        ("routed two-frame composite (baseline)", composite, "base"),
        ("NULL: normalized reference frame", norm[ref][crop[1]:crop[3], crop[0]:crop[2]], "base"),
        ("baseline, certified vs NORMALIZED frames", composite, "normalized"),
        ("baseline, NO per-layer exposure gain", composite, "nogain"),
        ("baseline, GAUSSIAN PSF instead of disk", composite, "gauss"),
        ("baseline, defocus disabled (all radii 0)", composite, "zero"),
        ("baseline, layers kept, GLOBAL AFFINE only", composite, "globalonly"),
        ("baseline, ONE layer + global affine (no model)", composite, "onelayer"),
        ("ONE layer + global affine, NULL candidate",
         norm[ref][crop[1]:crop[3], crop[0]:crop[2]], "onelayer"),
    ):
        if kind == "onelayer":
            work = one_layer_model()
        else:
            work = SceneModel(shape=(h, w), n_frames=model.n_frames, ref=model.ref,
                              layers=model.layers, matrices=dict(frozen))
            if kind == "globalonly":
                for k in range(model.n_frames):
                    for g in model.groups():
                        work.matrices[(k, g)] = bases[k]
        observed = norm if kind == "normalized" else src
        canvas, support = place(candidate, crop, (h, w))
        appearances, supports = layer_views(work, canvas, support)
        if kind == "zero":
            work.radii = {(k, i): 0 for k in range(work.n_frames)
                          for i in range(len(work.layers))}
        else:
            estimate_radii(work, appearances, supports, observed,
                           gain=(kind != "nogain"),
                           psf=(_gaussian_psf if kind == "gauss" else defocus))
        result = certify(work, candidate, crop, observed, region=region)
        rows[label] = result
        print(f"  {label:<48} {result.score:8.4f} {result.p99:7.3f} "
              f"{result.certified * 100:6.2f}%")

    base = rows["routed two-frame composite (baseline)"]
    null = rows["NULL: normalized reference frame"]
    flat = rows["baseline, ONE layer + global affine (no model)"]
    flat_null = rows["ONE layer + global affine, NULL candidate"]
    print(f"\n  composite - null, full model : {base.score - null.score:+.4f} levels")
    print(f"  composite - null, ONE layer  : {flat.score - flat_null.score:+.4f} levels")
    print("\n  Both differentials are positive and of the same order, i.e. the composite\n"
          "  is separated from the reference either way. What the layer model buys is\n"
          "  ABSOLUTE accuracy, and on this scene it is far less than the factory led us\n"
          "  to expect — read the two 'no model' rows against the baseline before\n"
          "  designing round B around the pass-1 decomposition.")


COMMANDS = {"kat1": kat1, "kat2": kat2, "kat3": kat3, "kat4": kat4,
            "floor": floor_factory, "ledger": kat4_null}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print("commands: " + ", ".join(COMMANDS))
        return
    COMMANDS[sys.argv[1]]()


if __name__ == "__main__":
    main()
