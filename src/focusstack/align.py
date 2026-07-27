"""Stage 1 — registration (alignment).

Why this stage exists: when a lens refocuses, the image magnification changes
slightly (objects appear to grow/shrink) — this is called *focus breathing*. The
camera or subject may also shift a little between frames. If we fuse unaligned
frames, sharp detail from one frame lands on the wrong pixels of another and we
get ghosting/doubling. So first we warp every frame onto a common coordinate
frame (that of a chosen reference frame).

We use OpenCV's ECC (Enhanced Correlation Coefficient) algorithm, which directly
estimates the geometric warp that best aligns two images by maximizing their
correlation — no feature detection needed, and it is robust to the brightness
differences a focus change can introduce.

One global warp is not enough on handheld stacks. Pure rotation about the lens
entrance pupil would be a single depth-independent homography, but a hand pivots
the *device*, so the camera centre also translates; image displacement then
varies roughly with inverse scene depth and near and far objects move by
different amounts. No single affine or homography can absorb both — fitting one
splits the difference, leaving the near content misregistered by several pixels
while the far content is fine.

The default therefore adds a second, depth-aware pass on top of the global warp:

1. the stack's own depth-from-focus map is split into depth bins at the
   *valleys* of its histogram, so an edge lands where the scene separates in
   depth rather than through the middle of an object;
2. each bin gets its own small translation-only ECC correction, which is the
   most constrained model that can express parallax and cannot invent shear or
   scale the global warp already owns;
3. those corrections are blended into one dense coordinate field by edge-aware
   bin memberships, and the field is then relaxed wherever it would *stretch*
   content rather than merely transport it;
4. the frame is resampled exactly once through the composed field, so the extra
   accuracy costs no additional interpolation softening.

Every step above is refusable. A bin with no texture, too few pixels, or a
diverged fit keeps the global warp, and a frame that earns no correction at all
comes out byte-identical to the global-only aligner. Stacks of fewer than three
frames have no depth proxy and skip the pass entirely.

`depth_model="joint"` selects an experimental alternative that estimates camera
motion, scene depth, and the depth-to-parallax calibration in alternation. It
registers moving sweeps better than the binned model but is not yet
non-regressing on still ones; see `research/FINDINGS.md` (F81).
"""

from __future__ import annotations

import warnings

import cv2
import numpy as np

from .io import to_gray_float

# Motion models, from most constrained to most general. `affine` (translation +
# rotation + scale + shear) is a good default because it captures focus breathing
# (scale) plus small camera motion without the extra freedom of a full homography.
_MOTION_MODES = {
    "translation": cv2.MOTION_TRANSLATION,
    "euclidean": cv2.MOTION_EUCLIDEAN,  # translation + rotation only
    "affine": cv2.MOTION_AFFINE,
    "homography": cv2.MOTION_HOMOGRAPHY,  # full perspective (8 DOF)
}

# Depth-binned refinement. The per-bin model is translation only: for a camera
# centre displaced by t, a scene point at depth Z shifts by roughly f*t/Z, so
# within a narrow depth band the parallax residual is close to constant. Two
# degrees of freedom per bin is also the strongest regularizer available — it
# cannot invent local shear or scale that the global warp already owns.
_REFINE_MAX_ITERATIONS = 200
_REFINE_EPS = 1e-5
# A refinement larger than this fraction of the image diagonal is treated as an
# ECC failure rather than parallax: real residual parallax after a global fit is
# a few pixels, while a diverged bin fit is arbitrarily large.
_REFINE_MAX_FRACTION = 0.015
# Bins holding fewer than this share of valid pixels do not get their own fit.
_REFINE_MIN_BIN_FRACTION = 0.06
# A bin with no texture cannot support a correlation fit at all.
_REFINE_MIN_GRADIENT = 1.0
# Edge-aware smoothing of the bin memberships, as a fraction of the diagonal.
# This is the field's transition width, and it trades two failure modes against
# each other: too wide smears a near object's correction across its own
# silhouette (visible stretching), too narrow leaves the correction speckled by
# depth-map noise. A real depth discontinuity SHOULD stay sharp — the parallax
# jump there is physical — so the guided filter's edge awareness, not the
# radius, is what carries smooth depth gradients.
_REFINE_MEMBERSHIP_FRACTION = 0.015
_REFINE_MEMBERSHIP_EPS = 1e-3
# Largest local field gradient tolerated before the displacement is relaxed. A
# sampling field may TRANSPORT content freely; what it must not do is stretch
# it, because that is geometry the camera never saw.
_REFINE_STRETCH_TOLERANCE = 0.10
# Widest disocclusion ribbon considered; beyond this the frame is so displaced
# that the binned model is out of its depth anyway.
_OCCLUSION_MAX_RADIUS = 24
# How abruptly depth must change to count as one surface passing in front of
# another, as a fraction of the full depth range. Absolute on purpose: it is a
# claim about the scene, so it must not loosen just because more bins were asked
# for. A gradual ramp fails it at every bin count.
_OCCLUSION_MIN_DEPTH_STEP = 0.10
# Joint motion/depth estimation. Tiles are the observation unit: enough of them
# to constrain seven parameters robustly, each large enough for phase
# correlation to be reliable.
_TILE_GRID = (6, 8)
_TILE_MIN_TEXTURE = 0.02
_TILE_MIN_RESPONSE = 0.10
_MIN_TILE_OBSERVATIONS = 12
_IRLS_PASSES = 3
_JOINT_ITERATIONS = 3
_RIDGE = 1e-3
_JOINT_INNER = 3


def _largest_valid_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return the largest axis-aligned rectangle containing only true pixels."""
    valid = np.asarray(mask, dtype=bool)
    if valid.ndim != 2:
        raise ValueError("validity mask must be two-dimensional")

    h, w = valid.shape
    heights = np.zeros(w, dtype=np.int32)
    best_area = 0
    best = (0, 0, 0, 0)

    for y in range(h):
        heights = np.where(valid[y], heights + 1, 0)
        stack: list[tuple[int, int]] = []
        for x in range(w + 1):
            height = int(heights[x]) if x < w else 0
            start = x
            while stack and stack[-1][1] > height:
                x0, previous = stack.pop()
                area = previous * (x - x0)
                if area > best_area:
                    best_area = area
                    best = (x0, y - previous + 1, x, y + 1)
                start = x0
            if height and (not stack or stack[-1][1] < height):
                stack.append((start, height))

    if best_area == 0:
        raise ValueError("aligned frames have no common valid image footprint")
    return best


def _homogeneous(warp: np.ndarray) -> np.ndarray:
    """Promote a 2x3 affine warp to a 3x3 matrix; pass 3x3 through."""
    if warp.shape == (3, 3):
        return warp.astype(np.float64)
    full = np.eye(3, dtype=np.float64)
    full[:2] = warp
    return full


def _matrix_field(matrix: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Dense sampling grid for a single homogeneous warp."""
    return _blended_coordinate_maps(
        [matrix], [np.ones(shape, dtype=np.float32)], shape
    )


def _tile_observations(
    ref_gray: np.ndarray,
    moving_gray: np.ndarray,
    valid: np.ndarray,
    depth: np.ndarray,
    max_shift: float,
    grid: tuple[int, int] = _TILE_GRID,
) -> list[tuple[float, float, float, float, float, float]]:
    """Sample the residual displacement field on a grid of tiles.

    Phase correlation gives each fully observed, textured tile a subpixel shift
    plus a response that doubles as a confidence weight, which is far cheaper
    than running ECC per tile and gives the motion fit many more constraints
    than a handful of depth bins could. Each observation carries the tile's own
    median depth, because depth is what the fit needs in order to tell
    rotation apart from camera-centre translation.
    """
    h, w = ref_gray.shape
    rows, columns = grid
    tile_h, tile_w = h // rows, w // columns
    if tile_h < 24 or tile_w < 24:
        return []
    window = cv2.createHanningWindow((tile_w, tile_h), cv2.CV_64F)

    observations = []
    for r in range(rows):
        for c in range(columns):
            y0, x0 = r * tile_h, c * tile_w
            box = (slice(y0, y0 + tile_h), slice(x0, x0 + tile_w))
            if not valid[box].all():
                continue
            reference_tile = ref_gray[box]
            if float(reference_tile.std()) < _TILE_MIN_TEXTURE:
                continue
            shift, response = cv2.phaseCorrelate(
                np.ascontiguousarray(reference_tile.astype(np.float64)),
                np.ascontiguousarray(moving_gray[box].astype(np.float64)),
                window,
            )
            if response < _TILE_MIN_RESPONSE:
                continue
            if float(np.hypot(*shift)) > max_shift:
                continue
            observations.append(
                (
                    x0 + tile_w / 2.0,
                    y0 + tile_h / 2.0,
                    float(np.median(depth[box])),
                    float(shift[0]),
                    float(shift[1]),
                    float(response),
                )
            )
    return observations


def _motion_design(x: float, y: float, u: float) -> tuple[list[float], list[float]]:
    """Rows of the small-motion flow model for one observation.

    Parameters are ``[wx, wy, wz, tx, ty, tz, s]``: three rotation rates, three
    camera-centre translations, and one focus-breathing scale. Rotation and
    breathing act the same at every depth; translation is multiplied by the
    inverse-depth proxy ``u``. That single difference is what lets the fit
    separate them — notably forward translation ``tz`` from breathing ``s``,
    which are both radial and are otherwise indistinguishable.
    """
    return (
        [x * y, -(1.0 + x * x), y, -u, 0.0, u * x, x],
        [1.0 + y * y, -x * y, -x, 0.0, -u, u * y, y],
    )


def _fit_motion_model(
    observations: list[tuple[float, float, float, float, float, float]],
    shape: tuple[int, int],
) -> np.ndarray | None:
    """Robustly fit the seven motion parameters to the tile observations."""
    if len(observations) < _MIN_TILE_OBSERVATIONS:
        return None
    h, w = shape
    scale = float(np.hypot(h, w))
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    # Centre the depth coordinate. Rotation acts at every depth and translation
    # acts in proportion to depth, but if depth never approaches zero in the
    # data those two bases are nearly parallel, and the solver answers with two
    # large opposed terms that cancel where the tiles are and diverge where they
    # are not. Measuring depth relative to the scene's own middle makes the
    # depth-independent and depth-varying parts genuinely separate questions.
    origin = float(np.median([o[2] for o in observations]))

    design, target, confidence = [], [], []
    for px, py, u, dx, dy, response in observations:
        x, y = (px - cx) / scale, (py - cy) / scale
        row_u, row_v = _motion_design(x, y, u - origin)
        design.append(row_u)
        target.append(dx / scale)
        design.append(row_v)
        target.append(dy / scale)
        confidence.extend([response, response])

    design = np.asarray(design, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    weights = np.asarray(confidence, dtype=np.float64)

    params = None
    for _ in range(_IRLS_PASSES):
        root = np.sqrt(weights)[:, None]
        weighted = design * root
        # Ridge term: prefer the smallest motion that explains the tiles, so a
        # poorly constrained direction returns ~0 rather than a wild guess.
        normal = weighted.T @ weighted
        ridge = _RIDGE * float(np.trace(normal)) / normal.shape[0]
        try:
            solution = np.linalg.solve(
                normal + ridge * np.eye(normal.shape[0]),
                weighted.T @ (target * root.ravel()),
            )
        except np.linalg.LinAlgError:
            return None
        if not np.isfinite(solution).all():
            return None
        params = solution
        residual = np.abs(design @ params - target)
        # Huber reweighting: a tile straddling a depth step reports a shift that
        # belongs to neither side, and must not drag the whole fit.
        cutoff = max(float(np.median(residual)) * 2.0, 1e-6)
        weights = np.asarray(confidence) * np.minimum(1.0, cutoff / np.maximum(residual, 1e-9))
    return np.concatenate([params, [origin]])


def _motion_displacement(
    params: np.ndarray,
    depth: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the fitted motion model at every pixel's own depth."""
    h, w = shape
    scale = float(np.hypot(h, w))
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64)
    )
    x = (grid_x - cx) / scale
    y = (grid_y - cy) / scale
    wx, wy, wz, tx, ty, tz, s, origin = params
    u = depth.astype(np.float64) - origin

    dx = wx * x * y - wy * (1.0 + x * x) + wz * y + u * (-tx + x * tz) + s * x
    dy = wx * (1.0 + y * y) - wy * x * y - wz * x + u * (-ty + y * tz) + s * y
    return (dx * scale).astype(np.float32), (dy * scale).astype(np.float32)


def _compose_field(
    map_x: np.ndarray,
    map_y: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold a newly measured residual into an existing sampling field.

    The field already maps reference coordinates to the ORIGINAL frame, so a
    residual measured against the currently-aligned frame is applied by
    resampling the field itself. Only coordinates are resampled here; the image
    is interpolated exactly once, at the end, however many iterations run.
    """
    h, w = map_x.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
    )
    sample_x = (grid_x + dx).astype(np.float32)
    sample_y = (grid_y + dy).astype(np.float32)
    common = dict(interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return (
        cv2.remap(map_x, sample_x, sample_y, **common),
        cv2.remap(map_y, sample_x, sample_y, **common),
    )


def _valley_edges(
    samples: np.ndarray,
    bins: int,
    resolution: int = 96,
) -> np.ndarray | None:
    """Bin edges at the most prominent minima of the depth histogram.

    Prominence — how far a minimum sits below the lower of the two peaks
    flanking it — ranks candidate splits by how genuinely separated the depths
    on either side are, so a shallow ripple inside one object's depth spread
    never outranks the real gap between an object and its background. Returns
    None when the histogram has no usable valley, leaving the caller on
    quantiles.
    """
    if bins < 2:
        return None
    histogram, edges = np.histogram(samples, bins=resolution, range=(0.0, 1.0))
    smoothed = cv2.GaussianBlur(
        histogram.astype(np.float32).reshape(1, -1), (9, 1), 2.0
    ).ravel()

    candidates = []
    for i in range(1, resolution - 1):
        if smoothed[i] <= smoothed[i - 1] and smoothed[i] <= smoothed[i + 1]:
            left = float(smoothed[:i].max())
            right = float(smoothed[i + 1:].max())
            prominence = min(left, right) - float(smoothed[i])
            if prominence > 0.0:
                candidates.append((prominence, i))
    if not candidates:
        return None

    candidates.sort(reverse=True)
    chosen = sorted(index for _, index in candidates[: bins - 1])
    centres = (edges[:-1] + edges[1:]) / 2.0
    return np.unique(
        np.concatenate([[0.0], centres[chosen], [1.0]])
    )


def _depth_bin_masks(
    depth: np.ndarray,
    valid: np.ndarray,
    bins: int,
) -> list[np.ndarray]:
    """Split the depth proxy into bins that follow the scene's own depth structure.

    Bin edges are placed in the *valleys* of the depth histogram, not at
    quantiles. This matters more than it sounds: an equal-population edge lands
    wherever a third of the pixels happen to fall, which can be straight through
    a single physical object. Because each bin then receives its own translation,
    such an edge puts a many-pixel discontinuity inside one rigid surface, and
    the result is a seam and a ghosted strip across the middle of the object.
    Valleys are where the scene genuinely separates in depth, which is exactly
    where a displacement step belongs. Quantiles remain the fallback for scenes
    with no clear depth structure.
    """
    samples = depth[valid]
    if samples.size == 0:
        return []
    edges = _valley_edges(samples, bins)
    if edges is None:
        edges = np.unique(np.quantile(samples, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 3:
        # A degenerate depth map (one focal plane owns everything) carries no
        # depth-dependent motion to correct.
        return []

    masks = []
    for k in range(edges.size - 1):
        low, high = edges[k], edges[k + 1]
        if k == edges.size - 2:
            masks.append(valid & (depth >= low) & (depth <= high))
        else:
            masks.append(valid & (depth >= low) & (depth < high))
    return masks


def _residual_translation(
    ref_gray: np.ndarray,
    moving_gray: np.ndarray,
    mask: np.ndarray,
    max_shift: float,
) -> np.ndarray | None:
    """Estimate one depth bin's leftover translation, or None if unsupportable.

    `moving_gray` is already globally aligned, so the fit starts at identity and
    only has to find a small residual. The mask is defined in reference
    coordinates, which the global pass has made a good approximation of the
    moving frame's coordinates too.
    """
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        _REFINE_MAX_ITERATIONS,
        _REFINE_EPS,
    )
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(
            ref_gray,
            moving_gray,
            warp,
            cv2.MOTION_TRANSLATION,
            criteria,
            mask.astype(np.uint8) * 255,
            5,
        )
    except cv2.error:
        return None

    if not np.isfinite(warp).all():
        return None
    if float(np.hypot(warp[0, 2], warp[1, 2])) > max_shift:
        return None
    return warp


def _bin_is_textured(gradient: np.ndarray, mask: np.ndarray) -> bool:
    """A correlation fit needs structure; a blank wall bin cannot vote."""
    if not mask.any():
        return False
    return float(gradient[mask].mean()) >= _REFINE_MIN_GRADIENT


def _blended_coordinate_maps(
    matrices: list[np.ndarray],
    weights: list[np.ndarray],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Combine per-bin warps into one dense sampling grid.

    Blending the *coordinates* (not the warped images) is what keeps this to a
    single resample: every output pixel gets one source location, interpolated
    once. It also regularizes the field — a bin's correction fades smoothly into
    its neighbours instead of tearing at the bin edge.
    """
    h, w = shape
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
    )
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    for matrix, weight in zip(matrices, weights):
        xs = matrix[0, 0] * grid_x + matrix[0, 1] * grid_y + matrix[0, 2]
        ys = matrix[1, 0] * grid_x + matrix[1, 1] * grid_y + matrix[1, 2]
        ws = matrix[2, 0] * grid_x + matrix[2, 1] * grid_y + matrix[2, 2]
        ws = np.where(np.abs(ws) < 1e-12, 1e-12, ws)
        map_x += weight * (xs / ws).astype(np.float32)
        map_y += weight * (ys / ws).astype(np.float32)
    return map_x, map_y


def _limit_field_stretch(
    map_x: np.ndarray,
    map_y: np.ndarray,
    base_x: np.ndarray,
    base_y: np.ndarray,
    tolerance: float,
    iterations: int = 6,
    sigma: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Relax the field wherever it would distort content, not merely move it.

    Two frames of a handheld stack genuinely disagree by a step at a depth
    discontinuity — the near object moves and the background does not — and no
    single-valued sampling field can express a step without either stretching
    content across it or tearing. Stretching is the visible failure: an object's
    own silhouette smears. So displacement is smoothed selectively, only where
    its local gradient exceeds `tolerance`, which trades a little residual
    misregistration in a thin band around depth steps for leaving no region
    geometrically distorted. Well-behaved interiors are untouched.
    """
    disp_x = map_x - base_x
    disp_y = map_y - base_y
    ksize = int(2 * round(3 * sigma) + 1)
    for _ in range(iterations):
        gradient = np.maximum(
            np.maximum(np.abs(np.gradient(disp_x, axis=1)), np.abs(np.gradient(disp_x, axis=0))),
            np.maximum(np.abs(np.gradient(disp_y, axis=1)), np.abs(np.gradient(disp_y, axis=0))),
        )
        excess = np.clip((gradient - tolerance) / max(tolerance, 1e-6), 0.0, 1.0)
        if float(excess.max()) <= 0.0:
            break
        excess = cv2.GaussianBlur(excess.astype(np.float32), (ksize, ksize), sigma)
        smooth_x = cv2.GaussianBlur(disp_x, (ksize, ksize), sigma)
        smooth_y = cv2.GaussianBlur(disp_y, (ksize, ksize), sigma)
        disp_x = (1.0 - excess) * disp_x + excess * smooth_x
        disp_y = (1.0 - excess) * disp_y + excess * smooth_y
    return (base_x + disp_x).astype(np.float32), (base_y + disp_y).astype(np.float32)


def _occlusion_mask(
    displacement_x: np.ndarray,
    displacement_y: np.ndarray,
    depth_step: np.ndarray,
    tolerance: float = 1.0,
) -> np.ndarray:
    """Pixels this frame cannot legitimately supply, because nothing was behind.

    Lateral camera motion does not merely shift a near object; it swings the
    object across the background, uncovering scene on one side and hiding it on
    the other. Those pixels have no correspondence at all — the observation
    simply does not exist in this frame — so no warp, however good, can produce
    them, and any value there is interpolated from the wrong surface. F80 threw
    away invented data at the outer border; this is the same rule applied where
    the invention happens in the interior.

    The test is geometric on purpose. Photometric agreement cannot be used in a
    focus stack, because frames legitimately disagree wherever defocus differs,
    which is everywhere that matters. Instead: if the displacement disagrees by
    more than `tolerance` pixels within a neighbourhood, then one sampling field
    is being asked to serve two surfaces moving differently, and the pixels
    between them belong to whichever surface wins — not reliably to this frame.

    Feed this the MEASURED per-region displacement, not the field that was
    finally applied. Uncovering is a fact about the scene and the camera, so it
    happened whether or not the correction chose to model it; a conservatively
    smoothed field would otherwise report that nothing was ever uncovered.

    `depth_step` is required and does real work: binning turns a smoothly
    receding surface, like a countertop running away from the camera, into a
    staircase of displacements, and those manufactured risers are not occlusion
    boundaries. A continuous surface hides nothing behind itself. Only where the
    scene's depth genuinely jumps can one surface pass in front of another, so
    the ribbon is admitted only near a real discontinuity.

    The ribbon's width is not free either: a foreground that moves Q px relative
    to its background uncovers a strip exactly Q px wide, so a pixel is at risk
    only if it lies within Q of a step of height Q. Testing that at one scale
    would either miss narrow ribbons or condemn whole regions around wide ones,
    so it is tested at a ladder of radii and a pixel fails if it fails any of
    them. Small steps therefore cost a couple of pixels, not a neighbourhood.
    """
    displacement_x = np.ascontiguousarray(displacement_x, dtype=np.float32)
    displacement_y = np.ascontiguousarray(displacement_y, dtype=np.float32)

    edge = np.ascontiguousarray(depth_step, dtype=np.uint8)
    if not edge.any():
        return np.zeros(displacement_x.shape, dtype=bool)

    mask = np.zeros(displacement_x.shape, dtype=bool)
    radius = 1
    while radius <= _OCCLUSION_MAX_RADIUS:
        window = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        spread = np.maximum(
            cv2.dilate(displacement_x, window) - cv2.erode(displacement_x, window),
            cv2.dilate(displacement_y, window) - cv2.erode(displacement_y, window),
        )
        # Within `radius` of a step at least `radius` tall, and within reach of
        # a real depth discontinuity. Never below the subpixel floor, where a
        # "step" is just interpolation noise.
        mask |= (spread > max(float(radius), tolerance)) & (cv2.dilate(edge, window) > 0)
        radius *= 2
    return mask


def _field_stretch(map_x: np.ndarray, map_y: np.ndarray) -> float:
    """Worst local stretch the sampling field imposes, as a deviation from 1.

    A blended field distorts content wherever the correction changes across
    space. This is the direct audit of that: a value near 0 means every region
    is transported rigidly, while a large value means some region is being
    stretched or compressed, which shows up as smeared or torn edges. The
    99.9th percentile ignores the single-pixel corners of the field.
    """
    dxdx = np.gradient(map_x, axis=1) - 1.0
    dxdy = np.gradient(map_x, axis=0)
    dydx = np.gradient(map_y, axis=1)
    dydy = np.gradient(map_y, axis=0) - 1.0
    deviation = np.maximum(
        np.maximum(np.abs(dxdx), np.abs(dydy)),
        np.maximum(np.abs(dxdy), np.abs(dydx)),
    )
    return float(np.percentile(deviation, 99.9))


def _linear_depth_field(
    base: np.ndarray,
    depth: np.ndarray,
    centers: list[float],
    shifts: list[tuple[float, float] | None],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit displacement as a straight line in the depth proxy, then go dense.

    The bins are a robust way to *sample* how displacement varies with depth,
    but they are an arbitrary quantization of a continuous quantity, and a
    piecewise-constant field tears at every bin edge. The physics is smoother
    than that: a camera centre displaced by t moves a point at depth Z by about
    f*t/Z, so displacement is linear in inverse depth. A phone focal sweep steps
    approximately uniformly in diopters (1/Z), which makes the normalized
    focus-winner index a usable stand-in for inverse depth — so fitting
    ``d = a + b*u`` across the bin samples turns three noisy measurements into a
    two-parameter model, and evaluating it at every pixel's own depth gives a
    dense field that is exactly as smooth as the depth map: rigid inside
    objects, sharp only where depth genuinely steps.
    """
    samples = [(u, s) for u, s in zip(centers, shifts) if s is not None]
    if len(samples) < 2:
        return None

    u = np.array([s[0] for s in samples], dtype=np.float64)
    design = np.stack([np.ones_like(u), u], axis=1)
    dx = np.array([s[1][0] for s in samples], dtype=np.float64)
    dy = np.array([s[1][1] for s in samples], dtype=np.float64)
    coefficients_x, *_ = np.linalg.lstsq(design, dx, rcond=None)
    coefficients_y, *_ = np.linalg.lstsq(design, dy, rcond=None)

    shift_x = (coefficients_x[0] + coefficients_x[1] * depth).astype(np.float32)
    shift_y = (coefficients_y[0] + coefficients_y[1] * depth).astype(np.float32)

    h, w = depth.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
    )
    # Residual first (reference coordinates -> coarse coordinates), then the
    # global warp, so a single resample reaches the original frame.
    res_x = grid_x + shift_x
    res_y = grid_y + shift_y
    map_x = (base[0, 0] * res_x + base[0, 1] * res_y + base[0, 2]).astype(np.float32)
    map_y = (base[1, 0] * res_x + base[1, 1] * res_y + base[1, 2]).astype(np.float32)
    denominator = base[2, 0] * res_x + base[2, 1] * res_y + base[2, 2]
    denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
    return (map_x / denominator).astype(np.float32), (map_y / denominator).astype(np.float32)


def _fit_inverse_depth(
    observations: dict[int, list],
    motions: dict[int, np.ndarray],
    shape: tuple[int, int],
    bins: int,
    previous: np.ndarray,
) -> np.ndarray:
    """Re-estimate each depth bin's inverse depth from how far its tiles moved.

    The focus-winner index says which bin is nearer than which, but not by how
    much: focal steps are not uniform in 1/Z, so displacement is monotone in the
    index and not proportional to it. Assuming proportionality is what made the
    parametric fits fail. Here the mapping is measured instead — given each
    frame's motion, every bin's inverse depth is whatever scalar best explains
    the displacement its own tiles actually showed, pooled across the stack.
    Scale is unidentifiable (only the product of depth and translation is
    observable), so the result is renormalized and the translations absorb it.
    """
    h, w = shape
    scale = float(np.hypot(h, w))
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    numerator = np.zeros(bins, dtype=np.float64)
    denominator = np.zeros(bins, dtype=np.float64)
    for index, frame_observations in observations.items():
        params = motions.get(index)
        if params is None:
            continue
        wx, wy, wz, tx, ty, tz, _s, _origin = params
        for px, py, tile_bin, dx, dy, weight in frame_observations:
            # Tiles carry the median bin index of their pixels, which lands
            # between bins when a tile straddles a depth step.
            bin_index = int(np.clip(round(tile_bin), 0, bins - 1))
            x, y = (px - cx) / scale, (py - cy) / scale
            # Everything the motion explains without depth, removed first.
            flat_u = wx * x * y - wy * (1.0 + x * x) + wz * y + _s * x
            flat_v = wx * (1.0 + y * y) - wy * x * y - wz * x + _s * y
            coefficient_u = -tx + x * tz
            coefficient_v = -ty + y * tz
            numerator[bin_index] += weight * (
                coefficient_u * (dx / scale - flat_u) + coefficient_v * (dy / scale - flat_v)
            )
            denominator[bin_index] += weight * (
                coefficient_u * coefficient_u + coefficient_v * coefficient_v
            )

    estimate = np.where(denominator > 1e-12, numerator / np.maximum(denominator, 1e-12), previous)
    spread = float(estimate.std())
    if not np.isfinite(spread) or spread < 1e-9:
        return previous
    # Fix the gauge: centred and unit-spread, with the sign that keeps the
    # bin ordering the focus sweep reported.
    estimate = (estimate - estimate.mean()) / spread
    if np.corrcoef(estimate, previous)[0, 1] < 0:
        estimate = -estimate
    return estimate


def _joint_motion_depth_fields(
    images: list[np.ndarray],
    coarse: list[np.ndarray],
    coarse_valid: list[np.ndarray],
    global_warps: list[np.ndarray | None],
    ref_index: int,
    iterations: int,
    stretch_tolerance: float,
    bins: int = 6,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict]:
    """Alternate between camera motion, scene depth, and the depth calibration.

    Three quantities are entangled and none is knowable alone. Separating
    rotation from parallax needs depth. Depth-from-focus needs frames already
    registered well enough that the sharpness contest is decided by focus rather
    than by misalignment. And converting the focus index into the inverse depth
    that parallax actually scales with needs the motion. So all three are solved
    in alternation, each pass using the others' current best estimate.

    Every iteration composes its correction into the sampling field rather than
    into the pixels, so the frame is still interpolated exactly once no matter
    how many iterations run.
    """
    from .fusion import depth_from_focus, guided_filter

    h, w = coarse[0].shape[:2]
    diagonal = float(np.hypot(h, w))
    max_shift = _REFINE_MAX_FRACTION * diagonal

    fields: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i, warp in enumerate(global_warps):
        if i == ref_index or warp is None:
            continue
        fields[i] = _matrix_field(_homogeneous(warp), (h, w))

    report: dict = {"bins": bins, "model": "joint", "frames": {}, "iterations": [], "reason": None}
    if not fields:
        report["reason"] = "no globally registered frames"
        return {}, report

    current = list(coarse)
    valid = list(coarse_valid)
    corrected_frames: set[int] = set()
    depth_step = np.zeros((h, w), dtype=np.uint8)

    for _iteration in range(iterations):
        depth = depth_from_focus(current)
        common = np.logical_and.reduce(valid)
        if common.mean() < 0.2:
            report["reason"] = "insufficient common footprint"
            break

        edges = _valley_edges(depth[common], bins)
        if edges is None:
            edges = np.unique(np.quantile(depth[common], np.linspace(0.0, 1.0, bins + 1)))
        if edges.size < 3:
            report["reason"] = "degenerate depth map"
            break
        bin_map = np.clip(np.digitize(depth, edges[1:-1]), 0, edges.size - 2)
        bin_count = int(edges.size - 1)
        probe = np.ones((5, 5), np.uint8)
        depth_step = (
            (cv2.dilate(depth, probe) - cv2.erode(depth, probe))
            > _OCCLUSION_MIN_DEPTH_STEP
        ).astype(np.uint8)

        centres = np.array(
            [float(np.median(depth[bin_map == b])) if (bin_map == b).any() else 0.0
             for b in range(bin_count)]
        )
        spread = float(centres.std())
        rho = (centres - centres.mean()) / spread if spread > 1e-9 else np.zeros(bin_count)

        ref_gray = to_gray_float(current[ref_index]).astype(np.float32) / 255.0
        observations = {}
        for i in fields:
            observations[i] = _tile_observations(
                ref_gray,
                to_gray_float(current[i]).astype(np.float32) / 255.0,
                common & valid[i],
                bin_map.astype(np.float64),
                max_shift,
            )

        motions: dict[int, np.ndarray] = {}
        for _inner in range(_JOINT_INNER):
            for i, frame_observations in observations.items():
                as_depth = [
                    (px, py, float(rho[int(np.clip(round(b), 0, bin_count - 1))]),
                     dx, dy, weight)
                    for px, py, b, dx, dy, weight in frame_observations
                ]
                params = _fit_motion_model(as_depth, (h, w))
                if params is not None:
                    motions[i] = params
            if not motions:
                break
            rho = _fit_inverse_depth(observations, motions, (h, w), bin_count, rho)

        if not motions:
            break

        # A calibrated inverse-depth map: piecewise by bin, then edge-aware
        # smoothed so the field steps only where the image says depth steps.
        rho_map = guided_filter(
            ref_gray,
            rho[bin_map].astype(np.float32),
            max(2, int(round(_REFINE_MEMBERSHIP_FRACTION * diagonal))),
            _REFINE_MEMBERSHIP_EPS,
        )

        fitted = 0
        magnitudes = []
        for i, params in motions.items():
            dx, dy = _motion_displacement(params, rho_map, (h, w))
            if not (np.isfinite(dx).all() and np.isfinite(dy).all()):
                continue
            if float(np.hypot(dx, dy).max()) > max_shift:
                # A fit that wants to move some pixel further than parallax ever
                # could is extrapolating past its observations, not measuring.
                continue

            fields[i] = _compose_field(*fields[i], dx, dy)
            current[i] = cv2.remap(
                images[i], *fields[i], cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            valid[i] = cv2.remap(
                np.full((h, w), 255, np.uint8), *fields[i], cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            ) == 255
            fitted += 1
            corrected_frames.add(i)
            magnitudes.append(float(np.hypot(dx, dy).mean()))
            report["frames"][i] = {
                "accepted": fitted,
                "params": [float(v) for v in params],
                "observations": len(observations[i]),
                "stretch": 0.0,
            }

        report["iterations"].append(
            {
                "fitted": fitted,
                "mean_correction_px": float(np.mean(magnitudes)) if magnitudes else 0.0,
                "rho": [float(v) for v in rho],
            }
        )
        if fitted == 0:
            break

    output = {}
    occlusion: dict[int, np.ndarray] = {}
    for i in sorted(corrected_frames):
        map_x, map_y = fields[i]
        base_x, base_y = _matrix_field(_homogeneous(global_warps[i]), (h, w))
        occlusion[i] = _occlusion_mask(map_x - base_x, map_y - base_y, depth_step)
        if stretch_tolerance > 0.0:
            map_x, map_y = _limit_field_stretch(
                map_x, map_y, base_x, base_y, stretch_tolerance
            )
        report["frames"][i]["occluded_fraction"] = float(occlusion[i].mean())
        report["frames"][i]["stretch"] = _field_stretch(map_x, map_y)
        output[i] = (map_x, map_y)
    report["occlusion"] = occlusion
    return output, report


def _depth_binned_fields(
    images: list[np.ndarray],
    coarse: list[np.ndarray],
    coarse_valid: list[np.ndarray],
    global_warps: list[np.ndarray | None],
    ref_index: int,
    bins: int,
    depth_model: str = "bins",
    stretch_tolerance: float = _REFINE_STRETCH_TOLERANCE,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict]:
    """Plan a depth-binned sampling field for each frame that earns one.

    Returns the fields keyed by frame index (frames with no accepted bin
    refinement are absent, and must keep the plain global warp so their output
    stays byte-identical to the global-only aligner) plus a diagnostic report.
    """
    from .fusion import depth_from_focus, guided_filter

    h, w = coarse[0].shape[:2]
    diagonal = float(np.hypot(h, w))
    max_shift = _REFINE_MAX_FRACTION * diagonal
    common_valid = np.logical_and.reduce(coarse_valid)

    report: dict = {"bins": 0, "frames": {}, "reason": None}
    if common_valid.mean() < 0.2:
        report["reason"] = "insufficient common footprint"
        return {}, report

    depth = depth_from_focus(coarse)
    masks = _depth_bin_masks(depth, common_valid, bins)
    if len(masks) < 2:
        report["reason"] = "degenerate depth map"
        return {}, report
    report["bins"] = len(masks)

    ref_gray = to_gray_float(coarse[ref_index]).astype(np.float32) / 255.0
    gradient = cv2.magnitude(
        cv2.Sobel(ref_gray * 255.0, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(ref_gray * 255.0, cv2.CV_32F, 0, 1, ksize=3),
    )

    # Edge-aware memberships: the guided filter softens each bin's influence
    # without smearing it across a real object boundary, which is exactly where
    # depth (and therefore the correction) genuinely jumps.
    radius = max(2, int(round(_REFINE_MEMBERSHIP_FRACTION * diagonal)))
    memberships = [
        np.clip(
            guided_filter(ref_gray, mask.astype(np.float32), radius, _REFINE_MEMBERSHIP_EPS),
            0.0,
            1.0,
        )
        for mask in masks
    ]
    total_membership = np.sum(memberships, axis=0)
    # Whatever membership is missing (image border, unbinned pixels) falls back
    # to the global warp, so the field is defined everywhere.
    fallback = np.clip(1.0 - total_membership, 0.0, 1.0)
    denominator = total_membership + fallback
    denominator = np.where(denominator < 1e-6, 1.0, denominator)
    memberships = [(m / denominator).astype(np.float32) for m in memberships]
    fallback = (fallback / denominator).astype(np.float32)

    # A genuine discontinuity: depth crossing most of a bin's width within a
    # few pixels. A gradual ramp never trips this, however many bins span it.
    probe = np.ones((5, 5), np.uint8)
    depth_step = (
        (cv2.dilate(depth, probe) - cv2.erode(depth, probe)) > _OCCLUSION_MIN_DEPTH_STEP
    ).astype(np.uint8)

    centers = [float(np.median(depth[mask])) if mask.any() else 0.0 for mask in masks]
    textured = [_bin_is_textured(gradient, mask) for mask in masks]
    populated = [
        bool(mask.sum() >= _REFINE_MIN_BIN_FRACTION * common_valid.sum()) for mask in masks
    ]

    fields: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    occlusion: dict[int, np.ndarray] = {}
    for i, image in enumerate(images):
        if i == ref_index or global_warps[i] is None:
            continue
        base = _homogeneous(global_warps[i])
        moving_gray = to_gray_float(coarse[i]).astype(np.float32) / 255.0

        matrices = [base]
        weights = [fallback]
        shifts: list[tuple[float, float] | None] = []
        accepted = 0
        for mask, is_textured, is_populated, membership in zip(
            masks, textured, populated, memberships
        ):
            residual = None
            if is_textured and is_populated:
                residual = _residual_translation(
                    ref_gray, moving_gray, mask & coarse_valid[i], max_shift
                )
            if residual is None:
                shifts.append(None)
                matrices.append(base)
            else:
                shifts.append((float(residual[0, 2]), float(residual[1, 2])))
                accepted += 1
                # coarse(x) = original(global * x) and the residual maps
                # reference coordinates onto coarse ones, so the composed warp
                # samples the ORIGINAL frame once.
                matrices.append(base @ _homogeneous(residual))
            weights.append(membership)

        report["frames"][i] = {"accepted": accepted, "shifts": shifts, "stretch": 0.0}
        if accepted == 0:
            continue
        if depth_model == "linear":
            field = _linear_depth_field(base, depth, centers, shifts)
            if field is None:
                continue
            map_x, map_y = field
        else:
            map_x, map_y = _blended_coordinate_maps(matrices, weights, (h, w))
        base_x, base_y = _matrix_field(base, (h, w))
        # Built from the per-bin shifts as MEASURED, on hard bin support: this
        # is the relative motion the scene actually underwent, before any
        # membership smoothing or stretch relaxation softened the steps out of
        # the applied field.
        hard_x = np.zeros((h, w), dtype=np.float32)
        hard_y = np.zeros((h, w), dtype=np.float32)
        for mask, shift in zip(masks, shifts):
            if shift is not None:
                hard_x[mask] = shift[0]
                hard_y[mask] = shift[1]
        occlusion[i] = _occlusion_mask(hard_x, hard_y, depth_step)
        if depth_model != "linear" and stretch_tolerance > 0.0:
            map_x, map_y = _limit_field_stretch(
                map_x, map_y, base_x, base_y, stretch_tolerance
            )
        report["frames"][i]["occluded_fraction"] = float(occlusion[i].mean())
        report["frames"][i]["stretch"] = _field_stretch(map_x, map_y)
        fields[i] = (map_x, map_y)

    report["occlusion"] = occlusion
    return fields, report


def align_stack(
    images: list[np.ndarray],
    ref_index: int | None = None,
    motion: str = "affine",
    max_iterations: int = 500,
    eps: float = 1e-6,
    crop_valid: bool = True,
    depth_bins: int = 4,
    depth_model: str = "bins",
    stretch_tolerance: float = _REFINE_STRETCH_TOLERANCE,
    return_report: bool = False,
) -> list[np.ndarray] | tuple[list[np.ndarray], dict]:
    """Align every frame to a reference frame.

    Args:
        images: list of BGR uint8 frames, all the same size.
        ref_index: which frame to treat as the fixed reference. Defaults to the
            middle frame, which tends to be geometrically closest to all others.
        motion: one of `_MOTION_MODES`.
        max_iterations, eps: ECC convergence criteria.
        crop_valid: crop every aligned frame to the largest rectangular region
            genuinely observed in every source frame. This prevents warp border
            fill from entering focus selection or fusion.
        depth_bins: number of depth-from-focus bins used for the depth-dependent
            parallax refinement. 0 or 1 disables it and restores pure global
            registration; it is also skipped for stacks of fewer than three
            frames, which carry no usable depth proxy.
        depth_model: "bins" (default) fits one translation per depth bin;
            "joint" is the experimental alternating motion/depth/calibration
            estimator, which is not yet non-regressing on still stacks.
        stretch_tolerance: largest local field gradient left unrelaxed. 0
            disables the limiter and permits visible smearing at depth steps.
        return_report: also return a diagnostic dict describing the refinement:
            the bin count, per-frame accepted corrections and their measured
            shifts, the worst field stretch, and the final crop rectangle.

    Returns:
        A new list of aligned BGR uint8 frames. With the default common-footprint
        crop, even the reference is cropped to match the other frames.
        Frames for which ECC fails to converge are returned unaligned, with a
        warning, rather than aborting the whole run.
    """
    if motion not in _MOTION_MODES:
        raise ValueError(f"Unknown motion model {motion!r}; choose from {list(_MOTION_MODES)}")
    warp_mode = _MOTION_MODES[motion]

    n = len(images)
    if ref_index is None:
        ref_index = n // 2

    # ECC wants single-channel float in [0, 1].
    def norm_gray(img: np.ndarray) -> np.ndarray:
        return to_gray_float(img) / 255.0

    ref = norm_gray(images[ref_index])
    h, w = ref.shape
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, eps)

    aligned: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    global_warps: list[np.ndarray | None] = []
    for i, img in enumerate(images):
        if i == ref_index:
            aligned.append(img)
            valid_masks.append(np.ones((h, w), dtype=bool))
            global_warps.append(None)
            continue

        moving = norm_gray(img)
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        try:
            # findTransformECC estimates the warp mapping `moving` onto `ref`.
            _, warp_matrix = cv2.findTransformECC(
                ref, moving, warp_matrix, warp_mode, criteria, None, 5
            )
            # WARP_INVERSE_MAP applies that warp to resample `img` into ref's frame.
            common = dict(
                dsize=(w, h),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            if warp_mode == cv2.MOTION_HOMOGRAPHY:
                warped = cv2.warpPerspective(img, warp_matrix, **common)
                warped_valid = cv2.warpPerspective(
                    np.full((h, w), 255, np.uint8),
                    warp_matrix,
                    dsize=(w, h),
                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
            else:
                warped = cv2.warpAffine(img, warp_matrix, **common)
                warped_valid = cv2.warpAffine(
                    np.full((h, w), 255, np.uint8),
                    warp_matrix,
                    dsize=(w, h),
                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
            aligned.append(warped)
            # Requiring 255 means bilinear sampling never touched synthetic
            # border data, rather than merely requiring the sample centre to
            # fall within the source canvas.
            valid_masks.append(warped_valid == 255)
            global_warps.append(warp_matrix)
        except cv2.error as e:
            warnings.warn(f"ECC alignment failed for frame {i}; using it unaligned. ({e})")
            aligned.append(img)
            valid_masks.append(np.ones((h, w), dtype=bool))
            global_warps.append(None)

    report: dict = {"bins": 0, "frames": {}, "reason": "disabled"}
    if depth_bins >= 2 and n >= 3:
        if depth_model == "joint":
            fields, report = _joint_motion_depth_fields(
                images, aligned, valid_masks, global_warps, ref_index,
                _JOINT_ITERATIONS, stretch_tolerance,
            )
        else:
            fields, report = _depth_binned_fields(
                images, aligned, valid_masks, global_warps, ref_index, depth_bins,
                depth_model, stretch_tolerance,
            )
        for i, (map_x, map_y) in fields.items():
            aligned[i] = cv2.remap(
                images[i],
                map_x,
                map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            refined_valid = cv2.remap(
                np.full((h, w), 255, np.uint8),
                map_x,
                map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            valid_masks[i] = refined_valid == 255

    # Per-pixel usability, kept SEPARATE from the rectangular crop above. The
    # crop answers "which pixels did every frame observe at all"; this answers
    # "which pixels can this particular frame legitimately supply", which
    # parallax makes a different question near every depth step.
    occlusion = report.get("occlusion", {})
    usable = [
        ~occlusion[i] if i in occlusion else np.ones((h, w), dtype=bool)
        for i in range(n)
    ]

    if crop_valid:
        common_valid = np.logical_and.reduce(valid_masks)
        x0, y0, x1, y1 = _largest_valid_rectangle(common_valid)
        aligned = [img[y0:y1, x0:x1].copy() for img in aligned]
        usable = [mask[y0:y1, x0:x1].copy() for mask in usable]
        report["crop"] = (x0, y0, x1, y1)

    # The reference frame is unwarped, so it always has a real observation
    # everywhere. That guarantees every output pixel keeps at least one usable
    # source and fusion can never be left with nothing to choose from.
    report["usable"] = usable
    report.pop("occlusion", None)

    if return_report:
        return aligned, report
    return aligned
