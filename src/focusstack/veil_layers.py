"""Conservative joint-layer recovery for licensed giant defocus veils.

This is the package form of F55's narrow, held-out specialist.  It does not
repair an already-fused image by amplifying its texture.  Instead it fits the
two captured focal frames to a two-layer image-formation model:

    O_i = H_near,i(alpha * N) + (1 - H_near,i(alpha)) * H_far,i(S)

Only corrections to observed near/far anchors are solved.  A candidate matte
must first reduce observation-domain error at 512 px and satisfy the frozen
semantic license.  At native resolution, a component is retained only when it
has the same sign under three regularizers and two plausible PSF
implementations.  Everything outside that licensed regime is identity.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import cv2
import numpy as np

from .focus import content_aware_energies
from .fusion import guided_filter
from .io import to_gray_float
from .reconstruct import _disk_blur


MODEL_SIDE = 512
# Native validation used 1536 px scenes.  A small container tolerance admits
# that exact regime while refusing unmeasured 2K/4K memory and scale behavior.
MAX_NATIVE_SIDE = 1600
RADIUS_FRACTION = 0.035
CHANNEL_OFFSETS = np.asarray((-0.04, 0.0, 0.04), np.float32)
FOCUS_POSITIONS = (0.15, 0.85)
NEAR_DEPTH = 0.15
FAR_DEPTH = 0.85
SOLVER_CONFIGS = ((2.0, 0.02), (8.0, 0.05), (32.0, 0.10))
# An owner-frame semantic fragment is admitted as missing foreground support
# only when adding it to the layer model reduces 512px observation error by
# this absolute amount.  P11 calibrated the margin above every harmful
# component in development; the unchanged rule is validated on a fresh factory
# extension before package promotion.
SUPPORT_FORWARD_MARGIN = 0.01
SUPPORT_MAX_AREA_FRACTION = 0.02
SUPPORT_MAX_OVERLAP_FRACTION = 0.20

BlurFunction = Callable[[np.ndarray, float], np.ndarray]


def _box_disk_blur(image: np.ndarray, radius: float) -> np.ndarray:
    """Second plausible PSF family used by the factory and F55 audit."""
    image_f = image.astype(np.float32, copy=False)
    if radius < 0.6:
        return image_f.copy()
    if radius <= 12:
        r = int(np.ceil(radius))
        yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
        kernel = ((xx * xx + yy * yy) <= radius * radius).astype(np.float32)
        kernel /= kernel.sum()
        return cv2.filter2D(image_f, -1, kernel)
    factor = max(1, int(radius / 6))
    height, width = image.shape[:2]
    small = cv2.resize(
        image_f,
        (max(1, width // factor), max(1, height // factor)),
        interpolation=cv2.INTER_AREA,
    )
    reduced_radius = max(1, int(round(radius / factor)))
    small = cv2.blur(
        small,
        (2 * reduced_radius + 1, 2 * reduced_radius + 1),
    )
    return cv2.resize(
        small,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)


def _blur_channels(
    image: np.ndarray,
    radii: np.ndarray,
    blur_fn: BlurFunction,
) -> np.ndarray:
    return np.stack(
        [
            blur_fn(image[..., channel], float(radii[channel]))
            for channel in range(3)
        ],
        axis=2,
    )


def _radii_for(max_radius: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
    near = [
        np.abs(NEAR_DEPTH - (focus + CHANNEL_OFFSETS)) * max_radius
        for focus in FOCUS_POSITIONS
    ]
    far = [
        np.abs(FAR_DEPTH - (focus + CHANNEL_OFFSETS)) * max_radius
        for focus in FOCUS_POSITIONS
    ]
    return near, far


def _prepare_model(
    alpha: np.ndarray,
    max_radius: float,
    blur_fn: BlurFunction,
) -> dict:
    near_radii, far_radii = _radii_for(max_radius)
    alpha3 = np.repeat(alpha[..., None], 3, axis=2)
    transmission = [
        1.0 - _blur_channels(alpha3, radii, blur_fn)
        for radii in near_radii
    ]
    return {
        "alpha": alpha.astype(np.float32),
        "near_radii": near_radii,
        "far_radii": far_radii,
        "transmission": transmission,
        "blur_fn": blur_fn,
    }


def _forward_layers(
    near: np.ndarray,
    far: np.ndarray,
    model: dict,
) -> list[np.ndarray]:
    premultiplied = model["alpha"][..., None] * near
    return [
        _blur_channels(premultiplied, near_radii, model["blur_fn"])
        + transmission
        * _blur_channels(far, far_radii, model["blur_fn"])
        for near_radii, far_radii, transmission in zip(
            model["near_radii"],
            model["far_radii"],
            model["transmission"],
        )
    ]


def _adjoint(
    residuals: list[np.ndarray],
    model: dict,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = model["alpha"][..., None]
    near = np.zeros_like(residuals[0], np.float32)
    far = np.zeros_like(residuals[0], np.float32)
    for residual, near_radii, far_radii, transmission in zip(
        residuals,
        model["near_radii"],
        model["far_radii"],
        model["transmission"],
    ):
        # Both admitted PSFs are symmetric. Reflect-border filtering is only
        # approximately self-adjoint at the outer image border; the candidate
        # support is required to be an interior semantic object.
        near += alpha * _blur_channels(residual, near_radii, model["blur_fn"])
        far += _blur_channels(
            transmission * residual,
            far_radii,
            model["blur_fn"],
        )
    return near, far


def _correction_regularizer(
    image: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """B^T B for B=(I-G_sigma), preserving correction DC."""
    high = image - cv2.GaussianBlur(
        image,
        (0, 0),
        sigma,
    )
    return high - cv2.GaussianBlur(
        high,
        (0, 0),
        sigma,
    )


def solve_layers(
    images: list[np.ndarray],
    alpha: np.ndarray,
    max_radius: float,
    *,
    smooth_lambda: float = 8.0,
    anchor_lambda: float = 0.05,
    regularizer_sigma: float = 1.0,
    blur_fn: BlurFunction = _box_disk_blur,
    iterations: int = 18,
) -> tuple[np.ndarray, dict]:
    """Fit corrections to the observed layers with conjugate gradients.

    ``images`` is ordered as ``[foreground-owner frame, other focal frame]``.
    The returned scene is a sharp-alpha composite, plus observation-domain
    diagnostics used for candidate admission.
    """
    if len(images) != 2:
        raise ValueError("joint veil inversion requires exactly two frames")
    if alpha.shape != images[0].shape[:2]:
        raise ValueError("alpha and frame dimensions do not match")

    observed = [image.astype(np.float32) for image in images]
    model = _prepare_model(alpha, max_radius, blur_fn)
    near0 = observed[0].copy()
    far0 = observed[1].copy()
    predicted0 = _forward_layers(near0, far0, model)
    residual0 = [
        observation - predicted
        for observation, predicted in zip(observed, predicted0)
    ]
    rhs_near, rhs_far = _adjoint(residual0, model)

    def normal(
        correction_near: np.ndarray,
        correction_far: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        applied_near, applied_far = _adjoint(
            _forward_layers(correction_near, correction_far, model),
            model,
        )
        applied_near += (
            smooth_lambda
            * _correction_regularizer(correction_near, regularizer_sigma)
            + anchor_lambda * correction_near
        )
        applied_far += (
            smooth_lambda
            * _correction_regularizer(correction_far, regularizer_sigma)
            + anchor_lambda * correction_far
        )
        return applied_near, applied_far

    def dot(
        near_a: np.ndarray,
        far_a: np.ndarray,
        near_b: np.ndarray,
        far_b: np.ndarray,
    ) -> float:
        return float(
            np.sum(near_a * near_b, dtype=np.float64)
            + np.sum(far_a * far_b, dtype=np.float64)
        )

    correction_near = np.zeros_like(observed[0])
    correction_far = np.zeros_like(observed[1])
    residual_near, residual_far = rhs_near.copy(), rhs_far.copy()
    direction_near, direction_far = residual_near.copy(), residual_far.copy()
    residual_norm = dot(
        residual_near,
        residual_far,
        residual_near,
        residual_far,
    )
    history = [
        float(
            np.sqrt(
                residual_norm / (residual_near.size + residual_far.size)
            )
        )
    ]

    for _ in range(iterations):
        normal_near, normal_far = normal(direction_near, direction_far)
        denominator = dot(
            direction_near,
            direction_far,
            normal_near,
            normal_far,
        )
        if not np.isfinite(denominator) or denominator <= 1e-12:
            break
        step = residual_norm / denominator
        correction_near += step * direction_near
        correction_far += step * direction_far
        residual_near -= step * normal_near
        residual_far -= step * normal_far
        new_norm = dot(
            residual_near,
            residual_far,
            residual_near,
            residual_far,
        )
        history.append(
            float(
                np.sqrt(new_norm / (residual_near.size + residual_far.size))
            )
        )
        if new_norm <= 1e-8 * max(residual_norm, 1e-12):
            break
        beta = new_norm / max(residual_norm, 1e-12)
        direction_near = residual_near + beta * direction_near
        direction_far = residual_far + beta * direction_far
        residual_norm = new_norm

    near = np.clip(near0 + correction_near, 0, 255)
    far = np.clip(far0 + correction_far, 0, 255)
    predicted = _forward_layers(near, far, model)
    before = float(
        np.mean(
            [
                np.abs(prediction - observation).mean()
                for prediction, observation in zip(predicted0, observed)
            ]
        )
    )
    after = float(
        np.mean(
            [
                np.abs(prediction - observation).mean()
                for prediction, observation in zip(predicted, observed)
            ]
        )
    )
    scene = (
        model["alpha"][..., None] * near
        + (1.0 - model["alpha"][..., None]) * far
    )
    return scene, {
        "forward_before": before,
        "forward_after": after,
        "cg_history": history,
        "near_correction_rms": float(
            np.sqrt(np.mean(correction_near * correction_near))
        ),
        "far_correction_rms": float(
            np.sqrt(np.mean(correction_far * correction_far))
        ),
    }


def candidate_is_licensed(candidate: dict) -> bool:
    """Frozen F55 high-precision observation/semantic admission rule."""
    features = np.asarray(candidate.get("feats", ()), np.float32)
    before = float(candidate.get("forward_before", np.nan))
    after = float(candidate.get("forward_after", np.nan))
    if features.size < 4 or not np.all(np.isfinite(features)):
        return False
    if not np.isfinite(before) or not np.isfinite(after):
        return False
    score, purity, _, area_fit = features[:4]
    forward_ratio = after / max(before, 1e-6)
    return bool(
        score > 0.5
        and purity > 0.85
        and area_fit > 0.9
        and forward_ratio < 0.85
    )


def stable_correction(
    base: np.ndarray,
    solved_images: Iterable[np.ndarray],
    *,
    correction_sigma: float,
) -> tuple[np.ndarray, dict]:
    """Project onto components whose sign survives every admitted model."""
    minimum = None
    maximum = None
    minimum_abs = None
    total = None
    total_square = None
    count = 0
    base_f = base.astype(np.float32)

    for solved in solved_images:
        correction = solved.astype(np.float32) - base_f
        if correction_sigma > 0:
            correction = cv2.GaussianBlur(
                correction,
                (0, 0),
                correction_sigma,
                borderType=cv2.BORDER_REFLECT,
            )
        if minimum is None:
            minimum = correction.copy()
            maximum = correction.copy()
            minimum_abs = np.abs(correction)
            total = correction.copy()
            total_square = correction * correction
        else:
            np.minimum(minimum, correction, out=minimum)
            np.maximum(maximum, correction, out=maximum)
            np.minimum(minimum_abs, np.abs(correction), out=minimum_abs)
            total += correction
            total_square += correction * correction
        count += 1

    if count == 0:
        raise ValueError("at least one solved image is required")
    mean = total / count
    same_sign = (minimum > 0) | (maximum < 0)
    retained = np.sign(mean) * minimum_abs * same_sign
    variance = np.maximum(total_square / count - mean * mean, 0)
    return retained, {
        "model_count": count,
        "stable_fraction": float(same_sign.mean()),
        "ensemble_spread_rms": float(np.sqrt(np.mean(variance))),
        "retained_rms": float(np.sqrt(np.mean(retained * retained))),
    }


def _resize_for_model(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, MODEL_SIDE / max(height, width))
    size = (
        max(2, round(width * scale)),
        max(2, round(height * scale)),
    )
    if size == (width, height):
        return image.copy()
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _candidate_evidence(
    images: list[np.ndarray],
    candidate: dict,
) -> dict | None:
    owner = int(candidate.get("owner", -1))
    if owner not in (0, 1):
        return None
    alpha = np.asarray(candidate.get("alpha"), np.float32)
    if alpha.shape != images[0].shape[:2] or not np.all(np.isfinite(alpha)):
        return None

    resized_images = [_resize_for_model(image) for image in images]
    height, width = resized_images[0].shape[:2]
    alpha_small = cv2.resize(
        alpha,
        (width, height),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    ordered = [resized_images[owner], resized_images[1 - owner]]
    _, evidence = solve_layers(
        ordered,
        alpha_small,
        RADIUS_FRACTION * max(height, width),
        smooth_lambda=8.0,
        anchor_lambda=0.05,
        regularizer_sigma=1.0,
        blur_fn=_box_disk_blur,
    )
    return {**candidate, **evidence}


def select_licensed_candidate(
    images: list[np.ndarray],
    candidates: list[dict],
) -> tuple[dict | None, dict]:
    """Physically rerank the semantic bank, then apply the frozen license."""
    measured = []
    for rank, candidate in enumerate(candidates[:4]):
        evidence = _candidate_evidence(images, candidate)
        if evidence is not None:
            evidence["rank"] = rank
            measured.append(evidence)
    if not measured:
        return None, {"reason": "no_valid_candidate"}

    selected = min(measured, key=lambda row: row["forward_after"])
    ratio = selected["forward_after"] / max(
        selected["forward_before"],
        1e-6,
    )
    report = {
        "candidate_count": len(measured),
        "candidate_rank": int(selected["rank"]),
        "forward_before": float(selected["forward_before"]),
        "forward_after": float(selected["forward_after"]),
        "forward_ratio": float(ratio),
    }
    if not candidate_is_licensed(selected):
        report["reason"] = "candidate_unlicensed"
        return None, report
    report["reason"] = "licensed"
    return selected, report


def complete_owner_support(
    images: list[np.ndarray],
    selected: dict,
    owner_masks: np.ndarray | None,
) -> tuple[np.ndarray, dict]:
    """Find physically licensed owner-only semantic fragments.

    The ordinary semantic bridge sees the already-fused image.  If fusion has
    mixed a small foreground appendage with sharp revealed background, that
    appendage may no longer be segmentable there.  The foreground-owner frame
    still observes it sharply.  We therefore inspect that frame's semantic
    masks for small nearby *satellites* that are not already part of the
    selected matte.

    A satellite is not trusted on appearance alone.  It is added to the
    two-layer alpha at model resolution and must independently reduce the
    captured-frame forward residual by ``SUPPORT_FORWARD_MARGIN``.  Accepted
    masks are used only to hard-select the observed owner and to veto
    background recovery; they never generate texture.
    """
    shape = images[0].shape[:2]
    empty = np.zeros(shape, bool)
    report = {
        "owner_support_candidate_count": 0,
        "owner_support_accepted_count": 0,
        "owner_support_pixels": 0,
        "owner_support_reason": "owner_masks_unavailable",
    }
    if owner_masks is None:
        return empty, report
    masks = np.asarray(owner_masks)
    if masks.ndim != 3 or masks.shape[1:] != shape:
        report["owner_support_reason"] = "owner_masks_invalid"
        return empty, report

    alpha = np.clip(np.asarray(selected["alpha"], np.float32), 0.0, 1.0)
    seed = alpha >= 0.5
    if not np.any(seed):
        report["owner_support_reason"] = "owner_seed_empty"
        return empty, report

    max_radius = RADIUS_FRACTION * max(shape)
    near_radius = max(3, int(round(1.5 * max_radius)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * near_radius + 1, 2 * near_radius + 1),
    )
    nearby = cv2.dilate(seed.astype(np.uint8), kernel) > 0
    min_area = max(
        8,
        int(round(20.0 * (max(shape) / 1536.0) ** 2)),
    )
    max_area = SUPPORT_MAX_AREA_FRACTION * seed.size
    owner = int(selected["owner"])
    guide = to_gray_float(images[owner]) / 255.0
    base_after = float(selected["forward_after"])
    trials = []

    for mask_index, raw_mask in enumerate(masks):
        mask = np.asarray(raw_mask) > 0
        area = int(mask.sum())
        if area < min_area or area > max_area:
            continue
        overlap = int((mask & seed).sum())
        if overlap / max(area, 1) >= SUPPORT_MAX_OVERLAP_FRACTION:
            continue
        # The fragment must belong to the selected object's immediate optical
        # neighborhood, not merely share its colors elsewhere in the frame.
        if float((mask & nearby).sum()) / area < 0.90:
            continue

        report["owner_support_candidate_count"] += 1
        mask_alpha = np.clip(
            guided_filter(
                guide.astype(np.float32),
                mask.astype(np.float32),
                2,
                1e-4,
            ),
            0.0,
            1.0,
        )
        augmented = np.maximum(alpha, mask_alpha)
        evidence = _candidate_evidence(
            images,
            {**selected, "alpha": augmented},
        )
        if evidence is None:
            continue
        improvement = base_after - float(evidence["forward_after"])
        if improvement > SUPPORT_FORWARD_MARGIN:
            trials.append(
                {
                    "index": int(mask_index),
                    "mask": mask,
                    "alpha": mask_alpha,
                    "forward_after": float(evidence["forward_after"]),
                    "improvement": float(improvement),
                }
            )

    if not trials:
        report["owner_support_reason"] = "no_forward_licensed_fragment"
        return empty, report

    combined_alpha = alpha.copy()
    combined_mask = empty.copy()
    for trial in trials:
        combined_alpha = np.maximum(combined_alpha, trial["alpha"])
        combined_mask |= trial["mask"]
    combined_evidence = _candidate_evidence(
        images,
        {**selected, "alpha": combined_alpha},
    )
    combined_improvement = (
        base_after - float(combined_evidence["forward_after"])
        if combined_evidence is not None
        else -np.inf
    )
    if combined_improvement <= SUPPORT_FORWARD_MARGIN:
        # Individually useful fragments can overlap or compete in the joint
        # model.  Retain only the strongest independently licensed fragment.
        best = max(trials, key=lambda row: row["improvement"])
        trials = [best]
        combined_mask = best["mask"].copy()
        combined_improvement = best["improvement"]
        combined_after = best["forward_after"]
    else:
        combined_after = float(combined_evidence["forward_after"])

    support = combined_mask & ~seed
    report.update(
        {
            "owner_support_accepted_count": len(trials),
            "owner_support_pixels": int(support.sum()),
            "owner_support_reason": "forward_licensed_owner_fragments",
            "owner_support_mask_indices": [
                int(trial["index"])
                for trial in trials
            ],
            "owner_support_forward_before": base_after,
            "owner_support_forward_after": combined_after,
            "owner_support_forward_improvement": float(combined_improvement),
        }
    )
    return support, report


def _fringe_mask(
    alpha: np.ndarray,
    max_radius: float,
    mask_sigma: float,
) -> np.ndarray:
    blurred_alpha = _box_disk_blur(alpha, 0.7 * max_radius)
    support = (
        (blurred_alpha > 0.05)
        & (blurred_alpha < 0.95)
        & (alpha < 0.5)
    )
    return cv2.GaussianBlur(
        support.astype(np.float32),
        (0, 0),
        mask_sigma,
        borderType=cv2.BORDER_REFLECT,
    )


def _ownership_gate(
    images: list[np.ndarray],
    owner: int,
    alpha: np.ndarray,
    spatial_scale: float,
) -> tuple[np.ndarray, dict]:
    """Suppress recovery where the stack supports the foreground owner.

    Observation-domain fit alone cannot distinguish a missing piece of a
    semantic matte from background: blur's null space can absorb that mistake
    into the solved far layer.  Focus ownership is independent evidence.  A
    confident owner win outside the proposed alpha therefore attenuates the
    application mask; uncertain pixels are left to the inverse consensus.
    """
    energies = np.stack(
        content_aware_energies(
            [to_gray_float(image) for image in images]
        ),
        axis=0,
    )
    winner = np.argmax(energies, axis=0)
    ordered = np.sort(energies, axis=0)
    dominance = np.clip(
        (ordered[-1] - ordered[-2]) / (ordered[-1] + 1e-6),
        0.0,
        1.0,
    )
    informative = ordered[-1] > np.median(ordered[-1])
    owner_confidence = (
        (winner == owner).astype(np.float32)
        * informative.astype(np.float32)
        * np.clip((dominance - 0.3) / 0.4, 0.0, 1.0)
    )
    smooth_veto = cv2.GaussianBlur(
        owner_confidence,
        (0, 0),
        2.0 * spatial_scale,
        borderType=cv2.BORDER_REFLECT,
    )
    gate = (alpha < 0.5).astype(np.float32) * (1.0 - smooth_veto)
    return np.clip(gate, 0.0, 1.0), {
        "owner_veto_mean": float(smooth_veto.mean()),
        "owner_veto_confident_fraction": float(
            (owner_confidence > 0.5).mean()
        ),
    }


def recover_giant_veil(
    images: list[np.ndarray],
    base: np.ndarray,
    candidates: list[dict],
    *,
    owner_masks_by_frame: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, dict]:
    """Recover one licensed giant veil or return ``base`` byte-for-byte."""
    report = {
        "fired": False,
        "reason": "not_evaluated",
        "model": "joint_two_layer_giant",
        "radius_fraction": RADIUS_FRACTION,
    }
    if len(images) != 2:
        report["reason"] = "requires_two_frames"
        return base, report
    if not candidates:
        report["reason"] = "no_candidate"
        return base, report
    if any(image.shape != base.shape for image in images):
        report["reason"] = "shape_mismatch"
        return base, report
    if max(base.shape[:2]) > MAX_NATIVE_SIDE:
        report["reason"] = "native_size_unvalidated"
        return base, report

    selected, selection_report = select_licensed_candidate(images, candidates)
    report.update(selection_report)
    if selected is None:
        return base, report

    owner = int(selected["owner"])
    alpha = np.clip(selected["alpha"].astype(np.float32), 0.0, 1.0)
    owner_masks = (
        owner_masks_by_frame[owner]
        if owner_masks_by_frame is not None
        and len(owner_masks_by_frame) == len(images)
        else None
    )
    owner_support, owner_support_report = complete_owner_support(
        images,
        selected,
        owner_masks,
    )
    report.update(owner_support_report)
    ordered = [images[owner], images[1 - owner]]
    spatial_scale = max(base.shape[:2]) / MODEL_SIDE
    max_radius = RADIUS_FRACTION * max(base.shape[:2])
    def solve_bank():
        # Yield one scene at a time: stable_correction accumulates its moments,
        # so native memory does not grow by six full-resolution RGB arrays.
        for blur_fn in (_box_disk_blur, _disk_blur):
            for smooth_lambda, anchor_lambda in SOLVER_CONFIGS:
                scene, _ = solve_layers(
                    ordered,
                    alpha,
                    max_radius,
                    smooth_lambda=smooth_lambda,
                    anchor_lambda=anchor_lambda,
                    regularizer_sigma=spatial_scale,
                    blur_fn=blur_fn,
                )
                yield scene

    correction, uncertainty = stable_correction(
        base,
        solve_bank(),
        correction_sigma=0.5 * spatial_scale,
    )
    report.update(uncertainty)
    if not np.all(np.isfinite(correction)):
        report["reason"] = "nonfinite_consensus"
        return base, report

    ownership, ownership_evidence = _ownership_gate(
        images,
        owner,
        alpha,
        spatial_scale,
    )
    report.update(ownership_evidence)
    mask = (
        _fringe_mask(alpha, max_radius, 2.0 * spatial_scale)
        * ownership
    )
    mask[owner_support] = 0.0
    if not np.any(mask > 1e-4):
        if not np.any(owner_support):
            report["reason"] = "empty_fringe"
            return base, report
        output = base.copy()
        output[owner_support] = images[owner][owner_support]
        report.update(
            {
                "fired": True,
                "reason": "licensed_owner_support_only",
                "owner": owner,
                "changed_pixels": int(
                    np.any(output != base, axis=2).sum()
                ),
            }
        )
        return output, report
    repaired_base = base.copy()
    repaired_base[owner_support] = images[owner][owner_support]
    output = np.clip(
        repaired_base.astype(np.float32) + correction * mask[..., None],
        0,
        255,
    ).astype(np.uint8)
    report.update(
        {
            "fired": True,
            "reason": "licensed_consensus",
            "owner": owner,
            "changed_pixels": int(np.any(output != base, axis=2).sum()),
        }
    )
    return output, report
