#!/usr/bin/env python3
"""S15 — causally attribute the remaining finest-band complement tail.

The current seven F62 fires improve every physical partition but retain a small
positive GT-credited error on smooth true-background veil pixels. This harness
reconstructs the exact package components, asserts byte identity with
``recover_giant_veil``, then separates:

1. focused-owner front reconstruction;
2. float rear-layer correction;
3. uint8 quantization;
4. correction strength; and
5. the correction's finest Gaussian band.

GT is used only to grade these offline counterfactuals. Runtime evidence remains
the captured frames, semantic proposals, focus ownership, and forward model.

Run:
    .venv/bin/python research/veil_fineband.py attribute
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from objocc_v2_eval import _score  # noqa: E402
from objocc_v2_gen import scenes  # noqa: E402
from t2_candidates import candidates_with_features  # noqa: E402
from veilband import fringe_mask as true_fringe_mask  # noqa: E402
from veilship import false_texture_error  # noqa: E402
from focusstack.fusion import fuse_perband  # noqa: E402
from focusstack.veil import estimate_noise_sigma  # noqa: E402
from focusstack.veil_layers import (  # noqa: E402
    MODEL_SIDE,
    RADIUS_FRACTION,
    SOLVER_CONFIGS,
    _box_disk_blur,
    _disk_blur,
    _fringe_mask,
    _ordered_visibility_gate,
    _owner_front_reconstruction_support,
    _ownership_gate,
    complete_owner_support,
    recover_giant_veil,
    refine_owner_candidate,
    select_licensed_candidate,
    solve_layers,
    stable_correction,
)


HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "objocc_v2", "fineband_cache_f62")
OUTPUT = os.path.join(HERE, "objocc_v2_f62_fineband_attribution.json")
CASES = (
    ("extension", "extension_007"),
    ("extension", "extension_034"),
    ("s12", "s12_025"),
    ("s16", "s16_034"),
    ("s19", "s19_000"),
    ("s19", "s19_012"),
    ("s19", "s19_013"),
)


def _load_cases() -> list[dict]:
    by_split = {
        split: {scene["sid"]: scene for scene in scenes(split)}
        for split in {split for split, _ in CASES}
    }
    return [by_split[split][sid] for split, sid in CASES]


def _components(scene: dict) -> dict:
    """Rebuild exact F62 arrays once, caching only ignored research data."""
    os.makedirs(CACHE, exist_ok=True)
    cache_path = os.path.join(CACHE, f"{scene['sid']}.npz")
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        return {key: cached[key] for key in cached.files}

    images = scene["frames"]
    base = fuse_perband(images, harden=0.5)
    candidates = candidates_with_features(scene, topk=4)
    owner_masks_by_frame = [
        np.load(
            os.path.join(
                scene["dir"],
                f"frame_{frame_index}.png.masks.npy",
            )
        )
        for frame_index in range(2)
    ]
    selected, selection_report = select_licensed_candidate(images, candidates)
    if selected is None:
        raise RuntimeError(
            f"{scene['sid']} lost its licensed candidate: {selection_report}"
        )
    owner = int(selected["owner"])
    owner_masks = owner_masks_by_frame[owner]
    selected, refinement_report = refine_owner_candidate(
        images,
        selected,
        owner_masks,
    )
    alpha = np.clip(selected["alpha"].astype(np.float32), 0.0, 1.0)
    owner_support, support_report = complete_owner_support(
        images,
        selected,
        owner_masks,
    )
    spatial_scale = max(1.0, max(base.shape[:2]) / MODEL_SIDE)
    max_radius = RADIUS_FRACTION * max(base.shape[:2])
    front_reconstruction = _owner_front_reconstruction_support(
        alpha,
        owner_masks,
        {**support_report, **refinement_report},
        max_radius,
        spatial_scale,
    )
    owner_copy = owner_support | front_reconstruction
    ordered = [images[owner], images[1 - owner]]

    def solve_bank():
        for blur_fn in (_box_disk_blur, _disk_blur):
            for smooth_lambda, anchor_lambda in SOLVER_CONFIGS:
                solved, _ = solve_layers(
                    ordered,
                    alpha,
                    max_radius,
                    smooth_lambda=smooth_lambda,
                    anchor_lambda=anchor_lambda,
                    regularizer_sigma=spatial_scale,
                    blur_fn=blur_fn,
                )
                yield solved

    correction, _ = stable_correction(
        base,
        solve_bank(),
        correction_sigma=0.5 * spatial_scale,
    )
    ownership, _ = _ordered_visibility_gate(
        images,
        owner,
        alpha,
        spatial_scale,
    )
    mask = (
        _fringe_mask(alpha, max_radius, 2.0 * spatial_scale)
        * ownership
    )
    mask[owner_copy] = 0.0
    repaired_base = base.copy()
    repaired_base[owner_copy] = images[owner][owner_copy]
    rebuilt = np.clip(
        repaired_base.astype(np.float32) + correction * mask[..., None],
        0,
        255,
    ).astype(np.uint8)
    package, report = recover_giant_veil(
        images,
        base,
        candidates,
        owner_masks_by_frame=owner_masks_by_frame,
    )
    if not report["fired"] or not np.array_equal(rebuilt, package):
        raise RuntimeError(f"{scene['sid']} attribution path drifted")

    payload = {
        "base": base,
        "repaired_base": repaired_base,
        "correction": correction.astype(np.float32),
        "mask": mask.astype(np.float32),
        "alpha_est": alpha.astype(np.float32),
        "owner_copy": owner_copy.astype(np.uint8),
        "owner": np.asarray(owner, np.int16),
    }
    np.savez_compressed(cache_path, **payload)
    return payload


def _quiet_mask(scene: dict) -> np.ndarray:
    support = true_fringe_mask(scene["alpha"], scene["max_r"])
    support &= scene["alpha"] < 0.5
    gt_f = scene["gt"].astype(np.float32)
    gt_mid = np.abs(
        gt_f - cv2.GaussianBlur(gt_f, (0, 0), 1.6)
    ).mean(axis=2)
    return support & (gt_mid <= 1.0)


def _fine_error(image: np.ndarray, gt: np.ndarray) -> np.ndarray:
    image_f = image.astype(np.float32)
    gt_f = gt.astype(np.float32)
    image_fine = image_f - cv2.GaussianBlur(image_f, (0, 0), 0.7)
    gt_fine = gt_f - cv2.GaussianBlur(gt_f, (0, 0), 0.7)
    return np.sqrt(np.mean((image_fine - gt_fine) ** 2, axis=2))


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.asarray(left, np.float64).ravel()
    y = np.asarray(right, np.float64).ravel()
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 32:
        return None
    x -= x.mean()
    y -= y.mean()
    denominator = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denominator <= 1e-12:
        return None
    return float(np.sum(x * y) / denominator)


def _gaussian_highpass_noise_stats(
    sigma: float,
    channel_count: int = 3,
) -> tuple[float, float]:
    """Expected mean-absolute 1.6px high-pass response under white noise.

    This is an analytic runtime proxy, not a GT-tuned threshold.  The impulse
    response reproduces OpenCV's discrete Gaussian implementation; the
    half-normal moments then give the expected mean and standard deviation
    after averaging independent color channels.
    """
    side = 65
    impulse = np.zeros((side, side), np.float32)
    impulse[side // 2, side // 2] = 1.0
    highpass = impulse - cv2.GaussianBlur(
        impulse,
        (0, 0),
        1.6,
        borderType=cv2.BORDER_CONSTANT,
    )
    response_sigma = float(sigma) * float(np.sqrt(np.sum(highpass**2)))
    mean_abs = response_sigma * np.sqrt(2.0 / np.pi)
    std_mean_abs = (
        response_sigma
        * np.sqrt(1.0 - 2.0 / np.pi)
        / np.sqrt(float(channel_count))
    )
    return float(mean_abs), float(std_mean_abs)


def _rear_structure_density(
    rear_frame: np.ndarray,
    noise_sigma: float,
    spatial_scale: float,
    *,
    z_threshold: float,
) -> tuple[np.ndarray, dict]:
    """Return regional positive evidence for rear structure above noise.

    Pointwise high-pass excursions become only *votes*.  A broad Gaussian
    density turns them into regional presence evidence, so their noisy
    amplitudes cannot directly stamp the inverse correction.
    """
    rear_f = rear_frame.astype(np.float32)
    observed = np.abs(
        rear_f - cv2.GaussianBlur(
            rear_f,
            (0, 0),
            1.6,
            borderType=cv2.BORDER_REFLECT,
        )
    ).mean(axis=2)
    expected, spread = _gaussian_highpass_noise_stats(noise_sigma)
    threshold = expected + float(z_threshold) * spread
    votes = (observed > threshold).astype(np.float32)
    density = cv2.GaussianBlur(
        votes,
        (0, 0),
        2.0 * spatial_scale,
        borderType=cv2.BORDER_REFLECT,
    )
    return density, {
        "noise_sigma": float(noise_sigma),
        "expected_noise_response": expected,
        "noise_response_spread": spread,
        "vote_threshold": threshold,
        "vote_fraction": float(votes.mean()),
        "density_mean": float(density.mean()),
    }


def _variant(
    scene: dict,
    base: np.ndarray,
    repaired_base: np.ndarray,
    correction: np.ndarray,
    mask: np.ndarray,
    *,
    strength: float = 1.0,
    keep_fine: float = 1.0,
) -> tuple[np.ndarray, dict]:
    low = cv2.GaussianBlur(
        correction,
        (0, 0),
        0.7,
        borderType=cv2.BORDER_REFLECT,
    )
    filtered = low + keep_fine * (correction - low)
    output = np.clip(
        repaired_base.astype(np.float32)
        + strength * filtered * mask[..., None],
        0,
        255,
    ).astype(np.uint8)
    return output, _score(scene, base, output)


def _applied_variant(
    scene: dict,
    base: np.ndarray,
    repaired_base: np.ndarray,
    correction: np.ndarray,
    mask: np.ndarray,
    *,
    quantizer: str = "full_truncate",
) -> tuple[np.ndarray, dict]:
    applied = correction * mask[..., None]
    if quantizer == "full_truncate":
        output = np.clip(
            repaired_base.astype(np.float32) + applied,
            0,
            255,
        ).astype(np.uint8)
    elif quantizer == "signed_truncate":
        delta = np.trunc(applied).astype(np.int16)
        output = np.clip(
            repaired_base.astype(np.int16) + delta,
            0,
            255,
        ).astype(np.uint8)
    elif quantizer == "signed_round":
        delta = np.rint(applied).astype(np.int16)
        output = np.clip(
            repaired_base.astype(np.int16) + delta,
            0,
            255,
        ).astype(np.uint8)
    else:
        raise ValueError(quantizer)
    return output, _score(scene, base, output)


def _smoothed_applied_variant(
    scene: dict,
    base: np.ndarray,
    repaired_base: np.ndarray,
    correction: np.ndarray,
    mask: np.ndarray,
    *,
    sigma: float,
    support: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Grade smoothing after correction and gate have been composed."""
    applied = correction * mask[..., None]
    applied = cv2.GaussianBlur(
        applied,
        (0, 0),
        sigma,
        borderType=cv2.BORDER_REFLECT,
    )
    if support is not None:
        applied *= support[..., None]
    delta = np.rint(applied).astype(np.int16)
    output = np.clip(
        repaired_base.astype(np.int16) + delta,
        0,
        255,
    ).astype(np.uint8)
    return output, _score(scene, base, output)


def _compact_metrics(metrics: dict) -> dict:
    return {
        key: metrics[key]
        for key in (
            "d_ssim",
            "d_mae",
            "mse_base",
            "mse_output",
            "d_false_texture",
            "changed_pixels",
            "changed_closer",
            "changed_worse",
            "partitions",
        )
    }


def attribute() -> None:
    rows = []
    variant_specs = {
        "front_only": (0.0, 1.0),
        "rear_strength_025": (0.25, 1.0),
        "rear_strength_050": (0.50, 1.0),
        "rear_strength_075": (0.75, 1.0),
        "rear_full": (1.0, 1.0),
        "rear_no_finest": (1.0, 0.0),
        "rear_finest_025": (1.0, 0.25),
        "rear_finest_050": (1.0, 0.50),
        "rear_finest_075": (1.0, 0.75),
    }
    extra_variant_names = (
        "rear_signed_truncate",
        "rear_signed_round",
        "rear_no_positive_gate",
        "rear_visibility_blur_3",
        "rear_visibility_blur_6",
        "rear_visibility_blur_12",
        "rear_noise_vote_z2_density02",
        "rear_noise_vote_z25_density02",
        "rear_noise_vote_z3_density02",
        "rear_noise_vote_z3_density05",
        "rear_noise_veto_z3_density10",
        "rear_noise_veto_z3_density20",
        "rear_noise_veto_z3_density30",
        "rear_noise_replace_z3_density10",
        "rear_noise_replace_z3_density20",
        "rear_noise_replace_z3_density30",
        "rear_applied_blur_07",
        "rear_applied_blur_14",
        "rear_applied_blur_28",
        "rear_applied_blur_07_reclip",
        "rear_applied_blur_14_reclip",
        "rear_applied_blur_28_reclip",
    )
    for scene in _load_cases():
        arrays = _components(scene)
        base = arrays["base"]
        repaired = arrays["repaired_base"]
        correction = arrays["correction"]
        mask = arrays["mask"]
        quiet = _quiet_mask(scene)
        applied = correction * mask[..., None]
        applied_fine = applied - cv2.GaussianBlur(
            applied,
            (0, 0),
            0.7,
            borderType=cv2.BORDER_REFLECT,
        )
        coverage = scene["coverage"][1]
        coverage_slope = cv2.magnitude(
            cv2.Sobel(coverage, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(coverage, cv2.CV_32F, 0, 1, ksize=3),
        )
        base_fine_error = _fine_error(base, scene["gt"])
        front_fine_error = _fine_error(repaired, scene["gt"])
        full_float = np.clip(
            repaired.astype(np.float32) + applied,
            0,
            255,
        )
        float_fine_error = _fine_error(full_float, scene["gt"])
        current = full_float.astype(np.uint8)
        current_fine_error = _fine_error(current, scene["gt"])
        variants = {}
        for name, (strength, keep_fine) in variant_specs.items():
            output, metrics = _variant(
                scene,
                base,
                repaired,
                correction,
                mask,
                strength=strength,
                keep_fine=keep_fine,
            )
            variants[name] = _compact_metrics(metrics)
        for name, quantizer in (
            ("rear_signed_truncate", "signed_truncate"),
            ("rear_signed_round", "signed_round"),
        ):
            _, metrics = _applied_variant(
                scene,
                base,
                repaired,
                correction,
                mask,
                quantizer=quantizer,
            )
            variants[name] = _compact_metrics(metrics)

        owner = int(arrays["owner"])
        spatial_scale = max(1.0, max(base.shape[:2]) / MODEL_SIDE)
        max_radius = RADIUS_FRACTION * max(base.shape[:2])
        front_gate, _ = _ownership_gate(
            scene["frames"],
            owner,
            arrays["alpha_est"],
            spatial_scale,
        )
        ordered_gate, _ = _ordered_visibility_gate(
            scene["frames"],
            owner,
            arrays["alpha_est"],
            spatial_scale,
        )
        rear_visibility = np.zeros_like(ordered_gate)
        valid_front = front_gate > 1e-6
        rear_visibility[valid_front] = (
            ordered_gate[valid_front] / front_gate[valid_front]
        )
        optical_fringe = _fringe_mask(
            arrays["alpha_est"],
            max_radius,
            2.0 * spatial_scale,
        )
        identity_support = optical_fringe * (arrays["alpha_est"] < 0.5)
        identity_support[arrays["owner_copy"] > 0] = 0.0
        for applied_sigma in (0.7, 1.4, 2.8):
            suffix = f"{int(10 * applied_sigma):02d}"
            for mode, support in (
                ("", None),
                ("_reclip", identity_support),
            ):
                _, metrics = _smoothed_applied_variant(
                    scene,
                    base,
                    repaired,
                    correction,
                    mask,
                    sigma=applied_sigma,
                    support=support,
                )
                variants[
                    f"rear_applied_blur_{suffix}{mode}"
                ] = _compact_metrics(metrics)
        no_positive_gate = optical_fringe * front_gate
        no_positive_gate[arrays["owner_copy"] > 0] = 0.0
        _, metrics = _applied_variant(
            scene,
            base,
            repaired,
            correction,
            no_positive_gate,
        )
        variants["rear_no_positive_gate"] = _compact_metrics(metrics)
        for sigma in (3.0, 6.0, 12.0):
            smoothed_rear = cv2.GaussianBlur(
                rear_visibility,
                (0, 0),
                sigma,
                borderType=cv2.BORDER_REFLECT,
            )
            alternative_mask = optical_fringe * front_gate * smoothed_rear
            alternative_mask[arrays["owner_copy"] > 0] = 0.0
            _, metrics = _applied_variant(
                scene,
                base,
                repaired,
                correction,
                alternative_mask,
            )
            variants[f"rear_visibility_blur_{int(sigma)}"] = (
                _compact_metrics(metrics)
            )

        noise_sigma = estimate_noise_sigma(scene["frames"])
        noise_reports = {}
        z3_density = None
        for z_threshold, density_floor, name in (
            (2.0, 0.02, "rear_noise_vote_z2_density02"),
            (2.5, 0.02, "rear_noise_vote_z25_density02"),
            (3.0, 0.02, "rear_noise_vote_z3_density02"),
            (3.0, 0.05, "rear_noise_vote_z3_density05"),
        ):
            rear_density, noise_report = _rear_structure_density(
                scene["frames"][1 - owner],
                noise_sigma,
                spatial_scale,
                z_threshold=z_threshold,
            )
            if z_threshold == 3.0:
                z3_density = rear_density
            rear_license = np.clip(
                rear_density / density_floor,
                0.0,
                1.0,
            )
            alternative_mask = optical_fringe * front_gate * rear_license
            alternative_mask[arrays["owner_copy"] > 0] = 0.0
            _, metrics = _applied_variant(
                scene,
                base,
                repaired,
                correction,
                alternative_mask,
            )
            variants[name] = _compact_metrics(metrics)
            noise_reports[name] = {
                **noise_report,
                "density_floor": density_floor,
                "license_mean": float(rear_license.mean()),
                "license_fraction": float((rear_license > 0.5).mean()),
            }
        assert z3_density is not None
        for density_floor in (0.10, 0.20, 0.30):
            rear_license = np.clip(
                z3_density / density_floor,
                0.0,
                1.0,
            )
            suffix = f"{int(100 * density_floor):02d}"
            for mode, alternative_mask in (
                ("veto", mask * rear_license),
                (
                    "replace",
                    optical_fringe * front_gate * rear_license,
                ),
            ):
                alternative_mask = alternative_mask.copy()
                alternative_mask[arrays["owner_copy"] > 0] = 0.0
                _, metrics = _applied_variant(
                    scene,
                    base,
                    repaired,
                    correction,
                    alternative_mask,
                )
                variants[
                    f"rear_noise_{mode}_z3_density{suffix}"
                ] = _compact_metrics(metrics)

        fine_excess = current_fine_error - base_fine_error
        float_excess = float_fine_error - front_fine_error
        sub_lsb = np.max(np.abs(applied), axis=2) < 1.0
        negative = np.any(applied < 0.0, axis=2)
        rounded = np.clip(np.rint(full_float), 0, 255).astype(np.uint8)
        rounded_ft, _ = false_texture_error(
            rounded,
            scene["gt"],
            scene["alpha"],
            scene["max_r"],
        )
        base_ft, quiet_n = false_texture_error(
            base,
            scene["gt"],
            scene["alpha"],
            scene["max_r"],
        )
        front_ft, _ = false_texture_error(
            repaired,
            scene["gt"],
            scene["alpha"],
            scene["max_r"],
        )
        float_ft, _ = false_texture_error(
            full_float,
            scene["gt"],
            scene["alpha"],
            scene["max_r"],
        )
        current_ft, _ = false_texture_error(
            current,
            scene["gt"],
            scene["alpha"],
            scene["max_r"],
        )
        row = {
            "sid": scene["sid"],
            "stratum": scene["stratum"],
            "quiet_pixels": quiet_n,
            "false_texture": {
                "base": base_ft,
                "front_only": front_ft,
                "rear_float": float_ft,
                "rear_uint8_truncate": current_ft,
                "rear_uint8_round": rounded_ft,
                "front_delta": front_ft - base_ft,
                "rear_float_delta_from_front": float_ft - front_ft,
                "truncate_delta_from_float": current_ft - float_ft,
                "round_delta_from_float": rounded_ft - float_ft,
            },
            "quiet_attribution": {
                "owner_copy_pixels": int(
                    (quiet & (arrays["owner_copy"] > 0)).sum()
                ),
                "rear_application_pixels": int((quiet & (mask > 1e-4)).sum()),
                "rear_changed_pixels": int(
                    (
                        quiet
                        & np.any(current != repaired, axis=2)
                    ).sum()
                ),
                "rear_sub_lsb_pixels": int(
                    (quiet & (mask > 1e-4) & sub_lsb).sum()
                ),
                "rear_negative_pixels": int(
                    (quiet & (mask > 1e-4) & negative).sum()
                ),
                "fine_error_worse_pixels": int(
                    (quiet & (fine_excess > 0.0)).sum()
                ),
                "fine_error_better_pixels": int(
                    (quiet & (fine_excess < 0.0)).sum()
                ),
                "applied_rms": float(
                    np.sqrt(np.mean(applied[quiet] ** 2))
                ),
                "applied_finest_rms": float(
                    np.sqrt(np.mean(applied_fine[quiet] ** 2))
                ),
                "correlation_excess_vs_applied_finest": _pearson(
                    fine_excess[quiet],
                    np.sqrt(np.mean(applied_fine[quiet] ** 2, axis=1)),
                ),
                "correlation_float_excess_vs_applied_finest": _pearson(
                    float_excess[quiet],
                    np.sqrt(np.mean(applied_fine[quiet] ** 2, axis=1)),
                ),
                "correlation_excess_vs_mask": _pearson(
                    fine_excess[quiet],
                    mask[quiet],
                ),
                "correlation_excess_vs_coverage_slope": _pearson(
                    fine_excess[quiet],
                    coverage_slope[quiet],
                ),
            },
            "blind_noise_evidence": noise_reports,
            "variants": variants,
        }
        rows.append(row)
        print(
            f"{scene['sid']}: ft base={base_ft:.5f} "
            f"front={front_ft - base_ft:+.5f} "
            f"float={float_ft - base_ft:+.5f} "
            f"trunc={current_ft - base_ft:+.5f} "
            f"round={rounded_ft - base_ft:+.5f}",
            flush=True,
        )

    variant_summary = {}
    for name in (*variant_specs, *extra_variant_names):
        selected = [row["variants"][name] for row in rows]
        variant_summary[name] = {
            "scene_count": len(selected),
            "ssim_positive": sum(row["d_ssim"] > 0 for row in selected),
            "mae_positive": sum(row["d_mae"] < 0 for row in selected),
            "mse_positive": sum(
                row["mse_output"] < row["mse_base"] for row in selected
            ),
            "false_texture_nonregressing": sum(
                row["d_false_texture"] <= 0 for row in selected
            ),
            "mean_d_ssim": float(
                np.mean([row["d_ssim"] for row in selected])
            ),
            "mean_d_mae": float(
                np.mean([row["d_mae"] for row in selected])
            ),
            "mean_d_false_texture": float(
                np.mean([row["d_false_texture"] for row in selected])
            ),
        }
    payload = {
        "experiment": "S15_f62_fineband_causal_attribution",
        "runtime_changed": False,
        "case_count": len(rows),
        "cases": [scene["sid"] for scene in _load_cases()],
        "rows": rows,
        "variant_summary": variant_summary,
    }
    with open(OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(json.dumps(variant_summary, indent=2), flush=True)
    print(f"wrote {OUTPUT}", flush=True)
    print(
        "DOCTRINE: GT graded causal counterfactuals; no public benchmark or "
        "source-similarity score selected a runtime rule.",
        flush=True,
    )


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] != "attribute":
        raise SystemExit("usage: veil_fineband.py attribute")
    attribute()


if __name__ == "__main__":
    main()
