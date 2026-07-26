#!/usr/bin/env python3
"""Generate the specialist-layer figures for docs/SHOWCASE.md.

Figures use the same conventions as make_showcase.py (docs/img/*.jpg, JPEG q85):

  spec_recon.jpg — contour reconstruction on a canonical thin-occluder scene
      where the SHIPPED gate (focusstack.gates.RECON_GATE) actually fires:
      [base perband | reconstructed | ground truth] at the most-differing crop.
  spec_veil.jpg  — retired veil-gain failure on a realistic object scene
      (mechanism at its clearest): [base | corrected | ground truth] fringe crop.
  spec_joint.jpg — the shipped owner-safe joint-layer recovery, including
      physically licensed owner-frame support, on a fresh holdout fire:
      [base | recovered physical scene | ground truth].
  spec_fence.jpg — the former real-data subtraction fire retained as an audit
      artifact: [base | former output | amplified difference] at the wire edge.

Crops are disagreement-guided (eyetool discipline), never hand-picked.
  inspection/ — an audit workbench with complete inputs, shipped output,
      GT-only diagnostics, edit/error maps, and two automatically selected crops
      for five representative joint-layer cases.

Run:  python research/make_showcase_specialists.py [recon|veil|fence|inspection|all]
"""
from __future__ import annotations
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from eyetool import _disagreement, _top_regions, _amplify_diff  # noqa: E402
from focusstack.fusion import fuse_perband  # noqa: E402
from focusstack.gates import RECON_GATE, predict_gain  # noqa: E402
from focusstack.reconstruct import (contamination_band, reconstruct_band,  # noqa: E402
                                    thin_matte_features)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
IMG = os.path.join(REPO, "docs", "img")
INSPECTION_IMG = os.path.join(IMG, "inspection")
INSPECTION_MANIFEST = os.path.join(REPO, "docs", "inspection_manifest.json")
os.makedirs(IMG, exist_ok=True)


def save(name, img, max_w=1620, q=85):
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, round(h * max_w / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(IMG, name), img, [cv2.IMWRITE_JPEG_QUALITY, q])
    print(f"  wrote {name}  ({img.shape[1]}x{img.shape[0]})")


def gap(h, w=6):
    return np.full((h, w, 3), 255, np.uint8)


def hstack(*imgs):
    h = min(i.shape[0] for i in imgs)
    row = []
    for k, im in enumerate(imgs):
        row.append(im[:h])
        if k < len(imgs) - 1:
            row.append(gap(h))
    return np.hstack(row)


def crop_at(imgs, center, half, zoom):
    """Same window from every image, zoomed with nearest (pixel-honest)."""
    h, w = imgs[0].shape[:2]
    y, x = center
    y0 = int(np.clip(y - half, 0, h - 2 * half))
    x0 = int(np.clip(x - half, 0, w - 2 * half))
    sl = (slice(y0, y0 + 2 * half), slice(x0, x0 + 2 * half))
    return [cv2.resize(im[sl], None, fx=zoom, fy=zoom,
                       interpolation=cv2.INTER_NEAREST) for im in imgs]


def _write_image(path, image, *, max_side=1200, quality=91):
    """Write a browser-sized diagnostic while retaining native coordinates."""
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    extension = os.path.splitext(path)[1].lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if extension in (".jpg", ".jpeg") else []
    if not cv2.imwrite(path, image, params):
        raise RuntimeError(f"could not write {path}")


def _mask_image(mask):
    """Perceptually useful monochrome mask without pretending it is RGB data."""
    gray = np.uint8(np.clip(mask, 0.0, 1.0) * 255.0)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _error_delta_map(base, output, gt):
    """Green means closer to GT; magenta means farther; gray means unchanged."""
    base_error = np.abs(base.astype(np.float32) - gt.astype(np.float32)).mean(2)
    output_error = np.abs(output.astype(np.float32) - gt.astype(np.float32)).mean(2)
    delta = base_error - output_error
    changed = np.any(output != base, axis=2)
    support = np.abs(delta[changed])
    scale = float(np.quantile(support, 0.98)) if support.size else 1.0
    strength = np.clip(np.abs(delta) / max(scale, 1e-3), 0.0, 1.0)[..., None]
    neutral = np.full((*delta.shape, 3), (42, 42, 42), np.float32)
    better = np.full_like(neutral, (68, 196, 82))
    worse = np.full_like(neutral, (196, 66, 210))
    color = np.where((delta >= 0)[..., None], better, worse)
    image = neutral * (1.0 - strength) + color * strength
    image[~changed] = (24, 24, 24)
    return np.uint8(np.clip(image, 0, 255)), scale


def _outcome_map(base, output, gt):
    """Discrete map of changed pixels: green closer, magenta worse, blue tied."""
    base_error = np.abs(base.astype(np.float32) - gt.astype(np.float32)).sum(2)
    output_error = np.abs(output.astype(np.float32) - gt.astype(np.float32)).sum(2)
    changed = np.any(output != base, axis=2)
    image = np.full((*base_error.shape, 3), (24, 24, 24), np.uint8)
    image[changed & (output_error < base_error)] = (68, 196, 82)
    image[changed & (output_error > base_error)] = (196, 66, 210)
    image[changed & (output_error == base_error)] = (210, 145, 60)
    return image


def _region_mae(image, gt, region):
    if not np.any(region):
        return float("nan")
    return float(
        np.abs(image.astype(np.float32) - gt.astype(np.float32))[region].mean()
    )


def _alpha_scores(estimated, truth):
    predicted = estimated >= 0.5
    actual = truth >= 0.5
    intersection = int((predicted & actual).sum())
    union = int((predicted | actual).sum())
    predicted_n = int(predicted.sum())
    actual_n = int(actual.sum())
    return {
        "iou": intersection / max(union, 1),
        "precision": intersection / max(predicted_n, 1),
        "recall": intersection / max(actual_n, 1),
        "estimated_pixels": predicted_n,
        "true_pixels": actual_n,
    }


def _crop_strip(images, center, half=120, zoom=2):
    """Same native crop from [frame0, frame1, base, output, GT, delta]."""
    return hstack(*crop_at(images, center, half, zoom))


def _rgb_pixel(image, x, y):
    """JSON-friendly RGB value from OpenCV's BGR storage."""
    return [int(value) for value in image[y, x, ::-1]]


def _inspection_ledger():
    sources = (
        ("development", "veillayers_p14_composed_parent_support_dev.json"),
        ("holdout-1", "veillayers_p14_composed_parent_support_holdout.json"),
        ("holdout-2", "veillayers_p14_composed_parent_support_second_holdout.json"),
    )
    ledger = []
    for split, filename in sources:
        with open(os.path.join(HERE, filename), encoding="utf-8") as handle:
            audit = json.load(handle)
        for row in audit["rows"]:
            evidence = row["report"]["veil_evidence"]
            ledger.append(
                {
                    "sid": row["sid"],
                    "index": row["index"],
                    "split": split,
                    "dg": row["dg"],
                    "d_global_mae": row["d_global_mae"],
                    "d_global_mse": row["d_global_mse"],
                    "d_psnr": row["d_psnr"],
                    "de_fringe": row["de_fringe"],
                    "d_false_texture": row["d_false_texture"],
                    "changed_closer": row["changed_closer"],
                    "changed_worse": row["changed_worse"],
                    "forward_ratio": evidence["forward_ratio"],
                    "stable_fraction": evidence["stable_fraction"],
                    "owner_veto_mean": evidence["owner_veto_mean"],
                    "owner_support_pixels": evidence.get(
                        "owner_support_pixels",
                        0,
                    ),
                    "owner_support_forward_improvement": evidence.get(
                        "owner_support_forward_improvement",
                        0.0,
                    ),
                    "changed_pixels": evidence["changed_pixels"],
                }
            )
    return ledger, [source[1] for source in sources]


def fig_inspection():
    """Build the owner-facing inspection dataset from the shipped veil operator."""
    from t2_candidates import candidates_with_features
    from t2_confidence import scenes
    from veilband import fringe_mask as true_fringe_mask
    from veilship import false_texture_error
    from focusstack.veil_layers import (
        MODEL_SIDE,
        RADIUS_FRACTION,
        _fringe_mask,
        _ownership_gate,
        complete_owner_support,
        recover_giant_veil,
        select_licensed_candidate,
    )

    # Deliberately includes adverse/weak cases, not only the strongest pictures.
    selections = {
        72: (
            "development · false-texture stress",
            "Largest disclosed false-texture-index tail in the licensed set.",
        ),
        75: (
            "development · weakest SSIM win",
            "Smallest positive GT-SSIM delta: a near-boundary license stress case.",
        ),
        114: (
            "holdout 1 · diagnosed foreground miss",
            "The reported black appendage at x=187, y=252 now exercises licensed owner-frame support.",
        ),
        122: (
            "holdout 1 · support and ownership stress",
            "A broad owner-support completion plus the independent focus-ownership protection.",
        ),
        147: (
            "holdout 2 · untouched confirmation",
            "Only licensed fire in the second scene-disjoint untouched holdout.",
        ),
    }
    diagnostic_points = {
        114: (
            187,
            252,
            "User-reported black foreground appendage",
        ),
        122: (
            804,
            521,
            "User-reported yellow foreground core",
        ),
    }
    all_scenes = list(scenes())
    cases = []
    for index, (role, why) in selections.items():
        sc = all_scenes[index]
        print(f"  inspection {sc['sid']}: running shipped recovery")
        base = fuse_perband(sc["frames"], harden=0.5)
        candidates = candidates_with_features(sc, topk=4)
        owner_masks = [
            np.load(
                os.path.join(
                    sc["dir"],
                    f"frame_{frame_index}.png.masks.npy",
                )
            )
            for frame_index in range(len(sc["frames"]))
        ]
        output, report = recover_giant_veil(
            sc["frames"],
            base,
            candidates,
            owner_masks_by_frame=owner_masks,
        )
        if not report["fired"]:
            raise RuntimeError(f"{sc['sid']} unexpectedly refused: {report}")

        selected, selection_report = select_licensed_candidate(sc["frames"], candidates)
        if selected is None:
            raise RuntimeError(f"{sc['sid']} lost licensed candidate: {selection_report}")
        owner = int(selected["owner"])
        estimated_alpha = np.clip(selected["alpha"].astype(np.float32), 0.0, 1.0)
        owner_support, support_report = complete_owner_support(
            sc["frames"],
            selected,
            owner_masks[owner],
        )
        if (
            int(owner_support.sum()) != report["owner_support_pixels"]
            or support_report["owner_support_accepted_count"]
            != report["owner_support_accepted_count"]
        ):
            raise RuntimeError(f"{sc['sid']} owner-support audit mismatch")
        spatial_scale = max(base.shape[:2]) / MODEL_SIDE
        ownership, _ = _ownership_gate(
            sc["frames"],
            owner,
            estimated_alpha,
            spatial_scale,
        )
        estimated_fringe = _fringe_mask(
            estimated_alpha,
            RADIUS_FRACTION * max(base.shape[:2]),
            2.0 * spatial_scale,
        )
        application_mask = estimated_fringe * ownership
        application_mask[owner_support] = 0.0
        changed = np.any(output != base, axis=2)
        true_fringe = true_fringe_mask(sc["alpha"], sc["max_r"])
        true_foreground = sc["alpha"] >= 0.5
        true_background = ~true_foreground & ~true_fringe

        error_map, error_scale = _error_delta_map(base, output, sc["gt"])
        outcome_map = _outcome_map(base, output, sc["gt"])
        edit_map = _amplify_diff(output, base, gain=8.0)
        heat_edit = _disagreement(base, output)
        base_error = np.abs(base.astype(np.float32) - sc["gt"].astype(np.float32)).mean(2)
        output_error = np.abs(output.astype(np.float32) - sc["gt"].astype(np.float32)).mean(2)
        heat_worse = np.maximum(output_error - base_error, 0.0) * changed
        edit_center = _top_regions(heat_edit, 1, 120)[0]
        worse_center = _top_regions(heat_worse, 1, 120)[0]

        case_dir = os.path.join(INSPECTION_IMG, sc["sid"])
        assets = {
            "frame0": "frame0.jpg",
            "frame1": "frame1.jpg",
            "base": "base.jpg",
            "output": "output.jpg",
            "gt": "gt.jpg",
            "estimated_alpha": "estimated_alpha.png",
            "true_alpha": "true_alpha.png",
            "application_mask": "application_mask.png",
            "owner_support": "owner_support.png",
            "protected": "protected.png",
            "edit_x8": "edit_x8.jpg",
            "error_delta": "error_delta.jpg",
            "outcomes": "outcomes.png",
            "crop_edit": "crop_edit.jpg",
            "crop_worse": "crop_worse.jpg",
        }
        images = {
            "frame0": sc["frames"][0],
            "frame1": sc["frames"][1],
            "base": base,
            "output": output,
            "gt": sc["gt"],
            "estimated_alpha": _mask_image(estimated_alpha),
            "true_alpha": _mask_image(sc["alpha"]),
            "application_mask": _mask_image(application_mask),
            "owner_support": _mask_image(owner_support.astype(np.float32)),
            "protected": _mask_image(
                np.maximum(1.0 - ownership, owner_support.astype(np.float32))
            ),
            "edit_x8": edit_map,
            "error_delta": error_map,
            "outcomes": outcome_map,
            "crop_edit": _crop_strip(
                [*sc["frames"], base, output, sc["gt"], error_map],
                edit_center,
            ),
            "crop_worse": _crop_strip(
                [*sc["frames"], base, output, sc["gt"], error_map],
                worse_center,
            ),
        }
        point_spec = diagnostic_points.get(index)
        if point_spec is not None:
            point_x, point_y, _ = point_spec
            assets["crop_reported"] = "crop_reported.jpg"
            images["crop_reported"] = _crop_strip(
                [*sc["frames"], base, output, sc["gt"], error_map],
                (point_y, point_x),
                half=80,
                zoom=3,
            )
        for key, filename in assets.items():
            max_side = 3200 if key.startswith("crop_") else 1600
            quality = 93 if key in {"frame0", "frame1", "base", "output", "gt"} else 90
            _write_image(
                os.path.join(case_dir, filename),
                images[key],
                max_side=max_side,
                quality=quality,
            )
            assets[key] = f"img/inspection/{sc['sid']}/{filename}"

        base_f = base.astype(np.float32)
        output_f = output.astype(np.float32)
        gt_f = sc["gt"].astype(np.float32)
        base_mae = float(np.abs(base_f - gt_f).mean())
        output_mae = float(np.abs(output_f - gt_f).mean())
        base_mse = float(np.mean((base_f - gt_f) ** 2))
        output_mse = float(np.mean((output_f - gt_f) ** 2))
        ft_base, ft_pixels = false_texture_error(
            base, sc["gt"], sc["alpha"], sc["max_r"]
        )
        ft_output, _ = false_texture_error(
            output, sc["gt"], sc["alpha"], sc["max_r"]
        )
        alpha_scores = _alpha_scores(estimated_alpha, sc["alpha"])
        support_pixels = int(owner_support.sum())
        support_true_foreground = int(
            (owner_support & true_foreground).sum()
        )
        diagnostic_point = None
        if point_spec is not None:
            point_x, point_y, point_label = point_spec
            half = 32
            component_count, component_labels = cv2.connectedComponents(
                owner_support.astype(np.uint8),
            )
            point_component = int(component_labels[point_y, point_x])
            if component_count <= 1 or point_component == 0:
                raise RuntimeError(
                    f"{sc['sid']} reported point lost owner support"
                )
            point_region = component_labels == point_component
            diagnostic_point = {
                "x": point_x,
                "y": point_y,
                "label": point_label,
                "region_half_width": half,
                "region_pixels": int(point_region.sum()),
                "rgb": {
                    "frame0": _rgb_pixel(sc["frames"][0], point_x, point_y),
                    "frame1": _rgb_pixel(sc["frames"][1], point_x, point_y),
                    "base": _rgb_pixel(base, point_x, point_y),
                    "output": _rgb_pixel(output, point_x, point_y),
                    "gt": _rgb_pixel(sc["gt"], point_x, point_y),
                },
                "estimated_alpha": float(
                    estimated_alpha[point_y, point_x]
                ),
                "true_alpha": float(sc["alpha"][point_y, point_x]),
                "owner_support": bool(owner_support[point_y, point_x]),
                "application_mask": float(
                    application_mask[point_y, point_x]
                ),
                "base_error": float(
                    np.abs(base_f - gt_f)[point_y, point_x].mean()
                ),
                "output_error": float(
                    np.abs(output_f - gt_f)[point_y, point_x].mean()
                ),
                "region_mae_base": _region_mae(
                    base,
                    sc["gt"],
                    point_region,
                ),
                "region_mae_output": _region_mae(
                    output,
                    sc["gt"],
                    point_region,
                ),
                "region_changed_closer": int(
                    (
                        point_region
                        & changed
                        & (output_error < base_error)
                    ).sum()
                ),
                "region_changed_worse": int(
                    (
                        point_region
                        & changed
                        & (output_error > base_error)
                    ).sum()
                ),
            }
        metrics = {
            "ssim_base": M.ref_ssim(base, sc["gt"]),
            "ssim_output": M.ref_ssim(output, sc["gt"]),
            "d_ssim": M.ref_ssim(output, sc["gt"]) - M.ref_ssim(base, sc["gt"]),
            "mae_base": base_mae,
            "mae_output": output_mae,
            "d_mae": output_mae - base_mae,
            "mse_base": base_mse,
            "mse_output": output_mse,
            "d_mse": output_mse - base_mse,
            "d_psnr": float(
                10.0 * np.log10(max(base_mse, 1e-12) / max(output_mse, 1e-12))
            ),
            "fringe_mae_base": _region_mae(base, sc["gt"], true_fringe),
            "fringe_mae_output": _region_mae(output, sc["gt"], true_fringe),
            "foreground_mae_base": _region_mae(base, sc["gt"], true_foreground),
            "foreground_mae_output": _region_mae(output, sc["gt"], true_foreground),
            "far_background_mae_base": _region_mae(base, sc["gt"], true_background),
            "far_background_mae_output": _region_mae(
                output, sc["gt"], true_background
            ),
            "false_texture_base": ft_base,
            "false_texture_output": ft_output,
            "d_false_texture": ft_output - ft_base,
            "false_texture_pixels": ft_pixels,
            "changed_pixels": int(changed.sum()),
            "changed_fraction": float(changed.mean()),
            "changed_closer": int((changed & (output_error < base_error)).sum()),
            "changed_worse": int((changed & (output_error > base_error)).sum()),
            "changed_on_true_foreground": int((changed & true_foreground).sum()),
            "changed_outside_true_fringe": int((changed & ~true_fringe).sum()),
            "application_coverage": float((application_mask > 1e-4).mean()),
            "owner_support_pixels": support_pixels,
            "owner_support_true_foreground": support_true_foreground,
            "owner_support_precision": (
                support_true_foreground / support_pixels
                if support_pixels
                else None
            ),
            "owner_support_mae_base": (
                _region_mae(base, sc["gt"], owner_support)
                if support_pixels
                else None
            ),
            "owner_support_mae_output": (
                _region_mae(output, sc["gt"], owner_support)
                if support_pixels
                else None
            ),
            "error_map_scale_gray": error_scale,
            "estimated_alpha": alpha_scores,
        }
        cases.append(
            {
                "sid": sc["sid"],
                "index": index,
                "role": role,
                "why": why,
                "native_width": int(base.shape[1]),
                "native_height": int(base.shape[0]),
                "owner": owner,
                "edit_center": [int(edit_center[1]), int(edit_center[0])],
                "worse_center": [int(worse_center[1]), int(worse_center[0])],
                "assets": assets,
                "metrics": metrics,
                "report": {
                    key: value
                    for key, value in report.items()
                    if isinstance(value, (str, bool, int, float, list))
                },
                "diagnostic_point": diagnostic_point,
            }
        )
        print(
            f"    ΔSSIM={metrics['d_ssim']:+.6f} "
            f"ΔMAE={metrics['d_mae']:+.4f} "
            f"closer/worse={metrics['changed_closer']}/{metrics['changed_worse']}"
        )

    ledger, sources = _inspection_ledger()
    from objocc_v2_gen import scenes as v2_scenes
    v2_cases = []
    for scene in list(v2_scenes("dev"))[:3]:
        source = os.path.join(scene["dir"], "vis.png")
        asset_dir = os.path.join(INSPECTION_IMG, "factory_v2")
        filename = f"{scene['sid']}.jpg"
        _write_image(
            os.path.join(asset_dir, filename),
            cv2.imread(source),
            max_side=3200,
            quality=94,
        )
        v2_cases.append(
            {
                "sid": scene["sid"],
                "stratum": scene["stratum"],
                "asset": f"img/inspection/factory_v2/{filename}",
                "core_fraction": scene["factory"]["core_fraction"],
                "inner_veil_fraction": scene["factory"][
                    "inner_veil_fraction"
                ],
                "defocus_radius": scene["factory"]["defocus_radius"],
            }
        )
    manifest = {
        "schema": 3,
        "title": "focusstack owner inspection lab",
        "generated_from": (
            "legacy V1 recovery audit plus physically audited V2 factory"
        ),
        "oracle_warning": (
            "Ground truth, true alpha, error maps, and GT metrics are audit-only. "
            "They are never inputs to runtime recovery. The giant-veil auto path is currently "
            "safety-disabled: the five deep cases and ten-row ledger below use the "
            "superseded V1 factory and remain only as reproducible diagnostics."
        ),
        "case_selection": (
            "Adversarial/diagnostic selection: weakest licensed win, largest "
            "false-texture tail, the user-reported scene-114 foreground miss, "
            "ownership stress, and a second scene-disjoint holdout."
        ),
        "audit_sources": sources,
        "factory_v2": {
            "status": (
                "Current validation foundation: exact circular aperture, explicit "
                "frame-specific coverage, cleaned foreground radiance, and separate "
                "solid/mixed/thin strata."
            ),
            "panel_order": (
                "near-focus frame · far-focus frame · all-in-focus GT · optical "
                "coverage classes (green complete core, yellow inner partial "
                "occlusion, magenta outer veil)"
            ),
            "cases": v2_cases,
        },
        "ledger": ledger,
        "cases": cases,
    }
    with open(INSPECTION_MANIFEST, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(
        f"  wrote {os.path.relpath(INSPECTION_MANIFEST, REPO)} "
        f"({len(cases)} deep cases, {len(ledger)} ledger rows)"
    )


def fig_recon():
    """Thin-occluder contour reconstruction, gate-verified on the shipped model."""
    from thinocc_gate import thin_scenes
    from reconstruct import estimate_alpha_v3
    # dg-ranked candidates from the F48 label cache (canonical thin, idx<120,
    # restricted to low indices so the factory build stays cheap)
    candidates = [15, 5, 35, 25, 2, 13]
    scs = thin_scenes(max(candidates) + 1)
    for idx in candidates:
        sc = scs[idx]
        a3, owner = estimate_alpha_v3(sc["frames"], sc["max_r"])
        if a3.max() <= 0:
            print(f"  thin_{idx}: no matte, skip")
            continue
        feats = thin_matte_features(sc["frames"], a3, sc["max_r"])
        gain = predict_gain(RECON_GATE, feats) if feats is not None else -1
        fires = feats is not None and gain >= RECON_GATE["margin"]
        print(f"  thin_{idx}: predicted gain {gain:+.4f} (margin "
              f"{RECON_GATE['margin']:+.4f}) -> {'FIRE' if fires else 'refuse'}")
        if not fires:
            continue
        base = fuse_perband(sc["frames"], harden=0.5)
        rec = reconstruct_band([sc["frames"][owner], sc["frames"][1 - owner]], a3,
                               contamination_band(a3, sc["max_r"]), base, sc["max_r"])
        dg = M.ref_ssim(rec, sc["gt"]) - M.ref_ssim(base, sc["gt"])
        print(f"    actual GT-SSIM delta {dg:+.4f}")
        heat = _disagreement(base, rec)
        (cy, cx), = _top_regions(heat, 1, 110)
        cells = crop_at([base, rec, sc["gt"]], (cy, cx), 110, 3)
        save("spec_recon.jpg", hstack(*cells))
        # companion: what the specialist sees, same crop — the owner frame, the
        # C3 difference matte, and the contamination band it re-renders
        band = contamination_band(a3, sc["max_r"])
        matte = cv2.cvtColor((a3 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        tint = base.astype(np.float32).copy()
        bm = band.astype(np.float32)[..., None]
        tint = tint * (1 - 0.55 * bm) + np.array([0.0, 140.0, 255.0]) * 0.55 * bm
        mcells = crop_at([sc["frames"][owner], matte,
                          np.clip(tint, 0, 255).astype(np.uint8)], (cy, cx), 110, 3)
        save("spec_matte.jpg", hstack(*mcells))
        return
    print("  no candidate fired — no figure written")


def fig_veil():
    """F54 failure panel: subtraction, rejected gain, and GT.

    Scene 31 is selected by the recorded realistic-object oracle audit because
    it is the worst low-matte-error counterexample—not for visual drama. The
    crop is located automatically where gain increases GT error inside the true
    fringe.
    """
    from t2_confidence import scenes
    from veilband import fringe_mask
    from veilgain import (
        WINNER,
        build_D_ca,
        calibrate_band_noise,
        fuse_perband_gain,
        glaw_sq,
    )
    from focusstack.fusion import _auto_levels

    sc = list(scenes())[31]
    D, ab, pm = build_D_ca(
        sc["frames"], sc["alpha"], sc["max_r"], owner=0, far_idx=1
    )
    levels = _auto_levels(sc["gt"].shape, None)
    ck, _ = calibrate_band_noise(sc["gt"].shape[:2], levels)
    subtraction = fuse_perband_gain(
        sc["frames"], {1: D}, ab, sc["alpha"], omega=0.0
    )
    hybrid = fuse_perband_gain(
        sc["frames"],
        {1: D},
        ab,
        sc["alpha"],
        g_law=glaw_sq,
        shrink_m=2.0,
        ck=ck,
        pm_by_far={1: pm},
        **WINNER,
    )
    fringe = fringe_mask(sc["alpha"], sc["max_r"])
    e_sub = np.abs(subtraction.astype(np.float32) - sc["gt"]).mean(axis=2)
    e_hybrid = np.abs(hybrid.astype(np.float32) - sc["gt"]).mean(axis=2)
    heat = np.maximum(e_hybrid - e_sub, 0.0) * fringe.astype(np.float32)
    (cy, cx), = _top_regions(heat, 1, 130)
    cells = crop_at(
        [subtraction, hybrid, sc["gt"]], (cy, cx), 130, 2.5
    )
    print(
        f"  {sc['sid']}: subtraction {M.ref_ssim(subtraction, sc['gt']):.4f}, "
        f"rejected gain {M.ref_ssim(hybrid, sc['gt']):.4f}; crop ({cy},{cx})"
    )
    save("spec_veil.jpg", hstack(*cells))


def fig_joint():
    """Shipped joint-layer recovery on an actually licensed holdout scene."""
    from t2_candidates import candidates_with_features
    from t2_confidence import scenes
    from veilship import false_texture_error
    from focusstack.veil_layers import recover_giant_veil

    sc = list(scenes())[114]
    base = fuse_perband(sc["frames"], harden=0.5)
    output, report = recover_giant_veil(
        sc["frames"],
        base,
        candidates_with_features(sc, topk=4),
        owner_masks_by_frame=[
            np.load(
                os.path.join(
                    sc["dir"],
                    f"frame_{frame_index}.png.masks.npy",
                )
            )
            for frame_index in range(len(sc["frames"]))
        ],
    )
    if not report["fired"]:
        print(f"  {sc['sid']}: package refused unexpectedly — no figure written")
        return

    dg = M.ref_ssim(output, sc["gt"]) - M.ref_ssim(base, sc["gt"])
    base_error = np.abs(
        base.astype(np.float32) - sc["gt"].astype(np.float32)
    ).mean()
    output_error = np.abs(
        output.astype(np.float32) - sc["gt"].astype(np.float32)
    ).mean()
    ft0, _ = false_texture_error(
        base,
        sc["gt"],
        sc["alpha"],
        sc["max_r"],
    )
    ft, _ = false_texture_error(
        output,
        sc["gt"],
        sc["alpha"],
        sc["max_r"],
    )
    heat = _disagreement(base, output)
    (cy, cx), = _top_regions(heat, 1, 130)
    cells = crop_at(
        [base, output, sc["gt"]],
        (cy, cx),
        130,
        2.5,
    )
    amplified = _amplify_diff(cells[1], cells[0], gain=5.0)
    print(
        f"  {sc['sid']}: rank={report['candidate_rank']} "
        f"forward_ratio={report['forward_ratio']:.3f}, "
        f"GT-SSIM delta={dg:+.5f}, MAE={base_error:.3f}->{output_error:.3f}, "
        f"false-texture={ft0:.3f}->{ft:.3f}; crop ({cy},{cx})"
    )
    save(
        "spec_joint.jpg",
        hstack(*cells, amplified),
        max_w=2160,
        q=88,
    )


def fig_fence():
    """Former real-data subtraction fire retained as an audit artifact.

    The enhanced output is cached (the semantic bridge takes minutes) so the
    historical figure composition can iterate cheaply.  Do not delete the
    cache: current ``enhance`` runs the replacement model, not this old branch.
    """
    from focusstack.enhance import enhance
    a = cv2.imread(os.path.join(HERE, "data", "standard", "c_05_1.tif"))
    b = cv2.imread(os.path.join(HERE, "data", "standard", "c_05_2.tif"))
    base = fuse_perband([a, b], harden=0.5)
    cache = os.path.join(HERE, "inspect", "fence_enhanced.npz")
    if os.path.exists(cache):
        out = np.load(cache)["out"]
    else:
        out, rep = enhance([a, b], base, log=print)
        print(f"  fence: report={rep}")
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, out=out)
    d = float(np.abs(out.astype(np.int16) - base.astype(np.int16)).mean())
    print(f"  fence: mean diff={d:.3f}")
    if d == 0:
        print("  nothing fired — no figure written")
        return
    heat = _disagreement(base, out, win=31)
    (cy, cx), = _top_regions(heat, 1, 70)
    cells = crop_at([base, out], (cy, cx), 70, 4)
    amp = _amplify_diff(*crop_at([out, base], (cy, cx), 70, 1), gain=16.0)
    amp = cv2.resize(amp, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    print(f"  crop ({cy},{cx})")
    save("spec_fence.jpg", hstack(cells[0], cells[1], amp))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    print("generating specialist figures -> docs/img/")
    if which in ("recon", "all"):
        fig_recon()
    if which in ("veil", "all"):
        fig_veil()
    if which in ("joint", "all"):
        fig_joint()
    if which in ("fence", "all"):
        fig_fence()
    if which in ("inspection", "all"):
        fig_inspection()
    print("done")
