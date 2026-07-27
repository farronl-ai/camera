#!/usr/bin/env python3
"""Regenerate the current S29 and ordinary-photo inspection workbench.

The historical showcase figures remain checked-in visual artifacts. Their
superseded generators are available in Git history; this script deliberately
contains only the live inspection path.

Run:  python research/make_showcase_specialists.py [inspection|all]
"""
from __future__ import annotations
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eyetool import _disagreement, _top_regions, _amplify_diff  # noqa: E402
from focusstack.fusion import fuse_perband  # noqa: E402

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


def _contact_sheet(images, *, columns=4, cell_width=320):
    """Label every source frame in a compact, click-through audit sheet."""
    cells = []
    for index, image in enumerate(images):
        height, width = image.shape[:2]
        scale = cell_width / width
        resized = cv2.resize(
            image,
            (cell_width, max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        header = np.full((34, cell_width, 3), (20, 20, 20), np.uint8)
        cv2.putText(
            header,
            f"frame {index:02d}",
            (10, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        cells.append(np.vstack([header, resized]))

    rows = []
    for start in range(0, len(cells), columns):
        row = cells[start : start + columns]
        target_height = max(cell.shape[0] for cell in row)
        row = [
            np.pad(
                cell,
                ((0, target_height - cell.shape[0]), (0, 0), (0, 0)),
                constant_values=20,
            )
            for cell in row
        ]
        while len(row) < columns:
            row.append(np.full_like(row[0], 20))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def _normal_photo_cases(selected_sids=None):
    """Run the actual default pipeline on ordinary real photographic stacks."""
    from focusstack.align import align_stack
    from focusstack.enhance import enhance
    from focusstack.focus import content_aware_energies
    from focusstack.fusion import (
        fuse_coherent,
        stack_consistency_route,
    )
    from focusstack.io import normalize_exposure, to_gray_float

    selections = (
        (
            "standard_c01",
            sorted(glob.glob(os.path.join(HERE, "data", "standard", "c_01_*.tif"))),
            "classic real pair · golfer",
            "A conventional two-frame photographic stack with a person, fine clothing edges, and a distant background.",
        ),
        (
            "standard_c05",
            sorted(glob.glob(os.path.join(HERE, "data", "standard", "c_05_*.tif"))),
            "classic real pair · fence",
            "The familiar fence/foliage scene, now shown through the current default path rather than the retired subtraction specialist.",
        ),
        (
            "standard_c10",
            sorted(glob.glob(os.path.join(HERE, "data", "standard", "c_10_*.tif"))),
            "classic real pair · trees and person",
            "A typical deep scene with two near trunks, a person, and cluttered foliage at several depths.",
        ),
        (
            "standard_c20",
            sorted(glob.glob(os.path.join(HERE, "data", "standard", "c_20_*.tif"))),
            "classic real pair · toy portrait",
            "A clean subject/background portrait with curved silhouette boundaries and reflective detail.",
        ),
        (
            "mobile_kitchen",
            sorted(glob.glob(os.path.join(HERE, "data", "mobiledepth", "Figure3", "kitchen", "*.jpg"))),
            "real phone sweep · kitchen · 12 frames",
            "A real handheld phone focal sweep over ordinary indoor objects; no synthetic blur and no all-in-focus GT.",
        ),
        (
            "mobile_smallmotion",
            sorted(glob.glob(os.path.join(HERE, "data", "mobiledepth", "Figure6", "smallmotion", "*.jpg"))),
            "real phone sweep · small motion · 14 frames",
            "A normal phone sweep with mild camera/scene motion, included to expose registration and fusion behavior outside ideal alignment.",
        ),
    )
    cases = []
    for sid, paths, role, why in selections:
        if selected_sids is not None and sid not in selected_sids:
            continue
        if len(paths) < 2:
            raise RuntimeError(f"{sid} is missing its input stack")
        raw = [cv2.imread(path) for path in paths]
        if any(image is None for image in raw):
            raise RuntimeError(f"{sid} contains an unreadable frame")
        print(f"  normal inspection {sid}: {len(raw)} real frames")
        aligned = normalize_exposure(align_stack(raw, motion="affine"))
        routed, instability = stack_consistency_route(aligned)
        if routed:
            base, shared_weights = fuse_coherent(
                aligned,
                harden=0.5,
                return_weights=True,
            )
        else:
            base = fuse_perband(
                aligned,
                harden=0.5,
                stack_consistency=False,
            )
            shared_weights = None
        output, report = enhance(aligned, base, harden=0.5)

        energies = np.stack(
            content_aware_energies(
                [to_gray_float(image) for image in aligned]
            ),
            axis=0,
        )
        winner = (
            np.argmax(shared_weights, axis=0)
            if shared_weights is not None
            else np.argmax(energies, axis=0)
        )
        winner_counts = np.bincount(
            winner.ravel(),
            minlength=len(aligned),
        )
        winner_shares = winner_counts / winner.size
        selection_map = cv2.applyColorMap(
            np.uint8(
                np.round(
                    winner.astype(np.float32)
                    / max(len(aligned) - 1, 1)
                    * 255.0
                )
            ),
            cv2.COLORMAP_TURBO,
        )
        edit_map = _amplify_diff(output, base, gain=8.0)
        changed = np.any(output != base, axis=2)
        source_mae = [
            float(
                np.abs(
                    output.astype(np.float32)
                    - image.astype(np.float32)
                ).mean()
            )
            for image in aligned
        ]

        def edge_energy(image):
            gray = to_gray_float(image)
            return float(
                np.mean(np.abs(cv2.Laplacian(gray, cv2.CV_32F)))
            )

        source_edge = [edge_energy(image) for image in aligned]
        case_dir = os.path.join(INSPECTION_IMG, "normal", sid)
        assets = {
            "inputs": "inputs.jpg",
            "aligned": "aligned.jpg",
            "first": "frame_first.jpg",
            "middle": "frame_middle.jpg",
            "last": "frame_last.jpg",
            "base": "base.jpg",
            "output": "output.jpg",
            "selection": "selection.png",
            "edit_x8": "edit_x8.jpg",
        }
        middle = len(raw) // 2
        images = {
            "inputs": _contact_sheet(raw),
            "aligned": _contact_sheet(aligned),
            "first": raw[0],
            "middle": raw[middle],
            "last": raw[-1],
            "base": base,
            "output": output,
            "selection": selection_map,
            "edit_x8": edit_map,
        }
        for key, filename in assets.items():
            _write_image(
                os.path.join(case_dir, filename),
                images[key],
                max_side=3600 if key in {"inputs", "aligned"} else 1800,
                quality=93 if key in {"base", "output"} else 90,
            )
            assets[key] = f"img/inspection/normal/{sid}/{filename}"

        cases.append(
            {
                "sid": sid,
                "role": role,
                "why": why,
                "frame_count": len(raw),
                # Feedback coordinates belong to the fused common footprint,
                # not to the larger uncropped source-frame canvas.
                "native_width": int(output.shape[1]),
                "native_height": int(output.shape[0]),
                "assets": assets,
                "metrics": {
                    "changed_pixels_after_base": int(changed.sum()),
                    "changed_fraction_after_base": float(changed.mean()),
                    "selection_instability": instability,
                    "shared_decision_routed": routed,
                    "active_winner_frames": int(
                        (winner_shares >= 0.005).sum()
                    ),
                    "winner_shares": [
                        float(value) for value in winner_shares
                    ],
                    "closest_source_mae": min(source_mae),
                    "output_edge_energy": edge_energy(output),
                    "median_source_edge_energy": float(
                        np.median(source_edge)
                    ),
                },
                "report": {
                    key: value
                    for key, value in report.items()
                    if isinstance(value, (str, bool, int, float))
                },
                "ground_truth": False,
            }
        )
    return cases


def refresh_normal_cases(selected_sids):
    """Refresh selected ordinary-photo cases without rerunning S29 recovery."""
    selected = set(selected_sids)
    replacements = {
        case["sid"]: case
        for case in _normal_photo_cases(selected)
    }
    missing = selected - replacements.keys()
    if missing:
        raise RuntimeError(f"unknown normal inspection cases: {sorted(missing)}")
    with open(INSPECTION_MANIFEST, encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["normal_cases"] = [
        replacements.get(case["sid"], case)
        for case in manifest["normal_cases"]
    ]
    with open(INSPECTION_MANIFEST, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(
        "  refreshed normal cases: "
        + ", ".join(sorted(replacements))
    )


def _rgb_pixel(image, x, y):
    """JSON-friendly RGB value from OpenCV's BGR storage."""
    return [int(value) for value in image[y, x, ::-1]]


def _v2_visibility_cases(selected_sids=None):
    """Build post-rule S29 cases with all physical partitions exposed."""
    from objocc_v2_eval import _score, candidates_with_features
    from objocc_v2_gen import scenes
    from focusstack.veil_layers import (
        MODEL_SIDE,
        RADIUS_FRACTION,
        _owner_geometry_consensus,
        _one_sided_rear_application_mask,
        recover_giant_veil,
        select_one_sided_owner_geometry,
    )

    selections = (
        (
            "s29",
            "s29_000",
            "post-rule primary · tiny connected foreground restored",
            "The fresh case that reproduced the user's small dark-piece failure. "
            "A strict native graph continuation now hard-selects the focused owner.",
        ),
        (
            "s29",
            "s29_002",
            "post-rule boundary · hardest silhouette",
            "The lowest-IoU licensed S29 geometry. It exposes boundary misses while "
            "still keeping rear application out of every protected GT partition.",
        ),
        (
            "s29",
            "s29_005",
            "post-rule all-veil · strongest direct recovery",
            "A nearly all-veil stress case with the largest S29 core and veil error "
            "reduction; included with the finest-band diagnostic dissent visible.",
        ),
        (
            "s29",
            "s29_007",
            "post-rule primary · ordinary object and SSIM dissent",
            "A representative butterfly/object case where direct physical errors "
            "improve even though global SSIM dissents.",
        ),
        (
            "s29",
            "s29_009",
            "post-rule primary · second graph continuation",
            "The independent S29 case that also licenses a bounded native graph "
            "continuation, providing a non-single-scene check of that mechanism.",
        ),
        (
            "s29",
            "s29_010",
            "post-rule boundary · narrow partial coverage",
            "A compact boundary-dominant object where hard foreground, boundary, "
            "and outward veil all improve with exact far identity.",
        ),
        (
            "s29",
            "s29_011",
            "post-rule all-veil · subpixel-speckle correction",
            "The source case whose two unresolved three-pixel mask islands were "
            "removed without deleting its real small opaque object.",
        ),
    )
    if selected_sids is not None:
        selections = tuple(
            row for row in selections if row[1] in selected_sids
        )
    diagnostic_points = {}
    loaded = {
        split: {scene["sid"]: scene for scene in scenes(split)}
        for split in {selection[0] for selection in selections}
    }
    cases = []
    for split, sid, role, why in selections:
        scene = loaded[split][sid]
        base = fuse_perband(scene["frames"], harden=0.5)
        candidates = candidates_with_features(scene, topk=4)
        owner_masks = [
            np.load(
                os.path.join(
                    scene["dir"],
                    f"frame_{frame_index}.png.masks.npy",
                )
            )
            for frame_index in range(2)
        ]
        selected, selection_report = select_one_sided_owner_geometry(
            scene["frames"],
            owner_masks,
        )
        if selected is None:
            raise RuntimeError(
                f"{sid} unexpectedly lost one-sided geometry: "
                f"{selection_report}"
            )
        output, report = recover_giant_veil(
            scene["frames"],
            base,
            candidates,
            owner_masks_by_frame=owner_masks,
            one_sided_selection=(selected, selection_report),
        )
        if not report["fired"]:
            raise RuntimeError(f"{sid} unexpectedly refused: {report}")
        owner = int(selected["owner"])
        estimated_alpha = np.clip(
            selected["alpha"].astype(np.float32),
            0.0,
            1.0,
        )
        owner_support = np.zeros(estimated_alpha.shape, bool)
        support_report = {
            "owner_support_accepted_count": 0,
            "owner_support_mask_indices": [],
            "owner_support_kinds": [],
        }
        spatial_scale = max(1.0, max(base.shape[:2]) / MODEL_SIDE)
        max_radius = RADIUS_FRACTION * max(base.shape[:2])
        (
            front_consensus,
            fringe_consensus,
            consensus_report,
        ) = _owner_geometry_consensus(
            estimated_alpha,
            owner_masks[owner],
            max_radius,
            spatial_scale,
        )
        inside_distance = cv2.distanceTransform(
            (estimated_alpha >= 0.5).astype(np.uint8),
            cv2.DIST_L2,
            5,
        )
        cross_frame_fragments = np.asarray(
            selected.get(
                "cross_frame_satellite_extent",
                np.zeros(estimated_alpha.shape, bool),
            ),
            bool,
        )
        front_reconstruction = (
            (
                inside_distance > max(1.0, spatial_scale)
            )
            & front_consensus
        ) | cross_frame_fragments
        if int(front_reconstruction.sum()) != report[
            "owner_front_reconstruction_pixels"
        ]:
            raise RuntimeError(f"{sid} owner/front audit mismatch")
        application_mask, _ = _one_sided_rear_application_mask(
            scene["frames"],
            owner,
            estimated_alpha,
            np.asarray(selected["rear_support_alpha"], np.float32),
            np.asarray(selected["front_extent"], bool),
            fringe_consensus,
            max_radius,
            spatial_scale,
            float(selected.get("front_veto_model_pixels", 3.0)),
        )
        ordered_visibility = application_mask
        owner_copy_support = owner_support | front_reconstruction
        application_mask[owner_copy_support] = 0.0

        score = _score(scene, base, output)
        gt = scene["gt"]
        changed = np.any(output != base, axis=2)
        base_error = np.abs(
            base.astype(np.float32) - gt.astype(np.float32)
        ).mean(axis=2)
        output_error = np.abs(
            output.astype(np.float32) - gt.astype(np.float32)
        ).mean(axis=2)
        error_map, error_scale = _error_delta_map(base, output, gt)
        outcome_map = _outcome_map(base, output, gt)
        edit_map = _amplify_diff(output, base, gain=8.0)
        edit_center = _top_regions(_disagreement(base, output), 1, 120)[0]
        heat_worse = np.maximum(output_error - base_error, 0.0) * changed
        worse_center = _top_regions(heat_worse, 1, 120)[0]

        sharp = scene["alpha"] >= 0.95
        coverage = scene["coverage"][1]
        inner = sharp & (coverage > 0.05) & (coverage < 0.95)
        outer = (scene["alpha"] < 0.05) & (coverage > 0.05)
        true_fringe = inner | outer
        true_foreground = scene["hard_ownership"]
        true_background = coverage <= 0.05
        support_pixels = int(owner_support.sum())
        support_true_foreground = int(
            (owner_support & true_foreground).sum()
        )
        front_pixels = int(front_reconstruction.sum())
        front_true_foreground = int(
            (front_reconstruction & true_foreground).sum()
        )

        case_dir = os.path.join(INSPECTION_IMG, "s29", sid)
        assets = {
            "frame0": "frame0.jpg",
            "frame1": "frame1.jpg",
            "base": "base.jpg",
            "output": "output.jpg",
            "gt": "gt.jpg",
            "coverage_classes": "coverage_classes.png",
            "estimated_alpha": "estimated_alpha.png",
            "true_alpha": "true_alpha.png",
            "hard_ownership": "hard_ownership.png",
            "ordered_visibility": "ordered_visibility.png",
            "application_mask": "application_mask.png",
            "owner_support": "owner_support.png",
            "front_reconstruction": "front_reconstruction.png",
            "protected": "protected.png",
            "edit_x8": "edit_x8.jpg",
            "error_delta": "error_delta.jpg",
            "outcomes": "outcomes.png",
            "crop_edit": "crop_edit.jpg",
            "crop_worse": "crop_worse.jpg",
        }
        images = {
            "frame0": scene["frames"][0],
            "frame1": scene["frames"][1],
            "base": base,
            "output": output,
            "gt": gt,
            "coverage_classes": cv2.imread(
                os.path.join(scene["dir"], "coverage_classes.png")
            ),
            "estimated_alpha": _mask_image(estimated_alpha),
            "true_alpha": _mask_image(scene["alpha"]),
            "hard_ownership": _mask_image(
                scene["hard_ownership"].astype(np.float32)
            ),
            "ordered_visibility": _mask_image(ordered_visibility),
            "application_mask": _mask_image(application_mask),
            "owner_support": _mask_image(owner_support.astype(np.float32)),
            "front_reconstruction": _mask_image(
                front_reconstruction.astype(np.float32)
            ),
            "protected": _mask_image(
                np.maximum(
                    1.0 - ordered_visibility,
                    owner_copy_support.astype(np.float32),
                )
            ),
            "edit_x8": edit_map,
            "error_delta": error_map,
            "outcomes": outcome_map,
            "crop_edit": _crop_strip(
                [*scene["frames"], base, output, gt, error_map],
                edit_center,
            ),
            "crop_worse": _crop_strip(
                [*scene["frames"], base, output, gt, error_map],
                worse_center,
            ),
        }
        point_spec = diagnostic_points.get(sid)
        if point_spec is not None:
            point_x, point_y, _ = point_spec
            assets["crop_reported"] = "crop_reported.jpg"
            images["crop_reported"] = _crop_strip(
                [*scene["frames"], base, output, gt, error_map],
                (point_y, point_x),
                half=80,
                zoom=3,
            )
        for key, filename in assets.items():
            max_side = 3200 if key.startswith("crop_") else 1600
            quality = (
                93
                if key in {"frame0", "frame1", "base", "output", "gt"}
                else 90
            )
            _write_image(
                os.path.join(case_dir, filename),
                images[key],
                max_side=max_side,
                quality=quality,
            )
            assets[key] = f"img/inspection/s29/{sid}/{filename}"

        diagnostic_point = None
        if point_spec is not None:
            point_x, point_y, point_label = point_spec
            half = 10
            point_region = np.zeros(base.shape[:2], bool)
            point_region[
                max(0, point_y - half) : min(
                    base.shape[0], point_y + half + 1
                ),
                max(0, point_x - half) : min(
                    base.shape[1], point_x + half + 1
                ),
            ] = True
            diagnostic_point = {
                "x": point_x,
                "y": point_y,
                "label": point_label,
                "region_half_width": half,
                "region_pixels": int(point_region.sum()),
                "rgb": {
                    "frame0": _rgb_pixel(
                        scene["frames"][0], point_x, point_y
                    ),
                    "frame1": _rgb_pixel(
                        scene["frames"][1], point_x, point_y
                    ),
                    "base": _rgb_pixel(base, point_x, point_y),
                    "output": _rgb_pixel(output, point_x, point_y),
                    "gt": _rgb_pixel(gt, point_x, point_y),
                },
                "estimated_alpha": float(
                    estimated_alpha[point_y, point_x]
                ),
                "true_alpha": float(scene["alpha"][point_y, point_x]),
                "owner_support": bool(owner_support[point_y, point_x]),
                "front_reconstruction": bool(
                    front_reconstruction[point_y, point_x]
                ),
                "application_mask": float(
                    application_mask[point_y, point_x]
                ),
                "base_error": float(base_error[point_y, point_x]),
                "output_error": float(output_error[point_y, point_x]),
                "region_mae_base": _region_mae(
                    base, gt, point_region
                ),
                "region_mae_output": _region_mae(
                    output, gt, point_region
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

        base_mse = score["mse_base"]
        output_mse = score["mse_output"]
        metrics = {
            **score,
            "d_mse": output_mse - base_mse,
            "d_psnr": float(
                10.0
                * np.log10(
                    max(base_mse, 1e-12) / max(output_mse, 1e-12)
                )
            ),
            "fringe_mae_base": _region_mae(base, gt, true_fringe),
            "fringe_mae_output": _region_mae(output, gt, true_fringe),
            "foreground_mae_base": _region_mae(
                base,
                gt,
                true_foreground,
            ),
            "foreground_mae_output": _region_mae(
                output,
                gt,
                true_foreground,
            ),
            "far_background_mae_base": _region_mae(
                base,
                gt,
                true_background,
            ),
            "far_background_mae_output": _region_mae(
                output,
                gt,
                true_background,
            ),
            "changed_fraction": float(changed.mean()),
            "changed_on_true_foreground": int(
                (changed & true_foreground).sum()
            ),
            "changed_outside_true_fringe": int(
                (changed & ~true_fringe).sum()
            ),
            "application_coverage": float(
                (application_mask > 1e-4).mean()
            ),
            "owner_support_pixels": support_pixels,
            "owner_support_true_foreground": support_true_foreground,
            "owner_support_precision": (
                support_true_foreground / support_pixels
                if support_pixels
                else None
            ),
            "owner_support_mae_base": (
                _region_mae(base, gt, owner_support)
                if support_pixels
                else None
            ),
            "owner_support_mae_output": (
                _region_mae(output, gt, owner_support)
                if support_pixels
                else None
            ),
            "front_reconstruction_pixels": front_pixels,
            "front_reconstruction_true_foreground": front_true_foreground,
            "front_reconstruction_precision": (
                front_true_foreground / front_pixels
                if front_pixels
                else None
            ),
            "front_reconstruction_mae_base": (
                _region_mae(base, gt, front_reconstruction)
                if front_pixels
                else None
            ),
            "front_reconstruction_mae_output": (
                _region_mae(output, gt, front_reconstruction)
                if front_pixels
                else None
            ),
            "error_map_scale_gray": error_scale,
            "estimated_alpha": _alpha_scores(
                estimated_alpha,
                scene["alpha"],
            ),
        }
        cases.append(
            {
                "sid": sid,
                "index": None,
                "split": split,
                "current_v2": True,
                "role": role,
                "why": why,
                "native_width": int(base.shape[1]),
                "native_height": int(base.shape[0]),
                "owner": owner,
                "edit_center": [
                    int(edit_center[1]),
                    int(edit_center[0]),
                ],
                "worse_center": [
                    int(worse_center[1]),
                    int(worse_center[0]),
                ],
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
            f"  S29 inspection {sid}: "
            f"ΔSSIM={metrics['d_ssim']:+.6f} "
            f"ΔMAE={metrics['d_mae']:+.4f}"
        )
    return cases


def fig_inspection():
    """Build the current-only owner workbench and ordinary-photo cohort."""
    from objocc_v2_gen import scenes as v2_scenes

    v2_cases = []
    for scene in list(v2_scenes("s29"))[:3]:
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

    audit_files = (
        "objocc_v2_s29_formation_audit.json",
        "objocc_v2_s29_geometry_audit.json",
        "objocc_v2_s29_ordered_visibility.json",
    )
    with open(
        os.path.join(HERE, audit_files[0]),
        encoding="utf-8",
    ) as handle:
        formation_audit = json.load(handle)
    with open(
        os.path.join(HERE, audit_files[1]),
        encoding="utf-8",
    ) as handle:
        geometry_audit = json.load(handle)
    with open(
        os.path.join(HERE, audit_files[2]),
        encoding="utf-8",
    ) as handle:
        ordered_audit = json.load(handle)

    current_cases = _v2_visibility_cases()
    normal_cases = _normal_photo_cases()
    manifest = {
        "schema": 7,
        "title": "focusstack owner inspection lab",
        "generated_from": (
            "F76 shipped one-sided recovery on frozen post-rule S29 opaque "
            "inputs plus the current default pipeline on ordinary real "
            "photographic stacks"
        ),
        "oracle_warning": (
            "Ground truth, true alpha, error maps, and GT metrics exist only for "
            "the exact-disk physical-stress cases and are audit-only; they never "
            "enter runtime recovery. The normal-photo cohort is real optical input "
            "without all-in-focus GT, so its numbers are descriptive rather than "
            "quality verdicts. Giant-veil auto is now enabled only for the "
            "validated two-frame regime and identity-refuses failed gates."
        ),
        "case_selection": (
            "All seven deep cases are frozen post-rule S29 inputs rerun through "
            "the shipped F76 path. They include the diagnosed tiny-continuation "
            "repair, hardest boundary, ordinary primary, and all-veil cases. "
            "No legacy formation or stale lower-panel input is shown."
        ),
        "normal_selection": (
            "Six ordinary real-photo stacks show the actual default pipeline: four "
            "classic two-frame photographs and two real phone focal sweeps. Every "
            "original frame is visible in a labeled contact sheet; aligned/normalized "
            "inputs are shown separately so registration artifacts cannot hide."
        ),
        "audit_sources": list(audit_files),
        "s12_summary": {
            "auto_enabled": True,
            "current_s29_scene_count": ordered_audit["scene_count"],
            "current_s29_fired_count": ordered_audit["fired_count"],
            "current_s29_all_partitions_nonregressing": ordered_audit[
                "fired_summary"
            ]["all_partitions_nonregressing"],
            "current_s29_protected_rear_overlap": int(
                sum(geometry_audit["rear_overlap_totals"].values())
            ),
            "formation_owned_invariant": formation_audit[
                "all_v2_owned_exactly_invariant"
            ],
            "formation_outer_veil_retained": formation_audit[
                "all_v2_retain_outer_veil"
            ],
            "current_case_count": len(current_cases),
            "false_texture_warning_count": sum(
                case["metrics"]["d_false_texture"] > 0
                for case in current_cases
            ),
            "normal_case_count": len(normal_cases),
        },
        "factory_v2": {
            "status": (
                "These are the same frozen S29 inputs used in every lower deep "
                "panel. Across all 12 scenes, changing only hidden background "
                "changes zero hard-owned V2 pixels; coverage is exactly one and "
                "an outward foreground veil remains."
            ),
            "panel_order": (
                "Every panel: near-focus · far-focus · all-in-focus GT · optical "
                "classes (green complete core, yellow inner partial, magenta "
                "outer veil). Click any panel for the full-size image."
            ),
            "cases": v2_cases,
        },
        "normal_cases": normal_cases,
        "cases": current_cases,
    }
    with open(INSPECTION_MANIFEST, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(
        f"  wrote {os.path.relpath(INSPECTION_MANIFEST, REPO)} "
        f"({len(current_cases)} physical deep cases, "
        f"{len(normal_cases)} ordinary-photo cases)"
    )



if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "inspection"
    if which in ("inspection", "all"):
        print("generating current inspection -> docs/img/inspection/")
        fig_inspection()
    elif which == "normal":
        if len(sys.argv) < 3:
            raise SystemExit("normal requires at least one case id")
        refresh_normal_cases(sys.argv[2:])
    else:
        raise SystemExit("use inspection, all, or normal CASE [CASE ...]")
    print("done")
