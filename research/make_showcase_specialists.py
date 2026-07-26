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
  inspection/ — an audit workbench with complete inputs, current research output,
      GT-only diagnostics, edit/error maps, and two automatically selected crops
      for every current V2 fire selected across successive validation splits.

Run:  python research/make_showcase_specialists.py [recon|veil|fence|inspection|all]
"""
from __future__ import annotations
import glob
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


def _normal_photo_cases():
    """Run the actual default pipeline on ordinary real photographic stacks."""
    from focusstack.align import align_stack
    from focusstack.enhance import enhance
    from focusstack.focus import content_aware_energies
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
        if len(paths) < 2:
            raise RuntimeError(f"{sid} is missing its input stack")
        raw = [cv2.imread(path) for path in paths]
        if any(image is None for image in raw):
            raise RuntimeError(f"{sid} contains an unreadable frame")
        print(f"  normal inspection {sid}: {len(raw)} real frames")
        aligned = normalize_exposure(align_stack(raw, motion="affine"))
        base = fuse_perband(aligned, harden=0.5)
        output, report = enhance(aligned, base, harden=0.5)

        energies = np.stack(
            content_aware_energies(
                [to_gray_float(image) for image in aligned]
            ),
            axis=0,
        )
        winner = np.argmax(energies, axis=0)
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
                "native_width": int(raw[0].shape[1]),
                "native_height": int(raw[0].shape[0]),
                "assets": assets,
                "metrics": {
                    "changed_pixels_after_base": int(changed.sum()),
                    "changed_fraction_after_base": float(changed.mean()),
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


def _v2_visibility_cases():
    """Build current front-first cases with all optical partitions exposed."""
    from objocc_v2_eval import _score
    from objocc_v2_gen import scenes
    from t2_candidates import candidates_with_features
    from focusstack.veil_layers import (
        MODEL_SIDE,
        RADIUS_FRACTION,
        _fringe_mask,
        _owner_front_reconstruction_support,
        _ordered_visibility_gate,
        complete_owner_support,
        recover_giant_veil,
        refine_owner_candidate,
        select_licensed_candidate,
    )

    selections = (
        (
            "extension",
            "extension_007",
            "diagnosed inner-partial failure · repaired",
            "The F60 counterexample: old global wins hid inner foreground damage. "
            "S12 now hard-selects the missed parent tail and requires positive rear evidence.",
        ),
        (
            "extension",
            "extension_034",
            "post-F60 reference fire",
            "The prior good extension fire, rerun through the same frozen ordered-visibility rule.",
        ),
        (
            "s12",
            "s12_025",
            "first post-S12 validation fire",
            "The only fire in the first 36-scene split generated after ordered visibility was frozen.",
        ),
        (
            "s16",
            "s16_034",
            "fresh-split counterexample · repaired",
            "This fresh-split counterexample exposed the need for focused-owner matte replacement; the final front-first rule is all-partition-positive on it.",
        ),
        (
            "s19",
            "s19_000",
            "post-final holdout · solid foreground",
            "A clean fire in the 72-scene split generated only after F62 was frozen: foreground core, inner partial occlusion, and outer veil all improve while far background is unchanged.",
        ),
        (
            "s19",
            "s19_012",
            "post-final holdout · broad solid foreground",
            "A second independent solid-stratum fire with a large reconstructed front region and nonregression in every optical partition.",
        ),
        (
            "s19",
            "s19_013",
            "post-final holdout · mixed foreground",
            "The independent mixed-stratum fire: complete core and far background are byte-stable while both partial-coverage partitions improve.",
        ),
    )
    diagnostic_points = {
        "extension_007": (
            1048,
            216,
            "User-reported mixed foreground · focused-owner reconstruction",
        ),
    }
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
        output, report = recover_giant_veil(
            scene["frames"],
            base,
            candidates,
            owner_masks_by_frame=owner_masks,
        )
        if not report["fired"]:
            raise RuntimeError(f"{sid} unexpectedly refused: {report}")
        selected, selection_report = select_licensed_candidate(
            scene["frames"],
            candidates,
        )
        if selected is None:
            raise RuntimeError(f"{sid} lost candidate: {selection_report}")
        owner = int(selected["owner"])
        selected, refinement_report = refine_owner_candidate(
            scene["frames"],
            selected,
            owner_masks[owner],
        )
        estimated_alpha = np.clip(
            selected["alpha"].astype(np.float32),
            0.0,
            1.0,
        )
        owner_support, support_report = complete_owner_support(
            scene["frames"],
            selected,
            owner_masks[owner],
        )
        spatial_scale = max(1.0, max(base.shape[:2]) / MODEL_SIDE)
        front_reconstruction = _owner_front_reconstruction_support(
            estimated_alpha,
            owner_masks[owner],
            {**support_report, **refinement_report},
            RADIUS_FRACTION * max(base.shape[:2]),
            spatial_scale,
        )
        if (
            int(owner_support.sum()) != report["owner_support_pixels"]
            or support_report["owner_support_accepted_count"]
            != report["owner_support_accepted_count"]
            or int(front_reconstruction.sum())
            != report["owner_front_reconstruction_pixels"]
        ):
            raise RuntimeError(f"{sid} owner/front audit mismatch")
        ordered_visibility, _ = _ordered_visibility_gate(
            scene["frames"],
            owner,
            estimated_alpha,
            spatial_scale,
        )
        application_mask = (
            _fringe_mask(
                estimated_alpha,
                RADIUS_FRACTION * max(base.shape[:2]),
                2.0 * spatial_scale,
            )
            * ordered_visibility
        )
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
        true_foreground = scene["alpha"] >= 0.5
        true_background = coverage <= 0.05
        support_pixels = int(owner_support.sum())
        support_true_foreground = int(
            (owner_support & true_foreground).sum()
        )
        front_pixels = int(front_reconstruction.sum())
        front_true_foreground = int(
            (front_reconstruction & true_foreground).sum()
        )

        case_dir = os.path.join(INSPECTION_IMG, "s12", sid)
        assets = {
            "frame0": "frame0.jpg",
            "frame1": "frame1.jpg",
            "base": "base.jpg",
            "output": "output.jpg",
            "gt": "gt.jpg",
            "coverage_classes": "coverage_classes.png",
            "estimated_alpha": "estimated_alpha.png",
            "true_alpha": "true_alpha.png",
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
            assets[key] = f"img/inspection/s12/{sid}/{filename}"

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
            f"  S12 inspection {sid}: "
            f"ΔSSIM={metrics['d_ssim']:+.6f} "
            f"ΔMAE={metrics['d_mae']:+.4f}"
        )
    return cases


def _fig_inspection_legacy():
    """Retired V1 inspection builder retained only for artifact reproduction."""
    from t2_candidates import candidates_with_features
    from t2_confidence import scenes
    from veilband import fringe_mask as true_fringe_mask
    from veilship import false_texture_error
    from focusstack.veil_layers import (
        MODEL_SIDE,
        RADIUS_FRACTION,
        _fringe_mask,
        _owner_front_reconstruction_support,
        _ordered_visibility_gate,
        complete_owner_support,
        recover_giant_veil,
        refine_owner_candidate,
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
        selected, refinement_report = refine_owner_candidate(
            sc["frames"],
            selected,
            owner_masks[owner],
        )
        estimated_alpha = np.clip(selected["alpha"].astype(np.float32), 0.0, 1.0)
        owner_support, support_report = complete_owner_support(
            sc["frames"],
            selected,
            owner_masks[owner],
        )
        spatial_scale = max(1.0, max(base.shape[:2]) / MODEL_SIDE)
        front_reconstruction = _owner_front_reconstruction_support(
            estimated_alpha,
            owner_masks[owner],
            {**support_report, **refinement_report},
            RADIUS_FRACTION * max(base.shape[:2]),
            spatial_scale,
        )
        if (
            int(owner_support.sum()) != report["owner_support_pixels"]
            or support_report["owner_support_accepted_count"]
            != report["owner_support_accepted_count"]
            or int(front_reconstruction.sum())
            != report["owner_front_reconstruction_pixels"]
        ):
            raise RuntimeError(f"{sc['sid']} owner/front audit mismatch")
        ownership, _ = _ordered_visibility_gate(
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
        owner_copy_support = owner_support | front_reconstruction
        application_mask[owner_copy_support] = 0.0
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
            "front_reconstruction": "front_reconstruction.png",
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
            "front_reconstruction": _mask_image(
                front_reconstruction.astype(np.float32)
            ),
            "protected": _mask_image(
                np.maximum(
                    1.0 - ownership,
                    owner_copy_support.astype(np.float32),
                )
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
        front_pixels = int(front_reconstruction.sum())
        front_true_foreground = int(
            (front_reconstruction & true_foreground).sum()
        )
        diagnostic_point = None
        if point_spec is not None:
            point_x, point_y, point_label = point_spec
            half = 32
            component_count, component_labels = cv2.connectedComponents(
                owner_copy_support.astype(np.uint8),
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
                "front_reconstruction": bool(
                    front_reconstruction[point_y, point_x]
                ),
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
            "front_reconstruction_pixels": front_pixels,
            "front_reconstruction_true_foreground": front_true_foreground,
            "front_reconstruction_precision": (
                front_true_foreground / front_pixels
                if front_pixels
                else None
            ),
            "front_reconstruction_mae_base": (
                _region_mae(base, sc["gt"], front_reconstruction)
                if front_pixels
                else None
            ),
            "front_reconstruction_mae_output": (
                _region_mae(output, sc["gt"], front_reconstruction)
                if front_pixels
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
    current_cases = _v2_visibility_cases()
    with open(
        os.path.join(HERE, "objocc_v2_s12_ordered_visibility.json"),
        encoding="utf-8",
    ) as handle:
        s12_audit = json.load(handle)
    with open(
        os.path.join(HERE, "objocc_v2_extension_ordered_visibility.json"),
        encoding="utf-8",
    ) as handle:
        extension_audit = json.load(handle)
    with open(
        os.path.join(HERE, "objocc_v2_s16_ordered_visibility.json"),
        encoding="utf-8",
    ) as handle:
        s16_audit = json.load(handle)
    manifest = {
        "schema": 5,
        "title": "focusstack owner inspection lab",
        "generated_from": (
            "S16 front-first refinement audit, physically audited V2 factory, "
            "and legacy V1 failure reproductions"
        ),
        "oracle_warning": (
            "Ground truth, true alpha, error maps, and GT metrics are audit-only. "
            "They are never inputs to runtime recovery. The giant-veil auto path "
            "remains safety-disabled. The first four deep cases show the current "
            "front-first research rule on exact-disk V2; the following five and ten-row "
            "ledger use the superseded V1 factory only for reproducible diagnostics."
        ),
        "case_selection": (
            "Current cases come first: the user-reported failure, the prior "
            "good reference, the first post-visibility fire, and the only fire "
            "from a second post-refinement 36-scene split. Five legacy V1 cases "
            "remain below for exact coordinate reproduction."
        ),
        "audit_sources": sources,
        "s12_summary": {
            "post_freeze_scene_count": s12_audit["scene_count"],
            "post_freeze_fired_count": s12_audit["fired_count"],
            "post_freeze_all_partitions_nonregressing": s12_audit[
                "fired_summary"
            ]["all_partitions_nonregressing"],
            "diagnostic_extension_fired_count": extension_audit[
                "fired_count"
            ],
            "diagnostic_extension_all_partitions_nonregressing": (
                extension_audit["fired_summary"][
                    "all_partitions_nonregressing"
                ]
            ),
            "current_case_count": len(current_cases),
            "post_refinement_scene_count": s16_audit["scene_count"],
            "post_refinement_fired_count": s16_audit["fired_count"],
            "post_refinement_all_partitions_nonregressing": s16_audit[
                "fired_summary"
            ]["all_partitions_nonregressing"],
            "false_texture_warning_count": sum(
                case["metrics"]["d_false_texture"] > 0
                for case in current_cases
            ),
        },
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
        "cases": current_cases + cases,
    }
    with open(INSPECTION_MANIFEST, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(
        f"  wrote {os.path.relpath(INSPECTION_MANIFEST, REPO)} "
        f"({len(current_cases) + len(cases)} deep cases, "
        f"{len(ledger)} ledger rows)"
    )


def fig_inspection():
    """Build the current-only owner workbench and ordinary-photo cohort."""
    from objocc_v2_gen import scenes as v2_scenes

    v2_cases = []
    formation_dir = os.path.join(
        HERE,
        "data",
        "objocc_v2",
        "formation_audit",
        "extension_007_opaque_primary_r12",
    )
    with open(
        os.path.join(
            HERE,
            "objocc_v2_extension_007_opaque_primary.json",
        ),
        encoding="utf-8",
    ) as handle:
        formation_report = json.load(handle)
    formation_asset_dir = os.path.join(
        INSPECTION_IMG,
        "factory_v2",
    )
    formation_filename = "extension_007_opaque_primary_r12.jpg"
    _write_image(
        os.path.join(formation_asset_dir, formation_filename),
        cv2.imread(os.path.join(formation_dir, "comparison.png")),
        max_side=6400,
        quality=96,
    )
    v2_cases.append(
        {
            "sid": formation_report["id"],
            "stratum": "same-scene opaque-primary rerender",
            "asset": (
                "img/inspection/factory_v2/"
                f"{formation_filename}"
            ),
            "core_fraction": formation_report["core_fraction"],
            "inner_veil_fraction": formation_report[
                "inner_veil_fraction"
            ],
            "defocus_radius": formation_report["new_defocus_radius"],
            "description": (
                "Top: old near · old far · new near · new far. "
                "Bottom: GT · old coverage · GT · new coverage. "
                "Same bird/background/placement/noise; only CoC radius changes "
                f"{formation_report['old_defocus_radius']:.1f}→"
                f"{formation_report['new_defocus_radius']:.1f}px. At "
                f"({formation_report['diagnostic_xy'][0]},"
                f"{formation_report['diagnostic_xy'][1]}) coverage changes "
                f"{formation_report['old_diagnostic_coverage']:.3f}→"
                f"{formation_report['new_diagnostic_coverage']:.3f}."
            ),
        }
    )
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

    audit_files = {
        "extension": "objocc_v2_extension_ordered_visibility.json",
        "s12": "objocc_v2_s12_ordered_visibility.json",
        "s16": "objocc_v2_s16_ordered_visibility.json",
        "s19": "objocc_v2_s19_ordered_visibility.json",
    }
    audits = {}
    for split, filename in audit_files.items():
        with open(os.path.join(HERE, filename), encoding="utf-8") as handle:
            audits[split] = json.load(handle)

    current_cases = _v2_visibility_cases()
    normal_cases = _normal_photo_cases()
    manifest = {
        "schema": 6,
        "title": "focusstack owner inspection lab",
        "generated_from": (
            "F62 front-first exact-disk audits plus current default-pipeline "
            "runs on ordinary real photographic stacks"
        ),
        "oracle_warning": (
            "Ground truth, true alpha, error maps, and GT metrics exist only for "
            "the exact-disk physical-stress cases and are audit-only; they never "
            "enter runtime recovery. The normal-photo cohort is real optical input "
            "without all-in-focus GT, so its numbers are descriptive rather than "
            "quality verdicts. The giant-veil auto path remains safety-disabled."
        ),
        "case_selection": (
            "All seven current F62 fires are shown: two diagnosed extension cases, "
            "the S12 validation fire, the repaired S16 counterexample, and all three "
            "fires from the genuinely post-final 72-scene S19 split. Legacy V1 deep "
            "cases are no longer included."
        ),
        "normal_selection": (
            "Six ordinary real-photo stacks show the actual default pipeline: four "
            "classic two-frame photographs and two real phone focal sweeps. Every "
            "original frame is visible in a labeled contact sheet; aligned/normalized "
            "inputs are shown separately so registration artifacts cannot hide."
        ),
        "audit_sources": list(audit_files.values()),
        "s12_summary": {
            "post_freeze_scene_count": audits["s12"]["scene_count"],
            "post_freeze_fired_count": audits["s12"]["fired_count"],
            "post_freeze_all_partitions_nonregressing": audits["s12"][
                "fired_summary"
            ]["all_partitions_nonregressing"],
            "diagnostic_extension_fired_count": audits["extension"][
                "fired_count"
            ],
            "diagnostic_extension_all_partitions_nonregressing": audits[
                "extension"
            ]["fired_summary"]["all_partitions_nonregressing"],
            "current_case_count": len(current_cases),
            "post_refinement_scene_count": audits["s16"]["scene_count"],
            "post_refinement_fired_count": audits["s16"]["fired_count"],
            "post_refinement_all_partitions_nonregressing": audits["s16"][
                "fired_summary"
            ]["all_partitions_nonregressing"],
            "post_final_scene_count": audits["s19"]["scene_count"],
            "post_final_fired_count": audits["s19"]["fired_count"],
            "post_final_all_partitions_nonregressing": audits["s19"][
                "fired_summary"
            ]["all_partitions_nonregressing"],
            "false_texture_warning_count": sum(
                case["metrics"]["d_false_texture"] > 0
                for case in current_cases
            ),
            "normal_case_count": len(normal_cases),
        },
        "factory_v2": {
            "status": (
                "The first panel is an immediate same-scene formation audit: "
                "extension_007 is rerendered with a 12px instead of 37.6px CoC "
                "radius, preserving all content and seeds. The remaining panels "
                "show the frozen V2 solid/mixed/thin regimes."
            ),
            "panel_order": (
                "Special comparison order is stated under its panel. Standard "
                "panels: near-focus · far-focus · all-in-focus GT · optical "
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
