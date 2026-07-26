#!/usr/bin/env python3
"""Evaluate the unchanged shipped recovery pipeline on the F60 V2 factory.

The renderer and development strata are frozen before the holdout split is
generated.  This script never trains or retunes a threshold:

    python research/objocc_v2_eval.py prep dev
    python research/objocc_v2_eval.py audit dev
    python research/objocc_v2_eval.py oracle dev
    python research/objocc_v2_eval.py prep holdout
    python research/objocc_v2_eval.py audit holdout
    python research/objocc_v2_eval.py oracle holdout
"""
from __future__ import annotations

import json
import importlib
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from objocc_v2_gen import scenes  # noqa: E402
from t2_candidates import candidates_with_features  # noqa: E402
from veilship import false_texture_error  # noqa: E402
from focusstack.bridge import run_bridge_many  # noqa: E402
from focusstack.focus import content_aware_energies  # noqa: E402
from focusstack.fusion import fuse_perband, guided_filter  # noqa: E402
from focusstack.io import to_gray_float  # noqa: E402
from focusstack.veil_layers import (  # noqa: E402
    REAR_EVIDENCE_DENSITY,
    VISIBILITY_COVERAGE_FLOOR,
    VISIBILITY_COVERAGE_RAMP,
    recover_giant_veil,
)

import metrics as M  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
package_enhance = importlib.import_module("focusstack.enhance")


def prepare(split: str) -> None:
    loaded = list(scenes(split))
    pass1_paths = []
    mask_inputs = []
    for scene in loaded:
        pass1 = os.path.join(scene["dir"], "pass1.png")
        cv2.imwrite(pass1, fuse_perband(scene["frames"], harden=0.5))
        pass1_paths.append(pass1)
        mask_inputs.extend(
            [
                pass1,
                os.path.join(scene["dir"], "frame_0.png"),
                os.path.join(scene["dir"], "frame_1.png"),
            ]
        )
    print(f"{split}: {len(loaded)} pass-1 images", flush=True)
    if run_bridge_many("depth", pass1_paths, timeout=1800) is None:
        raise RuntimeError("depth bridge failed")
    print(f"{split}: depth bridge complete", flush=True)
    if run_bridge_many("masks", mask_inputs, timeout=1800) is None:
        raise RuntimeError("mask bridge failed")
    print(f"{split}: mixed-base and owner-frame masks complete", flush=True)


def _mae(image: np.ndarray, gt: np.ndarray, support: np.ndarray) -> float | None:
    if not np.any(support):
        return None
    error = np.abs(image.astype(np.float32) - gt.astype(np.float32)).mean(axis=2)
    return float(error[support].mean())


def _partitions(scene: dict, base: np.ndarray, output: np.ndarray) -> dict:
    alpha = scene["alpha"]
    coverage = scene["coverage"][1]
    sharp = alpha >= 0.95
    partitions = {
        "complete_coverage_core": sharp & (coverage >= 0.95),
        "inner_partial_occlusion": (
            sharp & (coverage > 0.05) & (coverage < 0.95)
        ),
        "outer_veil": (alpha < 0.05) & (coverage > 0.05),
        "far_background": coverage <= 0.05,
    }
    return {
        name: {
            "pixels": int(mask.sum()),
            "mae_base": _mae(base, scene["gt"], mask),
            "mae_output": _mae(output, scene["gt"], mask),
        }
        for name, mask in partitions.items()
    }


def _score(scene: dict, base: np.ndarray, output: np.ndarray) -> dict:
    gt = scene["gt"]
    base_error = np.abs(base.astype(np.float32) - gt.astype(np.float32)).mean(
        axis=2
    )
    output_error = np.abs(
        output.astype(np.float32) - gt.astype(np.float32)
    ).mean(axis=2)
    changed = np.any(output != base, axis=2)
    false_base, false_n = false_texture_error(
        base,
        gt,
        scene["alpha"],
        scene["max_r"],
    )
    false_output, _ = false_texture_error(
        output,
        gt,
        scene["alpha"],
        scene["max_r"],
    )
    return {
        "ssim_base": M.ref_ssim(base, gt),
        "ssim_output": M.ref_ssim(output, gt),
        "d_ssim": M.ref_ssim(output, gt) - M.ref_ssim(base, gt),
        "mae_base": float(base_error.mean()),
        "mae_output": float(output_error.mean()),
        "d_mae": float(output_error.mean() - base_error.mean()),
        "mse_base": float(
            np.square(base.astype(np.float32) - gt.astype(np.float32)).mean()
        ),
        "mse_output": float(
            np.square(output.astype(np.float32) - gt.astype(np.float32)).mean()
        ),
        "changed_pixels": int(changed.sum()),
        "changed_closer": int((changed & (output_error < base_error)).sum()),
        "changed_worse": int((changed & (output_error > base_error)).sum()),
        "false_texture_base": false_base,
        "false_texture_output": false_output,
        "d_false_texture": false_output - false_base,
        "false_texture_pixels": false_n,
        "partitions": _partitions(scene, base, output),
    }


def owner_frame_candidates(scene: dict, topk: int = 4) -> list[dict]:
    """Apply the frozen semantic feature recipe to sharp-frame masks.

    F58/F59 used these masks only after a mixed-base candidate was licensed.
    V2 asks the more direct ordering question: can the observed sharp-owner
    silhouette itself propose the foreground layer before fusion has mixed it?
    Feature definitions and thresholds are unchanged.
    """
    frames = scene["frames"]
    height, width = frames[0].shape[:2]
    depth = np.load(os.path.join(scene["dir"], "pass1.png.depth.npy"))
    if depth.shape != (height, width):
        depth = cv2.resize(depth, (width, height))
    depth = (depth - depth.min()) / (np.ptp(depth) + 1e-9)
    grays = [to_gray_float(frame) for frame in frames]
    energies = np.stack(content_aware_energies(grays), axis=0)
    winner = np.argmax(energies, axis=0)
    ordered = np.sort(energies, axis=0)
    decisive = (
        (ordered[-1] - ordered[-2]) / (ordered[-1] + 1e-6) > 0.3
    ) & (ordered[-1] > np.median(ordered[-1]))

    proposed = []
    for owner in range(2):
        masks = np.load(
            os.path.join(scene["dir"], f"frame_{owner}.png.masks.npy")
        )
        for mask_index, raw_mask in enumerate(masks):
            mask = np.asarray(raw_mask) > 0
            area = int(mask.sum())
            if area < 400:
                continue
            depth_in = float(np.median(depth[mask]))
            depth_out = float(np.median(depth[~mask]))
            if depth_in <= depth_out:
                continue
            decisive_in = decisive & mask
            decisive_n = int(decisive_in.sum())
            if decisive_n < 50:
                continue
            ring = (
                cv2.dilate(
                    mask.astype(np.uint8),
                    np.ones((25, 25), np.uint8),
                )
                > 0
            ) & ~mask
            decisive_ring = decisive & ring
            ring_n = int(decisive_ring.sum())
            purity = float(
                (decisive_in & (winner == owner)).sum() / decisive_n
            )
            area_fit = min(
                1.0,
                float(
                    (decisive_in & (winner == owner)).sum()
                    / max(area, 1)
                    * 4.0
                ),
            )
            ring_other = (
                float(
                    (decisive_ring & (winner != owner)).sum() / ring_n
                )
                if ring_n > 50
                else 0.0
            )
            if ring_other < 0.5:
                continue
            score = purity * np.sqrt(area_fit) * ring_other
            alpha = np.clip(
                guided_filter(
                    grays[owner] / 255.0,
                    mask.astype(np.float32),
                    2,
                    1e-4,
                ),
                0.0,
                1.0,
            )
            snapped = alpha > 0.5
            iou = float((snapped & mask).sum()) / (
                float((snapped | mask).sum()) + 1e-6
            )
            proposed.append(
                {
                    "score": score,
                    "mask": mask,
                    "candidate": {
                        "feats": np.asarray(
                            [
                                score,
                                purity,
                                ring_other,
                                area_fit,
                                depth_in - depth_out,
                                area / (height * width),
                                iou,
                            ],
                            np.float32,
                        ),
                        "alpha": alpha,
                        "owner": owner,
                        "source": "owner_frame",
                        "source_mask_index": int(mask_index),
                    },
                }
            )

    selected = []
    for proposal in sorted(
        proposed,
        key=lambda row: row["score"],
        reverse=True,
    ):
        # Frame segmenters often return near-duplicate nested masks.  Keep the
        # highest-scoring observation; this is semantic deduplication, not an
        # outcome-trained rule.
        if any(
            float((proposal["mask"] & kept["mask"]).sum())
            / max(float((proposal["mask"] | kept["mask"]).sum()), 1.0)
            > 0.90
            for kept in selected
        ):
            continue
        selected.append(proposal)
        if len(selected) >= topk:
            break
    return [row["candidate"] for row in selected]


def audit(split: str) -> None:
    rows = []
    for scene in scenes(split):
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
        row = {
            "sid": scene["sid"],
            "stratum": scene["stratum"],
            "factory": scene["factory"],
            "candidate_count": len(candidates),
            "report": report,
            "metrics": _score(scene, base, output),
        }
        rows.append(row)
        metrics = row["metrics"]
        print(
            f"{scene['sid']} {scene['stratum']:5s} "
            f"fired={report['fired']} candidates={len(candidates)} "
            f"dSSIM={metrics['d_ssim']:+.6f} "
            f"dMAE={metrics['d_mae']:+.4f}",
            flush=True,
        )

    fired = [row for row in rows if row["report"]["fired"]]
    payload = {
        "factory": "objocc_v2_exact_disk",
        "split": split,
        "pipeline": "unchanged_f59_owner_parent_support",
        "thresholds_retuned": False,
        "scene_count": len(rows),
        "fired_count": len(fired),
        "rows": rows,
        "fired_summary": {
            "ssim_positive": sum(
                row["metrics"]["d_ssim"] > 0 for row in fired
            ),
            "mae_positive": sum(
                row["metrics"]["d_mae"] < 0 for row in fired
            ),
            "mse_positive": sum(
                row["metrics"]["mse_output"]
                < row["metrics"]["mse_base"]
                for row in fired
            ),
        },
    }
    path = os.path.join(
        HERE,
        f"objocc_v2_{split}_unchanged_pipeline.json",
    )
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload["fired_summary"], indent=2))
    print(f"{len(rows)} scenes, {len(fired)} fires -> {path}")


def owner_audit(split: str) -> None:
    """Audit owner-frame candidates under the frozen package license."""
    rows = []
    for scene in scenes(split):
        base = fuse_perband(scene["frames"], harden=0.5)
        candidates = owner_frame_candidates(scene, topk=4)
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
        row = {
            "sid": scene["sid"],
            "stratum": scene["stratum"],
            "candidate_count": len(candidates),
            "candidate_features": [
                candidate["feats"].tolist() for candidate in candidates
            ],
            "report": report,
            "metrics": _score(scene, base, output),
        }
        rows.append(row)
        metrics = row["metrics"]
        print(
            f"{scene['sid']} {scene['stratum']:5s} "
            f"fired={report['fired']} candidates={len(candidates)} "
            f"dSSIM={metrics['d_ssim']:+.6f} "
            f"dMAE={metrics['d_mae']:+.4f}",
            flush=True,
        )
    fired = [row for row in rows if row["report"]["fired"]]
    payload = {
        "factory": "objocc_v2_exact_disk",
        "split": split,
        "pipeline": "owner_frame_candidates_frozen_f59_package",
        "thresholds_retuned": False,
        "scene_count": len(rows),
        "fired_count": len(fired),
        "rows": rows,
        "fired_summary": {
            "ssim_positive": sum(
                row["metrics"]["d_ssim"] > 0 for row in fired
            ),
            "mae_positive": sum(
                row["metrics"]["d_mae"] < 0 for row in fired
            ),
            "mse_positive": sum(
                row["metrics"]["mse_output"]
                < row["metrics"]["mse_base"]
                for row in fired
            ),
        },
    }
    path = os.path.join(HERE, f"objocc_v2_{split}_owner_candidates.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload["fired_summary"], indent=2))
    print(f"{len(rows)} scenes, {len(fired)} owner fires -> {path}")


def composed(split: str) -> None:
    """Audit the exact auto-enhance entry point with cached V2 bridges."""
    original_bridge = package_enhance.run_bridge
    original_bridge_many = package_enhance.run_bridge_many
    original_enabled = package_enhance.VEIL_AUTO_ENABLED
    rows = []
    try:
        package_enhance.VEIL_AUTO_ENABLED = True
        for scene in scenes(split):
            base = fuse_perband(scene["frames"], harden=0.5)
            pass1 = os.path.join(scene["dir"], "pass1.png")

            def cached_bridge(kind, *_args, **_kwargs):
                suffix = ".depth.npy" if kind == "depth" else ".masks.npy"
                path = pass1 + suffix
                return path if os.path.exists(path) else None

            def cached_bridge_many(kind, image_paths, *_args, **_kwargs):
                if kind != "masks" or len(image_paths) != 3:
                    return None
                paths = [
                    pass1 + ".masks.npy",
                    os.path.join(scene["dir"], "frame_0.png.masks.npy"),
                    os.path.join(scene["dir"], "frame_1.png.masks.npy"),
                ]
                return (
                    paths
                    if all(os.path.exists(path) for path in paths)
                    else None
                )

            package_enhance.run_bridge = cached_bridge
            package_enhance.run_bridge_many = cached_bridge_many
            output, report = package_enhance.enhance(scene["frames"], base)
            row = {
                "sid": scene["sid"],
                "stratum": scene["stratum"],
                "report": report,
                "metrics": _score(scene, base, output),
            }
            rows.append(row)
            metrics = row["metrics"]
            print(
                f"{scene['sid']} {scene['stratum']:5s} "
                f"veil={report['veil_fired']} "
                f"recon={report['recon_fired']} "
                f"dSSIM={metrics['d_ssim']:+.6f} "
                f"dMAE={metrics['d_mae']:+.4f}",
                flush=True,
            )
    finally:
        package_enhance.run_bridge = original_bridge
        package_enhance.run_bridge_many = original_bridge_many
        package_enhance.VEIL_AUTO_ENABLED = original_enabled

    fired = [
        row
        for row in rows
        if row["report"]["veil_fired"] or row["report"]["recon_fired"]
    ]
    payload = {
        "factory": "objocc_v2_exact_disk",
        "split": split,
        "entry_point": "focusstack.enhance.enhance",
        "research_veil_override": True,
        "thresholds_retuned": False,
        "scene_count": len(rows),
        "fired_count": len(fired),
        "rows": rows,
        "fired_summary": {
            "ssim_positive": sum(
                row["metrics"]["d_ssim"] > 0 for row in fired
            ),
            "mae_positive": sum(
                row["metrics"]["d_mae"] < 0 for row in fired
            ),
            "mse_positive": sum(
                row["metrics"]["mse_output"]
                < row["metrics"]["mse_base"]
                for row in fired
            ),
        },
    }
    path = os.path.join(HERE, f"objocc_v2_{split}_composed.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload["fired_summary"], indent=2))
    print(f"{len(rows)} scenes, {len(fired)} composed fires -> {path}")


def oracle(split: str) -> None:
    """Measure the unchanged operator ceiling with exact factory support."""
    rows = []
    for scene in scenes(split):
        base = fuse_perband(scene["frames"], harden=0.5)
        candidate = {
            "alpha": scene["alpha"],
            "owner": 0,
            "feats": np.ones(7, np.float32),
        }
        output, report = recover_giant_veil(
            scene["frames"],
            base,
            [candidate],
        )
        row = {
            "sid": scene["sid"],
            "stratum": scene["stratum"],
            "report": report,
            "metrics": _score(scene, base, output),
        }
        rows.append(row)
        metrics = row["metrics"]
        print(
            f"{scene['sid']} {scene['stratum']:5s} "
            f"fired={report['fired']} "
            f"ratio={report.get('forward_ratio', float('nan')):.3f} "
            f"dSSIM={metrics['d_ssim']:+.6f} "
            f"dMAE={metrics['d_mae']:+.4f}",
            flush=True,
        )
    fired = [row for row in rows if row["report"]["fired"]]
    payload = {
        "factory": "objocc_v2_exact_disk",
        "split": split,
        "pipeline": "unchanged_f59_with_oracle_alpha",
        "thresholds_retuned": False,
        "oracle_fields": ["alpha", "owner"],
        "scene_count": len(rows),
        "fired_count": len(fired),
        "rows": rows,
        "fired_summary": {
            "ssim_positive": sum(
                row["metrics"]["d_ssim"] > 0 for row in fired
            ),
            "mae_positive": sum(
                row["metrics"]["d_mae"] < 0 for row in fired
            ),
            "mse_positive": sum(
                row["metrics"]["mse_output"]
                < row["metrics"]["mse_base"]
                for row in fired
            ),
            "core_mae_positive": sum(
                row["metrics"]["partitions"]["complete_coverage_core"][
                    "mae_output"
                ]
                <= row["metrics"]["partitions"]["complete_coverage_core"][
                    "mae_base"
                ]
                for row in fired
                if row["metrics"]["partitions"]["complete_coverage_core"][
                    "mae_base"
                ]
                is not None
            ),
        },
    }
    path = os.path.join(HERE, f"objocc_v2_{split}_oracle_ceiling.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload["fired_summary"], indent=2))
    print(f"{len(rows)} scenes, {len(fired)} oracle fires -> {path}")


def ordered_visibility_audit(split: str, *, oracle_alpha: bool) -> None:
    """Grade S12's asymmetric visibility rule without replacing F60 evidence."""
    rows = []
    for scene in scenes(split):
        base = fuse_perband(scene["frames"], harden=0.5)
        if oracle_alpha:
            candidates = [
                {
                    "alpha": scene["alpha"],
                    "owner": 0,
                    "feats": np.ones(7, np.float32),
                }
            ]
            owner_masks = None
        else:
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
        row = {
            "sid": scene["sid"],
            "stratum": scene["stratum"],
            "candidate_count": len(candidates),
            "report": report,
            "metrics": _score(scene, base, output),
        }
        rows.append(row)
        metrics = row["metrics"]
        partition_deltas = {
            name: (
                values["mae_output"] - values["mae_base"]
                if values["mae_base"] is not None
                else None
            )
            for name, values in metrics["partitions"].items()
        }
        print(
            f"{scene['sid']} {scene['stratum']:5s} "
            f"fired={report['fired']} candidates={len(candidates)} "
            f"dSSIM={metrics['d_ssim']:+.6f} "
            f"dMAE={metrics['d_mae']:+.4f} "
            f"parts={partition_deltas}",
            flush=True,
        )

    fired = [row for row in rows if row["report"]["fired"]]
    partition_names = (
        "complete_coverage_core",
        "inner_partial_occlusion",
        "outer_veil",
        "far_background",
    )

    def partition_delta(row: dict, name: str) -> float | None:
        values = row["metrics"]["partitions"][name]
        if values["mae_base"] is None:
            return None
        return values["mae_output"] - values["mae_base"]

    partition_summary = {}
    for name in partition_names:
        deltas = [
            delta
            for row in fired
            if (delta := partition_delta(row, name)) is not None
        ]
        partition_summary[name] = {
            "evaluated": len(deltas),
            "nonregressing": sum(delta <= 0 for delta in deltas),
            "mean_delta_mae": float(np.mean(deltas)) if deltas else None,
            "worst_delta_mae": float(np.max(deltas)) if deltas else None,
        }
    all_partitions_nonregressing = sum(
        all(
            delta is None or delta <= 0
            for name in partition_names
            for delta in [partition_delta(row, name)]
        )
        for row in fired
    )
    suffix = (
        "ordered_visibility_oracle"
        if oracle_alpha
        else "ordered_visibility"
    )
    payload = {
        "factory": "objocc_v2_exact_disk",
        "split": split,
        "pipeline": "s12_asymmetric_ordered_visibility",
        "oracle_fields": ["alpha", "owner"] if oracle_alpha else [],
        "owner_support": not oracle_alpha,
        "rear_evidence_density": REAR_EVIDENCE_DENSITY,
        "visibility_coverage_floor": VISIBILITY_COVERAGE_FLOOR,
        "visibility_coverage_ramp": VISIBILITY_COVERAGE_RAMP,
        "thresholds_retuned": False,
        "scene_count": len(rows),
        "fired_count": len(fired),
        "rows": rows,
        "fired_summary": {
            "ssim_positive": sum(
                row["metrics"]["d_ssim"] > 0 for row in fired
            ),
            "mae_positive": sum(
                row["metrics"]["d_mae"] < 0 for row in fired
            ),
            "mse_positive": sum(
                row["metrics"]["mse_output"]
                < row["metrics"]["mse_base"]
                for row in fired
            ),
            "false_texture_nonregressing": sum(
                row["metrics"]["d_false_texture"] <= 0 for row in fired
            ),
            "all_partitions_nonregressing": (
                all_partitions_nonregressing
            ),
            "partition_summary": partition_summary,
        },
    }
    path = os.path.join(HERE, f"objocc_v2_{split}_{suffix}.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload["fired_summary"], indent=2))
    print(f"{len(rows)} scenes, {len(fired)} S12 fires -> {path}")


def main() -> None:
    if (
        len(sys.argv) != 3
        or sys.argv[1]
        not in {
            "prep",
            "audit",
            "owner",
            "composed",
            "oracle",
            "ordered",
            "ordered-oracle",
        }
    ):
        raise SystemExit(
            "usage: objocc_v2_eval.py "
            "{prep|audit|owner|composed|oracle|ordered|ordered-oracle} "
            "{dev|holdout|extension|s12}"
        )
    command, split = sys.argv[1:]
    if split not in {"dev", "holdout", "extension", "s12"}:
        raise SystemExit("split must be dev, holdout, extension, or s12")
    {
        "prep": prepare,
        "audit": audit,
        "owner": owner_audit,
        "composed": composed,
        "oracle": oracle,
        "ordered": lambda selected_split: ordered_visibility_audit(
            selected_split,
            oracle_alpha=False,
        ),
        "ordered-oracle": lambda selected_split: ordered_visibility_audit(
            selected_split,
            oracle_alpha=True,
        ),
    }[command](split)


if __name__ == "__main__":
    main()
