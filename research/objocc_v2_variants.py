#!/usr/bin/env python3
"""Re-render controlled formation variants from an existing V2 scene.

This is deliberately separate from the random factory. It preserves the exact
source object, background, placement, and noise seeds while changing one
physical variable, so an inspector can compare inputs without scene-content
confounds.

Run:
    .venv/bin/python research/objocc_v2_variants.py extension_007_primary
"""
from __future__ import annotations

import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hires_gen import add_noise  # noqa: E402
from objocc_gen import LONG  # noqa: E402
from objocc_v2_gen import (  # noqa: E402
    DEFOCUS_DISTANCE,
    OUT,
    SRC,
    _prepare_object,
    _source_assets,
    coverage_classification,
    coverage_stats,
    render_layered_focal_pair,
)


HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT_OUT = os.path.join(OUT, "formation_audit")
PRIMARY_REPORT = os.path.join(
    HERE,
    "objocc_v2_extension_007_opaque_primary.json",
)


def _resize_source(path: str) -> np.ndarray:
    original = cv2.imread(path)
    height, width = original.shape[:2]
    scale = LONG / max(height, width)
    return cv2.resize(
        original,
        (
            int(round(width * scale)),
            int(round(height * scale)),
        ),
        interpolation=cv2.INTER_AREA,
    )


def _reconstruct_layers(split: str, sid: str) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    manifest_path = os.path.join(OUT, split, "manifest.json")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    row = next(item for item in manifest["scenes"] if item["id"] == sid)

    background_path = next(
        path
        for path in sorted(glob.glob(os.path.join(SRC, "*", "gt.png")))
        if os.path.basename(os.path.dirname(path)) == row["background"]
    )
    background = _resize_source(background_path)
    source = next(
        item
        for item in _source_assets()
        if os.path.basename(os.path.dirname(item[0])) == row["source"]
        and int(item[1]) == int(row["source_mask_index"])
    )
    foreground_crop, alpha_crop = _prepare_object(
        source[2],
        source[3],
        float(row["scale"]),
    )
    height, width = background.shape[:2]
    foreground = np.zeros((height, width, 3), np.float32)
    alpha = np.zeros((height, width), np.float32)
    px, py = (int(value) for value in row["placement_xy"])
    object_height, object_width = alpha_crop.shape
    alpha[py:py + object_height, px:px + object_width] = alpha_crop
    foreground[
        py:py + object_height,
        px:px + object_width,
    ] = foreground_crop
    return row, background, foreground, alpha


def _write_gray(path: str, image: np.ndarray) -> None:
    cv2.imwrite(
        path,
        np.round(np.clip(image, 0.0, 1.0) * 255).astype(np.uint8),
    )


def extension_007_primary() -> None:
    """Make the reported slender column retain an opaque interior."""
    split = "extension"
    sid = "extension_007"
    variant_sid = "extension_007_opaque_primary_r12"
    target_defocus_radius = 12.0
    row, background, foreground, alpha = _reconstruct_layers(split, sid)

    original = render_layered_focal_pair(
        background,
        foreground,
        alpha,
        float(row["max_radius"]),
        material_opacity=1.0,
    )
    original_dir = os.path.join(OUT, split, sid)
    for index, clean in enumerate(original["frames"]):
        stored = cv2.imread(
            os.path.join(original_dir, f"frame_{index}_clean.png")
        )
        rebuilt = np.clip(clean, 0, 255).astype(np.uint8)
        if not np.array_equal(rebuilt, stored):
            delta = int(
                np.max(
                    np.abs(
                        rebuilt.astype(np.int16)
                        - stored.astype(np.int16)
                    )
                )
            )
            raise RuntimeError(
                f"source reconstruction drifted for frame {index}: {delta}"
            )

    rendered = render_layered_focal_pair(
        background,
        foreground,
        alpha,
        target_defocus_radius / DEFOCUS_DISTANCE,
        material_opacity=1.0,
    )
    output_dir = os.path.join(AUDIT_OUT, variant_sid)
    os.makedirs(output_dir, exist_ok=True)
    scene_index = int(sid.rsplit("_", 1)[1])
    seed = 12001
    frames = [
        add_noise(
            np.clip(frame, 0, 255).astype(np.uint8),
            3.0,
            seed + 10 * scene_index + index,
        )
        for index, frame in enumerate(rendered["frames"])
    ]
    cv2.imwrite(
        os.path.join(output_dir, "gt.png"),
        np.clip(rendered["gt"], 0, 255).astype(np.uint8),
    )
    _write_gray(os.path.join(output_dir, "alpha.png"), alpha)
    for index, (frame, clean, coverage, extinction) in enumerate(
        zip(
            frames,
            rendered["frames"],
            rendered["geometry_coverage"],
            rendered["extinction"],
        )
    ):
        cv2.imwrite(os.path.join(output_dir, f"frame_{index}.png"), frame)
        cv2.imwrite(
            os.path.join(output_dir, f"frame_{index}_clean.png"),
            np.clip(clean, 0, 255).astype(np.uint8),
        )
        _write_gray(
            os.path.join(output_dir, f"coverage_{index}.png"),
            coverage,
        )
        _write_gray(
            os.path.join(output_dir, f"extinction_{index}.png"),
            extinction,
        )
    classes = coverage_classification(
        alpha,
        rendered["geometry_coverage"][1],
    )
    cv2.imwrite(os.path.join(output_dir, "coverage_classes.png"), classes)

    old_frames = [
        cv2.imread(os.path.join(original_dir, f"frame_{index}.png"))
        for index in (0, 1)
    ]
    old_classes = cv2.imread(
        os.path.join(original_dir, "coverage_classes.png")
    )
    gt = np.clip(rendered["gt"], 0, 255).astype(np.uint8)
    comparison = np.vstack(
        [
            np.hstack([old_frames[0], old_frames[1], frames[0], frames[1]]),
            np.hstack([gt, old_classes, gt, classes]),
        ]
    )
    cv2.imwrite(os.path.join(output_dir, "comparison.png"), comparison)

    stats = coverage_stats(alpha, rendered["geometry_coverage"][1])
    x, y = 808, 347
    report = {
        "id": variant_sid,
        "source_scene": sid,
        "material_model": "opaque_occluder",
        "material_opacity": 1.0,
        "old_defocus_radius": float(row["defocus_radius"]),
        "new_defocus_radius": target_defocus_radius,
        "diagnostic_xy": [x, y],
        "diagnostic_alpha": float(alpha[y, x]),
        "old_diagnostic_coverage": float(
            original["geometry_coverage"][1][y, x]
        ),
        "new_diagnostic_coverage": float(
            rendered["geometry_coverage"][1][y, x]
        ),
        **stats,
        "comparison_order": [
            "old near input",
            "old far input",
            "new near input",
            "new far input",
            "GT",
            "old coverage classes",
            "GT",
            "new coverage classes",
        ],
    }
    for report_path in (
        os.path.join(output_dir, "report.json"),
        PRIMARY_REPORT,
    ):
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {output_dir}")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] != "extension_007_primary":
        raise SystemExit(
            "usage: objocc_v2_variants.py extension_007_primary"
        )
    extension_007_primary()


if __name__ == "__main__":
    main()
