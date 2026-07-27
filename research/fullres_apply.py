"""Full-resolution delivery: estimate small, apply native (F107 work).

The alignment estimators are pixel-scaled and validated at ~800–1100 px wide
(profile lengths, support radii, shift caps, screening offsets). Running them
naively at 24 MP puts every measurement outside its validated regime — the near
object's displacement alone exceeds the 40 px match cap several times over.

The sampling-field architecture makes the fix structural rather than parametric
(PLAYBOOK §0: scale-adaptivity belongs in the structure, not in a number). A field
maps reference coordinates to source coordinates, so a field estimated at working
scale transfers exactly to native scale:

    field_native(X) = s * field_small(X / s)

Estimation happens where it is validated; the NATIVE pixels are resampled exactly
once through the scaled field, so full resolution costs no extra interpolation and
no re-validation of any estimator. Usable masks scale with nearest-neighbour, the
common-footprint crop is recomputed natively, and fusion runs on native pixels —
`fuse_perband` is structurally resolution-adaptive already.

    .venv/bin/python research/fullres_apply.py <frames...> -o out.png
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import focusstack.align as align_mod  # noqa: E402
from focusstack import io as fio  # noqa: E402
from focusstack.fusion import fuse_perband  # noqa: E402
from focusstack.io import to_gray_float  # noqa: E402

WORKING_WIDTH = 1080


def align_fullres(natives: list[np.ndarray], working_width: int = WORKING_WIDTH):
    """Native-resolution aligned frames + usable masks, estimated at working scale."""
    nh, nw = natives[0].shape[:2]
    scale = nw / float(working_width)
    small = [cv2.resize(f, (working_width, int(round(nh / scale))),
                        interpolation=cv2.INTER_AREA) for f in natives]
    h, w = small[0].shape[:2]
    n = len(small)
    ref_index = n // 2

    # Global ECC at working scale, exactly as align_stack does it.
    ref_gray = to_gray_float(small[ref_index]) / 255.0
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-6)
    coarse, coarse_valid, global_warps = [], [], []
    for i, image in enumerate(small):
        if i == ref_index:
            coarse.append(image)
            coarse_valid.append(np.ones((h, w), dtype=bool))
            global_warps.append(None)
            continue
        warp = np.eye(2, 3, dtype=np.float32)
        _, warp = cv2.findTransformECC(ref_gray, to_gray_float(image) / 255.0,
                                       warp, cv2.MOTION_AFFINE, criteria, None, 5)
        common = dict(dsize=(w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        coarse.append(cv2.warpAffine(image, warp, **common))
        coarse_valid.append(cv2.warpAffine(np.full((h, w), 255, np.uint8), warp,
                                           **common) == 255)
        global_warps.append(warp)

    fields, report = align_mod._depth_binned_fields(
        small, coarse, coarse_valid, global_warps, ref_index, 4,
    )
    occlusion = report.pop("occlusion", {})

    grid_x, grid_y = np.meshgrid(np.arange(nw, dtype=np.float32),
                                 np.arange(nh, dtype=np.float32))
    aligned, valid, usable = [], [], []
    for i in range(n):
        if i == ref_index:
            aligned.append(natives[i])
            valid.append(np.ones((nh, nw), dtype=bool))
            usable.append(np.ones((nh, nw), dtype=bool))
            continue
        if i in fields:
            map_small_x, map_small_y = fields[i]
        elif global_warps[i] is not None:
            map_small_x, map_small_y = align_mod._matrix_field(
                align_mod._homogeneous(global_warps[i]), (h, w))
        else:
            aligned.append(natives[i])
            valid.append(np.ones((nh, nw), dtype=bool))
            usable.append(np.ones((nh, nw), dtype=bool))
            continue
        # The transfer: field_native(X) = s * field_small(X / s). The resize IS
        # the evaluation of field_small at X/s; the multiply rescales its value.
        map_x = cv2.resize(map_small_x, (nw, nh), interpolation=cv2.INTER_LINEAR) * scale
        map_y = cv2.resize(map_small_y, (nw, nh), interpolation=cv2.INTER_LINEAR) * scale
        aligned.append(cv2.remap(natives[i], map_x, map_y, cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0))
        valid.append(cv2.remap(np.full((nh, nw), 255, np.uint8), map_x, map_y,
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0) == 255)
        if i in occlusion:
            usable.append(cv2.resize(occlusion[i].astype(np.uint8), (nw, nh),
                                     interpolation=cv2.INTER_NEAREST) == 0)
        else:
            usable.append(np.ones((nh, nw), dtype=bool))

    common_valid = np.logical_and.reduce(valid)
    x0, y0, x1, y1 = align_mod._largest_valid_rectangle(common_valid)
    aligned = [a[y0:y1, x0:x1] for a in aligned]
    usable = [u[y0:y1, x0:x1] for u in usable]
    return aligned, usable, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("frames", nargs="+")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    natives = [cv2.imread(p) for p in sorted(args.frames)]
    print(f"{len(natives)} frames at {natives[0].shape[1]}x{natives[0].shape[0]}")
    aligned, usable, report = align_fullres(natives)
    print("estimation:", report.get("motion_groups"))
    fused = fuse_perband(fio.normalize_exposure(aligned), usable=usable)
    cv2.imwrite(args.output, fused)
    print(f"wrote {args.output} ({fused.shape[1]}x{fused.shape[0]})")


if __name__ == "__main__":
    main()
