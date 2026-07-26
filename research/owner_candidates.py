"""Focused-owner proposal extraction used by the current V2 evaluator.

This is the surviving inference portion of the retired T2 training script.
Keeping it here avoids making the S29 path import old training, gate, and
veil-correction experiments.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from focusstack.focus import content_aware_energies  # noqa: E402
from focusstack.fusion import guided_filter  # noqa: E402
from focusstack.io import to_gray_float  # noqa: E402


def candidates_with_features(scene: dict, topk: int = 4) -> list[dict]:
    """Return locally supported focused-owner silhouettes and audit features."""
    pass1 = os.path.join(scene["dir"], "pass1.png")
    masks = np.load(pass1 + ".masks.npy")
    depth = np.load(pass1 + ".depth.npy")
    frames = scene["frames"]
    height, width = frames[0].shape[:2]
    if depth.shape != (height, width):
        depth = cv2.resize(depth, (width, height))
    depth = (depth - depth.min()) / (np.ptp(depth) + 1e-9)

    grays = [to_gray_float(frame) for frame in frames]
    energies = np.stack(content_aware_energies(grays), axis=0)
    winner = np.argmax(energies, axis=0)
    sorted_energy = np.sort(energies, axis=0)
    decisive = (
        (sorted_energy[-1] - sorted_energy[-2])
        / (sorted_energy[-1] + 1e-6)
        > 0.3
    ) & (sorted_energy[-1] > np.median(sorted_energy[-1]))

    proposals: dict[int, dict] = {}
    for mask_index, mask in enumerate(masks):
        support = mask > 0
        area = float(support.sum())
        if area < 400:
            continue
        depth_inside = float(np.median(depth[support]))
        depth_outside = float(np.median(depth[~support]))
        if depth_inside <= depth_outside:
            continue
        decisive_inside = decisive & support
        inside_total = float(decisive_inside.sum())
        if inside_total < 50:
            continue
        ring = (
            cv2.dilate(
                support.astype(np.uint8),
                np.ones((25, 25), np.uint8),
            )
            > 0
        ) & ~support
        decisive_ring = decisive & ring
        ring_total = float(decisive_ring.sum())
        for frame_index in range(len(frames)):
            owner_hits = float(
                (decisive_inside & (winner == frame_index)).sum()
            )
            purity = owner_hits / inside_total
            area_fit = owner_hits / area
            ring_other = (
                float(
                    (decisive_ring & (winner != frame_index)).sum()
                )
                / ring_total
                if ring_total > 50
                else 0.0
            )
            if ring_other < 0.5:
                continue
            score = (
                purity
                * np.sqrt(min(1.0, area_fit * 4))
                * ring_other
            )
            if (
                mask_index not in proposals
                or score > proposals[mask_index]["score"]
            ):
                proposals[mask_index] = {
                    "score": score,
                    "purity": purity,
                    "ring_other": ring_other,
                    "area_fit": min(1.0, area_fit * 4),
                    "margin": depth_inside - depth_outside,
                    "area_fraction": area / (height * width),
                    "mask_index": mask_index,
                }

    output = []
    for proposal in sorted(
        proposals.values(),
        key=lambda item: -item["score"],
    )[:topk]:
        support = masks[proposal["mask_index"]] > 0
        interior = (
            cv2.erode(
                support.astype(np.uint8),
                np.ones((15, 15), np.uint8),
            )
            > 0
        )
        decisive_interior = decisive & interior
        owner = (
            int(
                np.bincount(
                    winner[decisive_interior].ravel(),
                    minlength=len(frames),
                ).argmax()
            )
            if decisive_interior.sum() > 20
            else 0
        )
        alpha = np.clip(
            guided_filter(
                grays[owner] / 255.0,
                support.astype(np.float32),
                2,
                1e-4,
            ),
            0.0,
            1.0,
        )
        snapped = alpha > 0.5
        intersection = float((snapped & support).sum())
        union = float((snapped | support).sum()) + 1e-6
        features = np.array(
            [
                proposal["score"],
                proposal["purity"],
                proposal["ring_other"],
                proposal["area_fit"],
                proposal["margin"],
                proposal["area_fraction"],
                intersection / union,
            ],
            np.float32,
        )
        output.append(
            {
                "feats": features,
                "alpha": alpha,
                "owner": owner,
            }
        )
    return output
