#!/usr/bin/env python3
"""Eye-analysis 2.0 — disagreement-guided visual inspection.

Instead of hand-picking crop locations (which biases what you see), this finds
the regions where two method outputs DISAGREE most, and renders, for each:
side-by-side crops of [GT if given | A | B | amplified diff(A,B)] — so the eye is
pointed at exactly the pixels where the methods differ, with the difference
amplified beyond what unaided viewing can perceive.

Usage (as a library):
    from eyetool import compare
    compare({"blend": bl, "perband": pb}, gt=gt, out="research/analyze_out/x.png")
"""
from __future__ import annotations

import cv2
import numpy as np


def _disagreement(a: np.ndarray, b: np.ndarray, win: int = 61) -> np.ndarray:
    d = np.abs(a.astype(np.float32) - b.astype(np.float32)).sum(axis=2)
    return cv2.boxFilter(d, cv2.CV_32F, (win, win))


def _top_regions(heat: np.ndarray, k: int, half: int) -> list[tuple[int, int]]:
    """Greedy non-overlapping maxima of the heatmap."""
    h, w = heat.shape
    heat = heat.copy()
    out = []
    for _ in range(k):
        y, x = np.unravel_index(int(heat.argmax()), heat.shape)
        if heat[y, x] <= 0:
            break
        out.append((int(y), int(x)))
        y0, y1 = max(0, y - 2 * half), min(h, y + 2 * half)
        x0, x1 = max(0, x - 2 * half), min(w, x + 2 * half)
        heat[y0:y1, x0:x1] = 0                      # suppress neighborhood
    return out


def _amplify_diff(a: np.ndarray, b: np.ndarray, gain: float = 5.0) -> np.ndarray:
    """Mid-grey plus amplified signed difference — makes subtle changes visible."""
    d = (a.astype(np.float32) - b.astype(np.float32)) * gain + 127.0
    return np.clip(d, 0, 255).astype(np.uint8)


def compare(methods: dict[str, np.ndarray], gt: np.ndarray | None = None,
            out: str = "compare.png", k: int = 3, half: int = 120, zoom: float = 2.5):
    """Render the k most-disagreeing regions across all method pairs.

    Each row: [GT? | each method | amplified diff of the two most-different methods].
    Returns the list of (y, x) region centers used.
    """
    names = list(methods)
    # heat = max pairwise disagreement
    heat = None
    worst_pair = (names[0], names[-1])
    worst_val = -1.0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            hij = _disagreement(methods[names[i]], methods[names[j]])
            if heat is None:
                heat = hij.copy()
            else:
                heat = np.maximum(heat, hij)
            v = float(hij.max())
            if v > worst_val:
                worst_val, worst_pair = v, (names[i], names[j])

    centers = _top_regions(heat, k, half)
    h, w = heat.shape

    rows = []
    for (y, x) in centers:
        y0, x0 = np.clip(y - half, 0, h - 2 * half), np.clip(x - half, 0, w - 2 * half)
        sl = (slice(y0, y0 + 2 * half), slice(x0, x0 + 2 * half))
        cells = []
        if gt is not None:
            cells.append(gt[sl])
        cells += [methods[n][sl] for n in names]
        cells.append(_amplify_diff(methods[worst_pair[0]][sl], methods[worst_pair[1]][sl]))
        z = [cv2.resize(c, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST) for c in cells]
        rows.append(np.hstack(z))
    grid = np.vstack(rows)
    cv2.imwrite(out, grid)
    header = (["GT"] if gt is not None else []) + names + [f"diff({worst_pair[0]},{worst_pair[1]})x5"]
    print(f"wrote {out}  columns: {' | '.join(header)}  rows: top-{len(centers)} disagreement regions {centers}")
    return centers


if __name__ == "__main__":
    print(__doc__)
