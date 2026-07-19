#!/usr/bin/env python3
"""Generate a synthetic multi-focus image stack for testing/demoing.

We render one sharp, detail-rich "ground truth" scene, then produce N frames.
Frame i is sharp in vertical band i and progressively blurred in bands farther
away — mimicking a scene where different depths (here, different columns) come
into focus in different shots. Focus stacking these frames should recover
something close to the ground truth.

Usage:
    python scripts/make_synthetic_stack.py --out examples/synth -n 4
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np


def make_base(height: int, width: int, seed: int = 0) -> np.ndarray:
    """A high-frequency scene so focus differences are visible everywhere."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (3, 3), 0)  # tame single-pixel noise a touch

    # A few structured marks give recognizable, orientation-rich detail.
    for _ in range(12):
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        pt1 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        pt2 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        cv2.line(base, pt1, pt2, color, 2)
    cv2.putText(
        base, "FOCUS STACK", (width // 8, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX, width / 300.0, (255, 255, 255), 3, cv2.LINE_AA,
    )
    return base


def make_stack(base: np.ndarray, n: int, max_sigma: float = 6.0) -> list[np.ndarray]:
    """Build N frames, each sharp in one vertical band, blurred elsewhere."""
    height, width = base.shape[:2]
    edges = np.linspace(0, width, n + 1).astype(int)
    frames: list[np.ndarray] = []
    for i in range(n):
        frame = np.empty_like(base)
        for j in range(n):
            x0, x1 = edges[j], edges[j + 1]
            sigma = abs(i - j) / max(1, n - 1) * max_sigma
            segment = base[:, x0:x1]
            if sigma >= 0.3:
                k = int(2 * round(sigma * 2) + 1)  # odd kernel scaled to sigma
                segment = cv2.GaussianBlur(segment, (k, k), sigma)
            frame[:, x0:x1] = segment
        frames.append(frame)
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a synthetic multi-focus stack.")
    ap.add_argument("--out", default="examples/synth", help="Output directory.")
    ap.add_argument("-n", "--num", type=int, default=4, help="Number of frames.")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    base = make_base(args.height, args.width, args.seed)
    # Underscore prefix keeps it out of a `frame_*.png` glob.
    cv2.imwrite(os.path.join(args.out, "_ground_truth.png"), base)

    frames = make_stack(base, args.num)
    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(args.out, f"frame_{i:02d}.png"), frame)

    print(f"wrote {len(frames)} frames + ground truth to {args.out}/")


if __name__ == "__main__":
    main()
