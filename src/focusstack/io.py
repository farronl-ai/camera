"""Image input/output: expand inputs, load a stack, save results.

OpenCV loads images as HxWx3 arrays in **BGR** channel order (not RGB) and
dtype uint8. We keep that convention throughout the package and only convert to
float / grayscale where the math needs it.
"""

from __future__ import annotations

import glob
import os

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def _expand_inputs(inputs: list[str]) -> list[str]:
    """Turn a list of files/dirs/globs into a sorted, de-duplicated file list."""
    paths: list[str] = []
    for item in inputs:
        if os.path.isdir(item):
            for name in os.listdir(item):
                p = os.path.join(item, name)
                if os.path.isfile(p) and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                    paths.append(p)
        else:
            # glob.glob returns [item] for a plain existing path, or expands wildcards.
            paths.extend(glob.glob(item))

    # De-duplicate by absolute path, preserving nothing but final sort order.
    seen: set[str] = set()
    unique: list[str] = []
    for p in paths:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            unique.append(p)
    return sorted(unique)


def load_images(inputs: list[str]) -> list[tuple[str, np.ndarray]]:
    """Load an image stack.

    Returns a list of (basename, BGR uint8 array), sorted by path so the frame
    order is deterministic. All frames must share the same dimensions — that is
    what makes them a stack of the *same* scene.
    """
    paths = _expand_inputs(inputs)
    if not paths:
        raise FileNotFoundError(f"No image files matched: {inputs}")

    images: list[tuple[str, np.ndarray]] = []
    shape: tuple[int, ...] | None = None
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image: {p}")
        if shape is None:
            shape = img.shape
        elif img.shape != shape:
            raise ValueError(
                f"Image size mismatch: {p} is {img.shape}, expected {shape}. "
                "All frames in a focus stack must have identical dimensions."
            )
        images.append((os.path.basename(p), img))

    if len(images) < 2:
        raise ValueError("Focus stacking needs at least 2 images.")
    return images


def save_image(path: str, img: np.ndarray) -> None:
    """Write an image, creating parent directories as needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if not cv2.imwrite(path, img):
        raise IOError(f"Failed to write image: {path}")


def to_gray_float(img: np.ndarray) -> np.ndarray:
    """Convert a BGR uint8 image to a single-channel float32 luminance image.

    Focus/alignment math operates on brightness, not color, so we collapse the
    three channels to one and use float to avoid clipping in the derivatives.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32)
