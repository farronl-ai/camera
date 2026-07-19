import os

import cv2
import numpy as np

from focusstack.focus import focus_measure
from focusstack.pipeline import run


def _write_two_region_stack(directory):
    rng = np.random.default_rng(2)
    base = rng.integers(0, 256, (160, 160, 3)).astype(np.uint8)
    blur = lambda im: cv2.GaussianBlur(im, (15, 15), 6)

    a = base.copy()
    a[:, 80:] = blur(base)[:, 80:]
    b = base.copy()
    b[:, :80] = blur(base)[:, :80]

    pa = os.path.join(directory, "a.png")
    pb = os.path.join(directory, "b.png")
    cv2.imwrite(pa, a)
    cv2.imwrite(pb, b)
    return base, a, b, [pa, pb]


def _sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return focus_measure(gray).mean()


def test_pipeline_end_to_end(tmp_path):
    d = str(tmp_path)
    base, a, b, paths = _write_two_region_stack(d)
    out = os.path.join(d, "out.png")

    # align=False: the synthetic frames share geometry, and ECC on pure noise is
    # unnecessary here — we're exercising focus + fusion end-to-end.
    fused = run(paths, out, method="pyramid", align=False)

    assert os.path.exists(out)
    assert fused.shape == base.shape
    assert _sharpness(fused) > _sharpness(a)
    assert _sharpness(fused) > _sharpness(b)


def test_pipeline_max_method_with_debug(tmp_path):
    d = str(tmp_path)
    _, _, _, paths = _write_two_region_stack(d)
    out = os.path.join(d, "out.png")
    debug = os.path.join(d, "debug")

    run(paths, out, method="max", align=False, debug_dir=debug)

    assert os.path.exists(out)
    assert os.path.exists(os.path.join(debug, "selection.png"))
    # A focus map should be dumped per input frame.
    assert os.path.exists(os.path.join(debug, "focus_a.png"))
    assert os.path.exists(os.path.join(debug, "focus_b.png"))
