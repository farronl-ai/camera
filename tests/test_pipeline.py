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


def test_normalize_exposure_identity_and_recovery():
    from focusstack.io import normalize_exposure

    rng = np.random.default_rng(5)
    base = rng.integers(0, 200, (64, 64, 3)).astype(np.uint8)  # headroom, no clipping
    stack = [base.copy(), cv2.GaussianBlur(base, (9, 9), 3)]

    # near-identity on an undrifted stack (defocus preserves the mean)
    normed = normalize_exposure(stack)
    assert np.abs(np.stack(normed).astype(np.int16) - np.stack(stack).astype(np.int16)).mean() < 1.0

    # recovers a gain-drifted frame toward the stack median
    drifted = [stack[0], np.clip(stack[1].astype(np.float32) * 1.15, 0, 255).astype(np.uint8)]
    fixed = normalize_exposure(drifted)
    err_before = abs(float(drifted[1].mean()) - float(drifted[0].mean()))
    err_after = abs(float(fixed[1].mean()) - float(fixed[0].mean()))
    assert err_after < err_before * 0.35


def test_enhance_auto_identity_when_silent(tmp_path, monkeypatch):
    # no bridge (env points nowhere) + noise stack (gates silent) => byte-identical
    monkeypatch.setenv("FOCUSSTACK_BRIDGE_PYTHON", "/nonexistent/python")
    rng = np.random.default_rng(3)
    base = rng.integers(0, 255, (128, 128, 3)).astype(np.uint8)
    stack = [base.copy(), cv2.GaussianBlur(base, (9, 9), 3)]
    paths = []
    for i, im in enumerate(stack):
        pth = str(tmp_path / f"e{i}.png")
        cv2.imwrite(pth, im)
        paths.append(pth)
    a = run(paths, str(tmp_path / "on.png"), align=False, enhance="auto")
    b = run(paths, str(tmp_path / "off.png"), align=False, enhance="off")
    assert np.array_equal(a, b)


def test_enhance_auto_does_not_call_safety_disabled_veil_bridge(monkeypatch):
    import focusstack.enhance as enhance_module

    def unsafe_bridge_call(*_args, **_kwargs):
        raise AssertionError("safety-disabled veil branch called its bridge")

    monkeypatch.setattr(enhance_module, "run_bridge", unsafe_bridge_call)
    rng = np.random.default_rng(14)
    image = rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)
    stack = [image, cv2.GaussianBlur(image, (9, 9), 3)]
    output, report = enhance_module.enhance(stack, image.copy())

    assert output.shape == image.shape
    assert report["veil_fired"] == 0
    assert report["veil_disabled_safety"] is True
    assert report["bridge"] is False


def test_gates_predict_gain_shapes():
    from focusstack.gates import RECON_GATE, VEIL_GATE, predict_gain

    x = np.linspace(0.1, 0.9, 7).astype(np.float32)
    for gate in (VEIL_GATE, RECON_GATE):
        v = predict_gain(gate, x)
        assert isinstance(v, float) and np.isfinite(v)
        assert 0 < gate["margin"] < 0.1


def test_bridge_runner_graceful_absence():
    from focusstack.bridge import find_bridge_python, run_bridge

    # explicit-but-missing python resolves to None, and run degrades to None
    assert find_bridge_python("/nonexistent/python") is None
    assert run_bridge("depth", "/nonexistent/img.png", python="/nonexistent/python") is None
    # unknown bridge kind is also a graceful None (never raises)
    assert run_bridge("nope", "/nonexistent/img.png", python="/nonexistent/python") is None


def test_reconstruct_boundaries_safe_and_off_by_default(tmp_path):
    from focusstack.reconstruct import reconstruct_boundaries

    rng = np.random.default_rng(9)
    base = rng.integers(0, 255, (96, 96, 3)).astype(np.uint8)
    stack = [base.copy(), cv2.GaussianBlur(base, (9, 9), 3)]
    fused = base.copy()

    # graceful no-op on degenerate inputs (single frame; no occluder found)
    assert np.array_equal(reconstruct_boundaries([base], fused), fused)
    out = reconstruct_boundaries(stack, fused)
    assert out.shape == fused.shape and out.dtype == np.uint8

    # pipeline default OFF -> byte-identical to a run without the stage
    paths = []
    for i, im in enumerate(stack):
        p = str(tmp_path / f"f{i}.png")
        cv2.imwrite(p, im)
        paths.append(p)
    a = run(paths, str(tmp_path / "a.png"), align=False, verbose=False)
    b = run(paths, str(tmp_path / "b.png"), align=False, verbose=False,
            reconstruct_boundaries=False)
    assert np.array_equal(a, b)


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
