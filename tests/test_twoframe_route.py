"""The two-frame route: when it engages, when it declines, and the ported gate.

The route is F101's doctrine applied a second time — two architectures that win
on opposite scenes are ROUTED, never merged — so the tests that matter are about
the DECISION and about non-regression, not about fusion quality:

  * a scene with nothing stranded must be byte-identical with the route on or off;
  * a scene with a stranded object the two-frame path can place must engage;
  * a scene whose stranded object needs re-registration must decline (large-motion
    is the real case; here it is reproduced synthetically by moving the object
    further than the architecture's displacement licence);
  * and the ported validity gate must still catch a wrong layer fit, because a
    port is a new instrument (DEVSTYLE §12.1) — `global_stage` is checked against
    the shipped global stage the same way.
"""

import cv2
import numpy as np

from focusstack.align import align_stack
from focusstack.pipeline import run
from focusstack.twoframe import (GATE_TOL, EdgeEvidence, gate_shift, global_stage,
                                 twoframe_stack)


def _textured(seed, h, w):
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 90, np.float32)
    for _ in range(160):
        x0, y0 = rng.integers(0, w - 30), rng.integers(0, h - 14)
        shade = float(rng.integers(0, 255))
        cv2.rectangle(img, (int(x0), int(y0)),
                      (int(x0 + rng.integers(6, 28)), int(y0 + rng.integers(3, 12))),
                      shade, -1)
    return img


def _disk_blur(image, radius):
    """Real defocus is a disk, not a Gaussian (PLAYBOOK §0)."""
    r = int(round(radius))
    if r < 1:
        return image
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    kernel = (xx * xx + yy * yy <= r * r).astype(np.float32)
    return cv2.filter2D(image, -1, kernel / kernel.sum())


def _stranded_object_stack(step_px, frames=5, h=360, w=460, square_focus=None,
                           background_focus=0):
    """A textured square translating over a static background, with a focal sweep.

    The moving-square fixture of `test_motion_groups`, given the one thing a
    focus stack must have: each layer is sharp in its own frame. `square_focus`
    defaults to the reference, which is the kitchen's regime — the stranded
    object is sharpest where it needs no correction, and that is exactly when the
    two-frame path can serve it.
    """
    ref = frames // 2
    if square_focus is None:
        square_focus = ref
    background = _textured(3, h, w)
    square = _textured(7, 150, 150)
    stack = []
    for k in range(frames):
        sharp = background.copy()
        x0 = 150 + int(round(step_px * (k - ref)))
        sharp[100:250, x0:x0 + 150] = square
        mask = np.zeros((h, w), bool)
        mask[100:250, x0:x0 + 150] = True
        near = _disk_blur(sharp, 1.6 * abs(k - square_focus))
        far = _disk_blur(sharp, 1.6 * abs(k - background_focus))
        frame = np.where(mask, near, far)
        stack.append(cv2.cvtColor(np.clip(frame, 0, 255).astype(np.uint8),
                                  cv2.COLOR_GRAY2BGR))
    return stack, ref


def _still_stack(frames=4, h=240, w=320):
    """Frames that differ only by sensor noise: nothing is stranded."""
    rng = np.random.default_rng(11)
    base = _textured(5, h, w)
    return [cv2.cvtColor(np.clip(base + rng.normal(0, 2, base.shape), 0, 255)
                         .astype(np.uint8), cv2.COLOR_GRAY2BGR)
            for _ in range(frames)]


def _write(stack, directory):
    paths = []
    for i, frame in enumerate(stack):
        path = str(directory / f"f{i:02d}.png")
        cv2.imwrite(path, frame)
        paths.append(path)
    return paths


def test_route_declines_and_is_byte_identical_on_a_still_stack(tmp_path, capsys):
    """Non-regression by construction: nothing stranded, nothing changed."""
    paths = _write(_still_stack(), tmp_path)
    routed = run(paths, str(tmp_path / "on.png"), twoframe_route=True, verbose=True)
    log = capsys.readouterr().out
    plain = run(paths, str(tmp_path / "off.png"), twoframe_route=False)

    assert "fusion path: shipped depth-bin" in log
    assert "TWO-FRAME" not in log
    assert np.array_equal(routed, plain)


def test_route_engages_on_a_stranded_object_the_two_frame_path_can_place(tmp_path,
                                                                         capsys):
    """The kitchen's regime in miniature: an object the depth path cannot see,
    sharpest where it needs no correction."""
    stack, _ref = _stranded_object_stack(step_px=6.0)
    paths = _write(stack, tmp_path)
    routed = run(paths, str(tmp_path / "on.png"), twoframe_route=True, verbose=True)
    log = capsys.readouterr().out
    plain = run(paths, str(tmp_path / "off.png"), twoframe_route=False)

    assert "fusion path: TWO-FRAME" in log
    assert not np.array_equal(routed, plain)


def test_route_declines_when_the_object_needs_re_registration(tmp_path, capsys):
    """Large-motion's regime: the stranded object is sharpest far from the
    reference, so serving it means re-registering by more than the architecture's
    displacement licence. The composite is built, measured, and discarded."""
    stack, _ref = _stranded_object_stack(step_px=8.0, square_focus=0)
    paths = _write(stack, tmp_path)
    routed = run(paths, str(tmp_path / "on.png"), twoframe_route=True, verbose=True)
    log = capsys.readouterr().out
    plain = run(paths, str(tmp_path / "off.png"), twoframe_route=False)

    assert "DECLINED" in log
    assert "fusion path: shipped depth-bin" in log
    assert np.array_equal(routed, plain)


def test_ported_global_stage_still_matches_the_shipped_one():
    """A port is a new instrument. This one claims to be a VIEW of the shipped
    global stage that also returns its matrices; identical pixels is the claim."""
    stack, ref = _stranded_object_stack(step_px=4.0)
    coarse, warps, _valid = global_stage(stack, ref)
    shipped = align_stack(stack, ref_index=ref, depth_bins=0, crop_valid=False)
    for ours, theirs in zip(coarse, shipped):
        assert np.array_equal(ours, theirs)
    assert warps[ref] is None


def test_the_validity_gate_catches_an_injected_layer_error():
    """The promotion blocker the gate exists for: a wrong-but-plausible layer
    shift is invisible to `max_shift`, and F110 measured one taking the factory
    to 0.668 with nothing objecting. Inject +8 px and require the gate to recover
    the clean composite."""
    stack, ref = _stranded_object_stack(step_px=6.0)
    clean, info = twoframe_stack(stack, ref)
    assert len(info["pairs"]) >= 1

    # Into a real two-member pair: a degenerate one-frame region has no second
    # observation, so refusing it falls back to the unwarped reference and the
    # test would be measuring the fallback rather than the gate.
    pair_index = next(i for i, d in enumerate(info["diagnostics"])
                      if len(d["frames"]) == 2)
    error = {(pair_index, 0): (8.0, 0.0)}
    ungated, ungated_info = twoframe_stack(stack, ref, inject=error, gate=False)
    gated, gated_info = twoframe_stack(stack, ref, inject=error, gate=True)

    def compare(a, a_info, b, b_info):
        """Mean |difference| on the region both outputs actually observed.

        A wrong layer shift moves the common footprint, so the crops differ; each
        output is mapped back into REFERENCE coordinates before differencing. The
        research harness lost a run to exactly this trap (F110 H4).
        """
        ax0, ay0, ax1, ay1 = a_info["crop"]
        bx0, by0, bx1, by1 = b_info["crop"]
        x0, y0 = max(ax0, bx0), max(ay0, by0)
        x1, y1 = min(ax1, bx1), min(ay1, by1)
        left = a[y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0].astype(float)
        right = b[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0].astype(float)
        return float(np.abs(left - right).mean())

    damage = compare(ungated, ungated_info, clean, info)
    residual = compare(gated, gated_info, clean, info)
    assert damage > 1.0, damage                 # the error really is damaging
    assert residual < 0.25 * damage, (residual, damage)

    # And the error must not have been applied silently: the entry it was
    # injected into is no longer carrying it.
    injected = gated_info["diagnostics"][pair_index]["shifts"][0]
    clean_shift = info["diagnostics"][pair_index]["shifts"][0]
    if injected is not None and clean_shift is not None:
        assert abs(injected[0] - clean_shift[0]) < 2.0, (injected, clean_shift)


def test_the_ported_gate_separates_a_correct_fit_from_a_wrong_one():
    """The gate as an instrument, on a known answer (F110 KAT 6 re-run here).

    `warpAffine` moves content by -t and every estimator in this module returns
    the warp taking REFERENCE coordinates to MOVING ones, so the correct answer
    for a +12 px content move is -12. A correct fit must verify; an 8 px error
    must be contradicted, and the statistic must read the error's own size,
    because that reading is what the repair step uses as its correction.
    """
    grey = _textured(21, 320, 420) / 255.0
    h, w = grey.shape
    moved = cv2.warpAffine(grey.astype(np.float32), np.float32([[1, 0, -12], [0, 1, 0]]),
                           (w, h), borderMode=cv2.BORDER_REFLECT)
    evidence = EdgeEvidence([grey.astype(np.float32), moved], 0,
                            np.zeros((h, w), np.float32), np.ones((h, w), bool),
                            np.ones((h, w), np.float32))
    mask = np.zeros((h, w), bool)
    mask[60:260, 80:340] = True
    indices = evidence.indices_in(mask)
    assert len(indices) >= 20

    status, statistic, _why = gate_shift(evidence, 1, indices, (-12.0, 0.0))
    assert status == "verified", (status, statistic)
    assert statistic < GATE_TOL

    status, statistic, _why = gate_shift(evidence, 1, indices, (-12.0 + 8.0, 0.0))
    assert status == "contradicted", (status, statistic)
    assert abs(statistic - 8.0) < 1.0, statistic
