import cv2
import numpy as np

from focusstack.align import align_stack
from focusstack.motion_groups import overrides


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


def _square_stack(step_px, frames=5, h=360, w=460):
    """A textured square translating over a static textured background.

    This is the override's whole reason to exist in miniature: the depth path
    (given a flat depth map) applies nothing, while one compact object moves by
    `step_px` per frame. Everything is sharp, so only geometry matters.
    """
    background = _textured(3, h, w)
    square = _textured(7, 150, 150)
    ref = frames // 2
    stack = []
    for k in range(frames):
        frame = background.copy()
        x0 = 150 + int(round(step_px * (k - ref)))
        frame[100:250, x0:x0 + 150] = square
        stack.append(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2BGR))
    return stack, ref


def _run_overrides(stack, ref):
    grays = [s for s in stack]
    h, w = stack[0].shape[:2]
    valid = [np.ones((h, w), bool) for _ in stack]
    depth = np.zeros((h, w), np.float32)   # flat: no depth steps, all edges material
    return overrides(stack, stack, valid, ref, depth, lambda k, x, y: (0.0, 0.0))


def test_override_fires_on_an_object_the_depth_path_missed():
    stack, ref = _square_stack(step_px=4.0)
    chosen, report = _run_overrides(stack, ref)
    assert report["overridden"] >= 1, report

    # The chosen group must measure the square's true motion: +8 px at the far
    # frame (4 px/frame, two frames from the reference), recovered within 1.5 px.
    best = max(chosen, key=lambda c: c[0][175, 225])
    weight, motion = best
    far = motion[len(stack) - 1]
    assert abs(far[0] - 8.0) < 1.5 and abs(far[1]) < 1.5, far

    # And its correction must land on the square, not on the background.
    assert weight[175, 225] > 0.8          # square centre: owned in full
    assert weight[40, 40] < 0.1            # far background corner: untouched


def test_override_screens_out_a_scene_with_no_unexplained_motion():
    stack, ref = _square_stack(step_px=0.0)
    chosen, report = _run_overrides(stack, ref)
    assert chosen == []
    assert report.get("skipped"), report   # refused early, before the full pass


def test_align_stack_is_byte_identical_where_nothing_disagrees():
    """Non-regression by construction: agreeing scenes must not change at all."""
    rng = np.random.default_rng(11)
    frames = []
    base = _textured(5, 240, 320)
    for k in range(4):
        noisy = np.clip(base + rng.normal(0, 2, base.shape), 0, 255).astype(np.uint8)
        frames.append(cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR))
    on = align_stack(frames, motion_override=True)
    off = align_stack(frames, motion_override=False)
    for a, b in zip(on, off):
        assert np.array_equal(a, b)
