import cv2
import numpy as np

from focusstack.align import align_stack


def _center_mad(a, b, margin=40):
    """Mean absolute difference over a central crop (ignores border-fill artifacts)."""
    a = a[margin:-margin, margin:-margin].astype(np.float32)
    b = b[margin:-margin, margin:-margin].astype(np.float32)
    return float(np.abs(a - b).mean())


def _textured_scene(seed=0, h=480, w=640):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    cv2.putText(img, "ALIGN", (w // 6, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 4, cv2.LINE_AA)
    return img


def test_align_recovers_known_translation_and_scale():
    """align_stack should pull a deliberately warped frame back onto the reference."""
    ref = _textured_scene()
    h, w = ref.shape[:2]

    # Known misalignment: +8/+5 px shift and 1.5% scale-up (mimics focus breathing).
    warp = np.array([[1.015, 0.0, 8.0],
                     [0.0, 1.015, 5.0]], dtype=np.float32)
    moved = cv2.warpAffine(ref, warp, (w, h), borderMode=cv2.BORDER_REFLECT)

    aligned = align_stack(
        [ref, moved],
        ref_index=0,
        motion="affine",
        crop_valid=False,
    )
    recovered = aligned[1]

    before = _center_mad(ref, moved)
    after = _center_mad(ref, recovered)

    # Alignment should remove most of the registration error.
    assert after < before / 3.0


def test_reference_frame_is_returned_unchanged():
    ref = _textured_scene(seed=1)
    other = _textured_scene(seed=2)
    aligned = align_stack(
        [ref, other],
        ref_index=0,
        motion="affine",
        crop_valid=False,
    )
    # Without common-footprint cropping, the fixed reference passes through.
    assert np.array_equal(aligned[0], ref)


def _parallax_stack(near_shift=3.0, far_shift=0.6, frames=5, h=300, w=400):
    """A stack whose near and far content move by DIFFERENT amounts.

    This is the case a single global warp cannot express, so it is the only
    input that actually tests the depth-aware pass; frames that differ by one
    global transform would look perfectly aligned either way.
    """
    pad = 40
    background = _textured_scene(seed=11, h=h + 2 * pad, w=w + 2 * pad)
    foreground = _textured_scene(seed=12, h=h + 2 * pad, w=w + 2 * pad)
    alpha = np.zeros((h + 2 * pad, w + 2 * pad), np.float32)
    cv2.rectangle(alpha, (pad + 40, pad + 50), (pad + 210, pad + 240), 1.0, -1)

    def shift(layer, amount):
        matrix = np.float32([[1, 0, amount], [0, 1, 0]])
        return cv2.warpAffine(layer, matrix, (w + 2 * pad, h + 2 * pad),
                              borderMode=cv2.BORDER_REFLECT)

    def defocus(layer, radius):
        """Disk blur: the depth cue the alignment pass reads depth out of."""
        if radius < 1:
            return layer
        size = 2 * int(radius) + 1
        yy, xx = np.mgrid[-int(radius):int(radius) + 1, -int(radius):int(radius) + 1]
        kernel = ((xx ** 2 + yy ** 2) <= radius * radius).astype(np.float32)
        return cv2.filter2D(layer, -1, kernel / kernel.sum(),
                            borderType=cv2.BORDER_REPLICATE)

    reference = frames // 2
    near_focus, far_focus = 0, frames - 1
    stack = []
    for k in range(frames):
        step = k - reference
        near_alpha = defocus(
            shift(alpha, step * near_shift), abs(k - near_focus) * 1.4
        ).reshape(alpha.shape)[..., None]
        composited = (
            defocus(shift(foreground, step * near_shift).astype(np.float32),
                    abs(k - near_focus) * 1.4)
            * near_alpha
            + defocus(shift(background, step * far_shift).astype(np.float32),
                      abs(k - far_focus) * 1.4)
            * (1.0 - near_alpha)
        )
        stack.append(
            np.clip(composited[pad:pad + h, pad:pad + w], 0, 255).astype(np.uint8)
        )
    near_mask = (shift(alpha, 0.0)[pad:pad + h, pad:pad + w] > 0.5)
    return stack, near_mask, reference


def _plane_residual(reference, moved, mask):
    """Leftover translation inside one depth plane, measured independently."""
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    warp = np.eye(2, 3, dtype=np.float32)
    grey = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            for i in (reference, moved)]
    _, warp = cv2.findTransformECC(grey[0], grey[1], warp, cv2.MOTION_TRANSLATION,
                                   criteria, mask.astype(np.uint8) * 255, 5)
    return float(np.hypot(warp[0, 2], warp[1, 2]))


def test_depth_aware_pass_removes_depth_dependent_misregistration():
    """The near plane must stop moving relative to the reference, not just the far one."""
    stack, near_mask, reference = _parallax_stack()

    global_only = align_stack(stack, depth_bins=0, crop_valid=False)
    depth_aware = align_stack(stack, depth_bins=3, crop_valid=False)

    def worst_near_residual(aligned):
        return max(
            _plane_residual(aligned[reference], frame, near_mask)
            for index, frame in enumerate(aligned)
            if index != reference
        )

    before = worst_near_residual(global_only)
    after = worst_near_residual(depth_aware)
    assert after < before / 2.0, f"near-plane residual {before:.2f} -> {after:.2f} px"


def test_depth_aware_pass_leaves_an_unmoving_stack_alone():
    """No parallax to correct means no correction: refusal must be the default."""
    frames = [_textured_scene(seed=21, h=200, w=260) for _ in range(4)]
    aligned, report = align_stack(
        frames, depth_bins=3, crop_valid=False, return_report=True
    )
    for original, result in zip(frames, aligned):
        assert np.array_equal(original, result)
    assert all(entry["accepted"] == 0 for entry in report["frames"].values())


def test_depth_aware_pass_is_skipped_for_two_frame_stacks():
    """Two frames carry no depth-from-focus proxy, so the pass must not engage."""
    ref = _textured_scene(seed=31, h=200, w=260)
    warp = np.array([[1.0, 0.0, 4.0], [0.0, 1.0, 3.0]], dtype=np.float32)
    moved = cv2.warpAffine(ref, warp, (260, 200), borderMode=cv2.BORDER_REFLECT)

    with_pass = align_stack([ref, moved], depth_bins=3, crop_valid=False)
    without_pass = align_stack([ref, moved], depth_bins=0, crop_valid=False)
    for a, b in zip(with_pass, without_pass):
        assert np.array_equal(a, b)


def test_depth_aware_field_does_not_distort_content():
    """A sampling field may transport content, but never stretch it."""
    stack, _, _ = _parallax_stack()
    _, report = align_stack(stack, depth_bins=3, crop_valid=False, return_report=True)
    corrected = [f for f in report["frames"].values() if f["accepted"] > 0]
    assert corrected, "expected the parallax stack to earn a correction"
    for frame in corrected:
        assert frame["stretch"] < 1.0


def test_alignment_crops_to_pixels_observed_by_every_frame(monkeypatch):
    ref = _textured_scene(seed=3, h=120, w=160)
    other = ref.copy()
    warp = np.array(
        [[1.0, 0.0, 14.0], [0.0, 1.0, -9.0]],
        dtype=np.float32,
    )

    monkeypatch.setattr(
        cv2,
        "findTransformECC",
        lambda *_args, **_kwargs: (1.0, warp.copy()),
    )
    aligned = align_stack([ref, other], ref_index=0, motion="affine")

    # WARP_INVERSE_MAP samples source x+14, y-9, leaving a 14 px loss on the
    # right and 9 px loss at the top. Neither synthetic region may survive.
    assert aligned[0].shape == aligned[1].shape == (111, 146, 3)
    assert np.array_equal(aligned[0], ref[9:, :146])
