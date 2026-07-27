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
