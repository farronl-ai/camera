import cv2
import numpy as np

from focusstack.fusion import fuse_perband
from focusstack.veil_layers import (
    RADIUS_FRACTION,
    _box_disk_blur,
    _forward_layers,
    _fringe_mask,
    _ordered_visibility_gate,
    _ownership_gate,
    _prepare_model,
    candidate_is_licensed,
    complete_owner_support,
    recover_giant_veil,
    select_licensed_candidate,
    stable_correction,
)


def _physical_giant_stack(size=192, *, missing_satellite=False):
    yy, xx = np.mgrid[:size, :size]
    # Mid-scale rear structure makes the positive-observation premise explicit:
    # the ordered gate must be able to distinguish the rear-focus frame around
    # the occlusion band. A merely smooth gradient is deliberately insufficient.
    rear_texture = 55.0 * np.sin(xx / 2.7) * np.sin(yy / 3.7)
    background = np.stack(
        [
            50.0 + 0.4 * xx + rear_texture,
            90.0 + 20.0 * np.sin(xx / 5.0) - 0.6 * rear_texture,
            130.0 + 0.3 * yy + 0.8 * rear_texture,
        ],
        axis=2,
    ).astype(np.float32)
    alpha_main = (
        (xx - size / 2) ** 2 + (yy - size / 2) ** 2
        < (size * 0.22) ** 2
    ).astype(np.float32)
    alpha = alpha_main.copy()
    satellite = np.zeros_like(alpha)
    if missing_satellite:
        satellite[
            size // 2 - 5 : size // 2 + 5,
            size // 2 + 43 : size // 2 + 51,
        ] = 1.0
        alpha = np.maximum(alpha, satellite)
    foreground = np.empty_like(background)
    foreground[:] = (210.0, 60.0, 30.0)
    model = _prepare_model(
        alpha,
        RADIUS_FRACTION * size,
        _box_disk_blur,
    )
    frames = [
        np.uint8(np.clip(frame, 0, 255))
        for frame in _forward_layers(foreground, background, model)
    ]
    gt = np.uint8(
        np.clip(
            alpha[..., None] * foreground
            + (1.0 - alpha[..., None]) * background,
            0,
            255,
        )
    )
    candidate = {
        "feats": np.asarray(
            (0.8, 0.95, 0.8, 0.96, 0.2, 0.1, 0.95),
            np.float32,
        ),
        "alpha": alpha_main if missing_satellite else alpha,
        "owner": 0,
    }
    return frames, gt, candidate, satellite


def test_candidate_license_uses_semantics_and_forward_evidence():
    candidate = {
        "feats": np.asarray((0.6, 0.9, 0.7, 0.95), np.float32),
        "forward_before": 2.0,
        "forward_after": 1.0,
    }
    assert candidate_is_licensed(candidate)

    for key, value in (
        ("feats", np.asarray((0.5, 0.9, 0.7, 0.95), np.float32)),
        ("forward_after", 1.8),
    ):
        rejected = dict(candidate)
        rejected[key] = value
        assert not candidate_is_licensed(rejected)


def test_component_consensus_refuses_sign_disagreement():
    base = np.full((8, 8, 3), 100, np.uint8)
    positive = np.full_like(base, 104)
    stronger_positive = np.full_like(base, 108)
    disagreement = stronger_positive.copy()
    disagreement[:, :4, 1] = 96

    correction, evidence = stable_correction(
        base,
        (positive, stronger_positive, disagreement),
        correction_sigma=0.0,
    )

    assert np.all(correction[:, :4, 1] == 0)
    assert np.all(correction[:, 4:, 1] == 4)
    assert evidence["model_count"] == 3
    assert 0 < evidence["stable_fraction"] < 1


def test_ownership_gate_vetoes_observed_foreground_evidence():
    rng = np.random.default_rng(8)
    size = 96
    owner = np.full((size, size, 3), 128, np.uint8)
    owner[:, : size // 2] = rng.integers(
        0,
        256,
        (size, size // 2, 3),
        dtype=np.uint8,
    )
    other = cv2.GaussianBlur(owner, (15, 15), 5)
    other[:, size // 2 :] = owner[:, size // 2 :]

    gate, evidence = _ownership_gate(
        [owner, other],
        owner=0,
        alpha=np.zeros((size, size), np.float32),
        spatial_scale=1.0,
    )

    assert gate[:, :40].mean() < 0.01
    assert gate[:, 60:].mean() > 0.99
    assert evidence["owner_veto_confident_fraction"] > 0.4


def test_ordered_visibility_requires_positive_rear_observation():
    rng = np.random.default_rng(18)
    size = 120
    owner = np.full((size, size, 3), 128, np.uint8)
    rear = owner.copy()
    owner_texture = rng.integers(
        0,
        256,
        (size, size // 3, 3),
        dtype=np.uint8,
    )
    rear_texture = rng.integers(
        0,
        256,
        (size, size // 3, 3),
        dtype=np.uint8,
    )
    owner[:, : size // 3] = owner_texture
    rear[:, : size // 3] = cv2.GaussianBlur(
        owner_texture,
        (15, 15),
        5,
    )
    owner[:, -size // 3 :] = cv2.GaussianBlur(
        rear_texture,
        (15, 15),
        5,
    )
    rear[:, -size // 3 :] = rear_texture

    gate, evidence = _ordered_visibility_gate(
        [owner, rear],
        owner=0,
        alpha=np.zeros((size, size), np.float32),
        spatial_scale=1.0,
    )

    # Sharp foreground blocks the rear on the left.  The flat center has no
    # positive observation in either frame and therefore returns to identity.
    # Only the decisively rear-focused structure on the right is licensed.
    assert gate[:, :30].mean() < 0.01
    assert gate[:, 48:72].mean() < 0.01
    assert gate[:, -30:].mean() > 0.95
    assert evidence["rear_evidence_confident_fraction"] > 0.25
    assert evidence["ordered_visibility_fraction"] > 0.25


def test_fringe_softening_never_leaks_outside_predicted_coverage():
    alpha = np.zeros((128, 128), np.float32)
    cv2.circle(alpha, (64, 64), 24, 1.0, -1)
    radius = RADIUS_FRACTION * 128
    blurred = _box_disk_blur(alpha, 0.7 * radius)
    physical_support = (
        (blurred > 0.05)
        & (blurred < 0.95)
        & (alpha < 0.5)
    )
    fringe = _fringe_mask(alpha, radius, 4.0)

    assert np.any(fringe > 0)
    assert np.all(fringe[~physical_support] == 0)


def test_physical_reranking_and_joint_recovery_improve_scene_error():
    frames, gt, candidate, _ = _physical_giant_stack()
    selected, evidence = select_licensed_candidate(frames, [candidate])

    assert selected is not None
    assert evidence["reason"] == "licensed"
    assert evidence["forward_after"] < 0.85 * evidence["forward_before"]

    base = fuse_perband(frames, harden=0.5)
    output, report = recover_giant_veil(frames, base, [candidate])
    before = np.abs(base.astype(np.float32) - gt).mean()
    after = np.abs(output.astype(np.float32) - gt).mean()

    assert report["fired"] is True
    assert report["model_count"] == 6
    assert report["changed_pixels"] > 0
    assert after < before


def test_joint_recovery_refusal_is_byte_identical():
    frames, _, candidate, _ = _physical_giant_stack(96)
    base = fuse_perband(frames, harden=0.5)
    weak = dict(candidate)
    weak["feats"] = np.zeros(7, np.float32)

    output, report = recover_giant_veil(frames, base, [weak])
    assert report["fired"] is False
    assert report["reason"] == "candidate_unlicensed"
    assert np.array_equal(output, base)

    output, report = recover_giant_veil(frames + [frames[0]], base, [candidate])
    assert report["reason"] == "requires_two_frames"
    assert np.array_equal(output, base)


def test_owner_frame_satellite_repairs_missing_foreground_support():
    frames, gt, candidate, satellite = _physical_giant_stack(
        missing_satellite=True
    )
    selected, evidence = select_licensed_candidate(frames, [candidate])
    assert selected is not None, evidence

    owner_masks = np.stack([satellite.astype(np.uint8)], axis=0)
    support, support_evidence = complete_owner_support(
        frames,
        selected,
        owner_masks,
    )

    assert support_evidence["owner_support_accepted_count"] == 1
    assert support_evidence["owner_support_forward_improvement"] > 0.01
    assert np.mean(support[satellite > 0]) > 0.95

    base = fuse_perband(frames, harden=0.5)
    output, report = recover_giant_veil(
        frames,
        base,
        [candidate],
        owner_masks_by_frame=[
            owner_masks,
            np.zeros((0, *satellite.shape), np.uint8),
        ],
    )
    satellite_pixels = satellite > 0

    assert report["owner_support_pixels"] > 0
    assert np.array_equal(
        output[satellite_pixels],
        frames[0][satellite_pixels],
    )
    assert (
        np.abs(output.astype(np.float32) - gt)[satellite_pixels].mean()
        < np.abs(base.astype(np.float32) - gt)[satellite_pixels].mean()
    )


def test_owner_frame_parent_silhouette_completes_opaque_foreground():
    frames, _, candidate, satellite = _physical_giant_stack(
        missing_satellite=True
    )
    selected, evidence = select_licensed_candidate(frames, [candidate])
    assert selected is not None, evidence
    parent = np.maximum(
        candidate["alpha"],
        satellite,
    ).astype(np.uint8)[None]

    support, support_evidence = complete_owner_support(
        frames,
        selected,
        parent,
    )

    assert support_evidence["owner_support_accepted_count"] == 1
    assert support_evidence["owner_support_kinds"] == ["parent_silhouette"]
    assert support_evidence["owner_support_forward_improvement"] > 0.01
    assert (
        support_evidence[
            "owner_support_local_forward_improvement_ratio"
        ]
        > 0.05
    )
    assert np.mean(support[satellite > 0]) > 0.95


def test_owner_frame_fragment_without_physical_support_is_refused():
    frames, _, candidate, _ = _physical_giant_stack()
    selected, evidence = select_licensed_candidate(frames, [candidate])
    assert selected is not None, evidence

    size = frames[0].shape[0]
    unsupported = np.zeros((size, size), np.uint8)
    unsupported[
        size // 2 - 5 : size // 2 + 5,
        size // 2 + 43 : size // 2 + 51,
    ] = 1
    support, support_evidence = complete_owner_support(
        frames,
        selected,
        unsupported[None],
    )

    assert not np.any(support)
    assert support_evidence["owner_support_candidate_count"] == 1
    assert support_evidence["owner_support_accepted_count"] == 0
    assert (
        support_evidence["owner_support_reason"]
        == "no_forward_licensed_fragment"
    )

    unsupported_parent = np.maximum(
        candidate["alpha"],
        unsupported,
    ).astype(np.uint8)[None]
    support, support_evidence = complete_owner_support(
        frames,
        selected,
        unsupported_parent,
    )
    assert not np.any(support)
    assert support_evidence["owner_support_candidate_count"] == 1
    assert support_evidence["owner_support_accepted_count"] == 0
