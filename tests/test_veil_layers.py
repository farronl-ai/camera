import cv2
import numpy as np

from focusstack.fusion import fuse_perband
from focusstack.veil_layers import (
    RADIUS_FRACTION,
    _adjoint,
    _box_disk_blur,
    _cross_frame_satellite_support,
    _forward_layers,
    _fringe_mask,
    _one_sided_rear_application_mask,
    _owner_geometry_consensus,
    _owner_front_reconstruction_support,
    _ordered_visibility_gate,
    _ownership_gate,
    _prepare_model,
    candidate_is_licensed,
    complete_owner_support,
    recover_giant_veil,
    refine_owner_candidate,
    select_licensed_candidate,
    select_one_sided_owner_geometry,
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


def test_runtime_forward_model_never_admits_rear_inside_owned_support():
    size = 96
    yy, xx = np.mgrid[:size, :size]
    alpha = np.zeros((size, size), np.float32)
    alpha[25:71, 43:53] = 1.0
    foreground = np.full((size, size, 3), (190, 80, 35), np.float32)
    checker = ((xx + yy) % 2).astype(np.float32) * 220.0
    rear_a = np.repeat(checker[..., None], 3, axis=2)
    rear_b = 255.0 - rear_a
    model = _prepare_model(
        alpha,
        RADIUS_FRACTION * size,
        _box_disk_blur,
    )

    frames_a = _forward_layers(foreground, rear_a, model)
    frames_b = _forward_layers(foreground, rear_b, model)
    owned = alpha == 1.0

    for frame_a, frame_b in zip(frames_a, frames_b):
        np.testing.assert_array_equal(frame_a[owned], frame_b[owned])
    assert model["formation_model"] == "one_sided_opaque_v1"


def test_runtime_v2_protects_full_hard_support_after_alpha_antialiasing():
    size = 96
    yy, xx = np.mgrid[:size, :size]
    hard_ownership = np.zeros((size, size), bool)
    hard_ownership[25:71, 43:53] = True
    alpha = cv2.GaussianBlur(
        hard_ownership.astype(np.float32),
        (0, 0),
        0.5,
    )
    foreground = np.full((size, size, 3), (190, 80, 35), np.float32)
    checker = ((xx + yy) % 2).astype(np.float32) * 220.0
    rear_a = np.repeat(checker[..., None], 3, axis=2)
    rear_b = 255.0 - rear_a
    model = _prepare_model(
        alpha,
        RADIUS_FRACTION * size,
        _box_disk_blur,
        hard_ownership=hard_ownership,
    )

    frames_a = _forward_layers(foreground, rear_a, model)
    frames_b = _forward_layers(foreground, rear_b, model)

    assert np.any(
        hard_ownership & (alpha < 1.0)
    ), "the test must cover the old soft-alpha ownership loophole"
    for frame_a, frame_b in zip(frames_a, frames_b):
        np.testing.assert_array_equal(
            frame_a[hard_ownership],
            frame_b[hard_ownership],
        )
    assert model["formation_model"] == "one_sided_opaque_v2"


def test_one_sided_forward_and_adjoint_are_consistent():
    rng = np.random.default_rng(31)
    size = 72
    alpha = np.zeros((size, size), np.float32)
    alpha[20:52, 27:45] = 1.0
    model = _prepare_model(
        alpha,
        RADIUS_FRACTION * size,
        _box_disk_blur,
        hard_ownership=alpha >= 0.5,
    )
    near = np.zeros((size, size, 3), np.float32)
    far = np.zeros_like(near)
    residuals = [np.zeros_like(near), np.zeros_like(near)]
    interior = np.s_[15:-15, 15:-15]
    near[interior] = rng.normal(size=near[interior].shape)
    far[interior] = rng.normal(size=far[interior].shape)
    for residual in residuals:
        residual[interior] = rng.normal(size=residual[interior].shape)

    forward = _forward_layers(near, far, model)
    adjoint_near, adjoint_far = _adjoint(residuals, model)
    left = sum(
        float(np.sum(value * residual, dtype=np.float64))
        for value, residual in zip(forward, residuals)
    )
    right = float(
        np.sum(near * adjoint_near, dtype=np.float64)
        + np.sum(far * adjoint_far, dtype=np.float64)
    )

    np.testing.assert_allclose(left, right, rtol=2e-5, atol=2e-5)


def test_one_sided_geometry_selects_focused_owner_mask():
    frames, _, candidate, _ = _physical_giant_stack()
    true_mask = candidate["alpha"] > 0
    eroded = cv2.erode(
        true_mask.astype(np.uint8),
        np.ones((7, 7), np.uint8),
    )
    false_mask = np.zeros_like(eroded)
    false_mask[15:55, 10:40] = 1
    other_observation = cv2.dilate(
        true_mask.astype(np.uint8),
        np.ones((5, 5), np.uint8),
    )

    selected, evidence = select_one_sided_owner_geometry(
        frames,
        [
            np.stack([true_mask, eroded]).astype(np.uint8),
            np.stack([other_observation, false_mask]),
        ],
    )

    assert selected is not None, evidence
    assert selected["owner"] == 0
    assert evidence["one_sided_geometry_fired"] is True
    assert evidence["one_sided_geometry_competitor_margin"] > 0.05
    np.testing.assert_array_equal(selected["alpha"] >= 0.5, true_mask)


def test_one_sided_recovery_hard_selects_confident_foreground_interior():
    frames, _, candidate, _ = _physical_giant_stack()
    true_mask = candidate["alpha"] > 0
    eroded = cv2.erode(
        true_mask.astype(np.uint8),
        np.ones((7, 7), np.uint8),
    )
    false_mask = np.zeros_like(eroded)
    false_mask[15:55, 10:40] = 1
    other_observation = cv2.dilate(
        true_mask.astype(np.uint8),
        np.ones((5, 5), np.uint8),
    )
    owner_masks = [
        np.stack([true_mask, eroded]).astype(np.uint8),
        np.stack([other_observation, false_mask]),
    ]
    base = fuse_perband(frames, harden=0.5)

    output, report = recover_giant_veil(
        frames,
        base,
        [candidate],
        owner_masks_by_frame=owner_masks,
    )

    assert report["fired"] is True
    assert report["one_sided_geometry_fired"] is True
    assert report["owner_front_reconstruction_pixels"] > 0
    expected_front = cv2.fastNlMeansDenoisingColored(
        frames[0],
        None,
        2.0,
        2.0,
        3,
        7,
    )
    hard_front = np.all(output == expected_front, axis=2) & true_mask
    assert int(hard_front.sum()) >= report["owner_front_reconstruction_pixels"]
    assert (
        report["front_observation_source"]
        == "focused_owner_nlm_foreground_only"
    )
    assert report["rear_mask_front_veto_removed_pixels"] >= 0


def test_one_sided_owned_core_is_one_observation_not_a_blend():
    frames, _, candidate, _ = _physical_giant_stack()
    true_mask = candidate["alpha"] > 0
    eroded = cv2.erode(
        true_mask.astype(np.uint8),
        np.ones((7, 7), np.uint8),
    )
    other_observation = cv2.dilate(
        true_mask.astype(np.uint8),
        np.ones((5, 5), np.uint8),
    )
    false_mask = np.zeros_like(eroded)
    false_mask[15:55, 10:40] = 1
    owner_masks = [
        np.stack([true_mask, eroded]).astype(np.uint8),
        np.stack([other_observation, false_mask]),
    ]
    selected = select_one_sided_owner_geometry(frames, owner_masks)
    assert selected[0] is not None, selected[1]
    base = np.rint(
        0.37 * frames[0].astype(np.float32)
        + 0.63 * frames[1].astype(np.float32)
    ).astype(np.uint8)

    output, report = recover_giant_veil(
        frames,
        base,
        [candidate],
        owner_masks_by_frame=owner_masks,
        one_sided_selection=selected,
    )

    owner = int(selected[0]["owner"])
    denoised_owner = cv2.fastNlMeansDenoisingColored(
        frames[owner],
        None,
        2.0,
        2.0,
        3,
        7,
    )
    alpha = np.asarray(selected[0]["alpha"], np.float32)
    front_consensus, _, _ = _owner_geometry_consensus(
        alpha,
        owner_masks[owner],
        RADIUS_FRACTION * max(alpha.shape),
        1.0,
    )
    inside_distance = cv2.distanceTransform(
        (alpha >= 0.5).astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    hard_front = (inside_distance > 1.0) & front_consensus
    hard_front |= np.asarray(
        selected[0].get(
            "cross_frame_satellite_extent",
            np.zeros(alpha.shape, bool),
        ),
        bool,
    )
    assert np.array_equal(
        output[hard_front],
        denoised_owner[hard_front],
    )
    assert report["owner_front_reconstruction_pixels"] == int(
        hard_front.sum()
    )


def test_cross_frame_satellite_requires_positive_two_frame_geometry():
    size = 192
    completed = np.zeros((size, size), bool)
    cv2.circle(
        completed.view(np.uint8),
        (80, 96),
        35,
        1,
        -1,
    )
    satellite = np.zeros_like(completed)
    satellite[88:100, 118:126] = True
    unsupported = np.zeros_like(completed)
    unsupported[20:32, 20:28] = True
    owner_masks = np.stack(
        [completed, satellite, unsupported]
    ).astype(np.uint8)
    other_masks = np.stack(
        [
            cv2.dilate(
                completed.astype(np.uint8),
                np.ones((3, 3), np.uint8),
            ),
            satellite.astype(np.uint8),
        ]
    )

    support, evidence = _cross_frame_satellite_support(
        [owner_masks, other_masks],
        owner=0,
        completed=completed,
        max_radius=RADIUS_FRACTION * size,
    )

    assert np.mean(support[satellite]) > 0.95
    assert not np.any(support[unsupported])
    assert evidence["one_sided_satellite_accepted_count"] == 1


def test_one_sided_rear_mask_never_crosses_true_foreground():
    frames, _, candidate, _ = _physical_giant_stack()
    true_mask = candidate["alpha"] > 0
    eroded = cv2.erode(
        true_mask.astype(np.uint8),
        np.ones((7, 7), np.uint8),
    )
    other_observation = cv2.dilate(
        true_mask.astype(np.uint8),
        np.ones((5, 5), np.uint8),
    )
    false_mask = np.zeros_like(eroded)
    false_mask[15:55, 10:40] = 1
    owner_masks = [
        np.stack([true_mask, eroded]).astype(np.uint8),
        np.stack([other_observation, false_mask]),
    ]
    selected, evidence = select_one_sided_owner_geometry(
        frames,
        owner_masks,
    )
    assert selected is not None, evidence
    consensus = np.ones(true_mask.shape, bool)

    rear_mask, report = _one_sided_rear_application_mask(
        frames,
        selected["owner"],
        selected["alpha"],
        selected["rear_support_alpha"],
        selected["front_extent"],
        consensus,
        RADIUS_FRACTION * max(true_mask.shape),
        1.0,
    )

    assert not np.any((rear_mask > 1e-4) & true_mask)
    assert (
        report["rear_mask_after_geometry_corroboration_pixels"]
        >= report["rear_mask_active_pixels"]
    )
    assert report["rear_mask_presence_reason"] in {
        "reverse_reblur_exceeds_noise_floor",
        "insufficient_rear_noise_reference",
    }


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

    refined, refinement_evidence = refine_owner_candidate(
        frames,
        selected,
        parent,
    )
    assert refinement_evidence["owner_refinement_fired"] is True
    front = _owner_front_reconstruction_support(
        refined["alpha"],
        parent,
        {**support_evidence, **refinement_evidence},
        RADIUS_FRACTION * candidate["alpha"].shape[0],
        1.0,
    )
    assert np.any(front)
    assert np.all(refined["alpha"][front] >= 0.5)

    base = fuse_perband(frames, harden=0.5)
    output, report = recover_giant_veil(
        frames,
        base,
        [candidate],
        owner_masks_by_frame=[
            parent,
            np.zeros((0, *satellite.shape), np.uint8),
        ],
    )
    assert report["owner_front_reconstruction_pixels"] == int(front.sum())
    assert np.array_equal(output[front], frames[0][front])


def test_owner_geometry_consensus_rejects_disputed_mask_extension():
    size = 192
    yy, xx = np.mgrid[:size, :size]
    alpha = (
        (xx - size / 2) ** 2 + (yy - size / 2) ** 2
        < (size * 0.25) ** 2
    ).astype(np.float32)
    masks = []
    for index in range(4):
        mask = alpha > 0
        if index == 0:
            mask = mask.copy()
            mask[20:45, 15:35] = True
        masks.append(mask)

    front, fringe, evidence = _owner_geometry_consensus(
        alpha,
        np.stack(masks).astype(np.uint8),
        RADIUS_FRACTION * size,
        1.0,
    )

    assert evidence["owner_consensus_active"] is True
    assert evidence["owner_consensus_proposal_count"] == 4
    assert np.all(front[alpha > 0])
    assert not np.any(front[20:45, 15:35])
    assert not np.any(fringe[20:45, 15:35])


def test_owner_geometry_consensus_is_identity_with_one_proposal():
    size = 96
    alpha = np.zeros((size, size), np.float32)
    alpha[20:76, 20:76] = 1.0
    front, fringe, evidence = _owner_geometry_consensus(
        alpha,
        alpha[None].astype(np.uint8),
        RADIUS_FRACTION * size,
        1.0,
    )

    assert evidence["owner_consensus_active"] is False
    assert evidence["owner_consensus_proposal_count"] == 1
    assert np.all(front)
    assert np.all(fringe)


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
