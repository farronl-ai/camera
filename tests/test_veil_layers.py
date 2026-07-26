import cv2
import numpy as np

from focusstack.fusion import fuse_perband
from focusstack.veil_layers import (
    RADIUS_FRACTION,
    _box_disk_blur,
    _forward_layers,
    _ownership_gate,
    _prepare_model,
    candidate_is_licensed,
    recover_giant_veil,
    select_licensed_candidate,
    stable_correction,
)


def _physical_giant_stack(size=192):
    yy, xx = np.mgrid[:size, :size]
    background = np.stack(
        [
            50.0 + 0.4 * xx,
            90.0 + 20.0 * np.sin(xx / 5.0),
            130.0 + 0.3 * yy,
        ],
        axis=2,
    ).astype(np.float32)
    alpha = (
        (xx - size / 2) ** 2 + (yy - size / 2) ** 2
        < (size * 0.22) ** 2
    ).astype(np.float32)
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
        "alpha": alpha,
        "owner": 0,
    }
    return frames, gt, candidate


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


def test_physical_reranking_and_joint_recovery_improve_scene_error():
    frames, gt, candidate = _physical_giant_stack()
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
    frames, _, candidate = _physical_giant_stack(96)
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
