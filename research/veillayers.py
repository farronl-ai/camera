#!/usr/bin/env python3
"""F55 — joint two-layer inversion for wide-occluder recovery.

The retired hybrid repaired an already-fused band while pretending only the
far-focus remnant carried background evidence.  This experiment instead fits
both captured frames simultaneously:

    O_i = H_near,i(alpha * N) + (1 - H_near,i(alpha)) * H_far,i(S)

where N and S are the sharp foreground color and sharp background.  The solver
estimates only corrections to the observed owner/far frames, with a
high-frequency penalty on those corrections.  It therefore has no external
texture prior and no free generator.

First rungs: oracle alpha/radii, realistic object scenes, max-side 512.

Run:
    cd research
    ../.venv/bin/python veillayers.py p0
    ../.venv/bin/python veillayers.py p1
    ../.venv/bin/python veillayers.py p2
    ../.venv/bin/python veillayers.py p3
    ../.venv/bin/python veillayers.py p4
    ../.venv/bin/python veillayers.py p4h
    ../.venv/bin/python veillayers.py p5
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import metrics as M  # noqa: E402
from hardbench import disk_blur  # noqa: E402
from t2_candidates import candidates_with_features  # noqa: E402
from t2_confidence import scenes  # noqa: E402
from veilband import fringe_mask  # noqa: E402
from veilship import false_texture_error  # noqa: E402
from focusstack.fusion import fuse_perband  # noqa: E402


MAX_SIDE = 512
OFFSETS = np.asarray((-0.04, 0.0, 0.04), np.float32)
FOCUS = (0.15, 0.85)
NEAR_DEPTH = 0.15
FAR_DEPTH = 0.85


def blur3(image: np.ndarray, radii: np.ndarray) -> np.ndarray:
    return np.stack(
        [disk_blur(image[..., c].astype(np.float32), float(radii[c])) for c in range(3)],
        axis=2,
    )


def radii_for(max_r: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
    near = [
        np.abs(NEAR_DEPTH - (focus + OFFSETS)) * max_r for focus in FOCUS
    ]
    far = [np.abs(FAR_DEPTH - (focus + OFFSETS)) * max_r for focus in FOCUS]
    return near, far


def prepare_model(alpha: np.ndarray, max_r: float) -> dict:
    near_radii, far_radii = radii_for(max_r)
    transmission = [
        1.0 - blur3(np.repeat(alpha[..., None], 3, axis=2), radii)
        for radii in near_radii
    ]
    return {
        "alpha": alpha.astype(np.float32),
        "near_radii": near_radii,
        "far_radii": far_radii,
        "transmission": transmission,
    }


def forward_layers(near: np.ndarray, far: np.ndarray, model: dict) -> list[np.ndarray]:
    premult = model["alpha"][..., None] * near
    return [
        blur3(premult, rn) + transmission * blur3(far, rf)
        for rn, rf, transmission in zip(
            model["near_radii"], model["far_radii"], model["transmission"]
        )
    ]


def adjoint(residuals: list[np.ndarray], model: dict) -> tuple[np.ndarray, np.ndarray]:
    alpha = model["alpha"][..., None]
    near = np.zeros_like(residuals[0], np.float32)
    far = np.zeros_like(residuals[0], np.float32)
    for residual, rn, rf, transmission in zip(
        residuals,
        model["near_radii"],
        model["far_radii"],
        model["transmission"],
    ):
        # Disk kernels are symmetric. Reflect-border filtering is only
        # approximately self-adjoint at the outer image border; the recovery
        # support is far from that border in the factory.
        near += alpha * blur3(residual, rn)
        far += blur3(transmission * residual, rf)
    return near, far


def correction_regularizer(image: np.ndarray) -> np.ndarray:
    """B^T B for B=(I-G_sigma), a DC-preserving high-frequency penalty."""
    high = image - cv2.GaussianBlur(image, (0, 0), 1.0)
    return high - cv2.GaussianBlur(high, (0, 0), 1.0)


def solve_layers(
    images: list[np.ndarray],
    alpha: np.ndarray,
    max_r: float,
    initial_far: np.ndarray | None = None,
    smooth_lambda: float = 2.0,
    anchor_lambda: float = 0.02,
    iterations: int = 18,
) -> tuple[np.ndarray, dict]:
    """Solve for layer corrections with conjugate gradients on normal equations."""
    observed = [image.astype(np.float32) for image in images]
    model = prepare_model(alpha, max_r)
    near0 = observed[0].copy()
    far0 = (
        observed[1].copy()
        if initial_far is None
        else initial_far.astype(np.float32).copy()
    )
    predicted0 = forward_layers(near0, far0, model)
    residual0 = [obs - pred for obs, pred in zip(observed, predicted0)]
    rhs_n, rhs_f = adjoint(residual0, model)

    def normal(dn: np.ndarray, df: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        an, af = adjoint(forward_layers(dn, df, model), model)
        an += smooth_lambda * correction_regularizer(dn) + anchor_lambda * dn
        af += smooth_lambda * correction_regularizer(df) + anchor_lambda * df
        return an, af

    def dot(an: np.ndarray, af: np.ndarray, bn: np.ndarray, bf: np.ndarray) -> float:
        return float(np.sum(an * bn, dtype=np.float64) + np.sum(af * bf, dtype=np.float64))

    dn = np.zeros_like(near0)
    df = np.zeros_like(far0)
    rn, rf = rhs_n.copy(), rhs_f.copy()
    pn, pf = rn.copy(), rf.copy()
    rr = dot(rn, rf, rn, rf)
    history = [float(np.sqrt(rr / (rn.size + rf.size)))]
    for _ in range(iterations):
        qn, qf = normal(pn, pf)
        denom = dot(pn, pf, qn, qf)
        if not np.isfinite(denom) or denom <= 1e-12:
            break
        step = rr / denom
        dn += step * pn
        df += step * pf
        rn -= step * qn
        rf -= step * qf
        rr_new = dot(rn, rf, rn, rf)
        history.append(float(np.sqrt(rr_new / (rn.size + rf.size))))
        if rr_new <= 1e-8 * max(rr, 1e-12):
            rr = rr_new
            break
        beta = rr_new / max(rr, 1e-12)
        pn = rn + beta * pn
        pf = rf + beta * pf
        rr = rr_new

    near = np.clip(near0 + dn, 0, 255)
    far = np.clip(far0 + df, 0, 255)
    predicted = forward_layers(near, far, model)
    before = float(
        np.mean([np.abs(p - y).mean() for p, y in zip(predicted0, observed)])
    )
    after = float(
        np.mean([np.abs(p - y).mean() for p, y in zip(predicted, observed)])
    )
    high_n = dn - cv2.GaussianBlur(dn, (0, 0), 1.0)
    high_f = df - cv2.GaussianBlur(df, (0, 0), 1.0)
    data0_sum = sum(
        np.sum((p - y) ** 2, dtype=np.float64)
        for p, y in zip(predicted0, observed)
    )
    data_sum = sum(
        np.sum((p - y) ** 2, dtype=np.float64)
        for p, y in zip(predicted, observed)
    )
    normalizer = float(sum(y.size for y in observed))
    objective0 = data0_sum / normalizer
    objective = (
        data_sum
        + smooth_lambda
        * (
            np.sum(high_n * high_n, dtype=np.float64)
            + np.sum(high_f * high_f, dtype=np.float64)
        )
        + anchor_lambda
        * (
            np.sum(dn * dn, dtype=np.float64)
            + np.sum(df * df, dtype=np.float64)
        )
    ) / normalizer
    return (
        model["alpha"][..., None] * near
        + (1.0 - model["alpha"][..., None]) * far,
        {
            "forward_before": before,
            "forward_after": after,
            "objective_before": float(objective0),
            "objective_after": float(objective),
            "cg_history": history,
            "near_correction_rms": float(np.sqrt(np.mean(dn * dn))),
            "far_correction_rms": float(np.sqrt(np.mean(df * df))),
        },
    )


def resize_oracle(sc: dict) -> dict:
    h, w = sc["gt"].shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    size = (max(2, round(w * scale)), max(2, round(h * scale)))
    return {
        "sid": sc["sid"],
        "gt": cv2.resize(sc["gt"], size, interpolation=cv2.INTER_AREA),
        "alpha": cv2.resize(sc["alpha"], size, interpolation=cv2.INTER_AREA).astype(
            np.float32
        ),
        "frames": [
            cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            for frame in sc["frames"]
        ],
        "max_r": float(sc["max_r"]) * scale,
        "regime": float(sc["max_r"] / max(sc["gt"].shape[:2])),
    }


def apply_to_fringe(
    base: np.ndarray,
    solved: np.ndarray,
    sc: dict,
    correction_sigma: float = 0.0,
) -> np.ndarray:
    correction = solved.astype(np.float32) - base.astype(np.float32)
    if correction_sigma > 0:
        correction = cv2.GaussianBlur(
            correction, (0, 0), correction_sigma, borderType=cv2.BORDER_REFLECT
        )
    support = fringe_mask(sc["alpha"], sc["max_r"]) & (sc["alpha"] < 0.5)
    mask = cv2.GaussianBlur(support.astype(np.float32), (0, 0), 2.0)
    output = base.astype(np.float32) + correction * mask[..., None]
    return np.uint8(np.clip(output, 0, 255))


def stable_correction(
    base: np.ndarray,
    solved: list[np.ndarray],
    correction_sigma: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """Keep only the correction shared by a regularization ensemble.

    An inverse-problem component that changes sign when the physically
    reasonable regularization strength changes is not licensed by the
    observations. Stable components retain the smallest ensemble magnitude,
    making this a conservative projection rather than an averaging prior.
    """
    corrections = []
    for image in solved:
        correction = image.astype(np.float32) - base.astype(np.float32)
        if correction_sigma > 0:
            correction = cv2.GaussianBlur(
                correction,
                (0, 0),
                correction_sigma,
                borderType=cv2.BORDER_REFLECT,
            )
        corrections.append(correction)
    bank = np.stack(corrections, axis=0)
    same_sign = (np.min(bank, axis=0) > 0) | (np.max(bank, axis=0) < 0)
    mean = np.mean(bank, axis=0)
    correction = np.sign(mean) * np.min(np.abs(bank), axis=0) * same_sign
    return correction, {
        "stable_fraction": float(same_sign.mean()),
        "ensemble_spread_rms": float(np.sqrt(np.mean(np.var(bank, axis=0)))),
        "retained_rms": float(np.sqrt(np.mean(correction * correction))),
    }


def apply_correction_to_fringe(
    base: np.ndarray,
    correction: np.ndarray,
    sc: dict,
) -> np.ndarray:
    support = fringe_mask(sc["alpha"], sc["max_r"]) & (sc["alpha"] < 0.5)
    mask = cv2.GaussianBlur(support.astype(np.float32), (0, 0), 2.0)
    output = base.astype(np.float32) + correction * mask[..., None]
    return np.uint8(np.clip(output, 0, 255))


def score(output: np.ndarray, base: np.ndarray, sc: dict) -> dict:
    fringe = fringe_mask(sc["alpha"], sc["max_r"])
    error = np.abs(output.astype(np.float32) - sc["gt"].astype(np.float32)).sum(2)
    error0 = np.abs(base.astype(np.float32) - sc["gt"].astype(np.float32)).sum(2)
    ft, n = false_texture_error(output, sc["gt"], sc["alpha"], sc["max_r"])
    ft0, _ = false_texture_error(base, sc["gt"], sc["alpha"], sc["max_r"])
    return {
        "dg": M.ref_ssim(output, sc["gt"]) - M.ref_ssim(base, sc["gt"]),
        "de_fringe": float(error[fringe].mean() - error0[fringe].mean()),
        "d_false_texture": float(ft - ft0),
        "false_texture_pixels": n,
    }


def summarize(rows: list[dict], key: str) -> dict:
    values = np.asarray([row[key] for row in rows], np.float64)
    values = values[np.isfinite(values)]
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "worst": float(values.min()),
        "best": float(values.max()),
    }


def cmd_p0() -> None:
    configs = (
        (0.5, 0.01),
        (2.0, 0.02),
        (8.0, 0.05),
        (32.0, 0.10),
    )
    selected = list(scenes())
    # Deterministic cross-regime subset including F54's worst counterexample and
    # former apparent wins. Expand to all 100 only if this rung has a ceiling.
    indices = (0, 1, 2, 5, 7, 9, 21, 31, 42, 47, 56, 60, 72, 75, 81, 87, 91, 99)
    results = []
    for smooth_lambda, anchor_lambda in configs:
        rows = []
        print(
            f"config smooth={smooth_lambda:g} anchor={anchor_lambda:g}",
            flush=True,
        )
        for index in indices:
            sc = resize_oracle(selected[index])
            base = fuse_perband(sc["frames"], harden=0.5)
            solved, evidence = solve_layers(
                sc["frames"],
                sc["alpha"],
                sc["max_r"],
                smooth_lambda=smooth_lambda,
                anchor_lambda=anchor_lambda,
            )
            output = apply_to_fringe(base, solved, sc)
            outcome = score(output, base, sc)
            row = {
                "sid": sc["sid"],
                "regime": sc["regime"],
                **outcome,
                **evidence,
            }
            rows.append(row)
            print(
                f"  {sc['sid']} dg={outcome['dg']:+.5f} "
                f"dfr={outcome['de_fringe']:+.2f} "
                f"dft={outcome['d_false_texture']:+.3f} "
                f"obs={evidence['forward_before']:.2f}->{evidence['forward_after']:.2f}",
                flush=True,
            )
        aggregate = {
            key: summarize(rows, key)
            for key in ("dg", "de_fringe", "d_false_texture")
        }
        results.append(
            {
                "smooth_lambda": smooth_lambda,
                "anchor_lambda": anchor_lambda,
                "rows": rows,
                "aggregate": aggregate,
            }
        )
        print(json.dumps(aggregate, indent=2), flush=True)
    path = os.path.join(HERE, "veillayers_p0_faranchor.json")
    with open(path, "w") as handle:
        json.dump({"max_side": MAX_SIDE, "configs": results}, handle, indent=2)
    print(f"-> {path}", flush=True)
    print(
        "DOCTRINE: the solver estimates only corrections to observed layers, "
        "must reproduce every focal frame, and is judged on realistic-object GT "
        "including false-texture complements and worst scenes.",
        flush=True,
    )


def cmd_p1() -> None:
    """Audit the selected P0 model across the complete 100-scene factory."""
    smooth_lambda = 8.0
    anchor_lambda = 0.05
    correction_sigma = 0.5
    rows = []
    for index, original in enumerate(scenes()):
        sc = resize_oracle(original)
        base = fuse_perband(sc["frames"], harden=0.5)
        solved, evidence = solve_layers(
            sc["frames"],
            sc["alpha"],
            sc["max_r"],
            smooth_lambda=smooth_lambda,
            anchor_lambda=anchor_lambda,
        )
        output = apply_to_fringe(
            base, solved, sc, correction_sigma=correction_sigma
        )
        outcome = score(output, base, sc)
        correction = solved.astype(np.float32) - base.astype(np.float32)
        support = fringe_mask(sc["alpha"], sc["max_r"]) & (sc["alpha"] < 0.5)
        row = {
            "index": index,
            "sid": sc["sid"],
            "regime": sc["regime"],
            "alpha_area": float((sc["alpha"] >= 0.5).mean()),
            "support_area": float(support.mean()),
            "correction_rms": float(
                np.sqrt(np.mean(correction[support] ** 2))
            ),
            **outcome,
            **evidence,
        }
        rows.append(row)
        print(
            f"{sc['sid']} dg={outcome['dg']:+.5f} "
            f"dfr={outcome['de_fringe']:+.2f} "
            f"dft={outcome['d_false_texture']:+.3f} "
            f"obs={evidence['forward_before']:.2f}->{evidence['forward_after']:.2f}",
            flush=True,
        )
    aggregate = {
        key: summarize(rows, key)
        for key in (
            "dg",
            "de_fringe",
            "d_false_texture",
            "correction_rms",
            "forward_before",
            "forward_after",
        )
    }
    payload = {
        "max_side": MAX_SIDE,
        "smooth_lambda": smooth_lambda,
        "anchor_lambda": anchor_lambda,
        "correction_sigma": correction_sigma,
        "rows": rows,
        "aggregate": aggregate,
    }
    path = os.path.join(HERE, "veillayers_p1_oracle.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(aggregate, indent=2), flush=True)
    print(f"-> {path}", flush=True)
    print(
        "DOCTRINE: full-factory promotion depends on harmful tails and "
        "false-texture complements, never a favorable mean alone.",
        flush=True,
    )


def cmd_p2() -> None:
    """Full-factory audit of regularization-stable correction components."""
    configs = ((2.0, 0.02), (8.0, 0.05), (32.0, 0.10))
    correction_sigma = 0.5
    rows = []
    for index, original in enumerate(scenes()):
        sc = resize_oracle(original)
        base = fuse_perband(sc["frames"], harden=0.5)
        solved = []
        evidence_bank = []
        for smooth_lambda, anchor_lambda in configs:
            image, evidence = solve_layers(
                sc["frames"],
                sc["alpha"],
                sc["max_r"],
                smooth_lambda=smooth_lambda,
                anchor_lambda=anchor_lambda,
            )
            solved.append(image)
            evidence_bank.append(evidence)
        correction, uncertainty = stable_correction(
            base, solved, correction_sigma=correction_sigma
        )
        output = apply_correction_to_fringe(base, correction, sc)
        outcome = score(output, base, sc)
        row = {
            "index": index,
            "sid": sc["sid"],
            "regime": sc["regime"],
            **outcome,
            **uncertainty,
            "forward_before": evidence_bank[1]["forward_before"],
            "forward_after": evidence_bank[1]["forward_after"],
            "far_correction_rms": evidence_bank[1]["far_correction_rms"],
        }
        rows.append(row)
        print(
            f"{sc['sid']} dg={outcome['dg']:+.5f} "
            f"dfr={outcome['de_fringe']:+.2f} "
            f"dft={outcome['d_false_texture']:+.3f} "
            f"stable={uncertainty['stable_fraction']:.3f}",
            flush=True,
        )
    aggregate = {
        key: summarize(rows, key)
        for key in (
            "dg",
            "de_fringe",
            "d_false_texture",
            "stable_fraction",
            "ensemble_spread_rms",
            "retained_rms",
        )
    }
    payload = {
        "max_side": MAX_SIDE,
        "configs": configs,
        "correction_sigma": correction_sigma,
        "rows": rows,
        "aggregate": aggregate,
    }
    path = os.path.join(HERE, "veillayers_p2_consensus.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(aggregate, indent=2), flush=True)
    print(f"-> {path}", flush=True)
    print(
        "DOCTRINE: disagreement across defensible inverse solutions is "
        "uncertainty; only stable evidence may alter the image.",
        flush=True,
    )


def cmd_p3() -> None:
    """Audit consensus inversion with runtime semantic matte, true radius."""
    configs = ((2.0, 0.02), (8.0, 0.05), (32.0, 0.10))
    correction_sigma = 0.5
    rows = []
    refused = []
    for index, original in enumerate(scenes()):
        candidates = candidates_with_features(original, topk=1)
        if not candidates:
            refused.append({"index": index, "sid": original["sid"]})
            print(f"{original['sid']} REFUSE no semantic candidate", flush=True)
            continue
        candidate = candidates[0]
        sc = resize_oracle(original)
        h, w = sc["gt"].shape[:2]
        alpha_est = cv2.resize(
            candidate["alpha"], (w, h), interpolation=cv2.INTER_AREA
        ).astype(np.float32)
        owner = int(candidate["owner"])
        ordered = [sc["frames"][owner], sc["frames"][1 - owner]]
        base = fuse_perband(sc["frames"], harden=0.5)
        solved = []
        evidence_bank = []
        for smooth_lambda, anchor_lambda in configs:
            image, evidence = solve_layers(
                ordered,
                alpha_est,
                sc["max_r"],
                smooth_lambda=smooth_lambda,
                anchor_lambda=anchor_lambda,
            )
            solved.append(image)
            evidence_bank.append(evidence)
        correction, uncertainty = stable_correction(
            base, solved, correction_sigma=correction_sigma
        )
        model_sc = dict(sc, alpha=alpha_est)
        output = apply_correction_to_fringe(base, correction, model_sc)
        outcome = score(output, base, sc)
        row = {
            "index": index,
            "sid": sc["sid"],
            "regime": sc["regime"],
            "owner": owner,
            "semantic_features": candidate["feats"].tolist(),
            "alpha_error": float(np.abs(alpha_est - sc["alpha"]).mean()),
            **outcome,
            **uncertainty,
            "forward_before": evidence_bank[1]["forward_before"],
            "forward_after": evidence_bank[1]["forward_after"],
            "far_correction_rms": evidence_bank[1]["far_correction_rms"],
        }
        rows.append(row)
        print(
            f"{sc['sid']} dg={outcome['dg']:+.5f} "
            f"dfr={outcome['de_fringe']:+.2f} "
            f"dft={outcome['d_false_texture']:+.3f} "
            f"ae={row['alpha_error']:.3f} owner={owner}",
            flush=True,
        )
    aggregate = {
        key: summarize(rows, key)
        for key in (
            "dg",
            "de_fringe",
            "d_false_texture",
            "alpha_error",
            "stable_fraction",
            "ensemble_spread_rms",
            "retained_rms",
        )
    }
    payload = {
        "max_side": MAX_SIDE,
        "configs": configs,
        "correction_sigma": correction_sigma,
        "rows": rows,
        "refused": refused,
        "aggregate": aggregate,
    }
    path = os.path.join(HERE, "veillayers_p3_semantic_true_radius.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(aggregate, indent=2), flush=True)
    print(f"refused={len(refused)} -> {path}", flush=True)
    print(
        "DOCTRINE: a positive true-matte ceiling licenses testing runtime "
        "mattes; it does not transfer safety to them by assumption.",
        flush=True,
    )


def run_candidate_bank(start_index: int, filename: str) -> None:
    """Measure whether physical evidence can rerank the top-4 matte bank."""
    smooth_lambda = 8.0
    anchor_lambda = 0.05
    correction_sigma = 0.5
    scenes_out = []
    for index, original in enumerate(scenes()):
        if index < start_index:
            continue
        sc = resize_oracle(original)
        h, w = sc["gt"].shape[:2]
        base = fuse_perband(sc["frames"], harden=0.5)
        candidate_rows = []
        for rank, candidate in enumerate(candidates_with_features(original, topk=4)):
            alpha_est = cv2.resize(
                candidate["alpha"], (w, h), interpolation=cv2.INTER_AREA
            ).astype(np.float32)
            owner = int(candidate["owner"])
            ordered = [sc["frames"][owner], sc["frames"][1 - owner]]
            solved, evidence = solve_layers(
                ordered,
                alpha_est,
                sc["max_r"],
                smooth_lambda=smooth_lambda,
                anchor_lambda=anchor_lambda,
            )
            correction = solved.astype(np.float32) - base.astype(np.float32)
            correction = cv2.GaussianBlur(
                correction,
                (0, 0),
                correction_sigma,
                borderType=cv2.BORDER_REFLECT,
            )
            model_sc = dict(sc, alpha=alpha_est)
            output = apply_correction_to_fringe(base, correction, model_sc)
            outcome = score(output, base, sc)
            row = {
                "rank": rank,
                "owner": owner,
                "semantic_features": candidate["feats"].tolist(),
                "alpha_error": float(np.abs(alpha_est - sc["alpha"]).mean()),
                **outcome,
                **{
                    key: value
                    for key, value in evidence.items()
                    if key != "cg_history"
                },
            }
            candidate_rows.append(row)
            print(
                f"{sc['sid']} c{rank} dg={outcome['dg']:+.5f} "
                f"ae={row['alpha_error']:.3f} "
                f"obj={evidence['objective_before']:.2f}"
                f"->{evidence['objective_after']:.2f}",
                flush=True,
            )
        scenes_out.append(
            {
                "index": index,
                "sid": sc["sid"],
                "regime": sc["regime"],
                "candidates": candidate_rows,
            }
        )
    path = os.path.join(HERE, filename)
    with open(path, "w") as handle:
        json.dump(
            {
                "max_side": MAX_SIDE,
                "smooth_lambda": smooth_lambda,
                "anchor_lambda": anchor_lambda,
                "correction_sigma": correction_sigma,
                "scenes": scenes_out,
            },
            handle,
            indent=2,
        )
    print(f"-> {path}", flush=True)
    print(
        "DOCTRINE: reranking is licensed only by order-invariant observed "
        "evidence; oracle alpha error is analysis, never a selector.",
        flush=True,
    )


def cmd_p4() -> None:
    run_candidate_bank(0, "veillayers_p4_candidate_bank.json")


def cmd_p4h() -> None:
    run_candidate_bank(100, "veillayers_p4_fresh_holdout.json")


def candidate_is_licensed(row: dict) -> bool:
    """Predeclared P5 high-precision license; all features are order-invariant."""
    score, purity, _, area_fit, *_ = row["semantic_features"]
    forward_ratio = row["forward_after"] / max(row["forward_before"], 1e-6)
    return (
        score > 0.5
        and purity > 0.85
        and area_fit > 0.9
        and forward_ratio < 0.85
    )


def run_licensed_consensus(bank_filename: str, output_filename: str) -> None:
    bank = json.load(open(os.path.join(HERE, bank_filename)))
    originals = {index: sc for index, sc in enumerate(scenes())}
    configs = ((2.0, 0.02), (8.0, 0.05), (32.0, 0.10))
    fired = []
    for scene_row in bank["scenes"]:
        if not scene_row["candidates"]:
            continue
        chosen = min(
            scene_row["candidates"], key=lambda row: row["forward_after"]
        )
        if not candidate_is_licensed(chosen):
            continue
        index = int(scene_row["index"])
        original = originals[index]
        candidates = candidates_with_features(original, topk=4)
        candidate = candidates[int(chosen["rank"])]
        sc = resize_oracle(original)
        h, w = sc["gt"].shape[:2]
        alpha_est = cv2.resize(
            candidate["alpha"], (w, h), interpolation=cv2.INTER_AREA
        ).astype(np.float32)
        owner = int(candidate["owner"])
        ordered = [sc["frames"][owner], sc["frames"][1 - owner]]
        base = fuse_perband(sc["frames"], harden=0.5)
        solved = []
        for smooth_lambda, anchor_lambda in configs:
            image, _ = solve_layers(
                ordered,
                alpha_est,
                sc["max_r"],
                smooth_lambda=smooth_lambda,
                anchor_lambda=anchor_lambda,
            )
            solved.append(image)
        correction, uncertainty = stable_correction(base, solved)
        output = apply_correction_to_fringe(
            base, correction, dict(sc, alpha=alpha_est)
        )
        outcome = score(output, base, sc)
        row = {
            "index": index,
            "sid": sc["sid"],
            "rank": int(chosen["rank"]),
            "regime": sc["regime"],
            "alpha_error": float(np.abs(alpha_est - sc["alpha"]).mean()),
            "forward_ratio": float(
                chosen["forward_after"] / max(chosen["forward_before"], 1e-6)
            ),
            **outcome,
            **uncertainty,
        }
        fired.append(row)
        print(
            f"{sc['sid']} dg={outcome['dg']:+.5f} "
            f"dfr={outcome['de_fringe']:+.2f} "
            f"dft={outcome['d_false_texture']:+.3f}",
            flush=True,
        )
    path = os.path.join(HERE, output_filename)
    with open(path, "w") as handle:
        json.dump(
            {
                "source_bank": bank_filename,
                "configs": configs,
                "fired": fired,
                "aggregate": {
                    key: summarize(fired, key)
                    for key in ("dg", "de_fringe", "d_false_texture")
                },
            },
            handle,
            indent=2,
        )
    print(f"fired={len(fired)} -> {path}", flush=True)


def cmd_p5() -> None:
    run_licensed_consensus(
        "veillayers_p4_candidate_bank.json",
        "veillayers_p5_licensed_dev.json",
    )
    run_licensed_consensus(
        "veillayers_p4_fresh_holdout.json",
        "veillayers_p5_licensed_holdout.json",
    )
    print(
        "DOCTRINE: a license calibrated on inspected data counts only after "
        "the unchanged rule survives a fresh scene-disjoint holdout.",
        flush=True,
    )


def main() -> None:
    commands = {
        "p0": cmd_p0,
        "p1": cmd_p1,
        "p2": cmd_p2,
        "p3": cmd_p3,
        "p4": cmd_p4,
        "p4h": cmd_p4h,
        "p5": cmd_p5,
    }
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        raise SystemExit("usage: veillayers.py {p0|p1|p2|p3|p4|p4h|p5}")
    commands[sys.argv[1]]()


if __name__ == "__main__":
    main()
