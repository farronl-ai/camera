#!/usr/bin/env python3
"""F54 — hallucination-safe veil recovery graduation.

The first rung is intentionally easier than shipping: use the semantic matte
that runtime can produce but the factory's true blur radius.  If the corrected
hybrid cannot survive that condition, blind radius selection cannot rescue it.

Run:
    cd research
    ../.venv/bin/python veilship.py oracle

Outputs are small JSON evidence files; generated images/caches stay out of git.
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
from focusstack.enhance import _build_veil_D  # noqa: E402
from focusstack.fusion import fuse_perband  # noqa: E402
from focusstack.gates import VEIL_GATE, predict_gain  # noqa: E402
from focusstack.veil import (  # noqa: E402
    build_veil_model,
    estimate_noise_sigma,
    fit_chromatic_spread,
    forward_residual,
    unsupported_texture_score,
)


MAX_SIDE = 512


def resize_scene(sc: dict, alpha_est: np.ndarray) -> dict:
    """Resize a factory scene and its runtime matte without changing geometry."""
    h, w = sc["gt"].shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    size = (max(2, round(w * scale)), max(2, round(h * scale)))
    return {
        "sid": sc["sid"],
        "scale": scale,
        "max_r": float(sc["max_r"]) * scale,
        "gt": cv2.resize(sc["gt"], size, interpolation=cv2.INTER_AREA),
        "alpha": cv2.resize(sc["alpha"], size, interpolation=cv2.INTER_AREA).astype(
            np.float32
        ),
        "alpha_est": cv2.resize(
            alpha_est, size, interpolation=cv2.INTER_AREA
        ).astype(np.float32),
        "frames": [
            cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            for frame in sc["frames"]
        ],
    }


def false_texture_error(
    output: np.ndarray,
    gt: np.ndarray,
    alpha: np.ndarray,
    max_r: float,
) -> tuple[float, int]:
    """GT-credited error on smooth pixels the old contrast metric excluded.

    Select true background-side fringe pixels where the noise-free GT has little
    mid-scale energy, then measure finest-band output error against GT.  Unlike
    an output-only sharpness measure, real background texture cancels.
    """
    support = fringe_mask(alpha, max_r) & (alpha < 0.5)
    gt_f = gt.astype(np.float32)
    out_f = output.astype(np.float32)
    gt_mid = np.abs(gt_f - cv2.GaussianBlur(gt_f, (0, 0), 1.6)).mean(axis=2)
    quiet = support & (gt_mid <= 1.0)
    if quiet.sum() < 32:
        return float("nan"), int(quiet.sum())
    gt_fine = gt_f - cv2.GaussianBlur(gt_f, (0, 0), 0.7)
    out_fine = out_f - cv2.GaussianBlur(out_f, (0, 0), 0.7)
    error = np.sqrt(np.mean((out_fine - gt_fine) ** 2, axis=2))
    return float(error[quiet].mean()), int(quiet.sum())


def score(
    output: np.ndarray,
    base: np.ndarray,
    sc: dict,
) -> dict:
    fringe = fringe_mask(sc["alpha"], sc["max_r"])
    error = np.abs(output.astype(np.float32) - sc["gt"].astype(np.float32)).sum(2)
    base_error = np.abs(base.astype(np.float32) - sc["gt"].astype(np.float32)).sum(2)
    ft, ft_n = false_texture_error(output, sc["gt"], sc["alpha"], sc["max_r"])
    ft_base, _ = false_texture_error(base, sc["gt"], sc["alpha"], sc["max_r"])
    return {
        "dg": M.ref_ssim(output, sc["gt"]) - M.ref_ssim(base, sc["gt"]),
        "de_fringe": float(error[fringe].mean() - base_error[fringe].mean()),
        "false_texture": ft,
        "d_false_texture": ft - ft_base,
        "false_texture_pixels": ft_n,
    }


def summarize(rows: list[dict], key: str) -> dict:
    values = np.asarray([row[key] for row in rows], np.float64)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"n": 0}
    return {
        "n": int(len(finite)),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "p10": float(np.quantile(finite, 0.10)),
        "p90": float(np.quantile(finite, 0.90)),
        "worst": float(finite.min()),
        "best": float(finite.max()),
    }


def cmd_oracle() -> None:
    rows = []
    for i, original in enumerate(scenes()):
        candidates = candidates_with_features(original, topk=1)
        if not candidates:
            print(f"{original['sid']}: no semantic candidate", flush=True)
            continue
        candidate = candidates[0]
        sc = resize_scene(original, candidate["alpha"])
        frames = sc["frames"]
        sigma = estimate_noise_sigma(frames)
        # Factory render uses near-vs-far radii {0.66, 0.70, 0.74} * max_r.
        radii = sc["max_r"] * np.asarray((0.66, 0.70, 0.74), np.float32)
        model = build_veil_model(
            frames,
            sc["alpha_est"],
            radii,
            owner=int(candidate["owner"]),
            far_idx=1 - int(candidate["owner"]),
            sigma=sigma,
        )
        base = fuse_perband(frames, harden=0.5)
        sub_model = dict(model, gain_strength=0.0)
        subtraction = fuse_perband(frames, harden=0.5, veil_models=[sub_model])
        hybrid = fuse_perband(frames, harden=0.5, veil_models=[model])

        s_sub = score(subtraction, base, sc)
        s_hybrid = score(hybrid, base, sc)
        true_ab = disk_blur(sc["alpha"], float(radii[1]))
        matte_error = float(np.abs(sc["alpha_est"] - sc["alpha"]).mean())
        row = {
            "sid": sc["sid"],
            "regime": float(original["max_r"] / max(original["gt"].shape[:2])),
            "semantic_features": candidate["feats"].tolist(),
            "owner": int(candidate["owner"]),
            "matte_error": matte_error,
            "sigma": sigma,
            "sub": s_sub,
            "hybrid": s_hybrid,
            "hybrid_vs_sub_dg": s_hybrid["dg"] - s_sub["dg"],
            "hybrid_vs_sub_false_texture": (
                s_hybrid["false_texture"] - s_sub["false_texture"]
            ),
            "forward_base": forward_residual(base, frames, model),
            "forward_sub": forward_residual(subtraction, frames, model),
            "forward_hybrid": forward_residual(hybrid, frames, model),
            "unsupported_texture": unsupported_texture_score(
                base, hybrid, frames, model
            ),
            "true_ab_mean": float(true_ab[model["mask"] > 0.05].mean()),
        }
        rows.append(row)
        print(
            f"{sc['sid']} coc={row['regime']:.3f} aerr={matte_error:.3f} "
            f"sub={s_sub['dg']:+.5f} hybrid={s_hybrid['dg']:+.5f} "
            f"h-s={row['hybrid_vs_sub_dg']:+.5f} "
            f"ftΔ={s_hybrid['d_false_texture']:+.3f}",
            flush=True,
        )

    aggregate = {}
    for method in ("sub", "hybrid"):
        for key in ("dg", "de_fringe", "d_false_texture"):
            aggregate[f"{method}.{key}"] = summarize(
                [row[method] for row in rows], key
            )
    aggregate["hybrid_vs_sub_dg"] = summarize(rows, "hybrid_vs_sub_dg")
    aggregate["hybrid_vs_sub_false_texture"] = summarize(
        rows, "hybrid_vs_sub_false_texture"
    )
    output = {"max_side": MAX_SIDE, "rows": rows, "aggregate": aggregate}
    path = os.path.join(HERE, "veilship_oracle.json")
    with open(path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(aggregate, indent=2), flush=True)
    print(f"-> {path}", flush=True)
    print(
        "DOCTRINE: the hybrid was judged against scene GT, forward evidence, "
        "and the textureless complement; visual plausibility was not accepted "
        "as truth.",
        flush=True,
    )


def cmd_subtraction() -> None:
    """Audit a pattern-limited subtraction fix under the shipped gate."""
    rows = []
    total_candidates = 0
    total_fired = 0
    for original in scenes():
        candidates = candidates_with_features(original, topk=4)
        total_candidates += len(candidates)
        fired = [
            candidate
            for candidate in candidates
            if predict_gain(VEIL_GATE, candidate["feats"]) >= VEIL_GATE["margin"]
        ]
        total_fired += len(fired)
        if candidates:
            sc = resize_scene(original, candidates[0]["alpha"])
        else:
            sc = resize_scene(original, np.zeros_like(original["alpha"]))
        frames = sc["frames"]
        base = fuse_perband(frames, harden=0.5)
        shared_fields: dict[int, np.ndarray] = {}
        smooth_fields: dict[int, np.ndarray] = {}
        chromatic_fields: dict[int, np.ndarray] = {}
        fired_evidence = []
        blur_sigma = 0.004 * max(base.shape[:2])

        for candidate in fired:
            alpha = cv2.resize(
                candidate["alpha"],
                (base.shape[1], base.shape[0]),
                interpolation=cv2.INTER_AREA,
            ).astype(np.float32)
            owner = int(candidate["owner"])
            far_idx = 1 - owner
            radius = 0.012 * max(base.shape[:2])
            shared = _build_veil_D(frames, alpha, radius, owner, far_idx)
            smooth = cv2.GaussianBlur(shared, (0, 0), blur_sigma)

            base_radius = 0.7 * radius
            radii, fit = fit_chromatic_spread(
                frames, alpha, base_radius, owner, far_idx
            )
            model = build_veil_model(
                frames,
                alpha,
                radii,
                owner,
                far_idx,
                sigma=estimate_noise_sigma(frames),
            )
            chromatic = cv2.GaussianBlur(model["D"], (0, 0), blur_sigma)
            shared_fields[far_idx] = shared_fields.get(far_idx, 0) + shared
            smooth_fields[far_idx] = smooth_fields.get(far_idx, 0) + smooth
            chromatic_fields[far_idx] = (
                chromatic_fields.get(far_idx, 0) + chromatic
            )
            fired_evidence.append(
                {
                    "prediction": predict_gain(VEIL_GATE, candidate["feats"]),
                    "features": candidate["feats"].tolist(),
                    "spread": fit["spread"],
                    "spread_fit_error": fit["fit_error"],
                    "spread_fit_margin": fit["fit_margin"],
                }
            )

        outputs = {"identity": base}
        if fired:
            outputs.update(
                {
                    "shipped": fuse_perband(
                        frames, harden=0.5, veil_D=shared_fields
                    ),
                    "smooth": fuse_perband(
                        frames, harden=0.5, veil_D=smooth_fields
                    ),
                    "chromatic_smooth": fuse_perband(
                        frames, harden=0.5, veil_D=chromatic_fields
                    ),
                }
            )
        method_scores = {
            name: score(output, base, sc) for name, output in outputs.items()
        }
        row = {
            "sid": sc["sid"],
            "regime": float(original["max_r"] / max(original["gt"].shape[:2])),
            "candidate_count": len(candidates),
            "fired_count": len(fired),
            "blur_sigma": blur_sigma,
            "fired_evidence": fired_evidence,
            "methods": method_scores,
        }
        rows.append(row)
        if fired:
            print(
                f"{sc['sid']} fired={len(fired)} "
                f"ship={method_scores['shipped']['dg']:+.5f}/"
                f"{method_scores['shipped']['d_false_texture']:+.3f} "
                f"smooth={method_scores['smooth']['dg']:+.5f}/"
                f"{method_scores['smooth']['d_false_texture']:+.3f} "
                f"CA={method_scores['chromatic_smooth']['dg']:+.5f}/"
                f"{method_scores['chromatic_smooth']['d_false_texture']:+.3f}",
                flush=True,
            )

    aggregate = {
        "scenes": len(rows),
        "candidates": total_candidates,
        "fired_candidates": total_fired,
        "fired_scenes": sum(row["fired_count"] > 0 for row in rows),
    }
    for method in ("shipped", "smooth", "chromatic_smooth"):
        method_rows = [
            row["methods"][method] for row in rows if method in row["methods"]
        ]
        for key in ("dg", "de_fringe", "d_false_texture"):
            aggregate[f"{method}.{key}"] = summarize(method_rows, key)
    result = {"max_side": MAX_SIDE, "rows": rows, "aggregate": aggregate}
    path = os.path.join(HERE, "veilship_subtraction.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(aggregate, indent=2), flush=True)
    print(f"-> {path}", flush=True)
    print(
        "DOCTRINE: only the shipped gate fired; smoothing removed unsupported "
        "fine correction bands, and every changed scene was judged against GT "
        "including textureless fringe pixels.",
        flush=True,
    )


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"oracle", "subtraction"}:
        raise SystemExit("usage: veilship.py {oracle|subtraction}")
    {"oracle": cmd_oracle, "subtraction": cmd_subtraction}[sys.argv[1]]()


if __name__ == "__main__":
    main()
