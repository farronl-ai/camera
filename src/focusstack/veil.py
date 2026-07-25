"""Forward models and audits for wide-occluder veil recovery.

The veil specialist is allowed to synthesize detail only because the observed
far-focus frame still contains an attenuated remnant of that detail.  This
module keeps the physical model, blind noise estimate, and observation-domain
audit together so the gain path cannot silently drift away from the render it
claims to invert.

A model is a plain dictionary to keep the optional fusion hook lightweight:

```
{
    "far_idx": int,
    "D": HxWx3 additive haze field,
    "ab": HxWx3 blurred alpha,
    "pm": HxWx3 blurred foreground premultiplication,
    "alpha": HxW sharp semantic matte,
    "mask": HxW background-side veil support,
    "radii": (B, G, R) blur radii in pixels,
    "sigma": conservative observed noise estimate,
}
```

Nothing here decides whether a model is trustworthy enough to fire.  The
outcome gate owns that decision; these functions expose the evidence it needs.
"""

from __future__ import annotations

import cv2
import numpy as np

from .reconstruct import _disk_blur


_NOISE_KERNEL = np.array(
    [[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]],
    dtype=np.float32,
)


def estimate_noise_sigma(images: list[np.ndarray]) -> float:
    """Estimate sensor-noise sigma conservatively from observed frames.

    The 3x3 Immerkaer high-pass kernel has noise standard deviation ``6*sigma``.
    Taking the median absolute response on the lowest-gradient 35% of pixels
    suppresses scene texture; the minimum across frames/channels avoids treating
    residual detail as noise.  A small floor prevents under-shrinking nearly
    noiseless or heavily quantized inputs.
    """
    estimates: list[float] = []
    for image in images:
        for c in range(image.shape[2]):
            x = image[..., c].astype(np.float32)
            response = np.abs(
                cv2.filter2D(x, cv2.CV_32F, _NOISE_KERNEL, borderType=cv2.BORDER_REFLECT)
            )
            gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
            grad = cv2.magnitude(gx, gy)
            flat = grad <= np.quantile(grad, 0.35)
            if flat.any():
                # median(|N(0, (6 sigma)^2)|) = 0.67448975 * 6 sigma
                estimates.append(float(np.median(response[flat]) / (0.67448975 * 6.0)))
    if not estimates:
        return 3.0
    return float(np.clip(min(estimates), 1.5, 20.0))


def build_veil_model(
    images: list[np.ndarray],
    alpha: np.ndarray,
    radii: tuple[float, float, float] | list[float] | np.ndarray,
    owner: int,
    far_idx: int,
    sigma: float | None = None,
) -> dict:
    """Build a per-channel forward veil model from blind inputs.

    Channel-specific radii are mandatory: F53 showed that a shared radius leaves
    chromatic residuals which multiplicative recovery turns into false texture.
    The returned ``pm`` is retained separately because the post-subtraction
    remnant still contains ``ab * pm``; failing to remove it extends foreground
    texture beyond the true silhouette.
    """
    if len(images) < 2:
        raise ValueError("Veil recovery needs at least two frames")
    if not (0 <= owner < len(images) and 0 <= far_idx < len(images)):
        raise ValueError("owner/far_idx outside the image stack")
    if owner == far_idx:
        raise ValueError("owner and far_idx must differ")

    a = np.clip(np.asarray(alpha, np.float32), 0.0, 1.0)
    if a.shape != images[0].shape[:2]:
        raise ValueError("alpha shape must match the image frames")
    rr = np.asarray(radii, np.float32)
    if rr.shape != (3,) or not np.isfinite(rr).all() or (rr <= 0).any():
        raise ValueError("radii must be three positive finite BGR values")

    owner_f = images[owner].astype(np.float32)
    far_f = images[far_idx].astype(np.float32)
    near_pm = owner_f * a[..., None]
    D = np.empty_like(far_f)
    ab = np.empty_like(far_f)
    pm = np.empty_like(far_f)
    for c, radius in enumerate(rr):
        ab[..., c] = _disk_blur(a, float(radius))
        pm[..., c] = _disk_blur(near_pm[..., c], float(radius))
        D[..., c] = (pm[..., c] - near_pm[..., c]) + far_f[..., c] * (
            a - ab[..., c]
        )

    # The green channel is the shared geometric support; per-channel fields
    # retain chromatic differences within it.
    band = ((ab[..., 1] > 0.02) & (ab[..., 1] < 0.98) & (a < 0.5)).astype(np.float32)
    band = cv2.GaussianBlur(band, (0, 0), 2.0)
    return {
        "far_idx": int(far_idx),
        "D": D * band[..., None],
        "ab": ab,
        "pm": pm,
        "alpha": a,
        "mask": band,
        "radii": rr,
        "sigma": float(estimate_noise_sigma(images) if sigma is None else sigma),
    }


def render_far_observation(scene: np.ndarray, model: dict) -> np.ndarray:
    """Push a claimed sharp scene back through the model into the far frame."""
    return model["pm"] + (1.0 - model["ab"]) * scene.astype(np.float32)


def forward_residual(
    scene: np.ndarray,
    images: list[np.ndarray],
    model: dict,
    clip: float = 30.0,
) -> float:
    """Robust observation-domain residual in the modeled veil shell.

    Lower is better.  This is necessary rather than sufficient evidence: blur
    has a null space, so the value is a gate feature and audit—not a standalone
    truth certificate.
    """
    pred = render_far_observation(scene, model)
    obs = images[int(model["far_idx"])].astype(np.float32)
    mask = np.asarray(model["mask"]) > 0.05
    if mask.sum() < 32:
        return float("inf")
    residual = np.abs(pred - obs).mean(axis=2)
    return float(np.mean(np.minimum(residual[mask], clip)))


def unsupported_texture_score(
    base: np.ndarray,
    corrected: np.ndarray,
    images: list[np.ndarray],
    model: dict,
) -> float:
    """GT-free proxy for fine texture unsupported by observed remnants.

    The score only examines modeled background-side veil pixels whose strongest
    observed mid-scale structure is at or below the estimated noise floor.
    It measures finest-band energy introduced by the correction there, normalized
    by the expected high-pass sensor noise.  Correct recovery may still receive
    a non-zero score, so this is deliberately a gate feature rather than a hard
    source-similarity veto.
    """
    mask = np.asarray(model["mask"]) > 0.05
    sigma = max(float(model.get("sigma", 3.0)), 1e-3)

    def highpass(image: np.ndarray, blur_sigma: float) -> np.ndarray:
        x = image.astype(np.float32)
        return x - cv2.GaussianBlur(x, (0, 0), blur_sigma)

    observed_mid = np.maximum.reduce(
        [np.abs(highpass(image, 1.6)).mean(axis=2) for image in images]
    )
    quiet = mask & (observed_mid <= 1.5 * sigma)
    if quiet.sum() < 32:
        return 0.0
    introduced = highpass(corrected, 0.7) - highpass(base, 0.7)
    energy = np.sqrt(np.mean(introduced * introduced, axis=2))
    return float(energy[quiet].mean() / (sigma + 1e-6))


def fit_chromatic_spread(
    images: list[np.ndarray],
    alpha: np.ndarray,
    base_radius: float,
    owner: int,
    far_idx: int,
    spreads: tuple[float, ...] = (0.0, 0.04, 0.08, 0.12),
) -> tuple[np.ndarray, dict]:
    """Fit a bounded BGR radius spread by forward-rendering the observation.

    Absolute radius selection needs outcome calibration (the raw residual is
    radius-biased on textured backgrounds).  For one proposed base radius this
    function only chooses the small chromatic spread, preventing three
    unconstrained channel fits from chasing unrelated background texture.
    """
    a = np.clip(np.asarray(alpha, np.float32), 0.0, 1.0)
    far = images[far_idx].astype(np.float32)
    max_radius = float(base_radius) * (1.0 + max(spreads))
    support = cv2.dilate(
        (a > 0.15).astype(np.uint8),
        np.ones((2 * max(1, int(np.ceil(max_radius))) + 1,) * 2, np.uint8),
    )
    plate = np.stack(
        [
            cv2.inpaint(
                np.clip(far[..., c], 0, 255).astype(np.uint8),
                support,
                5,
                cv2.INPAINT_TELEA,
            )
            for c in range(3)
        ],
        axis=2,
    ).astype(np.float32)
    owner_pm = images[owner].astype(np.float32) * a[..., None]

    rows: list[tuple[float, float]] = []
    for spread in spreads:
        radii = np.asarray(
            [base_radius * (1.0 - spread), base_radius, base_radius * (1.0 + spread)],
            np.float32,
        )
        errors = []
        for c, radius in enumerate(radii):
            ab = _disk_blur(a, float(radius))
            pm = _disk_blur(owner_pm[..., c], float(radius))
            pred = pm + plate[..., c] * (1.0 - ab)
            shell = (a < 0.5) & (ab > 0.02) & (ab < 0.98)
            if shell.sum() < 32:
                errors.append(30.0)
            else:
                errors.append(
                    float(np.mean(np.minimum(np.abs(pred[shell] - far[..., c][shell]), 30.0)))
                )
        rows.append((float(np.mean(errors)), float(spread)))

    rows.sort()
    best_error, best_spread = rows[0]
    second_error = rows[1][0] if len(rows) > 1 else best_error
    radii = np.asarray(
        [
            base_radius * (1.0 - best_spread),
            base_radius,
            base_radius * (1.0 + best_spread),
        ],
        np.float32,
    )
    return radii, {
        "spread": best_spread,
        "fit_error": best_error,
        "fit_margin": float(second_error - best_error),
    }
