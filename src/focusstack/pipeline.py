"""Orchestration: run the three stages end-to-end.

    load  ->  align  ->  (focus measure)  ->  fuse  ->  save

`run()` is the single entry point used by both the CLI and library callers. When
`debug_dir` is set it also writes intermediate visualizations so you can *see*
what each stage did (aligned frames, focus maps, and — for the max method — a
colorized map of which source frame won each pixel).
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from . import io as fio
from .align import align_stack
from .focus import content_aware_energies, focus_measures
from .fusion import (depth_from_focus, fuse_blend, fuse_decision, fuse_max,
                     fuse_perband, fuse_pyramid)


def _focus_maps(images: list[np.ndarray], method: str) -> list[np.ndarray]:
    """Per-frame focus maps; content_aware needs all frames (cross-frame contrast)."""
    if method == "content_aware":
        return content_aware_energies([fio.to_gray_float(im) for im in images])
    return focus_measures(images, method=method)


def _normalize_map(m: np.ndarray) -> np.ndarray:
    """Scale a float map to a 0-255 uint8 image for viewing."""
    lo, hi = float(m.min()), float(m.max())
    if hi - lo < 1e-9:
        return np.zeros(m.shape, dtype=np.uint8)
    return ((m - lo) / (hi - lo) * 255.0).astype(np.uint8)


def _colorize_selection(index_map: np.ndarray, n: int) -> np.ndarray:
    """Map frame-index -> color so each source region is a distinct hue."""
    scaled = (index_map.astype(np.float32) / max(1, n - 1) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_JET)


def _twoframe_route(align_report: dict, method: str, n: int) -> tuple[bool, str]:
    """First half of the two-frame routing rule: is there a stranded object?

    Two methods that win on OPPOSITE scenes are routed, never merged (F101). The
    two-frame architecture (`twoframe.py`) eliminates F108's streak on the kitchen
    sweep and un-doubles the cat figurine there, but costs 0.0015 GT-SSIM against
    the shipped depth-bin path on the analytic factory, whose two planes depth
    separates cleanly — exactly F101's shape.

    The signature tested here is the shipped alignment's OWN measurement: whether
    the motion-group override fired. That override fires when a compact object's
    measured motion disagrees with the depth path's fit at the object's own
    location by more than 5 px — i.e. precisely when the scene contains an object
    the depth bins strand. Measured: kitchen 2 groups overridden, large-motion 3,
    IMG-46 1; analytic factory, zero-motion and small-motion 0 (declined here, and
    therefore byte-identical to the pre-route output by construction — no
    two-frame work is even started on them).

    Firing is NECESSARY but not sufficient: the second half of the rule is the
    architecture's own licence (`twoframe.shift_licence`), checked on the composite
    it actually builds, because large-motion showed a stranded object the
    two-frame path cannot serve. See the caller.

    Deliberately NOT a quality comparison between the two outputs: no-reference
    metrics cannot adjudicate an alignment change (F81a), so a runtime A/B would
    be a coin toss wearing a number.
    """
    if method != "perband":
        return False, f"method is {method!r}, and the two-frame path fuses per-band"
    if n < 3:
        return False, "fewer than 3 frames carry no usable focal statistics"
    groups = align_report.get("motion_groups") or {}
    overridden = int(groups.get("overridden") or 0)
    if overridden < 1:
        skipped = groups.get("skipped")
        return False, ("the motion-group override did not fire"
                       + (f" ({skipped})" if skipped else ""))
    return True, (f"the motion-group override fired on {overridden} object(s) — "
                  "a stranded-object scene")


def run(
    inputs: list[str],
    output: str,
    method: str = "perband",
    align: bool = True,
    align_motion: str = "affine",
    align_depth_bins: int = 4,
    align_depth_model: str = "bins",
    align_motion_override: bool = True,
    twoframe_route: bool = True,
    focus_method: str = "content_aware",
    levels: int | None = None,
    harden: float = 0.5,
    weight_scale: float = 1.0,
    normalize_exposure: bool = True,
    reconstruct_boundaries: bool = False,
    enhance: str = "auto",
    depth_out: str | None = None,
    boundary_out: str | None = None,
    debug_dir: str | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Run focus stacking on `inputs` and write the result to `output`.

    Returns the fused BGR uint8 image.
    """

    def log(msg: str) -> None:
        if verbose:
            print(f"[focusstack] {msg}")

    usable: list[np.ndarray] | None = None
    named = fio.load_images(inputs)
    names = [name for name, _ in named]
    images = [img for _, img in named]
    # The two-frame route re-registers from the ORIGINAL frames (it composes the
    # global stage into its own per-layer geometry so native pixels are resampled
    # exactly once), so keep them before alignment overwrites the list.
    sources = images
    routed = False
    log(f"loaded {len(images)} frames: {', '.join(names)}")

    if align:
        log(f"aligning frames (motion={align_motion}, depth_bins={align_depth_bins}) ...")
        images, align_report = align_stack(
            images,
            motion=align_motion,
            depth_bins=align_depth_bins,
            depth_model=align_depth_model,
            motion_override=align_motion_override,
            return_report=True,
        )
        corrected = sum(
            1 for frame in align_report["frames"].values() if frame["accepted"] > 0
        )
        if corrected:
            log(f"depth-aware pass corrected {corrected} frame(s) "
                f"across {align_report['bins']} depth bins")
        groups = align_report.get("motion_groups", {})
        if groups.get("overridden"):
            log(f"motion-group override corrected {groups['overridden']} object(s) "
                f"whose depth bin was fitted to other content")
        elif groups.get("skipped"):
            log(f"motion-group override skipped: {groups['skipped']}")
        # Pixels parallax uncovered: present in some frames, absent in others.
        usable = align_report.get("usable")
        if usable is not None:
            withheld = float(np.mean([1.0 - m.mean() for m in usable]))
            if withheld > 0:
                log(f"withholding {withheld * 100:.1f}% of pixels per frame "
                    f"as parallax-uncovered")

        if twoframe_route:
            routed, why = _twoframe_route(align_report, method, len(images))
            if not routed:
                log(f"fusion path: shipped depth-bin — {why}")

    if normalize_exposure:
        log("normalizing per-frame exposure/WB drift ...")
        images = fio.normalize_exposure(images)
        # A per-frame channel gain commutes with the warp, so the routed path
        # gets the same correction applied to the frames it registers itself.
        if routed:
            sources = fio.normalize_exposure(sources)

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        for name, img in zip(names, images):
            stem = os.path.splitext(name)[0]
            fio.save_image(os.path.join(debug_dir, f"aligned_{stem}.png"), img)

    twoframe_fused = None
    if routed:
        # The two-frame path is align AND fuse in one: it elects a frame pair per
        # region, warps each member by one rigid transform, fuses the pair, and
        # stitches. It therefore replaces both stages above for this stack; the
        # shipped alignment above still ran, and its report is what routed here.
        from .twoframe import fuse_twoframe

        log("building the two-frame composite ...")
        twoframe_fused, tf_report = fuse_twoframe(sources, harden=harden)
        log(f"two-frame: pairs {tf_report['pairs']}, frames used "
            f"{tf_report['frames_used']} of {len(sources)}, "
            f"{tf_report['refusals']} layer(s) refused by the validity gate, "
            f"largest layer shift {tf_report['max_layer_shift']:.1f} px "
            f"(licence {tf_report['shift_licence']:.1f} px)")
        # SECOND HALF OF THE ROUTING RULE. A composite that had to translate an
        # elected layer further than the arc's refinement scale is re-registering,
        # not refining — the regime F109 named in advance as this architecture's
        # failure case, and the one where its own disocclusion refusal withdraws
        # the very member it elected (measured on large-motion: the sharp,
        # correctly-fitted playing-card box is withdrawn over 91% of its pair and
        # comes back reference-defocused). The composite is discarded and the
        # shipped output stands.
        routed = bool(tf_report["within_licence"])
        if not routed:
            twoframe_fused = None
            log("fusion path: shipped depth-bin — the two-frame composite was "
                "built and DECLINED: it had to re-register a layer beyond the "
                "refinement scale, where its own refusal withdraws the object it "
                "was serving")

    if routed:
        log("fusion path: TWO-FRAME — a stranded object, served within the "
            "architecture's displacement licence")
        fused = twoframe_fused
    elif method == "max":
        log(f"computing focus maps (measure={focus_method}) ...")
        fmaps = _focus_maps(images, focus_method)
        log("fusing (per-pixel maximum sharpness) ...")
        fused, index_map = fuse_max(images, fmaps)
        if debug_dir:
            for name, fm in zip(names, fmaps):
                stem = os.path.splitext(name)[0]
                fio.save_image(os.path.join(debug_dir, f"focus_{stem}.png"), _normalize_map(fm))
            fio.save_image(
                os.path.join(debug_dir, "selection.png"),
                _colorize_selection(index_map, len(images)),
            )
    elif method == "pyramid":
        log("fusing (Laplacian pyramid) ...")
        fused = fuse_pyramid(images, levels=levels)
        if debug_dir:
            # Focus maps aren't used by the pyramid method, but dump them anyway
            # so the sharpness of each frame is visible alongside the result.
            for name, fm in zip(names, _focus_maps(images, focus_method)):
                stem = os.path.splitext(name)[0]
                fio.save_image(os.path.join(debug_dir, f"focus_{stem}.png"), _normalize_map(fm))
    elif method == "decision":
        log("fusing (guided-filter decision map) ...")
        fused, weights = fuse_decision(images, focus_method=focus_method, harden=harden,
                                       weight_scale=weight_scale, return_weights=True)
        if debug_dir:
            # The refined per-frame weight maps are the heart of this method —
            # dump them so the (clean, edge-aligned) selection is visible.
            for name, wmap in zip(names, weights):
                stem = os.path.splitext(name)[0]
                fio.save_image(os.path.join(debug_dir, f"weight_{stem}.png"), _normalize_map(wmap))
    elif method == "blend":
        log("fusing (guided multi-band blend) ...")
        fused, weights = fuse_blend(
            images, focus_method=focus_method, levels=levels, harden=harden,
            weight_scale=weight_scale, return_weights=True
        )
        if debug_dir:
            for name, wmap in zip(names, weights):
                stem = os.path.splitext(name)[0]
                fio.save_image(os.path.join(debug_dir, f"weight_{stem}.png"), _normalize_map(wmap))
    elif method == "perband":
        log("fusing (per-band edge-aware) ...")
        fused = fuse_perband(images, harden=harden, usable=usable)
    else:
        raise ValueError(
            f"Unknown method {method!r}; use 'blend', 'perband', 'decision', 'pyramid', or 'max'."
        )

    if enhance == "auto" and method == "perband" and routed:
        # F56 licenses the enhance specialists for the fused output of frames the
        # caller can point at; a stitched per-region composite is not that, and
        # the licence does not transfer. Skipped rather than silently applied.
        log("enhance: skipped on the two-frame route (not licensed for a "
            "stitched composite)")
    elif enhance == "auto" and method == "perband":
        from .enhance import enhance as _enhance
        fused, rep = _enhance(images, fused, harden=harden, log=log)
        if rep["veil_fired"] or rep["recon_fired"]:
            log(f"enhance: veil={rep['veil_fired']} recon={rep['recon_fired']} "
                f"(bridge={'yes' if rep['bridge'] else 'no'})")

    if reconstruct_boundaries:
        from .reconstruct import reconstruct_boundaries as _recon
        log("reconstructing boundary bands (experimental) ...")
        fused = _recon(images, fused)

    if boundary_out:
        from .reconstruct import stack_boundary
        log("computing stack boundary map ...")
        b = stack_boundary(images)
        fio.save_image(boundary_out, (np.clip(b, 0, 1) * 255.0).astype(np.uint8))
        log(f"wrote boundary map {boundary_out}")

    if depth_out:
        log("computing depth-from-focus map ...")
        d = depth_from_focus(images, focus_method=focus_method)
        fio.save_image(depth_out, (d * 255.0).astype(np.uint8))
        log(f"wrote depth map {depth_out}")

    fio.save_image(output, fused)
    log(f"wrote {output}")
    return fused
