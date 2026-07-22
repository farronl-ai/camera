"""Command-line front end for focusstack.

Examples:
    focusstack images/*.jpg -o stacked.png
    focusstack shots/ -o out.png --method max --debug-dir debug -v
"""

from __future__ import annotations

import argparse
import sys

from .pipeline import run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="focusstack",
        description="Focus stacking: merge multiple differently-focused frames "
        "into one all-in-focus image.",
    )
    p.add_argument("inputs", nargs="+", help="Input images: files, globs, or a directory.")
    p.add_argument("-o", "--output", required=True, help="Output image path.")
    p.add_argument(
        "--method",
        choices=["perband", "blend", "decision", "pyramid", "max"],
        default="perband",
        help="Fusion method (default: perband — per-band edge-aware: multi-scale "
        "decision + halo-free, best all-rounder across resolutions). Others: "
        "'blend' (single-scale guided multi-band; strongest on hard low-res halo "
        "boundaries), 'decision' (image-space guided), 'pyramid', 'max'.",
    )
    p.add_argument(
        "--no-align",
        action="store_true",
        help="Skip registration (use if frames are already pixel-aligned).",
    )
    p.add_argument(
        "--align-motion",
        choices=["translation", "euclidean", "affine", "homography"],
        default="affine",
        help="Geometric model for alignment (default: affine).",
    )
    p.add_argument(
        "--focus-measure",
        dest="focus_method",
        choices=["laplacian", "gradient", "tenengrad", "mod_laplacian", "content_aware"],
        default="content_aware",
        help="Sharpness operator (default: content_aware — routes "
        "laplacian<->mod_laplacian per pixel by local contrast; non-regressing on "
        "clean data, better on smooth content).",
    )
    p.add_argument(
        "--levels",
        type=int,
        default=None,
        help="Pyramid levels for the pyramid method (default: auto).",
    )
    p.add_argument(
        "--harden",
        type=float,
        default=0.0,
        help="Defocus-spread rejection strength 0..1 for blend/decision (default: 0 off). "
        "Hardens the blend toward hard-selection where one frame is confidently sharpest, "
        "so out-of-focus 'spread' can't bleed into bright/thin structures.",
    )
    p.add_argument(
        "--debug-dir",
        default=None,
        help="Directory to write intermediate visualizations.",
    )
    p.add_argument(
        "--weight-scale",
        type=float,
        default=1.0,
        help="Compute the (smooth) fusion weights at this fraction of resolution "
        "then upsample — a high-res speedup (default 1.0 = full). 0.5 is ~quality-"
        "neutral; focus/confidence stay full-res so thin structures are preserved.",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Speed preset for high-res: image-space decision fusion + "
        "--weight-scale 0.5 (~1.5x faster, quality-neutral-or-better). "
        "Overrides --method/--weight-scale unless those are set explicitly.",
    )
    p.add_argument(
        "--no-normalize-exposure",
        action="store_true",
        help="Disable per-frame exposure/WB drift correction (on by default; "
        "defocus preserves the mean, so frame-mean differences are exposure, not "
        "focus — near-identity on undrifted stacks).",
    )
    p.add_argument(
        "--enhance",
        choices=["auto", "off"],
        default="auto",
        help="Gated specialist enhancement (default auto): contour reconstruction "
        "and — when a torch bridge environment is available — wide-occluder veil "
        "correction, each firing only where its outcome-trained gate predicts a "
        "win. Identity when gates stay silent. 'off' disables. perband method only.",
    )
    p.add_argument(
        "--reconstruct-boundaries",
        action="store_true",
        help="EXPERIMENTAL: re-render occlusion-boundary bands as fresh matte "
        "composites (sharper hard edges where a defocused occluder veils the "
        "background). Validated on matte-occlusion benchmarks; may alter other "
        "content — off by default.",
    )
    p.add_argument(
        "--depth-out",
        default=None,
        help="Also write a depth-from-focus map (uint8 PNG; near=dark, far=bright "
        "for a near-to-far ordered stack). A free byproduct of the fusion decision; "
        "more frames give finer depth quantization.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Print progress.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    method, weight_scale = args.method, args.weight_scale
    if args.fast:  # speed preset — honored unless the user set these explicitly
        if method in ("perband", "blend"):
            method = "decision"
        if weight_scale == 1.0:
            weight_scale = 0.5
    try:
        run(
            inputs=args.inputs,
            output=args.output,
            method=method,
            align=not args.no_align,
            align_motion=args.align_motion,
            focus_method=args.focus_method,
            levels=args.levels,
            harden=args.harden,
            weight_scale=weight_scale,
            normalize_exposure=not args.no_normalize_exposure,
            reconstruct_boundaries=args.reconstruct_boundaries,
            enhance="off" if args.fast else args.enhance,
            depth_out=args.depth_out,
            debug_dir=args.debug_dir,
            verbose=args.verbose,
        )
    except Exception as e:  # noqa: BLE001 - surface a clean message to the user
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
