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
        choices=["blend", "decision", "pyramid", "max"],
        default="blend",
        help="Fusion method (default: blend — guided multi-band blending; "
        "halo-free AND multi-scale). Others: 'decision' (single-scale guided), "
        "'pyramid' (Laplacian pyramid), 'max' (per-pixel argmax).",
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
        choices=["laplacian", "gradient"],
        default="laplacian",
        help="Sharpness operator (default: laplacian).",
    )
    p.add_argument(
        "--levels",
        type=int,
        default=None,
        help="Pyramid levels for the pyramid method (default: auto).",
    )
    p.add_argument(
        "--debug-dir",
        default=None,
        help="Directory to write intermediate visualizations.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Print progress.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(
            inputs=args.inputs,
            output=args.output,
            method=args.method,
            align=not args.no_align,
            align_motion=args.align_motion,
            focus_method=args.focus_method,
            levels=args.levels,
            debug_dir=args.debug_dir,
            verbose=args.verbose,
        )
    except Exception as e:  # noqa: BLE001 - surface a clean message to the user
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
