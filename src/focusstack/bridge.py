"""Bridge runner — torch-dependent helpers in an EXTERNAL python, via subprocess.

The main package never imports torch. Semantic helpers (monocular depth,
segmentation masks) run in a separate environment (e.g. a `.venv312` with CPU
torch) through the scripts in `focusstack/bridges/`. Resolution order for that
environment: explicit argument > FOCUSSTACK_BRIDGE_PYTHON env var > a
`.venv312/bin/python` found at the working directory or the repository root.

Every function degrades gracefully: when no bridge environment exists (or a
bridge run fails), callers receive None and MUST fall back to identity
behavior — this is what makes bridge-dependent stages safe to enable by
default.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def find_bridge_python(explicit: str | None = None) -> str | None:
    """Locate the external python that hosts torch bridges, or None."""
    if explicit:
        return explicit if os.path.exists(explicit) else None
    env = os.environ.get("FOCUSSTACK_BRIDGE_PYTHON")
    if env:
        return env if os.path.exists(env) else None
    for root in (Path.cwd(), Path(__file__).resolve().parents[2]):
        cand = root / ".venv312" / "bin" / "python"
        if cand.exists():
            return str(cand)
    return None


def run_bridge(kind: str, image_path: str, python: str | None = None,
               timeout: int = 900) -> str | None:
    """Run the `kind` bridge ('depth' | 'masks') on image_path.

    Returns the produced .npy path, or None on any failure (no environment,
    crash, timeout) — never raises.
    """
    outputs = run_bridge_many(
        kind,
        [image_path],
        python=python,
        timeout=timeout,
    )
    return outputs[0] if outputs else None


def run_bridge_many(
    kind: str,
    image_paths: list[str],
    python: str | None = None,
    timeout: int = 900,
) -> list[str] | None:
    """Run one bridge process over several images, or return ``None``.

    Torch/model import dominates the semantic-mask bridge startup.  Loading the
    model once also lets enhancement inspect the mixed base and both captured
    frames without tripling that cost.  The result is all-or-nothing so callers
    never silently pair outputs from different bridge runs.
    """
    if not image_paths:
        return []
    py = find_bridge_python(python)
    if py is None:
        return None
    script = Path(__file__).parent / "bridges" / f"{kind}_bridge.py"
    if not script.exists():
        return None
    try:
        subprocess.run(
            [py, str(script), *(str(path) for path in image_paths)],
            check=True,
            timeout=timeout,
            capture_output=True,
        )
    except Exception:
        return None
    suffix = "depth" if kind == "depth" else "masks"
    outputs = [f"{path}.{suffix}.npy" for path in image_paths]
    return outputs if all(os.path.exists(path) for path in outputs) else None
