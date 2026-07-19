"""focusstack — focus stacking / extended depth-of-field image synthesis.

Given several photos of the same scene, each focused at a different depth,
this package synthesizes one image that is sharp everywhere.

The pipeline has three stages (see the correspondingly named modules):
    1. align  — register frames onto a common coordinate frame (focus breathing)
    2. focus  — measure local sharpness (high-frequency energy) per pixel
    3. fusion — combine the sharpest content from each frame

`run()` (from `pipeline`) ties them together; `cli` is the command-line front end.
"""

from .pipeline import run

__all__ = ["run"]
__version__ = "0.1.0"
