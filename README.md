# focusstack

Merge multiple photos of the same scene — each focused at a different depth —
into a single image that is sharp everywhere. This is **focus stacking**, also
called producing an **extended depth-of-field (EDOF)** image. It's used in macro,
product, microscopy, and landscape photography, wherever one shot can't hold the
whole scene in focus.

## How it works

The pipeline has three stages, each a small signal/image-processing problem:

1. **Registration / alignment** (`align.py`) — refocusing a lens slightly changes
   magnification ("focus breathing"), and the camera may shift. Unaligned frames
   fuse into ghosts, so every frame is warped onto a common reference frame using
   OpenCV's ECC image alignment.
2. **Focus measure** (`focus.py`) — sharpness is high-frequency energy. For each
   pixel we score "how in focus is this here?" using the magnitude of the
   Laplacian (2nd derivative) or the image gradient, pooled over a small window.
3. **Fusion** (`fusion.py`) — combine the sharp parts:
   - `max`: per pixel, copy from the sharpest frame. Simple; can show seams/noise.
   - `pyramid` (default): **Laplacian-pyramid fusion** — decompose each frame into
     frequency bands, keep the highest-energy content per band, and collapse back.
     Seamless and halo-resistant.

## Install

Requires Python ≥ 3.10.

```bash
cd camera
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"        # runtime deps + pytest
```

## Usage

```bash
# Basic: stack a folder (or a glob) of frames into one sharp image.
focusstack images/*.jpg -o stacked.png

# See what each stage does (writes aligned frames, focus maps, selection map).
focusstack images/ -o stacked.png --debug-dir debug -v

# Alternative fusion method and options.
focusstack images/*.png -o out.png --method max --focus-measure gradient
focusstack images/*.png -o out.png --no-align --levels 5
```

Run `focusstack --help` for all options. You can also invoke it as
`python -m focusstack ...`, or call `focusstack.run(...)` from Python.

## Try it with no camera

Generate a synthetic multi-focus stack and process it:

```bash
python scripts/make_synthetic_stack.py --out examples/synth -n 4
focusstack examples/synth/frame_*.png -o out/stacked.png --debug-dir out/debug -v
```

Compare `out/stacked.png` against `examples/synth/_ground_truth.png`.

## Tests

```bash
pytest
```

## Roadmap

The classic pyramid pipeline here is a strong, interpretable baseline. Planned
directions:

- **Alignment**: feature-based / multi-scale ECC for large shifts.
- **Quality**: edge-halo suppression, smoother weight maps, per-frame denoising.
- **Leading edge**: a learning-based fusion backend (e.g. IFCNN / U²Fusion /
  MFF-GAN) behind the same CLI — that's where current multi-focus fusion research
  lives.

## Layout

```
src/focusstack/   package (align, focus, fusion, pipeline, cli)
scripts/          synthetic stack generator
tests/            pytest suite
```
