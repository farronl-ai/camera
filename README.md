# focusstack

Merge multiple photos of the same scene — each focused at a different depth —
into a single image that is sharp everywhere. This is **focus stacking**, also
called producing an **extended depth-of-field (EDOF)** image. It's used in macro,
product, microscopy, and landscape photography, wherever one shot can't hold the
whole scene in focus.

> 📸 **[Progress Showcase](docs/SHOWCASE.md)** — the engine explained visually:
> real before/after photos with zooms, the intuition and the math behind each
> idea, the evidence across every validated regime, and the road ahead.
>
> 🔬 **[Owner Inspection Lab](docs/INSPECTION.html)** — the engineering view:
> all current physical-stress fires plus ordinary real-photo stacks, every input
> frame, aligned/normalized inputs, base/output slider, visibility/support/front
> reconstruction maps, GT-only error maps where truth exists, exact regional
> metrics, adversarial crops, and explicit optional coordinate feedback.

## How it works

The pipeline has three stages, each a small signal/image-processing problem:

1. **Registration / alignment** (`align.py`) — refocusing a lens slightly changes
   magnification ("focus breathing"), and the camera may shift. Unaligned frames
   fuse into ghosts, so every frame is warped onto a common reference frame using
   OpenCV's ECC image alignment.
2. **Focus measure** (`focus.py`) — sharpness is high-frequency energy. For each
   pixel we score "how in focus is this here?" from the Laplacian (2nd derivative),
   gradient, Tenengrad, or modified Laplacian, pooled over a small window. The
   default `content_aware` operator *routes* per pixel between the Laplacian
   (best on texture) and the modified Laplacian (best on smooth low-contrast
   surfaces, where the signed Laplacian's curvatures cancel) by local contrast —
   non-regressing on clean data, better on smooth content.
3. **Fusion** (`fusion.py`) — combine the sharp parts:
   - `perband` (default): **per-band edge-aware fusion** — makes the focus decision
     *and* an edge-aware guided weight at *each* pyramid band (not one global
     weight), so the decision is multi-scale like `pyramid` *and* halo-free like
     `blend`. A fixed small guided radius per band means the effective scale grows
     with resolution automatically — multi-scale by construction, no magic numbers.
     Best all-rounder across resolutions (validated on ground truth at low *and*
     high res, and by disagreement-guided visual inspection on real optical defocus).
   - `blend`: **guided multi-band blending** — one edge-aware weight map (as in
     `decision`) applied per Laplacian-pyramid band (Burt & Adelson multiresolution
     blending). Halo-free and seamless, but its *decision* is single-scale; the
     guided radius/pooling are resolution-adaptive (≈ the circle of confusion).
   - `decision`: **guided-filter decision-map fusion** — decide per pixel which frame
     is in focus, refine with a *guided filter* so it snaps to real edges, blend in
     image space. Crisp and halo-free, but single-scale. (Used by `--fast`.)
   - `pyramid`: **Laplacian-pyramid fusion** — keep the highest-energy content per
     band, collapse back. Seamless and intrinsically multi-scale, but can ring
     (halo) around thin high-contrast objects at a focus boundary.
   - `max`: per pixel, copy from the sharpest frame. Simple; crisp but speckly.

   The ladder, in one line each: `max` decides per pixel (speckle); `decision`
   cleans that decision edge-aware (single-scale); `pyramid` decides per band
   (multi-scale, but hard-select → halos); `blend` = edge-aware decision + multiband
   reconstruction (but one decision for all bands); `perband` = edge-aware decision
   *per band* — multi-scale **and** halo-free, which is why it's the default.

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

# Alternative fusion methods and options.
focusstack images/*.png -o out.png --method pyramid --levels 5
focusstack images/*.png -o out.png --method max --focus-measure gradient
# Skip alignment for already-registered frames (avoids a needless resample).
focusstack images/*.png -o out.png --no-align

# Defocus-spread rejection: keeps bright/thin structures (dots, wires, hairs)
# crisp instead of letting an out-of-focus frame's "spread" bleed in as a dim
# blob. Recommended for scenes with bright points or fine structures.
focusstack images/*.png -o out.png --harden 0.5

# High-res speed preset (~1.5x faster, quality-neutral-or-better): image-space
# decision fusion + weights computed at half-resolution then upsampled (the
# weights are smooth, so this is near-lossless; thin structures stay full-res).
focusstack big/*.png -o out.png --fast --harden 0.5

# Default auto enhancement can re-render licensed thin contours. For exactly
# two frames it can also recover a licensed giant occlusion veil: opaque
# ownership is selected as discrete focused-owner regions, rear recovery is
# forbidden inside foreground/boundary support, and every failed gate returns
# the ordinary fusion byte-for-byte.
focusstack bracket/*.png -o out.png --enhance auto
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

Validated on the standard MFIF benchmark (real Lytro-style focus pairs): all three
methods reach the achievable sharpness upper bound globally, but only `decision`
stays clean at focus boundaries. Planned directions:

- **Alignment**: feature-based / multi-scale ECC for large shifts.
- **Quality**: tune the guided-filter radius/eps per scene; per-frame denoising;
  small-region cleanup on the decision map before refinement.
- **Leading edge**: a learning-based fusion backend (e.g. IFCNN / U²Fusion /
  MFF-GAN) behind the same CLI — that's where current multi-focus fusion research
  lives.

## Layout

```
src/focusstack/   package (align, focus, fusion, pipeline, cli)
scripts/          synthetic stack generator
tests/            pytest suite
```
