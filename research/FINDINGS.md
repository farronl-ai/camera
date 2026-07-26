# Current findings

This is the compact scientific state of the project. It is not a chronological
lab notebook; superseded experiments and their reports remain in Git history.
Read `MISSION.md` first, then this file, `OCCLUSION_FORMATION.md`, and
`STATE.md`.

## The result we are building

The target is the latent physical scene, not an ideal blend of what the camera
recorded. Selection among focal frames is the safe floor. Recovery above that
floor is allowed only when it is anchored in observations and survives the
forward formation model. Plausible but unobserved synthesis is out of scope.

Public and common validation standards are insufficient by themselves:

- pseudo-ground truth made by another focus-stacker inherits that stacker's
  ceiling and can score true veil removal as damage;
- source-similarity metrics penalize a correct recovery precisely because the
  recovered scene differs from every corrupted source;
- global SSIM can dissent from large, spatially coherent improvements and can
  hide narrow boundary failures;
- synthetic benchmarks are useful only when their renderer obeys the physical
  invariant being tested.

The evidence lattice is therefore: analytic latent GT where available,
observation-domain re-rendering everywhere, physical partition metrics,
changed-pixel accounting, and direct visual inspection for artifacts and edit
location. The eye does not certify invented detail.

## Shipped floor: ordinary focus stacking

- `perband` is the default fusion method. It makes an edge-aware focus decision
  at every pyramid band, retaining pyramid's scale coverage without its hard
  boundary selection.
- Exposure/white-balance normalization is default-on because realistic drift
  materially harms fusion and mean-based normalization is blur-invariant.
- Confidence hardening protects thin bright structures and rejects defocus
  spread where one frame is decisively sharp.
- The `--fast` path is a documented CPU tradeoff, not quality-neutral.
- Real handheld sweeps exposed a separate regime: residual motion can make the
  single-decision `blend` method more robust than `perband`. Alignment and
  focus breathing remain practical limits.

These are the selection floor. They do not solve cross-boundary coefficient
contamination or recover a rear surface attenuated by a foreground veil.

## Load-bearing recovery findings

### 1. Defocus does not make an opaque foreground transparent

The original object-occlusion factory violated this. Ordinary radius blur of a
composited image pulled hidden background inward beneath foreground ownership.
That trained and graded the solver against impossible inputs.

The current `one_sided_opaque_v2` renderer separates:

- pre-antialias hard ownership;
- foreground radiance;
- outward foreground spread;
- rear transmission/coverage;
- antialiased boundary support; and
- latent all-in-focus GT.

Changing only the hidden background changes exactly zero hard-owned input
pixels. Foreground radiance may spread outward; rear radiance may not bleed
inward through complete opaque coverage. Thin or all-veil geometry is a named
stress class, not evidence that substantial opaque objects are transmissive.

### 2. Ownership is discrete; veil recovery is ordered

A focused foreground observation is positive front-surface evidence. It cannot
be revoked because the other frame weakly suggests rear structure. The
pipeline therefore:

1. associates same-object semantic proposals across focal frames;
2. selects locally supported focused-owner regions;
3. completes only observed graph-connected continuations and corroborated tiny
   fragments;
4. hard-selects focused-owner radiance for accepted foreground;
5. excludes rear correction from foreground, its protected boundary, and far
   identity regions; and
6. applies rear recovery only where the focal transformation positively
   reveals rear information.

This repaired the small-black-piece failures. Uncertainty protects possible
foreground; it does not become a license to blend foreground and background,
and it does not invent foreground texture.

### 3. Solve the formation, not the fused image

The old correction-after-fusion family was structurally wrong. Once focal
observations have been mixed into a base image, foreground radiance and rear
transmission are no longer identifiable. A veil specialist receiving that
mixture can reasonably interpret base-stage foreground/background blending as
veil and cannot undo the ownership mistake.

The surviving two-frame path retains both original observations and solves a
paired ordered layer model. Its numerical inverse, regularization, and
multi-model consensus were already strong; the major gains came from repairing
formation, ownership, source attribution, and routing.

### 4. The boundary line was a transmission-model failure

The final faint vertical line on `s29_010` was not best treated as a seam to
blur. Circular absolute filters damaged legitimate image content because their
shape did not follow the veil. Contour-relative continuous feathering improved
integration but could not decide whether a low-frequency dip was true
background or formation residue.

The load-bearing fix is direct ordered inversion near the material boundary:

```text
B_direct = (O_rear - V_front) / T_rear
```

where:

- `O_rear` is the rear-focused captured observation;
- `V_front` is predicted foreground veil radiance in that observation; and
- `T_rear` is the spatially varying rear transmission coefficient.

The coefficient varies across the aperture-coverage transition, and the veil
itself may be brighter or darker than the rear surface. A subtraction-only or
constant-coefficient scheme leaves bright or dark contour lines.

Two details close the visual failure:

- choose the local PSF hypothesis by the lower two-frame forward residual
  rather than averaging incompatible disk/box explanations;
- allow surrounding-trend extrapolation only for a proposed component whose
  re-rendered observation falls below the local detection floor. If the
  component should have survived capture, its absence vetoes extrapolation.

The integration contour is the 10% cross-PSF aperture-coverage transition, not
an arbitrary maximum-radius ring. High-frequency integration operates on the
relative correction field along the veil contour. Low-frequency correction
uses a continuous 75% contour-relative taper. No absolute circular blur is in
the final path.

### 5. Formation state belongs upstream

The base/original-frame stage is where these quantities are identifiable:

- `V_front`;
- `T_rear`;
- local PSF selection or weights;
- per-frame forward residual/disagreement; and
- component-specific observation detection floor.

They should become an explicit formation-state object handed to recovery and
composition. The current implementation is numerically valid because
`recover_giant_veil` still receives both original frames and recomputes the
state internally. The refactor should compute it once upstream; recovery must
never try to infer it from the fused base.

## Current checkpoint

Commit `bf99365` is the visually frozen F78 transmission-boundary checkpoint.
The inspector's right side for `s29_010` was judged near-perfect by the user.
The quick cohort was intentionally limited to `s29_002`, `s29_007`, and
`s29_010`.

Recorded quick-cohort deltas:

| Scene | ΔSSIM | ΔMAE | Interpretation |
|---|---:|---:|---|
| `s29_002` | +0.004768 | -0.6100 | strong direct improvement |
| `s29_007` | -0.001796 | -0.2181 | direct error and visual result improve; SSIM dissents |
| `s29_010` | +0.004015 | -0.2189 | user-validated boundary result |

The broader frozen S29 result before F78 licensed 11 scenes and refused one.
All 11 improved MAE/MSE and every physical partition, with exactly zero rear
application in protected foreground/far regions and exact far identity. Five
improved global SSIM. This is evidence for the narrow two-frame, validated-size
opaque path—not a general camera claim.

The current inspection page contains seven S29 deep cases and six ordinary
real-photo stacks. It shows all original inputs, aligned inputs, base/output,
ownership and rear-application maps, GT-only diagnostics where truth exists,
and explicit region selection. Legacy formation cohorts were removed.

## Retired approaches

- Correction after fusion: cannot recover identifiable layer state.
- Symmetric foreground/background blending: violates ordered visibility.
- Radius blur of a composited opaque image: leaks hidden rear content inward.
- Global confidence as a substitute for component observability: too coarse.
- Circular absolute seam filters: shape-unaware image damage.
- Blind exterior slope extrapolation: invents visible components.
- Hard tuning against one aggregate metric or one diagnosed scene.
- Treating pseudo-GT or source similarity as latent-scene truth.

## What is not yet proven

- F78 needs a fresh, frozen cross-family split; do not tune further on
  `s29_010`.
- Disk and box hypotheses are only controlled rungs. Compound-lens,
  field-dependent, linear-light, sensor-noise, and ISP families are untested.
- Transparent foreground is a separate model with separate latent layers and
  transmission parameters. Do not weaken opaque ownership to approximate it.
- N-frame recovery, multiple occluders, broad CoC range, >1600 px solving, and
  real macro/product truth are open.
- The formation-state handoff is not yet explicit in the pipeline API.
- Alignment/focus breathing remain a dominant real handheld limitation.

## Working rule

Preserve the liked F78 numerical path while testing its scope. When a new case
fails, first determine whether the input formation, ownership geometry, source
attribution, inversion, or integration is wrong. Do not start by tuning a
global threshold or smoothing the symptom.
