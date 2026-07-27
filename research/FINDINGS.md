# Current findings

This is the compact scientific state of the project. It is not a chronological
lab notebook; superseded experiments and their reports remain in Git history.
Read `MISSION.md` first, then this file, `OCCLUSION_FORMATION.md`, and
`STATE.md`.

## F83 — Occlusion-edge blur orders the boundary correctly, and ordering does not help

An occlusion boundary is the near object's own silhouette, so its sharpness
follows the FOREGROUND through the sweep: crisp in the frame that focuses the
occluder, blurred in the frame that focuses what lies behind. The focus index
read *on* a contour therefore names the occluder's depth — ordinal front/back
evidence that focus magnitude alone cannot supply (Marshall et al., JOSA A 1996).

The cue is real and was measured directly. Within 4 px of the true silhouette,
the background reads the FOREGROUND's focal frame (1) rather than its own (4);
past 8 px it recovers its own. Everything the cue promises is confirmed:

- **Localization**: the depth map alone cannot place a contour — its steps sit a
  median of 32 px from the true silhouette, because a 0.10 threshold is half a
  focal step on a 6-frame stack and depth noise swamps it. Intensity edges keep
  only those whose two sides, sampled 10 px out along the edge normal, differ by
  >= 1.5 focal frames: median error 4.8 px, 61% within 8 px.
- **Ordering**: per contour the cue is 77.3% reliable, far too noisy to decide
  ownership pixel by pixel. But a monotone sweep shares ONE bit — near is either
  the low index or the high one — and voting that bit across thousands of contour
  pixels settles it correctly. Given the bit, "lower index is nearer" is 100%
  accurate, and the resulting front mask agrees with the true near plane 91.1%.

And it does not help, which is the finding. Refusing only the background side is
WORSE than refusing both:

| refusal | GT-SSIM | withheld |
|---|---:|---:|
| none | 0.966007 | 0% |
| background side only | 0.971433 | 4.84% |
| foreground side only | 0.975415 | 7.03% |
| both sides | **0.978509** | 11.86% |

The premise was that the occluder is opaque and present in every frame, merely
displaced, so it loses nothing and needs no refusal. That is true of the
*surface* and false of the *observation*. When the occluder is out of focus its
own matte is blurred, so its boundary pixels are foreground/background mixtures
carrying background colour onto the object — unusable for exactly the same
reason the uncovered ribbon is. The foreground side turns out to contribute
MORE of the gain than the background side it was supposed to be spared for.

Two-sided refusal (F82) therefore stands. `research/occlusion_order.py` keeps the
contour localizer and the ordering vote, both validated, since they are reusable
and the negative must stay reproducible; neither is in the runtime path.

Consequence worth carrying: near a defocused silhouette, BOTH surfaces are
compromised, and the reason differs on each side — missing correspondence behind,
matte mixing in front. Any future boundary work should assume the whole
neighbourhood is suspect rather than just the far half.

## F82 — Parallax uncovers scene, and the uncovered ribbon must be discarded

Lateral camera motion does not merely shift a near object; it swings it across
the background, revealing scene on one side and hiding it on the other. Those
pixels have no correspondence in that frame at all, so no warp can supply them
and any value there is interpolated off the wrong surface. Fusion then mixes a
bottle's outer edge with what was behind it. F80 discarded invented data at the
outer border; this is the same rule applied where the invention happens in the
interior, as per-pixel validity rather than a rectangle.

On the analytic parallax factory this is worth as much again as the alignment
fix itself:

| | GT-SSIM | PSNR | withheld |
|---|---:|---:|---:|
| global affine only | 0.875425 | 23.343 | — |
| depth-aware alignment | 0.966007 | 29.660 | 0% |
| **+ disocclusion refusal** | **0.978509** | **31.816** | 11.86% |

Real sweeps at the shipped default (4 bins), near-content residual and the
fraction each frame withholds:

| Sweep | near residual | withheld/frame | Q_SSIM |
|---|---:|---:|---:|
| kitchen, 12 frames | 2.190 → 0.544 px | 4.32% | 0.911196 → 0.910074 |
| large-motion, 14 frames | 1.540 → 0.907 px | 4.09% | 0.942354 → 0.928407 |
| small-motion, 14 frames | 0.469 → 0.227 px | 0.94% | 0.981779 → 0.980734 |
| zero-motion, 14 frames | 0.046 → 0.031 px | 0.11% | 0.986779 → 0.986336 |

The withheld fraction tracks how much the camera actually moved, which is the
behaviour to demand: a still stack discards essentially nothing.

Four properties had to hold, and three of them were wrong on the first attempt:

1. **The test must be geometric, never photometric.** In a focus stack frames
   legitimately disagree wherever defocus differs, so "these pixels disagree"
   cannot separate occlusion from blur.
2. **Derive it from the MEASURED per-region displacement, not the applied
   field.** Uncovering is a fact about the scene, so it happened whether or not
   the correction modelled it. Reading it off the smoothed, stretch-limited
   field reported almost nothing on the kitchen sweep (0.06% withheld) precisely
   where the visible defect is.
3. **The ribbon is as wide as the step is tall.** A foreground moving Q px
   relative to its background uncovers a Q px strip. Testing at one scale with a
   fixed tolerance condemned 38.6% of the large-motion frame; testing at a ladder
   of radii, where radius r requires a step of at least r, brought the same
   sweep to 4.09% without losing the real ribbons.
4. **Only genuine depth discontinuities qualify.** A continuously receding
   surface hides nothing, but binning turns it into a staircase of displacements.
   The gate must be an absolute depth jump; expressing it as a fraction of bin
   width made it loosen as bins increased, which is backwards, and it was
   effectively inert until fixed.

Bin count moved to 4 on the same evidence: kitchen's near residual improves
1.481 → 0.544 px going from 3 to 4 bins. Registration keeps improving at 5 on
the two 14-frame sweeps (large-motion 0.300 px) but withholds more, and the
optimum is scene-dependent, so 4 is a compromise rather than a discovered value.

### F82a — GT and no-reference metrics disagree about refusal, and GT is right

Q_SSIM prefers *less* masking on every real sweep, and it is wrong to. It scores
the fused image against its locally sharpest source, so deliberately refusing an
untrustworthy-but-sharp source always looks like a loss to it. On the factory,
where truth exists, refusal is worth +0.0125 GT-SSIM and +2.2 dB. This is F81a
again with a sharper edge: the no-reference metrics cannot see the difference
between using data and using data that should not exist.

### F82b — Rejected: median-stabilizing the depth map before the step test

Depth-from-focus speckle scatters some refusals onto surfaces that never
occluded anything, so a median filter ahead of the step test looked obviously
right. It made concentration on the true silhouette *worse* (2.81x -> 2.07x) and
withheld more. Not kept. The ribbon is currently ~2.8x concentrated on the
silhouette, and the residual scatter is unexplained.

## F81 — Depth-dependent parallax needs a depth-binned field, not a better global warp

F79 and F80 handled the *consequences* of unresolved parallax. This is the
geometry itself. A handheld rotation pivots the device, not the entrance pupil,
so the camera centre translates and displacement scales with inverse depth. One
global warp splits the difference: on all three moving phone sweeps the residual
translation inside near content was 2.0–2.5x the residual in far content, which
is the inverse-depth signature measured directly rather than inferred.

The default now adds a second pass on top of the global warp: depth bins from
the stack's own depth-from-focus map, one translation-only ECC correction per
bin, blended into a single dense coordinate field by edge-aware memberships and
resampled exactly once. Registration improves on every moving sweep:

| Sweep | near residual | far residual | Q_SSIM | sharp | align |
|---|---:|---:|---:|---:|---:|
| synthetic parallax + GT | 2.735 → 0.481 px | 1.245 → 0.402 px | GT 0.8754 → **0.9660** | — | — |
| kitchen, 12 frames | 2.172 → 0.981* | 0.855 → 0.916 | 0.911430 → 0.909417 | 19.36 → 19.27 | 1.2 → 2.0 s |
| large-motion, 14 frames | 1.539 → 0.981 | 0.729 → 0.358 | 0.942463 → 0.940672 | 18.21 → 18.51 | 1.3 → 4.5 s |
| small-motion, 14 frames | 0.470 → 0.221 | 0.379 → 0.217 | 0.981784 → 0.982400 | 15.45 → 15.47 | 0.4 → 3.3 s |
| zero-motion, 14 frames | 0.046 → 0.034 | 0.026 → 0.040 | 0.986796 → 0.986316 | 16.28 → 16.29 | 0.4 → 3.3 s |

*kitchen near residual 2.172 → 1.453.

The isolated synthetic case carries the verdict because it is the only one with
GT, and because a stack that differs by one global transform cannot test this
pass at all — it must be built with near and far content moving by different
amounts. +0.0906 GT-SSIM and +6.3 dB. Its remaining error sits on silhouettes,
which is genuine disocclusion: parallax uncovers background that no
single-valued warp can recover.

Four things had to be right, and each was found by a failure:

1. **Bin edges belong at depth-histogram valleys, not quantiles.** Equal-population
   edges cut through the middle of one physical object, giving it a 13 px step
   across its own surface: a visible seam and ghosted strip inside the book on
   the large-motion sweep. Valleys split where the scene separates.
2. **The field must be stopped from stretching content.** A sampling field may
   transport pixels freely, but a large correction changing quickly across space
   smears geometry the camera never saw. Relaxing displacement wherever its local
   gradient exceeds 0.10 *raised* probe GT-SSIM (0.9579 → 0.9628), so the stretch
   was pure damage. Membership width cannot substitute: narrowing it makes stretch
   worse (5.9 vs 2.7 at the extremes), because the same jump crosses fewer pixels.
3. **Refusal at every level.** Untextured bins, underpopulated bins, diverged
   fits, and sub-three-frame stacks all fall back to the global warp, and a frame
   earning no correction is byte-identical. The zero-motion sentinel is left alone.
4. **Bin count stopped being a tuning knob** once edges came from valleys: 3 and 4
   requested bins both resolve to the same 2 real bins on the probe.

Cost is 1.5–3.5x alignment time, which is a small share of a full run.

### F81a — No-reference metrics cannot adjudicate an alignment change

Q_SSIM compares the fused image to its *locally sharpest source*, and alignment
changes what the sources are. Scoring each fused output against the other
variant's source set collapses both to ~0.72–0.82, and the depth-aware output
scores marginally *higher* on the global variant's sources than the reverse.
Within-variant Q_SSIM therefore partly measures self-consistency with a possibly
misregistered stack, which a misaligned stack can win. The real-data Q_SSIM
deltas above (±0.002) are not evidence either way; the registration residuals,
the GT probe, and disagreement-guided crops are. Localizing the one dissent that
looked real (large-motion, −0.0137) is what exposed the quantile-edge seam, so
the metric earned its keep as a *pointer* while being useless as a verdict.

### F81b — Tested and rejected: parametric depth-to-displacement models

Displacement is proportional to 1/Z, so a model linear in the depth proxy looks
principled. It loses: probe GT-SSIM 0.9313 vs 0.9565 for bins, with higher
stretch. The focus-winner index is monotone in inverse depth but not affine in
it, and multiplying a noisy continuous proxy by a ~20 px coefficient turns every
depth-map wiggle into a displacement wiggle. Binning quantizes that noise away
and assumes nothing about the mapping.

A full alternating estimator — camera rotation, translation, breathing scale,
depth, *and* the depth-to-parallax calibration, each refined from the others
across iterations — is implemented behind `depth_model="joint"`. It converges
(mean correction 1.13 → 1.11 → 0.35 px) and achieves the best registration of
any variant (large-motion near residual 0.568 px vs 0.981 binned, 1.539 global),
but it is **not promotable**: it moves the zero-motion sentinel from 0.046 to
0.247 px, inventing motion where there is none, and scores worst on Q_SSIM
everywhere. The diagnosed wall is the observation model, not the solver — tile
shifts are sound (84% agree within 1 px with independent ECC), yet the
rigid-motion-plus-depth model explains only ~25–50% of them, so the fit
generalizes poorly to pixels whose depth differs from the tiles'.

## F80 — Alignment output is the all-frame observed footprint

A rotated or translated frame does not observe the same scene rectangle as the
reference. The old aligner nevertheless filled missing warp regions by
reflection, allowing invented border pixels to compete as focus evidence. The
default now warps an explicit validity mask with every frame, intersects those
masks across all N frames, and crops the complete stack to the largest
axis-aligned rectangle containing only all-valid pixels. Bilinear edge samples
are rejected too: validity requires that interpolation never touched padding.

This is deliberately an intersection, not a union with per-pixel fallback.
Every output pixel has a real observation in every focal frame, so selection
confidence remains comparable and later reconstruction never mistakes missing
coverage for defocus. On the 12-frame kitchen sweep the common footprint is
`727×502` from `774×518` inputs. Inspector feedback coordinates now use that
cropped output geometry.

## F79 — Fragmented N-frame ownership routes to one cross-band decision

The real 12-frame kitchen sweep was not a veil-recovery failure. Raw framing
shifts, focus breathing, and depth-dependent parallax remain after a single
global affine alignment, and independent per-band decisions could therefore
select different misregistered frames at different frequencies. Pure rotation
about the entrance pupil would be one depth-independent homography, but normal
handheld rotation pivots around the device/body and includes camera-center
translation; image displacement then varies approximately with inverse scene
depth. The visible symptom was scattered focus ownership and doubled
text/edges.

The default now measures the fraction of locally isolated focus winners whose
lead over the runner-up is weak. This is label-invariant and does not penalize
decisive real depth boundaries. Above `0.115`, N-frame `perband` obtains one
shared guided decision, snaps it to a hard region-coherent frame choice, and
uses that choice across all pyramid bands. Fine detail therefore comes from
exactly one frame while the Gaussian weight pyramid feathers only coarser
transitions. Stable stacks and every two-frame stack retain the original
per-band path.

Three real phone sweeps were the deliberately small validation set:

| Sweep | Instability | Route | Q_SSIM old → new |
|---|---:|---|---:|
| kitchen, 12 frames | 0.1250 | coherent | 0.889760 → 0.912199 |
| zero-motion, 14 frames | 0.0880 | per-band identity | 0.986347 → 0.986347 |
| large-motion, 14 frames | 0.1220 | coherent | 0.930018 → 0.943014 |

The first soft shared-weight attempt improved the metric and decision map but
still visibly doubled the Lubriderm label, cat, and vertical contours. It was
not accepted as the fix. Hard direct copying removed those ghosts but made
jagged seams. The final coherent multiband composition removes the doubled
images without those hard-copy seams and restores edge energy
(`46.12 → 55.07` kitchen, `46.62 → 55.40` large-motion). The kitchen inspector
shows this final output and its actual decision. Dense optical flow and
focus-breathing compensation remain open alignment work. A more flexible
single global homography is not the answer; the geometric successor must allow
depth-varying motion through regularized dense flow or depth-binned local
transforms.

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
