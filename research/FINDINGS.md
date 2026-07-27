# Current findings

This is the compact scientific state of the project. It is not a chronological
lab notebook; superseded experiments and their reports remain in Git history.
Read `MISSION.md` first, then this file, `OCCLUSION_FORMATION.md`, and
`STATE.md`.

## F91 — The magnification was a measurement artifact; the bottle is pure translation

F87, F88 and F90 all rest on one number: the kitchen bottle's width growing
138 → 160 px (14%) across the raw sweep. That number is wrong, and a rigid object
proves it — **the bottle does not grow vertically**. Its top and bottom straddle the
optical axis, so any real magnification must push them apart:

| | frame 8 | frame 9 | frame 10 |
|---|---:|---:|---:|
| raw vertical growth | +1.68 px (1.0064) | +2.77 px (1.0105) | +4.01 px (1.0152) |
| after global affine | −0.39 px (0.9985) | −0.25 px (0.9991) | −0.22 px (0.9992) |

Raw magnification is ~1.5%, not 14%, and the global affine removes it. A rigid
object cannot grow 14% in one axis and 1.5% in the other, so the horizontal width
series came from the left-edge detector failing — the same left edge F87's own
rigidity test had already flagged as unreliable, on the same object.

A validated instrument confirms it. `edge_similarity.fit_region_similarity` recovers
scale and translation from edge-normal displacements; on known-answer warps of the
real frame it returns 1.020 → 1.0200, 1.050 → 1.0499, 0.970 → 0.9692, with
translations to 0.1 px. Applied to the bottle it measures scale 1.003–1.008 with
tx +16.02 at frame 11 (truth +19.2). **The bottle undergoes essentially pure
translation.**

What this retires:
- F87's "14% breathing" — artifact.
- F88's "breathing is upstream and blocks object grouping" — unsupported. The IoU
  improvement measured there was real, but its cause is not established; the "crude
  de-breathing" applied a scale of up to 12.5% derived from the broken width series,
  so whatever helped, it was not removing breathing.
- F90's "depth-dependent magnification is forward camera translation" — the near/far
  comparison used horizontal-only span fits, which are ill-conditioned for this
  object because its edges span ux 111–252 and never approach zero, so `s·ux` is
  nearly constant across it and scale trades against translation. The better-
  conditioned vertical measurement (uy crosses zero) says scale ≈ 1.000.

What survives: **the region-grab-bag problem (F84/F85) is the real blocker**, exactly
as first diagnosed. A bin covering 55% of the frame fits its majority and hands the
bottle +2.3 px where it needs +19.2; motion-driven splitting recovers it (+18.8 px).
The open problem is the factory regression and a principled split gate — not scale.

**Method, and this is the expensive part.** Three findings were built on a
measurement that was never checked against a known answer, on a scene with no GT.
DEVSTYLE §12 rule 1 — written earlier the same day — says exactly not to do that.
The rule was correct, and being written down did not make it applied. Two habits
follow: (a) conceptual work belongs on the factory, where the transform is known,
with the real scene used to confirm rather than to discover; (b) a geometric claim
about a rigid object must be checked on BOTH axes, because rigidity is a free
consistency test and it costs one measurement.

## F90 — The residual magnification is forward camera translation, not breathing

F87/F88 concluded that focus breathing survives the global affine and prescribed
fixing it at the global stage. **The prescription was wrong**, and the measurement
that refutes it is simple: breathing is depth-INDEPENDENT, so if the residual were
breathing it would magnify near and far content equally.

In the aligned kitchen stack, object-level magnification measured with the validated
edge fit:

| frame | bottle (near) | far background |
|---|---:|---:|
| 7 | 1.0085 | 1.0025 |
| 8 | 1.0406 | 1.0122 |
| 9 | 1.0790 | 1.0240 |
| 10 | 1.0845 | 1.0319 |

Near magnifies 2.6–3.3× more than far. That is depth-scaled magnification, which is
the forward-translation (`t_z`) term of camera motion — the hand drifting toward the
scene — not the lens breathing.

Confirmed from the other side by adding a breathing axis to the analytic factory
(`BREATHING_PER_FRAME`): with a true 2.5%/frame magnification applied, the residual
scale measured after the global affine is 1.0013…0.9994. The affine absorbs genuine
breathing essentially completely, on the factory and on the kitchen (residual
±1.5%). What it cannot absorb is the part that varies with depth.

**Consequence: no global stage can fix this**, which retires the F88 next-step. The
per-region model needs a radial SCALE term, and the evidence is direct — fitting one
kitchen region with similarity instead of translation drops its p90 tile residual
from 18.45 px to 2.75 px (frame 9) and 7.75 px to 0.64 px (frame 11).

**And the region must be object-sized before its scale is measurable.** The bottle's
region covers 55% of the frame and fits scale ≈1.0004 while the bottle itself needs
~1.08 — the same majority problem that defeats its translation. So scale belongs in
the region model from the start of the split iteration rather than being added after
objects are found. That is what dissolves the apparent circularity in F88 ("objects
need motion, motion needs objects"): the iteration converges only if each round's
model can represent what the object is actually doing.

Method note: the first comparison of translation against similarity was invalid
because the two residuals were computed on different tile geometries (48 px vs
40 px). Same model, same tiles, or the numbers are not an A/B.

## F81–F89 — Alignment arc: parallax, disocclusion, and object geometry

One arc, consolidated. Nine findings, six of which overturned a claim made earlier
in the same arc; the corrections are kept because each names a trap.

### Shipped (F81, F82) — default on, runtime path

A handheld rotation pivots the DEVICE, not the lens entrance pupil, so the camera
centre translates and displacement scales with inverse depth. Measured directly: on
every moving phone sweep the residual inside near content was 2.0–2.5× the residual
in far content. One global warp splits the difference; F79/F80 handled the
consequences of that, this handles the geometry.

`align_stack` adds a depth-aware pass — depth-from-focus bins cut at histogram
VALLEYS, one translation-only ECC correction per bin, blended into one dense
coordinate field, relaxed where it would stretch content, resampled exactly once —
plus per-pixel refusal of scene that parallax uncovered, carried into fusion as a
`usable` mask (separate from F80's rectangular crop).

| | GT-SSIM | PSNR |
|---|---:|---:|
| global affine only | 0.875425 | 23.34 dB |
| + depth-aware alignment | 0.966007 | 29.66 dB |
| + disocclusion refusal | **0.978509** | **31.82 dB** |

Real sweeps, near-content residual and fraction withheld per frame: kitchen
2.190 → 0.544 px / 4.32%; large-motion 1.540 → 0.907 px / 4.09%; small-motion
0.469 → 0.227 px / 0.94%; zero-motion 0.046 → 0.031 px / 0.11%. What is withheld
tracks how much the camera moved, which is the behaviour to demand. Cost is
1.5–3.5× alignment time.

Four properties are load-bearing and each was found by failing first:
1. **Valley bin edges, not quantiles** — an equal-population edge cut through the
   middle of one object, putting a 13 px step across its own surface (visible seam
   and ghost strip on the large-motion book).
2. **Stop the field stretching** — relaxing displacement where its local gradient
   exceeds 0.10 RAISED probe GT-SSIM (0.9579 → 0.9628), so the stretch was pure
   damage. No membership width substitutes: narrowing makes stretch worse.
3. **Refusal at every level** — untextured/underpopulated bins, diverged fits and
   sub-three-frame stacks fall back to the global warp; a frame earning no
   correction is byte-identical.
4. **Disocclusion ribbons must come from MEASURED per-bin displacement**, never the
   smoothed applied field (which reported 0.06% on kitchen), and be sized by the
   step: a foreground moving Q px uncovers a Q px strip, tested at a ladder of radii
   where radius r demands a step ≥ r. A single-scale test condemned 38.6% of a frame.

### The metrics cannot adjudicate this (F81a, F82a)

Q_SSIM scores the fused image against its locally sharpest source, and alignment
changes what the sources ARE. Cross-scoring each output against the other variant's
sources collapses both to ~0.72–0.82. Within-variant scores therefore partly measure
self-consistency with a possibly misregistered stack, which a misaligned stack can
win; the ±0.002 real-sweep deltas are not evidence either way. Refusal makes it
worse: deliberately declining an untrustworthy-but-sharp source always looks like a
loss, yet on the factory it is worth +0.0125 GT-SSIM. Judge alignment by
per-depth-region residual, the analytic factory, and disagreement crops. The metric
still earned its keep as a POINTER — localizing its one real-looking dissent is what
exposed the quantile-edge seam.

### Rigorous negatives (do not reopen without new evidence)

- **Parametric depth→displacement models (F81b).** Displacement is ∝1/Z so a model
  linear in the depth proxy looks principled; it loses (0.9313 vs 0.9565) because the
  focus-winner index is monotone but not affine in inverse depth, and multiplying a
  noisy proxy by a ~20 px coefficient turns depth wiggle into displacement wiggle.
  Binning quantizes that noise away and assumes nothing about the mapping.
- **Joint motion/depth/calibration estimator (F81b).** Alternating rotation,
  translation, breathing scale, depth and the depth-to-parallax calibration converges
  (corrections 1.13 → 1.11 → 0.35 px) and achieves the BEST registration of any
  variant (large-motion 0.568 px), but invents motion on the zero-motion sentinel
  (0.046 → 0.247 px). Available as `depth_model="joint"`, off by default. The wall is
  the observation model, not the solver: tile shifts are sound (84% agree within 1 px
  with independent ECC) yet the rigid-motion-plus-depth model explains only 25–50%
  of them.
- **One-sided disocclusion refusal (F83).** Every part of the occlusion-edge-blur cue
  works: contour localization 32 px → 4.8 px, the global ordering bit votes
  correctly, the front mask agrees 91.1% with the known near plane. And refusing only
  the background side LOSES: background-only 0.971433, foreground-only 0.975415, both
  0.978509. The occluder is opaque and geometrically present — but out of focus its
  own matte blurs, so its boundary pixels are foreground/background mixtures carrying
  background colour onto the object. Near a defocused silhouette BOTH sides are
  compromised, by different mechanisms. Instrument kept at
  `research/occlusion_order.py`; the conclusion does not survive.
- **Veiling as a hard mask (F84).** Direction and width are both derivable (foreground
  spreads outward; width = distance from the occluder's own focal frame, since
  contrast-over-gradient saturates by 2 px of blur and is useless on textured frames).
  It behaves correctly — the occluder's focal frame veils nothing — and still loses:
  neutral in the parallax-dominant regime, −0.002 in the veil-dominant regime built to
  favour it, −0.001 with both strong, and −0.0024 even at `harden=0`, so it is not
  redundancy. Refusing veiled background forces fusion onto frames where the
  BACKGROUND is defocused, trading contaminated-but-sharp for clean-but-blurry.
  **Refusal is the wrong verb for partial contamination:** disocclusion earns a hard
  mask because the observation does not exist, veiling does not because it does. The
  same physics as a soft down-weight (`harden`) gains 0.980438 → 0.982593.
- **Median-stabilizing the depth map before the step test (F82b)** made silhouette
  concentration worse (2.81× → 2.07×).
- **Connected-coherence gating of splits (F86)** changed nothing; the factory's extra
  regions are coherent, not confetti.

### Why the kitchen bottle resisted, and what fixed the diagnosis (F84–F86)

A depth bin is a RANGE, not an object. The bin holding the bottle covers 55.4% of the
frame with a p90 internal tile residual of 14.44 px; the bottle is 8.7% of it, needs
+19.2 px, and receives +2.3 px because ECC follows the majority. Raising the
acceptance cap changes nothing — the correction is never proposed.

This also corrects the F81/F82 headline honestly: kitchen near-residual 2.190 → 0.544
px was averaged over the near half by depth median and never measured the bottle,
which is why a metric improved while the picture did not.

Grouping by MEASURED MOTION rather than depth recovers it (`research/adaptive_bins.py`):
clustering each region's tiles by their residual across every frame concatenated takes
the bottle's region from 55.4% of frame at +2.47 px to 7.0% at +18.56 px, and +18.8 px
end to end, with the ghost slab visibly gone. Necessary details: pool split evidence
across frames by MAX not median (a sweep's many near-reference frames hide an object
stranded in the few that moved); snap regions to image structure rather than the tile
grid; and relax `_REFINE_MIN_BIN_FRACTION` and the acceptance cap together, since a
7%-of-frame region asking 19 px is rejected by both.

**Splitting needs a merge rule (F86).** Subdividing a rigid plane gives each piece its
own noisier fit, so a surface with no discontinuity is transported by different amounts
in different places. Regions whose fitted motion agrees across the sweep are one
object — zero-motion collapses 5 regions to 2, kitchen keeps the bottle separate.
Tolerance is sharply bounded above: at 3 px the factory's two PLANES merge and one
compromise fit across a real depth boundary costs 0.889 GT-SSIM.

Not promoted: splitting regresses the analytic factory (0.9785 → 0.9734, partly
recovered to 0.9740 by merging) and no tile-confidence floor serves both scenes
(0.05 buys the bottle and costs the factory; 0.35 the reverse).

### Focus breathing is upstream of all of it (F87 corrected, F88)

The bottle's two edges disagreed by up to 16.8 px, growing monotonically with defocus.
First reading: a defocus bias, since a bright out-of-focus object spreads over its
darker surround. **Wrong.** Measuring the width directly in the RAW frames: 138, 138,
139, 139, 139, 139, 140, 141, 143, 148, 155, 160 px — 14% real magnification, monotone,
before any correlation is involved. It is focus breathing, and the global affine
removes only part of it (aligned widths still run 140 → 154).

Structural consequence, larger than the correction: **per-bin translation cannot
express residual breathing**, because a scale error moves an off-centre object's two
edges by different amounts. Translation-only is the most constrained model that can
express parallax, and it is — but not this. And a rigid object's edges need NOT show
zero differential in the image, only in the scene; the rigidity test must target the
expected breathing scale or it will indict sound measurements.

Object separation confirms breathing is the blocker (F88). Measured against a known
silhouette: on the factory the object lands in one region at 79.1% IoU (depth bins
alone already manage that). On the kitchen the bottle FRAGMENTS — purity doubles to
30.8% but coverage collapses 85.9% → 22.2% — which is what a translation-only model
must do to a magnifying object. Removing breathing crudely (one global scale per
frame from the bottle's own width, ~40% of it) drops regions 7 → 5 and lifts bottle
IoU 14.8% → 23.8%, coverage 22% → 42%. **The region machinery is sound and was being
fed geometry it cannot represent. Fix breathing at the global stage first.**

### Object motion from edges (F87, F89)

Textureless interiors have nothing to correlate; edges do. Integrating a vertical edge
along its length turns a weak local match into a strong 1-D measurement, and rigidity
carries it to the interior. Correlate GRADIENT profiles, not intensity. Trust only the
normal component (aperture problem) and combine differently-oriented edges.

**Interior edges make the one-object question falsifiable (F89).** Two edges give two
measurements for two unknowns, so the system is exactly determined — solvable, never
testable, and "one object breathing" is indistinguishable from "two objects moving".
Under magnification a rigid object's edge displacement is LINEAR IN X, so each interior
edge is a constraint the hypothesis can fail:

| frame | confident edges | translation | magnification | rms resid | verdict |
|---|---:|---:|---:|---:|---|
| 7 | 11 | +4.07 | 1.0048 | **0.33** | one object |
| 8 | 8 | +8.26 | 1.0254 | 1.15 | one object |
| 9 | 7 | +10.83 | 1.0530 | 2.16 | borderline |
| 10 | 5 | +9.44 | 1.0897 | 5.41 | interior detail gone |
| 11 | 3 | −4.47 | 1.0151 | 24.03 | interior detail gone |

Eleven edges agreeing to 0.33 px rms is positive proof of one object, with translation
and magnification separately identified. The test degrades exactly where physics says
it must — interior detail is low-contrast and blurs away off the focal plane — which
is repairable, because motion is smooth along the sweep:

| frame-11 estimate | value |
|---|---:|
| depth-bin fit | +2.3 px |
| quadratic extrapolation from 3 usable frames | +23.98 px |
| **linear extrapolation from 3 usable frames** | **+18.88 px** |
| truth (ECC over the bottle region) | +19.2 px |

**Order of estimation:** measure near an object's focal plane where the evidence
exists, test rigidity there, then propagate. Do not try to measure an object where it
is most defocused. With three points, use the simplest model the physics allows — the
quadratic is exactly determined and extrapolates 25% long.

### Instruments built by this arc

`parallax_gen.py` (analytic two-plane parallax factory with GT — alignment cannot be
tested on frames differing by one global transform), `adaptive_bins.py` (motion
splitting + merge), `edge_motion.py`, `boundary_probe.py`, `occlusion_order.py`.

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
- Depth-dependent magnification (forward camera translation) is the dominant
  unsolved real-handheld limitation. It is not removable globally and needs a
  per-region scale term on object-sized regions (F90, correcting F87/F88).
- Object-level region grouping is measured but unpromoted: it recovers the kitchen
  bottle and regresses the analytic factory (F85/F86).
- The per-region two-frame architecture and its stitch stage are unbuilt.

## Working rule

Preserve the liked F78 numerical path while testing its scope. When a new case
fails, first determine whether the input formation, ownership geometry, source
attribution, inversion, or integration is wrong. Do not start by tuning a
global threshold or smoothing the symptom.
