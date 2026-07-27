# Current findings

This is the compact scientific state of the project. It is not a chronological
lab notebook; superseded experiments and their reports remain in Git history.
Read `MISSION.md` first, then this file, `OCCLUSION_FORMATION.md`, and
`STATE.md`.

## F87 — Edges carry object motion where interiors cannot, once blur bias is removed

F85/F86 group by per-tile residual, which has nothing to measure inside a flat
surface. The Lubriderm bottle is mostly blank white, so tile evidence fragments
it and the tile-based fit gave the bottle +2.3 px where it needs +19.2.

Its edges are another matter. Integrating a vertical edge along its length turns
a weak local match into a strong one-dimensional measurement, and three physical
facts make edges sufficient for the whole object: a rigid object's edges all
move together; the bottle does not approach or recede, so there is no scale to
solve for; and a flat interior bounded by co-moving edges therefore inherits
their motion instead of needing evidence of its own. The aperture problem is
handled by taking only the normal component and combining differently-oriented
edges around one outline.

Measured on the kitchen sweep, frame 11 against the reference, where truth is
+20 px by direct edge scan and +19.2 px by ECC over the bottle region:

| method | left edge | right edge | differential |
|---|---:|---:|---:|
| per-tile residual (F85) | — | — | fit was +2.3 px overall |
| intensity profile, wide band | +1.18 | +17.98 | +16.81 |
| gradient profile, wide band | +13.62 | +19.97 | +6.36 |
| gradient profile, edge-centred window | +13.46 | **+19.85** | +6.39 |

**CORRECTED — the widening is mostly REAL, and it is focus breathing.** The
first reading of this was that defocus spreads a bright object over its surround
and biases the apparent edge outward. That explanation is wrong, and measuring
the bottle's width directly in the RAW frames is what refuted it: 138, 138, 139,
139, 139, 139, 140, 141, 143, 148, 155, 160 px across the sweep. A 14%
magnification change, monotone, present before any correlation is involved. The
lens refocusing changes magnification, which is the focus breathing this stage
has always existed to remove, and the global affine removes only part of it —
aligned widths still run 140 -> 154.

Gradient profiles do still help (frame 11's differential falls 16.81 -> 6.39 and
the strong edge lands within 0.15 px of truth), so some bias exists on top. But
the bulk of the differential is a real scale change, not a measurement artifact.

The consequence is structural and matters more than the measurement: **per-bin
translation cannot express residual breathing.** Translation-only was chosen as
the most constrained model that can express parallax, which it is — but a
residual scale on a 140 px object lying off-centre requires its two edges to
move by DIFFERENT amounts, exactly the +13.5 / +19.9 measured here. No
translation can do that. Either the global stage must remove breathing properly
(its affine currently compromises between scale and depth-varying parallax), or
the per-region model needs a scale term. This is now the most likely reason the
bottle resists correction, ahead of anything in F85/F86.

A rigid object's edges therefore do NOT have to show zero differential in the
image, only in the scene. The rigidity constraint must be stated against an
expected breathing scale, or it will keep indicting good measurements.

**Rigidity is still a usable detector, but not with a zero target.** Two edges
of one object must agree once the frame's magnification is accounted for. With
breathing folded in, the constraint as first written would reject sound
measurements; with it removed, the disagreement that remains is genuine evidence
about which reading to distrust.

Also worth recording: the correlator used for all of this was wrong on first
write, reporting -shift/8 because of time-domain zero padding, and produced a
completely plausible table of small consistent numbers. It was caught only by
feeding it known synthetic shifts. Validate a measuring instrument against a
known answer before believing anything it says about real data.

Open: the left edge still reads ~6 px low. Its window very likely straddles
background at a different depth, whose motion pulls the correlation. Narrowing
onto object support — which the split/merge work already produces — is the next
step, and the two lines of work meet there.

## F86 — Splitting needs a merge rule: pieces that move alike are one object

F85's residual splitting recovered the kitchen bottle and regressed the analytic
factory. The cause is not what two guesses predicted. The factory's extra
regions are neither confetti (a connected-coherence gate changed nothing) nor
caused by an over-permissive confidence floor. They are coherent pieces of ONE
RIGID PLANE, each handed its own independent ECC fit — and small regions fit
more noisily, so a surface with no discontinuity in it gets transported by
slightly different amounts in different places.

So object integrity is a MERGE rule, not a shape rule, and it uses the same
evidence as the split read the other way: regions whose fitted motion agrees
across the whole sweep are one object, however the split arrived at them.

| merge tolerance | factory (2 real regions) | kitchen bottle |
|---|---|---|
| none | 5 regions, 0.973263 | +18.8 px |
| 1 px | 4 regions, 0.974036 | +18.8 px |
| 2 px | 4 regions, 0.974036 | +18.8 px |
| 3 px | 4 regions, **0.889273** | +18.8 px |
| 5 px | 3 regions, 0.888564 | +18.2 px |

Zero-motion collapses 5 regions to 2, which is the behaviour to demand. The
tolerance is sharply bounded above: past 2 px it starts merging the factory's
two PLANES, whose motions differ by about 9 px, and a single compromise fit
across a real depth boundary is far worse than any over-splitting.

Net state: kitchen's bottle is corrected end to end (+2.5 -> +18.8 px, the ghost
slab visibly gone), the factory still costs 0.0045, and the remaining gap is
over-splitting that the merge rule does not fully undo. Still not promoted.

## F85 — Residual-driven splitting recovers the bottle, and is not yet promotable

F84 showed the depth bin holding the kitchen bottle covers 55% of the frame and
is fitted to +2.3 px where the bottle needs +19.2. The reframe that follows:
stop grouping by depth and group by MEASURED MOTION. Depth is only a seed, and
"wants the same correction" is the operational definition of an object here.

`research/adaptive_bins.py` splits each region by clustering its per-tile
residual, with the feature being the residual across EVERY frame concatenated —
an object is an object in every frame, so its residual profile is a far stronger
signature than any one frame's number. It works:

| level | bottle's region | bottle's share of it | frame-11 fit |
|---|---:|---:|---:|
| 0 (depth bins) | 55.4% of frame | 8.7% | +2.47 px |
| 1 | 11.5% | 24.0% | +4.96 px |
| 2 | 7.0% | 29.9% | **+18.56 px** |

End to end the corrected region reaches +18.8 px, and the crop shows it: the
ghost slab over the background beside the bottle is gone and the shelf behind it
is sharp up to a clean edge.

Three details were each necessary and each found by failing first:

1. **Pool the split evidence across frames by MAX, not median.** A sweep spends
   most frames near the reference where everything fits, so a median hides an
   object stranded in the few frames that moved. Median-gating refused to split
   the kitchen at all.
2. **Regions must not inherit the tile grid.** A blocky support puts staircase
   transitions into the sampling field. Tiles vote; the vote is snapped to image
   structure with a guided filter.
3. **The runtime's own guards blocked the result.** `_REFINE_MIN_BIN_FRACTION`
   (6%) and the 14 px acceptance cap reject a 7%-of-frame region asking for
   19 px. This also explains F84's puzzle that raising the cap alone did
   nothing: without the split the correction was never proposed, so there was
   nothing for the cap to reject.

**Not promotable.** Splitting regresses the analytic factory, whose bins are
already homogeneous, and no single tile-confidence floor serves both scenes:

| tile confidence floor | factory GT-SSIM | kitchen bottle |
|---|---:|---:|
| 0.05 | 0.978509 -> 0.973263 | +18.8 px |
| 0.20 | 0.978509 -> 0.973263 | +7.8 px |
| 0.35 | 0.978509 -> 0.978496 | +1.9 px |

A magnitude threshold is the wrong instrument, because it asks "is this residual
big" when the question is "is this residual real". The physical test is
available and unused: a genuine object's residual must scale with each frame's
motion and hold its direction, while a garbage phase correlation — from a tile
straddling the disoccluded zone, which is what the factory keeps splitting on —
will not. Gate on proportionality to frame motion, not on size.

## F84 — Veiling is real but must not be a hard mask; depth bins are the actual blocker

Two instruments were built for the two defects visible at the kitchen bottle,
deliberately on the smallest data that can show them (the analytic factory plus
one real sweep). `research/boundary_probe.py` runs both.

**Veiling — negative, in the regime built to favour it.** A defocused occluder
spreads its own material outward over the background and never pulls background
inward, so a mask can take its direction from the occluder and its width from
that occluder's defocus. Width needs no blur estimator: contrast-over-gradient
saturates by 2 px of blur (0.25/0.78/1.02/1.58 at radii 0/1/4/12) and reads 4-7
on real frames where texture swamps the window, but F83's contour reading
already names the occluder's focal frame, so the width is just how far a frame
sits from it. That version behaves correctly — the occluder's own focal frame
veils nothing — and still loses:

| regime | no refusal | ribbon | veil | ribbon+veil |
|---|---:|---:|---:|---:|
| parallax-dominant | 0.966007 | **0.978509** | 0.966610 | 0.978728 |
| veil-dominant | 0.982593 | 0.982604 | 0.980566 | 0.980593 |
| both strong | 0.970096 | **0.973777** | 0.968295 | 0.972802 |

It is not redundancy with `harden`: at `harden=0` the veil mask still costs
-0.0024. Refusing veiled background forces fusion onto frames where the
BACKGROUND is defocused, trading contaminated-but-sharp for clean-but-blurry,
and GT prefers the former. Meanwhile `harden` — the same physics expressed as a
soft down-weight — gains 0.980438 -> 0.982593 in that regime.

The lesson generalizes beyond veiling: per-pixel validity can only refuse, and
refusal is the wrong verb for contamination that is partial. A veiled pixel is a
mixture in some proportion, not an absence. Disocclusion earns a hard mask (F82)
because the observation genuinely does not exist; veiling does not, because it
does. Any future boundary evidence should ask which of those two it is.

**Bin homogeneity — strong positive.** A depth bin is a range, not an object. On
the kitchen sweep, frame 11 against the reference:

| bin | share of frame | fitted | tile residual median / p90 | contains |
|---|---:|---:|---:|---|
| 0 | 55.4% | (+2.33, +0.13) | 3.10 / **14.44** px | the bottle, 8.7% of the bin |
| 1 | 15.9% | (-0.62, +0.14) | 1.54 / 2.60 px | |
| 3 | 17.7% | (-0.94, -0.03) | 2.13 / **10.91** px | |

The bottle needs +19.2 px and its bin is fitted to +2.3 px by the other 91% of
its pixels. Raising the acceptance cap changes nothing — the correction is not
rejected, it is never proposed, because ECC over a bin follows the majority.
This is why the near-residual headline in F81/F82 (2.190 -> 0.544 px on kitchen)
did not translate into a better-looking bottle: averaged over the near half by
depth median, it never measured the object that fails.

Next work belongs here, not on veiling: bins must be split until each is
homogeneous, driven by the per-tile residual this instrument already measures.

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
