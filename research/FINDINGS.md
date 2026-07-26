# Adaptive focus-stacking — running findings log

Persistent notes from the autonomous marathon. Newest first. Pairs metric numbers
with conceptual reasoning and visual inspection (metrics guide, don't decide).

---

## F62 — Front geometry must be repaired before rear recovery

The user's second `extension_007` inspection at native `(1048,216)` exposed the
remaining cascade-order error. The far-focus input was not itself nonphysical at
that point: true alpha is 1.0, far-frame foreground coverage is 0.545, and a
finite aperture therefore records sharp rear texture plus a defocused foreground
contribution. The failure was downstream. The mixed-base alpha was only 0.718
and placed the point one pixel inside its boundary, so ordered visibility
correctly blocked the rear solver but left the mixed base untouched. The
forward-licensed focused-owner silhouette placed the same point 12.6 pixels
inside the bird. Protection without front reconstruction was only half the rule.

The first hypothesis—reuse an accepted parent-tail license to copy its eroded
interior—fixed all three current cases, but was not promoted. A new 36-scene S16
split produced one fire, `s16_034`, that lost globally (ΔSSIM −0.000486,
ΔMAE +0.0013), worsened inner partial by +0.0338 and far background by +0.0375,
and had **zero** parent-interior copy pixels. The new copy rule was exonerated;
the counterexample instead showed the older mixed-base matte overextending into
clean rear background. Its alpha IoU was 0.748. A sharp-owner mask in the
already-computed bank had 0.952 true IoU and reduced captured-frame forward
residual from 1.501 to 1.408, but the additive parent rule rejected it because
it contained only 79.1% of the overextended seed. An additive support mechanism
cannot remove a bad foundation.

F62 makes the cascade front-first:

1. physically rerank the mixed-base candidate as before;
2. search the selected owner's sharp-frame masks for the same object (IoU≥0.70,
   seed containment≥0.75, area ratio 0.5–1.5);
3. let that focused-owner silhouette *replace* the mixed-base alpha only when it
   remains licensed and improves whole-frame forward residual by >0.01;
4. inside the refined silhouette, hard-select the focused owner only where the
   point is at least one model-scale pixel inside the boundary and both PSF
   families predict partial far-frame coverage (>10% front and >10% rear);
5. run ordered rear recovery on the remaining outer veil.

The owner copy is therefore neither blanket foreground replacement nor a
learned score patch. It uses a directly observed front layer, excludes the
uncertain semantic edge and complete optical core, and precedes the inverse
solver that otherwise has to reinterpret a mixed pixel.

Frozen reruns:

- `extension_007`: the reported pixel changes from base RGB `[65,63,60]` to
  focused-owner `[65,65,68]` versus GT `[69,69,66]`; its 21×21 neighborhood MAE
  falls 7.786→2.887. Globally ΔSSIM +0.001083, ΔMAE −0.0657, inner partial
  −0.507, outer veil −0.968, core/far identity.
- `extension_034`: ΔSSIM +0.001797, ΔMAE −0.1246, inner partial −2.015,
  outer veil −3.497, core/far identity.
- `s12_025`: ΔSSIM +0.001343, ΔMAE −0.0851, inner partial −0.694,
  outer veil −1.059, core/far identity.
- repaired fresh-split counterexample `s16_034`: the exact prior loss flips to
  ΔSSIM +0.000582,
  ΔMAE −0.1296 and MSE 19.53→16.17; 64,207/9,601 changed pixels move
  closer/worse. Inner partial improves −2.167, outer veil −1.507, complete core
  is byte-identical, and far background is effectively identity (−0.0000026).
  The other 35 S16 scenes refuse exactly. S16 exposed the failure and therefore
  remains development evidence, not a post-final holdout.
- genuinely post-final S19: 72 scenes were generated with seed 21001 only after
  the final F62 code and thresholds were frozen. Three fire (`s19_000`,
  `s19_012`, `s19_013`); all three improve SSIM, MAE, and MSE and every
  core/inner/outer/far partition. Mean partition ΔMAE is −0.0022 complete core,
  −0.5915 inner partial occlusion, −1.5991 outer veil, and exactly 0.0 far
  background. The other 69 scenes refuse byte-identically. Disagreement-guided
  visual crops confirm that all three edits remove mixed/ghosted foreground
  boundary content toward the sharp GT silhouette; residual magenta is sparse
  rather than a coherent displaced region.

The fine-band complement remains open, but improves rather than being hidden:
all seven current fire deltas remain positive, ranging +0.00138 to +0.00963.
Auto remains off. The inspector now contains all seven current V2 fires, adds
focused-owner refinement/front-reconstruction maps, includes the exact
`(1048,216)` crop, and removes the already-reviewed legacy V1 deep cases. A
separate ordinary-photo cohort runs the actual default pipeline on four real
two-frame photographs plus two real phone sweeps, with every original and
aligned/normalized frame visible. Five are exact post-fusion refusals; the gated
contour specialist fires broadly on the golfer pair (71,170 pixels, 26.3%).
With no all-in-focus GT this is exposed for direct inspection, not counted as a
win.

**Shipping action:** `VEIL_AUTO_ENABLED=False`. S13's hierarchical owner-mask
gap is closed for the narrow licensed regime; S15's finest-band tail still
blocks runtime activation.

## F61 — Rear recovery requires positive visibility; absence of foreground is not evidence

> **F62 scope correction:** ordered visibility closed the first spatial tail,
> but a second fresh split exposed overextended mixed-base geometry. Focused-
> owner matte replacement plus front reconstruction now precede this rule. F61
> remains the rear-stage invariant, not the complete cascade.

The user's scene-114/122 diagnosis exposed the invariant that F60's corrected
factory made measurable but the runtime still did not encode. Focused foreground
is an on-focal occlusion and must veto the rear layer. A defocused foreground can
reveal the rear through aperture coverage, but rear recovery is licensed only
where another focused frame positively observes rear structure. The previous
application mask used “outside estimated alpha and not confidently foreground.”
That is only an absence of veto; it is not evidence that the rear was seen.

S12 now separates the two observations. `_ordered_visibility_gate` builds a
front-owner veto and an independent rear-focus evidence density. Its product is
then intersected with a conservative optical band: both exact-disk and box-disk
approximations must assign at least 10% coverage to each layer, with a continuous
10% ramp and a hard support clamp. The latter matters: an initial 5% one-kernel
version leaked tiny far-background tails on two already-seen holdout cases
(+0.00037/+0.00063 MAE). Cross-PSF consensus returns both to exact identity.

The same audit found a second, simpler scale error. A small parent fragment was
required to improve whole-image forward error by 5%, even though its optical
influence covers only a local PSF neighborhood. `extension_007` was therefore
rejected despite 98.1% seed containment, 0.936 IoU, 0.861 true-foreground
precision on 2,539 novel pixels, and a 67.6% forward improvement in the affected
neighborhood. The absolute whole-frame improvement remains mandatory; the
relative 5% test now uses the proposed support dilated by the model CoC.

Frozen evidence after the rule:

- oracle-alpha development: 9/9 fires improve SSIM/MAE/MSE; complete core,
  inner partial, and far background are byte-identical while outer veil improves
  by −1.743 MAE mean (worst −0.128);
- oracle-alpha holdout: 8/8 fires improve SSIM/MAE/MSE with the same three
  identity partitions; outer veil improves −1.154 mean (worst −0.054);
- diagnosed runtime extension: both fires improve every physical partition.
  `extension_007` flips inner partial from +0.415 MAE under F60 to −0.136 and
  far background from +0.0002 to exact identity; `extension_034` remains
  positive and accepts parent plus satellite support;
- a new 36-scene post-freeze S12 split yields one conservative fire,
  `s12_025`: ΔSSIM +0.001056, ΔMAE −0.0542, inner partial −0.220,
  outer veil −0.877, complete core and far background exact identity.
  The other 35 scenes refuse exactly.

This closes the reported *spatial ownership* tail and validates the ordered
visibility concept independently of the cases that shaped it. It does not ship
the giant-veil path. All three deep runtime fires raise the GT-credited
fine-band false-texture error by about +0.0096 to +0.0100 on otherwise-smooth
veil pixels. Visual inspection shows sparse boundary pinpoints rather than a
coherent invented texture field, but the complement metric is deliberately
strict and must not be waived. The next task is causal localization and an
identity-or-shrink decision for those pixels.

**Shipping action:** `VEIL_AUTO_ENABLED=False`. The inspection page puts the
three exact-disk V2 S12 cases first, with both inputs, optical partitions,
estimated/true alpha, ordered visibility, application/protection/support maps,
amplified edits, maximum-regression crops, and exact region metrics. The five
legacy V1 cases remain only for coordinate-note reproduction.

## F60 — The optical factory, not another threshold, resets the veil verdict

The user correctly challenged the input frames after inspecting them rather than
trusting the aggregate scores. The precise diagnosis has two parts. A severely
defocused foreground really can become partially occluding: different aperture
rays see foreground or the revealed background, so sharp alpha=1 does not imply
frame-specific coverage=1. At the reported `scene_122` point, the far-focus
coverage is only 0.505 at a 37.7 px defocus radius; it is physically an inner
veil even though the old inspection page showed only sharp alpha. But the V1
factory also had a genuine implementation defect: `hardbench.disk_blur()` changed
every radius above 12 px into downscale→**box** blur→upscale while the benchmark
continued to call it a disk. It used FastSAM silhouettes as exact alpha, retained
source-photo edge contamination in the cutout RGB, and did not expose
frame-specific coverage. The old 10/10 and P13/P14 ledgers remain reproducible
mechanism diagnostics, but they no longer promote or ship the veil path.

The F60 V2 factory replaces that foundation before changing the algorithm:

- offline convolution uses the exact circular aperture at native resolution and
  is unit-tested against a brute aperture average;
- two layers sit exactly at the two focal planes, making the premultiplied
  near/far equations exact rather than an independent-blur approximation;
- uncertain source-mask boundary RGB is replaced by nearest eroded-interior
  foreground radiance before compositing;
- every scene saves near/far frame-specific coverage plus a class map: green
  complete-coverage core, yellow inner partial occlusion, magenta outer veil;
- development, untouched holdout, and post-rule extension each cycle through
  explicit `solid`, `mixed`, and `thin` strata. Solid scenes preserve 55–80%
  complete-coverage core; thin/all-veil scenes are labeled rather than pooled.

The visual invariant now matches the optics and the user's intended audit:
background contributes nowhere inside the green complete-coverage core. In the
yellow/magenta partial-coverage band it is expected and measured, not mistaken
for opaque foreground damage.

The unchanged mixed-base runtime gate fires on 0/12 V2 development and 0/12
untouched-holdout scenes, returning exact identity. That is safe but means the V1
runtime evidence does not transfer. The operator ceiling survives: with oracle
factory alpha/owner but unchanged forward license and solver, 9/12 development
scenes fire and all nine improve SSIM, MAE, MSE, and complete-core MAE. On the
untouched holdout, 8/12 oracle cases fire; seven improve SSIM/MAE and all eight
improve MSE and do not damage complete-core MAE. The dissent,
`holdout_003`, is tiny globally (ΔSSIM −0.000026, ΔMAE +0.0018) but real:
its correction slightly worsens outer veil/far background while leaving opaque
core and inner foreground byte-identical. Forward ratio 0.590 cannot detect that
spatial tail.

Moving the existing ownership idea upstream was tested without retuning any
threshold: sharp owner-frame masks use the same score/purity/area-fit/ring and
captured-frame license as mixed-base masks. One development and one untouched
holdout scene fire, both improving global SSIM/MAE/MSE. The holdout fire,
however, worsens complete-core and inner-partial MAE because a hierarchical
child mask contains missed support but is neither a detached satellite nor a
full parent. A physically licensed overlapping-patch experiment repairs that
case, but it was derived from the holdout and therefore is **not promoted**.
The owner-frame fallback remains research-only.

After freezing V2, parent thresholds, and the production mixed-base path, a new
36-scene extension produces two fires. Both improve SSIM/MAE/MSE. `extension_034`
independently accepts parent-silhouette plus satellite support and improves
complete-core, inner-partial, outer-veil, and far-background MAE—so F58/F59's
discrete ownership fixes survive the corrected optics. `extension_007`, however,
improves global errors and outer veil while worsening inner-partial MAE by +0.415
gray and false texture by +0.0159. The runtime cannot yet distinguish these two
fires. That is enough to retain the model and enough to refuse shipment.

**Shipping action:** `VEIL_AUTO_ENABLED=False`. `--enhance auto` keeps gated
contour reconstruction but no longer launches the semantic bridge or joint veil
solver. The V1 inspector is labeled legacy; its exact scene-114/122 diagnostics
remain available. A new V2 section shows the actual input frames beside explicit
optical coverage. Re-enable requires positive observed-background evidence that
protects inner partial foreground, then a fresh post-rule extension with no
physical-partition regression. Global benchmark unanimity cannot substitute.

## F59 — Parent-silhouette completion carries front/back ordering into recovery

> **F60 scope correction:** the ownership mechanism remains valid and is
> independently observed on the corrected V2 extension, but the V1 scores below
> no longer support shipment. The giant-veil auto path is safety-disabled.

The `scene_122` report at native `(x=804, y=521)` exposed the larger form of
F58's ownership failure. Frame 0 contains a sharp yellow foreground edge and
frame 1 contains the revealed rear surface. The mixed-base semantic alpha is
zero at the point even though true alpha is one, and the local focus-energy
winner is split almost evenly across the missed 727-pixel foreground component.
The old satellite-only rule therefore left the correct base pixel available for
the veil specialist to overwrite with rear-frame content: RGB `[240,189,8]`
became `[165,138,46]` while GT is `[245,202,9]`. This is not a better veil
solver problem. It is a missing front/back ordering constraint.

The owner-frame segmenter had already supplied the needed observation: its main
mask contains 96.3% of the licensed seed, has 0.909 IoU with that seed, and adds
a 4,306-pixel tail with 0.94 GT foreground precision. P13 therefore generalizes
owner support from detached satellites to high-overlap parent silhouettes. A
parent must contain at least 90% of the licensed seed, reach at least 0.80 IoU,
add no more than 2% image area, keep at least 90% of that novel support inside a
1.5-CoC neighborhood, and reduce captured-frame forward MAE by both the existing
0.01 absolute margin and a new 5% relative margin. Only the novel pixels are
hard-copied from the observed owner and excluded from veil correction. The
semantic model proposes a correspondence; re-rendered observations license it;
neither GT nor a public image-quality score participates at runtime.

At the reported pixel, the composed output now selects frame 0 exactly:
`[248,202,13]`, reducing mean RGB error 6.33→2.33 instead of increasing it to
60.33. In the connected 669-pixel reported region, MAE falls 8.06→4.39 and
changed pixels grade 569 closer / 83 worse. Whole-scene `scene_122` improves from
the user's prior +0.001420 SSIM / −0.1546 MAE / 53,613:18,016
closer:worse checkpoint to +0.001792 / −0.1879 / 54,713:16,642. True-foreground
MAE falls by 1.87 gray/channel. The five-case inspection lab is regenerated
through the same composed package: foreground MAE now improves on `scene_75` and
`scene_122`; three smaller tails remain explicitly open.

The 90% containment, 0.80 IoU, and 5% relative-fit thresholds were frozen before
creating scenes 175–199. Only two fresh scenes pass the older giant-veil license.
Both independently accept a parent silhouette. On `scene_178`, the same package
with support disabled is harmful on every direct measure
(SSIM −0.000740, MAE +0.0313, MSE +1.0885, fringe +1.037); the frozen parent rule
flips all four positive (+0.000108, −0.0161, −0.3770, −0.762). `scene_184`
improves further from +0.000819/−0.1554/−8.392/−4.157 to
+0.001594/−0.1777/−9.008/−4.714. This causal, post-rule extension matters more
than a pooled benchmark mean: it directly tests whether the new ownership
constraint prevents the failure it claims to prevent.

P14 then exercises the exact `enhance()` composition across the ten established
fires, six prior-extension fires, and two fresh fires. Direct and composed
outputs match; contour reconstruction remains silent. Seventeen of eighteen
improve SSIM, and all eighteen improve MAE, MSE/PSNR, and true-fringe error. The
sole SSIM tail remains prior `scene_172` (−0.000416) with no owner-support fire,
so this mechanism neither caused nor conceals it. False-texture deltas remain
slightly positive and the scope is still exactly two frames, fixed giant CoC,
semantic bridge, and <=1600 px. Parent completion repairs a discrete
identifiability failure; it does not close continuous support uncertainty,
outer-fringe evidence, general CoC/N-frame recovery, or real-scene truth.

## F58 — Owner-frame support completion repairs the reported mixed-ownership fragment

The `scene_114` diagnostic at native `(x=187, y=252)` isolated an upstream
support failure rather than a veil-solver failure. The true alpha there is
0.882, but the pass-1 semantic alpha is exactly zero. Frame 0 observes the small
black foreground appendage; frame 1 mostly reveals sharp checkerboard/background.
Per-band focus energy therefore votes for the wrong frame on 557/605 nearby
missed-foreground pixels. Hardening the focus winner would make the failure worse,
and the veil specialist cannot infer that its input base already mixed two
different scene owners.

P11 adds a second semantic observation at the correct point in the pipeline:
automatic masks from the sharp foreground-owner frame. A fragment is eligible
only when it is at most 2% of the image, overlaps the already licensed matte by
less than 20%, and lies at least 90% within a 1.5-CoC neighborhood of that matte.
Appearance is not the license. Adding that fragment to the two-layer formation
model must independently reduce 512 px captured-frame forward MAE by more than
0.01; the combined fragment set must pass the same margin. Accepted pixels are
copied only from the observed owner frame and hard-veto veil correction. They do
not alter the inverse solver's alpha and never synthesize texture.

On the ten established exact-package fires, this admits owner support on six
scenes and preserves positive SSIM, MAE, MSE/PSNR, and fringe-L1 outcomes on all
ten. `scene_114` accepts mask 15: 522 pixels at 0.883 GT precision, forward MAE
1.5638→1.5330. At the reported pixel, RGB error falls 28.0→8.33; across the
entire support component, MAE falls 25.01→13.83 and changed pixels grade
440 closer / 76 worse. The full scene moves from the old +0.000931 SSIM /
−0.1697 MAE checkpoint to +0.001156 / −0.1885.

The 0.01 margin was frozen before generating a new 25-scene factory extension.
Six fresh scenes pass the older giant-veil license; owner support fires on two.
Against the same package with support disabled, `scene_152` improves slightly and
`scene_159` gains an additional +0.00023 SSIM, −0.018 MAE, and −1.80 MSE. Neither
accepted fresh case regresses. P12 then runs all established and fresh licensed
fires through the composed `enhance()` entry point: joint recovery fires, contour
reconstruction remains silent, and every score matches the direct package.
The focused and full suites pass 42 tests, including refusal of a plausible
owner fragment that does not improve the physical forward model.

This is a narrow mechanism repair, not closure of F57. Four of the five deep
cases still have slightly worse aggregate true-foreground MAE, and the disclosed
false-texture index remains positive. Fresh `scene_172` also retains a small
−0.00042 SSIM tail with zero owner-support pixels; that belongs to the older
candidate license, not this new mechanism. FRONTIER 19i therefore remains open
for continuous support uncertainty, outer-fringe evidence, and spatial tail
control. The inspection lab now includes the owner-support mask, support-only
metrics, physical-fit margin, exact RGB values, and a fixed crop at the reported
coordinate.

## F57 — The owner lab turns aggregate wins into a spatial repair agenda

The shipped F56 claim survives its direct package measures, but the new five-case
owner inspection lab shows why “10/10 positive” is a checkpoint, not a stopping
condition. The lab deliberately deep-inspects the weakest licensed SSIM win, the
largest false-texture tail, the ownership-leak stress case, and both scene-disjoint
holdouts. Every case exposes both original frames, base/output/GT, estimated and
true alpha, runtime application/protection masks, signed edits, continuous and
discrete GT error maps, and automatically selected maximum-edit and
maximum-regression crops.

The spatial partition changes the interpretation without overturning the narrow
ship:

- all five still improve global SSIM, MAE, MSE/PSNR, and true-fringe L1;
- **true-foreground MAE worsens on all five** by +0.022 to +0.870 gray/channel;
- 12,948 to 83,416 changed pixels fall outside the hard GT fringe support;
- far-background MAE is essentially neutral/improved on three, but worsens
  +0.0067 and +0.0592 gray on `scene_75`/`scene_114`;
- semantic alpha IoU spans 0.823–0.926; `scene_122` has only 0.857 recall and
  remains the clearest support-uncertainty case even after ownership protection;
- the already disclosed false-texture tail remains positive on all five.

“Owner-safe” therefore means the independent owner projection changed a failing
global squared-error tail into a passing one; it does **not** mean zero foreground
or off-fringe damage. The next mechanism target is support-tail control: measure
continuous owner/candidate uncertainty, require positive observed background
evidence before editing the outer fringe, and project the correction more
aggressively where semantic support and focus ownership disagree. Do not tune this
against SSIM. The new acceptance view is a vector of physical partitions plus
worst-region maps; global metric unanimity cannot close a spatially coherent tail.

The workbench itself is now reproducible (`docs/INSPECTION.html`,
`docs/inspection_manifest.json`, generated by the shipped operator). GT-only aids
are visually and textually separated from runtime evidence. An explicit large-view
region picker—not an image click side effect—can add native coordinates to feedback
notes. This makes user observations executable test cases rather than screenshots
without provenance.

## F56 — Owner evidence closes the matte null space; narrow joint veil recovery ships

The P8 package candidate exposed one last benchmark disagreement on fresh-holdout
`scene_122`: SSIM, global MAE, fringe L1, and the number of changed pixels closer
to GT improved, but global MSE/PSNR worsened slightly. Spatial decomposition—not
metric voting—located the squared-error tail inside **true foreground pixels omitted
by the semantic matte**. The joint solver still re-rendered the observations well:
blur's null space had simply allowed those foreground pixels to be explained as far
background. Regularizer and PSF consensus cannot detect a model-support error shared
by every solve.

The fix uses an independent observed remnant already present in the stack. Focus
dominance identifies pixels decisively owned by the near-focused frame; a
resolution-scaled smooth veto attenuates the recovery there, and the final mask is
strictly zero on the proposed foreground. On `scene_122`, this flips MSE from
+0.726 to −1.107, PSNR from −0.040 to +0.061 dB, raises SSIM from +0.000062
to +0.000262, and improves fringe L1 from −1.40 to −1.77 gray. This is an
identifiability repair, not a benchmark filter: the same operator also flips both
formerly SSIM-dissenting development rows positive.

P9 audits the exact package entry point, including 512 px physical candidate
reranking, the frozen semantic/forward license, fixed 3.5%-image CoC, three
regularizers, two PSF families, scaled native priors, ownership projection, and
identity refusal:

- development: **7/7 fires positive on SSIM, MAE, MSE/PSNR, and fringe L1**;
  mean deltas +0.00095 SSIM, −0.114 MAE, −4.49 MSE, +0.249 dB, −3.33 fringe;
- first scene-disjoint holdout: **2/2 positive on every direct measure**;
- second untouched 25-scene holdout: **1/1 positive on every direct measure**
  (+0.00113 SSIM, −0.150 MAE, −6.11 MSE, +0.396 dB, −4.99 fringe);
- the two holdouts fire on **0/24 moderate-CoC scenes**. Across development and
  both holdouts, the fixed giant hypothesis refuses all 66 moderate scenes.

The smooth-region false-texture index remains a small positive boundary-frequency
tail (+0.007 to +0.017 gray), so it is reported rather than called zero. Automated
worst-disagreement panels with GT show corrections localized to the physical contour,
moving toward scene truth without the broad pattern extension that triggered F53.
The shipped scope is deliberately narrow: exactly two frames, max side <=1600,
bridge available, licensed candidate, giant CoC only; otherwise byte-identical
identity. The old correction-after-fusion function remains only for research
reproduction and is unreachable from auto enhancement.

P10 closes composition rather than assuming specialist independence. Running all
ten licensed scenes through the public `focusstack.enhance.enhance` entry point with
their real cached bridge outputs fires joint recovery 10/10 and contour
reconstruction 0/10, reproducing every P9 score exactly. A real 14-frame
MobileDepth zero-motion sweep reports `requires_two_frames`, launches no bridge or
veil solve, leaves the contour gate silent, and is byte-identical. A clean wheel
build contains the solver plus both bridge scripts. Public `fuse_blend` /
`fuse_decision` harden defaults were also corrected to the validated 0.5 used by
CLI and pipeline; retired fusion hooks and the ungated boundary flag now say
reproduction-only.

## F55 (P0) — Joint two-layer inversion escapes F54's model-class failure
The replacement solves both captured-frame equations simultaneously for sharp
foreground/background layers, estimates corrections around observed owner/far
frames, and renders the all-focus composite. It has no texture generator or
external image prior. On an 18-scene realistic-object oracle-alpha/radius rung
at max-side 512, all regularization settings have positive mean GT-SSIM
(+0.00405 to +0.00497) and 17/18 scenes improve. At the selected
`smooth=8, anchor=0.05`, true-fringe error drops 8.10 gray levels on average and
smooth-region false-texture error drops 0.0268; the sole SSIM loss is
`scene_31` at −0.00015, versus the retired hybrid's −0.0304 failure.

The observed far frame is the correct conservative background anchor. Replacing
it with the already-fused result worsens both recovery and false-texture tails:
fusion is an output to beat, not independent evidence. A 0.5 px low-pass on the
*correction only* reduces the remaining P0 loss to −0.000067 and the positive
false-texture tail to +0.044 gray, but stronger filtering introduces boundary
frequency error.

The full 100-scene oracle P1 survives: mean GT-SSIM +0.00408, 99/100 positive,
mean true-fringe error −7.48 gray. The one loss is −0.000067; two scenes have
small positive fringe-error tails (+0.03/+0.59). False-texture error improves
−0.0274 gray on average but has a +0.092 tail. Eye inspection shows that worst
false-texture row has a visibly correct broad veil reduction and only 89 metric
pixels at the high-contrast contour; `scene_31` does show the weak boundary
mismatch predicted by its negative scores. This is the first positive
realistic-object ceiling after F54, not a promotion: analytic solver-uncertainty
control, blind inputs, native resolution, refusal, and real identity tests remain
open.

P2 then converts regularization sensitivity into an analytic uncertainty
projection: solve at three defensible strengths, keep only components whose sign
agrees, and use the smallest magnitude. This closes the global oracle tail:
**100/100 GT-SSIM positive**, mean +0.00383, worst +0.000053. It fixes
`scene_31`'s fringe miss; only `scene_40` retains a small +0.35 gray fringe loss
alongside +0.00029 global. False-texture mean/tail improve to
−0.0243/+0.0654 gray. The remaining positive false-texture cases are localized
boundary-frequency discrepancies, not the broad pattern extension visible in
F53, but remain part of the audit. This is evidence that solver disagreement is
useful uncertainty, while forward residual alone is not.

P3 exposes the next wall: the current semantic matte/owner chain is not an
identifiable input to the solver. With semantic matte but factory-true radius,
98 candidates average −0.00455 GT-SSIM (worst −0.0430), mean fringe error
worsens +0.74 gray, and only 16 improve both verdicts. Low mean alpha error does
not certify edge/ownership correctness: `scene_60` reports 0.006 but loses
−0.00869. Every positive candidate has owner index 0 only because this factory
always stores the near-focus frame first; gating on the index would be data-order
leakage, not physics. Direct semantic integration is rejected. The next
mechanism test is correction consensus across plausible matte displacements;
failure there redirects to observation-fitted alpha or better matting.

The displacement consensus does fail: erode/original/dilate stability reduces
bad magnitudes but cannot rescue wrong region/owner models. P4 finds a more
specific opportunity in *candidate ranking*. A good (`|alpha error|<0.05`) mask
exists in 56/98 top-4 banks versus 28/98 top-1; minimum post-solve forward MAE
selects 53 good masks. Yet its outcomes still average −0.00488 with only 24/98
positive, proving physical reranking cannot turn an inaccurate boundary into an
accurate one. A simple order-invariant development rule (semantic
score/purity/area-fit plus forward-residual ratio) isolates 7/98 all-positive
cases. Because those thresholds were chosen after inspecting the full set, this
is a calibration lead—not held-out evidence. A fresh factory extension is
required before any gate claim.

The fresh 25-scene mixed-regime holdout validates that narrow claim. The frozen
four-feature rule fires 2/25, both giant-CoC; after regularization consensus,
the outcomes are +0.00303/−7.61 gray fringe and +0.00075/−2.13. Thus a
high-precision semantic subset exists under true radius. Smooth-region
false-texture still rises +0.029/+0.017 gray at the boundary-frequency tail, so
this is not a zero-tail promotion. Broad radius consensus (1.2–4.5% image CoC)
is a conditional negative: it erases most recovery and flips one development
and one holdout fire negative. Blind radius remains the admission blocker.

P6 replaces radius estimation with regime refusal. A fixed 3.5%-of-image CoC
model plus the frozen observation license reproduces all 9 giant fires and
rejects all 54 moderate scenes across development + fresh holdout; every
max-side-512 fire improves global and fringe fidelity. This conditionally
resolves radius for a narrow specialist, not arbitrary stacks.

Native P7 catches another scale bug. Fixed 512 px priors yield 3/9 negative
global scores and +0.064 mean false-texture. Scaling regularizer σ, correction
σ, and blend-mask σ by resolution restores fresh holdout 2/2 positive and cuts
false-texture to +0.009. Development is 5/7 global-positive, but all 7 improve
true-fringe L1 (mean −4.26 gray); visual panels show the physical veil moves
toward GT without F53-style pattern extension. The remaining −0.00063/−0.00066
global SSIM rows are a real metric disagreement at a thin contour, not grounds
to overclaim. Production-PSF mismatch is the next mandatory rung.

P8 closes that rung with component consensus across the factory box PSF and the
production downscaled-disk PSF. Fresh native holdout remains 2/2 SSIM-positive;
all nine fires improve fringe L1 and false-texture stays below +0.015 gray.
Two development SSIM rows remain negative, but direct global physical error
overturns that isolated verdict: MAE −0.0526/−0.0468 gray, MSE −1.637/−1.334,
PSNR +0.175/+0.072 dB, and net changed pixels closer to GT
+28,781/+46,711. Eye panels agree with direct error. This is the concrete
post-doctrine benchmark case: retain the dissenting SSIM result, but admit the
operator on the full physical evidence lattice rather than optimizing SSIM.

## SYNTHESIS — current best understanding (read this instead of F1–F23 in sequence)

**The mission (2026-07-22).** The goal has graduated: produce the image TRUE TO THE
PHYSICAL OBJECTS captured — not the best mix of what the camera received. Selection
is the floor; model-based synthesis (remove modeled corruption, restore attenuated
amplitude, re-composite where truth survives) is the ceiling being built. At this
level the evaluation apparatus is part of the method: no-ref metrics punish scene
recovery (F45), pseudo-GT inherits its toolmaker's ceiling (17a), and negatives are
conditional on their bench (F27→19). Only analytic factories, observation-domain
audits, and the eye can see the frontier.

**The engine.** Every fusion method = a DECISION (which frame is sharpest, where) +
a RECONSTRUCTION (combine without artifacts). Four properties determine quality:

1. **Edge-aware decision** (guided filter) — else speckle/seams (`max`'s failure).
2. **Confidence-hardened decision** (`harden`) — else defocus *spread* bleeds in and
   thin structures (hairs/wires) grey out. One mechanism, both failure modes.
3. **Multi-scale decision** — else the method fails at whichever scale its window
   wasn't sized for (the cause of every low↔high-res ranking reversal we hit).
4. **Multi-band reconstruction** — else seams at region transitions.

`perband` (the default) is the only method with all four → best all-rounder,
GT-validated at low res (0.9918) and high res (0.9021). `blend` lacks (3) — one
single-scale decision broadcast to all bands. `pyramid` lacks (1)+(2) — halos.
The scale-theory arc (global magic number F19 → measured local-scale map F20 →
per-band F22) resolved cleanly: **put scale-adaptivity in the band structure
itself** — a fixed small window per band IS local scale, at every location, with
no scale-estimation step and no magic numbers.

**The metric.** The global composite is now **0.3·Q_ABF_MS + 0.7·Q_SSIM** (F26:
multi-scale Q_ABF — gradient transfer per pyramid level, mean-pooled — fixed the
high-res collapse the same way perband fixed the engine; best at BOTH regimes,
+0.785/+0.869). **Q_SSIM alone** for all per-region/local decisions (Q_ABF
anti-correlates locally, F12); Q_MI rejected outright; no-ref magnitudes are
suspect on atypical content (near-black microscopy) — trust orderings confirmed
by eye. With true GT available, **GT-SSIM is the verdict** — over both no-ref
metrics and the unaided eye's sense of "clean."

**The eye.** Aggregate metrics hide localized artifacts (halos <1% of pixels); the
unaided eye misjudges fidelity (F21). **Eye-analysis 2.0** (`eyetool.py`): crop
where methods *disagree most* + amplified-difference views (+GT when available) —
point the eye at the informative pixels instead of guessing crop locations.

**Structure & operators.** `harden` (confidence-hardening) unifies spread-rejection
and thin-structure preservation. `content_aware` routes laplacian↔mod_laplacian by
local contrast. Operator choice is LOW-leverage; structural/scale handling is where
the quality lives. **Exposure/WB drift is corrected by default** (F28: defocus
preserves the mean, so frame-mean differences are exposure — per-frame gain to the
stack median; near-identity when undrifted). N-frame stacks are a strength, not a
risk (F24/F29: quality rises with N; broad weights denoise; no spread import).

**Boundaries (E-phase, F30–F36).** The decision/reconstruction split is now proven:
residual hard-edge error is COEFFICIENT CONTAMINATION (the lens mixes both sides into
the captured pixels) — unfixable by any decision scheme (even oracle decisions lose,
F33); fixable only by matte-aware RE-RENDERING of the boundary band (F34: −22% ceiling;
F35: buildable −16% with global win on-model). But the reconstruction assumes a
formation model and must be GATED on evidence the model holds — it regresses off-model
(F36) and ships default-OFF (--reconstruct-boundaries) pending veil-evidence gating
(16c). The Boundary Engine itself (stack ∪ semantic channels, fused F=0.55, 2.6x
Canny; DA-V2 sees through defocus and camouflage) stands as a validated data product;
orthogonality of evidence channels is a design requirement, not a nicety (F30–F32).

**Routing & gates (F44–F56) — the fifth pillar.** Specialists compose; none replaces
the generalists. Each specialist is paired with its REGIME and its MATTE CLASS
(edge-stamping reconstruction needs ~px-precision C3 mattes on thin structures;
field-subtracting veil correction needs region-precision mask mattes on wide
occluders — one matte pipeline cannot serve both, F46). The historical promotion
recipe was the UNIFIED GATE RECIPE: regime-matched candidates → features incl. matte-edge
quality → ridge regression on the ACTUAL outcome from factory GT → fire margin set
on train by the every-scene property, verified held (F47: reconstruction held 5/30
fired all-positive mean +0.0083; veil 2/2 good). Safety is structural; recall grows
with features/data and is proportional to effect size (margins eat small-effect
recall first). No-ref source-similarity metrics CANNOT audit synthesis corrections
(F45) — GT-trained gates are the only valid audit until a synthesis-aware no-ref
signal exists. F54 strengthened the rule: a gate inherits the blind spots of its
labels and factory, so the historical veil gate/operator remains retired. F56's
replacement is independently licensed by semantic boundary evidence plus
observation-domain improvement. F62 first replaces mixed-base geometry with a
forward-winning focused-owner silhouette, directly reconstructs its optically
mixed front interior, and only then invokes F61's positive rear-observation gate.
Contour reconstruction is the only live enhancement specialist; the narrow
giant-veil solver remains a validated but runtime-disabled research mechanism.

**Scene recovery (F51–F62, FRONTIER 19).** The first multiplicative veil result was
a NARROW ORACLE-FACTORY win, not a general mechanism. F54's realistic-object audit
overturned the promotion thesis: even true matte + true radius can lose badly on
natural object/background combinations, while semantic-matte + true-radius hybrid
averages −0.0032 GT-SSIM (worst −0.0304). Channel-specific D and foreground-premult
removal fix two real artifacts but not the model-class error. The core deficit law
assumes the far remnant is the only surviving background evidence; actual fusion
already admits frequency-dependent background evidence from other frames. A direct
evidence-accounting replacement also fails because separating that contribution at
the matte boundary is itself ill-conditioned. **Multiplicative gain and the historical
subtraction branch remain retired**: their four native-resolution fires are
microscopic and fail the expanded worst-case/fringe/false-texture property. Their
successor does not repair fusion: it jointly solves both
captured formation equations for observed foreground/background layer corrections.
Regularizer + cross-PSF component consensus, a frozen physical candidate license,
fixed giant-CoC refusal, focused-owner geometry refinement/front reconstruction,
and ordered rear visibility yield spatially safe
exact-disk V2 fires across development, holdout diagnosis, and a fresh post-rule
extension (F61). The giant-veil branch remains safety-disabled because its
fine-band complement metric still detects a small positive tail. The key
limitations are both narrow recall/scope and unresolved finest-band error:
two frames, <=1600 px, giant CoC, semantic bridge, licensed candidate only.
The general lessons are joint evidence accounting, analytic uncertainty, and—
most importantly—null-space/false-texture auditing before any scene-recovery claim.
Stack gaps (F52): where NO frame is sharp, one Wiener FFT at the known disk scale
recovers +0.054 gap-SSIM (worst +0.034, off-gap zero); RL's iteration knob hits
Lucy's noise-fitting turnover; radius error degrades gracefully (±15% keeps most).
Gap recovery awaits its gate + blind DFF radius before ship.

**Learning.** Classical foundation first. Learned per-tile routing matches the
oracle; distillation matches classical quality in one pass (speed win needs a GPU);
per-tile no-GT labels are unreliable (~19% GT agreement) → train on GT dev-labels,
deploy feature-only. No answer key at inference — fully achieved.

---

## F54 — User-caught hallucination overturns the veil promotion thesis; auto veil branch safety-disabled
The F53 mechanism fixes were real but insufficient. The required harder judge was
the 100-scene objects-as-occluders factory, not the four simple wideocc blobs that
made F51 look general. With runtime semantic mattes but the FACTORY-TRUE radius
(an easier-than-shipping oracle), 98 candidates at max-side 512 give: subtraction
mean dg −0.00025/worst −0.00937; corrected multiplicative hybrid mean
**−0.00319**, median −0.00146, worst **−0.03041**, only 20/98 positive; true-fringe
error worsens +1.62 on average. Low matte mean-error is not safety: scene_31 has
|αerr|=0.0096 and still loses −0.0304. A true-alpha + true-radius spot audit also
contains severe losses (to −0.032), proving this is not merely a blind-radius or
semantic-matte problem. The wideocc all-positive result was a benchmark-class
conditional, and F51's “division redeemed” generalization is withdrawn.

Mechanism: `coef = G − w_far` restores the entire estimated scene deficit from the
far remnant while ignoring background information already present through the
other frames' frequency-dependent fusion contributions. It can therefore
double-restore natural background structure. A new direct evidence-accounting
formula was implemented and stopped after its first 14 scenes all worsened: owner
background/foreground separation at the matte boundary is itself ill-conditioned.
Radius-bank consensus likewise helped giant veils but harmed moderate scenes; raw
forward residual was radius-biased; a single blind radius fit had mean relative
error 0.43. These are mechanism negatives, not knobs awaiting a sweep.

The SHIPPED subtraction path was then audited separately at native resolution under
its exact locked gate (four fires among 366 candidates/100 scenes). Global dg was
{+0.000059,+0.000030,−0.000003,+0.000150}; one fire worsened true-fringe error by
+0.031; smooth-region false-texture deltas were
{+0.0005,+0.0011,+0.0117,+0.0101}. Its benefit is too small to spend trust on and
the old every-fire property no longer holds under the expanded instrument. A
pattern-limited low-pass D retained tiny gains but did not consistently reduce
false texture, so it was not promoted.

**Shipping action:** `VEIL_AUTO_ENABLED=False`; `--enhance auto` no longer launches
the semantic bridge or fires veil subtraction, and reports
`veil_disabled_safety=True`. Contour reconstruction remains live. The veil model,
gate, and experiments remain reproducible but cannot affect product output. New
tests make any accidental bridge call fail. The showcase and outward claims must
describe the veil arc as a retired research result until a different identifiable
model passes natural-object oracle, semantic-matte, native-resolution, false-texture,
and worst-case gates.

The benchmark lesson is larger than this feature: selective contrast metrics and
mean SSIM both missed the user's visible failure; re-degradation can also miss
null-space invention. Every restoration bench now needs the metric complement,
model-class diversity, native resolution, per-scene tails, and an explicit
identity/refusal baseline before “shippable” is uttered.

## F53 — User-caught over-extension artifact → chromatic-D model fix + the false-texture instrument (bench blind spot closed)
Farron spotted occluder texture extending past the silhouette in the F51 evidence
crops — flagrant to the eye, invisible to every metric in the bench. Three-layer
diagnosis: (1) pm-residual — the subtraction remnant is ab·pm_b + (1−ab²)·far and
the gain amplified the blurred-occluder-texture term; analytic fix (subtract
ab_k·L_k(pm_b)) real but minor. (2) CHROMATIC model mismatch, the main term: the
factory renders ca=0.04 (per-channel disk radii {0.66,0.70,0.74}·max_r); the
channel-shared D left a purple/green mottle residual the gain amplified outward.
build_D_ca (per-channel ab/pm/D + per-channel gain) flips the hybrid's fringe
|err| from worse-than-subtraction to BETTER (rough_texture 43.6→37.7) and
improves the home-regime worst case (8-bit m2 −0.0015→−0.0011; float worst
−0.0004 ≈ tie). Mottle reduced, not eliminated: deep-veil zones stay
under-recovered (the analytic shrink scales with the gain — it suppresses its
own correction at extreme attenuation; a knowable trade, open). (3) BENCH BLIND
SPOT: contrast_ratio EXCLUDES textureless-GT pixels — exactly where false
texture lands. New instrument: **false-texture index** (band-texture energy of
the output over fringe pixels where GT is smooth). Verdict on the full matrix
(final2): in-regime hybrid ft 0.62–0.67 vs GT's own 0.60 — the artifact is at
noise level after the fix; OFF-regime ft 0.79–0.85 vs GT 0.66 — the index
cleanly flags cross-regime firing as hallucination-adding, in float too
(structural, not quantization). The index would have caught the artifact
automatically; it joins the bench for all synthesis work. Note for the honest
record: home-regime dgs vs the NEW baseline is smaller than F51's headline
(+0.0011 vs +0.0019 float) because chromatic-D strengthens subtraction itself —
the baseline moved, the total win grew. Blind-side implication for 19d: real
data needs ca estimated or absorbed per-channel into the matte (19e).
**Superseded by F54:** those fixes cure two visible residual terms but do not make
the multiplicative operator valid on realistic object scenes; the auto veil branch
is now safety-disabled.

## F52 — FRONTIER 20 first pass: gap deconvolution WORKS — Wiener one-shot at the known scale recovers half the oracle gap; scale selection is the open problem
Stack-gap recovery (gapfill.py): where NO frame is sharp, selection is structurally
blurred (the F33 limit); scene recovery admits deconvolution with the known disk PSF.
GAP FACTORY: real-photo GT sharp everywhere, 3 wavy depth bands {0.15, 0.5, 0.85},
frames focused {0.15, 0.85} → middle band carries r_gap=6.5px disk blur in BOTH
frames (exact-kernel regime); gap-eval eroded from seams; σ=3; 4 fine_detail
backgrounds (data/hires). P0: baseline gap contrast 0.39–0.79 (mean 0.51 — ~10x the
veil arc's headroom); ORACLE RL8 on noiseless blurred GT: +0.040..+0.067 gap-SSIM,
off-gap 0.0000 → ceiling justifies estimators. LADDER on noisy fused stacks (all
off-gap deltas 0.0000 throughout — the gap mask contains everything):
**R8 Wiener one-shot WINS: λ=0.05 → gap-SSIM +0.0544 mean, worst +0.0340** (after
the replicate-pad fix; one FFT). R6 RL monotone to k=15 (+0.0366) then LUCY'S
TURNOVER MEASURED at k=40: contrast still rising (0.740) while worst-case fidelity
goes NEGATIVE (−0.0088) — noise-fitting exactly as Lucy 1974 warned; early stopping
is the regularizer. R7 gain-controlled RL ≈ plain RL (slightly better worst case).
EYE (before/after fix): FFT circular-wrap stripes at image borders — the R8 litscan
pitfall CONFIRMED then eliminated (replicate-pad 4r); post-fix panels show real
recovery (feather barbs, bark cracks restored toward GT), faint overshoot at the
strongest edges, no off-gap change. R9 scale robustness: ±15% radius error keeps
most of the win (+0.036/+0.041), ±30% still positive — graceful, never harmful
on-model. R9 scale SELECTION: naive re-blur-residual selector is DEGENERATE toward
under-deconvolution (picks 0.7·r always: reblur∘deconv → identity as r→0) — the
uncalibrated version of Levin 2007's selector fails for the reason their learned
λ_k exists. Bound on all claims: oracle radius + on-model PSF + symmetric gap;
DFF-estimated radius, asymmetric gaps, real stacks = the next phase.

## F51 — FRONTIER 19 lands: multiplicative veil recovery REDEEMED inside the restoration system; F27's epitaph written
The MISSION's first constructive scene-recovery experiment (veilgain.py; giant-CoC
wideocc factory, oracle alpha, 8-bit first). The hybrid = 16d subtraction + clamped
residual gain in the perband loop + analytic shrink. Rung ladder, one variable each:
P0 headroom REAL (post-subtraction strong-band contrast 0.877; subtraction alone
moves it 0.919→0.923 — the structural ceiling as predicted); measured attenuation
g(ab) far shallower than 1−ab (0.64 vs 0.20 @ ab=0.8). H1 (w_far·(G−1), 36 configs):
every config beats subtraction, off-band 0.0000, but contrast plateaus 0.900 —
w_far DILUTION diagnosed. H1b: deficit form (coef = G − w_far) + sq law (1−ab², the
derived physics) → contrast ratio 1.000 EXACT, at dg −0.0006 (noise rides the gain).
H2 analytic shrink (T = m·(1+ab)·σ·c_k·coef, σ=3 KNOWN, c_k calibrated — zero blind
estimation) m=2 → dg_vs_sub +0.0004 at cr 0.956: amplitude converted to net GT-SSIM
gain. FINAL (10 backgrounds × 2 CoC × {8-bit, float}): **coc0.04/float m2 +0.0019
mean, worst +0.0006 — ALL TEN scenes positive**; coc0.04/8-bit +0.0007 mean, worst
−0.0015 (3/10 small negatives, all three FLIP POSITIVE in float — the 8-bit wall
isolated). Eye pass: corrections hug the fringe, veil-band darkness/texture visibly
restored toward GT, NO ringing/halos/banding; mild speckle in recovered dark regions.
Negatives, all mechanism-diagnosed: H1a full-image placement loses decisively
(−0.005..−0.010 — placement was half of F27's verdict; F40 idiom confirmed 3rd time);
H3 guided-denoise of the correction loses everywhere (the attenuated band is too weak
a guide at amplified scale); H4 cross-scale coherence gate ≈ null at σ=3; H5
decomposition: float OUTPUTS do not rescue the 8-bit losers → the wall is INPUT-side
structured quantization (output bit depth exonerated; R5/MAP-AC debanding stays an
open conditional rung). Off-regime (coc 0.012): ungated harm (mean −0.0012, worst
−0.0034; strong-veil band nearly empty on 7/10) — F46's third rhyme: specialists fire
only in their regime; gate retrain on HYBRID outcomes is the promotion path.
Provenance: 10 real-photo backgrounds (data/hires: fine_detail×4, foliage×2,
metal_specular×2, rough_texture, hard_edges), blob occluders, disk PSF, σ=3, oracle
alpha (blind-matte chain is the next phase). F27's epitaph, verbatim as FRONTIER 19
hoped: THE IDEA WAS RIGHT, THE SYSTEM AROUND IT WAS MISSING — division redeemed by
placement (in-loop, w-aware), form (deficit coefficient), law (measured attenuation,
not assumed), clamps (t0/ω/fringe-mask), and an analytic noise model.

## F50 — F49-class audit: five drift points between shipped product and evidence, all fixed
Systematic sweep for default-vs-evidence mismatches and promised-but-unwired
functionality, inspired by the harden catch. Found and fixed: (1) fuse_perband's
LIBRARY default was still harden=0.0 (pipeline flipped in F49 but library callers got
the unvalidated config) — now 0.5; (2) --boundary-out was promised in the 16b plan and
task ledger but NEVER WIRED — now shipped (stack-evidence boundary map as a data
product, mirroring --depth-out); (3) --reconstruct-boundaries still advertised as
"experimental" though superseded by the gated --enhance path and known to regress
off-model (F36) — now marked DEPRECATED with the honest reason; (4) --fast still
claimed "quality-neutral-or-better" — true vs the OLD blend default, false since
perband: measured −0.005..−0.025 GT-SSIM — help now states the cost; (5) --levels help
claimed pyramid-only while blend consumes it and perband ignores it — corrected.
Verified clean: the F26 metric composite adoption DID land (metric_weights.json).
Meta-lesson (extends F49): help text and speed-preset claims are EVIDENCE CLAIMS —
when the default method changes, every comparative claim in the CLI is invalidated
and must be re-measured, not just the benchmarks.

## F49 — harden defaults ON (0.5): the shipped default now matches the validated configuration
User-caught inconsistency: every benchmark, both specialist gates, and the composed
pass were validated at harden=0.5, while the CLI/pipeline default was 0.0 — the
shipped default was a configuration no evidence covered (and the pipeline was even
overriding enhance's internal 0.5 with it). Regime-spanning A/B before flipping:
Real-MFF x15 mean +0.0001 (worst −0.0003 = noise), objocc −0.0001, microscopy −0.0001,
thin-structure hires_mixed **+0.0054** (harden's home mechanism, F9). Default flipped
to 0.5 across CLI/pipeline; help text corrected (applies to perband too); showcase
run-line now shows how to DISABLE it instead of enable it. Lesson: when a config
value appears in every experiment's invocation, it has de facto become part of the
method — promote it to the default or the shipped product silently diverges from all
published evidence.

## F48 — Composed pass + --enhance auto SHIPPED: the half-marathon closes
The composed two-specialist stage is in the package (enhance.py; pipeline default
enhance="auto", perband only, --fast forces off; identity when bridge absent or gates
silent — tested byte-identical). Gate finalization took full distribution coverage:
the recon gate is now trained on THREE families (canonical thin, varied-CoC/mixed,
object-only where the right answer is NEVER FIRE — the F46 lesson completed: partial
expansion had made the object-contour mistake MORE confident, −0.0217, before the
object family taught refusal). Final gate: 320 labeled scenes, train 23/240 fired
all-positive, held 19/80 with 18/19 positive (min −0.0007). COMPOSED VERDICT on 75
unseen scenes: objocc-held clean (recon correctly silent), thin-held clean, mixed 16
fires with wins to +0.020 (mean fired ≈ +0.006) and TWO persistent outliers
(−0.0024/−0.0033) that survived three distribution extensions — documented as the
honest worst-case bound, not distribution-fixable with current features. REAL-DATA
SPOT SWEEP through the shipped path: 13/14 byte-identical (10 Real-MFF GT untouched,
kitchen/largemotion/microscopy untouched); ONE fire on the fence (the one image with
true veil physics), tiny footprint (0.012 mean diff), metrics unchanged — eyetool crop
queued. Ship stance: default auto justified by strongly-positive mean, bounded
worst-case (−0.0033 on 2/75 hardest synthetics), and real-data conservatism;
package-vs-research parity proven (identical mattes, 0/30 fire flips). iphone12 GT
remains the promotion gate for any stronger claim.

## F47 — BOTH specialists gate-locked under one rule: outcome regression + property-driven margin
Reconstruction's gate, built on its home regime (120-scene thin-structure factory, C3
difference-mattes, no bridges): ungated hit rate 56% with 10x the veil's effect sizes;
edge-quality features (transition-shell sharpness + silhouette-on-edge alignment — the
F46 theory features) lifted ranking materially; the closing step was the PROPERTY-
DRIVEN MARGIN (train-only: fire threshold above the worst harmful prediction + eps):
train min dg −0.0002, **held 5/30 fired ALL positive, mean +0.0083**. The same rule
retro-applied to the veil gate excludes its one violator: held 2/2 good, worst +0.0000
— at a real recall cost (17→4 scenes; small effect sizes mean the safety margin eats
proportionally more recall). UNIFIED GATE RECIPE, now standard: (1) candidates from
regime-matched matting; (2) features incl. matte-edge quality; (3) ridge regression on
the ACTUAL outcome (dg) from factory GT; (4) fire margin set on train by the property,
verified held. Recall grows with features/data; safety is structural. Coverage today:
reconstruction ~17% of its regime (large wins), veil ~4% (small wins). Next: composed
two-specialist pass + --enhance packaging; iphone12 GT remains the real-data verdict.

## F46 — Cross-regime gate test CONFIRMS the scale-split adversarially; specialists need regime-matched mattes
Ported the outcome-gate to reconstruction ON THE WIDE-OBJECT factory: fired 39/100,
outcomes overwhelmingly harmful (contour errors explode, worst global −0.086) — and
that is the 16d scale-split theory PASSING an adversarial test: wide occluders'
boundary error is HAZE (veil's domain); their defocus-softened contours cannot be
re-rendered from region-precision mattes. The two specialists have OPPOSITE matte
requirements: reconstruction stamps a hard edge (needs ~1-2px matte precision — C3
difference-mattes on thin structures, its proven regime F35); veil correction
subtracts smooth fields (region-precision mask mattes suffice — F44/F45). Corollary:
one matte pipeline cannot serve both; routing must pair each specialist with its
regime AND its matte class. Reconstruction's gate will be built on a thin-structure
factory with C3 mattes. (The ridge gate also failed to rank these outcomes — its
features carry no contour-precision signal; a matte-edge-sharpness feature is the
missing input for any future cross-regime firing.)

## F45 — Fire-rate refinement locked: per-candidate outcome-regression gate; no-ref self-audit is INVALID for synthesis corrections
Checkpoint deepening (100-scene factory, 366 candidates, 2 CoC regimes), seven gate
iterations, each from a diagnosed failure: per-candidate routing (top-K masks gated
independently, corrections compose per owner group) lifted coverage 6/42→17/100 with
the RIGHT regime skew (14/58 giant-CoC vs 3/42 moderate — firing where P0 says the
effect exists); outcome-REGRESSION (predict dg, fire above +3e-4) replaced label
proxies entirely. Residual: ONE violator in 100 (scene_46, −0.0027 global) whose
features are twins of winners — irreducible with current features. Attempted fix
worth its own lesson: a runtime q_ssim output self-check REVERTED THE GT-VERIFIED WINS
along with the loser — **no-reference source-similarity metrics cannot audit a
correction whose success means deviating from every source** (the de-hazed output
matches GT, not the hazy far frame). GT-trained gating is the only valid audit until
a synthesis-aware no-ref signal exists. Ship stance per DEVSTYLE: veil correction
stays default-OFF (one-in-100 known violator); the gate + machinery are locked and
packaged-ready; promotion path = feature to split the scene_46 twin OR iphone12 GT.

## F44 — Outcome-trained routing gate satisfies the every-scene property (T2 complete)
The routing layer works when the gate predicts THE OUTCOME, not a proxy. Arc: (a) matte
lands 3/7→8/28 after ring-contrast + owner-fix (interior majority winner); (b) 42-scene
factory + snap-consistency feature + margin-tau: still one held false-fire — because the
aerr<0.05 LABEL was a proxy (a small mean alpha error can hide a misplaced edge);
(c) LABEL = the chain's actual outcome on factory GT (fringe improves AND global drop
< 0.0005) → logistic gate at tau=0.823: 6/42 fire, EVERY fire a win or neutral, worst
global delta −0.0001. The every-scene ≥ baseline property that default-ON requires now
HOLDS on the honest object benchmark. Conservative firing (14%) is the correct trade —
withheld wins cost nothing, regressions cost trust; recall can grow with features/data.
Also: fringe-band D clamp (physics hygiene, kept), outcome labels cost nothing extra
(the factory holds GT). Remaining for default-ON: T3 all-regime no-harm gates + T4
--enhance auto packaging (identity when bridge absent / gate silent).

## F43 — Mask matting v1: benchmark pathology CONFIRMED by looking; benchmark must upgrade to objects-as-occluders
T1 (FastSAM masks + seed/depth selection) scored WORSE than depth-thresholding
(|a err| 0.18-0.45, owner wrong on all 8) — and one look at the mask visualization
explained everything: (1) FastSAM fragments/misses our pasted texture-pastiche blobs
(they are not objects; SAM-class models segment real-object statistics); (2) it
segments every BOKEH DISK in the background photos as an object, and DA-V2 rates those
bright circles near → the selector picks background masks. The wideocc benchmark is
simultaneously too artificial for semantic models AND carries real-photo artifacts
that spoof them — the F32/F42 pathology line, now terminal for this benchmark as a
semantic-matte judge. FIX (next): OBJECTS-AS-OCCLUDERS generator — FastSAM cuts real
objects (true silhouette = GT alpha) from source photos; composite those with the
existing defocus physics. Also fixes T2's label factory. Infra shipped this stretch:
bridge_masks.py (FastSAM, torch/torchvision pair fixed 2.13/0.28+cpu), maskmatte.py
(seed/depth mask selection — logic reusable once the benchmark is honest).
T0 note: Sync.com is JS/E2E-encrypted — iphone12 requires a one-time MANUAL browser
download (link in REAL_DATA.md); nothing else blocks on it.

## F42 — Blind veil chain PROVEN end-to-end (1 scene); matte reliability blocks default-ON
16e stack-seeded semantic matting (focus seeds pick the owner + depth range; DA-V2
provides dense boundaries; components must contain seeds): when the matte lands
(|a err|=0.008, scene 02_c0.012), the FULLY BLIND chain improves both fringe
(26.8→24.8) and global (0.9652→0.9655) — the complete mechanism works with no oracle
anywhere. But the matte lands in ~2/8 scenes: global Otsu fails on bg-internal depth
(F32 pathology), and seeding fixes owner selection only partially (giant-CoC scenes
still flip owner). Self-gating attempt: NO candidate confidence signal (depth margin,
seed coverage, compactness) separates good from bad mattes — the worst matte has the
HIGHEST depth margin (confidently wrong owner). Per DEVSTYLE, no threshold-hunting on
n=8: default-ON is not evidence-supportable today. SHIPPED: fuse_perband(veil_D=,
veil_far_idx=) — the weight-scaled in-loop correction as identity-gated package
infrastructure (byte-identical when None, tested). Path to default-ON, concretely:
(a) segmentation-mask matting (SAM-class) instead of depth thresholding — masks are
the object-shaped signal, depth only orders them; (b) a learned matte-confidence
model trained on our GT scenes (we have unlimited labeled mattes from the generators);
(c) iphone12 GT for real-data verdicts. The half-marathon checkpoint stands at:
mechanisms proven (16b/16d), infrastructure shipped, matte = the last wall.

## F41 — Veil correction: mechanism fully validated; the matte is (again) the missing input
Completing the 16d ladder isolated everything. O2 (TRUE alpha + estimated content —
near premult from the owner frame, far observed): fringe AND global improve on all 8
scenes, both CoC regimes (e.g. 44.9→38.2 fringe, 0.9430→0.9464 global) — estimated
content is good enough; the matte alone separates success from failure. P2 (estimated
alpha, two iterations: coherent-winner matte, then decisive-label propagation via
distance transform — the F26 borrow-from-nearest-signal move): alpha err stuck at
0.09-0.29 because sky-scale TEXTURELESS regions are fundamentally ambiguous to focus
evidence; global regresses. Conclusion: weight-scaled in-loop veil correction is a
proven mechanism awaiting a proven matte — and the semantic channel's demonstrated
strength (F31: crisp closed wide-object boundaries, camouflage included) is exactly
this input. Design forward: correct_veil accepts an external alpha (two-pass semantic
bridge; classical fallback stays off); P3 gates + packaging once semantic-alpha lands.
Pattern note (three arcs now: 16b, 16c, 16d): every reconstruction-family mechanism
validates on oracle mattes and stalls on classical matte estimation — the matte IS the
frontier, and it is semantic.

## F40 — Veil correction (16d): weight-scaled forward-model subtraction — oracle ceiling strong; band-limitation premise refuted
P0 (measure first): wide-occluder haze is REAL only at giant CoC (fringe = 19-25% of
total error at r=0.04·dim, with the predicted coarse-band bump; at moderate CoC the
fringe is 6-9% and mostly fine-band = reconstruction's domain). P1 (unweighted D
subtracted from the fused pyramid, band-limited): FAILS — worse everywhere at moderate
CoC, marginal at giant. Diagnosis → O1b: the haze enters the output ONLY through the
far frame's per-band fusion weights, so the correction must be w_far-scaled and live
INSIDE the perband loop (Farron's slots-into-the-band-machinery instinct, exactly).
O1b (exact D, weight-scaled, ALL bands): fringe −18..−37%, **global SSIM +0.005..+0.019
on every scene in BOTH CoC regimes** — the strongest oracle ceiling since F34. Theory
correction, logged honestly: the band-limitation premise was WRONG — F27's amplifiers
come from DIVISION; forward-model SUBTRACTION is safe at every band. The F27-evasion
is subtraction-not-division, not band exclusion. (Coarse-only windows barely help.)
P2 (blind estimator: alpha from coherent coarse winner region — wide occluders are the
EASY matting case — near premult from owner frame, far observed) running next.

## F39 — Veil-evidence licensing: rigorous negative (category mismatch), reverted
Built the 16c veil-evidence check (per component: far-frame energy suppressed inside
the veil vs sharp outside). Both formulations (whole-ribbon medians; veil-core vs
outside-ribbon) no-op'd EVERYTHING — including the on-model occ win. Mechanism, fully
understood: THIN structures never produce a strong veil (blurred alpha of a 1-3px line
peaks ~0.1-0.3 — F27's "thin-structure haze is small" rhyming back), so an
energy-suppression license can never fire exactly where reconstruction wins; the
on-model win's mechanism is SHARP-CONTOUR RE-COMPOSITE, not deep-veil removal — the
license verified the wrong physics. Reverted to the F38 state (plane-step + energy
floor: −16% on-model, real-neutral). A correct license must be SILHOUETTE-side, not
veil-side: contour sharp in the owner frame + multi-plane gap across it (already in
the plane-step ribbon) + owner-side matte contrast — plus GT verdicts (iphone12) as
the promotion gate. Default stays OFF; the check design is the next 16c iteration.

## F38 — Reconstruction on real stacks: gate arc lands strongest on-model result; real-data now neutral, benefit unproven
Tested the "true beast" on real handheld sweeps. v1 fired on OBJECT INTERIORS (kitchen
alphafire eye-check: label text, in-focus texture — dominance = ordinary DOF at that
plane, F36's mechanism on real data; 5.9% of pixels). Gate iterations, each from a
measured mechanism: (1) depth-evidence-only ribbon — vacuous on deep stacks (winner
flips are UBIQUITOUS at N=12: each object at its own plane is continuous depth, not
occlusion); (2) absolute plane-step on guided depth — killed the N=2 on-model win
(guided smoothing erases the step, F30 blindness); (3) **raw-winner median-filtered
plane-step** (occlusion = multi-plane jump; N=2 flip = full jump — one rule, both
regimes) + **energy floor** (steps only count where focus is decisive — textureless
argmax is noise, F26): occ bband e2 −16% AND global +0.003, improving on ALL scenes on
BOTH metrics — the strongest on-model result of the arc — while real-data impact is
NEUTRAL (q_ssim ±0.001, ~1 gray-level diff on ~5% of pixels) and fire visibly moved
from interiors toward true silhouettes (eye-checked; specular/noisy surfaces still
misfire). Honest status: reconstruction does no real-data harm and its on-model
mechanism is strong, but real-data BENEFIT is undemonstrated — remaining: per-component
VEIL-EVIDENCE confirmation (does the far side actually show a veil?) and GT verdicts
(iphone12). Stays default-OFF.

## F37 — FIRST REAL-HANDHELD RESULT: blend beats perband (misalignment robustness is a new regime axis)
mobiledepth quick suite (real optical, N=12-14, graded handheld motion, no GT; F25
protocol: q_ssim ordering + eye): blend > perband > pyramid on ALL four sequences, and
the blend-perband gap GROWS with motion (zero +0.001 / small +0.002 / large +0.016 /
kitchen +0.023). Eye-confirmed on kitchen: perband shows double-edge GHOSTING (doubled
text strokes, pot-rim echoes) — residual misalignment leaks through per-band fine
decisions, which can flip per band on offset content; blend's single coarse decision
broadcast to all bands picks one frame's content consistently → ghost-free, slightly
softer. Our four-property theory implicitly assumed ALIGNED frames — misalignment
robustness is a FIFTH regime axis no synthetic benchmark tested (all were perfectly
aligned). Both methods score low absolutely (kitchen 0.89-0.91): alignment residuals +
focus breathing, not fusion, are the dominant real-data quality limiter → FRONTIER 7/8
now top practical priority. NOT flipping the default on one no-GT dataset: iphone12
(real photographic + pseudo-GT) is the decider; also motivates a misalignment-robust
perband variant (cross-band decision consistency). Real-data honesty: the user's
too-clean-synthetic warning materialized exactly as predicted.

## F36 — C3 gates: wins ON-model, regresses OFF-model → shipped default-OFF; gating is the successor
All-regime gates on the buildable reconstruction: occ (matte-model data) passes with
graceful radius degradation (0.5x/1x/2x: improve/best/≈baseline); fence eye shows the
target artifact genuinely fixed (wires thinner, defocus halo reduced) with mild side
effects (faint plate-ghost bands, few off-wire edits). BUT: hires_mixed regresses
(bband +6.7, global −0.017), Real-MFF −0.0035, nframe/drift −0.0016, layered mildly
harmed. Diagnosis: **focus dominance fires on every in-focus region** (the other frame
is always defocused there — that is ordinary depth-of-field, NOT occlusion), so the
matte stamps corrections onto non-occluder content; C3's formation model (matte
composite + disk veil) only matches matte-composited data. Per the plan's pre-stated
rule: shipped as default-OFF experimental (--reconstruct-boundaries;
focusstack/reconstruct.py, tests for no-op safety + off-by-default identity).
Lessons: (1) a reconstruction that ASSUMES a formation model must GATE on evidence the
model holds (veil evidence in the far frame — energy suppression at the support — not
mere dominance); (2) benchmark-matched wins do NOT transfer — cross-generator gates
are the honest test (occ_gen win + mixed_gen loss = model overfit made visible).
Successor (FRONTIER 16c): veil-evidence applicability gating + validation on real
bracketed captures (#13) before any promotion.

## F35 — Buildable matte reconstruction WORKS: C3 recovers 73% of the oracle ceiling
Closing the F34 gap took two diagnosed iterations, each driven by decomposition + eye:
C2 (B-ribbon gating + owner-guided snap) still regressed globally — mixed-oracle rungs
isolated the cause: matte VALUES were fine (est-value/true-mask ≈ baseline), the
SUPPORT was the killer (true-value/est-mask −0.033 global), and the eye-check showed
why: energy-dominance support is ~10px wide around 2px structures and merges into
blobs where structures are dense — support-based mattes can never be thin.
C3 = DIFFERENCE MATTING: inpaint the owner frame over the generous support to get a
background plate; alpha = robust-normalized |owner − plate| — nonzero exactly ON the
structure, however thin; light owner-guided snap. Result on occ (all 4 scenes win):
bband err 22.0→18.5 (−16%; ceiling −22% → 73% recovered), in-band SSIM 0.9446→0.9561,
**global 0.9410→0.9461 (improves)**. The buildable reconstruction now beats baseline
with no oracle inputs. Lessons: mixed-oracle rungs decompose a 2-factor failure in one
experiment; matte thinness must come from CONTENT (difference vs plate), not from
detector support width.

## F34 — Matte-aware boundary RECONSTRUCTION: ceiling validated (-22% bband err); matte quality is the bottleneck
16b rung B (TRUE sharp alpha, everything else from frames): bband k=2 err 21.8→16.9
(-22%), in-band SSIM 0.9471→0.9672, global IMPROVES 0.944→0.949 — the physics-correct
formulation works: out = obs_near + (1-α)(far_est − blur(far_est)) with far_est =
veil-strength blend of observed far frame and inward inpainting, overriding ONLY the
strong-veil ribbon. Three implementation lessons en route (each measured): replacing
the whole band with inpaint destroys good data (-0.08 global); α·obs_near double-counts
alpha (systematic on thin structures — obs_near already contains the composite); the
faint-veil zone must keep perband (fused output beats raw frames there).
Rung C (naive focus-dominance matte): bband ≈ baseline and GLOBAL REGRESSES (0.9221) —
false-positive support applies reconstruction where none belongs. THE MATTE IS THE
BOTTLENECK, and this recasts the boundary engine's role: not a decision-guide (dead per
F33) but the matte-support + ownership provider for reconstruction — B gates WHERE,
near-side says WHO owns the contour, the owner frame yields the sharp silhouette.
Next: B-gated matte estimation (guided-filter the side mask with the OWNER frame as
guide inside the high-B ribbon), then all-regime gates.

## F33 — E4 rigorous negative: decision-side boundary integration cannot cash the bband error
Ablation (guide-enrichment / eps-modulation / both, estimated B): null vs baseline
(29.6→29.9 err). Oracle ladder then killed the lever class entirely: a PERFECT GT
boundary map through the same levers is null (27.6 vs 27.4), and a PERFECT per-pixel
decision (true nearest-plane winner, hard weights) is WORSE (39.9, global 0.9326 vs
0.9503) — soft adaptive decisions beat the "correct" hard assignment. Mechanism: the
residual boundary error is COEFFICIENT CONTAMINATION — defocus physics mixes
cross-boundary content into the band coefficients themselves (same physics as F27's
inversion negative); no weight map, however perfect, can unmix it, and hard selection
amplifies it. Deepens F27's label: "decision-boundary error" is really reconstruction-
physics error. Consequences: (1) E4 guide/eps injections stay in the code (identity-
gated, harmless, may matter on real data with appearance-quiet boundaries) but are NOT
promoted; (2) E5's ownership rule expectations tempered — oracle_dec already embodies
boundary hard-select and lost; (3) the honest remaining levers for near-perfect hard
lines are RECONSTRUCTION-side: matte-aware rendering of the boundary band or
supersampled boundary reconstruction — logged to FRONTIER as the successor push; the
boundary engine's value (B + near-side, F31/F32) stands for depth/segmentation uses.

## F32 — Channel fusion works: fused F=0.55 (2.6x Canny); two confounds diagnosed
E3 quantitative on layered GT (tol ±3px): stack 0.443, semantic 0.400, **fused 0.551**
(max- and mean-fusion tie) vs Canny 0.215 — the orthogonal channels combine; fusion
also lifts camo recall (e.g. 0.76 on scene_04). Two confounds found and resolved:
(a) **Input quality gates the semantic channel**: DA-V2 on the artifact-ridden
winner-take-all composite scored sem F=0.31; on the clean perband fusion 0.40 →
the TWO-PASS architecture (pass-1 perband → depth net → boundary-aware pass-2) is
now evidence-backed, not a convenience.
(b) **Benchmark pathology**: layered scenes use photos as flat layers; DA-V2 correctly
sees the photos' INTERNAL 3D objects (balls, people), which layer-GT scores as false
positives (sem F=0.06-0.11 exactly on object-rich backgrounds) — the eval UNDER-measures
the semantic channel, and the ~0 focus↔DA-V2 Spearman on these scenes is the same
pathology (flat in layer-depth, 3D in scene-depth). Calibration must be validated on
real scenes; boundary-F is an intermediate diagnostic — the phase gate remains bband
error after integration (don't over-optimize the intermediate).

## F31 — Semantic depth channel visually validated: orthogonality is real
Depth-Anything-V2-Small via the .venv312 bridge, eye-checked on two probes:
(a) layered scene_00: blob object crisp with CLOSED boundaries; the hole in the blob
correctly shows far depth through it (nesting topology); the CAMOUFLAGE patch stands
out as its own depth despite texture-matching — object priors segment what appearance
statistics cannot; (b) the real fence pair, run on the FAR-FOCUSED frame where the
fence is an optical smear: DA-V2 reconstructs the fence lattice as nearest object and
players as distinct depth silhouettes — semantic priors recover geometry optically
ABSENT from the gradients. This channel supplies boundaries + near-side EVERYWHERE,
including where our stack channel is blind (F30 guide-blindness) and where appearance
is quiet. Conventions noted: DA-V2 higher=nearer (inverse-depth); focus-depth
lower=nearer -> calibration must be rank/monotonic. Next (E3 cont.): quantitative
boundary P/R/F from semantic-depth discontinuities vs stack vs fused; focus-depth
cross-calibration; then E4 integration with B as its OWN guide channel (per F30).

## F30 — Stack boundary channel doubles Canny; two honest lessons reshape E3/E4
E2 on layered GT scenes (tolerance ±3px): stack channel (defocus-robust max-over-frames
edges + winner discontinuities + focus-depth gradient) F=0.443 vs Canny F=0.215 — the
physics channel carries real information appearance lacks. But absolute F is modest and
two findings matter more than the win:
(a) **Camo-probe correction:** offset-crop camouflage leaves a texture-PHASE seam that
Canny sees (camo recall 0.76 vs stack 0.40) — the probe tests low-contrast seams, not
zero-appearance boundaries; true iso-appearance probes need texture synthesis (logged,
not built). Don't over-trust your own probe's construction.
(b) **Guide-blindness (architectural):** depth_from_focus smooths with a LUMINANCE-guided
filter → it smooths ACROSS depth boundaries that lack luminance contrast → our stack
depth evidence inherits appearance-blindness through its guide, exactly where
orthogonality matters. Consequence for E4: boundary data B must enter guided decisions
as its OWN guide channel (or via winner-consistency guides), NOT filtered through
luminance — else the integration partially collapses back to the parallel vector.
Also: edges sub-channel fires on intra-object texture (precision ceiling) — the
semantic channel's job to fix.

## F29 — Deep-stack bright-source spread (1b): no defect; harden holds at N=8
Harshest spread scene (26px CoC bright bars/dots), planes AT the depths: N=8 matches
N=2 (0.9881 vs 0.9888 GT-SSIM with harden; ring error 55.1 vs 53.6) — distant-frame
leakage does NOT import bokeh spread even under maximal stress; harden's benefit
persists at depth, mildly attenuated (F24's conf erosion). F24's open sub-case closed.
Probe-design lesson: the first probe placed focus planes BETWEEN the scene depths
(nothing exactly sharp) and returned absurd 0.42 SSIM — caught by sanity-checking
magnitudes against priors. Benchmarks must cover the depths present in the scene.

## F28 — Exposure/WB drift: real failure mode, clean theory-backed fix, promoted DEFAULT-ON
Injected realistic auto-exposure wobble (±12% gain + slight WB tilt) into a GT stack:
perband drops 0.9594→0.9340, blend 0.9583→0.9270 — drift genuinely breaks fusion.
Fix exploits a blur invariant: **defocus preserves the mean**, so within a stack any
frame-mean difference is exposure, not focus → per-frame per-channel scalar gain to
the stack-median means. Recovers to 0.9572 (−0.002 of clean) and is near-identity on
undrifted stacks (0.5 gray-level rounding; SSIM −0.0003) — the gate that justified
default-ON (--no-normalize-exposure to opt out). Caveat documented: clipped highlights
slightly break mean preservation; gains stay bounded. io.normalize_exposure, pipeline
stage after alignment.

## F27 — Occlusion de-veiling: rigorous NEGATIVE (headroom probes must decompose by CAUSE)
Probe said 57% of perband's occ-benchmark error lives in the veil fringe (2.1x density)
— so we built matte inversion: far_est=(obs−blur(near·α,r))/(1−blur(α,r)). Result:
WORSE than baseline at every rung of an oracle ladder — estimated (0.9219), oracle
α+premult (0.9401), oracle + noiseless (0.9731), oracle + noiseless + exact per-channel
PSF (0.9762) — all below plain perband (0.9437 noisy / 0.9783 noiseless). Mechanism,
fully diagnosed: (a) with THIN structures the blurred-α haze is small, so there is
little removable veil; (b) the fringe error is actually DECISION-BOUNDARY error, which
inversion doesn't touch; (c) inversion divides by coverage (1−α_blur), amplifying even
uint8 quantization up to 4x at the guard — costing more than the haze it removes.
Two lessons: **headroom probes must decompose error by CAUSE, not location** (57%-in-
fringe conflated boundary error with haze); and **oracle ladders turn a mystery negative
into an understood one** (each rung eliminated an explanation: estimation, noise, PSF).
Revisit conditions (FRONTIER 3b): large opaque occluders with wide substantial-α fringes
+ float pipeline + denoise-aware regularized unmixing — otherwise inversion stays dead.

## F26 — Multi-scale Q_ABF fixes the metric across resolutions; depth byproduct shipped
B4: q_abf_ms (gradient transfer per pyramid level, MEAN-pooled — sum lets the noisy
fine level dominate by pixel count) — Spearman vs GT: high-res +0.783 (plain q_abf had
collapsed to +0.109), low-res +0.294 (≈ plain +0.323). New composite (0.3·q_abf_ms +
0.7·q_ssim) beats the old at BOTH regimes: +0.785 vs +0.719 low-res, +0.869 vs +0.686
high-res (even beats q_ssim alone there). Adopted as the default composite — ONE
trustworthy global metric across resolutions; F17's regime split is resolved
structurally (per-region selection stays q_ssim per F12). Same cure as the engine:
scale-adaptivity in the pyramid structure.
B5: depth_from_focus + --depth-out shipped (winner-index/(N-1), guided-smoothed).
Honest scope: two-region test passes; on an N=8 continuous-gradient real-photo scene
r=0.59 vs true depth on TEXTURED pixels, r=0.27 overall — depth-from-focus is only
observable where there is texture (classic DFF limitation; flat regions are
noise-driven). A coarse byproduct, not a depth sensor; useful for masks/occlusion
reasoning, documented as such.

## F25 — Honesty checks STRENGTHEN perband: α-matte occlusion + REAL microscopy defocus
Two independent reality checks; perband's crown survived both.
B3 (occlusion-aware generator): replaced hard per-pixel depth indexing with a proper
layered α-matte defocus (a blurred foreground semi-transparently VEILS the background;
blurred alpha) — the physics leading-edge MFIF papers target. Re-ranked on this honest
benchmark: perband 0.9434 > pyramid 0.9308 > blend 0.9283 overall; near-structures
perband 0.9550 > pyramid 0.9534 >> blend 0.9221. The crown WIDENS, and blend suffers
most under honest occlusion (its single-scale decision can't handle the translucent
veil). The synthetic conclusion was not an artifact.
B2 (REAL optical data): BBBC006 microscopy z-stacks (N=3, genuine defocus, focus at
z16 ±6). perband produces visibly the sharpest fusion (recovers nuclei chromatin
texture); pyramid softens/halos the nuclei; blend between. q_ssim ordering perband 0.98
> blend 0.96 > pyramid 0.69 — direction eye-confirmed. Metric caveat (checked, not
assumed): pyramid's low MAGNITUDE is partly a q_ssim artifact of the ~85%-black
background (q_ssim fragile on near-uniform regions); the ORDERING is real, the number
inflated — trust eye+ordering. Closes the standing "real optical defocus" gap for the
microscopy domain: perband holds on non-synthetic defocus. (Methodology, recurring: an
alarming metric number was checked against the eye before any conclusion.)

## F24 — N-frame blind spot: NOT a defect — the "dilution" is beneficial denoising
Probed the biggest blind spot (everything was 2-frame; real stacks are 5–50). Result
overturns the hypothesis, and how it was overturned is the lesson.
- Quality RISES with N for every method (perband 0.875→0.959→0.977 at N=2/4/8): more
  planes = finer depth sampling + multi-frame noise averaging. perband stays best ∀N.
- H1 (harden conf-collapse): mild — conf erodes with N on structures (0.216→0.124) but
  does not collapse to 0; harden's effect is small on these scenes anyway.
- H2 (weight dilution): REAL by measurement (at N=8, weight mass on the true-sharpest
  frame ~0.35; far-plane leakage 0.50 > adjacent 0.21 — *looked* harmful). BUT the
  direct A/B — a top-K energy gate that removes distant-frame weight — makes quality
  monotonically WORSE (K=2 0.957 < K=3 0.966 < K=all 0.978). Mechanism (eye-confirmed):
  on smooth content blurred≈sharp, and averaging many frames REDUCES sensor noise; the
  broad weight distribution is doing multi-frame denoising. top-K throws that away.
- Conclusion: NO fix needed; the hypothesized defect is a strength, and a "fix" would
  have regressed both SSIM and visible noise. **Methodology win: a pathological-looking
  internal measurement (50% weight on "wrong" frames) is NOT evidence of harm — only
  the end-to-end A/B (remove it → does quality improve?) is the verdict, and here it
  validated the existing design.** Open sub-case (FRONTIER): extreme defocus-spread from
  bright point sources across many frames — distant leakage could import spread there;
  harden's domain, not stressed by these scenes.

## F23 — perband refined (correctness fixes) + promoted to DEFAULT
Fresh-eyes review found two defects in fuse_perband: (a) base band was a plain MEAN
(imports the defocused frame's low-frequency spread) → now blended with the coarsest
detail band's weights propagated down; (b) coarse-band windows exceeded the band size
(a radius ≈ the whole band degenerates the guided filter into a global mean, ~50/50
blending) → radius/energy_ksize now capped per band. A/B (v1 vs refined v2):
high-res 0.9014→0.9021, low-res 0.9919≈0.9918, fence composite unchanged but output
differs (mean|diff| 0.55). Refined perband now leads EVERY GT-measured regime:
Real-MFF 0.9918 > blend 0.9915 > pyramid 0.9913; high-res 0.9021 > pyramid 0.8926 >
blend 0.8704. On the one no-GT case (fence) the composite prefers blend (-0.005) but
**eye-analysis 2.0 (disagreement-guided crops + amplified diff) favors perband** —
sharper wires, no halo; the composite deficit corresponds to no visible defect.
DEFAULT switched to perband (pipeline+CLI; --fast now maps perband→decision).

## F22 — PER-BAND edge-aware fusion = best-of-both at ALL resolutions (Farron's insight)
Farron's point: blend's pyramid is only in the RECONSTRUCTION; its DECISION is
single-scale (one guided weight broadcast to all bands), whereas pyramid decides
per-band (multi-scale). Fix: make the decision per-band too — at EACH Laplacian band,
decide from that band's energy AND refine with a guided filter (guided by that band's
Gaussian image). A FIXED small radius per band => effective full-res radius grows with
scale automatically (the pyramid "starts at finest pixels and moves up") — multi-scale
by construction, NO magic number.
Result (`fuse_perband`): BEST at high-res (0.9014 > pyramid 0.8926 > blend 0.8704) AND
low-res Real-MFF (0.9926 ≥ blend 0.9923 ≥ pyramid 0.9920); nearly halo-free on the
Lytro fence (composite 0.9091, between blend 0.9143 and pyramid 0.9060; visually mostly
clean vs pyramid's clear halo). So it inherits pyramid's multi-scale strength AND
blend's halo-freeness — dominates pyramid everywhere, beats/ties blend except the
hardest low-res halo case (fence, -0.005 vs blend). Promoted as `--method perband`
(recommended for high-res). Kept `blend` default (still the fence halo-champion).
Lesson: give the DECISION the multi-scale structure, not just the reconstruction.

## F21 — Course-correction: PYRAMID is the best realization of the local-scale principle
Adding pyramid to the fine-structure comparison (with TRUE GT) flips my earlier
visual read: on the fine near structures pyramid is MOST faithful (near-SSIM
0.71–0.77) vs my local (0.55) and global (0.50); pyramid also wins aggregate
(0.8926). Content-routing (local at fine boundaries, pyramid elsewhere) gained only
+0.001 overall and HURT the structures (I routed toward the worse option there per GT).
Resolution crossover: low-res clean Real-MFF blend≈pyramid (0.9923≈0.9922); low-res
HARD boundaries (Lytro fence) blend>pyramid (halo); high-res pyramid>blend (+0.022).
Two lessons (added to PLAYBOOK):
  (a) **With true GT, GT-referenced fidelity is the verdict** — a "cleaner-looking"
      result (my local) can be LESS faithful than a sharper one with a faint halo
      (pyramid). The "look, don't trust the metric" rule targets NO-REFERENCE metrics
      + aggregates hiding LOCAL defects; it does NOT override GT-referenced fidelity.
  (b) **The best realization of a principle may be an existing algorithm, not your
      bespoke retrofit.** Farron's local/multi-scale thesis is CORRECT — and pyramid
      (intrinsically multi-scale) embodies it better than my explicit content-measured
      local-scale guided-blend. Don't fall in love with your own mechanism.
Decision: keep `blend` the default (halo-safe all-rounder; high-res data is synthetic-
only and pyramid halos on hard boundaries). Method choice is resolution/content-
dependent — a selector (blend low-res / pyramid high-res) is the principled next step,
pending REAL high-res optical-defocus data to confirm pyramid's high-res win isn't a
synthetic-benchmark artifact. analyze.py/local_fuse.py retained as validated research.

## F20 — CONTENT-MEASURED LOCAL SCALE beats the global magic number (Farron's thesis, confirmed)
The F19 global scale (0.012·max_dim) only helps object-scale depth splits; it DESTROYS
fine details at FINE-SCALE depth boundaries (thin near structures over a far
background) because the window is coarser than the boundary. Built the right benchmark
(hires_mixed: thin near wires/dots over real far photos, GT) and a content-analysis
stage (analyze.py): Canny edges + a per-pixel LOCAL STRUCTURE-SCALE map measured from
fine-detail energy (small on detail: 17–26px, large on smooth: 31–34px — MEASURED, not
assumed). local_fuse.py sets the guided scale per-pixel from that map (interp over K
discrete scales + multiband).
Result on fine near structures: LOCAL >> global (+0.04 to +0.14 SSIM; e.g. 08:
0.070→0.206, 09: 0.248→0.363) and **visually best — sharp AND clean**. Global blurs
them; the thesis is confirmed: measure local scale, no magic number.
Nuance (honest, and the recurring lesson): **pyramid wins AGGREGATE** (0.8926 vs local
0.8787 vs global 0.8704) because it is intrinsically multi-scale — it already embodies
the local-scale principle — BUT it HALOS on the fine high-contrast structures (visible
in the crop, hidden by the mean). So LOCAL is visually cleanest on structures; pyramid
is strongest on detailed backgrounds. Best-of-both = CONTENT-ROUTE between local-guided
(clean fine boundaries) and pyramid (multi-scale detail) — the L4 next step. Lesson to
PLAYBOOK: fusion scale must be MEASURED locally from structure; and pyramid's aggregate
again hid a boundary halo — look at the structures, not the mean.

## F19 — FIX: resolution-adaptive guided params — blend beats pyramid at high-res, no low-res regression
Root cause of F18 confirmed by experiment: scaling the guided radius + focus pool
with resolution monotonically improves high-res blend (default 0.788 → 0.80). Fix:
`radius`/`smooth_ksize` default to None → auto = `max(8/9, round(0.012·max_dim))`
(≈ the CoC). Results: **byte-identical to the old fixed 8/9 at low-res** (40/40
Real-MFF pairs — the floor guarantees no regression) and **default blend now 0.7998
> pyramid 0.7933** at 3072px, visually recovering the soft foliage detail. Promoted
(it's the package default now). Nuance: pyramid still edges it on a few extreme
large-smooth-defocus stacks (05 foliage, 08 texture) — a future content-routing
opportunity, not chased now. Lesson (added to PLAYBOOK): **every fixed-pixel-size
operator must scale with resolution/CoC**; a param tuned at one resolution silently
mismatches another. (Metric analogue F17 — scale-aware Q_ABF — noted as follow-up.)

## F18 — HIGH-RES METHOD REVERSAL: pyramid beats guided-blend at high-res (root: fixed-scale params)
On 10 high-res (3072px) GT stacks (real Wikimedia photos + depth-dependent disk
defocus + chromatic aberration + noise, CoC~37px), `pyramid` WINS overall
(GT-SSIM 0.7933, worst-tile 0.4734) vs the guided-blend family (~0.788, worst ~0.44)
— a REVERSAL of the low-res finding (where blend/decision beat pyramid's halos).
Visually confirmed on foliage: blend is softer (fine detail smeared), pyramid keeps
detail. **Root cause:** the guided-blend's FIXED-pixel params (guided radius 8, focus
pool 9) are tiny vs a 37px CoC, so the weight is mismatched to large-CoC structure
and blends in blur; pyramid is inherently multi-scale so it adapts. Lesson: the
low-res-optimal default is NOT high-res-optimal; fixed-pixel operators must SCALE
with resolution/CoC. (Testing scale-aware params next.) This is exactly the
"high-res is where it differs" the user predicted.

## F17 — The metric ALSO doesn't transfer to high-res (q_abf collapses)
Re-validating per-stack Spearman vs GT-SSIM at 3072px: q_abf +0.12 (vs ~+0.30
low-res — its fixed 3×3 Sobel only sees the finest scale), q_ssim +0.874 (STRONGER
than low-res), composite (0.3 q_abf+0.7 q_ssim) +0.714 — now DRAGGED DOWN by q_abf.
**q_ssim alone beats the composite at high-res.** Same root cause as F18 (fixed-scale
operators). Fix: scale-aware/multi-scale Q_ABF, or drop its weight at high-res.
Reinforces PLAYBOOK: don't assume the metric transfers; re-validate at each regime.

## F16 — Real-data validation: MFFW blocked; validated on real Lytro optical defocus
MFFW (real + hard defocus-spread) has no accessible download (ResearchGate page
only) — a wall, not a blocker. But Lytro (in standard/) is REAL light-field optical
defocus. On 20 Lytro pairs, `--fast` (decision + weight_scale 0.5) composite 0.9275
vs full blend 0.9285 (-0.001) and is VISUALLY INDISTINGUISHABLE (fence crop). So the
engine + speed path hold on real optical defocus. Overall validation coverage: real
content + GT (Real-MFF 710), real optical defocus (Lytro 20), synthetic + hard + GT
(benchmark). The remaining gap is real+hard-together (MFFW/UHD) — download-blocked.

## F15 — High-res CPU speedup: quality-safe ceiling ~1.5x (profiled, not assumed)
Profiled fuse_blend at 1K/2K/4K: weight pipeline (focus energies + guided filter)
~55%, multiband blend+pyramids ~42%, weight pyramids ~0%. Since weights are smooth,
subscaling helps — BUT naive full-subscale (weight_scale on everything) greyed thin
bright structures (defeats harden: -0.003 GT-SSIM on defocus_spread, visibly grey
wires) even though Real-MFF said -0.0001 (clean-data mirage again — LOOK). Fix:
subscale ONLY the guided-filter smoothing; keep focus/confidence/decision full-res
so harden still hard-selects thin structures. Result (quality-safe):
- `blend`, weight_scale 0.5: ~1.3x, Real-MFF -0.0002.
- `decision` (image-space, skips pyramids): with harden it BEATS blend on structural
  scenes (spread 0.9923 vs 0.9894) and ties on clean; ~1.2x.
- **`--fast` = decision + weight_scale 0.5: ~1.5x, Real-MFF -0.0003, better on hard.**
The dramatic 2-3x needs the quality-sacrificing full-subscale or a GPU. On CPU the
opencv engine is already fast, so ~1.5x is the honest quality-safe ceiling. Exposed
`--weight-scale` + `--fast`; weight_scale=1.0 is byte-identical (test-asserted).

## F14 — Distillation MATCHES classical quality in one pass; speed is a GPU story
Distilling the classical engine (content_aware+harden) into the same tiny FCN:
distilled CNN 0.9885 held-out GT-SSIM vs classical 0.9888 — a quality MATCH in one
forward pass (the engine IS distillable). But on CPU the CNN is 74.8 ms/img vs the
classical 34.9 ms/img (0.5x — SLOWER): the hand-optimized numpy/opencv engine
(integral-image guided filter, pyramids) is very fast on CPU, so the CNN's speed
win is a GPU/high-res-batch story this box can't show. Honest end-state: learned
fusion can reproduce the engine's quality; the "faster" half of "faster + excellent"
needs a GPU. Fulfills the staged plan's M4 and its version-bridge intent.

## F13 — M4 version-bridge works; self-supervised CNN < classical (as expected)
Provisioned py3.12 + CPU torch 2.13 via uv (torch has no 3.14 wheel) — the version
bridge works. A tiny FCN trained PURELY self-supervised (gradient-retention +
smooth-weight loss, no GT) reaches 0.9711 held-out Real-MFF GT-SSIM vs classical
0.9888. So from-scratch self-supervised learning is feasible but does NOT beat the
mature classical engine (guided edge-aware weights + multiband + content-routing +
hardening) — expected for a small CPU net vs a domain-knowledge-rich pipeline. The
route to "fast AND excellent" is DISTILLATION (train the CNN to reproduce the
classical engine in one forward pass) — testing next. Confirms the staged plan: AI
-learning rests on, and currently trails, the conceptual/algorithmic foundation.

## F12 — The GLOBAL composite must NOT be used for per-tile/region decisions
30-scene per-tile study (mean per-tile Spearman vs GT-SSIM over 6 candidate tunes):
q_abf ANTI-correlates per-tile (-0.241; -0.388 smooth), so the global composite
(0.3 q_abf + 0.7 q_ssim, great for GLOBAL ranking) is actively BAD per-tile
(-0.239; -0.642 smooth). Plain q_ssim is the best simple per-tile metric (+0.217);
a content-ROUTED metric (lowfreq on smooth, q_ssim+gradcons on textured) is best
(+0.278) — the metric-level analogue of the content_aware operator. But even the
best per-tile no-ref metric is MODEST: per-tile no-GT discrimination is inherently
hard. Consequences: (1) retroactively validates M3 (train on GT dev-labels, deploy
feature-only — don't rely on a no-ref metric to label per-tile); (2) per-region
SELECTION should use q_ssim, never the q_abf-laden global composite; (3) the "no
answer key even for per-tile training" ideal is only partly reachable now.

## F11 — Recommended engine (content_aware + harden) is non-regressing and scales
Real-MFF (200 clean pairs): content_aware+harden0.5 = 0.9914 vs default 0.9913
(neutral, worse on only 17/200) -> promoted content_aware to the DEFAULT operator;
harden stays opt-in (recommend 0.5 for bright/thin/spread scenes). High-res (2048)
hard scenes: recommended beats baseline on both (defocus_spread 0.9914 vs 0.9907;
gradient_metal 0.9615 vs 0.9582), on global AND worst-tile, and visibly restores
bright wires baseline greys out. Structural wins scale with resolution.

## F10 — Confidence-hardening UNIFIES spread-rejection AND thin-structure ("hair") preservation
The same `harden` mechanism that rejects defocus spread also preserves thin 1px
hairs: both are "hard-select where one frame is confidently the unique sharp
source." On a 1px-hair scene: baseline 0.9847 -> harden0.8 0.9874, and visually the
hairs go from faint/greyed (baseline blends toward the frame where the hair is
absent) to dark+crisp. So NO separate hair-isolation layer is needed — one
theory-grounded mechanism covers both structural failure modes. (adaptive 0.9887
still edges global SSIM but does NOT preserve the hairs; the real best is adaptive
WITH harden.) Don't-over-engineer corollary to F9.

## F9 — Defocus-spread rejection (confidence-hardening): orthogonal, stacks, VISIBLE
Where one frame is confidently sharpest (thin/bright structures), push the guided
weight back toward hard one-hot selection so the other frame's defocus SPREAD (dim
wide blob) can't bleed in; keep soft blending where ambiguous. On a harsh-spread
scene (bright near bars/dots, CoC~26px): baseline 0.9853, adaptive 0.9880,
spread_reject 0.9870, **adaptive+spread_reject 0.9888** (best). Crucially VISUAL:
baseline AND adaptive both render bright dots/bars GRAY (spread bleed); hardening
restores them bright+crisp — a big visual difference the metric barely registers
(+0.001), exactly the "metric-blind, eye-obvious" case. It's ORTHOGONAL to operator
routing and STACKS with it. Promoted as `harden` (0..1, default 0 = off) on
fuse_blend/fuse_decision + --harden CLI flag. Don't-throw-baby-out: kept adaptive,
added the layer, combined.

## F8 — M3 learned per-tile routing beats best-single-tune, MATCHES the oracle
Numpy MLP (8 content features -> best of 6 tunes), trained on GT-supervised dev
labels, deployed feature-only (no GT at inference). Held-out hard collage scenes:
learned routing 0.8580 > best-single-tune 0.8559 > content_aware 0.8554, and it
EQUALS the per-tile oracle upper bound (0.8580). Tile-accuracy is only 54% but
tunes are near-equivalent where confused, so fusion quality saturates near oracle.
Read: per-tile operator/param routing works and is near-optimal, but its CEILING
is small (+0.002 here). Bigger headroom is structural (defocus-spread rejection,
thin-structure isolation), not operator choice -> next focus.

## F7 — Content-aware operator routing = non-regressing best-of-both (promoted)
`content_aware` focus op (blend laplacian/mod_laplacian per pixel by local contrast,
c=1-exp(-ref/tau)): matches laplacian exactly on Real-MFF (0.9913, no regression) and
on defocus_spread, improves smooth-gradient metal (0.9621 vs lap 0.9586; mod 0.9675 is
the ceiling). Hand-tuned tau is a floor for M3 to beat. Committed to the package.

## F6 — Per-tile tune diversity is real; no-GT per-tile labeling is NOT yet reliable
Per-tile best-tune over collage scenes: all 6 candidate tunes win some tiles (GT dist
~[34,22,40,16,25,55]) — strong justification for learned per-region routing. BUT the
composite metric's per-tile best-tune agrees with GT's only ~15% of the time. So the
no-reference metric is fine for GLOBAL ranking (F1) yet too weak to LABEL per-tile tune
choice without GT. Implication: train M3 on GT-supervised dev labels, deploy feature-only
(no GT at inference); closing the no-GT-labeling gap needs metric refinement (future).

## F5 — Composite metric is BLIND (even backwards) on smooth low-texture content
On the `gradient_metal` hard scene (smooth brushed surface, gradual focus), GT-SSIM
ranks **mod_laplacian best (0.9675)** vs laplacian/baseline 0.9586 — a real, distributed
win (14% lower mean error, closer to GT on 57% of pixels). But the **composite metric
ranks baseline ABOVE mod_laplacian** (0.7637 vs 0.7507) — backwards. Cause: q_ssim/q_abf
need structure/edges to grade; on smooth surfaces they can't see the difference.
Consequences: (a) operator selection must be **content-aware**, not purely metric-driven,
for smooth regions; (b) the metric needs a smooth-content term or a fallback. Visual
check: the win is perceptually subtle here (both look near-GT), so this is more a
metric-trust lesson than a big visual gain — but it's exactly the "don't trust the
metric alone" case.

## F4 — Scene-dependence is REAL on hard (disk-defocus) data
Hard benchmark (disk/bokeh PSF, defocus spread, noise, 1024px):
- `gradient_metal`: mod_laplacian > adaptive > baseline (see F5).
- `defocus_spread`: **adaptive best** (GT-SSIM 0.9904 vs 0.9883 baseline), and the
  advantage is larger on the **worst local tile** (0.9719 vs 0.9616, +0.010) — adaptive
  helps most exactly in the hardest region. Local (tiled) metrics reveal wins the global
  mean shrinks.
On clean-Gaussian data (Real-MFF + synth archetypes) operators nearly tie — scene-
dependence only manifests under realistic defocus.

## F3 — Real-MFF is near-ceiling; global tuning has tiny headroom
M1 global search: composite +0.0002, held-out GT-SSIM +0.0003 over baseline (positive,
transfers — objective validated end-to-end, but tiny). laplacian dominates 198/200 scenes.

## F2 — Region-adaptive > global > baseline on GT (monotonic, small)
Held-out Real-MFF: adaptive 0.9897 > global 0.9893 > baseline 0.9889. Premise validated;
magnitude limited by near-ceiling data.

## F1 — Composite objective calibrated & validated (M0)
0.3·Q_ABF + 0.7·Q_SSIM; mean per-pair Spearman vs true GT-SSIM = +0.72. Q_MI was
**rejected** (anti-correlated, +rewards ghosting/speckle). Sharpness alone unreliable.

## Open threads / next
- HIGH-RES: re-run hard benchmark at 2K/4K — do NOT assume 1024 conclusions transfer;
  defocus-spread & thin structures scale with resolution and may reward complex algorithms.
- Content-aware operator selection (smooth -> mod_laplacian) independent of the blind metric.
- Thin-structure ("hair/wire") isolation layer so outliers don't break region partitioning.
- Defocus-spread rejection (consistency check vs sharpest source).
- Metric refinement for smooth/low-texture regions.
