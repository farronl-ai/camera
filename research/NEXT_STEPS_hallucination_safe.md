# NEXT STEPS — hallucination-safe veil recovery to shipping

Active objective (Farron, 2026-07-25): stop veil recovery from extending
foreground texture/patterns into background regions where they do not physically
exist, and admit a replacement into `--enhance auto` only if it clears the
expanded hallucination audit.

This is not a cosmetic cleanup. False detail outside the occluder is a mission
violation even when global SSIM rises. The shipping claim is therefore stronger
than "the average benchmark improved": every fired correction must remain
remnant-anchored, match the observation-domain forward model, add no measurable
false texture, and preserve the existing never-harm property across regimes.

## What is already known

- F53 found two concrete causes of the visible over-extension:
  1. a channel-shared blur model leaves chromatic residuals which gain turns into
     purple/green mottle;
  2. the post-subtraction remnant still contains blurred foreground premultipled
     texture (`ab * pm_b`), which gain can extend beyond the silhouette.
- The oracle research fix (`build_D_ca` + `pm_by_far`) reduces in-regime false
  texture to near the GT noise floor, but assumes true alpha, true radius,
  factory chromatic offsets, and known sensor noise.
- The existing shipped specialist performs subtraction only. It uses a fixed
  `0.012 * max_dim` radius and its gate was trained on subtraction outcomes, so
  neither the blind model nor the gate currently licenses multiplicative gain.
- Off-regime firing remains structurally harmful. A better operator without a
  better refusal mechanism is not shippable.

## F54 decision: the multiplicative formulation is retired

The realistic-object oracle audit overturned the prior promotion path. With
semantic mattes but true radii, the corrected hybrid averaged −0.00319 GT-SSIM
(worst −0.03041) over 98 candidates; only 20 were positive and true-fringe error
worsened on average. More decisively, true-matte + true-radius spot checks still
lost as much as −0.032. The model—not merely the blind inputs—is wrong outside
the simple wideocc factory.

`coef = G − w_far` assumes the far remnant is the only surviving background
evidence. Actual fusion already contains scale-dependent background evidence
from other frames, so the hybrid can double-restore it. A direct
evidence-accounting variant also failed because separating the owner's
background contribution at the matte boundary is ill-conditioned.

Shipping safety has been restored immediately: the subtraction veil branch is
disabled in `--enhance auto`; contour reconstruction remains live. The next
recovery attempt must replace the formulation rather than tune the gain.

## Why ordinary benchmarks are insufficient

The failure landed in pixels excluded by `contrast_ratio`; global SSIM diluted
it; source-similarity no-reference metrics score correct scene recovery as
damage; commercial pseudo-GT contains the same selection-era veil and cannot
credit removing it. A benchmark can therefore be green while the output visibly
invents texture.

Required evidence is a lattice:

1. analytic-factory GT, including the complement of every selective metric;
2. observation-domain re-degradation against the captured frames;
3. disagreement-guided visual inspection for artifacts and edit location;
4. real stacks for refusal/identity behavior, never as truth certification;
5. per-scene and worst-case results, not only means.

## Shipping requirements

- [x] **S0 — reproduce and instrument.** Reproduce the user-caught artifact and
      retain `false_texture` as a first-class label. Add a GT-free analogue based
      on forward-model residual / unsupported high-frequency energy so runtime
      code can refuse without access to truth.
- [x] **S1 — blind chromatic model bank (conditional negative).** Fit bounded per-channel chromatic
      spread for a small bank of plausible base radii from the observed
      owner/far pair and semantic matte. Do not force a point estimate: retain
      candidate disagreement as uncertainty, and reject low-confidence or
      out-of-range banks. No hard-coded factory CA. Result: point fits and raw
      residual selection are too unstable; bank disagreement is useful evidence
      but does not rescue the operator.
- [x] **S2 — blind noise + safe hybrid (operator rejected).** Port deficit-form in-loop gain,
      foreground-premult residual removal, analytic shrink, base-band exclusion,
      fringe clamp, and a strict correction cap. Estimate noise conservatively
      from observed flat-region high-pass residuals. Result: noise estimation
      works; the hybrid fails realistic oracle scenes, so gate training would
      learn around a broken model and was stopped.
- [ ] **S3 — replacement outcome gate.** Train only after the replacement
      operator has a positive realistic-object oracle ceiling. Then use regime-matched
      objects-as-occluders scenes spanning moderate, giant, chromatic, and
      never-fire regimes. Features must include radius-fit confidence,
      forward-model residual change, layer-solver uncertainty, correction
      magnitude, and the GT-free false-texture proxy. Margin is set from harmful training outcomes
      and verified on a scene-disjoint holdout.
- [x] **S4 — package safety.** `--enhance auto` no longer calls the veil branch
      or semantic bridge; it reports the safety disable and retains contour
      reconstruction. Identity/refusal is the shipped veil behavior.
- [ ] **S5 — verification.** Add focused unit tests for channel-specific
      modeling, unsupported-texture refusal, kill-switch identity, and gate
      shape. Run the full test suite plus factory holdout, off-regime families,
      composed specialists, mobiledepth refusal checks, and eye panels.
- [ ] **S6 — checkpoint close.** Distribute findings to SYNTHESIS, PLAYBOOK,
      DEVSTYLE, and FRONTIER; refresh outward-facing docs if a capability ships;
      run the F49/F50 drift sweep; clean-tree/origin/test audit; commit and push
      logical checkpoints without author trailers.

## Blind-radius result: point estimation is the wrong abstraction

Candidate model for channel `c`:

`far_hat_c(r) = disk_blur(owner_c * alpha, r) + plate_c * (1 - disk_blur(alpha, r))`

where `plate` is conservatively inpainted from the observed far frame outside a
maximum veil support. On 20 object-occluder scenes, constrained shared-radius
fits still had mean relative error 0.43, median 0.30, p90 0.67, and maximum 2.56.
A pure high-pass correlation variant was similarly poor. A raw forward-residual
selector was also radius-biased on the wide-occluder factory, repeating the
scale-selection degeneracy Levin et al. warn about for circular apertures.

Therefore a blind radius point estimate is a conditional negative, not the
shipping path. The active design is a calibrated candidate bank:

1. fit only the small chromatic spread within each proposed base radius;
2. form the hybrid correction for every physically plausible radius;
3. compute per-band correction consensus and disagreement across the bank;
4. retain only remnant-supported consensus, attenuating high-disagreement
   components as inverse-problem uncertainty;
5. use scene-disjoint factory labels to calibrate outcome and false-texture
   refusal. Raw forward residual remains a feature, never the selector.

This changes the question from “which uncertain model is true?” to “what
recovery survives the plausible model set?” It may leave some contrast
unrecovered; that is the correct trade when the alternative is invented detail.

## Active replacement: joint two-layer inversion, not correction-after-fusion

The failed hybrid tries to repair a fused image after the fusion weights have
mixed two unknown layers. The physically cleaner formulation solves the two
captured equations together for the sharp foreground premultiplication `P` and
sharp background `S`:

`O_i = H_near,i P + (1 - H_alpha,i alpha) * H_far,i S + noise`

for every focal frame `i`, then renders the all-focus scene
`P + (1 - alpha) * S`. This accounts for every frame's background transfer
before fusion instead of pretending the far remnant is the sole observation.

First rung: oracle alpha/radii, small crops, linear operators with
Tikhonov/TV regularization and quantization-bin projection. Required stopping
rule: the joint solver must beat subtraction on every realistic-object oracle
scene and reduce false texture before any blind radius/matte work resumes.
Failure at this rung retires wide-veil inversion entirely; success earns the
model-bank/refusal work above.

### F55 P0 checkpoint — a positive oracle ceiling appears

The first 18-scene, max-side-512 cross-regime rung is qualitatively different
from F54. Solving both observed frames jointly with the observed far frame as
the background anchor gives mean GT-SSIM deltas from +0.00405 to +0.00497 over
the untouched fused baseline across the regularization sweep; 17/18 scenes
improve. At `smooth=8, anchor=0.05`, true-fringe absolute error falls by 8.10
gray levels on average and smooth-region false-texture error falls by 0.0268 on
average. The remaining scene (`scene_31`) loses only −0.00015 GT-SSIM, compared
with −0.0304 for the retired hybrid under its semantic-matte audit.

Two controls constrain the interpretation:

- anchoring the background to the fused result is worse, particularly in the
  false-texture tail, because it preserves the very mixed-frequency artifact
  the inversion is meant to remove;
- Gaussian-limiting only the recovered correction at 0.5 px improves the P0
  worst case to −0.000067 and the false-texture mean/tail, while stronger
  filtering creates its own boundary-frequency error.

P1 ran the chosen far-anchor model over all 100 oracle scenes. Mean GT-SSIM is
+0.00408, 99/100 scenes are positive, and mean true-fringe absolute error falls
by 7.48 gray levels. The lone loss is `scene_31` at −0.000067; only
`scene_31`/`scene_40` have tiny positive fringe-error deltas (+0.03/+0.59).
Smooth-region false-texture error improves by 0.0274 gray on average, but 31/97
measurable scenes have positive deltas and the tail reaches +0.092 gray.

Eye inspection of the worst false-texture row (`scene_45`, only 89 qualifying
quiet pixels) shows a large veil reduction toward GT and no visible invented
pattern; the tail is concentrated at the high-contrast contour. In contrast,
`scene_31` visibly confirms a weak-scene boundary mismatch. The oracle model
class has therefore passed the “worth continuing” test but not the strict
every-scene property.

P2 treats sensitivity to regularization as inverse-problem uncertainty. Three
solves (`smooth/anchor={2/.02,8/.05,32/.10}`) vote per component; a correction
is retained only when all signs agree, at the smallest ensemble magnitude.
Across all 100 oracle scenes this makes global GT-SSIM strictly positive:
mean +0.00383, worst +0.000053. `scene_31` also flips to a −0.24 gray
true-fringe improvement. One small fringe tail remains (`scene_40`, +0.35 gray
despite +0.00029 global); mean fringe improvement is −6.85 gray. False-texture
mean/tail improve from P1 to −0.0243/+0.0654 gray but do not become uniformly
nonpositive.

The consensus is a mechanism improvement, not a learned gate: uncertainty
changes the operator itself. `scene_40` is now a refusal candidate because its
benefit is microscopic and the expanded property is mixed. Do not tune away
strong-scene gains to force this row.

P3 decisively rejects direct use of the current semantic matte chain. Even with
factory-true radius, 98 candidates average −0.00455 GT-SSIM (worst −0.0430),
mean fringe error worsens +0.74 gray, and only 16 candidates improve both global
and fringe scores. Mean alpha error is 0.135; worse, low mean error is not
sufficient (`scene_60`: 0.006 alpha error but −0.00869 GT-SSIM due wrong
ownership/boundary placement). All positive rows happen to have owner 0 because
the factory always orders the near-focused frame first; using that index as a
gate feature would be order leakage and is forbidden.

Next analytic rung: make matte placement part of uncertainty. Solve a small
erode/original/dilate matte ensemble and retain only correction components stable
under both inverse regularization and plausible boundary displacement. This may
turn edge sensitivity into refusal locally. If it cannot rescue the semantic
ceiling, do not train a gate on it; the required replacement is genuinely
observation-fitted alpha or a higher-precision matting model. Blind radius,
native resolution, real-stack identity, and package integration remain
untouched.

That matte-displacement probe is negative: it attenuates severe errors but does
not change their sign and can worsen the fine-detail tail. P4 instead audits all
top-4 candidates. The best-of-four bank contains an alpha-error `<0.05` candidate
in 56/98 scenes versus 28/98 for semantic top-1. Selecting the minimum
post-solve forward residual recovers 53 such candidates, so the observations
substantially fix *ranking*. They cannot fix a missing/inaccurate matte:
physically reranked single-solve outcomes still average −0.00488 and only 24/98
are positive.

A transparent development-only license identifies a high-precision subset:
semantic score `>0.5`, purity `>0.85`, area-fit `>0.9`, and forward MAE ratio
`<0.85` after physical candidate selection. It fires 7/98 and all seven improve
both global and fringe scores in the inspected set. This is not yet a holdout:
the thresholds were derived after examining these outcomes. Generate a fresh,
scene-disjoint object-occluder set before treating the rule as a gate. No owner
index may enter the rule.

A fresh 25-scene holdout was generated after freezing that rule: 12 moderate
(`CoC=0.02`) and 13 giant (`CoC=0.035`) scenes, with new placements and full
FastSAM/DA-V2 bridge outputs. The unchanged rule fires 2/25, both giant, and
both improve global/fringe fidelity. After regularization consensus:
`scene_114` is +0.00303 GT-SSIM / −7.61 gray fringe; `scene_122` is
+0.00075 / −2.13. This is a valid high-precision true-radius holdout result.
False-texture deltas remain small positive (+0.029/+0.017), localized to
boundary-frequency mismatch; do not call the false-texture tail closed.

Blind radius remains open. A deliberately broad 1.2–4.5% CoC consensus is too
conservative and not uniformly safe: it reduces the nine licensed gains by
roughly an order of magnitude and makes dev `scene_99` / held `scene_122`
negative. The next blind test is a fixed 3.5% giant-veil hypothesis with the
same forward-ratio license. This is justified only if it refuses every fresh
moderate scene and preserves the giant fires; otherwise radius remains a hard
blocker.

## Doctrine

The binding rule is that recovery must stay rooted in observed remnants. A
correction that looks plausible but adds unsupported texture is a failure,
regardless of mean benchmark gains. Shipping requires evidence that the claimed
scene re-renders into the observations and that selective metrics' excluded
pixels are explicitly audited.
