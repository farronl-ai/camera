# NEXT STEPS — hallucination-safe veil recovery to shipping

Active objective (Farron, 2026-07-25): the multiplicative veil-recovery hybrid
must stop extending foreground texture/patterns into background regions where
they do not physically exist, then graduate from the oracle research rig into
the shipped `--enhance auto` path.

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

- [ ] **S0 — reproduce and instrument.** Reproduce the user-caught artifact and
      retain `false_texture` as a first-class label. Add a GT-free analogue based
      on forward-model residual / unsupported high-frequency energy so runtime
      code can refuse without access to truth.
- [ ] **S1 — blind chromatic forward fit.** Estimate per-channel veil radii from
      the observed owner/far pair and semantic matte by rendering candidate
      observations back into the far frame. Reject low-confidence or
      out-of-range fits. No hard-coded factory CA.
- [ ] **S2 — blind noise + safe hybrid.** Port deficit-form in-loop gain,
      foreground-premult residual removal, analytic shrink, base-band exclusion,
      fringe clamp, and a strict correction cap. Estimate noise conservatively
      from observed flat-region high-pass residuals.
- [ ] **S3 — hybrid outcome gate.** Retrain on regime-matched
      objects-as-occluders scenes spanning moderate, giant, chromatic, and
      never-fire regimes. Features must include radius-fit confidence,
      forward-model residual change, D magnitude, gain exposure, and the
      GT-free false-texture proxy. Margin is set from harmful training outcomes
      and verified on a scene-disjoint holdout.
- [ ] **S4 — package integration.** `--enhance auto` may use the hybrid only when
      S1–S3 all license it; otherwise retain safe subtraction or identity.
      Absence/failure of semantic bridges remains byte-identical.
- [ ] **S5 — verification.** Add focused unit tests for channel-specific
      modeling, unsupported-texture refusal, kill-switch identity, and gate
      shape. Run the full test suite plus factory holdout, off-regime families,
      composed specialists, mobiledepth refusal checks, and eye panels.
- [ ] **S6 — checkpoint close.** Distribute findings to SYNTHESIS, PLAYBOOK,
      DEVSTYLE, and FRONTIER; refresh outward-facing docs if a capability ships;
      run the F49/F50 drift sweep; clean-tree/origin/test audit; commit and push
      logical checkpoints without author trailers.

## Current experiment: blind radius fitting

Candidate model for channel `c`:

`far_hat_c(r) = disk_blur(owner_c * alpha, r) + plate_c * (1 - disk_blur(alpha, r))`

where `plate` is conservatively inpainted from the observed far frame outside a
maximum veil support. Search a shared base radius plus a bounded chromatic spread
rather than three unconstrained radii. Score robust observation residual only in
the background-side veil shell. The fit must beat a null/neighbor-radius model
by a margin; otherwise gain refuses to fire.

Initial prototype note: unconstrained per-channel fitting is unstable on textured
backgrounds. The constrained shared-radius/chromatic-spread search is the active
path; its error distribution on all 100 existing object-occluder scenes is the
next checkpoint.

## Doctrine

The binding rule is that recovery must stay rooted in observed remnants. A
correction that looks plausible but adds unsupported texture is a failure,
regardless of mean benchmark gains. Shipping requires evidence that the claimed
scene re-renders into the observations and that selective metrics' excluded
pixels are explicitly audited.
