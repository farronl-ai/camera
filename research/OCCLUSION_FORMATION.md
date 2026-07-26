# Occlusion formation audit — geometry, veil, and material transmission

Status: active design note, 2026-07-26. Read with `MISSION.md`,
`NEXT_STEPS_scenerecovery.md`, and F60–F78 in `FINDINGS.md`.

## 2026-07-26 transmission-boundary integration checkpoint

`s29_010` exposed a second formation invariant after ownership was already
correct: the background-focused observation is not a clean background plate at
the end of a bright foreground veil.  It contains both foreground radiance
spread and attenuated rear radiance:

```text
O_rear = V_front + T_rear * B
B_direct = (O_rear - V_front) / T_rear
```

Therefore a dark or bright boundary feature seen in `O_rear` cannot be copied
as latent background merely because the rear layer is globally high-confidence.
Likewise, smoothing the final image or extrapolating the exterior trend is not
an inverse.  The runtime now uses the paired one-sided formation model to
subtract predicted foreground veil radiance and divide by remaining rear
transmission where that division is locally usable.

The formation stage owns these quantities. While both original frames are
available it must emit, per frame/model/pixel:

- predicted foreground radiance contribution `V_front`;
- rear transmission `T_rear`;
- selected or weighted PSF hypothesis;
- forward residual/model disagreement; and
- the observation-domain detection floor used for component-specific vetoes.

Boundary integration consumes that formation state. It must not estimate these
fields from the fused base: fusion has already pooled frequency-dependent
evidence from both frames, so the decomposition is no longer identifiable.
The current implementation computes the fields inside `recover_giant_veil`
because that function still receives both originals; the intended durable
architecture is one upstream solve and an explicit formation-state handoff.

Two further distinctions are load-bearing:

- **Support is not the integration contour.** Nonzero aperture support extends
  to the maximum PSF radius, but the visible transition is centered where
  modeled foreground coverage becomes material. The integration contour uses
  the conservative cross-PSF 10% coverage level and tapers continuously around
  it; the full licensed veil recovery footprint remains unchanged.
- **Scene confidence is not feature observability.** For a proposed latent
  low-frequency component, forward-project that component through each
  plausible PSF. If it would exceed the local captured-frame noise floor, its
  absence vetoes extrapolation. Only a component censored by the veil/noise may
  be inferred from the clean exterior trend.

The box-like and disk PSF inverses are selected locally by their two-frame
forward residual instead of averaged or rejected for disagreement. On the
diagnosed `_010` strip, the disk inverse explained the observations and nearly
matched analytic GT; the box hypothesis over-subtracted the bright veil.
Relative high-frequency integration follows the local contour normal on the
correction field. Circular absolute-image filters and unconstrained slope
extrapolation are explicitly rejected.

Commit `bf99365` is the visually validated checkpoint: the inspector's
right-side `_010` output was judged near-perfect. This remains a three-case
quick-cohort result (`s29_002/007/010`), not evidence for transmission,
arbitrary CoCs, real compound-lens PSFs, or all camera/ISP regimes.

## 2026-07-26 narrow shipping checkpoint

Full post-rule S29 licenses 11 scenes and exactly refuses one.  All 11 improve
MAE/MSE and every physical partition, rear application is zero in all protected
regions, and far background is exact.  GT-side worst-tile inspection finds no
invented fine texture behind the remaining diagnostic dissent.  The real
composed bridge reproduces the validated ownership path.

Exactly two-frame, validated-size one-sided opaque recovery is therefore live
in `--enhance auto`; failed gates remain byte-identical.  This does not promote
transmission, N-frame layer inversion, or arbitrary CoC regimes.

## 2026-07-26 post-rule S29 checkpoint

S29 separates resolved small ownership from source-mask speckle.  A connected
native graph continuation can hard-select the focused owner under strict
partial-overlap, size, distance, presence, and rear-semantic bounds even when
downsampled direction is mildly rear-biased by its surroundings.  Disconnected
source components expected below eight rendered pixels are not physical
factory truth; meaningful ~50-pixel pieces remain.

Regenerated S29 passes the hidden-background invariant 12/12 with outward veil
retained.  Blind geometry licenses 11/refuses one, mean IoU is `0.96566`, and
rear application is zero in hard/core/soft-edge/boundary/far regions.  Full
reconstruction remains open.

## 2026-07-26 negative-geometry preservation checkpoint

A rear-focused patch cannot subtract support unless the enclosing foreground
object is itself cross-frame corroborated.  The minimum containment is `0.75`.
Below it, focal/segmentation disagreement is ambiguity and possible opaque
support wins.  This restores the historical V1 false carve to zero rear overlap
without changing the legitimate S28 arch (`0.894` containment).

S27 preserves every physical partition 6/6; S25 preserves zero rear overlap in
all protected regions and improves all primary partitions, with one inherited
`+0.00584` ordinary-boundary dissent already smaller than its prior value.
S29 is the newly seeded post-rule verdict.

## 2026-07-26 focused-owner source checkpoint

Foreground radiance remains a discrete focused-owner observation after
ownership is selected.  Low-texture support may use stronger NLM (`h=3`) and
textured support uses `h=2`, but both operate on that same owner frame; observed
high-gradient contours are copied byte-exact from it.  No edge correction,
denoising route, or hard-copy route blends the rear frame into opaque support.

The complete S28 rerun licenses ten scenes and exactly refuses two.  All ten
improve MAE/MSE and all physical partitions, rear application is zero in every
protected foreground/boundary/far region, and far background remains exact.
Because S28 exposed and shaped this routing rule, it is development evidence;
preservation and a new post-rule V2 split remain before promotion.

## 2026-07-26 inverse checkpoint after fresh failures

F72 closes the four S28 geometry failure classes without changing the preserved
numerical solver.  A material exterior-connected region that decisively chooses
the rear focal plane is negative ownership evidence; rear-frame graph
connectivity expands that seed, and neither GrabCut nor satellites may re-add
it.  Conversely, a bounded native graph fragment that transforms frontward and
has strong reverse-reblur presence is a direct focused-foreground observation,
so it is hard-selected.  Rear-mask weights below `0.075` are unstable
sub-quantization tails and return to exact identity.

The resulting S27/S28 geometry audits have zero rear application in hard
ownership, core, hard soft edge, boundary, and far background.  Five causal
S28 outputs improve every physical partition and retain exact far identity.
Full split and post-rule validation remain; the inspector stays frozen.

## 2026-07-26 fresh split checkpoint

S28 is the first frozen split generated after F70.  Its twelve-scene direct
formation audit passes the full pre-antialias ownership invariant: changing
only the hidden rear layer changes zero V2 hard-owned pixels, every hard-owned
coverage value is exactly one, and every scene retains an exterior veil.  V1
changes 39,850 hard-owned pixels under the same counterfactual.

The inverse is not promoted.  Before any solve bank, the GT-attributed geometry
audit found one broad semantic rear opening (`s28_002`), two narrow
hard-ownership rear-mask leaks (`s28_005`, `s28_006`), and one far-background
leak (`s28_011`).  The broad case begins in the source proposal rather than the
generator or solver: its focused semantic mask contains 38,503 true rear
pixels, and later completion expands them.  A decisively rear-focused connected
component supplies independent evidence for the opening, while owner-proposal
consensus already protects focused-owner hard copy.  The remaining task is to
keep that rear region out of the layer model (or refuse the hypothesis), then
close the narrow rear-mask leaks without consuming legitimate exterior veil.

Do not regenerate the inspector yet.

## 2026-07-26 pause — restart from the hidden-background invariant

The current inspector's recovery result is a strong checkpoint and must be
preserved.  Later boundary-ownership work remains an uncommitted experiment;
do not let it replace the last pushed solver or use it to excuse a malformed
input.

The input formation is still **not accepted** merely because a region metric or
the old alpha-core counterfactual passes.  The user's visual diagnosis is the
authority for the failure: foreground defocus must never transport hidden rear
detail into a region claimed as opaque foreground.  F69 reproduced the loophole
and implemented the direct proof below; a compact visual carpet and fresh
validation cohort remain open before inverse work resumes:

1. Hold foreground radiance, geometry, focus, PSF, and noise fixed.
2. Render once over a high-contrast checkerboard and once over its inverse.
3. Compare the defocused-foreground observations over the full hard ownership
   support saved **before** antialiasing or PSF spread.  The difference must be
   exactly zero there, not merely small on an eroded `alpha == 1` subset.
4. Display that support and the magnified absolute-difference image beside both
   renders.  Any recognizable checker structure inside it invalidates the
   generator.
5. Foreground radiance may be redistributed using foreground samples only and
   may spread outward into an explicitly labeled silhouette veil.  Blur of an
   already composed foreground/background image, or any normalization that can
   import rear samples into hard support, is forbidden.

The old test covered only `alpha == 1` after resize and Gaussian antialiasing.
On the actual diagnosed bird source, V1 changed 7,397/55,222 hard-owned pixels
when only the hidden background was inverted. `one_sided_opaque_v2` now keeps
that pre-antialias binary ownership as a separate field and changes 0/55,222,
exactly, while retaining outward foreground spread.

F70 also pairs the inverse forward/adjoint with this same hard support. Its
compact S27 development carpet has zero rear correction in GT hard ownership,
core, soft edge, ordinary boundary, and far background for 6/6 scenes while
improving or preserving every corresponding direct-error partition. Discrete
graph completion always sources radiance from the focused owner; ambiguous
support may deny rear synthesis but cannot fabricate foreground texture. S27
shaped the mechanism, so unseen V2 validation remains required.

The inspection page seen at this pause mixed the new input example at the top
with stale historical input frames in lower scene panels.  It is therefore not
evidence that every displayed input used the repaired formation.  Leave the
page frozen now; rebuild all panels from one final formation version only after
the direct counterfactual above passes and the inverse pipeline is revalidated.

## User correction and pause checkpoint — supersedes the primary input contract

The historical exact-disk renderer uses aperture-coverage convolution on both
sides of the sharp silhouette. Even after reweighting S23 and making the
same-scene `r=12` example, that operator can admit focused rear detail *inside*
the latent opaque-foreground support. The user rejected that as the primary
formation contract for this project. Reducing the CoC merely reduces the
violation; it is not the fix. The primary S25+ renderer now implements the
one-sided contract below; the historical model remains only as a named stress
formation.

The primary opaque generator and paired recovery model use this one-sided
ownership operator:

1. inside confidently owned latent foreground support, rear throughput is
   exactly zero in every focal observation;
2. foreground defocus may spread foreground radiance outward beyond that
   support, producing the exterior veil;
3. the blur operator must not symmetrically erode/open the opaque support and
   reveal sharp rear texture inward;
4. true material transmission remains a separate, explicitly labeled model.

Existing V2/S23 aperture-mixed scenes remain useful as a named
alternative/stress formation model, but they no longer define the primary
opaque input contract. The inspector remains intentionally frozen at the
previous S23 checkpoint while S25 is unfinished.

### Resume progress

- [x] Add `one_sided_opaque_v1` without changing historical split
      reproducibility. Its far frame uses normalized foreground spread
      `H(alpha*F)/H(alpha)` and coverage `max(alpha, H(alpha))`.
- [x] Add direct counterfactual tests: replacing a hidden checkerboard with its
      inverse cannot change any `alpha=1` far-focus pixel, while the foreground
      PSF still extends outward.
- [x] Generate the first compact six-scene S25 development sample. Every scene
      has saved owned-core fraction `1.0`, owned-inner fraction `0.0`, zero rear
      throughput on every stored `alpha=255` pixel, and a nonempty outward veil.
- [x] Pair the runtime forward/adjoint solver with the one-sided operator,
      including a numerical adjoint test and a hidden-background
      counterfactual.
- [x] Select foreground geometry from original focal frames under the paired
      formation model, prefer containing near-tied silhouettes, corroborate
      support across frames, and hard-copy only the eroded, locally supported
      focused foreground.
- [x] Resolve the `s25_000` dissent causally. The conservative mask
      intersection had only `0.635` foreground recall, so the rear solver
      entered missed true foreground. Focal-pair graph-cut completion raises
      its silhouette IoU to `0.982`; the hard front uses only an NLM-denoised
      focused-owner observation, while rear correction requires the original
      cross-frame geometry and a separate front veto.
- [x] Add explicit foreground-support, semantic-boundary, outer-veil, and
      far-background audits. Hard ownership means selecting a foreground layer;
      any foreground denoising must use only foreground-owned observations and
      may never blend the hidden rear layer inward.
- [x] Freeze the compact S25 development rule. All six runs have zero rear-mask
      overlap with GT foreground core, GT antialiased boundary, and far
      background. All six improve MAE/MSE, foreground core, and exterior veil,
      with exact far-background identity. Five improve SSIM; `s25_003` changes
      `-0.000150` while improving direct errors and preserving ownership.
- [x] Generate S26 after the first freeze. Its first 15 scenes exposed two
      distinct ownership failures: rear application crossed a missed opaque
      boundary on `s26_013`, and reached pure rear on `s26_014`. S26 therefore
      became development evidence.
- [x] Preserve the F66 focused-RGB completion/solver while adding only causal
      safeguards: cross-frame-proven satellites, a front-direction veto
      confined to the immediate boundary extension, and a reverse-reblur
      presence threshold measured from the scene's rear noise floor.
- [x] Re-freeze the six-scene S25 carpet. Rear application is zero in GT
      core/boundary/far for 6/6; all six improve MAE/MSE/core/veil and preserve
      exact far identity; 5/6 improve SSIM.
- [ ] Finish attribution of `s26_014`'s remaining hard-front far extension,
      freeze without weakening the working solver, then generate a genuinely
      new S27 validation split. Regenerate the inspector only once at the end.

## Why this note exists

The inspection at `extension_007` native `(1048,216)` raised the right
foundation-level question: are we learning to invert a physically valid opaque
occlusion, or compensating for accidental transparency in the synthetic input?
The answer must be explicit before any more inverse-stage tuning.

The audit found that V2 does **not** alpha-overlay a sharp background through an
opaque foreground. Its far-focus equation is an aperture average:

```text
O_far = H(alpha * F) + (1 - H(alpha)) * B
```

For each aperture ray, the ray either hits opaque foreground or sees focused
background. `H(alpha)` is therefore frame-specific foreground *coverage*, not
material opacity. Where `H(alpha)=1`, background contributes exactly zero.
Where `0<H(alpha)<1`, different rays see different layers and their irradiances
average at the sensor.

The reported point is a measured example:

- sharp geometric alpha: `1.000`;
- distance inside the sharp silhouette: about `12 px`;
- far-frame defocus-disk radius: about `38 px`;
- far-frame foreground coverage: `0.545`.

Thus 54.5% of aperture rays hit opaque foreground and 45.5% see background.
That point is an **inner partial-coverage veil pixel**, not a complete-coverage
opaque core and not a transmissive material. A sufficiently slender opaque
object can be partial-coverage over its entire apparent width under severe
defocus. This is physical, and the renderer is unit-tested against a brute
aperture sum.

A second inspection at native `(808,347)` tests the full-column intuition. The
sharp foreground is only 46 px thick vertically in that column (46–50 px in the
nearby sampled columns), while the far-frame CoC is about 75 px in diameter.
The reported pixel has alpha `1.000`, coverage `0.725`, and lies about 16 px
inside the sharp support. The column has no pixels with coverage at least 0.95:
background is consequently visible to some extent throughout the blurred
foreground column. That observation is real and important, but it identifies
slender/all-veil opaque geometry—not material transmission. The inspector and
cohort labels must expose the CoC-to-local-thickness relationship directly.

The inspection confusion is nevertheless evidence of a benchmark-design
problem. Sharp alpha alone makes an inner veil look like damaged opaque
interior. Equal cycling through `solid`, `mixed`, and `thin` also gives
nearly-all-veil geometries more authority than their role in ordinary
photography warrants. Finally, the factory has no genuine material-transmission
mode at all. Those problems must be fixed without replacing correct aperture
physics with a hard opaque overlay.

## Three regimes that must never be pooled

### 1. Opaque occluder with substantial complete-coverage core

Material transmittance is zero. The far-focused frame may reveal background
only in the aperture-defined inner/outer veil. The complete-coverage core is a
hard invariant: `coverage=1 => background throughput=0`.

This becomes the primary opaque validation regime. It should dominate normal
promotion claims and ordinary inspector examples.

### 2. Opaque but slender / severe all-veil geometry

Material transmittance is still zero, but the CoC is large relative to the
object. Most or all sensor pixels integrate both foreground-hit and
background-visible rays. This is valid optics, but a harder and less typical
regime. It stays as an explicitly named stress cohort, not as evidence that the
foreground material is translucent and not pooled equally into the primary
opaque claim.

### 3. Transmissive foreground material

The foreground has nonzero material transmittance even for rays that intersect
it. Geometric coverage and optical extinction are now different fields. For a
scalar opacity `q=1-tau` on geometric support `alpha`, an initial nonrefractive
model is:

```text
A = alpha * q
O_near = A*F + (1-A)*H_far(B)
O_far  = H_near(A*F) + (1-H_near(A))*B
GT     = A*F + (1-A)*B
```

The two focal observations transform the front and rear layers differently, so
the latent foreground/background can remain identifiable when enough texture
and focus evidence survive. This is a distinct inverse problem, not a relaxed
opaque rule. The factory must save geometric coverage, extinction/opacity,
foreground radiance, background radiance, and composite GT separately so layer
recovery can be graded rather than hidden inside one image score.

Colored transmission, refraction, scattering, and internal blur are later
strata. They must not be silently approximated by scalar alpha during the first
identifiability experiment.

### 4. Malformed/adversarial overlay

A sharp background simply blended through an allegedly opaque, fully covered
foreground is physically inconsistent with regimes 1–2. Such inputs may be
retained only as a named robustness/adversarial set. They cannot train, select,
or validate the primary physical operator.

## Pipeline consequences

1. Preserve front-first ordering from F61/F62. A focused opaque owner remains a
   hard rear veto in complete coverage.
2. Do not apply that hard veto to genuine transmission. Infer or license an
   extinction field separately from geometric support.
3. Keep layer formation and layer recovery paired. An opaque candidate is
   re-degraded by the opaque model; a transmissive candidate by the
   transmissive model. Cross-model residuals can help route, but forward fit
   alone is not a truth certificate because blur has a null space.
4. Grade opaque core, opaque inner veil, outer veil, transmitted foreground,
   transmitted rear layer, and far background separately. Global SSIM/MAE
   cannot substitute for these partitions.
5. Keep F62's current fires as mechanism diagnostics, but do not continue
   tuning the fine-band tail until the new cohort taxonomy shows which tail
   survives in primary opaque-core scenes.
6. Add an explicit per-pixel aperture/coverage explanation to the inspector so
   a sharp-silhouette coordinate cannot be mistaken for complete optical
   coverage.

## Execution checkpoint

- [x] Audit V2 equation and brute-aperture unit test.
- [x] Numerically classify the reported `extension_007` point.
- [x] Add unambiguous material-model and optical-regime metadata.
- [x] Reweight/generate primary opaque-core and named all-veil stress cohorts.
- [x] Require local owner-mask supermajority before irreversible opaque copy or
      its PSF footprint; S23's causal far-background leak is repaired.
- [x] Add a scalar-transmission renderer/factory path with saved latent layers.
- [x] Add formation-specific tests and the same-scene opaque inspector panel.
- [x] Re-evaluate F62 by regime before returning to S15: all seven S23 fires
      occur in the primary opaque cohort; boundary/all-veil cohorts refuse.
- [ ] Generate the compact transmissive development cohort and establish its
      oracle layer-separation ceiling.
- [ ] Build an oracle-transmission ceiling, then attempt blind model routing and
      opacity/layer estimation from captured focal transformations.

## Doctrine

The target is the physical scene that could have produced the observations.
Neither source similarity nor a convenient synthetic compositor may define
truth. Correct aperture mixing is retained; material transmission is modeled
explicitly; malformed inputs are isolated rather than optimized into the main
pipeline.
