# Occlusion formation audit — geometry, veil, and material transmission

Status: active design note, 2026-07-26. Read with `MISSION.md`,
`NEXT_STEPS_scenerecovery.md`, and F60–F65 in `FINDINGS.md`.

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
- [ ] Resume on the single remaining S25 dissent: `s25_000` improves global MAE
      and the exterior veil but has a tiny foreground-core MAE regression
      (`+0.0096`) and ΔSSIM `-0.000283`. Separate raw owner-frame sensor noise
      from missed support before changing any established recovery component.
- [ ] Add explicit foreground-support, semantic-boundary, outer-veil, and
      far-background audits. Hard ownership means selecting a foreground layer;
      any foreground denoising must use only foreground-owned observations and
      may never blend the hidden rear layer inward.
- [ ] After the recovery rule freezes, generate a fresh S26 validation split;
      regenerate the inspector only once at the end.

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
