# The scene-model ALIGNER, round 1: the machinery and its KATs on the factory

`research/aligner.py`. FRONTIER §7b's output contract, built: the N input frames
resampled ONCE into reference geometry through a PIECEWISE sampling field, each
carrying honest GAPS, with the interface mirroring `align_stack`'s —
`(aligned, usable, report)` — so `fuse_perband(aligned, usable=usable)` consumes
it unchanged. ANALYTIC FACTORY ONLY (`parallax_gen`, 6 frames, reference 3, near
3.2 px/frame against far 0.7, 4.6x). The kitchen is round 2's scene; nothing here
has been run on it and no number here should be assumed to transfer (F117's `c`
is the standing proof that they do not).

## Headline

| rung | GT-SSIM |
|---|---|
| reference frame alone | 0.965826 |
| aligner, RECOVERED pieces, gaps ON | **0.970842** |
| shipped depth-bin path | 0.972808 |
| aligner, TRUE pieces, fitted affines | **0.986238** |
| aligner, TRUE pieces, TRUE affines | 0.986258 |
| runtime two-frame (F117) | 0.984455 |
| oracle one-resample floor (F115) | 0.989487 |

Attributed gap, one substituted term per rung with everything downstream
recomputed by the module itself (F110's trap: never substitute a term *plus* its
consequences):

```
pieces                      +0.015396      <- the entire remaining gap
transforms                  +0.000019      <- solved
gaps are worth              +0.007000      (gaps ON minus gaps OFF, true geometry)
resample + fusion remainder +0.003229      (against the 0.989487 floor)
```

**The finding of the round: with correct pieces the aligner contract BEATS the
runtime two-frame path (0.986238 > 0.984455) and lands 0.0032 from the oracle
one-resample floor — and the transform contributes 0.00002 of what is left.**
Everything that is missing is the PIECE FINDER. That is a much better place to be
than the reverse, because the transform was the charter's designated hard
sub-problem (2) and the piece cut was assumed to be the easy half.

## What each commitment cost to build, and what it measures

### 1. Pieces cut at occluding contours (K1) — the crude part

Method: `depth_from_focus` (pass 1's own depth proxy) -> median at the guided
filter's own diameter (2*8+1 = 17 px; speckle finer than the filter's support is
not resolved, and a median preserves a real jump) -> local RANGE across that same
diameter, because a real jump is SPREAD over it and a 3x3 Sobel therefore
under-reads it -> ribbon where the range exceeds half a focal step -> connected
components of the complement, each >= `MIN_ARBITRABLE` (200 px, borrowed), as
watershed markers -> `cv2.watershed` on the REFERENCE FRAME, so the cut lands on
the image's own strongest contour (1 px) inside the ribbon the smoothing widened,
and closure is guaranteed by watershed rather than hoped for.

Measured on the factory: 24 pieces where truth has 2 (three surfaces: background,
rectangle, circle). Purity against the true near mask: mean 0.954, worst 0.687 —
so pieces mostly respect the silhouette. Cut RECALL (true silhouette covered by a
cut within the 2.30 px matte band + 1 px) **74.74%**; cut PRECISION (cut pixels
within that band) **23.71%**, median cut distance 21 px. Both bars FAIL as stated
in the charter, and the failure is one-sided and legible: **over-segmentation of
a single surface, not misplacement of the real silhouette.**

Root cause, measured not guessed: `depth_from_focus`'s field on this scene is
noisy at the scale of the jump. The near/far levels separate cleanly in the mean
(0.289 vs 0.699) but a threshold that closes the true silhouette ribbon also
raises ribbon inside the background, and one that suppresses the background
speckle leaves the silhouette ribbon OPEN (probed: at a closure that leaves 0.19%
ribbon the cut sits a median 1.40 px from the true contour — excellent
localization — but the complement is ONE component, so nothing is separated).
There is no threshold that does both, which by DEVSTYLE §12.3 means the threshold
is the wrong instrument and something physical is standing behind it.

**The named cure, NOT built (round 2):** a cut must be justified by a MOTION
difference. Two pieces separated by a spurious cut have the same affine, so
merging adjacent pieces whose fitted affines agree within `GATE_TOL` over the
frame collapses the over-segmentation without touching the real silhouette, whose
two sides differ by 2.5 px/frame here. The ingredients are already computed
(`report["travel"]`, `report["matrices"]`); only the merge is missing. This is
also the clause that makes the finder scene-independent, because it replaces a
depth-field threshold with a measured motion difference.

Ramp-wholeness is tested at the INSTRUMENT level (`kat_ramp`) and passes: on a
synthetic depth field that is a pure linear ramp the cut finder cuts NOWHERE
(ramp slope 0.00150/px against the 0.02500/px threshold); with one step added it
cuts at x=299..300 and nowhere else. It is NOT tested on the factory scene, and
cannot be: `parallax_gen`'s planes are fronto-parallel and its `viewpoint` applies
ONE scalar shift per layer, so no parameter of it produces a depth ramp
(`BREATHING_PER_FRAME` adds magnification, which is depth-INDEPENDENT — that is
exactly what makes breathing separable from parallax, and exactly why it cannot
stand in for a ramp). Round 2's kitchen countertop is the real ramp test, and
F114 already recorded that its depth is continuous.

### 2. One affine per piece per frame (K2) — solved

Fitted from MATERIAL edges inside the piece through F116's dense normal-profile
matcher: each site gives one scalar constraint `n . u(p) = d(p)`, six unknowns
against thousands of sites (§12.4 — an exactly-determined fit proves nothing), two
trimmed reweightings at 2.5 MAD, three iterations, global affine as the prior.
LIMB edges are excluded by construction (sites are eroded away from every cut by
`CONTOUR_HALF + CONTOUR_SPAN`): a silhouette's apparent motion is a mixture of two
surfaces, so fitting a piece's motion from its own limb imports the neighbour's
motion (F92).

Scoped to the transform alone, i.e. on TRUE pieces (§12.2 — the number must be
computed on the thing under test): worst |tx| error over all (frame, piece)
**0.159 px against the 0.2 px bar, PASS.** Per-piece isolation probe, true masks,
identity prior: near plane 9.420/3.213/-3.269/-6.889 against truth
9.6/3.2/-3.2/-6.4; far plane 2.083/0.686/-0.694/-1.408 against 2.1/0.7/-0.7/-1.4.
The two largest errors are both at the ends of the sweep where the piece is most
defocused (k=5, near radius 4.6 px) — PLAYBOOK's recorded bias, a blurred profile
correlating confidently at about zero shift, so a per-frame fit on a blurred piece
is biased SHORT.

Two disciplines paid for themselves inside the round:

- **`MATCH_SIGN` is MEASURED, not assumed** (`kat_sign`, §12.1). The first build
  guessed the matcher's sign convention and the KAT caught it immediately. It also
  caught a subtler error in the KAT itself: the normal component of a +2 px x
  displacement is `2*nx`, which FLIPS with the normal's own direction, so the
  median shift over x-facing sites is meaningless (it read -1.273 for a +2 px
  move) and only `shift/nx` is the instrument's answer. Corrected, it reads
  **+2.001 px per unit normal for a +2.000 px displacement** over 1335 sites.
- **The motion SERIES must fall back to the PRIOR, never to the identity.** One
  weighted line per affine parameter through the identity at the reference (§12.6
  — the simplest model the physics allows; a constant per-frame camera translation
  makes every parameter linear in k). A piece with no usable evidence has no
  weight, and the first build then produced the IDENTITY for it — which asserts
  the camera did not move, the exact silent invention F81 was caught making, and
  it showed up as a series error of precisely `3.2*|k-ref|` on the near pieces.
  Now such a piece is UNVERIFIABLE per F106: it declines the correction and keeps
  the global stage's geometry.

**The silhouette gate** (K2b) is F116's instrument re-pointed as a per-frame
alignment gate at the cut band. On the RECOVERED pieces it does not separate:
truth 10.75% of confident sites flagged against 16.15% for an injected +1.5 px
near-piece error, a 1.5x separation, VERDICT not passed. That 10.75% baseline is
not the gate misfiring — it is the over-segmented cut set putting 5010 "contour"
pixels a median 21 px away from any real silhouette, i.e. the gate is being asked
about contours that are not occluding contours. The gate consumes
`report["cut"]`, so it inherits K1's failure directly and should be re-run as the
FIRST check after the merge clause lands. Its consequence is correct and already
wired: a failing site produces a GAP band, never a forced fit.

### 3. Gaps (K3) — the wall-smear class is structurally absent

Composed in ONE fixed order — off-footprint, occlusion, silhouette gate,
photometric veto, the reference's own matte band, then a 1 px dilation for
interpolation support — and **monotonicity is ASSERTED per frame**, because
refusal composition is non-monotone in general (scenemodel_NOTES §23a: deleting a
rewrite pixel upstream can delete a downstream clause's SEED and make it withdraw
less). Every clause only removes.

Occlusion is computed exactly as the charter specifies: each nearer piece's
silhouette PUSHED into frame k by ITS OWN transform, DILATED by ITS modeled
defocus radius there (`R = c*|k - peak|`, F83), then read back at the place the
target pixel samples from — one `warpAffine` plus one nearest-neighbour `remap`
per (frame, occluder). **The F116 wall smear entered exactly through the missing
dilation, and it is load-bearing here: with it, `usable_k` is FALSE over
88.2-95.3% of truth's own occlusion mask, and the leakage — far content sampled
where truth says the surface was not seen, measured on truth's occlusion core
eroded by the matte band — is 0 px in every frame.** That is the structural
guarantee the contract exists for, and it holds on the recovered pieces too.

IoU against truth's occlusion mask is **0.017 mean**, which looks catastrophic and
is not what it appears: recall is 0.88-0.95 (the occlusions are all found), and
the IoU is destroyed by the DENOMINATOR — 69k-98k gap pixels against 800-2800 truly
occluded ones. Those extra gaps are overwhelmingly the photometric veto firing on
impure pieces (the veto ledger shows the photometry clause taking k=0 from 0.9071
to 0.7456), which is the veto doing its job on a geometry that is wrong, plus the
matte band of 24 spurious cuts. IoU becomes the right summary only once the pieces
are right; until then recall + leakage are the honest pair, and both pass.

A clause the charter did not name but the factory forced: **pieces whose measured
travel differs by less than `GATE_TOL` are at the SAME depth and cannot occlude
each other.** Without it an over-segmented single surface occludes ITSELF at every
spurious cut. Occlusion order comes from parallax magnitude (displacement is
linear in inverse depth, so the piece that travels further per frame is nearer) —
a direct measurement from transforms the module already fitted, and a strictly
better instrument than F83's focal-peak polarity proxy, which F114 found REFUSES
on today's factory.

### 4. Application

One `cv2.remap` per frame with the discontinuous field. Piece boundaries are NOT
blended (F81's blend is the soft geometry F106 outlawed) — the jump is legal and
the gaps absorb it. `usable` is nearest-neighbour throughout: a mask is never
interpolated into existence.

## `c`, measured on this scene

F117's open item is that `c` does not transfer (factory 1.161, kitchen 0.684,
large-motion ~0.4) and nothing in the runtime measures it. `measure_blur_rate`
measures it here: each piece's own sharpest frame is its focal peak by definition,
the disk radius that best carries the peak frame's appearance to frame k is frame
k's radius (integer ladder + one parabolic sub-pixel step), regressed through the
origin. On true pieces: **c = 1.120 against the factory's true `BLUR_PER_STEP` =
1.15 (2.6% low), and peaks recovered exactly — near 1, far 4.** Closer to truth
than the shipped 1.161. On the recovered 24 pieces it reads 1.716, i.e. the
measurement degrades with the segmentation, which is the expected coupling and one
more reason the merge clause comes first.

## Interfaces round 2 inherits

```python
aligned, usable, report = aligner.align(frames, ref=None, matrices=None,
                                        labels=None, verbose=False)
```

- `aligned[k]` uint8, reference geometry, one resample, defocused as observed.
- `usable[k]` bool, reference geometry. `usable[ref]` is all-True by construction
  (the reference observed the reference viewpoint, matte band included), which is
  what guarantees every pixel has at least one usable frame and keeps
  `fuse_perband` out of a degenerate all-refused window.
- `labels=` substitutes piece masks (the oracle rung, and round 2's hook for a
  better cut); `matrices=` substitutes `{(k, piece): 3x3}` transforms. Both are
  single-term substitutions: everything downstream (blur rate, peaks, order,
  groups, gaps) is recomputed from them.
- `report` carries `labels, pieces, groups, order, travel, c, peaks, radius,
  matrices, raw_matrices, prior, slopes, stages, gates, cut, ribbon, withheld`.
  `stages` is the per-frame veto ledger, in composition order, and is the first
  thing to read when a frame withholds more than expected.

Commands: `aligner.py` runs K1, K2, K3, K4 and the ladder; `aligner.py ramp` runs
the ramp KAT alone. `out/aligner/` holds the 6 aligned frames, the 6 gap overlays
(gaps in red), the piece map (white = cuts), the fused result and the truth.

## What round 2 must know

1. **Build the affine-agreement merge clause FIRST.** It is the whole remaining
   gap (+0.0154) and every other number in this file is contaminated by its
   absence — the gate's 10.75% baseline, the 0.017 IoU, and c = 1.716 all improve
   for free when the pieces are right.
2. **The contour-cut method is NOT kitchen-ready.** It relies on a focal-peak
   field with a resolvable jump, and F114 measured the kitchen's depth to be a
   continuous RAMP whose bands saturate `RADIUS_MAX`. On a ramp the current finder
   produces ONE piece (correctly, per the ramp KAT) and therefore no occluding
   contours at all — which is right for the countertop and wrong for the bottle
   standing on it. The kitchen needs the cut driven by MOTION disagreement, with
   the depth jump as a prior rather than the arbiter, and that is the same clause
   as (1) run in the other direction.
3. **The transform is done; do not re-open it.** 0.159 px worst error, 0.00002 of
   the gap. If a kitchen number looks like a motion problem, suspect the pieces.
4. **Left crude, deliberately:** the silhouette gate's band is
   `ceil(max reference radius) + 1` for every piece rather than per-piece; the
   photometric veto runs whole-frame rather than per-piece (correct, but it means a
   veto cannot be attributed to a piece without re-running); `measure_blur_rate`
   uses a Laplacian energy for the peak rather than the repo's
   `content_aware_energies`; and `travel` uses translation magnitude only, so a
   piece that rotates without translating would be mis-ordered.
5. **Do not import 1.161.** Measure c per scene with `measure_blur_rate`; it is
   the one number F117 left unmeasured in the runtime.
