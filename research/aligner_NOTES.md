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

# Round 2: the affine-agreement merge, and the aligner's first kitchen

Round 1 left one thing to build and measured its price: the piece CUT was the whole
remaining gap (+0.0154 of 0.0186), and focal-peak thresholding was diagnosed as the
wrong instrument (§12.3 — no threshold both closes the true silhouette ribbon and
stays quiet inside one surface). Round 2 built the named cure and then took the
module to the kitchen.

## The merge, and the model order it forced

**The clause as designed.** Depth/focal jumps are demoted to a SEED that proposes
candidate cuts; the arbiter is measured MOTION. Adjacent pieces (a surface is
contiguous, F93) whose fitted affines explain each other's support to within
`GATE_TOL` (1.5 px) in EVERY frame are ONE surface and are merged; the merged piece
is refit on its merged support; iterate to a fixed point with piece count strictly
decreasing, ASSERTED. Two clauses, split by whether a measurement exists:
AGREEMENT (both sides measured their own motion) and ADOPTION (a piece whose motion
is UNVERIFIABLE cannot agree with anything, because agreement is a measurement — it
is absorbed by the neighbour it shares the longest border with, and the refit that
follows re-measures the pair jointly, so a wrong adoption surfaces as the merged
piece's own residual instead of being locked in). Chaining (a~b, b~c, a!~c) is
guarded twice: candidates are consumed in increasing order of disagreement, and each
test is between the two GROUPS' area-weighted affines, so a group that has already
grown must still explain the newcomer's support.

**The first build of it did nothing, and the reason is the finding of the round.**
With round 1's per-piece 6-DoF fits the merge fired ZERO agreement merges: the
pairwise disagreements inside ONE surface ran 1.75 to 53.31 px, i.e. an order of
magnitude above the tolerance, purely from the linear part's extrapolation noise on
a small piece. **Motion agreement is not measurable from an over-parameterized fit.**
The physics gives the right model order: rotation and magnification (the camera's
breathing) are DEPTH-INDEPENDENT — that is exactly the property that lets ONE global
affine carry them, and exactly why round 1 could separate breathing from parallax —
while PARALLAX is a translation whose magnitude is linear in inverse depth. So a
piece's residual on top of the global affine is, to first order, a pure TRANSLATION:
`PIECE_DOF = 2`, two unknowns against thousands of normal-profile constraints, which
stays determined where six unknowns do not (§12.4, §12.6). With that one change the
merge fired: 24 -> 6 -> 5 -> 5 pieces, disagreements inside a surface 0.15 px against
2.0-7.9 px across the true silhouette — a clean separation, and the first
scene-independent statement of what a cut is.

**But 2 DoF is the wrong model for the TRANSFORM, and the ladder caught it
immediately.** On TRUE pieces the 2-DoF fit reads 3.570 px worst against the 6-DoF
fit's 0.159 px, and the true-pieces rung fell 0.986238 -> 0.975629. The cause is the
PRIOR: the whole-frame affine's linear part is a compromise across depths (it absorbs
some differential parallax as shear/scale), and only six degrees of freedom can shed
it — translation alone cannot undo a wrong linear part away from the sites' centroid.
So the two jobs take different model orders for a stated reason: **2 DoF decides the
CUT (it must be stable on small pieces), 6 DoF is the TRANSFORM (it runs only on the
merged pieces, which are large).** `FINAL_DOF = 6` refits once after the fixed point.
A graduated fallback (6 DoF, else 2 DoF, else UNVERIFIABLE-keeps-the-prior per F106)
costs 0.0003 on the factory and is kept because a piece that would otherwise carry
whole-frame geometry is the F81 class of silent invention.

## The factory KATs, after the merge

| | round 1 | round 2 | bar |
|---|---|---|---|
| pieces (truth: 2 labels / 3 surfaces) | 24 | **5** | ~3 |
| cut PRECISION (within the 2.30 px matte band) | 23.71% | **43.01%** | >90% FAIL |
| median cut distance to the true contour | 21 px | **6 px** | — |
| cut RECALL (silhouette covered by a cut) | 74.74% | **68.23%** | >90% FAIL |
| purity, worst / mean | 0.687 / 0.954 | **0.933 / 0.963** | >=0.98 MISS |
| K2 worst series RMS, RECOVERED pieces | 7.567 (24-piece) | **2.264** | 0.2 FAIL |
| K2 worst \|tx\| error, TRUE pieces | 0.159 | **0.159** | 0.2 PASS |
| K2b gate separation (truth vs +1.5 px) | 10.75% vs 16.15%, 1.5x | **15.04% vs 17.68%, 1.2x** | 4x FAIL |
| K3 mean IoU vs truth's occlusion mask | 0.017 | **0.037** | rise 10x MISS (2.2x) |
| K3 wall-smear leakage | 0 px | **0 px** | 0 PASS |
| K4 GT-SSIM, recovered pieces + gaps | 0.970842 | **0.980686** | ->0.9862 |
| K4 gaps OFF | — | 0.945656 | — |
| withheld pixel-frames | 19%+ | 11.61% | — |
| `c`, recovered pieces | 1.716 | **1.212** | 1.15 |
| `c`, true pieces | 1.120 | 1.120 | 1.15 |

Attributed gap, recomputed by the module (one substituted term per rung):

```
pieces                      +0.005552   <- was +0.015396 (2.8x smaller)
transforms                  +0.000019   <- unchanged, still solved
gaps are worth              +0.007000
resample + fusion remainder +0.003229
```

**Honest verdict: the merge bought two thirds of the piece term (+0.0098 of the
+0.0154 available) and no stated bar flipped to PASS.** What improved is exactly what
the diagnosis predicted (piece count 24->5, cut precision 1.8x, median cut distance
3.5x closer, purity worst 0.687->0.933, `c` 1.716->1.212, K4 +0.0098). What did not:

1. **5 pieces, not 3.** The two surviving spurious cuts are inside the background,
   between pieces whose measured disagreement is 2.0-2.6 px — above `GATE_TOL` but
   far below the true silhouette's 4.9-7.9 px. Their disagreement is FIT ERROR on
   impure pieces, not depth. `GATE_TOL` is the wrong tolerance for this test: it is
   the displacement nothing in this arc RESOLVES, whereas what is needed is the
   uncertainty of the two fits being compared. The instrument the next round wants
   is a model-selection test — does ONE affine on the union explain both supports as
   well as two do, given their own residuals — not a fixed pixel bar.
2. **Cut RECALL fell 6.5 points.** The merge cannot ADD a cut, so recall can only
   fall, and it did: two boundary segments of the true silhouette were removed
   because the pieces either side agreed within tolerance. That is the seed's failure
   (the ribbon was open there), not the merge's, and it is why the charter's plan of
   "seed with depth, arbitrate with motion" is only half-built: nothing yet PROPOSES
   a cut from motion where the depth field is silent.
3. **The K2b gate still does not separate (1.2x, and it got WORSE).** Round 1 blamed
   the 10.75% baseline on the over-segmented cut set; the merge fixed the cut set and
   the baseline ROSE to 15.04%. So round 1's attribution was wrong, and this is now a
   live defect in the GATE, not an inherited one. It is the first thing to isolate
   next (§12.2: score it on the thing under test — one piece, one frame, one injected
   offset, at a cut that is known to be a real silhouette).

## The kitchen — the aligner's first real scene (`aligner.py kitchen`)

12 frames, 774x518, `normalize_exposure`, reference 6. `align_stack`'s own crop on
this stack is (15, 8, 742, 510) — verified by the command, so `scene_model`'s
canonical box coordinates apply unchanged. Elapsed 233 s. Log:
`out/aligner/kitchen_round2.log`.

**Merge fixed point: 33 -> 13 -> 11 -> 11 pieces** (5 agreed + 15 adopted, then
2 agreed, then none; 30/22/17 cuts HELD by motion disagreement). Monotone asserted.
`c` measured on THIS scene: **1.403 px/frame** (F117's kitchen value from a different
instrument was 0.684 — the two are not comparable and neither was imported).

**The piece map (`out/aligner/kitchen_pieces.png`) — the verdict is
UNDER-segmentation, and it is the wrong half.** One piece carries 239669 px (60% of
the frame) and it contains the countertop AND the glass pitcher AND the Coke bottle
AND the cocoa tin AND the Sprite AND the Lubriderm. The other ten pieces are wall,
backdrop and lower-counter fragments of 2.5k-38k px.

* F114's prediction is CONFIRMED where it was made: the continuous countertop ramp
  is not cut. Motion agreement keeps a ramp whole, exactly as designed, and the
  focal-peak seed never proposed a cut inside it.
* The round's stated expectation for the discrete objects FAILED: they did not
  separate. They are inside the counter's piece, sharing the counter's affine — which
  is the F112/F108 defect class restated in the aligner's own terms.
* Mechanism, and it is the same tolerance problem the factory showed: the kitchen's
  differential parallax is a few px/frame, so an object's residual translation
  differs from the counter's by an amount COMPARABLE TO `GATE_TOL` (1.5 px), and the
  first merge round consumes it. On the factory the same tolerance was comfortable
  because the differential was 2.5 px/frame against a 0.15 px within-surface
  disagreement. **`GATE_TOL` is a resolution bound, not a fit-uncertainty bound, and
  using it as the merge tolerance is the single decision most in need of replacement.**
* `travel` (round 1's own flagged crudeness, note 4) is now visibly broken here: it
  reads 7822 px/frame for a 5349 px piece and 351 for another, because it evaluates
  the affine's translation at the ORIGIN, so a 6-DoF linear-part error extrapolates
  wildly. Occlusion ORDER is derived from it, so the order on this scene is not
  trustworthy. Fix is one line and was not made under the clock: evaluate the
  displacement at the piece's own CENTROID.

**Gap statistics, per frame, attributed to the clause that withdrew (the failure mode
to name is silent over-refusal, so here it is, loudly).** Mean **57.19%** of
pixel-frames gapped; the reference frame is 0.00% by construction. Range 39.2% (k=5)
to **82.1%** (k=11). Attribution of the mean: photometry **27.21**, matte + 1 px
dilation **12.04**, silhouette gate **8.19**, occlusion **7.05**, off-footprint
**2.69**. F114 predicted large holes and they are here, but the dominant clause is
NOT the geometry — it is the photometric veto, at 3.9x the occlusion clause, which is
the factory's signature for "the veto is doing its job on a geometry that is wrong."
With the objects welded to the counter, that is exactly the expected reading.

**The canonical instruments, handed to the UNMODIFIED `fuse_perband(usable=...)`.**

| candidate | box 1 | box 2 | box 3 | box 4 | flank mean / >12 |
|---|---|---|---|---|---|
| routed default (recorded) | 1.20/2 | 2.04/13 | 1.19/19 | 1.03/17 | 0.897 / 0.01% |
| aligner + gaps + `fuse_perband` | 3.31/11 | 3.01/18 | **0.72/24** | **10.05/81** | 1.378 / 0.08% |

Focus energy in the same boxes: aligner 32.5 / 23.0 / 83.4 / 30.9 against the
reference frame's 32.2 / 22.1 / 83.2 / 22.8 — so every box is at or above the
reference and box 4 is markedly sharper, i.e. the changes are sharpenings, not
washes. Box 3's MEAN improves on the routed default (0.72 vs 1.19). **Box 4 is a
defect: mean 10.05 against 1.03 and a max of 81** — that is the yellow-rag alias
region, whose content sits on objects the piece map welded to the counter, so it is
the same cause as the piece verdict and not a separate failure. The flank regresses
mildly (0.08% > 12, 16 px spread over x587-658 y255-418 — spread, so not the knob).
No bar was to be forced and none was; the cause of the one real regression is
attributed to the piece cut.

**THE WALL-SMEAR TEST — the contract's signature, and it holds.**
`out/aligner/kitchen_wall_gaps.png`, the wall right of the cocoa tin (inspection
x285-365 y50-170), 12 per-frame tiles with gaps in red plus the fused result and the
reference. Gap fraction in that box: k0 71%, k1 67%, k2 70%, k3 57%, k4 57%, k5 49%,
**k6 (reference) 0%**, k7 67%, k8 72%, k9 78%, k10 79%, k11 79%. The wall band the
tin's parallax swings across is withdrawn in a legible vertical ribbon in every
non-reference frame, and **the FUSED crop shows the wall and the tin's lid clean — no
smear, no doubled edge.** The F116 defect class cannot form: the frames that did not
observe that wall are not asked about it. That is the whole reason the contract
exists, and it is now demonstrated on the scene where the defect was originally found.

**Whole-frame deliverables.** `out/aligner/kitchen_fused.png` (the crop, 727x502) and
`out/inspect/kitchen_aligner.png`, registered to `out/inspect/kitchen_reference.png`
by template match: **score 1.0000, PASS** (bar 0.99), crop origin (16, 8). The
certifier was SKIPPED on the clock — `forward_certify.certify` wants a `SceneModel`
built by `model_from_pass1`, which is a separate pass-1 run; it is the first thing to
add and it is the only no-reference arbiter this scene has.

**ZERO-MOTION ANCHOR (`aligner.py sentinel`) — fails, at a 1 px rim, with the cause
fully attributed.** Built as `tests/test_twoframe_route.py::_still_stack` builds it
(frames differing only by sensor noise). Result: **1 piece** (correct), fitted
transform `tx = -0.00255 px` (identity to 2.6 milli-pixels), and the veto ledger
takes NOTHING at occlusion, gate or photometry (0.9952 -> 0.9952 -> 0.9952 -> 0.9952).
The only losses are **off-footprint 0.48% and the 1 px gap dilation that doubles it**
to 0.96-1.60% per frame — i.e. the outermost rim, because a field that is 0.0026 px
short of identity samples 0.0026 px outside the frame. Zero interior gaps. F101's
anchor demands byte-identical, so this MUST be closed before promotion, and the fix
is small and principled rather than a tolerance: a fitted transform whose
displacement is below what the arc can resolve IS the identity (the same reasoning
`GATE_TOL` already encodes), so snap it, and let the footprint clause admit a
half-pixel of border slack.

## What remains, for the integration/promotion round

1. **Replace `GATE_TOL` as the merge tolerance with a fit-uncertainty test.** Both
   scenes' remaining segmentation error is this one decision: the factory holds two
   spurious cuts at 2.0-2.6 px (fit error, not depth) and the kitchen merges real
   objects at under 1.5 px (depth, not fit error). A fixed pixel bar cannot separate
   those; a model-selection test can — does ONE affine on the union explain both
   supports as well as two do, measured against the residuals the fits already
   report. This is the single highest-value item and it is why no piece bar passed.
2. **Nothing yet PROPOSES a cut from motion.** The merge can only remove cuts, so cut
   RECALL can only fall (74.74% -> 68.23%), and on the kitchen the objects were never
   seeded at all. The charter's design is half-built until a motion-disagreement
   field can open a cut the depth seed missed.
3. **Close the zero-motion anchor** (identity snap + border slack, above).
4. **`travel` -> displacement at the piece CENTROID**, one line; occlusion order
   currently rests on a quantity that reads 7822 px/frame on the kitchen.
5. **Isolate the K2b silhouette gate.** It is now a live defect, not an inherited
   one: the merge cleaned the cut set and the gate's truth baseline ROSE (10.75% ->
   15.04%, separation 1.5x -> 1.2x). Round 1's attribution of it to the cut was wrong.
6. **The certifier on the kitchen output** — the only no-reference arbiter for the
   boxes' disagreement with the routed default, and unrun.
7. **The photometric veto is whole-frame** (round 1's note 4), so 27% of the frame
   being withdrawn cannot be attributed to a piece without re-running. Per-piece
   attribution is what would turn the gap ledger into a diagnosis.
8. Known and expected, not a bug: the glass pitcher violates one-surface-per-pixel
   and is inside the big piece; it is gapped per frame by the veto, as F117 recorded
   it would be.

# Round 3 — the guide, wired

The reference-collapse diagnosis named two root causes and both were WIRING, not
estimation. Round 3 wired them and measured the price on both scenes.

1. **Motion-group-seeded cuts.** Nothing proposed a cut from motion, so the
   kitchen's objects stayed fused to the counter mega-piece, inherited its wrong
   affine, and were correctly gapped by the veto in exactly the sharp frames. The
   organ that finds them already exists, ships in the runtime, and is proven
   (F93/F100/F102): `focusstack.motion_groups.overrides`. It is now called from
   `aligner.seed_from_motion_groups` at **the same entry point `align.py` uses**,
   `overrides(images, coarse, valid, ref_index, depth, displacement_at)`, with the
   aligner's own per-piece series substituted for the depth bins inside
   `displacement_at` — so "a group whose motion disagrees with its ENCLOSING
   PIECE" is the question the existing organ already answers.
2. **The two-axis equivalence test.** The cut/merge decision was translation-only,
   which merges different-depth objects that translate alike while scaling
   differently — and `motion_components` measured this scene at up to 4.3% forward
   translation. The decision fit is now 3 DoF (translation + isotropic scale about
   the piece's OWN site centroid) and the decision is the residual PAIR
   (Δtranslation, Δscale) compared against the FIT'S OWN uncertainty per axis.

## The tolerance, and why `k` is defensible

`fit_affine` returns the least-squares covariance of the decision fit, inflated by
an EFFECTIVE SAMPLE SIZE: sites along one contour do not carry independent errors
(the profile matcher's error at a site is dominated by that contour's texture and
defocus), so the variance is multiplied by the measured sites-per-connected-contour
ratio. Without that inflation the naive sigma sat below the instrument floor and
the "uncertainty" test degenerated into a fixed bar (7 factory pieces, K4 0.977485).

`DECISION_K` is calibrated on the factory, where truth adjudicates, and it is a
PLATEAU rather than a point:

| k | pieces | mean purity | worst purity | cut recall |
|---|---|---|---|---|
| 12 | 7 | 0.964 | 0.687 | 68.23% |
| 20 | 6 | 0.969 | 0.933 | 68.23% |
| **25** | **6** | **0.969** | **0.933** | **68.23%** |
| 30 | 6 | 0.969 | 0.933 | 68.23% |
| 35 | 4 | 0.869 | **0.657** | 42.72% |
| 45 | 3 | 0.869 | 0.620 | 18.81% |

k=20/25/30 produce the IDENTICAL piece map; k>=35 falls off a cliff — 4 pieces, but
the NEAR PLANE has been absorbed into a far piece (worst purity 0.657, recall
42.72%). A decision that is insensitive across a 1.5x range of its only free number
and then fails abruptly is evidence that the quantity being scaled is the right
one; a tuned threshold has no such plateau. 25 is the plateau's centre.

## THE FINDING OF THE ROUND: the transform instrument's capture range is ±6 px

The first kitchen run failed in a way that is worth more than the round's other
numbers. Seeded pieces were carved with the bottle's own motion group (pass-1
motion +24.49 px at k=11), and the merge then RE-ABSORBED them — legitimately,
because their refits read Δt 0.70 px against the mega-piece. The refits were
wrong: `fit_affine` accepts a site only when `|shift| < SM.CONTOUR_HALF` (**6
px**), so a ~20 px object displacement is entirely OUTSIDE the dense
normal-profile matcher's capture range. Started from the global affine, every site
on the bottle was rejected and the fit collapsed to ~+3.9 px, i.e. to the geometry
the seed exists to contradict.

The cure is the charter's own principle read operationally: **a seeded piece's fit
PRIOR is pass 1's measured group motion** (`prior_k @ T(dx, dy)`), which puts the
piece inside the instrument's capture range so the fit can refine rather than
reject. `motion_series`' F106 fallback follows the same prior — an unverifiable
seeded piece keeps pass 1's measurement, never the global affine. With that, the
same pieces are HELD by the two-axis test at Δt 19.94 px and 12.81 px.

Consequence for anyone using this module: **no fitted per-piece motion above ~6 px
can be believed unless it was PRIORED there.** That bound was invisible for two
rounds because the factory's largest differential is 3.2 px/frame.

## Results, both scenes, honestly

**Factory (regression gate).** 6 pieces (round 2: 5), K4 with gaps **0.980472**
against round 2's 0.980686 (−0.000214, one extra far/far cut held), gaps OFF
0.944404, reference alone 0.965826. Mean purity **0.969** (up from 0.963), worst
0.933 (unchanged), cut recall 68.23% (unchanged), precision 40.52% (was 43.01%),
**wall-smear leakage 0 px in every frame** (unchanged), K2 on true pieces 0.159 px
(unchanged), K3 recall 0.686–0.902 (unchanged). `motion_groups` proposed **nothing**
on the factory (`overridden: 0`), so the entire factory delta is Change 2's.
Verdict: holds within one piece and 2e-4 of score; not an improvement.

**Kitchen.** 3 pieces (round 2: 11), travel **5.86 / 24.78 / 12.12 px/frame** —
sane for the first time (round 2 reported 7822 and 350, fixed by measuring travel
at the piece CENTROID). `motion_groups`: 242 features, 3 groups, **2 overridden**,
1 rejected as majority, 155 unexplained points. Mean gap **54.32%** (round 2:
57.19%) but the photometric component **ROSE to 33.83%** (round 2: 27.21%), now
attributed per piece: tin 71.0%, counter mega-piece 39.5%, seeded mid-depth piece
36.2%. Focus energy vs reference / vs the routed default: box1 +13.9/+13.7%, box2
+17.2/+13.6%, box3 +2.8/+2.4%, box4 +47.3/+47.4%, knob +28.7/+27.7%, back shelf
+27.8/+20.1%, counter +10.5/+8.7% — round 2 read within 1–2% of the reference
everywhere. Box |Δ| 8.70/26 · 5.19/32 · 1.88/38 · 10.37/78 against the routed
default's 1.20/2 · 2.04/13 · 1.19/19 · 1.03/17; flank mean 2.579 with 4.91% >12
against 0.897/0.01%. Wall test PASSES (far frames 64%/57% gapped, reference 0%).
Inspector registration 1.0000.

## And the honest verdict, from the pictures rather than the numbers

`out/aligner/GUIDE_boxes.png` (routed | aligner | reference) kills the reading the
energy table invites. **The energy rose because content MOVED, not because it
sharpened.** Box 2's lid is visibly displaced with a doubled edge; box 4's rag has
a hard bright rim present in neither the routed path nor the reference; box 3's
Coke top is shifted with a gap sliver beside it. §12.8 exactly: both times a story
survived the numbers in this arc, an image killed it.

The cause is visible in `out/aligner/kitchen_pieces.png`: **the Lubriderm bottle is
NOT its own piece.** The seeded support is `hull ∩ focal band`, and the band was
read as the [5, 95] percentile of the focal signature inside the group's own claim
core — on the kitchen that core is impure, so the band came out **3.0–11.0 of 12
frames** and trimmed essentially nothing. The surviving seeded piece therefore
carries the bottle's MOTION over a mid-depth BLOB (cocoa tin + yellow rag + stove
wall + shelf + the bottle's top), and one affine over that blob misplaces all of
it. The gates then opened — correctly, on their own terms — over wrong geometry.

So: Change 2 lands and is calibrated. Change 1 is wired, fires, and its cut is now
HELD instead of merged away — but its SUPPORT CONSTRUCTION is the next defect, and
until it is fixed the kitchen's sharpening must not be claimed as a win.

## What remains, in priority order

1. **The seeded support is the whole remaining kitchen gap.** The focal band must be
   derived from the group's FEATURES' own focal frames (each feature's
   `_focal_frames` value, which `motion_groups` already computes internally) — not
   from a percentile of the claim core, which is contaminated by everything the
   convex hull swept in. Failing that, split the carved support by connected
   focal-signature component and let the two-axis test decide each piece.
2. **Round 2's items 1 and 2 are closed**; items 3 (zero-motion identity snap — the
   sentinel still shows 0.96–1.60% gaps and |M − I| 5.11e-3), 5 (the K2b silhouette
   gate, separation 1.2x), 6 (the certifier on the kitchen), and 8 (the glass
   pitcher self-vetoes — stated, not fought) stand. Item 4 (travel at the centroid)
   and item 7 (per-piece gap attribution) are done.
3. **The photometric veto went UP, and that is a symptom, not a regression to tune
   away**: 33.83% withdrawal is the veto correctly refusing the misplaced blob. It
   should fall on its own when item 1 lands. Per-piece measured `c` is still not
   built; `c` reads 1.844 px/frame here against round 2's 1.403 on the same scene,
   which is itself a warning that a whole-frame `c` is being fitted to whichever
   geometry the round produced.
4. The factory's −2e-4 is inside the granularity of one piece. If the plateau's
   lower edge (k=20) is preferred for regression safety it produces the identical
   piece map, so the choice is free.
