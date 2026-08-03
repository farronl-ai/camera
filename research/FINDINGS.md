# Current findings

This is the compact scientific state of the project. It is not a chronological
lab notebook; superseded experiments and their reports remain in Git history.
Read `MISSION.md` first, then this file, `OCCLUSION_FORMATION.md`, and
`STATE.md`.

## F118 — The aligner exists: the transform is solved, the wall smear is unconstructable, and one decision remains

Aligner rounds 1–2 (Opus, manager-verified: commits and evidence images
inspected; the ladder's arithmetic checked). `research/aligner.py` implements
the FRONTIER contract: piecewise-affine backward fields cut at occluding
contours, gaps proposed by geometry and vetoed by photometry, one resample,
output `(aligned, usable, report)` consumed by UNMODIFIED `fuse_perband`.

**The contract's ceiling is real and high.** On the factory with TRUE pieces:
**0.986238** — above every path ever measured here (runtime two-frame
0.984455), 0.0032 from the one-resample floor. Transform estimation is
SOLVED: worst 0.159 px, worth 0.00002 of score. **The wall-smear class is
structurally absent**: zero occluded pixels leaked on the factory (truth
adjudicated), and on the kitchen the signature test passes with per-frame
proof — every non-reference frame gaps the wall band the cocoa tin's parallax
sweeps (49–79% gapped; reference 0%), and the fused tile shows no smear and
no doubled edge.

**The model-order finding (both halves cost a failed attempt):** rotation and
magnification are depth-INDEPENDENT (why one global affine carries
breathing), parallax is a per-piece translation — so **2 DoF decides the
cut** (6-DoF fits fired zero merges: extrapolation noise 1.75–53 px swamps
agreement) **and 6 DoF, refit once on merged support, is the transform**
(2-DoF transforms cost 0.9862 → 0.9756: the global prior's linear part is
depth-contaminated). The affine-agreement merge converges 24 → 5 monotonely
and bought two thirds of the piece term (+0.0098 of +0.0154; K4 recovered
pieces 0.9707 → **0.9807** with gaps, 0.9457 without — the gaps themselves
buy +0.035 on imperfect pieces). Added physics the factory forced: pieces
with equal travel are at one depth and cannot occlude each other; occlusion
order comes from parallax magnitude (better than F83's focal-peak proxy).

**The kitchen names the one remaining decision.** Under-segmentation of the
wrong half: the ramp correctly stays whole, but the discrete objects share
the counter's affine (differential parallax ~ `GATE_TOL`, consumed by the
merge) — box 4 reads 10.05/81 vs the routed default's 1.03/17, attributed to
the cut, while box 3's mean BEATS routed and every box is a sharpening, not
a wash (focus energy ≥ reference in all four). Mean 57.2% of pixel-frames
gapped, photometric veto dominant (27.2%) — the veto doing its job on a
geometry that is wrong. The decision: **the merge tolerance must be a
fit-uncertainty (model-selection) test, not the resolution bound
`GATE_TOL`** — and nothing yet PROPOSES a cut from motion (a merge can only
remove, so recall can only fall). Zero-motion anchor fails ONLY at a 1 px
off-footprint rim (identity to 0.0026 px, zero interior gaps) — an identity
snap closes it; F101 demands byte-identity before any promotion. `c` is
confirmed per-scene (true-piece factory 1.120, recovered 1.212, kitchen
1.403; the runtime still carries 1.161 unmeasured). The full 8-item
promotion list is in `research/aligner_NOTES.md` Round 2.

## F117 — The runtime retirements: physics in the default path, and a licence honestly left alone

Round C (Opus, time-boxed, manager-verified: factory 0.984455 and the kitchen
box table reproduced exactly; both evidence crops inspected; 96 tests).
Retirements 1–2 shipped in `src/focusstack/twoframe.py`; retirement 3
(licence-before-render) DROPPED per time-box, still open.

**Retirement 1: `SURFACE_SIGMA` had two jobs, and the research port covered
only one.** The faithful cross-convolution port reproduced its KAT to three
decimals and then FAILED box 1 (6.14/61 → 10.42/156, F112's defect returning
crisper): the global sigma had also been POOLING the decision. Shipped as
cross-convolution + unanimous verdict over `content_aware_energies`' own
smoothing window (borrowed, not invented; 5/7/9/11 sweep is a plateau).
Runtime results: four boxes **1.20/2 · 2.04/13 · 1.19/19 · 1.03/17** (were
6.14/61 · 3.49/17 · 3.31/101 · 6.23/127), flank **0.897 / 0.01%>12**, factory
**0.984455**, sentinels byte-identical, large-motion still declined.
**Honest cost:** the kitchen default is now more conservative — 67.6% of
routed pixels changed, whole-frame focus energy 48.1 → 43.5 (reference 42.7);
box 1's pot and box 4's rag lose texture toward the reference. The certifier
backs the trade (input-routed 9.52 → 7.89; second pass on the new base 7.17),
and the factory's ground truth gained +0.005 — but the USER's eyes validated
the old default, and layer 1 of the inspector is now this new output: their
re-inspection is the outstanding audit.

**Retirement 2: pair-aware refusal, licensed by retirement 1.** Where F82's
geometric suspicion was the ONLY objection, a present member the appearance
evidence licenses now beats the reference fallback — trinary, with two
restrictions the factory's ground truth wrote (evidence must DISCRIMINATE
between members; the member must be modelled sharper than the reference;
unrestricted, tier-2 content was worse than the fallback it replaced,
0.9844 → 0.9814). Large-motion box: **47.4% recovered**, focus energy
31.6 → 74.2 (frame 0 holds 152.4) — legible but soft and mottled. **The 14 px
licence stays**: the repaired composite does not yet match the shipped
override's crispness there.

Open from the round: the blur-rate constant `c` DOES NOT TRANSFER across
scenes (factory 1.161, kitchen 0.684, large-motion ~0.4 — it over-blurs 2×
there) and nothing in the runtime measures it — pooling is currently
absorbing the error; `research/layer_decompose.py` now runs with pooling via
the default radii (flagged); the second pass's certified coverage on the new
conservative base shrank to 29.6% and its box-1 max reads 62 vs its recorded
61 — the second pass's rewrite role shrinks as the default grows
reference-faithful, which is the expected dynamic on the way to the aligner
contract.

## F116 — Contour continuity: the veto that polices position, not contrast

Round B3a (Opus, manager-verified: kitchen bars reproduced through the module's
own run; both defect crops inspected). The instrument F115 proved missing: a
rewrite may not MOVE a strong contour that the input composite and the
reference frame agree on. Implemented as clause 4 of `scene_model.py`'s veto
stack, composed LAST per §23a's non-monotonicity hazard, strict subset held
(kitchen 69294 → 67873 px, 0 rescued).

**The KAT inverts a recorded hazard into the required property.** PLAYBOOK's
"a blurred profile correlates confidently against a sharp one at about zero
shift" — a defocus BIAS when fitting motion — is exactly what makes profile
correlation along the normal a defocus-INVARIANT test of contour stasis.
Known answers: +1.00/+2.00 px read 1.000/2.000; sharpening (disk 1/2/4 → sharp)
reads ≤0.005 px and flags 0 of 120 agreed sites. Injected strip displacements:
70.9%@1px / 78.4%@2px against normal-observable sites (the shortfall from raw
32.5%/47.0% is the aperture problem, honestly scored), 0-px control clean,
off-strip false alarms 0.00%. On the factory, adjudicated by ground truth:
6 flags of 4114 agreed contour px, of which 2 are false (0.049%) and 4 had
genuinely moved. Both kitchen residuals detected blind from coordinates alone.
Profile geometry fixed by a pre-stated KAT rule before any bar was read —
and the sweep shows (half, span) = (6, 1) vs 2 is exactly box 1 at 61 vs 83.

**Bars: the arc's standing miss is closed.** Box 1 max 98 → **61** (= routed,
0 px above; mean improves 6.14 → 5.46; focus energy 44.8 vs routed 41.8 — the
withdrawal cost none of the resolved background). Box 4's junction profile
returns to the reference's shape (row means 142.9/93.5 vs reference
140.6/101.5). Every user box now at or under its routed max (61/16/101/119).
Factory 0.982034 (−0.000028 for the clause); knob 1.01×; flank 0.23%/29
identical; 94 tests; inspection layer registration 1.0000.

Open, ranked: the kitchen flags **19.10%** of agreed contour px against the
factory's 0.15% — a 127× ratio, so the factory's false-alarm bound DOES NOT
TRANSFER; 1421 kitchen px revert with no ground truth adjudicating any of
them. Bounded risk: the clause only withdraws to the input, so its worst case
is lost sharpening, never new structure — but the certifier is structurally
blind to this defect class (it can only fail to object, and the clause costs
it +0.024 levels), so the user's eyes are the only instrument that can audit
those reverts. The instrument under-reads below ~0.3 px; the 2 px slack is
what covers a cluster.

## F115 — The scene-model second pass ships a composite, and aggregation turns out to buy nothing

Round B2 of the scene-model arc (FRONTIER §7b), Opus-built.
`research/scene_model.py` assembles per-layer appearance from the frames and
rewrites the routed two-frame composite ONLY where B1 owns the pixel and the
certifier prefers the result. `forward_certify.py`, `src/**` and `tests/**` are
byte-identical to F114 — the file-scope allowance to add a certifier hook was
measured unnecessary and not used. Write-up in `research/scenemodel_NOTES.md`.

**`SURFACE_SIGMA` is retired by physics.** F112 logged the low-pass scale as an
unresolved §12.3 split (factory wants 2, kitchen wants 8, 4.0 clears every bar);
F112/R5 named the invariant and did not build it. The answer is not to make the
low-pass EXCEED the residual defocus difference but to REMOVE it exactly: two
observations of one surface satisfy `m (x) disk(R_r) == r (x) disk(R_m)`, so
each is convolved with the OTHER's disk. Cross-convolution, no PSF-family
mismatch (PLAYBOOK §0: defocus is a disk), radii integer by construction, `R =
c·|k − peak|` with `c` regressed from the certifier's own KAT-2'd radius search
— **1.161 px/frame against the factory's true 1.15**. On the COMMITTED fixture
and the committed pass marks it clears all four F112/R3 clauses and is exact
where σ=4 failed: disk defocus at 8 and 12 px reads **1.000 / 1.000** against
0.931 / 0.759, while a moved occluder still reads 0.002–0.010. One scale-free
number survives (`SIGMA0`, the sampling scale) and a 0.5/1.0/2.0 sweep does not
move either verdict. NOT yet ported into `twoframe.same_surface` — `src/**` was
read-only.

**Bars.** Factory GT-SSIM **0.979453 → 0.981104** (certifier 3.3051 → 2.9080).
Kitchen certifier **9.5242 → 8.7700** on a rewrite covering 25.4% of the crop,
byte-identical elsewhere (asserted in code). The F112 knob is REPAIRED: its
certifier differential goes from 2.13× the frame mean to **0.54×**, i.e. quieter
than average. The F108 flank box goes 0.57% → 0.40% > 12, and the 10 surviving
pixels sit at x662–669 y240–242 — the knob's own top edge, above its recorded
10×70 box. The pale sliver resolves into real background structure.

**The round's main result is a negative.** Once averaging across a focus
disagreement OR an unverified geometry is forbidden — and both must be —
multi-frame aggregation beats the per-region best SINGLE admissible frame by
+0.0032 on the factory and **−0.0407 on the kitchen, i.e. not at all**. The
second pass's win came from better ADMISSION, not from more data. Getting there
cost an F106 violation the eyes caught and no aggregate did: a first build
averaged any frame within 1.0 px of the sharpest modelled radius, including the
41-of-72 (frame, layer) fits whose gate returned UNVERIFIABLE and kept the global
affine. It scored BETTER on the certifier and was visibly softer (Lubriderm
label illegible; focus energy on rewritten pixels 0.969× the input, now
**1.133×**). The suspicion of a certifier sharpness bias was then tested rather
than assumed — the same composite Gaussian-blurred scores 9.52 → 9.68 → 10.12 →
11.10 at σ 0/0.5/1/2, so **the certifier has no sharpness bias on the kitchen**;
it was blind, not wrong, and an image killed the story for the third time (§12.8).

**Ordering is non-load-bearing, measured.** The visibility test uses no ordering
at all (F114 §9: F83's bit refuses on both scenes), and against the ordered
variant it produces IDENTICAL composites — the occlusion refusal fires on 0.025%
/ 0.160% of owned pixel-frames, because B1's 5 px boundary band already declines
a wider ribbon than the adjacent layers' differential motion. decompose_NOTES
§9's prediction, confirmed: it becomes load-bearing the moment anything completes
occluded content, and nothing here does.

**The factory remainder, re-attributed:** motion +0.0020, segmentation +0.0047,
and **assembly + render 0.0122 — of which 0.0105 is ONE round-trip resample of
the ground truth**. 86% of what survives true masks and true geometry is the
price of moving a frame, so B3's target is not a better estimator. (The oracle
ladder's first build substituted the true TOTAL shift into the residual's slot
and scored 0.9414 against the estimate's 0.9811 — F110's trap, same ladder,
caught by the same discipline.) Also open: a visible sharpness step at the
rewrite frontier (high-frequency 5.30 vs low-frequency 2.28, so not a gain
mismatch); the kitchen's radius-model residual is 3.5× its own slope, which is
F114's ramp quantization arriving in a new instrument.

**Correction rounds (manager-driven, two).** The manager's EYES rejected the
first build: the numbers above all passed, yet the rewrite had introduced new
localized structure in the user's own boxes (a wall streak and pump spur, a
silhouette line, shelf streaks and dashes). Diagnosis beat the hypothesis: the
worst offenders were never frontier effects but **components admitted with no
verdict at all** — `regions_of` filtered small components out of the ledger,
and a veto that only removes pixels cannot skip what never reached it (30
components, 1304 px, zero coverage). Fix: one local-veto rule at three scales
with NO new tuned number — every component judged; a pooled cluster clause at
the certifier's own measured sensitivity floor (its localkat: a 9×9 defect
puts 65.5% of its residual on itself, a 5×5 only 23.7% — certifier-only local
clauses are blind below a few dozen pixels); a frontier agreement clause.
Withdrawal IMPROVED the factory (0.981104 → **0.982061**) and the kitchen
(8.770 → **8.697**), knob repair 0.54× → 0.95× (still passing), flank 0.23%
max 29, box 2 restored to 16 ≤ 17. Structural guarantee: each correction's
rewrite set is a strict subset of the last, pixel-identical where both write.

**The second correction was built, priced, and honestly NOT shipped.** The
manager-prescribed clause (abstention near a geometric boundary is refusal)
was falsified on both residuals: neither is boundary-adjacent under B1's own
band — box 1's fleck is interior to one layer (B1 drew no edge there, so an
F92 licence cannot be cashed through B1's band) and 60 of box 4's 104 px are
positively certified. Two durable findings instead: (1) **refusal composition
is non-monotone** — deleting a rewrite pixel upstream can delete a frontier
SEED and make a downstream clause withdraw LESS (the strict-subset assertion
caught 438 rescued pixels breaking the knob to 1.58× before any bar was read);
(2) **focus energy cannot be evidence that rewritten content is real** — both
residuals RAISE it while moving a contour (the fleck translates the silhouette
one pixel; the shelf contour is displaced ~2 rows and steepened) because focus
energy is monotone in edge contrast and blind to edge position. The residual
class is sub-arbitrable interior GEOMETRY, and the missing instrument is
**contour continuity** — a rewrite may not move a strong contour the input and
reference agree on. That is round B3's charter, with the temporally-coherent
motion series. Final accepted state: box maxes 98/16/101/116 (box 1's 98 > 61
recorded as a proven-geometric MISS, 6 px), 94 tests, inspection layer
registered 1.0000.

## F114 — Decomposition by focal signature: ownership is requirement-dependent, and a layer is a quantization of a ramp

Round B1 (Opus, manager-verified: ladder and kitchen KAT-4 re-run and
reproduced; committed code gives segmentation **+0.138**, floor **1.367** —
the report's 0.100/1.328 was pre-final; verdict unchanged).
`research/layer_decompose.py` replaces pass-1's winner-map masks with a dense
focal-signature decomposition carrying trinary ownership (owned /
boundary-band / unknown; band half-width 5 px = the focus operator's pooling
radius, derived not tuned).

**The F98 reopening was justified, and the answer is requirement-dependent.**
Under the OWNERSHIP requirement (content assembly) the decomposition wins
decisively on the factory: purity 0.754/0.898 → **0.989/0.996** against true
plane masks, cross-plane contamination 16.1%/13.1% → **0.6%/0.5%**, the
attributed segmentation term 2.925 → **0.138** (21×; +1.895 in the
coverage-control configuration with the boundary band off — both reported,
and the control also puts 1–3 clusters inside the verified-clean flank, which
is the measured argument FOR the band). Under pass 1's SUPPORT requirement on
the continuous-depth kitchen it does not beat the winner map at matched
coverage — F98's negative stands where it was measured and does not extend.

**The kitchen's lesson: a depth LAYER is a quantization of a depth RAMP.**
The countertop recedes continuously; one disk radius per band is a poor
forward model (bands saturate `RADIUS_MAX`). Pass 1's winner map is a bad
ownership map and a good BLUR map — it groups by which frame is sharpest,
which is exactly what the renderer consumes. The scene model wants a hybrid:
discrete ownership where occlusion boundaries exist, continuous blur/depth
elsewhere.

**Knob attribution moved; still not localized.** With the new masks the
absolute floor drops (10.90 → 9.52) and the sliver improves (absolute rank
3 → 1), flank stays at ZERO clusters at all nine settings — but the knob's
peak (18.5) sits under ~15 larger localized differentials concentrated in the
deep-background band where the model is crudest. Those are model error, not
undiscovered defects. It needs a REGION-SCOPED null (per-layer appearance is
the material for one), not a better rank.

**Stale claim caught: F83's ordering bit REFUSES on today's factory** —
`occlusion_order.py` prints `near_is_low_index=None` (truth: True), share
0.542 vs its own 0.05 margin, most likely since F96 enriched surface texture.
Contour localization still holds (5.1 px). FINDINGS' "global polarity bit
voted correctly" is corrected here: occlusion ordering is currently a
focal-peak proxy whose guard refuses, and it is NOT reliable enough to
assemble content that some frames occlude. Certifier/GT agreement held
throughout (two-frame 3.31 < shipped 4.15; GT composite wins outright).

For round B2: trinary fractions factory 81.0/17.2/1.8, kitchen 57.4/35.4/7.2
(near layers own least — 18.6%); **motion is now the largest model term**
(+0.812 of 1.367), and raw per-frame fits are non-physical (−18.83 px beside
−5.26) — a temporally-coherent per-layer motion series is the missing
estimator; kitchen certified coverage 43.3% with the drop honestly
reclassified (24.0% boundary, ~23% declined, ~9.7% clipping).

## F113 — The certifier: forward-render consistency is the first GT-free arbiter

Round A of the scene-model second-pass arc (FRONTIER §7b), Opus-built,
manager-verified (KAT-3 and KAT-4 re-run and reproduced exactly; scope and
evidence overlays inspected). `research/forward_certify.py` renders every
source frame from a candidate composite plus pass-1's scene model (layers,
motions, focal ladder, exposure gains) and compares against the RAW frames in
their own geometry — evading F81a, since alignment never touches those.

**KATs 1–3 pass.** Renderer vs the factory at true parameters: MAE 0.42 levels
(0.12 after the factory's own uint8 truncation), identity at the reference.
The per-layer defocus estimator recovers known radii 100% exactly and degrades
LOUDLY under injected geometry error (residual 1.66 → 6.42) — blur does not
silently absorb misregistration. Ranking: two-frame 5.96 < shipped 6.51,
agreeing with GT-SSIM; the ground-truth composite scores best of all four (no
sharpness penalty); self-consistency 0.0000.

**KAT-4, honestly split.** On the routed kitchen: the pale sliver localizes at
rank 3/16 absolute; the verified-clean flank is QUIET at all nine detector
settings; the F112 knob is elevated 2.3× with the right sign but is NOT
localizable — ~7× under the real-scene detection floor. Measured, not excused,
and it is round B's acceptance test.

**The floor is attributed, and the attribution is the finding.** Factory
decomposition of the model-error floor: renderer+PSF+exposure 0.44, +motion
0.81, +layer segmentation +2.93 (3.6× everything else, localized p99 47.6) =
3.92; the null candidate's own defocus adds only 0.71. Pass-1's layer masks
are the focus contest's winner map — **a fusion SELECTOR shaped like a
decomposition** — and any second pass assembling per-layer appearance inherits
that error multiplied. Fix the decomposition before building on it. Because
segmentation error is shared by every candidate, the certifier's primary
output is the DIFFERENTIAL (candidate − null) map; the absolute map is a
model-error carpet. Real-scene sensitivity today: ~10 levels absolute, a few
levels differential, certified coverage 65.6% on the kitchen.

§12 paid three times inside the round: swapped layer radii in the truth model;
disocclusion invented by nearest-fill and then certified as observed; and the
whole instrument ranking the factory composites BACKWARDS while it took the
composed global affine as the only geometry. A broken instrument does not look
broken — it looks like data.

Also measured for round B: per-layer motion beyond the global affine is worth
only −0.10 levels on the kitchen; the focal ladder is the largest single term
(−2.79); a per-layer scalar gain absorbs `normalize_exposure` to 0.015 levels
(solved, don't build for it); and raw per-frame fits from the existing
estimator produce non-physical motion series (−18.83 px beside −5.26) — a
reconstruction needing all N frames cannot get them frame-by-frame.

## F112 — The focus contest's missing precondition: both members must see the same surface

Fourth Opus round, manager-steered mid-diagnosis and manager-verified. The user
inspected the routed kitchen output and marked four defects (pot rendered in
front of the Lubriderm bottle; a faint second lid; a Coke-edge alias; a rag
alias). All four were ONE mechanism, in two halves.

**Half one (the steer): pair fusion made a geometric decision softly.** Pair
members are misregistered BY DESIGN outside their own layer, so choosing which
member supplies a pixel is choosing between two geometries — and per-band soft
weights (`fuse_perband`, harden 0.5, commented "as the shipped path does") leak
the misplaced member's coarse bands wherever energy contrast is weak. That IS
the faint-alias signature. The disease and its cure were already in the repo:
`fuse_perband`'s own docstring routes unstable N-frame stacks to `fuse_coherent`
(one shared edge-aware decision, made ONE-HOT before multiband reconstruction),
but the `len(images) > 2` guard excluded pairs — the caller that needs it
unconditionally. Fix: every pair now fuses through `fuse_coherent`.

**Half two (the agent, measured before changing anything): F109 §2's claim
holds only if both members observe the same surface.** "The layer a member gets
wrong is the layer it is defocused in" fails where parallax swung an occluder:
the members then see different OBJECTS, and the contest picks the more textured
one, not the nearer one. In flaw 1's box the pair's near-layer mask covered
0.0% — the bottle's smooth white silhouette lost every pixel to the sharp pot
print behind it, and every downstream gate (`_pair_refusal`, depth step, layer
boundary) was blind together because all are keyed on that mask. This is F108's
wall one level in, and it is why the steer's layer-mask ownership prior was
measured unusable and rightly abandoned. Fix: `same_surface` — defocus is a
low-pass, so one surface still agrees once both members are low-passed and two
surfaces do not; each member is admitted only where its low-passed appearance
agrees with the unwarped reference (a `GATE_TOL` shift, the measured 2%
exposure residual and a 1-level noise floor are explained). KAT'd both ways:
defocus/shift/gain must not trip it; a moved occluder must, at 0.000 in the
vacated and covered strips.

**USER-VALIDATED.** The user re-inspected the routed kitchen output against
the shipped path and the reference frame in the inspector and judged it "by
far the best" — the four marks that opened this round are closed by the same
eyes that made them.

**Verified numbers.** Factory GT-SSIM (runtime module, manager-reproduced
exactly): 0.971310 → **0.979453**, now BEATING shipped 0.972808 — F109's
promotion blocker is gone, and F110's "what is left is the architecture" was
right about the category but wrong about the part: it was the fusion. Fix 1
alone is factory byte-identical (its planes are textured everywhere — why two
rounds of factory scoring never saw this) and makes flaw 1 WORSE (one-hots the
wrong winner): the precondition is not optional. All four user boxes clean by
manager's eyes (`out/inspect/ROUND3_flaw{1..4}.png`). Sentinels byte-identical;
large-motion still declined. 94 tests.

**Instrument note (§12).** The round's recorded flank 0.76/9/0.00% was measured
on registered 8-bit inspection layers. On the CANONICAL F108 instrument
(original coordinates, vs the normalized reference frame) the composite reads
mean **1.11** (was 2.24), max 45, 0.57% > 12 — the entire >12 tail is one
30×70 px dark background knob at (659–669, 243–313) where the one-hot decision
now picks a single darker/crisper member rendering instead of the old soft
blend toward the reference. Sub-visible at 1×; recorded, not excused.

Open, ranked: `SURFACE_SIGMA` is an unresolved §12.3 split (factory wants 2,
the kitchen boxes want 8; 4.0 is the smallest value clearing every bar; the
physical per-pixel version is designed in `twoframe_NOTES.md`, unbuilt); a
faint pale sliver survives at the bottle's left silhouette; the knob above;
`fuse_perband`'s `len > 2` guard still excludes every other 2-image caller in
`fusion.py`; the `img46_routed.png` inspector layer is STALE (IMG-46 frames are
not in the repo).

## F111 — The two-frame route ships: engaged where it wins, vetoed where it loses

Third Opus round, manager-verified (port KATs exact; byte-identity reproduced
independently through the full pipeline; all review crops inspected). In the
package: `src/focusstack/twoframe.py`, routed in `pipeline.run` (default on,
`--no-twoframe-route`), 92 tests.

**The routing rule, validated with an amendment.** The candidate — engage where
the motion-group override fired — proved necessary but NOT sufficient: the route
also requires the composite to place every elected layer within the existing
refinement licence (1.5% of the diagonal, `_REFINE_MAX_FRACTION`, borrowed not
invented). Per scene: factory / zero-motion / small-motion — shipped,
byte-identical (`np.array_equal` on full pipeline output, route on vs off).
Kitchen (2.1 px of 14.0 licence) and IMG-46 (6.9 of 20.2) — two-frame.
Large-motion (19.2 px, over licence) — composite built and DISCARDED.

**The large-motion veto is the finding.** The two-frame composite there does not
garble the box — it loses it: the box is sharpest at frame 0 and needs +18.9 px;
the pair fits it correctly (0.51 px verified), then F82's disocclusion refusal
withdraws that member over 91% of the pair and the box returns
reference-defocused. F109 §2 predicted the regime. The architecture vetoes its
own route, and the shipped override output ships there — text crisp, confirmed
on the review crop.

**Kitchen through the routed pipeline**: flank mean 2.24 / max 11 / 0.00% > 12
(bar 2.2 missed by 0.04 with the cause measured, not excused: normalize_exposure
±1.5% per-frame gains; the same composite reads 2.16 from raw sources; shipped
moves 5.98 → 6.69 under the same gains). The F108 streak remains eliminated in
the shipped default path.

Open, ranked: a pair-aware refusal that prefers a present-but-defocused member
over a reference-defocused one would repair large-motion and move the route's
boundary; a declined composite is still rendered (~4 s wasted — the licence check
could precede rendering); the licence's scale is borrowed rather than fitted, on
one real scene per side; `--enhance` is correctly skipped on the routed path
(F56's licence does not cover a stitched composite).

## F110 — Two-frame hardened: verified gate, edge-refined fits, oracle corrected

Second Opus round (write-up in `research/twoframe_NOTES.md`, hardening section).
Verified by the manager: factory number reproduced through the agent's own eval
path; kitchen flank confirmed clean on an independent render.

- **Validity gate**: forward-verify a layer fit by asking the layer's own material
  edges whether the layer stopped moving (focal-weighted, observability-aware).
  Trinary per F106: verified applies; contradicted repairs from the gate's own
  measurement, refusing only if re-verification fails; unverifiable declines the
  correction but keeps the observation. Injected +8 px error: 0.9282 ungated →
  **0.9710 gated**; +20 px: 0.9712. Silent on correct fits (factory and kitchen
  identical gate on/off). The predecessor's 0.668 case class is closed.
- **Estimation**: rigid layer geometry +0.0044, edge-refined fits +0.0016 —
  factory 0.9658 → **0.9713** (shipped 0.9728).
- **The prototype's oracle was wrong**, caught by KAT discipline applied unasked:
  exact analytic shifts run THROUGH the architecture score 0.9688 — below the
  hardened run — so F109's "the gap is estimation" claim is false; the remaining
  −0.0015 is architecture (degenerate single-frame regions, stitch, refusal
  trade). F109 stands corrected on that point.
- Full-res: candidates transfer by matrix conjugation `S·M·S⁻¹` (F107 with no
  field resize), KAT'd at 2×.
- Known gap for routing: one kitchen pair region (27.7% of frame) has zero
  material features and declines its correction; F104 may licence limb edges for
  VERIFICATION there. Tri-modal tiles measured (78% trip a relative bar that a
  receding countertop trips everywhere) — deliberately not built on.

## F109 — The two-frame architecture passes the F108 acceptance test

Prototyped by an Opus 5 subagent under the §10 delegation discipline
(`research/twoframe.py`, full write-up and negative deliverables in
`research/twoframe_NOTES.md`); verdict verified independently by the reviewing
agent on its own render.

The architecture is the user's design: per region, elect the frame where its
foreground is sharpest and the frame where its background is sharpest; each frame
of a pair is warped by ONE rigid transform (global affine ∘ layer shift, composed,
single resample); `fuse_perband` per pair; one-hot ownership; `multiband_blend`
stitch. The structural property that closes F108's gap: each pair member owns
exactly one depth layer, so the layer it gets wrong is by construction the layer it
is defocused in, and the focus contest discards it — no depth-dependent field
inside a region, hence no ramp, no stretch, no soft geometric blend, and no need
for per-pixel evidence where none exists.

**Acceptance (the Lubriderm streak): passed.** On the measured low-contrast flank:
shipped mean |Δ| 5.98, 16.34% of pixels >12; two-frame mean |Δ| **1.96**, **0.00%**
>12. Visually confirmed by the reviewer: the tan column is gone, the flank clean,
the edge single. The deeper reason is humbling for the whole arc: the bottle is
sharpest AT the reference, so regional pair election never asks the far frames to
supply that surface — the +19.97 px correction F100–F108 fought for is simply not
needed under this architecture. The cat figurine, doubled in the shipped output,
is also single. Cost 2.2 s using 5 of 12 frames.

**Honest ledger.** Factory GT-SSIM 0.9658 vs shipped 0.9728 (−0.0070) — and the
oracle rung proves it is ESTIMATION, not architecture: exact per-layer shifts give
0.9730, and two-source fusion is ~2.7× more sensitive to a misplaced layer than
N-frame. A wrong-but-in-range layer shift once took the factory to 0.668 with
nothing objecting — a validity gate on layer fits is the promotion blocker.
Notable measured non-result: hard paste and multiband stitching are
indistinguishable here because every candidate shares the reference geometry —
F79's seam is geometric and does not arise.

Integration prerequisites (in the notes, unstarted): layer-fit validity gate;
close the ~0.3 px layer-shift error (focal-weighted edge-profile estimator);
route between shipped and two-frame per F101 rather than replace; third frame for
three-mode regions; F107 rescaling.

## F108 — The wrong-group wipe, four kept fixes, and one precisely-characterized wall

The user reported a column of foreground displaced leftward with yellow background
in its true place, beside the Lubriderm bottle's lower right. The investigation
found one severe bug, three concept completions worth keeping, and one artifact
that survived five attempts and is now a named open problem.

**The severe bug: a wrongly-elected scene-wide group wipes every correction under
its hull.** A near-still background group (168 features spanning the frame, ~0
motion) was being chosen as an override; the trinary write then replaced the field
under its frame-wide hull with `base + ~0` — global-only geometry, rendering the
bottle ~19 px displaced with wall content in its true position. Its ownership
simultaneously voided the unexplained-motion refusal there. Two fixes, both kept:

1. **Disagreement is measured at the group's MEMBERS (median), never at a
   centroid** — one pixel of a scene-spanning group lands anywhere.
2. **The override's license is MINORITY rescue.** A group whose hull covers most
   of the frame IS the majority the global stage already fit; it may not override
   (hull area ≤ 35% of frame).
3. **Ownership requires explanation.** A disagreeing evidence point is accounted
   for only if its owner's motion matches the point's own measured shift — 
   coverage alone is territory, and territory must not void the refusal net.

**Also kept, measured positive: the bins' membership transition band joins the
refusal gate.** The band is soft-blended geometry — F106's sin, still living in the
bins stage since F81, priced in before refusal existed — and its depth-step-gated
ribbon cannot catch it at smooth junctions (F104). Refusing the band where the
adjacent bins' fitted shifts genuinely pull apart raised the analytic factory's
GT-SSIM 0.970768 → 0.973225. Also kept: unexplained-point refusal now chains points
into cluster hulls (objects are contiguous, F93) and applies to ALL moving frames —
the near-reference exemption was disproved by direct composite (ref-only in the
artifact box is clean; ref±1's ±3-4 px fan is real).

**The wall, stated precisely.** The tan streak persists. Mechanism PROVEN by
composite: it is the mixing of near-reference frames' fanned, uncorrected edges in
the low-contrast stretch of a partially-corrected object (white flank against light
wall). Five coverage attempts failed for one shared reason — every available gate
is blind exactly there: no detectable edges (evidence-driven refusal has nothing to
chain), the label evidence above is EXPLAINED and thus correctly excluded, the
guided depth ramps smoothly across the boundary (depth-value and depth-step gates
both fail), and a depth-adjacent dilation stops at the same ramp (rolled back after
buying +12% withholding for nothing). This is F98's zone-coverage gap at the
refusal level: pixels that carry no evidence can be neither corrected nor
selectively refused. The capability that closes it — support that does not depend
on local evidence, e.g. the two-frame architecture's per-region frame pairs, or a
non-edge coverage channel — is the next build, and this artifact is its acceptance
test.

State after the session: kitchen residuals f8 +0.78 / f11 +1.32 px, withheld
22.5%, factory GT-SSIM improved to 0.973225, all sentinels byte-identical, 85
tests. The streak is the known remaining defect, smaller than any of its ancestors.

## F107 — Full resolution by estimating small and applying native

The alignment estimators are pixel-scaled and validated at ~800–1100 px; at 24 MP
every measurement leaves its regime (the near object's displacement alone exceeds
the 40 px match cap several times over). The sampling-field architecture makes the
fix structural: a field maps reference coordinates to source coordinates, so

    field_native(X) = s * field_small(X / s)

Estimation runs where it is validated; the NATIVE pixels are resampled exactly once
through the scaled field; usable masks scale nearest; the crop is recomputed
natively; `fuse_perband` is structurally resolution-adaptive already
(`research/fullres_apply.py`). On IMG-46 at 5712×4284 this delivers an
artifact-free native result — the nutrition table is fully legible at 24 MP down to
"Pantothenic Acid 5.1mg", barcode digits resolved, cap single.

**And F106's safety net earned its keep on its first outing.** In this run the
grouping found one group instead of two and the override did not fire — the same
frames, downscaled in memory rather than through the saved 1080 JPEGs, moved the
greedy seed sampling enough to lose the bottle group. Under the old behaviour that
would have meant ghosts. Instead the 90 unexplained-motion points triggered
refusal, and the bottle renders clean with source-limited sharpness rather than
corrupted. Refusal converted a model failure from an artifact into a soft spot.

Open, honestly: the consensus grouping is provenance-sensitive (seed subsampling at
`pool[::len/24]` is brittle to pixel-level input changes) — it should seed from the
screening evidence, which already knows where unexplained motion clusters; and the
estimate-small/apply-native path lives in a research script and belongs in the
pipeline as the default for large inputs.

## F106 — A geometric decision cannot be soft, and unexplained motion obliges refusal

The user looked at the Lubriderm pump and saw three of it — one real, two faint —
and yellow wall bleeding in at the bottle's lower right. The same signature was
measured on IMG-46: label colour smeared ~20 px past the bottle's silhouette. Two
distinct causes, one concept each.

**1. The override was applied with soft weights, and a sampling field cannot be
blended.** A 0.5 weight between two displacement fields does not blend appearances —
it invents a THIRD position that neither surface ever occupied, and the support
ramp then renders exactly the observed smears. The repo already held this rule three
times over (F79's hard region choice, F82's hard refusal, the stretch limiter's
transport-not-stretch); the override had been exempted from the limiter and its
thin support-edge refusal under-covered the ramp. Application is now TRINARY:
inside the support (weight ≥ 0.9) the group's motion applies in full; outside
(≤ 0.1) the depth path stands; the ring between is REFUSED in every frame whose
geometry there is materially ambiguous (> 2 px), because the ring's true content is
unknowable from a frame that moved.

**2. Measured, unexplained motion was being dropped, and it must be refused.** The
pump's ghosts came from farther out: the pump sat OUTSIDE the support entirely — a
coverage-matching loop broke on the first frame where motion was visible, letting
one noisy measurement decide (fixed: any visible frame may confirm, none may veto,
per F99's zero-bias). But the deeper rule is that the screening pass had already
MEASURED the pump's limb edges moving ~19 px that nothing modelled, and then threw
that evidence away. Every disagreeing feature that no chosen support corrects is now
returned as unexplained motion, and its neighbourhood is refused in all frames
beyond ref±1: if it cannot be corrected, it must not be allowed to ghost. This is
F82's principle extended from "the observation does not exist" to "the observation
exists and nothing accounts for it".

Results: the Lubriderm pump is single and clean, the cap smear gone, the yellow-wall
bleed at the lower right gone to a faint edge tint; the IMG-46 bottle cap's ghost
step gone. Kitchen residuals improved again (f8 +0.78, f11 +1.32 px). Factory and
zero-motion remain byte-identical. 85 tests.

Honest costs: kitchen now withholds 20.2% of pixels per frame (was 7.1) — refused
regions fall back to the reference and its neighbours, which is visibly right here
but is a lot of withholding; IMG-46 withholds 5.6%. And a faint tint remains along
the bottle's lower-right silhouette. Both logged open.

## F105 — First unseen-scene validation: IMG-46

Four handheld 24 MP frames (glass table, transparent water bottle up close, specular
glass bowl, hills at infinity — an adversarial depth range), downscaled to 1080 px
per the working rule of reducing resolution rather than the scene. The full default
pipeline, untouched and untuned:

- 2.3 s end to end; 3 depth bins corrected; the motion-group override fired on
  exactly one object — the near water bottle, which is the kitchen-bottle situation
  replayed on data none of this was fitted to; 1.1% withheld.
- Override off, the bottle's nutrition label and barcode fuse doubled and
  illegible; override on, both resolve. The bottle CAP text ("RECYCLE...") resolves
  too — that is F104's limb-edge coverage extending the footprint over an
  untextured part, exactly as on the Lubriderm pump.
- Against the user's earlier full-resolution auto output from the older build, the
  new result at working resolution matches the label sharpness and renders the
  glass-table texture behind the bottle sharper.

Named limitation: this ran at reduced resolution. The motion-group parameters
(PROFILE_HALF, SUPPORT_RADIUS, STRIDE, screening offsets) are pixel-scaled and
validated at ~800–1100 px; native 24 MP needs resolution-adaptive scaling of those
— the same disease PLAYBOOK §0 already names for windows and metrics. That is the
path to full-resolution delivery, and it is engineering, not research.

## F104 — Coverage by motion-matched limb edges; refusal at the support boundary

The user looked all around the bottle and found what the right-edge residual could
not show: ghosted pump and cap, a doubled left shoulder, smudges at the base. All of
them ON the object but OUTSIDE the override's support — the group's fitting features
are the printed label, so its hull covered the label and the untextured rest of the
object still fused under the bin's contradicted +2 px.

**Fix 1: limb edges may decide COVERAGE, never the fit.** F92 bans limb edges from
rigid fitting because their view-dependent bias is a few px. But for "does this area
move +19 px with the group or +2 px with its bin" — a gap of >5 px by construction —
that bias is irrelevant. Any edge (limb included, plus material edges no group
claimed) whose own measured shift matches the group's motion within 3 px, at a frame
where that motion is visible along the edge's normal (the F103 vacuousness rule
again), joins the group's FOOTPRINT, chained outward from the fitted members so
nothing attaches across the frame. The pump outline moves with the bottle; the
background does not. Result: pump, shoulders and base corrected; the right-edge
residual also improved (f11 +2.22 → +1.80) because the whole object now moves as one.

**Fix 2: the override's support boundary joins the refusal gate.** The remaining
large-motion artifact — a white smear of mixed content where the card box meets the
table — sat in the support-transition band, which the depth-step gate never admitted
for refusal because the guided depth map is fuzzy at exactly that junction. But the
override KNOWS where it created a displacement discontinuity, and a measured >5 px
disagreement is stronger evidence of a real object boundary than the smoothed depth
map. The transition band of every chosen group is now refused (F82's rule, fed by
better evidence). The smear is reduced to a faint trace; withheld rises to 7.1%
(kitchen) and 8.9% (large-motion), the honest price of not mixing.

Remaining, honestly: a small dark fleck on the bottle's cap and faint low-left body
streaks — far below the previous artifact level, unexplained, logged open. Factory,
zero-motion and small-motion remain byte-identical; 85 tests.

## F103 — Fresh-eyes review of the override: two conceptual bugs, one accidental
## quadratic, and large-motion finally inspected

A model change brought fresh eyes over the arc's shipped code. Four defects, each
now fixed and tested (85 tests, +3 for the override, which had none):

**1. The still-stack gate was wrong by construction.** It gated the override on the
largest BIN shift — but a bin fit is a majority fit, and hiding the minority's
motion is precisely the failure the override exists to correct. The kitchen passed a
1.0 px gate at 2.46 px: the arc's flagship fix survived its own gate by 2.5×, on the
only scene it was validated on. And the gate never achieved its purpose, since bin
noise exceeds 1 px even on a still stack. **A majority statistic cannot gate a
minority-rescue path.** Replaced by evidence screening inside `overrides()`: a few
frames near the reference (±2, ±4 — near, because off the focal plane features read
zero confidently, F99) are checked for any cluster of features whose motion the
depth path does not already explain.

**2. Vacuous consistency: the aperture problem let static features join moving
groups.** A feature whose edge normal is perpendicular to a group's motion predicts
d·n ≈ 0, measures ≈ 0, and passes the residual test while carrying no evidence.
Measured: 37 static background features joined a moving group, ballooning its convex
hull across the frame (correction weight 0.28 at a corner 150 px from the object)
and biasing the group's perpendicular motion toward zero. Membership now requires
the group's motion to be OBSERVABLE along the feature's normal; perpendicular
features that genuinely sit on the object are recovered by a spatial attachment pass
(consistent AND adjacent to an informative member), which restores coverage without
letting anything attach across the frame.

**3. An accidental quadratic.** The consensus loop evaluated a full residual array
once per candidate feature (`residual(motions)[i]` inside a comprehension).
Rewriting it for bug 2 removed it: kitchen 23.2 → 3.1 s, zero-motion 45.7 → 4.0 s,
small-motion 47.4 → 5.4 s. With screening, the sweeps that override nothing now
cost ~4 s and return byte-identical output.

**4. Frames whose bins were all rejected could never receive an override** (the
application loop iterated only frames holding bin fields), and `displacement_at`
returned None for them so they did not even count toward disagreement. Both fixed:
the group's motion applies over the plain global warp there. Also fixed: the
unsupported-fit fallback in `_group_motion` (the F99 zero-bias trap, reapplied),
dead `bin_shifts` plumbing with a comment that mis-described it, and the missing
pipeline/CLI exposure (`--no-motion-override`) and docstrings.

**Large-motion, inspected at last.** The 28% of changed pixels are the playing-card
box: the bins path renders its text doubled and garbled ("LASeMeG"), the override
reads "LAS VEGAS" cleanly and matches the reference geometry. The −0.004 Q_SSIM that
had been logged as a caution was F81a's blindness a third time. One new, small,
localized artifact: a white smear where the box's bottom-right corner meets the
table — the support boundary zone — logged as open.

After fixes: kitchen f8 +8.4 → +1.0 px, f11 +20.1 → +2.0 px; factory, zero-motion
and small-motion byte-identical to `motion_override=False`.

## F102 — The motion-group override ships: the bottle is corrected in the runtime

The targeted override F101 specified, now in the package
(`src/focusstack/motion_groups.py`, wired into `align.py` behind
`motion_override=True`). The depth-bin path is untouched; a motion group replaces it
only where that group's own measured motion disagrees with what the depth path
applies AT THE GROUP'S OWN LOCATION by more than 5 px.

| sweep | groups overridden | pixels changed | Q_SSIM | align time |
|---|---:|---:|---:|---:|
| kitchen | 2 of 3 | 17.3% | 0.910336 → **0.918951** | 23.2 s |
| zero-motion | 0 of 1 | **0.00%** | identical | 45.7 s |
| small-motion | 0 of 1 | **0.00%** | identical | 47.4 s |
| large-motion | 3 of 4 | 28.2% | 0.927970 → 0.923764 | 66.8 s |
| analytic factory | 0 of 1 | **0.00%** | 0.970768 → 0.970768 | — |

The kitchen bottle's right-edge residual falls from +8.38/+11.98/+16.45/+20.14 px at
frames 8–11 to **+1.17/+1.60/+2.14/+2.51**, and the ghost band and doubled wordmark
along that edge are gone — the crop now matches the reference frame closely while
keeping the fused all-in-focus background. Non-regression is by construction rather
than by tuning: where nothing disagrees, nothing changes, exactly and bit-for-bit.

Three bugs stood between the research result and the runtime one, and all three were
in plumbing rather than in any model:

1. **The consensus radius was ported wrong.** At 1.4 px the target object's group
   does not form at all; at 2.0 px it forms at 93% purity. The research result used
   2.0 and the port hardcoded 1.4, so the override fired on the wrong group entirely.
2. **Re-limiting the field erased the correction.** The bin field is stretch-limited,
   and the override then deliberately introduces a step at the object's own boundary
   — a real depth discontinuity, not smearing. Relaxing it again left the bottle at
   +19.94 px instead of ~+1.5. That band is now refused the way F82 refuses every
   such band, rather than smoothed.
3. **Disagreement must be measured where the group lives.** Comparing each group
   against one frame-wide bin shift picked the wrong bin; the test now samples the
   depth path's actual applied displacement at the group's own centroid.

Honest caveats:
- **Cost.** Alignment goes from ~3 s to 23–67 s on these sweeps, because every
  material edge is profile-matched against every frame. The still-stack gate written
  to avoid this does NOT fire — some bin shift exceeds 1 px even on the zero-motion
  sentinel — so the two sweeps that override nothing still pay ~46 s to discover it.
- **Large-motion loses 0.004 Q_SSIM** while changing 28% of pixels. Q_SSIM cannot
  adjudicate an alignment change (F81a), so this is neither exoneration nor
  conviction; that sweep has not been inspected visually and should be before this is
  enabled by default for it.

## F101 — Motion groups and depth bins win on opposite scenes; the fix is a targeted override

Both blockers from F100 were closed. Seeding the grouping with the focal signature
(F97) makes it ENGAGE on the analytic factory — 2 groups, 97% near-plane purity,
where motion consensus alone collapsed to one — and the disocclusion refusal now
carries through the group path. The sentinels stay clean: zero-motion and
small-motion change by 0.04 and 0.10 px against the global-only result.

But scored on the same crop, the factory regresses:

| | factory GT-SSIM | kitchen bottle residual |
|---|---:|---:|
| shipped depth bins + refusal | **0.970768** | +19.97 px |
| motion groups + refusal | 0.900156 | **+1.47 px** (motion-seeded) / +4.44 (focal-seeded) |

The reason is not a defect in either method. The factory's two planes are cleanly
separated BY DEPTH, so depth bins are near-ideal there, while feature-hull supports
overlap and under-correct — the group differential reaches 3.19 px where the truth is
about 5.0. The kitchen is the opposite: depth cannot isolate the bottle at all (F99),
and only motion grouping does.

There is also a real tension inside the grouping itself. Focal seeding raises the
bottle group's purity from 92.9% to 100% but SHRINKS it from 14 features to 12, so
its convex hull covers less of the object and the residual worsens from +1.47 to
+4.44 px. Purity and support coverage trade against each other, and support coverage
is what actually determines whether the correction lands.

**Integration should therefore be a targeted override, not a replacement.** Keep the
shipped depth-bin path exactly as it is, and apply a motion group's own correction
only where that group demonstrably disagrees with the bin it sits in — which is the
kitchen bottle's case and is not the factory's. That is non-regressing by
construction, it needs no arbitration between two whole pipelines, and the
disagreement test is already measured (a bin fitted to +2.3 px containing a group
measured at +18.5 px is not a marginal call).

## F100 — Motion-group alignment finally corrects the bottle

Wiring together what F99 identified as already-solved-but-disconnected: material
edges (F92), motion-consensus grouping (F93), focal-proximity trust plus temporal
propagation (F89), and support built from a group's own features rather than by
segmenting the frame (F98's unexplored route).

`research/group_align.py`. On the kitchen sweep the grouping isolates the bottle at
92.9% purity in a 14-feature group, that group measures **+18.54 px at frame 11**
against ~+19.2 truth, and the correction lands:

| frame | global-only residual | motion-group residual |
|---|---:|---:|
| 8 | +9.09 px | **+1.68 px** |
| 9 | +12.53 px | **+1.10 px** |
| 10 | +16.21 px | **+1.67 px** |
| 11 | +19.97 px | **+1.47 px** |

About a 92% reduction, and visible: the ghost band and doubled strip along the
bottle's right edge are substantially gone.

Two bugs stood between "the motion is measured correctly" and "the correction is
applied", and both were in the painting rather than the estimation:

1. **Claim strength must be a GATE, not a scale factor.** Attenuating each group's
   correction by its guided-support value shaved 30% off it at the bottle's own label
   centre — a pixel the group owned outright with weight 1.00 — and 42% at its edge,
   because a guided-filtered seed stays below 1 even deep inside an object. A pixel a
   group clearly owns must get that group's motion in full.
2. **Support must be the convex hull of a group's features, not their
   neighbourhoods.** Features cluster on whatever part of an object carries texture —
   here the printed label — so circles around them left the bottle's top and bottom
   entirely unclaimed and uncorrected (applied dx +0.35 and +0.00). An object is
   connected; the hull of its features is a far better first guess at its body.

Sharpening the memberships was tried first and is NOT the fix: it moved frame 11 from
+14.8 to +14.3 across a 4x sharpness range, because the limiter was the claim gate
and the missing support, not the blending.

**Not integrated, and two things block it.** The method declines to engage where the
grouping yields fewer than two groups, which is correct on the zero-motion sentinel
(0.03 px introduced) but wrong on the analytic factory, where two planes exist and
the consensus still collapses to one group — the same collapse F96 diagnosed and the
focal-signature grouping (F97) is meant to fix, unwired here. And the group path does
not yet carry the disocclusion refusal the shipped path applies, so the fusion
comparisons run above are not like-for-like and the Q_SSIM figures from them should
be ignored (F81a applies doubly).

## F99 — Depth cannot drive the correction; motion grouping must. (What was missed)

Asked to re-read the arc and find what was missed that would fix the shipped output.
The answer is a wrong turn taken at F94 and never revisited.

F89 measured the kitchen bottle at **+18.88 px against +19.2 truth** by measuring its
interior edges at frames near the reference and PROPAGATING temporally. F93 then
identified which features belong to it (92.9–100% pure). Those two together are the
fix. Instead the arc went to F94–F97 building increasingly general scene models, and
F98 spent itself on pixel regions, and the shipped path was never touched.

Worse, the models were all keyed on DEPTH, and depth cannot do this job here. Fitting
displacement as a continuous function of depth — no bins, five knots, evaluated per
pixel — still fails, and the diagnostic is decisive:

| frame | curve at the bottle's depth | the bottle's own features |
|---|---:|---:|
| 0 | −5.01 | **−18.71** (n=18) |
| 5 | −1.77 | −2.13 (n=24) |
| 8 | +3.39 | **+7.80** (n=24) |
| 9 | +3.33 | **+9.87** (n=22) |
| 11 | +3.16 | +2.63 (n=14, blurred) |

The bottle's features give a clean monotone series that extrapolates to ~+19. The
depth-keyed curve never exceeds ±5 px at ANY frame, including frames where the bottle
is perfectly measurable. **Other content shares the bottle's depth VALUE while having
different motion**, so a depth-keyed fit averages it away — the same majority problem
as bins, which continuity does not cure because it was never a quantization problem.

Two further traps found on the way, both already in the playbook and both re-hit:
- Tiles cannot measure this at all. Phase correlation resolves about a quarter of the
  patch, so 40 px tiles saturate near 10 px, the bottle's 19 px is invisible, and the
  fitted curve reports that the scene barely moved.
- **Match confidence does not detect defocus bias.** A blurred profile correlates
  *confidently* against a sharp one at ≈0 shift, so degraded features report high
  confidence for "no motion" on exactly the objects that moved most. Any support or
  trust measure built on match confidence will be fooled; trust must be keyed on the
  feature's own focal distance instead.

Standing consequence: **key the correction on MOTION GROUPS, not on depth.** The
groups isolate the object (F93), their own features give an accurate motion series
where evidence exists, and F89's temporal propagation covers the frames where it does
not. Spatial support for a motion group is a much easier problem than general
segmentation, because the group is spatially compact and its own features bound it —
which is where F98's negative should have sent the work, and did not.

## F98 — Feature-level grouping works; turning it into pixel regions does not yet

F97's focal signature is an excellent per-FEATURE measurement, and F93/F96 group
features into objects well: the kitchen bottle comes out 92.9% pure as a feature
group, 97% self-consistent under the joint model. The obvious next step — use those
groups as the region masks the alignment fits — does not work yet.

| region construction | factory GT-SSIM | kitchen bottle IoU | bottle correction |
|---|---:|---:|---:|
| shipped valley depth bins | **0.970768** | ~14% | +2.3 px |
| sparse features, guided propagation | 0.904696 | 12.3% | +2.1 px |
| dense per-pixel focal peak, Otsu split | 0.969599 | 18.0% | +4.6 px |
| (truth) | — | — | +19.2 px |

Sparse-to-dense propagation is clearly wrong — a guided filter over sparse seeds
produced a region covering 72% of the kitchen frame, and cost the factory 0.066
GT-SSIM. Dense per-pixel focal peaks with subpixel interpolation are far better and
essentially tie the shipped bins on the factory, but still fail to isolate the
bottle: its best region reaches 18% IoU and the correction only +4.6 px.

**The gap is now precisely localized.** Grouping is solved at the level of features,
where the evidence lives — a feature has a focal curve, a normal displacement, and a
confidence. Region masks are a question about PIXELS, most of which have no such
evidence at all: the bottle's interior is blank white, so nothing there votes on
which object it belongs to, and whatever fills it in is doing so by spatial
propagation rather than measurement. Both constructions tried here fill it in badly.

This is the same shape as the project's older matte lesson: support must come from
CONTENT, not from a detector's convenient reach. A sparse set of confident features
plus a propagation rule is not the same object as the thing a human sees, and the
propagation rule is doing the real work while pretending to be plumbing.

Worth stating plainly: the alignment does not need pixel regions in order to use this
grouping. A per-feature motion model can be evaluated at any pixel through the
motion field it implies, without ever committing to a hard region boundary. That
route is unexplored and avoids the failure above entirely.

Also clustering-method notes, since each cost a run: single-linkage on an absolute
gap chains straight through a bimodal focal distribution whose tails meet (one group,
always); gap-relative-to-median has the same failure for the same reason; Otsu asks
the right question — is this better described as two clumps than one — and splits
correctly with no tuned distance.

## F96/F97 — Joint scene motion, and defocus as the orthogonal grouping channel

**F96 — objects, their depths and camera motion solved together.** F93 defines an
object as a maximal feature set admitting one rigid motion; F95 makes inverse depth
explicit and shared across frames, blaming its residual on four quantized bins and on
motion varying within a bin. Both complaints have one answer: let each OBJECT be its
own depth.

```text
d_i,k . n_i = omega_k (rot) + rho_o(i) [ (-tx_k + ux tz_k) nx + (-ty_k + uy tz_k) ny ]
```

Four camera parameters per frame, one inverse depth per object, plus the assignment —
each linear or trivial given the others, so it alternates, and the assignment step
becomes a physical test instead of a clustering heuristic. On the kitchen sweep it
converges from rms 2.66 to **1.49 px** (F95 managed 2.00), the target object is 97%
self-consistent, and its predicted shift at frame 8 is +7.61 px against ~+7.4 truth.

At frame 11 it predicts +11.06 against +19.2, because by then the bottle's own
features are blurred away and its motion is inferred from far content where rho gives
weak leverage. That is F89's rule again: where an object's own evidence is gone,
propagate its measured motion temporally rather than predicting it from the global
model.

**Two model errors found by the factory, once the factory could see them.** The
analytic factory yielded only 12–21 material edges because its synthetic texture had
no surface detail — almost every edge was a silhouette, which the material/limb test
correctly rejects. Adding printed-style surface texture brought it to 63–70 and
immediately exposed:

1. *Grouping must use the model that CANNOT explain a depth difference.* With a
   similarity model, one consensus swallowed the whole two-plane factory and reported
   an excellent 0.92 px residual for entirely the wrong structure — a radial term can
   imitate two spatially separated regions translating differently. Translation-only
   grouping is now mandatory (F93 had already found it better conditioned).
2. *Maximizing consensus SIZE rewards the sloppy compromise fit.* A model sitting
   midway between two planes counts both as inliers.

**F97 — no motion threshold serves two scenes, and defocus is the invariant that
replaces it.** Sweeping the inlier threshold shows the same trap as F85:

| inlier px | factory objects | near-plane purity | kitchen bottle purity |
|---|---:|---:|---:|
| 1.4 | 1 | 51.8% | 100% |
| 1.0 | 2 | 100% | 1.7% |

Each scene wants what the other cannot use. The physical invariant behind it is
DEFOCUS, which is orthogonal to motion entirely: features at one depth sharpen and
blur together across the sweep no matter what they are doing. Measuring each
material feature's own sharpness curve and taking its peak frame:

| | measured focal frame | truth | within-group spread |
|---|---:|---:|---:|
| factory near plane | 1.03 | frame 1 | 0.30 |
| factory far plane | 3.89 | frame 4 | 0.56 |
| kitchen bottle | 5.93 | ~frame 6 | 0.39 |
| kitchen elsewhere | 7.98 | many depths | 1.73 |

Focal frames are recovered to within 0.1 frame, groups separate by 2–2.9 frames
against a within-group spread of 0.3–0.6, and it works on both scenes with identical
settings. Depth grouping should therefore come from the focal signature, with motion
consistency confirming rigidity WITHIN a depth rather than being asked to discover
depth by itself.

Not yet tested: the same channel should also distinguish veiling from ordinary
defocus, since a veiled feature is attenuated one-sidedly by a foreground spreading
over it while a defocused one spreads symmetrically. That is the next orthogonal
signal and it is unbuilt.

## F95 — Depth belongs IN the decomposition, and is recoverable only up to an affine

Clarifying what F94's table was for: the components are not depth-independent. They
are characteristically DIFFERENT from one another, and depth-dependence is part of
what makes them different — it is not a nuisance to be set aside. The quadrant
sign test happens not to need depth, but that is one check inside the method, not
the method. Any decomposition must carry depth explicitly, and F94's four quantized
bins are a crude stand-in for a quantity continuous in 1/Z.

The physically correct form makes inverse depth a parameter, and exploits that it is
a property of the SCENE — one ρ per depth, shared by every frame, while motion is
per-frame:

```text
d_i,k . n_i = omega_k (rot basis) + rho_b [ (-tx_k + x tz_k) nx + (-ty_k + y tz_k) ny ]
```

Solved by alternating (motion given ρ, then ρ given motion), which is linear in each
half. On the kitchen sweep: 2419 observations, 226 material edges, 11 frames.

- **rms residual 2.00 px** against an observed displacement rms of 3.58 px — about
  69% of the motion explained, where F81b's tile-based version of the same idea
  managed 25–50%. The improvement is entirely in the features: material edges
  measured by gradient profile, not tiles.
- **ρ = +2.90, −0.21, −0.77, −0.12** across bins ordered near to far: one clearly
  separated near bin (the bottle's) and three far bins mutually indistinguishable.
  Sensible for this scene, and not monotone across all four.

**ρ is identifiable only up to an affine reparameterization**, and the reason is
F94's confound: a pan is indistinguishable from every depth translating equally, so
adding a constant to ρ can be absorbed by the motion. Dropping the uniform term (as
F94 requires for identifiability) leaves pan/tilt nowhere to go but into ρ·t, which
is what inflates one bin's ρ. Practical consequences:

- report and use only the depth-VARYING part of ρ; its offset and overall scale are
  gauge, not measurement;
- do not expect a monotone ρ to validate the fit — bins whose true depths are close
  will order arbitrarily within noise;
- this is the same wall as F81b's "monotone but not affine", now with its cause
  named rather than observed.

The residual 2 px is not yet explained. Candidates in order of suspicion: depth
quantization at four bins, genuine object-level motion differences within a bin
(which is what F93's grouping exists to capture), and measurement noise, which the
material-edge agreement puts at roughly 1 px.

## F94 — Reading camera motion off all edges at once, and what is not identifiable

Each component of camera motion leaves a different spatial signature, so with a
couple of hundred edges the problem is heavily over-determined and the components can
be measured rather than argued about:

| component | spatial pattern | depth |
|---|---|---|
| breathing | radial | independent |
| forward translation | radial | scaled by 1/Z |
| pan / rotation | near-uniform shift | independent |
| lateral translation | uniform direction | scaled by 1/Z |

The quadrant sign pattern separates radial from uniform with no depth at all (under a
radial component the left half moves left while the right half moves right); depth
then splits each pair, since only the translational components scale with inverse
depth. That hierarchy is exactly why this session burned three findings confusing
breathing with forward translation — both are radial, and depth is the ONLY thing
that separates them.

**Structural result: a pan is not identifiable.** A uniform shift is
indistinguishable from every depth translating equally, so the two are confounded in
image motion and only the depth-VARYING part of translation can be recovered. The
first version of this fit included both a uniform term and per-depth translations and
was therefore singular; it silently reported an applied +8 px shift as +4.00, split
evenly between the duplicate columns. The known-answer test caught it, the model was
restated (rotation shared; radial and translation per depth; no uniform term), and
the same test then returned +8.01. Breathing is now defined as the radial component
its depth bins SHARE, and forward translation as the part that varies between them.

Known-answer validation on a real frame: applied radial +2% reads +0.0197, applied
rotation +0.5° reads +0.491, applied uniform +8 px reads +8.01, and each stays within
0.03 of zero when its component is absent.

Kitchen sweep, 226 material edges, 4 depth bins:

| frame | breathing | radial spread (forward) | rotation | tx spread (lateral) | rms |
|---|---:|---:|---:|---:|---:|
| 0 | 0.9968 | 0.0433 | −0.574° | 10.63 px | 2.75 |
| 5 | 1.0012 | 0.0069 | −0.169° | 2.46 px | 0.52 |
| 10 | 1.0017 | 0.0201 | +0.495° | 7.17 px | 2.71 |

Breathing is ~1.000 across the whole sweep, confirming F91 that the global affine
removes it. Rotation is real and monotone through the sweep. Forward translation is
real but small — up to 4.3% radial spread — which partly vindicates F90's instinct
while leaving F91's correction of its magnitude intact. Lateral parallax dominates.

This is the instrument that should have existed before F87. It answers "what motion
is actually in this stack" in one pass, with a residual, instead of inferring it from
one object's width.

Limitation: the analytic factory yields only 12–21 material edges because its
synthetic texture mostly fails the limb/material test, so component recovery is
validated on known warps of a REAL frame rather than on the factory. The factory
needs richer surface texture before it can validate this instrument end to end.

## F93 — An object is a maximal feature set admitting one rigid motion

The anti-fragmentation model, stated so it can be tested rather than tuned. Every
admissible feature i sits at x_i with unit normal n_i and contributes exactly one
scalar per frame — its displacement along its own normal, all the aperture problem
permits. Features belong to one object iff a single motion per frame explains all of
them, in every frame:

```text
d_i,k . n_i  =  [ s_k (x_i - c) + t_k ] . n_i     for all i in the object, all k
```

An object is then a maximal consensus set under that model, found greedily from
spatially local seeds. Admissibility is physical, not heuristic: material edges only
(a curved object's limb is view-dependent, F92 — rejected here by testing whether
depth steps across the edge), and a feature counts in a frame only where its match is
confident, since detail blurs away off the focal plane (F89).

**It keeps the object whole.** On the kitchen sweep, 226 material features segment
into a 171-feature background, a 20-feature group, and a 21-feature group of which
**20 lie inside the bottle box** — with only 3 bottle-box features leaking into the
background. The bottle survives as one object, which is what every previous
grouping attempt failed to do.

**Motion parameterization is a separate problem from grouping, and it is where this
still loses accuracy.** Fitting the full similarity to that object produces a
monotonically rising scale (0.965 → 1.133) absorbing motion that is really
translation, and a non-monotone tx. Translation-only on the identical grouping is
clean and monotone:

| frame | 1 | 3 | 5 | 7 | 8 | 9 | 11 |
|---|---:|---:|---:|---:|---:|---:|---:|
| tx | −13.76 | −8.31 | −2.12 | +3.56 | +7.36 | +9.09 | +9.41 |
| rms | 1.90 | 1.33 | 0.57 | 0.72 | 1.90 | 3.84 | 8.65 |

Confirming F91/F92 from a third direction: this object translates and does not scale.
**Fit scale only where it is identifiable** — the radius term must change sign across
the object's features, and even then a compact off-centre object will trade scale
against translation unless the sign-changing features are numerous and well measured.

**Propagate from NEARBY frames, not all reliable ones.** Linear propagation from
every reliable frame (1–8) reaches +16.07 at frame 11 against a truth of +19.2, while
F89's propagation from only the adjacent frames reached +18.88. The drift is not
linear across a whole sweep, so distant frames bias the local slope. Combined with
F89's rule — measure near the object's focal plane — the estimator wants frames that
are near in BOTH senses: near the focal plane for evidence, near the target frame for
extrapolation.

Standing gap: grouping is solved, the remaining error is in the motion estimate, and
none of this is wired into the runtime yet.

## F92 — Silhouette edges of a curved object are not material features

A cylinder's left and right edges are LIMB edges: the silhouette lies where the
surface turns away from the camera, so it slides around the object as the viewpoint
moves and does not track a fixed material point. Its top and bottom rims do. That
predicts precisely the asymmetry that derailed F87–F91 — apparent width changing
under lateral camera motion while height does not — with no magnification involved.

Measured on the kitchen bottle, separating printed label edges (material) from the
silhouette (limb):

| frame | material edges (n=9) | spread | left limb | right limb |
|---|---:|---:|---:|---:|
| 8 | **+8.93** | 0.95 px | +4.52 | +8.28 |
| 9 | **+12.25** | 1.30 px | +3.23 | +11.95 |
| 10 | +14.40 | 14.46 | +1.50 | +15.69 |
| 11 | −16.59 | 41.27 | −0.07 | +19.66 |

At frames 8–9 nine material edges agree to about 1 px: the object is rigid and
purely translating, and the fit is clean. The LEFT limb is the outlier, reading half
the material value, while the right limb tracks material closely. Material
translation advances ≈+3.3 px per frame, extrapolating to ≈+19 px at frame 11
against a truth of +19.2. Frames 10–11 degrade because printed detail blurs away
(F89), not because the features are wrong.

**Rule: fit rigid motion on material (texture/printed) edges. Treat the silhouette
of a curved object as view-dependent and exclude it from a rigidity fit** — or the
constraint indicts the object instead of the feature. This also revises F87's
rigidity test, which assumed all edges are material: on a curved object a non-zero
differential can be legitimate limb motion rather than a bad measurement.

It also explains why the interior-edge test of F89 was the strongest instrument in
the arc: interior edges are exactly the material ones. The property that made it work
was never "more constraints", it was "the right constraints".

Standing caution for this project's data: kitchen objects are bottles, cans and
jars — curved almost everywhere. Silhouette-based motion estimation is systematically
biased on them, and flat-faced subjects (books, boxes, labels facing the camera) will
not reveal the problem.

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
