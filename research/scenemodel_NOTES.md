# Scene-model reconstruction — round B2 of the scene-model second pass

The first round of this arc that changes a pixel. `research/scene_model.py` is
new. **`research/forward_certify.py` was NOT modified** — see §11. No runtime
change, no test change, `src/**` and `tests/**` untouched (`git diff` against
B1's commit over those paths is empty).

```
.venv/bin/python research/scene_model.py kat        # the physical same-surface test
.venv/bin/python research/scene_model.py slope      # the blur ladder's known answer
.venv/bin/python research/scene_model.py factory    # bar A + the attributed remainder
.venv/bin/python research/scene_model.py kitchen    # bars B, C, D + both ledgers
.venv/bin/python research/scene_model.py orderguard # what the ordering-free rule costs
.venv/bin/python research/scene_model.py render     # out/inspect/kitchen_scenemodel.png
```

---

## 1. Verdict first

**The second pass works, it is non-regressing by construction, and the two
things it was built to prove are both measured: the `SURFACE_SIGMA` §12.3 split
is retired by physics, and the F112 knob is repaired. The round's largest
finding is a negative one — once you refuse to average across a focus
disagreement OR an unverified geometry, multi-frame aggregation buys
essentially nothing over selecting the single sharpest admissible observation.**

| bar | question | verdict |
|---|---|---|
| A | factory GT-SSIM ≥ 0.979453 | **PASS** — **0.981104** (+0.001651), and the certifier agrees (3.3051 → 2.9080) |
| B1 | knob repaired: box ≤ 1.5× the frame differential | **PASS** — 2.13× → **0.54×** |
| B2 | F108 flank box back to 0.00% > 12 | **MISS**, 0.57% → **0.40%**; the residual is 10 px at x662–669 y240–242, i.e. the knob's own top edge one to three rows above its recorded box |
| B3 | the four F112 user boxes must not regress | **2 of 4 regress on `|Δ|` vs the reference, and the cause is measured**: box 1 goes 6.14 → 7.48 while its FOCUS ENERGY goes 41.8 → **54.7**. Box 3 is flat (+0.02, identical max). §6. |
| B4 | the pale sliver, before/after, eyes-honest crop | **Resolved into sharp background structure** (`out/certify/kitchen_sliver.png`), with one ~1 px bright residue at the silhouette. §6c |
| C | every rewritten region certifier-better, else revert | **PASS by construction.** The veto fired on **11 of 21** kitchen regions and **0 of 3** factory regions; 2 kitchen regions were unarbitrable and were reverted too |
| D | byte-identity outside the rewrite | **PASS**, asserted in code. Kitchen rewrites 25.38% of the crop, factory 82.18% |
| E | `out/inspect/kitchen_scenemodel.png` registered to the existing layer | **PASS** — registration score **1.0000** |

---

## 2. What the reconstruction is

Input: the routed two-frame composite that ships today
(`twoframe_stack(normalize_exposure(src))`) plus B1's decomposition. Output: the
same composite with OWNED pixels rewritten from the frames, and nothing else
touched.

```
for each OWNED layer i, for each frame k:
  GEOMETRY    global affine (o) layer shift, composed, ONE resample; the menu is
              pass 1's own (affine / rigidified), chosen by pass 1's own edge
              evidence, gated by `twoframe.gate_shift`, trinary per F106/F110
  VISIBILITY  decline where ANY other layer's footprint, carried into frame k by
              its own transform and pulled back through layer i's, lands here —
              ordering-FREE, because F114 §9 says the ordering bit refuses
  ADMISSION   the PHYSICAL same-surface test (§3)
  AGGREGATE   the sharpest admitted observation, plus every other admitted
              observation with the SAME integer disk radius that also VERIFIED
              and mutually agrees; nothing else is averaged in (§4)
  else        the input composite's pixel stands — NO COMPLETION
then          certify per region; revert every region the certifier does not prefer
```

Nothing outside OWNED is written, so F101's non-regression is structural rather
than argued, and bar D is an `assert` in `assemble`, not a claim in prose.

---

## 3. `SURFACE_SIGMA`, retired — the verdict

F112/R4 logged the low-pass scale as an unresolved §12.3 split: the analytic
factory wants 2, the kitchen boxes want 8, 4.0 is the smallest value clearing
every bar. R5 named the physical invariant and did not build it. It is built.

**The design R5 named was to make the low-pass EXCEED the residual defocus
difference. The better move is to REMOVE it, exactly.** Two observations of one
latent surface are `L (x) disk(R_m)` and `L (x) disk(R_r)`, so blurring each by
the OTHER's disk makes them identical:

```
m (x) disk(R_r)  ==  L (x) disk(R_m) (x) disk(R_r)  ==  r (x) disk(R_m)
```

Cross-convolution. No PSF family mismatch, because the family used to match is
the family the physics uses (PLAYBOOK §0: real defocus is a DISK). Disk radii
are integers by construction, so the ladder is the exact parameter space.
`R(k, p) = c · |k − peak(p)|` — PLAYBOOK's validated blur proxy, per pixel, with
`c` measured per scene (§5).

### KAT, on the COMMITTED fixture and against the COMMITTED pass marks

`tests/test_twoframe_route.py::test_same_surface_is_blind_to_defocus_...`'s own
scene and bars, so the two versions are directly comparable.

| clause | bar | physical | global σ=4 (R3) |
|---|---|---|---|
| disk defocus r = 1 / 2 / 4 / 6 | > 0.97 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 0.999 / 0.983 |
| **disk defocus r = 8 / 12** | > 0.97 | **1.000 / 1.000** | **0.931 / 0.759** |
| shift 0.5 / 1.0 / 1.5 px | > 0.97 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| shift 2.0 / 4.0 px (should refuse) | — | 0.986 / **0.691** | 0.981 / 0.800 |
| gain ×1.015 / ×1.019 | > 0.97 | 1.000 / 1.000 | 1.000 / 1.000 |
| moved occluder 4 / 8 / 20 px, strip | < 0.05 | **0.010 / 0.003 / 0.002** | 0.000 |
| …elsewhere | > 0.95 | 1.000 / 1.000 / 1.000 | ~0.985 |

**Verdict: the split is retired.** The factory wanted σ small because a large
low-pass costs sharpness; the kitchen wanted σ large because a small one cannot
absorb a 12 px defocus difference. The cross-convolved test needs neither: it is
*exactly* blind to defocus at every radius (the row where σ=4 failed) *and*
refuses a moved occluder at 0.002–0.010 (the row σ=8 would have blurred away).
The two scenes' original bars are both cleared by one construction with no free
scale — and bar A confirms it end to end, since the factory's GT-SSIM went UP.

**One number survives, and it is not load-bearing.** `SIGMA0 = 1.0 px` is the
sampling-scale residual (the disk model, the bilinear resample and the pixel
grid are not distinguishable below a pixel). Swept at 0.5 / 1.0 / 2.0 the two
verdicts that matter do not move: defocus-12 reads 1.000 / 1.000 / 1.000 and
occluder-8 reads 0.000 / 0.003 / 0.000.

**The honest caveat.** `src/**` is read-only this round, so the physical test was
NOT put back through `twoframe_stack` and the runtime still ships `SURFACE_SIGMA
= 4.0`. The retirement is demonstrated on the instrument's own KAT and inside
the assembly. Porting it is a one-function change to `twoframe.same_surface`
plus a per-scene `c`; whoever does it must re-run F112's four boxes and the
factory, because the physical test is measurably STRICTER at 4 px of
misregistration (0.691 vs 0.800) and that will move refusal shares.

---

## 4. The round's main negative: aggregation buys nothing

The manager's architecture asked for sharpness-weighted aggregation with
mutual-consistency screening. It is built, and **measured, it is a wash.**

| | factory | kitchen |
|---|---:|---:|
| scene-model, certifier | 2.9080 | 8.8390 |
| region-scoped null (best SINGLE admissible frame) | 2.9048 | 8.7983 |
| differential | **+0.0032** | **+0.0407** |
| per-region: aggregation beats its own best single frame | 1 of 3 | 12 of 19 |
| mean members per rewritten pixel | 1.03 | 1.30 |

Aggregation wins the majority of individual regions on the kitchen and loses on
total, which is what "a wash" looks like. The reconstruction is therefore, in
practice, a per-pixel **selection** of the sharpest verified admissible
observation. That is the correct shape for an all-in-focus composite and it is
worth stating plainly for B3: the multi-frame averaging that a "scene model"
suggests is not where the remaining quality is.

### How it got there — an F106 violation the eyes caught and no aggregate did

The first build averaged every admitted frame within 1.0 px of the sharpest
modelled radius. It scored **better** on the certifier (8.7229) and was
**visibly softer**: the Lubriderm label went illegible on
disagreement-guided crops (PLAYBOOK §I.2). Two compounding causes:

1. **The equal-blur set was defined by a tolerance on a model whose residual is
   larger than the tolerance.** On the kitchen `c = 0.684 px/frame` with a
   residual rms of **2.42 px** — the layer-as-ramp-quantization problem F114
   measured, arriving here. A 1.0 px tolerance on a model that wrong admitted
   ~3 frames.
2. **It averaged across UNVERIFIED geometries.** 41 of 72 kitchen (frame, layer)
   fits return UNVERIFIABLE and keep the global affine. Blending two
   observations placed by two unverified geometries is exactly F106 — a
   photometric blend standing in for a geometric decision — and the Lubriderm
   moves ~3 px/frame.

Fixed by removing the tolerance (disk radii are integers, so "same blur" is
equality) and requiring both members to have VERIFIED. An unverifiable
observation is still used, per F110's trinary; it is used **alone**, as a
one-hot decision.

Mean focus energy, each build measured on its OWN rewritten set (the two builds
rewrite different pixels, so the denominators differ and the ratio is the number
to read):

| build | rewritten | input there | scene-model there | ratio |
|---|---:|---:|---:|---:|
| first (tolerance + unverified members) | 33.7% of the crop | 36.71 | 35.59 | **0.969×** |
| as shipped (equality + verified only) | 25.4% of the crop | 28.10 | 31.84 | **1.133×** |

Whole-frame focus energy: input 48.084, first build 47.967, shipped **49.372**
(the reference frame reads 42.685).

**The instrument note that goes with it (§12.1/§12.8).** The certifier preferred
the softer composite, so the natural suspicion was a sharpness bias — KAT-3
tested for one on the factory and could not test the kitchen, which has no GT.
So it was tested directly: the same composite, deliberately Gaussian-blurred,
scored **9.5242 → 9.6815 → 10.1176 → 11.1045** at σ = 0 / 0.5 / 1.0 / 2.0. The
certifier is monotonically correct on sharpness on the kitchen; it simply cannot
see a defect that is 3% of focus energy on a quarter of the frame. *An aggregate
was not wrong, it was blind, and an image killed the story* — the third time in
this project's record (§12.8).

---

## 5. The blur ladder's scale, and its known answer

`c` is not assumed. It is regressed through the origin from the certifier's own
forward radius search — the instrument KAT-2 measured recovering a known radius
100% exactly, including from a real imperfect composite:

| scene | measured c | truth | residual rms |
|---|---:|---:|---:|
| factory | **1.161** px/frame | `BLUR_PER_STEP = 1.15` | 0.29 px |
| kitchen | 0.684 px/frame | — | **2.42 px** |

The factory recovers its own constant to **1.0%**. The kitchen's residual is
3.5× its own slope, which is not noise — it is F114's finding arriving in a new
place: a depth LAYER is a quantization of a depth RAMP, so one radius per
(frame, layer) is a poor forward model on a receding countertop. Everything the
kitchen assembly does downstream of `c` inherits that crudeness, and it is the
single clearest lever B3 has.

---

## 6. The kitchen's canonical instruments

Every instrument was reproduced against its recorded value BEFORE being used on
anything new (§12.1), and one of those reproductions corrected the record:

* the four F112/R3 user boxes reproduce R3's maxima exactly (61 / 17 / 101 / 127);
  their coordinates are not in the repo and were recovered from the headers
  burnt into `out/inspect/ROUND3_flaw{1..4}.png` — they are now constants in
  `scene_model.USER_BOXES`, so the next round does not have to do that;
* **the canonical F108 flank instrument is the plain BOX x560–670 y240–420 vs
  the normalized reference, not the brightness/variance flank MASK.** The box
  reproduces F112's note exactly (mean 1.114, max 45, 0.57% > 12, and 0.00% with
  the knob box removed). The flank MASK (`grey > 170 & std < 4`, from
  `twoframe_probe.py`) reads 0.76 / 9 / 0.00% and **cannot see the knob at all**
  — the knob is dark, so the mask excludes 0 of its 781 px. Two different
  instruments have been sharing one name in the record.

### (a) the knob — repaired

| | input routed | scene-model |
|---|---:|---:|
| certifier differential, box mean | 3.524 | **0.477** |
| …frame mean | 1.653 | 0.887 |
| **ratio (bar ≤ 1.50×)** | 2.13× | **0.54×** |
| `|Δ|` vs reference, box / frame ratio | 1.05× | 0.87× |

F112 diagnosed the knob as a one-hot member quirk — a single darker/crisper
member rendering where the old soft blend leaned on the reference.
Multi-observation admission erases exactly that class of defect, and it did:
the knob is now *quieter than the frame average* on the certifier's own scale.
68.6% of the box is OWNED and 67.9% was rewritten; the remaining 31.4% is B1's
boundary band, which is why the repair is not total.

### (b) the flank box — 0.57% → 0.40%, and the residual is still the knob

The 10 surviving > 12 pixels outside the knob box sit at **x662–669, y240–242**
— inside the knob's x range, one to three rows above the recorded box's top
edge. Round A already recorded that "F112 describes one 30×70 px knob at
(659–669, 243–313)" is a 10×70 box for a 30×70 object. So the bar reads MISS
literally and the tail is the same object, smaller. Max rises 45 → 53 on those
rows: the rewrite reaches the OWNED/BOUNDARY frontier there and steps against
the un-rewritten band (§7).

### (c) the four user boxes, and the counter-instrument

| | box 1 | box 2 | box 3 | box 4 |
|---|---:|---:|---:|---:|
| input, mean/max `|Δ|` | 6.14 / 61 | 3.49 / 17 | 3.31 / 101 | 6.23 / 127 |
| scene-model | **7.48 / 107** | 2.63 / 63 | 3.33 / 101 | 5.00 / 116 |
| input, mean focus energy | 41.8 | 23.3 | 85.7 | 31.8 |
| scene-model | **54.7** | 23.1 | 86.0 | 35.3 |

Box 1 is the only real movement and it moves both ways at once: 22% further
from the reference and **31% sharper**. F112/R6.4 already rejected this metric
as an arbiter for precisely this reason — it is scored against the reference
frame, so refusing everything scores perfectly and any legitimate sharpening of
a locally-defocused background scores worse. The certifier is the arbiter that
cannot be gamed that way, and the region that owns box 1 — layer 5, 79019 px,
covering 436 of the box's 693 pixels, 62.9% of the box rewritten — reads
**−1.95 levels**, the strongest improvement of any region on the scene.

Box 3 moved +0.02 with an identical maximum: flat, not a regression.

**The sliver** (`out/certify/kitchen_sliver.png`, 6×, input | scene-model |
reference). The pale strip at the bottle's left silhouette is gone — replaced by
resolved background structure (a pot rail and its shadow) that the input carried
as a defocused smear and the reference does not resolve at all. One ~1 px bright
residue survives a few rows lower, at the silhouette itself. Reported, not
chased.

---

## 7. What the rewrite costs: a seam, and it is not a gain mismatch

The rewrite frontier is visible at 4× on disagreement-guided crops. Decomposed:

| at rewritten pixels adjacent to the frontier (29690 px) | mean | p95 |
|---|---:|---:|
| `|new − input|` | 6.30 | 22.0 |
| its LOW-frequency part (σ = 4) | 2.28 | 7.3 |
| its HIGH-frequency part | **5.30** | **18.7** |

Per-region DC offsets between the assembly and the input are ≤ 4 levels. **The
seam is a sharpness step, not a level step**, so the obvious fix — a per-region
scalar or DC match, the instrument F113 says already solves exposure — would buy
at most a third of it. Feathering inside the rewrite would preserve bar D (only
OWNED pixels would move) but would trade a visible edge for a visible blur band
at a depth boundary, which is the one place this project has spent five rounds
learning not to soften. Left as a measured cost. 26% of the pixels that newly
exceed 12 levels from the reference are within 3 px of the frontier, 53% within
10 px — so the seam is a real contributor and not the majority.

---

## 8. Ordering is non-load-bearing, measured

F114 §9 forbids assembling content that some frames occlude on the strength of
the current ordering, since F83's contour bit refuses (`near_is_low=None` on
BOTH scenes). So the visibility test does not use the ordering at all: any other
layer's footprint is a possible occluder. `orderguard` prices that choice
against the ordered variant.

| scene | guard | no geom | off-frame | **occluded** | diff surface | admitted |
|---|---|---:|---:|---:|---:|---:|
| factory | any | 0.00% | 0.00% | **0.025%** | 4.08% | 95.89% |
| factory | nearer | 0.00% | 0.00% | 0.016% | 4.09% | 95.90% |
| kitchen | any | 6.20% | 0.00% | **0.160%** | 13.82% | 79.82% |
| kitchen | nearer | 6.20% | 0.00% | 0.092% | 13.85% | 79.86% |

(share of owned pixel-frames; "no geom" is the gate's CONTRADICTED verdict)

The two guards produce **identical composites**. The reason is structural and is
decompose_NOTES §9's own prediction, now measured: B1's boundary band already
declines a 5 px ribbon at every layer boundary, and on both scenes the
differential motion between adjacent layers is smaller than that ribbon, so the
visibility test has almost nothing left to refuse. **Ordering is non-load-bearing
for a reconstruction that does not complete occluded content, and it becomes
load-bearing the moment one does.** The ordering-free rule is kept because it
costs nothing and removes an assumption.

The refusal that IS load-bearing is the physical same-surface test: 4.1% of
owned pixel-frames on the factory, 13.8% on the kitchen.

---

## 9. The factory's remainder, attributed

Same ladder discipline as round A's `floor`: each rung replaces exactly ONE
estimated quantity with the known answer.

| rung | GT-SSIM | remainder |
|---|---:|---:|
| as measured | 0.981104 | 0.018896 |
| + TRUE per-layer geometry | 0.983095 | 0.016905 |
| + TRUE masks as well | **0.987824** | **0.012176** |
| (the null: the reference frame) | 0.965880 | 0.034120 |

```
  motion estimation                        +0.001991
  layer segmentation                       +0.004729
  assembly + render + the crop's content    0.012176   <-- the largest term
    of which ONE round-trip resample        0.010513   <-- 86% of it
  = the remainder to 1.0                    0.018896
```

**86% of what survives true masks and true geometry is the price of moving a
frame.** Warping the ground truth by a true matrix and back — nothing else
changed, no model error of any kind — reads GT-SSIM 0.989487. Every
non-reference observation an assembly uses pays that once (PLAYBOOK §0: compose,
resample once — this IS once). The remaining ~0.0017 is the render and the crop.

Read against the certifier's own floor ladder this is a different ranking, and
both are right about their own question: the certifier's floor said MOTION was
the largest model term, because the certifier scores a forward render where
misplacement is loud; GT-SSIM on the composite says RESAMPLING is the largest,
because SSIM is a structure metric and interpolation is a low-pass.

**Motion refinement was out of scope and did not need to be in it**: on the
factory it is worth +0.001991, an eighth of the remainder.

### The oracle's own trap, caught by the KAT

The first build of this ladder substituted the true per-layer SHIFT into the
slot B1's *residual* occupies — where it gets composed onto the global affine.
The true shift is the total reference-to-frame displacement, so composing it
double-counted the affine and the oracle scored **0.9414 against the estimate's
0.9811**. An oracle that loses to the thing it bounds is a broken instrument,
which is exactly how F110 found the last version of this ladder wrong. The rung
now replaces the whole matrix.

---

## 10. The region-scoped null (F114 §7)

F114 asked for one because the global null carpet hides region-scale defects.
Built as the same assembly with aggregation removed: per owned region, exactly
one admissible observation, the sharpest.

| | factory | kitchen |
|---|---:|---:|
| global null (the reference frame) | 2.1917 | 7.8398 |
| region-scoped null (best single frame) | 2.9048 | 8.7983 |
| scene-model | 2.9080 | 8.8390 |
| input routed composite | 3.3051 | 9.5242 |
| scene-model − global null | +0.7163 | +0.9993 |
| scene-model − scoped null | +0.0032 | +0.0407 |

The scoped null does what F114 wanted: it is 0.7–1.0 levels *above* the global
null, i.e. it removes most of the carpet that the defocused reference frame's
easy agreement was providing, and it puts the comparison on content that is
actually all-in-focus. On the knob it is what makes the repair legible (§6a).
Its own honest limit is §4: it is nearly the same image as the scene-model, so
it bounds aggregation, not assembly.

---

## 11. NEGATIVE deliverables — tried, measured, rejected

1. **Gaussian second-moment blur matching** (`sigma = sqrt(R2²−R1²)/2`).
   REJECTED for cross-convolution: it leaves a PSF-family residual that grows
   with radius (defocus agreement 0.867 at r=12 against 1.000). KAT-2 rung C
   already said a family mismatch shows up in the residual and never in the
   parameter; this is the same statement one level down.
2. **Replacing F112's `tol·|∇|` linearization with the exact statement** —
   minimize the disagreement over a shift grid at ±GATE_TOL, charge only the
   sub-grid remainder to the gradient. BUILT and MEASURED WORSE on every row of
   the committed fixture: it buys a moved occluder a GATE_TOL-wide sliver at
   each strip edge (0.263 admitted at a 4 px move, against 0.010 linearized) and
   simultaneously admits more of a 4 px misregistration (0.885 vs 0.691). The
   linearization is the tighter bound here, not merely the cheaper one.
3. **A blur tolerance on the equal-blur set** (1.0 px = the sampling scale).
   REJECTED, §4: the kitchen's radius model has a 2.42 px residual, so a 1.0 px
   tolerance is a tolerance on a model that is wronger than the tolerance. Disk
   radii are integers; equality needs no constant.
4. **Averaging observations whose geometry only DECLINED verification.**
   REJECTED on F106 and on measurement (§4): focus energy on rewritten pixels
   0.969× the input, and the Lubriderm label visibly illegible.
5. **A per-region DC/scalar match to hide the frontier seam.** NOT DONE, §7:
   measured, the seam is 5.30 high-frequency against 2.28 low-frequency, so a
   gain match addresses at most a third of it and would be a cosmetic fit to a
   real sharpness step.
6. **Feathering the rewrite frontier.** NOT DONE. It would preserve bar D, and
   it would soften a depth boundary. Named for B3 with the measurement attached.
7. **Modifying `forward_certify.py`.** The file scope ALLOWED a region-scoped-null
   hook and **none was needed**: a scoped null is a CANDIDATE composite, and
   `certify` already takes a candidate and a region, and `Certification.unexplained`
   is already per-pixel. The allowance was not used, so default-path identity is
   not a claim to verify — the file is byte-identical to B1's commit, as are
   `src/**` and `tests/**`.
8. **Tuning anything to the four user boxes.** NOT DONE, for F112/R6.4's reason.
   The counter-instrument (focus energy) is reported beside them so the metric's
   direction is visible, not so it can be optimized.
9. **Running large-motion.** NOT DONE — out of scope by instruction. §12 states
   the transfer as a hypothesis with its mechanism, unmeasured.
10. **Sweeping the kitchen's layer count** — still not done, still the obvious
    next experiment (F114 §6b), and §5 now gives it a sharper motivation than
    F114 had: the radius model's residual is 3.5× its own slope.

---

## 12. The pair-aware-refusal transfer to large-motion — hypothesis, not measured

F111 left this open: on large-motion the box is sharpest at frame 0 and needs
+18.9 px; the pair fits it correctly (0.51 px verified), then F82's disocclusion
refusal withdraws that member over **91% of the pair** and the box comes back
reference-defocused. The named cure was "a pair-aware refusal that prefers a
present-but-defocused member over a reference-defocused one".

This architecture already has both halves of that, and they are worth stating
precisely because the mechanism — not the outcome — is what transfers:

* **It refuses OBSERVATIONS, not FRAMES, and only where the geometry says
  another surface actually stands there.** F82's refusal is keyed on the pair's
  differential layer motion applied as a width through `_occlusion_mask`, so at
  19 px of swing it covers most of the frame. The visibility test here asks the
  pixel-wise question — is this pixel covered by another layer's *warped
  footprint*? — and at 19 px of swing that is a 19 px band along each boundary,
  not 91% of anything. Measured on the two scenes available it refuses 0.03% and
  0.16% of owned pixel-frames (§8).
* **A present-but-defocused observation already outranks no observation.**
  `best_frame` is an argmin over the ADMITTED set only, so when the sharpest
  frame is refused the next-sharpest admitted one supplies the pixel, defocused
  and correctly placed. Only when nothing at all is admitted does the input
  composite stand. That is F111's preference order, implemented.

**What is NOT established.** Large-motion's input composite is the shipped
override output, because the two-frame route declines there (19.2 px > the 14.0
px licence) — so a B2 pass on that scene would be refining a different input
than the two scenes here, with a decomposition nobody has looked at, and B1's
propagated per-layer motion would have to carry ~19 px rather than ~3. The
licence question is about the two-frame ROUTE and does not obviously bind a
refinement pass that never re-registers the whole frame. That is a real
opportunity and it is a measurement, not an argument. B3 should run it.

---

## 13. Honest limits

1. **Two scenes**, one analytic. Every number above is conditional on that, and
   §5 is the reason it bites: the kitchen's blur model has a residual 3.5× its
   own slope and the factory's has 0.25×, so the two scenes are not testing the
   same instrument quality.
2. **The kitchen rewrites 25.4% of the crop after the veto**, from 60.4% before
   it. More than half of what the assembly proposed was reverted, and 2 of the
   21 regions had too little certified evidence for the certifier to arbitrate
   at all and were reverted on F106's rule. The second pass is currently a
   small, careful edit, not a reconstruction.
3. **The veto's granularity is coarse.** Region 1 (48013 px) was reverted whole
   on a +0.091 level mean. A finer partition would keep its good parts, and
   would have less certified evidence per region to decide with. Not swept.
4. **`SIGMA0` and the veto's `MIN_ARBITRABLE` are the only two free numbers
   left**, both borrowed, and only the first was swept.
5. **The frontier seam** (§7) is a visible artifact at 4× that no bar in this
   round measures.
6. **The certifier's real-scene sensitivity is unchanged** (~10 levels absolute,
   a few differential) and the scene-model's improvement on the kitchen
   (−0.7608) is comfortably above it, but individual region verdicts near ±0.05
   levels are not.
7. **The physical same-surface test is an APPEARANCE test** and cannot separate
   two surfaces that look alike — measured, not inferred: with the occluder's
   replacement drawn from the scene's own texture instead of a flat square, the
   vacated strip reads 0.759 / 0.681 / 0.613 admitted at 4 / 8 / 20 px. This is
   a property of the question. It is why visibility is checked geometrically
   before admission, and it is why §8's finding that visibility refuses almost
   nothing is a caution rather than a reassurance.

---

## 14. What round B3 / C should know

**Take from here:**

* `same_surface_physical` and `match_blur`. The knob is gone from the
  same-surface test; port it into `twoframe.same_surface` and re-run F112's
  boxes (§3's caveat).
* `blur_slope` — a per-scene circle-of-confusion constant with a 1.0% known
  answer, and its residual is a free diagnostic of how badly the layers
  quantize the scene's depth ramp.
* The canonical instruments, now constants: `USER_BOXES`, `KNOB`, `FLANK_BOX`,
  and the correction in §6 about which "F108 flank" is which.
* The certifier has NO sharpness bias on the kitchen (§4). That rung did not
  exist before and it licenses using the certifier on sharpness-changing edits.

**Do not take from here:**

* **The idea that a scene model wins by averaging frames.** §4. Once averaging
  across a focus disagreement or an unverified geometry is forbidden — and both
  must be — what is left is selection, and selection is what pass 1 already
  does. The second pass's win came from *better admission*, not from more data.
* **Ordering, still.** §8 says it is non-load-bearing *because nothing here
  completes occluded content*. A completion pass makes it load-bearing
  immediately, and F83's bit still refuses on both scenes.

**Three threads with a mechanism attached:**

* **The resample is now the factory's largest remainder term** (0.0105 of
  0.0189, §9). An assembly that reads a frame at sub-pixel positions cannot
  avoid one interpolation — but it could avoid *needing* one, by preferring the
  observation whose required displacement is nearest an integer, or by a single
  higher-order kernel. Nothing in the record has measured either.
* **The kitchen's layer count / ramp quantization** (§5, F114 §6b). It now has
  two independent symptoms: `RADIUS_MAX` saturation, and a radius-model residual
  3.5× the slope. F95's inverse-depth parameterization is the natural home.
* **Large-motion** (§12). The refusal mechanism that F111 named is already
  built here; nobody has pointed it at the scene it was named for.
