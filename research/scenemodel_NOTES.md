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

**One number the charter should see.** FRONTIER's amendment of 2026-08-02
designates "a temporally-coherent per-layer motion series refined against render
residual" as round B3, on F114's finding that motion is the largest remaining
term in the CERTIFIER's model floor (+0.812 of 1.367). That finding stands and
this round did not touch it — but on the composite's own GT-SSIM the same
factory ladder prices motion at **+0.0020 of a 0.0189 remainder**, an eighth,
against **0.0105 for one resample** (§9). Both are true of their own question: a
forward render punishes misplacement loudly, and SSIM punishes interpolation.
B3 should decide which question it is answering before it starts, because the
two rankings point at different builds. The amendment's principle — pass 1 as
prior, the certifier as likelihood, revision only where it beats the prior on
held-out physics — is exactly what §4's never-degrade veto implements at the
appearance level, and it is directly reusable for a motion series.

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

---

# CORRECTION ROUND (B2/R2) — the veto was blind at every scale below a region

The manager's eyes found three defect classes inside B2's own composite, in the
user's own acceptance boxes, that no aggregate in §1's table could see. This
section is the diagnosis, the fix, and its measured price. `research/scene_model.py`
is the only file changed; `src/**`, `tests/**`, `forward_certify.py` and
`layer_decompose.py` are untouched. 94 tests pass.

```
.venv/bin/python research/scene_model.py localkat   # NEW: the local arbiter's KAT
.venv/bin/python research/scene_model.py kitchen    # writes out/certify/B2R2_defect{1,2,3}.png
```

## 15. The overlay verdict, before anything was built

B2's numbers reproduce exactly before diagnosis (§12.1): certifier 9.5242 →
8.7700, 25.38% rewritten, boxes 61/17/101/127 → 107/63/101/116, knob 2.13× →
0.54×. The manager's coordinates are in inspection-layer space; the inspection
layer is the composite crop shifted **one pixel in x** (verified by exact array
match, not by re-registration), so composite = inspection + 1 in x.

Five distinct clusters, found by thresholding "new structure" — content that
differs from BOTH the routed input and the reference frame — and overlaid on the
rewrite map and its frontier:

| cluster | where (composite coords) | px | component | local certifier Δ | median depth from the frontier |
|---|---|---:|---|---:|---:|
| box 2, wall streak | x588–592 y57–64 | 17 | region 0 (79019 px, KEPT) | **+3.87** | **1.4** |
| box 2, pump spur | x545–567 y62–90 | 24 | **171 px component** | no coverage | **1.0** |
| box 1, bottle line | x473–507 y135–156 | 111 | region 0 | **+0.98** | 8.2 |
| box 4, shelf streak | x433–483 y196–214 | 147 | region 0 | **+3.15** | 8.0 |
| box 4, bright dashes | x467–481 y248–266 | 18 | **31 px component** | no coverage | **1.0** |

(baseline: over ALL rewritten pixels the median depth from the frontier is 12.6,
and 6.7% lie within 1.5 px of it)

**The hypothesis is half right, and the half that is wrong is the important
half.** Three of the five clusters are frontier phenomena — but not the measured
"sharpness step at the rewrite frontier" of §7. They are *rewrite islands*: two
of them live in connected components of 171 and 31 pixels, and one sits on the
1–2 px edge of a large one. The other two — the box 1 pale line and the box 4
shelf streaks, which are also the two largest — sit **~8 px inside a solid
rewrite**, at or below the frontier-adjacency of the rewrite as a whole. Frontier
depth does not explain them. Globally the frontier is only a weak predictor:
17.7% of new-structure pixels lie within 1.5 px of the frontier against 12.0% of
all rewritten pixels.

**The LIMB half of the hypothesis is right, and the picture shows why.** In
`out/certify/B2R2_defect2.png` the rewrite reaches into the Lubriderm bottle as a
wedge that crosses its left silhouette, and the pale line is at the wedge's tip.
F92 exactly: the silhouette of a curved object is a limb, it slides with
viewpoint, and the cross-convolved same-surface test cannot refuse it because a
low-passed white bottle edge and a low-passed pale background agree.

### 15a. The bug the overlay actually found: a size filter in front of a veto

`regions_of` dropped every rewrite component below `TF.MIN_LAYER_PIXELS`. That
reads like a sensible ledger tidy-up and is not one, because `apply_veto` only
ever REMOVES pixels: **a component that never reaches the ledger is not skipped,
it is admitted** — written with no verdict of any kind. On the kitchen that was
30 components and 1304 px, 1.41% of the final rewrite, with ZERO certifier
coverage between them, and it held two of the five defect clusters.

The rule that should have governed them already existed and already says the
right thing: fewer than `MIN_ARBITRABLE` certified pixels means UNARBITRATED,
and F106 reverts an unarbitrated change rather than waving it through. Nothing
needed inventing. The filter needed deleting.

## 16. Known-answer test first: what can a LOCAL certifier verdict see?

`local_veto` asks the certifier a question it has only ever been asked about
whole regions, so it is a new instrument and gets a KAT before it is believed
(§12.1, `scene_model.py localkat`). Three +40-level squares of known side are
pasted into the input composite at well-certified sites; a perfect arbiter would
put all the extra unexplained residual inside them, and a forward renderer
cannot, because it convolves the composite with each layer's defocus disk before
comparing.

| side | frame score | positive mass inside the square | within 10 px | pooled peak K=7 | K=15 | K=25 |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | +0.0028 | **23.7%** | 31.0% | 14.76 | 3.41 | 1.22 |
| 9 | +0.0191 | **65.5%** | 77.4% | 32.73 | 11.30 | 3.98 |
| 15 | +0.0518 | 57.7% | 62.8% | 38.80 | 29.63 | 11.58 |
| 25 | +0.2935 | 70.7% | 35.5% | 39.29 | 36.69 | 32.47 |

(the "within 10 px" column is a fixed radius, so it falls again at side 25
simply because a 25×25 square is wider than the disc that measures it)

**A 9×9 defect is arbitrable; a 5×5 one is not, at any pooling scale.** Two
thirds of a 9×9's residual lands on itself and the frame moves 0.019 levels; a
quarter of a 5×5's does and the frame moves 0.003, under the certifier's own
real-scene differential sensitivity (F113). This is the number that decided the
design: a certifier-only local clause has a floor of a few dozen pixels, and
three of the five defect clusters are under it. It is also KAT-4's finding
arriving in a new place — the knob was "~7× under the real-scene detection
floor" for the same reason.

## 17. The local veto: one rule, three scales, no new free number

> *A rewrite must be arbitrable at the scale at which it is written.*

1. **per COMPONENT.** `regions_of` no longer filters by area, so every component
   gets a verdict and the unarbitrable ones revert. (§15a — a hole, not a
   threshold.)
2. **per CLUSTER** (`local_veto`). The certifier differential pooled over
   `LOCAL_WINDOW = ceil(sqrt(MIN_ARBITRABLE)) = 15` px — the quorum the region
   rule already demands, made square — thresholded at the region rule's own
   `> 0.0`. Where the window does not hold the quorum the clause **abstains** and
   the region verdict stands (kitchen: 32011 px), because issuing a verdict on
   less evidence than the module already demands would be inventing sensitivity
   §16 says the certifier does not have.
3. **per FRONTIER** (`quiet_frontier`). A frontier is where the composite
   switches SOURCE, and the switch is invisible exactly where the two sources
   agree. F112's `agreement_budget` is already a per-pixel statement of an
   explainable difference (a GATE_TOL displacement against the local gradient,
   the exposure residual, sensor noise); applied to the switch instead of to the
   admission it says `|rewrite − input| ≤ budget` at every frontier pixel. Where
   it is not, that pixel and everything within `FRONTIER_SLACK = ceil(GATE_TOL)
   = 2` px of it reverts.

The budget's own physics puts the cost in the right place with no help: on a
smooth surface the gradient term vanishes and the budget collapses to a few
levels, so a frontier crossing flat wall is cut back hard — which is exactly
where a seam is visible — while along a real edge the budget is tens of levels
and the frontier is left alone. **This is not feathering** (F79; negative
deliverable 6 of the first round): nothing is blended, nothing is softened, no
depth boundary is crossed. The rewrite withdraws, hard, to where its own edge is
quiet.

The three constants are `MIN_ARBITRABLE` made square, the region rule's own
zero, and `GATE_TOL` rounded up. The correction introduces no new tuned number.

### The structural guarantee that replaces "trust me"

The corrected rewrite is a strict subset of B2's, and pixel-identical where both
rewrite — **verified, 0 px changed that B2 did not change.** So every pixel of
the corrected composite is either the shipped routed input or a pixel B2 already
had, and the correction cannot have introduced a new artifact class. That is
worth more than any amount of visual inspection of the parts nobody flagged.

## 18. The ladder — one clause per rung (kitchen)

| rung | rewritten | certifier | knob ratio | box 1 | box 2 | box 3 | box 4 | box1 focus | d1a/d1b/d2/d3a/d3b |
|---|---:|---:|---:|---|---|---|---|---:|---|
| input routed | — | 9.5242 | 2.13× | 6.14/**61** | 3.49/**17** | 3.31/**101** | 6.23/**127** | 41.8 | 0/0/0/0/0 |
| V0 = B2, region veto only | 25.4% | 8.7700 | 0.54× | 7.48/107 | 2.63/63 | 3.33/101 | 5.00/116 | 54.7 | 17/24/114/147/18 |
| V1 + every component judged | 25.0% | 8.7599 | 0.54× | 7.48/107 | 2.64/63 | 3.31/101 | 4.91/116 | 54.7 | 17/**0**/114/147/**0** |
| V2 + cluster clause | 20.6% | 8.6916 | 0.59× | 7.26/107 | 2.64/63 | 3.31/101 | 4.91/116 | 50.5 | 17/0/**48**/136/0 |
| **V3 + frontier clause (shipped)** | **19.0%** | **8.6971** | **0.95×** | 6.19/98 | 2.65/**16** | 3.31/**101** | 5.02/**116** | 44.7 | **0**/0/19/93/0 |

Clause 1 is nearly free (−0.4% of the rewrite) and kills two of five clusters
outright. Clause 2 is where the certifier improvement comes from (−0.068
levels). Clause 3 is where the box maxima and the last frontier cluster come
from, and it is the clause that costs.

### What the frontier slack costs, priced rather than chosen

| slack (one application) | rewritten | certifier | knob | box 2 max | d1a |
|---:|---:|---:|---:|---:|---:|
| none | 20.6% | 8.6916 | 0.59× | 63 | 17 |
| 1 px | 19.9% | 8.6866 | 0.64× | 18 | 6 |
| **2 px = ceil(GATE_TOL)** | **19.0%** | **8.6971** | **0.95×** | **16** | **0** |
| 3 px | 18.1% | 8.7098 | 1.35× | 16 | 0 |
| 5 px = the boundary band | 16.1% | 8.7551 | **2.42× FAIL** | 16 | 0 |

and iterating the clause to a fixed point is where it turns destructive: at 2 px
slack, two rounds take the knob to 2.32× and three to 3.17×. **One application,
deliberately** — each further round is a fresh revert with no new evidence behind
it. The residual loud frontier is reported instead (653 px of a 9489 px
frontier, 6.9%).

## 19. Re-acceptance bars, measured

| bar | routed | B2 | corrected | verdict |
|---|---:|---:|---:|---|
| factory GT-SSIM ≥ 0.981104 | 0.979453 | 0.981104 | **0.982061** | **PASS**, +0.000957 over B2 |
| factory certifier | 3.3051 | 2.9080 | **2.6725** | improved |
| kitchen certifier | 9.5242 | 8.7700 | **8.6971** | **PASS**, better than B2 |
| kitchen, whole certified frame vs input | — | −0.7608 | **−0.8290** | improved |
| F112 knob ≤ 1.5× frame | 2.13× | 0.54× | **0.95×** | **PASS**, repair survives, weakened |
| F108 flank > 12 | 0.57% | 0.40% | **0.23%** (max 45 → **29**) | best of the three |
| box 1 max ≤ 61 | 61 | 107 | **98** | **MISS** — argued below |
| box 2 max ≤ 17 | 17 | 63 | **16** | **PASS** |
| box 3 max ≤ 101 | 101 | 101 | **101** | **PASS** |
| box 4 max ≤ 127 | 127 | 116 | **116** | **PASS** |
| registration of the inspection layer | — | 1.0000 | **1.0000** | **PASS** |
| tests | 94 | 94 | **94** | **PASS** |
| byte-identity outside the rewrite | — | asserted | **asserted** | **PASS** |

Whole-frame new structure (differs from BOTH the input and the reference by more
than 12 levels): **8183 px → 4293 px.** Rewrite share 25.38% → 18.99%; on the
factory 82.17% → 51.12%, and the GT-SSIM went UP through that withdrawal, which
is the cleanest possible confirmation that the region veto was keeping content
that hurt.

### Box 1, argued honestly: the bar is MISSED and the escape does not apply

Six pixels of 693 exceed the routed maximum, at **x482–484 y150–153** — down
from 15 in B2. They are supplied by frame 9 (the box's focus rises monotonically
with frame index, 5.3 at frame 0 to 187.3 at frame 11, so frame 9 is a genuinely
far-focused observation of background). But the escape clause asks for PROOF of
sharper content, and the numbers do not give it: focus energy at those six pixels
is 103.9 (routed) → 112.0 (corrected) against 102.0 in the reference — a 7.8%
rise — while `|corrected − routed|` there averages **86 levels**. An 86-level
change buying 8% of focus energy, three pixels from a curved bottle's silhouette,
is a residual limb-crossing admission (F92), not resolved detail. **Reported as
a MISS.** The box as a whole is nonetheless better than B2 on every reading:
mean 7.48 → 6.19, max 107 → 98, and the pale line itself is gone from
`B2R2_defect2.png`.

Note also what box 1's focus energy does across the round: 41.8 → 54.7 (B2) →
44.7. The local veto gives back about two thirds of B2's sharpening in that box.
That is the never-degrade rule working as designed — B2's sharpening there was
partly bought with the very content the manager flagged — but it is a real cost
and F112/R6.4's warning still applies to reading either number alone.

## 20. The three defects, by eye

`out/certify/B2R2_defect{1,2,3}.png`, four panels at 6×: ROUTED (input) | B2
region veto only | CORRECTED | REFERENCE frame 6. Four and not three because the
round is a correction, and without the B2 panel a reader cannot tell a defect
that was fixed from one that was never there.

1. **Box 2 — CLEAN.** The pale diagonal wall streak and the spur on the pump's
   left limb are both gone; the wall reads as the reference does. Box 2's max
   |Δ| is 16 against the routed 17.
2. **Box 1 — the line is gone**, and a ~4 px pale nub survives on the
   silhouette at y≈150 (the six pixels of §19). The resolved background
   structure behind the bottle is retained.
3. **Box 4 — the bright dashes are gone.** The dark streaks at the shelf edge
   are REDUCED, not removed (147 px of new structure → 93), and the honest
   reading is that most of what remains there is not a smear but a sharpening
   the eye reads as blocky: focus energy in that band is 62.4 (routed) → 82.3
   (corrected), against 42.0 in the reference. It should be looked at again; it
   is the weakest of the three results.

## 21. NEGATIVE deliverables of the correction round

1. **The frontier clause as a conditional erosion peeled to a fixed point.**
   BUILT FIRST and REJECTED on measurement. It does not converge (still peeling
   after 200 passes, 8312 px), because at any place where the rewrite legitimately
   differs from the input over a wide area it simply eats the area; and it is
   simultaneously BLOCKED wherever a single in-budget pixel row separates the
   frontier from the loud content behind it, so it missed the very cluster it was
   built for. End to end it took the kitchen certifier to 8.9227 — worse than the
   region veto alone — and the knob to 2.32×. The bounded dilate-once form
   (§17.3) is strictly better on every reading. *A termination condition is not
   the same thing as a bound.*
2. **Protecting pixels the certifier positively prefers from the frontier
   clause** (revert only where `pooled ≥ 0`). Built, measured, and it changes
   almost nothing (knob 0.95× → 0.93×, d3a residue 93 → 97) — because §16 says
   the certifier cannot see clusters this small either way, so its "preference"
   there is not evidence. Dropped as complexity with no measured effect.
3. **A morphological opening of the rewrite** at radii 1–5, as a filament rule.
   Measured, and it catches only what clause 1 already catches more
   principledly: at every radius it removes 0% of the box 2 wall streak and 0%
   of the box 1 and box 4 clusters. Redundant.
4. **A pooling window of 9 px for the cluster clause.** It is a no-op by
   construction — 81 pixels cannot meet a 200-pixel quorum — which is the
   quorum doing its job, and worth recording so nobody re-tries it.
5. **Tuning any clause to the four user boxes.** NOT DONE. `FRONTIER_SLACK` is
   `ceil(GATE_TOL)`; §18 prices 1/2/3/5 px openly so the choice is visible, and
   1 px misses box 2 by one level while 3 px costs 40% of the knob repair.

## 22. What this round changes about the record

* **§13.3 of the first round is closed and was understated.** "The veto's
  granularity is coarse" was recorded as a missed opportunity — a finer partition
  would keep more good pixels. It was also a correctness hole in the other
  direction: below the partition's own minimum size the veto was not coarse, it
  was *absent*.
* **§7's frontier seam is now partly paid for.** Clause 3 addresses exactly the
  case §7 measured and declined to fix — a sharpness step at the frontier — and
  does it by withdrawing rather than by blending, so negative deliverable 6
  (feathering) stays rejected. The measured residual is 653 loud frontier pixels
  of 9489.
* **A general rule for this project's vetoes.** *A never-degrade rule that
  evaluates only at one scale grants unconditional authority at every scale
  below it.* B2's veto was correct region by region and still shipped five new
  artifacts, because a mean over 79019 pixels cannot see 17. Any future
  arbiter — the motion series B3 is designated to build included — needs its own
  answer to "what is the smallest thing this can see?", and §16's KAT is the
  shape of that answer.
* **The eyes beat the aggregates for the fourth time in this project's record**
  (§12.8, and F115 counted three). This time the aggregate that was blind was
  itself a never-degrade veto built to protect against exactly this.

## 23. Micro-round B2R3: the boundary-abstention clause, REJECTED on measurement

The two residual defects of §19–§20 were diagnosed as one class — rewrites that
cannot be positively arbitrated **and** sit at a layer silhouette — and the cure
proposed was one clause with no new tuned number:

> *Abstention near a geometric boundary is refusal.* Inside B1's own boundary
> band (`dec.diag["band"]`, half-width `BAND = 5`), dilated by this module's own
> `FRONTIER_SLACK = ceil(GATE_TOL) = 2` px, a rewrite survives only with a
> POSITIVE certifier verdict at quorum; abstention or no coverage reverts.

Licence: F92 (a curved object's limb is view-dependent and never trustworthy
from a moved frame) and F106 (what cannot be arbitrated must not be applied).
Built exactly as specified — `scene_model.py boundary` prices it — and **not
shipped.** The shipping path is byte-identical to 5ec37d7.

### Predictions, pre-registered, against outcomes

| # | prediction | outcome |
|---|---|---|
| 1 | the box-1 fleck reverts (uncertified, limb-adjacent) | **FALSE** — 0 of its 11 px are in the zone |
| 2 | the box-4 junction reverts, the rag's interior sharpening survives | **HALF** — interior survives; the junction loses 11 of 104 new-structure px |
| 3 | the knob repair survives (interior, certified) | **FALSE in the principled order** — 1.58×, the brief's own hard stop |
| 4 | the factory barely moves | **TRUE** — 0.982061 → 0.982060 |

### The measurements

| variant | kept px | certifier | subset of A | knob | box 1/2/3/4 max | flank | factory GT-SSIM |
|---|---:|---:|---|---:|---|---:|---:|
| **A shipped (5ec37d7)** | 69294 | **8.6971** | — | **0.95×** | 98 / 16 / 101 / 116 | 0.23% | **0.982061** |
| B clause **then** frontier | 64489 | 8.7419 | **NO (+438)** | **1.58× FAIL** | 101 / **28** / 101 / 116 | 0.30% | 0.982030 |
| C frontier **then** clause | 66030 | 8.7252 | yes, −3264 | 1.05× | 98 / 17 / 101 / 116 | 0.22% | 0.982060 |

Order B is the principled one — a refusal belongs with the cluster clause it
modifies, so the frontier clause sees the true final frontier — and it fails two
bars. Order C is the only safe one, holds every bar, and **fixes neither defect**
for a cost of 0.028 certifier levels and 3264 pixels. Two rejections for two
different reasons, and both are worth keeping.

### 23a. Why order B breaks the knob: `quiet_frontier` is NOT monotone in its input

The knob box is 31.4% BOUNDARY and 44.0% of it lies inside the dilated zone, so
the obvious suspicion is that the clause ate certified repair pixels. It did
not: of the 438 px that B keeps and A reverts, **0 are in the zone and 0 are
positively certified.** The mechanism is the frontier clause's own shape.
`bad = keep & ~inner & loud` is SEEDED by loud pixels lying ON the frontier, and
each seed grows a 2 px disc. Deleting a seed upstream deletes its disc. The
clause removed 32 of the 653 loud frontier pixels, `quiet_frontier`'s own
withdrawal fell 5976 → 5349 px, and 438 pixels the shipped pipeline reverts
survived — 7 of them inside the knob, which is the whole 0.95× → 1.58×.

**Shrinking a rewrite mask can make a downstream clause withdraw LESS.** This is
a composition hazard for every future refusal in this module, not a fact about
this one, and it is now recorded in `quiet_frontier`'s docstring. Two
consequences: a new clause is only safely composable AFTER the frontier clause,
where it can only remove; and **the strict-subset assertion is the instrument
that catches it** — B was caught by the subset check before any bar was read.

### 23b. Why the clause cannot reach either defect: the diagnosis was wrong

Both halves of "at or near a layer silhouette, with no certifier coverage" fail
against measurement.

* **Box 1's fleck is not near a boundary B1 knows about.** It is layer 5 on both
  sides, 3.6–6.6 px from the nearest band pixel and therefore **8.6–11.6 px from
  the nearest label edge**. No dilation of B1's band by a constant this module
  already owns reaches it. §15 recorded the reason without naming it: "the
  rewrite reaches into the Lubriderm bottle as a wedge that crosses its left
  silhouette" — a wedge crossing a silhouette *is* the statement that B1 put no
  edge there. **A clause licensed by F92 cannot be cashed through a
  decomposition that does not know the limb exists.**
* **Box 4's junction is interior and largely certified.** Median distance to the
  band **10.1 px**; of its 104 new-structure pixels, **60 have a cluster quorum
  and all 60 are positively certified**. The clause is not entitled to them and
  the brief's "no certifier coverage to defend it" is not what the instrument
  says.

### 23c. What the round did settle: both residuals are geometry, not detail

The escape clause offered was "PROVEN sharper-than-reference content — focus
energy up AND structure matching a raw far-focused frame". **Both defects clear
the focus-energy half and fail on inspection, and they fail the same way.**

Box 1's fleck, printed pixel by pixel (BGR, reference | routed | shipped):

```
y151  x483  ( 22, 80, 92) ( 19, 77, 89) ( 91,177,190)     dark -> BRIGHT
y151  x484  (135,189,200) (131,185,196) ( 77,167,181)   bright -> DARK
```

The dark/bright silhouette edge has been **translated one pixel right**, not
resolved. Focus energy rises 102.0 → 108.1 because a *moved* edge carries more
energy than a soft one, not because anything was resolved. Box 4 is the same
story two rows wide: the shelf/shadow contour's row means go
`179 127 67 38` (routed) → `183 157 117 35` (shipped) against
`165 133 88 43` (reference) — the contour is displaced ~2 rows and steepened,
which is what the eye reads as *blocky*, and focus energy rises 64.9 → 82.8
against the reference's 44.1. See `out/certify/B2R3_defect{1,2,3}.png`.

**Focus energy cannot serve as evidence that rewritten content is real.** It is
monotone in edge contrast and blind to edge POSITION, so a displaced silhouette
scores exactly like a resolved one. F112/R5.2's counter-instrument survives as a
counter-instrument — it correctly refuses the "everything got softer" story —
but §20.3's reading of box 4 ("most of what remains is a sharpening the eye
reads as blocky") is **withdrawn**: it is a 2 px contour displacement. Box 1's
MISS at max 98 stands, now proven rather than argued.

### 23d. Recommendation

Ship nothing. The residual class is **sub-arbitrable interior geometry**: too
small for the certifier (§16's KAT floor is a few dozen pixels), too far from
any boundary B1 draws for a geometric refusal, and invisible to `agreement_budget`
because at a high-contrast contour the budget is tens of levels (median 13.9 at
d3a) while the defect is a 1–2 px *displacement* that stays inside it. The
missing instrument is not another veto over the existing evidence — it is a
statement about **contour continuity**: a rewrite may not move a strong image
contour that both the input and the reference agree on. That is a new arbiter
with its own KAT, which is B3's size, not a micro-round's.

Also open, and cheap: 26% of the frontier at d3a is loud AGAIN after one
application (33 of 126), so the residual-loud-frontier figure of §17 (653 of
9489, 6.9%) is concentrated exactly where the eye complains.
