# Per-region two-frame architecture — prototype findings

Prototype of the user's F86-era proposal, on the books since F86 and named in
F108 as the capability that could close the zone-coverage gap. Code:
`research/twoframe.py` (+ `research/twoframe_probe.py`). Nothing in `src/`,
`tests/` or any other research file was touched; 85 tests pass.

```
.venv/bin/python research/twoframe.py kat        # instrument known-answer tests
.venv/bin/python research/twoframe.py factory    # analytic GT factory
.venv/bin/python research/twoframe.py oracle     # the architecture's ceiling
.venv/bin/python research/twoframe.py variants   # the two free choices, A/B'd
.venv/bin/python research/twoframe.py kitchen    # F108 acceptance test + crops
.venv/bin/python research/twoframe_probe.py [streak]
```

---

## 1. Verdict first

**The F108 acceptance test PASSES.** The tan streak on the Lubriderm bottle's
lower-right flank is gone, measured and seen. It also clears a second artifact
the shipped path produces and nobody had scored — the cat figurine renders
doubled in the shipped output and single in this one.

**It is not yet a promotion.** On the analytic factory it costs 0.0070 GT-SSIM
against the shipped depth-bin path, and an oracle rung proves that entire gap is
per-layer shift ESTIMATION, not the architecture: the identical two-frame
construction given exact shifts scores 0.972958, i.e. the shipped path's number.

---

## 2. Mechanism — why this closes a gap no local gate could

F108 characterised the wall precisely: in a smooth low-contrast stretch there is
*no local evidence*, so no local gate can act. Every refusal mechanism the arc
built is evidence-driven, and all of them are blind there simultaneously.

The two-frame architecture does not need local evidence, and the reason is worth
stating in full because it is more than "two frames are better conditioned":

1. **The pair choice is a REGIONAL statistic.** A tile pools ~10⁴ focal-peak
   votes weighted by focus contrast, so a textureless pixel simply abstains and
   the region still decides. This is the channel F108 asked for: support that
   does not depend on per-pixel evidence.
2. **Each frame of a pair owns exactly ONE depth layer**, so it can be warped by
   ONE RIGID translation. There is no depth-dependent field inside a region —
   hence no field discontinuity, no stretch limiter, and no soft geometric blend
   (F106 is satisfied structurally rather than by a rule).
3. **Geometry and focus are co-diagnostic.** The layer a frame gets wrong is, by
   construction, the layer it is defocused in, so the focus contest discards it
   without being told to.
4. **The fan cannot form.** F108's proven mechanism is the mixing of
   near-reference frames whose edges are uncorrected by ±3–4 px. Two sources,
   one of them sharp, cannot fan.

And the payoff on the kitchen turns out to be sharper than predicted:

> **The bottle is sharpest AT THE REFERENCE (frame 6).** The whole F99→F108 arc
> spent itself teaching frames 8–11 to place a bottle they are simultaneously
> most blurred in and most displaced in. The regional focal statistics simply do
> not ask them to. Measured provenance in the bottle box: pair (6,6) covers
> 24 346 px with median |fused − frame 6| = 0.0, pair (6,11) covers 12 485 px
> with median |fused − frame 6| = 6.0 against 25.0 for frame 11. The +19.97 px
> correction is not made better — it is not needed.

That is the architectural claim, and it is falsifiable: it holds only where each
region's layers each have a frame that is both sharp and near-reference enough
to place. It would fail on a sweep whose near object is sharpest at an extreme.

---

## 3. Instruments — known-answer tested before use (§12.1)

This arc lost three findings to an unvalidated instrument, so both new ones were
tested against known answers first (`twoframe.py kat`).

| KAT | question | result |
|---|---|---|
| 1 | does the global stage reproduce the shipped one? | worst per-pixel difference **0** vs `align_stack(depth_bins=0)` |
| 2 | can the coarse-to-fine masked ECC find a large shift? | −2/−5/−12/−20/−30 px recovered to **0.000 px** |
| 3 | does the pair chooser find the right frames? | factory: **(1, 4)** in 52 of 88 tiles — the analytic truth |
| 4 | are the per-layer shifts right? | near−far differential recovers **87–100%** of the analytic parallax |

KAT 2 exists because a single-scale ECC from identity does not find a 19 px
shift and phase correlation saturates at ¼ of the patch (PLAYBOOK §0b); the
estimator runs on a 3-level Gaussian pyramid.

KAT 4 needed its own thinking: the global affine has already absorbed an unknown
share of the scene motion, so the absolute per-layer residual has no closed
form. The DIFFERENTIAL does — whatever the affine took, it took from both layers
equally — so `near − far` must equal `−(k−ref)·(3.2 − 0.7)`. The ratios are
0.87 / 0.90 / 0.92 / 0.98 / 1.00 across frames 0,1,2,4,5. The under-read is on
the side of the reference where the near object has swung across background that
the reference-frame mask still counts as "near"; it is ~0.3–1.0 px and it is
exactly the residual the factory number below is paying for.

---

## 4. Measurements

### 4a. Analytic GT factory (`parallax_gen.py`, unmodified)

| variant | GT-SSIM |
|---|---:|
| shipped depth bins, no refusal | 0.957808 |
| **shipped depth bins + refusal** | **0.972808** |
| two-frame, no refusal | 0.962869 |
| **two-frame + refusal + reference fallback** | **0.965810** |

Reproduced baseline is 0.972808, not the 0.973225 F108 records; the difference
is not explained here and the shipped path was not touched, so it is logged
rather than chased.

**Oracle ladder — where the 0.0070 actually lives** (PLAYBOOK §0b: run the
ladder before building estimators):

| rung | GT-SSIM |
|---|---:|
| 2-frame, unaligned | 0.749356 |
| **2-frame, exact per-layer shifts (oracle)** | **0.972958** |
| 2-frame oracle, near layer 1 px wrong | 0.946205 |
| 6-frame oracle, per-layer | 0.979758 |

Three things follow, and they are the most useful numbers in this note:

* The architecture's ceiling on this scene **equals the shipped path** (0.972958
  vs 0.972808). Two frames are not the limitation.
* It is **2.7× more sensitive to a shift error** than the arc is used to: 1 px of
  near-layer error costs 0.0268 GT-SSIM, because with two sources a misplaced
  layer is misplaced *everywhere* — no other frame dilutes it. The shipped
  measured result sits 0.0072 below its own oracle, i.e. ~0.3 px of residual.
* Two frames cost **0.0068** against a six-frame oracle even on a strictly
  two-plane scene, which is the honest price of the architecture itself.

### 4b. Kitchen sweep — F108's acceptance test

Streak region x 600–700, y 230–400 (uncropped), reference frame 6. Scored on the
**3311 px low-contrast white flank** inside that box — the part where F108's own
bar applies ("compositing that region from the reference alone is clean"), so
the reference frame is the local truth there.

| output | mean \|Δ\| | max \|Δ\| | flank px over 12 |
|---|---:|---:|---:|
| shipped depth-bin path | 5.98 | 65 | **16.34%** |
| **two-frame** | **1.96** | **10** | **0.00%** |

Crops: `out/depth_align/TF_streak_ref_shipped_twoframe.png` (top row reference |
shipped | two-frame, bottom row 5× amplified difference from the reference). The
shipped amplified diff shows the streak as a bright vertical strip on the flank;
the two-frame diff shows a smooth low-frequency field there and nothing else.

Bottle right-edge offset of the OUTPUT against the reference frame: shipped
+0.02 px, two-frame +0.00 px. Per-frame residuals for context (unchanged
instruments, `group_align.edge_shift`):

| frame | global only | shipped path | two-frame |
|---|---:|---:|---|
| 8 | +9.09 | +0.95 | frame not used at the bottle |
| 9 | +12.53 | +1.15 | frame not used at the bottle |
| 10 | +16.21 | +1.53 | frame not used at the bottle |
| 11 | +19.97 | +1.83 | frame not used at the bottle |

Pairs elected on the kitchen, and what each measured:

```
(6, 8)  19.8% of area   refused 20.4%   [ref, (-1.57,-0.21)]
(6,11)  26.7%           refused 18.8%   [ref, (-1.97,+0.04)]
(6, 6)   8.8%           refused  0.0%   [ref]
(4, 6)  27.7%           refused 11.8%   [(-2.00,-0.23), ref]
(11,11)  9.2%           refused  0.0%   [(-1.08,-0.15)]
(9,11)   7.7%           refused 42.1%   [(-1.16,-0.06), (-6.89,+0.10)]
tiles judged single-layer: 45/160;  frames actually used: 4, 6, 8, 9, 11 of 12
```

Note "refused" here is not the shipped path's `withheld`: a refused pixel falls
back to the pair's other member (or, where both are refused, to the unwarped
reference), so it is still a real observation of that surface.

### 4c. Two artifacts and a broad sharpness gain, by eye

`out/depth_align/TF_disagreement.png` (eyetool 2.0, automatic crops at the three
regions where the outputs disagree most, so the windows are not hand-picked):
the two-frame output is sharper than the shipped one in **all three** — the
stove display and the "FARBERWARE" pot legible where shipped is soft, the
countertop grain resolved, the Hot Cocoa label crisp. The likely cause is that
the shipped path withholds 22.5% of pixels per frame (F106/F108) and those
regions fall back to reference-quality; the two-frame path has no equivalent
whole-frame withholding because its pair choice already puts a sharp source in
every region.

`out/depth_align/TF_cat_ref_shipped_twoframe.png` (hand-picked window, recorded
exactly: y 365–480, x 320–470, so it diagnoses and does not promote): the cat
figurine is **visibly doubled** in the shipped output — double mouth, double ear
outlines, smeared fur — and single and clean in the two-frame output. This is a
previously unscored defect of the shipped path on its own validation scene.

### 4d. Cost

2.22 s end to end on the kitchen against the shipped path's 3.26 s (align +
fuse), on the same machine, same run. Five renders of the full frame plus five
two-frame fusions is cheaper than twelve-frame alignment with the motion-group
override. 5 of 12 frames are read.

---

## 5. NEGATIVE deliverable — everything tried and rejected

Each cost a run; each is recorded with its mechanism so a later session can tell
whether its conditions have changed.

1. **Scoring the acceptance test over the whole streak box.** REJECTED as an
   instrument, not as a result. The box contains background the fusion is
   *supposed* to change, so the measurement said two-frame was WORSE (mean 9.91
   vs shipped 2.28) while the flank-scoped measurement of the identical images
   said 1.96 vs 5.98. DEVSTYLE §12.2 relearned the hard way: scope the metric to
   the thing that is failing.
2. **One ECC per frame (its own layer only).** REJECTED. A frame warped by one
   rigid translation has no internal disagreement, so disocclusion is invisible
   and the refusal gate can never fire. Measured consequence: a white ribbon of
   the background frame's own displaced bottle rendered beside the correctly
   placed one. The fix is to measure EVERY frame against EVERY layer — the
   frame's own layer says where to put it, the difference between its two layer
   motions says how far its near layer swung, which is the ribbon width (F82).
3. **Refusal with no fallback.** REJECTED. With two sources, blocking one forces
   the other, which at a disocclusion ribbon is the frame whose near layer is
   displaced — refusal swaps one wrong observation for another. Factory
   0.954678 (worse than no refusal at all, 0.962869). Admitting the *unwarped
   reference* as the geometry of last resort, only where both members are
   refused, gives 0.965810. This is F82's own answer ("refused zones become
   reference-quality") and it is what makes refusal viable in a pair.
4. **Refusal keyed on the guided depth step alone.** REJECTED as sole evidence
   (kept in a union). Flank still shows max 15 and 0.85% over 12, because the
   guided depth map ramps across exactly the junctions that matter (F104/F108).
   The pair's own layer boundary — where the focus contest between the two
   members flips — is a depth discontinuity by construction and cannot ramp;
   with it the flank reaches 0.00%. Union of the two is the default, and scores
   identically to the pair edge alone here.
5. **Eroding the layer fit mask for boundary purity** (both sides of a defocused
   silhouette are compromised, so shrink the mask). REJECTED in both forms.
   Eroding the *texture-gated* mask is a category error — it decimates a speckle
   pattern rather than shrinking a region — and collapsed the factory to 0.889 /
   0.918 / 0.513 at 3/7/13 px. Eroding the *dense* layer and re-gating is the
   correct implementation and still loses: 0.964534 (3 px), 0.668484 (7 px),
   0.512685 (13 px). The compact near plane runs out of textured support.
6. **Hard region paste instead of the multiband stitch.** NOT rejected, but the
   expected result did not appear. F79 found hard region-copy seams
   unacceptable; here paste and multiband differ by mean 1.14 levels with 0.81%
   of pixels over 8 and **no visible seam in either**
   (`out/depth_align/TF_stitch_ablation.png`). The mechanism is that all
   candidates are registered to the SAME reference geometry, so neighbouring
   candidates differ in which frame supplied a pixel, not in where content sits
   — the seam F79 fought was a geometric one. Multiband is kept anyway (it is
   free and strictly safer), but the honest read is that the stitch was expected
   to be the risk and, in this construction, is not.
7. **Segmenting regions into objects.** NOT ATTEMPTED, deliberately. F98 settled
   that turning feature groups into pixel regions is the open problem and that
   the propagation rule ends up doing the real work while looking like plumbing.
   A tile claims nothing; it is only a locality over which focal statistics are
   pooled. Gap-based clustering of the focal distribution was likewise not used —
   Otsu, per F98.
8. **The five refusal-side approaches F108 already closed** were not retried.

---

## 6. Honest costs, open risks

* **Not non-regressing.** −0.0070 GT-SSIM on the factory. Depth bins are
  near-ideal on a cleanly depth-separated scene (F101), which is exactly what
  the factory is, so this is the expected shape of the trade — but it is a real
  regression and blocks promotion as a replacement.
* **No sanity gate on a layer fit.** In the 7 px erosion run the estimator
  returned a badly wrong shift that was still inside `max_shift`, and GT-SSIM
  fell to 0.668 with nothing objecting. The architecture has no equivalent of
  F106's "measured motion that nothing explains obliges refusal" for its own
  estimates. This is the first thing to build before it is trusted anywhere.
* **Two frames cannot express more than two layers.** A region with three depths,
  or a continuously receding surface, is served by two frames or by one. 45 of
  160 kitchen tiles were judged single-layer; the model has no way to *report* a
  three-mode region, it just picks the best two. The 6-frame oracle being 0.0068
  above the 2-frame oracle on a strictly two-plane scene is the floor of this
  cost, not the ceiling.
* **The multiband stitch leaks low frequencies** across candidate boundaries
  (mean 1.14 levels vs paste). Small, but it is a defocused candidate's coarse
  bands entering a sharp region, and it will grow with the number of pairs.
* **`MAX_PAIRS = 6` was hit on the kitchen.** The merge is greedy by tile weight;
  a tile can be forced onto a pair up to `MERGE_TOL` frames from what it asked.
* **Resolution.** Every parameter (`TILE`, `STRIDE`, `MAX_SHIFT_FRACTION`,
  `MIN_LAYER_PIXELS`) is pixel-scaled and only exercised at 774×518 and 560×420.
  F107's disease exactly; the estimate-small/apply-native transfer
  (`field_native(X) = s·field_small(X/s)`) applies unchanged here because every
  candidate's geometry is a single homogeneous matrix, which is the easiest
  possible thing to rescale.
* **`--enhance` was not touched.** F56 licenses it for exactly two frames
  ≤1600 px and that licence does not transfer to a stitched composite.
* **Two scenes only**, per the brief and DEVSTYLE §13. Nothing here is validated
  on a scene where the near object is sharpest far from the reference, which is
  the regime that would test §2's claim hardest.

---

## 7. What integration would require

In priority order, because the first item alone may close the factory gap:

1. **Improve the per-layer shift, or refuse it.** The oracle says 0.0072 of
   GT-SSIM is sitting in ~0.3 px of estimation error. Candidates, cheapest
   first: (a) iterate the fit once against the corrected frame, composing into
   the matrix so the pixels are still resampled once; (b) replace the masked ECC
   with `group_align`'s edge-profile estimator, which is the arc's most-validated
   motion instrument and already handles the defocus bias by focal weighting;
   (c) a validity gate that refuses a fit whose two layers order implausibly.
2. **Route rather than replace.** The same F101 logic that produced the
   motion-group override applies: keep the shipped path and use the two-frame
   composite only where it demonstrably differs — here, where a region's focal
   statistics are decisively bimodal AND one of its layers is sharpest at a
   frame the shipped path corrects badly. Non-regressing by construction.
3. **Let a region take three frames when it has three modes.** The Otsu split is
   already recursive in spirit; the cost is one more render and one more stitch
   boundary, and the 6-frame oracle says up to 0.0068 is available.
4. **Then, resolution.** Per F107, and easier here than anywhere else in the
   pipeline because each candidate's geometry is one matrix.

The prototype deliberately leaves the runtime untouched (DEVSTYLE §13: iterate
in `research/` so nothing needs re-validation and the promotion decision stays
open until evidence closes it).

---

# Hardening pass — 2026-08-02

Second Opus session, on the promotion blockers §7 listed. Code: `research/twoframe.py`
only; nothing in `src/`, `tests/` or any other research file was touched; 85 tests
pass. New entry points:

```
.venv/bin/python research/twoframe.py kat2       # KATs for the two new instruments
.venv/bin/python research/twoframe.py hardening  # the two acceptance tests
.venv/bin/python research/twoframe.py fullres    # full-resolution transfer KAT
.venv/bin/python research/twoframe.py modes      # do kitchen tiles hold 3 modes?
```

## H1. Verdict first, and one correction to §4a

**The validity gate is built and both halves of its acceptance pass.** A deliberate
+8 px error in a layer shift takes the factory 0.971310 → 0.928212 ungated, and the
gate returns it to **0.970986** — inside 0.0004 of the clean run. On correct fits the
gate is silent: factory and kitchen are bit-identical with it on and off, zero
refusals on either scene.

**The factory gap is 79% closed: 0.965810 → 0.971310** against the shipped path's
0.972808. The kitchen keeps its F108 acceptance: flank mean |Δ| **2.16**, max 10,
**0.00%** over 12.

**And §4a's headline was wrong, which is the most important thing in this note.**
The oracle ladder there concluded "that entire gap is per-layer shift ESTIMATION,
not the architecture". It cannot support that: every rung of it bypasses
`twoframe_stack` — two frames, two analytic shifts, one `fuse_perband`, no global
affine, no tiling, no stitch, no refusal, no crop. Feeding the same exact shifts
THROUGH the architecture scores **0.968782**, and the measured hardened run scores
0.971310 — i.e. the architecture BEATS its own exact-per-layer-shift oracle. Two
things follow:

* The gap was never mostly estimation. It was **geometry** (H3), and the estimator
  contributed 0.0016 of the 0.0055 recovered.
* "Exact per-layer shift" is not this architecture's optimum, and that is a
  structural fact, not a fluke: a pair member is rendered WHOLE and its coarse
  bands enter the fusion outside its own layer, so the score-optimal single
  translation for a frame sits slightly between its two layers' exact shifts. Do
  not chase analytic exactness past ~0.5 px here; it is the wrong target.

An oracle that skips the machinery measures the idea, not the build. Both rungs are
now printed side by side by `twoframe.py oracle`.

## H2. TASK 1 — the validity gate

### Instruments first (§12.1), `twoframe.py kat2`

| KAT | question | result |
|---|---|---|
| 5 | does the focal-weighted edge-profile fit agree with the ECC, in sign and value? | −2/−5/−12/−20 px and two 2-D shifts recovered to **0.000 px** by both |
| 6 | does the gate separate right from wrong? | 0.0/0.5/1.0 px error → **verified**; 2/4/8/20 px → **contradicted**, statistic 1.97/3.98/7.97/19.67 |

KAT 5 exists because the two estimators must share a sign convention before either
can verify the other; both return the warp taking REFERENCE coordinates to MOVING
ones. KAT 6 fixes `GATE_TOL = 1.5 px` between the two measured populations: real
correct kitchen fits verify at 0.01–0.32 px, and the smallest error the gate must
catch reads 1.97.

### Design

A fitted layer shift is applied only after forward verification: apply it, then ask
the layer's **own material edges** whether the layer has stopped moving.

* **Independent of the fitter.** Profile correlation along edge normals versus
  masked ECC over a dense region — different evidence, different failure modes.
* **Material only** (`motion_groups._material_features`), so a curved object's
  view-dependent limb slide is not read as fit error (F92).
* **Focal-weighted**, so defocus cannot bias it toward the reassuring answer of
  zero (F99).
* **Observability-aware.** The residual is solved as a translation and only its
  components along observable eigen-directions of the weighted normal moment are
  gated — a feature whose normal is perpendicular to a motion agrees with it
  vacuously (F103).
* **No image warp.** Sampling the moving frame's profile at `M(x)` *is* sampling
  the warped image at `x`, so verification costs one profile per feature.
* **Free of a `max_shift` illusion.** The 0.668 fit was inside `max_shift`; a
  plausibility bound cannot tell a correct fit from a wrong one of the same size.

### Trinary, per F106, and repair before refusal

The verdict is **verified / contradicted / unverifiable**, and the three do
different things — this is PLAYBOOK §0's "is the observation absent, or mixed?"
asked of a *fit*:

* **verified** → apply.
* **contradicted** → the evidence says the layer is still displaced. But the
  verification did not only say NO, it measured HOW WRONG, and that measurement is
  the correction. The fit is **repaired** from it (up to `GATE_REPAIRS = 4`
  iterations, to `GATE_CONVERGED = 0.05 px`, preferring doubly-focal evidence) and
  re-verified. Only a fit that still fails is refused.
* **unverifiable** → no evidence either way. The *correction* is declined (the
  frame keeps the global stage's geometry, the baseline everything already accepts)
  but the *observation* is not thrown away. Refusing here would gut the
  architecture: the whole point is to use frames the reference cannot see well.

A refused member supplies nothing and the unwarped reference is admitted in its
place — §5.3's measured best fallback, now reached by a gate rather than by a
disocclusion probe.

Repair is not decoration. Refusal alone took the eroded-mask disaster from 0.668 to
**0.916**, because refusing the near layer over 82% of the frame hands it to a
reference that is defocused there. One repair pass takes it to 0.957. Refusal is
the floor, not the plan.

### The caught-error demonstration (`twoframe.py hardening`)

| scenario | gate off | gate on | vs clean 0.971310 |
|---|---:|---:|---:|
| +2 px into pair 0 layer 0 | 0.943368 | **0.971188** | −0.000122 |
| **+8 px into pair 0 layer 0** | 0.928212 | **0.970986** | **−0.000324** |
| +20 px into pair 0 layer 0 | 0.928210 | **0.971226** | −0.000084 |
| both layers, x and y (+8,−3) and (−6,+4) | 0.823803 | **0.968267** | −0.003043 |
| §6's own case: fit masks eroded 7 px | 0.670662 | **0.957371** | −0.013939 |
| eroded 13 px | 0.513248 | **0.898597** | −0.072713 |

The gate reads the injected error almost exactly (rms 3.68 for +8 px, 8.39 for
+20 px) and repairs to 0.02–0.04 px residual. The erosion rows stay short of clean
because erosion damages *every* fit and sub-`GATE_TOL` damage legitimately passes —
the gate promises to catch geometric errors, not to undo a bad mask.

**Silence on correct fits.** Factory GT-SSIM 0.971310 with the gate on and off;
kitchen flank 2.16 both ways; **zero refusals on either scene**. The only thing the
gate says on the kitchen is `unverifiable` for pair (4,6)'s frame-4 layer, which has
63 337 dense pixels and **0 material features** — the reference is defocused there,
so Canny finds nothing to verify with. That is honest, and it costs that pair its
−2.00 px correction; measured by eye
(`out/depth_align/TF_hardened_vs_prototype.png`, disagreement-guided) it shows as
slightly softer countertop grain and no new artifact anywhere.

## H3. TASK 2 — where the factory gap actually was

### The geometry, not the estimate

The prototype composed the **global affine** into every render: `matrix = base @
T(shift)`. But the global affine is fitted across BOTH depth layers at once, so it
absorbs differential parallax as a spurious scale — F96's failure exactly ("a radial
term imitates two separated regions translating differently").

Measured on the factory, which has `BREATHING_PER_FRAME = 0.0` so every
non-translational term in its affine is over-fit: frame 1's affine reads scale
0.9954, spreading **±1.3 px** of sampling error across the frame even when the
layer's own shift is analytically exact. At 0.0268 GT-SSIM per px of near-layer
error (§4a), that term alone is the factory gap.

The architecture's own premise — one member, one layer, **one rigid translation** —
was never honoured. `_rigidify` evaluates the composed transform at the layer's own
centre of support and collapses it to the translation that reproduces it there.
Nothing is invented; the same displacement is applied rigidly as claimed.

Worth **+0.0044** on the factory (0.965810 → 0.970242 with the prototype's own fit).

### But not everywhere — and the split is decided by evidence, not a threshold

The kitchen's global affine is doing real work: frame 11 reads shear 0.0286 and
**y-scale 1.0285**, which is ±7 px of vertical variation across the frame. Blanket
rigidification costs the kitchen (flank 1.96 → 2.39). Two scenes wanting opposite
values is DEVSTYLE §12.3's signal that the choice is the wrong instrument.

So both are **proposed** and the one that forward-verifies better is applied, per
(frame, layer), using the same instrument the gate uses — the focal-weighted RMS of
what is still displaced at the layer's material edges, taken RAW rather than as a
solved translation, because a solved translation absorbs exactly the
spatially-varying error that distinguishes the two candidates. The factory picks
rigid, the kitchen picks affine, and it picks per layer (kitchen pair (6,8) layer 1
chooses rigid while (6,11) chooses affine). No threshold anywhere.

### The edge-profile estimator

`edge_refined_shift`: the ECC pyramid supplies the RANGE (the only instrument here
that finds a 19 px shift from identity), and the focal-weighted edge-profile fit
closes the residual on material edges. Each is used where it is valid — the profile
is ±28 px long, so it measures a residual well and a gross displacement badly.

**One F99 extension was needed and is the interesting part.** `group_align` weights
only the MOVING side's focal distance, because it measures an object near its own
focal plane and propagates. Here the pairing is inverted: a layer supplied by frame
11 is one the *reference* is deeply defocused in, so every profile match on it is
blurred-against-sharp. Measured: three of four kitchen layer fits moved TOWARD ZERO
against the ECC (frame 8's own layer −1.57 → −0.98) — F99's defocus bias arriving
from the side focal weighting was not watching. Requiring a feature to be sharp in
**both** frames makes the estimator DECLINE where it cannot see instead of returning
a confident under-read.

The GATE deliberately does *not* apply that symmetric weighting, and the asymmetry
is F104's rule transplanted: a defocused profile match carries a few TENTHS of a
pixel of bias, which forbids it from producing a rigid fit and is irrelevant to a
question whose wrong answer is several pixels out. **It may verify what it may not
estimate.** Measured both ways: 0.01–0.32 px on real correct fits, 7.97 px on a
deliberate +8 px error.

### The ladder

| configuration | factory GT-SSIM | kitchen flank mean | >12 |
|---|---:|---:|---:|
| shipped depth-bin path | **0.972808** | 5.98 | 16.34% |
| two-frame PROTOTYPE (ecc, affine, no gate) | 0.965810 | 1.96 | 0.00% |
| ecc fit, rigid layer geometry | 0.970242 | 2.39 | 0.00% |
| edge fit, affine layer geometry | 0.966606 | 2.07 | 0.00% |
| edge fit, rigid layer geometry | 0.971852 | 2.42 | 0.00% |
| **HARDENED (edge fit, verified geometry, gate)** | **0.971310** | **2.16** | **0.00%** |
| also verifying the fit choice (ecc vs edge) | 0.970256 | 2.16 | 0.00% |
| in-architecture oracle: exact per-layer shifts | 0.968782 | — | — |

TASK 2's bar was factory ≥ 0.9720. **0.971310 misses it by 0.00069** and sits 0.0015
under the shipped path, against the prototype's 0.0070. The bar was set from the
0.9730 rung, which H1 shows is not this architecture's ceiling — the honest ceiling
statement is that the remaining 0.0015 is NOT estimation (the fits verify at
0.02–0.17 px on the factory) and NOT layer geometry (both candidates are tested).
What is left is the architecture: the degenerate single-frame regions (14.2% + 4.1%
of the factory frame take one frame and are therefore defocused in the other plane),
the stitch, and refusal's own trade. Chasing it means changing the architecture, not
the estimator.

## H4. TASK 3 — full resolution (`twoframe_fullres`, `twoframe.py fullres`)

Easier here than anywhere else in the pipeline, exactly as §6 predicted: each
candidate's geometry is ONE homogeneous matrix, so the F107 rule
`field_native(X) = s·field_small(X/s)` becomes an exact matrix conjugation

    M_native = S @ M_small @ S⁻¹,    S = diag(s, s, 1)

with no resize of a coordinate map and no interpolation of a field. Ownership and
usable masks are per-pixel decisions, not geometry, and scale nearest-neighbour as
`fullres_apply` scales its occlusion masks. Native pixels are resampled exactly once.

KAT: kitchen frames upscaled 2× as the "natives", `working_width = 774`, output
downscaled and compared to the working-resolution output of the same run.

| quantity | mean \|Δ\| | over 12 |
|---|---:|---:|
| **native (downscaled) vs working** | **2.41** | **1.21%** |
| floor — the KAT's own 2× up / area-down round trip | 0.88 | 0.33% |
| control — the reference frame vs the working output | 6.72 | 13.20% |

2.7× the resample floor and 2.8× below the control: the transfer is sound, and the
excess over the floor is the fusion band structure resolving genuinely different
detail at 2×, not a geometry error. Crops are recomputed natively (as
`fullres_apply` does) so the two outputs do not start at the same pixel — comparing
them without intersecting the crops reports mean 13.2 and 28% over 12, all of it the
comparison's own misregistration. That trap cost a run and is worth remembering.

## H5. TASK 4 — three focal modes, measured, not built

`twoframe.py modes`, using the architecture's own Otsu bar recursively and nothing
new: of 160 kitchen tiles, **137 (85.6%) are two-mode and 125 (78.1%) carry a THIRD
mode clearing the same quality bar**. The brief's threshold was 5%.

So the demand is real and enormous — but read it carefully before acting on it. The
recursive Otsu bar is a *relative* variance criterion: a second split of an already
split side clears 0.45 easily whenever the side has any spread at all, and a
continuously receding surface (a countertop) will trip it everywhere. This
measurement says "two layers is a coarse description of most tiles", which was never
in doubt; it does NOT say a third frame would pay. Nothing was built, per the brief.
A measurement that would decide it: whether a third frame's own layer verifies (H2's
instrument) at a residual the two-frame pair cannot reach.

## H6. NEGATIVE deliverables — this pass

1. **Routing the CROSS-LAYER fits through the edge estimator and the gate.**
   REJECTED, and it was the session's sharpest trap. Those entries are never
   rendered; they exist only to give F82 its disocclusion width, the DIFFERENTIAL
   between a frame's two layer motions — the one quantity here with a closed form
   and the one KAT 4 validated the masked ECC against (87–100% of the analytic
   parallax). Sending them through the new machinery collapsed the differential and
   with it the refusal that closes the streak: pair (6,8) refused 20.4% → **0.0%**,
   pair (4,6) 11.8% → 0.0%, kitchen flank 1.96 → **3.34**, 2.14% over 12. The
   factory *rose* to 0.976941 doing it, which is the trap: a worse differential
   (3.41 vs the ECC's 3.75 against a truth of 5.00) under-refuses, and
   under-refusing flatters the factory while breaking the named acceptance test.
   Different question, different instrument, each used where it was known-answer
   tested.
2. **Verifying the ecc-vs-edge fit choice as well as the geometry.** NOT rejected on
   mechanism, rejected on measurement: 0.970256 vs 0.971310 on the factory,
   identical on the kitchen. The two fits differ by less than the selector's own
   noise, so the extra choice adds variance and no information. Geometry is
   selected; the fit is not.
3. **Blanket rigid layer geometry.** REJECTED as a rule (kept as a candidate).
   Factory +0.0044, kitchen flank 1.96 → 2.39. The kitchen's affine carries a real
   y-scale of 1.0285 by frame 11.
4. **Blanket edge-profile refinement with moving-side focal weighting only.**
   REJECTED. Kitchen flank 1.96 → 3.26 with 1.93% over 12, mechanism identified as
   reference-side defocus bias (H3). Fixed by symmetric weighting, not by tuning.
5. **Refusal without repair on a contradicted fit.** REJECTED. 0.916 where repair
   reaches 0.957 (erode 7); 0.9365 vs 0.9710 (+8 px injection). Refusal is correct
   as the floor and wasteful as the first response, because the verification has
   already measured the correction.
6. **The multiband stitch as the flank's problem.** INVESTIGATED, not changed. The
   flank is 91.5% owned by the degenerate reference-only pair (6,6), whose candidate
   is the reference exactly, so its 1.96–2.16 mean |Δ| is entirely low-frequency
   leak from neighbouring candidates through the stitch. Hard paste drops the mean
   to 0.57–0.89 but raises the tail (max 13–22, 0.4–3.5% over 12), because the leak
   is what was smearing a real difference at the (6,11) boundary below the
   threshold. Neither is clearly better and the acceptance test is defined on the
   default, so the default stands — but the flank number is measuring the stitch as
   much as the fits, and a future pass should know that.

## H7. What remains for runtime integration

1. **Route, do not replace** (F101), still unstarted and still the right shape: the
   two-frame path now loses the factory by 0.0015 rather than 0.0070, but "loses by
   less" is not "non-regressing". Use it where a region's focal statistics are
   decisively bimodal AND one of its layers is sharpest at a frame the shipped path
   corrects badly.
2. **The unverifiable layer.** Kitchen pair (4,6) has 63 337 dense pixels and zero
   material features, so 27.7% of the frame silently declines its correction. Limb
   edges are forbidden from the fit (F92) but explicitly allowed to decide coverage
   (F104) — the same licence may extend to VERIFICATION, where the alternatives
   differ by pixels. Worth one probe.
3. **The degenerate single-frame regions** are the largest remaining factory cost
   (18.3% of the frame taking one frame). `OTSU_MIN_QUALITY` decides them and has
   never been swept; it is a threshold, so per §12.3 look for the physical test
   first.
4. `twoframe_fullres` is validated at 2× on one scene. F107's own warning about
   provenance sensitivity applies: the pair election is re-run on downscaled pixels
   and its greedy merge ORDER changed under a resize round trip (the same pairs, a
   different order).
5. `--enhance` still untouched (F56 does not licence it for a stitched composite).

---

# Runtime integration — routing, 2026-08-02

Third Opus session. The architecture is now in the package
(`src/focusstack/twoframe.py`) and ROUTED from `pipeline.run`; `align.py`,
`fusion.py` and `motion_groups.py` were not touched. 92 tests pass (85 + 7).

```
focusstack shots/*.jpg -o out.png -v          # route on by default
focusstack shots/*.jpg -o out.png --no-twoframe-route
```

## I1. The port, known-answer tested before use

A port is a new instrument (§12.1), so the runtime module was required to
reproduce the research numbers exactly before anything was wired to it. It does,
to every printed digit:

| KAT | research | port |
|---|---|---|
| global stage vs `align_stack(depth_bins=0)` | 0 | **0** |
| analytic factory GT-SSIM | 0.971310 | **0.971310** |
| +8 px injected layer error, gate off / on | 0.928212 / 0.970986 | **0.928212 / 0.970986** |
| kitchen flank mean / max / >12 | 2.16 / 10 / 0.00% | **2.16 / 10 / 0.00%** |
| full-res transfer at 2x (vs floor 0.95, control 7.08) | 2.41, 1.21% | **2.41, 1.21%** |

Ported: pair election, the focal-weighted edge-refined fit, the verified geometry
choice, the validity gate with repair, F82 pair refusal with reference fallback,
the multiband stitch, and the F107 transfer (used above `WORKING_WIDTH = 1100`).
NOT ported, deliberately: every ablation knob (`fit="ecc"`, blanket rigid/affine
layer geometry, hard paste, mask erosion, oracle shifts). Those were settled in
F110; reopening one means re-running the research harness, not adding a runtime
parameter. The one test hook kept is `inject`, because the gate's acceptance test
is the reason the gate exists.

The full-res comparison has a trap worth restating: compare the native output
against the analysis run INSIDE the same call (`working_fused`), not against a
separately-elected working run. Across two elections the same KAT reads 3.91
instead of 2.41 — it is measuring the pair merge order's sensitivity to a resize
round trip (H7 item 4), not the transfer.

## I2. The routing rule, as VALIDATED — and the half of it that was wrong

The brief's candidate: engage where the shipped alignment's motion-group override
fired. Measured decisions:

| scene | groups overridden | two-frame max layer shift | licence | route |
|---|---:|---:|---:|---|
| analytic factory | 0 | (not built) | — | shipped, byte-identical |
| zero-motion | 0 (screened) | (not built) | — | shipped, byte-identical |
| small-motion | 0 (screened) | (not built) | — | shipped, byte-identical |
| kitchen | 2 | 2.1 px | 14.0 | **TWO-FRAME** |
| IMG-46 | 1 | 6.9 px | 20.2 | **TWO-FRAME** |
| large-motion | 3 | **19.2 px** | 14.0 | shipped (composite discarded) |

**Half of it holds, and half of it was falsified on large-motion.** Firing is the
right NECESSARY condition: it is the shipped path's own measurement that a
compact object's motion disagrees with its depth bin by >5 px at the object's own
location, i.e. that the scene strands an object. Where it does not fire, no
two-frame work is started at all, so the factory's 0.9728 and the sentinels are
preserved by construction rather than by measurement.

But firing is NOT sufficient. On large-motion the override fires on three groups
and the two-frame path loses the very object F103 fixed:

> The playing-card box is sharpest at frame 0 of 14 and needs +18.9 px. The pair
> elects frame 0 for it and fits it CORRECTLY (verified at 0.51 px) — and then
> F82's disocclusion refusal withdraws that member over 91% of the pair's area,
> because the ribbon it must refuse is as wide as the correction. The box comes
> back reference-defocused: `out/depth_align/ROUTE_largemotion_{box,wordmark}.png`
> shows the shipped output's crisp "LAS VEGAS" beside the composite's blur, and a
> template match of the reference frame's own box scores 1.000 against the
> composite (it IS the reference) and 0.910 against the shipped output.

This is not a surprise so much as a prediction coming due: §2 above stated the
architecture's licence as falsifiable before any of this was measured — "it holds
only where each region's layers each have a frame that is both sharp and
near-reference enough to place. It would fail on a sweep whose near object is
sharpest at an extreme." Large-motion is that sweep. The kitchen is its opposite
and that is exactly why it wins there: the bottle is sharpest AT the reference,
so the correction is not needed at all.

**The validated rule is therefore two-part**, and the second half is a question
only the composite can answer, so it is asked after building it:

1. the shipped alignment's motion-group override fired (a stranded object), and
2. the composite placed every elected layer within the arc's own refinement
   scale — `align._REFINE_MAX_FRACTION`, 1.5% of the frame diagonal.

The scale is borrowed rather than invented: that constant is already this
project's statement of how large a per-region displacement can be before it stops
being a refinement of the global warp and becomes re-registration. Two frames are
the wrong instrument for re-registration — 2.7x the shift sensitivity of an
N-frame fusion (H1), and a refusal with only one other source to fall back to.
The separation is not marginal: 2.1 and 6.9 px served, 19.2 px declined.

Rejected alternatives for the second half:
* **A no-reference quality A/B between the two outputs.** F81a: no-reference
  metrics cannot adjudicate an alignment change. A coin toss wearing a number.
* **The concession statistic** — the contrast-weighted share of the frame where
  the composite withdraws an elected member and leaves nothing sharp behind.
  Measures the defect directly and is honest, but reads factory 7.2%, kitchen
  5.0%, IMG-46 5.3%, large-motion 26.3%: the separation is 5x where the
  displacement rule's is 3-9x, and its own focal tolerance is a second threshold
  that moves with sweep length. Kept as the mechanism explanation, not the gate.

## I3. Per-scene validation (small data only, DEVSTYLE §13)

* **factory / zero-motion / small-motion: BYTE-IDENTICAL** to the pre-route
  pipeline output (`np.array_equal`, not a metric). Large-motion too, since a
  declined composite is discarded.
* **kitchen: routed, and the F108 acceptance holds on the shipped pipeline's own
  output** — flank mean |Δ| **2.24**, max **11**, **0.00%** over 12, against the
  shipped path's 6.69 / 67 / 17.67%. The bar was mean ≤ 2.2: missed by 0.04, and
  the cause is measured, not hand-waved — `normalize_exposure` applies a ±1.5%
  per-frame gain to the members (the reference's own gain is exactly 1.0), so the
  same composite reads 2.16 from raw sources and 2.24 from normalized ones, while
  the shipped baseline moves 5.98 → 6.69 for the same reason. The structural half
  of the bar (0.00% over 12) passes exactly.
* **IMG-46: routed** (1 group overridden, 6.9 px of 20.2). The bottle label is
  sharp and clean in both paths — `out/img46/ROUTE_bottle.png`, reference |
  shipped | routed — with no doubling or ghosting in the routed crop. Call it
  neutral-to-slightly-crisper by eye; nothing here justifies a stronger claim.
* **large-motion: declined**, evidence above.

## I4. Deliberate integration choices

* **`enhance` is skipped on the routed path** and says so in the log. F56 licences
  it for the fused output of frames the caller can point at; a stitched
  per-region composite is not that, and the licence does not transfer (§6).
* **Exposure normalization** is applied to the frames the route registers itself,
  because a per-frame channel gain commutes with the warp.
* **`--method` other than `perband` declines the route**, since the two-frame
  path fuses per-band internally and would silently ignore the request.
* **The routed output keeps its own crop**, which differs from the shipped crop by
  a pixel or two; `--depth-out`/`--boundary-out` still describe the shipped
  aligned stack.

## I5. What remains

1. **The cost of a declined composite.** Large-motion builds the two-frame render
   (~4 s) and throws it away. The displacement is knowable from the layer fits
   alone, so the decision could be taken before rendering — worth doing if the
   route ever runs on big stacks.
2. **The second half of the rule has one real scene on each side** (kitchen and
   IMG-46 served, large-motion declined) plus two synthetic ones in the tests.
   That is thin, and the honest statement is that the SCALE is borrowed from a
   validated constant rather than fitted to these three points.
3. **The large-motion mechanism is a repairable defect, not a law.** The elected
   member was correct and the refusal withdrew it; F82's ribbon is being applied
   at full width to a pair whose other member cannot cover it. A pair-aware
   refusal that prefers a defocused-but-present observation over a
   reference-defocused one might reclaim that scene — and would move the route's
   boundary, so it should be measured before the licence is tuned.
4. Items H7.2-H7.5 (the unverifiable layer, the degenerate single-frame regions,
   full-res provenance sensitivity, `--enhance`) are untouched by this pass.

---

# Round 3 — the precondition the focus contest never had, 2026-08-02

Fourth Opus round. The user inspected the ROUTED kitchen output region by region
and marked four defects. Code touched: `src/focusstack/twoframe.py` and
`tests/test_twoframe_route.py` (94 tests, was 92). `research/twoframe.py` was
deliberately NOT mirrored — see R6.

```
.venv/bin/python -m pytest -q tests/test_twoframe_route.py    # 9 tests, 2 new
```

## R1. Verdict first

**All four marked defects are one mechanism, and the architecture's own §2 claim
is what failed.** §2 said geometry and focus are co-diagnostic: "the layer a
frame gets wrong is, by construction, the layer it is defocused in, so the focus
contest discards it without being told to." That holds only if both members
observe the SAME SURFACE at the pixel. Where parallax has swung an occluder they
do not — one member sees the foreground, the other sees the background it
uncovered — and the contest is then comparing two different objects. It picks
the more TEXTURED one, which is not the nearer one.

The measurement that settles it, taken before anything was changed: in flaw 1's
box (the background pot rendering IN FRONT of the Lubriderm bottle) the pair's
own near-layer mask `dense[0]` covers **0.0%** of the box. The bottle's surface
there is smooth white; the pot 2 m behind it carries sharp print; frame 11 wins
the focus contest on every pixel, so the layer mask assigns the bottle's own
silhouette to the FAR layer and hands it to frame 11. Every evidence-driven gate
in the module — `_pair_refusal`, the depth step, the layer boundary — is
downstream of that mask, so all of them were blind together. **This is F108's
wall recurring one level in**: not "no local evidence about focus", but "no local
evidence about focus that means what the code assumes it means".

Two changes close it. Measured on the routed pipeline output, against the
reference frame, in the user's own boxes:

| user box | before mean / max | after mean / max | shipped path, same box |
|---|---:|---:|---:|
| 1 pot in front of the bottle | 23.08 / 131 | **5.96 / 61** | 15.13 / 140 |
| 2 second copy of the lid | 13.66 / 122 | **3.55 / 17** | 4.19 / 103 |
| 3 Coca-Cola right edge alias | 9.52 / 85 | **3.82 / 101** | 16.35 / 125 |
| 4 blurry alias of the rag | 11.62 / 132 | **6.10 / 127** | 21.05 / 149 |

and everywhere else:

| quantity | before | after | bar |
|---|---:|---:|---|
| analytic factory GT-SSIM (two-frame) | 0.971310 | **0.979453** | ≥ 0.9700 |
| shipped depth-bin path, identical crop | 0.972808 | 0.972808 | — |
| kitchen F108 flank mean / max / >12 | 2.24 / 11 / 0.00% | **0.76 / 9 / 0.00%** | ≤ 2.3, 0% |
| factory / zero-motion / small-motion | byte-identical | **byte-identical** | required |
| large-motion | declined (19.2 px > 14.0) | **declined (19.2 px > 14.0)** | required |

**The factory regression that blocked promotion since F109 is gone.** The
two-frame path scored 0.9658 (F109), 0.9713 (F110), and now **0.9795** — it
BEATS the shipped depth-bin path by +0.0066 on the analytic factory, on the
identical crop, having lost to it by 0.0070 two rounds ago. F110's "what is left
is the architecture" was right about the category and wrong about which part:
it was the FUSION, not the stitch or the degenerate regions.

## R2. Fix 1 — a pair must be fused coherently, not per band

`twoframe_stack` fused each pair with `fuse_perband(..., harden=0.5)`, commented
"as the shipped path does". The analogy is false and the manager's steer named it
exactly: in the shipped path every frame shares ONE geometry, so per-band soft
weights only ever mix two renderings of the same surface in the same place. Pair
members are misregistered BY DESIGN outside their own layer, so choosing which
member supplies a pixel is choosing between two GEOMETRIES — and F106 says a
geometric decision cannot be soft. Per-band soft weights leak the misplaced
member's coarse bands wherever the energy contrast is weak.

`fuse_perband`'s own docstring documents this failure for unstable N-frame stacks
and routes them to `fuse_coherent`; the guard is `len(images) > 2`, so pairs were
excluded from the protection they need MOST. `fuse_coherent` makes one shared
edge-aware decision, one-hots the winner (fine detail from exactly one source, so
no double edge can form) and still feathers coarse transitions through the
Gaussian weight pyramid. Both call sites now use it.

Measured alone, without the second fix:

| | flank | box 1 | box 2 | box 3 | box 4 | factory |
|---|---:|---:|---:|---:|---:|---:|
| before | 2.24 | 23.08 | 13.66 | 9.52 | 11.62 | 0.971310 |
| fuse_coherent only | **1.01** | 41.47 | 13.35 | **6.12** | 15.76 | 0.971310 |

Read honestly, this is a **partial** result and the steer's prediction that it
would close flaws 2/3/4 held for exactly one of them:

* **flaw 3 closed** — that alias really was per-band soft mixing (9.52 → 6.12,
  and the doubled edge is gone by eye).
* **the flank halved**, which is F108's own acceptance region.
* **flaw 1 got WORSE** (23.08 → 41.47), as predicted: one-hotting makes the
  contest's wrong winner win outright, so the pot renders crisply instead of
  faintly.
* **flaws 2 and 4 barely moved.** They are not soft mixing; 2 is a disocclusion
  ghost and 4 is a doubled silhouette, and a one-hot decision reproduces both.
* **the analytic factory is BYTE-IDENTICAL** with either fusion. Its two planes
  are textured everywhere, so the contest is unanimous and one-hotting changes
  nothing — which is precisely why two rounds of factory scoring never saw this.

## R3. Fix 2 — the same-surface precondition (`same_surface`)

The steer's own agreement clause, implemented at the scope the measurement
demanded. It proposed using `layer_masks` as an ownership prior and allowing the
focus contest only in the contested ribbon where the members agree geometrically.
The ownership half cannot work here and R1 says why: **the layer mask is built
from the same broken contest** (`dense[0]` = 0.0% inside flaw 1's box), so a
prior taken from it hands flaw 1 to the wrong member and the box is nowhere near
a "ribbon". The AGREEMENT half is the load-bearing part, and it must apply
everywhere, not only in a ribbon.

The test needs no texture, and PLAYBOOK §0 supplies it in one line: **defocus is
a low-pass, always.** Two observations of one surface must therefore agree once
both are low-passed past their own defocus, however textureless the surface is;
two different surfaces do not, and their disagreement is the contrast between
them. So each member is admitted only where its low-passed appearance agrees with
the **unwarped reference's** — the reference being the one frame that is the
authority on what is VISIBLE in the composite's own geometry. Trinary and hard,
never ramped (F106).

What "agree" may mean is set by measurement, and every term is borrowed:

* a residual displacement up to `GATE_TOL` (1.5 px) is the module's own statement
  of a fit that VERIFIED, so a disagreement that a `GATE_TOL` shift explains is
  subtracted as `tol · |∇|`. This is F106's unexplained-motion rule asked per
  pixel — refuse what no motion the geometry admits can account for. It is also
  what stops the test firing along every correctly-registered edge in the frame.
* `normalize_exposure` leaves a multiplicative residual; measured on this sweep
  the largest is 1.85% (frame 4 reads 0.9815), so 2% of the local level is free.
* sensor noise survives the low-pass at p99 = 0.6–0.9 levels, so 1 level is free.

A refused member falls back to the pair's other member, and where both are
refused, to the unwarped reference — §5.3's measured fallback, unchanged. The
reference member agrees with itself everywhere, so a pair containing the
reference always has a source. **Degenerate one-frame regions now get the same
treatment**: a lone elected frame that disagrees with the reference is covered by
the reference there, carried as one `usable` entry more than the pair has frames,
which is the convention `twoframe_fullres` already reads.

### KAT before belief (§12.1), `tests/test_twoframe_route.py`

| question | answer |
|---|---|
| does DISK defocus trip it? (the premise) | radius 1/2/4/6 px → agreement 1.000 / 1.000 / 0.999 / 0.983 |
| …at larger radii? | 8 px → 0.931, 12 px → 0.759 — **it does**, see R5 |
| does a shift the geometry tolerates trip it? | 0.5 / 1.0 / 1.5 px → 1.000; 2 px → 0.981; 4 px → 0.800 |
| does the exposure residual trip it? | ×1.015 / ×1.02 → 1.000; ×1.05 → 0.455 |
| does a MOVED occluder trip it? | 4/8/20 px → agreement 0.000 in the vacated and newly-covered strips, 0.987–0.984 everywhere else |

The knees sit exactly where they were designed to: at `GATE_TOL` for
displacement, at the measured gain for exposure, and at zero for a real occlusion
swap.

## R4. `SURFACE_SIGMA` — two scenes want opposite values, and it is logged as such

The low-pass scale is the one free number, and it behaves exactly as DEVSTYLE
§12.3 warns:

| sigma | factory GT-SSIM | flank | box 1 | box 2 | box 3 | box 4 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | **0.981650** | 0.81 | 11.98 | 4.45 | 4.85 | 8.56 |
| 3 | 0.980871 | 0.78 | 9.84 | 3.95 | 4.15 | 7.52 |
| **4 (shipped)** | **0.979453** | **0.76** | **5.96** | **3.55** | **3.82** | **6.10** |
| 6 | 0.976212 | 0.80 | 4.53 | 2.83 | 3.15 | 3.95 |
| 8 | 0.974249 | 0.85 | 3.66 | 2.28 | 2.56 | 2.60 |

The factory (which has GROUND TRUTH) wants it small; the kitchen boxes want it
large. The box metric is measured against the reference frame and therefore
rewards refusal tautologically, so it is NOT a fair arbiter of this number and
was not used as one — 4.0 is the smallest value that clears every acceptance bar,
and the KAT says what it buys: a disk defocus of radius ≤ 6 px does not trip the
test. Every value in the table beats the pre-round state on every column, so the
choice is not load-bearing for the result; it is load-bearing for how much
sharpness is given up, and that is the open item in R5.

## R5. Honest costs and what is NOT fixed

1. **The physical invariant behind `SURFACE_SIGMA` was not built.** The low-pass
   must exceed the RESIDUAL DEFOCUS DIFFERENCE between the two frames, and that
   is a per-pixel quantity the module already has the ingredients for: `peak` is
   the focal frame per pixel and the arc's validated blur proxy is distance from
   the object's focal frame (PLAYBOOK §0c forbids contrast-based blur
   estimation). A spatially-varying low-pass keyed on `|frame − peak|` would
   remove the threshold. Not attempted — economy, and the fixed value clears
   every bar.
2. **Refusal roughly doubles**, per pair over its own area: (6,8) 0.12 → 0.16,
   (6,11) 0.19 → 0.31, (11,11) 0.00 → 0.41, (9,11) 0.15 → 0.30, (4,6) 0.00 →
   0.14. The composite is now within 2 levels of the reference over 67.7% of the
   frame, against 20.7% before. That reads alarming and is not: mean focus energy
   over the frame goes **15.03 → 15.56** (reference 13.08), i.e. the output is
   SHARPER than before while agreeing with the reference far more often. The
   refusals land where the member was wrong.
3. **A faint pale sliver survives at the Lubriderm's left silhouette in flaw 1**
   (`out/inspect/ROUND3_flaw1.png`, a few px wide). Small, at a silhouette, and
   unexplained. Logged, not chased.
4. **Box 3's max rises 85 → 101** while its mean halves. Seven pixels in one 3×3
   cluster on the Coke bottle's cap ring, where the composite is SHARPER than the
   reference; p99 in that box is 38. Not a new artifact by eye.
5. **Box 4's reference is defocused**, so its "vs reference" number is not a
   clean bar and is reported with the shipped path as a second comparator
   (21.05). Boxes 1 and 2 have a locally sharp reference and are fair bars;
   box 3's reference is sharp on the bottle.
6. **`out/inspect/img46_routed.png` is now STALE.** IMG-46's frames are not in
   the repo, so that inspection layer could not be regenerated; only the kitchen
   layer was. The round-3 change certainly moves IMG-46's routed output.
7. **The 2-frame vs N-frame fusion question is now open again.** `fuse_coherent`
   was chosen for pairs on mechanism, and it is right; but `fuse_perband`'s
   `len(images) > 2` guard is still there in `fusion.py` and still excludes every
   2-image caller from the protection. Not touched — out of scope this round, and
   it is a shipped-path change.

## R6. NEGATIVE deliverables and deliberate non-actions — this round

1. **`fuse_coherent` alone as the whole fix.** REJECTED as sufficient, with
   numbers (R2): it closes flaw 3 and halves the flank, makes flaw 1 nearly twice
   as bad, and leaves flaws 2 and 4 where they were. Kept, because it is right
   and because the precondition needs a one-hot decision to be worth making.
2. **An ownership prior taken from `layer_masks`.** REJECTED on measurement:
   `dense[0]` covers 0.0% of flaw 1's box, so the prior is built from the very
   contest that failed. The layer masks remain what they always were — support
   for the fits and the seed for F82's ribbon — and are not promoted to
   ownership.
3. **Restricting the agreement test to the layer boundary ribbon.** REJECTED:
   flaw 1's box is deep inside layer 1's mask, not in any ribbon, so a
   ribbon-scoped test cannot fire there at all.
4. **Choosing `SURFACE_SIGMA` by the four boxes.** REJECTED as an instrument
   (§12.2): the boxes are scored against the reference, so "refuse everything"
   scores perfectly. The factory's ground truth is the arbiter that cannot be
   gamed that way, and it wants the opposite; 4.0 is the smallest value clearing
   all bars.
5. **Mirroring the fix into `research/twoframe.py`.** NOT DONE, deliberately.
   That module is the F109/F110 ablation harness and the recorded ladders in this
   note must stay reproducible from it; changing its fusion would silently
   invalidate every row above. The runtime module is now ahead of it, and any
   future ablation must be re-ported rather than assumed.
6. **Raising the refusal fallback question again** (does a refused-everywhere
   pair want a third frame?). NOT ATTEMPTED; H5's answer stands.

---

# Round 4 — the runtime retirements, 2026-08-02

Fifth Opus round. Code touched: `src/focusstack/twoframe.py`,
`tests/test_twoframe_route.py` (96 tests, was 94). Retirement 3
(licence-before-render) was NOT attempted — the session was time-boxed and the
manager ranked it last; it remains open exactly as I5.1 states it.

## R4.1 Retirement 1 — `SURFACE_SIGMA` retired into physics, and the half it did
## not cover

Ported F115's cross-convolution faithfully first: `m (x) disk(R_r)` against
`r (x) disk(R_m)`, radii `c*|k - peak|` with `c = 1.161`. The port reproduces
`research/scene_model.py kat` to three decimals (defocus 1/2/4/6/8/12 all 1.000
against the global sigma's 0.931/0.759 at 8/12; shift 2.0 -> 0.986, 4.0 -> 0.691;
occluder 0.010/0.003/0.002) — that is the port's own known-answer test.

**And faithful was not enough.** On the routed kitchen box 1 went 6.14/61 ->
10.42/156 with its focus energy RISING 41.8 -> 56.1: F112's defect returning
crisper. Diagnosis, measured: `SURFACE_SIGMA` had two jobs.

1. absorbing the residual defocus DIFFERENCE — retired exactly by the physics;
2. POOLING the decision. A per-pixel level agreement is not evidence about a
   SURFACE — a textured intruder crosses the occluded surface's level at
   scattered pixels, and matching the defocus exactly makes the test see those
   coincidences (agreement inside box 1 rose 8.9% -> 14.3% for the member that
   renders the pot in front of the bottle). F108's wall, again.

Job 2's window is borrowed, not chosen: the test polices the FOCUS CONTEST, and
the contest is decided on energies pooled by `content_aware_energies`'s own
`smooth_ksize`, so a gate may not be finer-grained than the decision it polices.
Read from that function's signature so the two cannot drift.

| variant | factory GT-SSIM | box 1 | box 2 | box 3 | box 4 | flank |
|---|---:|---:|---:|---:|---:|---:|
| recorded (global sigma=4) | 0.979453 | 6.14/61 | 3.49/17 | 3.31/101 | 6.23/127 | 1.114 / 0.57% |
| cross-convolution only | 0.981607 | 10.42/156 | 3.81/19 | 4.16/84 | 6.62/122 | 1.275 / 0.90% |
| pooling only (sigma=4) | 0.981893 | 1.81/5 | 2.10/12 | 1.57/86 | 1.60/94 | 0.845 / 0.02% |
| **both (shipped)** | **0.984385** | **1.20/2** | **2.04/13** | **1.19/19** | **1.08/4** | **0.897 / 0.01%** |

Both halves earn their place: pooling alone leaves box 3 and box 4 maxima at 86
and 94 against 19 and 4; cross-convolution alone fails as above. Quorum measured,
not assumed: majority pooling reads factory 0.978678 (below the recorded bar) and
opening leaves box 2 at 3.58/20 over its 3.49/17. Window 5/7/9/11 reads
0.984209 / 0.984312 / 0.984385 / 0.984376 — a plateau, not a knee. `SIGMA0`
0.5/1.0/2.0 reads 0.984518 / 0.984385 / 0.983964, not load-bearing.

**The cost, measured and shown.** Whole-frame mean focus energy 48.08 -> 43.46
(reference frame 42.69): the composite refuses more and is less sharpened. By
eye (`out/certify/C_retirement1.png`) the Lubriderm label is untouched and the
flank is unchanged, while box 1's pot and box 4's rag lose texture toward the
reference. The GT-free arbiter agrees with the change: the certifier reads the
new base composite at **7.8911** levels against the recorded 9.5242.

## R4.2 Retirement 2 — the pair-aware refusal, and what the ground truth said

F111's designated repair, implemented as a trinary preference in the two-member
fallback chain: sharp member, then a PRESENT member the appearance evidence
licenses, then the reference. It is licensed by the retirement above —
`align._occlusion_mask` says outright that "photometric agreement cannot be used
in a focus stack, [since] frames legitimately disagree wherever defocus
differs", and that sentence is exactly what the cross-convolution retires.

Large-motion, pair (0,3), which owns 94.6% of the playing-card box: F82 withdraws
the correctly-fitted member 0 over 99.7% of the frame, so the box came back
reference-defocused. It now keeps **47.4%** of the box, focus energy **31.6 ->
74.2** (reference 31.6, sharp source frame 0 152.4), review crops in
`out/certify/C_largemotion.png`.

**Two restrictions, both forced by the factory's ground truth.** Unrestricted,
the preference costs the factory 0.984385 -> 0.981380, and on the pixels it
claims the ground truth says its content is WORSE than the reference fallback it
replaced (5.23 vs 4.37 levels) — the research KAT's own recorded limit arriving
("an appearance test cannot separate two surfaces that look the same"). So the
evidence must DISCRIMINATE (if both members read same-surface it has not answered
F82's question) and the member must be MODELLED SHARPER than the reference at the
pixel. With both, the factory reads **0.984455** (+0.00007 — the ground truth no
longer objects) and the box keeps 47.4% instead of 64.6%.

## R4.3 NEGATIVE deliverables

1. **A per-scene `c`.** The shipped 1.161 is the factory's. On large-motion it
   over-blurs the member by ~2x (`R_ref` 7.05 px where the picture says ~3), and
   the kitchen's own regression is 0.684. The kitchen/factory bars are insensitive
   across 0.4-1.161 (factory 0.984595 / 0.984692 / 0.984385) and collapse at 2.0.
   `c` does not transfer, and nothing in the runtime measures it.
2. **Cleaning F82's speckled seed.** The pair's layer boundary covers 31.9% of the
   large-motion frame (a clean split would be 1-2%), so it looked like the cause
   of the blanket refusal. Opening/closing it at the pooling window moves member
   0's usable share 7.5% -> 8.1% and nothing else: `depth_step` (11.1%) plus a
   19 px differential is enough on its own. Hypothesis rejected.
3. **A displaced copy of the SAME fine texture** reads 0.61 agreement at 12 px —
   the instrument's recorded limit, and the reason F82's geometric check is
   restricted rather than removed.
