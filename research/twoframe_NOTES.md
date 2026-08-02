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
