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
