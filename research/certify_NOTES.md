# Forward-render consistency certifier — round A of the scene-model second pass

Instrument only. No reconstruction, no runtime change. `research/forward_certify.py`
is new; nothing else in the repo was touched. Evidence images are in `out/certify/`
(gitignored).

```
.venv/bin/python research/forward_certify.py kat1     # renderer vs the factory
.venv/bin/python research/forward_certify.py kat2     # blur estimator vs known sigma
.venv/bin/python research/forward_certify.py kat3     # GT ranking + sharpness bias
.venv/bin/python research/forward_certify.py floor    # attribute the model-error floor
.venv/bin/python research/forward_certify.py kat4     # known kitchen defects
.venv/bin/python research/forward_certify.py ledger   # what each piece of the model buys
```

---

## 1. Verdict first

**Three of the four acceptance tests pass; the fourth passes in two of its three
clauses and fails the third, and the failure is diagnosed rather than argued
away.**

| KAT | question | verdict |
|---|---|---|
| 1 | does the renderer reproduce the factory from true parameters? | **PASS** — 0.42 levels MAE, 0.12 with the factory's own uint8 truncation |
| 2 | does the defocus estimator recover a known radius? | **PASS** — 100% exact on every rung, including from a real composite |
| 3 | does the score rank two composites as ground truth does? | **PASS** — and ground truth itself scores best of all four candidates |
| 4a | does the known pale sliver light up? | **PASS** — rank 3/16 absolute, 10/13 differential |
| 4b | does the verified-clean flank stay quiet? | **PASS** — 0 spurious clusters at all nine detector settings |
| 4c | does the F112 knob light up? | **FAIL** — elevated 2.3x but never clusters in the top 20 |

**The one number that should shape round B.** On the analytic factory, where truth
exists, the forward model's total error is 3.92 levels and it decomposes as:

```
renderer + PSF family + exposure   0.442      <- the instrument itself, ~quantization
+ pass-1 MOTION estimation        +0.812
+ pass-1 LAYER SEGMENTATION       +2.925      <- 3.6x the motion term
= full model floor                 3.920
```

Pass-1's per-layer *motions* are good enough to forward-render with. Pass-1's layer
*masks* are not, and they are the reason the certifier's real-scene sensitivity is
what it is. Round B should not assume the focus-contest masks are a scene
decomposition; they are a fusion selector that happens to be shaped like one.

---

## 2. What the instrument is, and why it evades F81a

Pass 1 produces a composite in the reference frame's geometry plus, for every
frame, how each layer moved and where the focus stepped. Read forward instead of
backward, that is a renderer:

```
composite (reference geometry)
  -> per (frame, layer) rigid transform        [pass-1 motions]
  -> per (frame, layer) disk defocus           [pass-1 focal ladder]
  -> painter's composite in depth order        [pass-1 layer ordering]
  == a prediction of frame k, in frame k's own coordinates
```

The comparison is against the **raw frame in its own geometry**. That is the whole
point. PLAYBOOK §0 rules no-reference metrics unusable for an alignment change
"because they score against sources that alignment itself moves" (F81a). A raw
frame's own coordinates are the one thing in this pipeline that no stage can move,
so a residual there is a statement about the model and the composite and never
about the registration used to score them. The self-consistency check confirms the
plumbing carries that property: the reference frame, certified as a composite
against the reference frame, reads **0.0000 levels** on both scenes.

### Trinary honesty, structurally rather than by rule

The renderer carries an `unknown` channel through *exactly* the same composite
arithmetic as the image. A layer's appearance is known only where it is visible; a
defocus kernel that reaches behind an occluder, off the composite's crop, or out of
frame therefore mixes in content nothing recorded, and the pixel is **boundary**,
not certified. Clipped pixels are **excluded** — outside the model's linear range,
so their residual measures the sensor. Three counts, reported separately, no ramp
(F106).

The distinction that makes this work is two masks per layer, not one:

* `mask` — where the layer is **visible**, i.e. where its appearance is observed;
* `extent` — where the surface **exists**, visible or not.

Collapsing them was a real bug in the first build, caught by KAT-1b: the backdrop
layer claimed to be observed everywhere including behind the foreground, so the
renderer confidently certified disocclusion strips it had invented by nearest-fill.
Frames 0 and 1 read p99 34.4 and 28.3; with the two masks separated they read
**1.08 and 1.04**. An instrument that reports a fabrication as certified data is
worse than no instrument.

---

## 3. KAT-1 — the renderer, against the factory it was modelled on

`parallax_gen.py` *is* a forward renderer, so its own constants are the known
answer: two planes, lateral shifts in a 4.57:1 ratio, disk radii
`round(|k - focus| * 1.15)`, a matte blurred with its own layer. The truth model is
rebuilt inside `forward_certify` (the factory is read-only and `build_stack`
returns only frames and the reference-viewpoint truth).

**Rung (a) — padded canvas, nothing unobserved.**

| frame | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| MAE (levels) | 0.471 | 0.401 | 0.446 | 0.414 | 0.329 | 0.443 |
| max | 1.4 | 1.2 | 1.4 | 1.0 | 1.5 | 1.4 |
| SSIM | 0.99976 | 0.99968 | 0.99974 | **1.000000** | 0.99948 | 0.99975 |

Mean MAE **0.4173** levels, max deviation 1.5 levels anywhere in the sweep. The gap
is not waved at: `parallax_gen` writes frames with `.astype(np.uint8)`, which
*truncates*, and warps uint8 layers so each sample is rounded before the blur. Apply
the same truncation to the prediction and the mean MAE falls to **0.1199** levels —
i.e. the renderer and the factory agree to within the factory's own quantization.
The reference frame renders at SSIM 1.000000 exactly.

**Rung (b) — reference geometry, only what a composite could know.** The far plane
is observable only where the near plane does not cover it, and nothing outside the
reference field of view exists.

| frame | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| MAE on certified | 0.712 | 0.593 | 0.687 | 0.643 | 0.482 | 0.680 |
| p99 | 1.08 | 1.04 | 1.10 | 1.00 | 1.03 | 1.08 |
| certified | 90.9% | 94.7% | 95.7% | 96.8% | 95.9% | 92.2% |
| boundary | 9.2% | 5.2% | 4.3% | 3.2% | 4.1% | 7.8% |

p99 never exceeds 1.10 levels — the quantization floor — and the cost of honesty is
3.2–9.2% of the frame declared boundary, largest in the frames whose near plane has
swung furthest. That band is the disocclusion, correctly refused rather than
guessed.

---

## 4. KAT-2 — the defocus estimator

PLAYBOOK §0c closes contrast-over-gradient blur estimation: it saturates by 2 px
and reads texture. That negative is about a measurement taken on the blurred image
alone. This is a different question — *given the sharp appearance the model already
holds, which radius makes the render match the observation* — so it is a different
instrument, and it got its own KAT before being believed. Disk radii are integers by
construction, so the integer grid is the exact parameter space.

| rung | exact hits | mean signed error | max abs error | best-fit MAE |
|---|---:|---:|---:|---:|
| A. true appearance, disk PSF, gain fitted | **100%** | +0.00 px | 0 px | 0.410 |
| B. true appearance, disk PSF, no gain | **100%** | +0.00 px | 0 px | 0.539 |
| C. true appearance, **Gaussian** PSF | **100%** | +0.00 px | 0 px | 0.764 |
| D. **real** appearance (routed two-frame composite) | **100%** | +0.00 px | 0 px | 1.661 |
| E. real appearance **+3 px geometry error** on the near plane | 50% | +0.58 px | 2 px | **6.416** |

Two results matter more than the headline.

**Rung C: the argmin survives a wrong PSF family; the residual does not.** A
Gaussian recovers the same integer radius as the disk that generated the data, and
pays 0.764 levels against 0.410. So a matched argmin is *not* evidence that the PSF
family is right — the family shows up in the floor, never in the parameter. This is
the same shape as the arc's standing warning that a confident number can agree with
nothing.

**Rung E: blur partially absorbs misregistration, but cannot hide it.** Three pixels
of injected geometry error inflate the affected layer's radius by 1–2 px — so the
radius search *is* a channel through which geometry error leaks into the blur
estimate, and any future use of these radii as a depth cue must account for it. But
the best-fit MAE goes 1.661 → 6.416, a factor of 3.9. The absorption is weak and
loud; a candidate cannot buy silence with it. That is what licenses re-fitting radii
per candidate in KAT-3.

---

## 5. KAT-3 — the ranking, and one bug the KAT caught

Same stack, same crop `(4, 1, 553, 419)`, same scene model, four candidates.

| candidate | GT-SSIM | certifier (levels) | p99 | certified |
|---|---:|---:|---:|---:|
| **ground truth composite** | — | **4.1539** | 40.30 | 78.65% |
| two-frame route | **0.979453** | **5.9629** | 40.62 | 78.65% |
| shipped depth-bin | 0.972808 | 6.5118 | 41.39 | 78.65% |
| null (reference frame) | — | 4.6977 | 36.11 | 82.25% |

**Ground truth prefers two-frame; the certifier prefers two-frame. AGREE.**

**The sharpness-bias check is the rung that was not asked for and should have
been.** The certifier renders by *blurring* a candidate, so at an edge a sub-pixel
geometry error costs a sharp candidate more than a soft one — which would be F81a's
trap reappearing through the model instead of through the metric. The ground-truth
composite is correct by construction, so it must win outright. It does: **4.1539,
best of all four**, beating the blurred reference by 0.54 and the best real
composite by 1.81. There is no sharpness penalty on this scene.

Note that the null composite (4.6977) *beats both real composites*. This is not the
instrument malfunctioning — the reference frame is a real, correctly-registered
observation, and its only fault is being defocused where the layers are not at its
focal plane. Both real composites carry registration and fusion error the reference
does not. It does mean the null cannot be used as a pass/fail bar, only as a scale.

### The bug KAT-3 caught before anything was believed

The first run **ranked the composites backwards** (shipped 7.463, two-frame 7.593)
on a null floor of 6.345. The cause was in the model, not the ranking: the geometry
was `global affine ∘ per-layer translation`, taken from pass 1 as-is. The factory
has zero breathing, so every non-translational term in that affine is over-fit —
`twoframe._rigidify`'s own docstring says so and measures it ("frame 1's affine
reads scale 0.9954, spreading ±1.3 px of sampling error across the frame even when
the layer's own shift is exact"). Measured here with the *true* layer masks, the
composed affine's implied displacement at the far plane ran from +6.9 px at x=0 to
+2.0 px at x=560 against a true +2.1 px everywhere.

The fix is not new machinery. Pass 1 already builds exactly this menu — the composed
affine and its rigid collapse — and chooses per layer by verification. The certifier
cannot borrow pass 1's verdict, because pass 1 only ever renders the two or three
frames it elected while this renders all of them, so it re-decides with the arbiter
it does have: which candidate makes the forward render match. On the factory it
chooses **rigid 15/18**, which is exactly what F110 measured pass 1 itself choosing
there. The floor fell 6.345 → 4.698 and the ranking corrected.

Geometry is chosen **once, on the null appearance, and frozen** before any candidate
is scored — a candidate must not pick the geometry that flatters it. Radii and gains
are re-fitted per candidate, on the licence rung E establishes.

---

## 6. The model-error floor, attributed (`floor`)

A floor is not a finding until it has an owner. Each rung holds the appearance
fixed at the true all-in-focus scene and changes one piece of the model.

| rung | score | p99 | certified |
|---|---:|---:|---:|
| 1. true masks + true motions (the renderer's own floor) | 0.4418 | 0.66 | 94.2% |
| 2. true masks + pass-1 estimated motions | 1.2540 | 3.58 | 94.0% |
| 3. pass-1 masks + true motions | 3.3670 | 47.59 | 81.2% |
| 4. pass-1 masks + pass-1 motions (**full model floor**) | 3.9200 | 37.64 | 79.2% |
| 5. rung 4 with the null composite | 4.6335 | 33.76 | 83.0% |

**The layer segmentation costs +2.93 levels; the motion estimation costs +0.81.**
The null candidate's own defocus adds only +0.71, so **85% of the null bound is the
model**, not the candidate — which corrects the charter's framing of the null test:
it bounds model error from above but is dominated by it, and on this scene the
dominant term inside it is segmentation.

Rung 3's p99 of 47.6 is the sharper statement: segmentation error is not a diffuse
haze, it is *localized and large*, sitting exactly where a real defect would. That
is why the certifier reports a **differential** map (candidate minus null) as well
as an absolute one. Model error is a property of the model, identical for every
candidate scored through it, so it cancels in the difference and stops masquerading
as a fault in the composite.

The subtraction is not tautological in the direction that would matter. The null is
a single defocused frame and the radius search may only *add* blur, so a null render
cannot match a frame in which a layer is sharp. A composite that is sharper than the
reference **and correct** scores below the null; sharper and wrong scores above it.
Agreeing with the reference buys nothing — which is the exact failure that made
F112/R4 reject the four-box metric as an arbiter.

---

## 7. KAT-4 — the kitchen, and where it falls short

12 frames, 774×518, routed composite from `twoframe_stack(normalize_exposure(src))`,
crop `(15, 8, 742, 510)`, certified against the **raw** jpgs so the exposure
residual is a nuisance the model must absorb rather than a difference quietly
removed first. Model: 10 layers in 9 geometry groups. Self-consistency **0.0000**.
Runtime 49 s.

```
routed composite : 10.9001 levels   p99 58.7   certified 65.6%  boundary 24.8%  excluded 9.6%
null (reference) :  9.4046 levels   p99 56.6
DIFFERENTIAL     :  +1.3832 levels  p99 +14.67
```

| region (original coords) | abs mean | abs max | **diff mean** | diff max | px |
|---|---:|---:|---:|---:|---:|
| F112 knob, x659–669 y243–313 | 9.553 | 28.17 | **3.208** | 23.55 | 672 |
| pale sliver, bottle's left silhouette | 43.575 | 104.56 | 0.841 | 17.16 | 689 |
| clean flank, x560–670 y240–420 | 7.151 | 100.32 | 0.324 | 23.55 | 18514 |
| **F108 flank MASK (canonical instrument)** | 2.744 | 14.45 | **0.318** | 1.87 | 3202 |
| whole certified frame | 10.900 | 173.84 | 1.386 | 89.99 | 325158 |

**Clause b — the flank is quiet, and this is the strongest result of the four.**
On F108's canonical flank mask (bright, low-variance, where the reference is the
local truth), the differential reads **0.318 against a frame mean of 1.386** — 4.4x
below average — and its maximum over 3202 pixels is 1.87 levels. No cluster appears
inside the flank box at **any** of the nine detector settings tried. The region
F108 spent five failed attempts on, and F109–F112 finally cleaned, is independently
confirmed clean by an instrument that never looks at the reference frame for its
verdict.

**Clause a — the pale sliver is found.** Rank 3 of 16 in the absolute map
(box `(491,145,509,169)`, peak 104.6) and rank 10 of 13 in the differential. This is
F112/R5's open item 3, "a faint pale sliver survives at the Lubriderm's left
silhouette… small, at a silhouette, and unexplained", located by a method that had
no knowledge of it.

**Clause c — the knob is not found, and here is why.** Detection limit, measured
rather than tuned:

| percentile | min area | clusters | KNOB rank | SLIVER rank | in clean flank |
|---:|---:|---:|---:|---:|---:|
| 99.5 | 25 | 13 | – | 10 | 0 |
| 99.5 | 3 | 93 | 70 | 10 | 0 |
| 99.0 | 25 | 35 | **25** | 12 | 0 |
| 98.0 | 25 | 48 | 22 | 14 | 0 |

The knob never reaches the top 20. Three facts, in order of how much they explain:

1. **In absolute terms the knob is a quiet region** — 9.553 against a frame mean of
   10.900. The forward model explains it *better* than average. It is elevated only
   in the differential (3.208 vs 1.386, 2.3x), i.e. the composite explains the sweep
   worse there than the reference does. So the defect is real and correctly signed,
   and only the differential map sees it at all.
2. **The signal is ~7x under the floor.** A 3.2-level differential on a scene whose
   null reads 9.4 is not a detection. The floor ladder says that floor is mostly the
   pass-1 layer segmentation, so this is the same finding as §6 arriving at the
   kitchen.
3. **The recorded box is imprecise and dilutes what signal there is.** F112 describes
   "one 30×70 px dark background knob at (659–669, 243–313)" — a 10×70 box for a
   30×70 object. Inspected at 4x (the knob is a stove knob on a dark panel), the box
   covers the knob's right sliver and about 50 rows of clean countertop below it. The
   knob's hot pixels form a thin ring that a 5×5 close plus a 25-px area floor does
   not survive.

Not offered as an excuse: **KAT-4 fails clause c.** The instrument's real-scene
sensitivity floor is around 10 levels absolute / a few levels differential, and the
F112 knob is under it.

**What the top differential clusters actually are.** Looked at, not inferred
(`out/certify/kat4_kitchen_differential.png`): the Sprite bottle's right silhouette,
the Hot Cocoa can's right silhouette, the cat figurine's outline, the Lubriderm's
left edge (the sliver). All are object silhouettes at depth discontinuities — the
family F106/F108/F112 have been fighting throughout — so they are credible rather
than obviously spurious, but **none is independently verified** and this round does
not claim them. The absolute map by contrast
(`out/certify/kat4_kitchen_absolute.png`) is a broad carpet over all textured
content: it is dominated by model error and should not be used to hunt defects.

---

## 8. Crudeness ledger — what each piece of the model is worth (`ledger`, kitchen)

| configuration | score | p99 | certified |
|---|---:|---:|---:|
| routed two-frame composite (baseline) | 10.9001 | 58.72 | 65.6% |
| null: normalized reference frame | 9.4046 | 56.60 | 67.0% |
| baseline, certified vs **normalized** frames | 10.8850 | 58.44 | 65.7% |
| baseline, **no** per-layer exposure gain | 11.1207 | 59.08 | 65.6% |
| baseline, **Gaussian** PSF instead of disk | 11.0214 | 57.26 | 61.0% |
| baseline, **defocus disabled** (all radii 0) | 13.6918 | 70.71 | 82.6% |
| baseline, layers kept, **global affine only** | 11.0002 | 61.82 | 62.2% |
| baseline, **one layer + global affine** (no model at all) | 13.4059 | 72.68 | 87.1% |
| one layer + global affine, null candidate | 12.0982 | 71.69 | 87.2% |

Read against the baseline:

* **Defocus modelling is worth −2.79 levels** — the single largest term. The focal
  ladder is the most load-bearing pass-1 output this instrument consumes.
* **The layer decomposition is worth −2.51 levels**, and costs 21.5 points of
  certified coverage (87.1% → 65.6%). It buys accuracy with honesty about
  disocclusion, which is the right trade, but the price is steep.
* **Per-layer motion beyond the global affine is worth −0.10 levels.** On this
  scene the layer motions are almost worthless to the forward model. The factory
  ladder agrees in direction (+0.81 for motion vs +2.93 for segmentation) and this
  is the stronger version of it: after the global affine, what remains on the
  kitchen is not motion the certifier can use.
* **Exposure: −0.22 levels**, and certifying against normalized rather than raw
  frames moves the score by 0.015. The per-layer scalar gain absorbs
  `normalize_exposure`'s ±1.5–2% residual essentially exactly, so exposure is a
  solved nuisance here, not a limit.
* **PSF family: −0.12 levels** for disk over Gaussian, plus 4.6 points of coverage.
  Real, small, and consistent with KAT-2 rung C: the family shows up in the floor.
* The differential is +1.4955 with the full model and +1.3077 with no model at all,
  so the composite is separated from the reference either way.

---

## 9. Honest limits

1. **Real-scene sensitivity is ~10 levels absolute, a few levels differential.**
   Set by the model floor, which the factory attributes mostly to segmentation.
   Defects below it (the F112 knob) are invisible.
2. **Rigid-per-layer misses rotation and within-layer scale.** Not separately
   measured this round — the factory has no rotation and the kitchen's per-layer
   motion turned out to be worth 0.10 levels, so there was nothing to attribute it
   to. Named, not quantified.
3. **The geometry menu is five candidates chosen by forward fit.** That is real
   freedom. It is constrained by being frozen on the null appearance before any
   candidate is scored, and every option comes from pass 1 — but a menu chosen by
   the same residual the instrument reports is a circularity that a future round
   should close with an independent arbiter (pass 1's own edge-evidence gate is the
   obvious candidate).
4. **Layer ordering is by median focal peak, lower = nearer.** True on both scenes
   here and false on a far-to-near sweep. It is nearly non-load-bearing because
   ordering only matters where warped layers overlap, which is the boundary band the
   instrument refuses anyway — but it is an unguarded assumption.
5. **Certified coverage on the kitchen is 65.6%**, falling to 47–56% in the extreme
   frames. A third of the frame is not certified in an average frame. The instrument
   is honest about this and it is a real limit on what it can arbitrate.
6. **Nearest-fill extends each layer's appearance past its mask.** Bounded (it only
   places content some pixel of the same layer shows, F56) and fully accounted for
   by the unknown channel, but it is still a fabrication living inside the render.
7. **Two scenes.** One analytic factory with two planes and no rotation, one real
   sweep. Everything above is conditional on that.

---

## 10. NEGATIVE deliverables — tried and rejected, with numbers

1. **One mask per layer (visible == exists).** REJECTED, and it was a bug, not a
   simplification: the backdrop then claims to be observed behind the foreground and
   the renderer certifies invented disocclusion. Factory frames 0/1 p99 34.4/28.3
   → **1.08/1.04** once `mask` and `extent` were separated. This is the single most
   important line of code in the module.
2. **`global affine ∘ per-layer translation`, taken from pass 1 unmodified, as the
   only geometry.** REJECTED on measurement: it **ranks the two factory composites
   backwards** (shipped 7.463 vs two-frame 7.593 against a GT ranking of the
   reverse) on a floor of 6.345. The affine absorbs differential parallax as a
   spurious scale on a scene with zero breathing. Replaced by pass 1's own
   affine/rigid menu; the factory then chooses rigid 15/18 and the ranking corrects.
3. **Per-frame masked-ECC layer shifts, used as measured.** REJECTED as the sole
   source. On the kitchen they produce non-physical series — `pair(4,6)/L0` reads
   −18.83 px at frame 0 and −5.26 at frame 1, a 13.5 px jump between adjacent
   frames, while frames 2–10 of `pair(6,11)/L0` form a clean 3.4 px/frame ramp. A
   layer is only measurable near its own focal plane (F99). Replaced by a
   focal-weighted Theil-Sen slope propagated along the sweep (PLAYBOOK §0's own
   recipe; §12.6's linear model), kept **alongside** the raw measurement in the menu
   rather than instead of it.
4. **Grouping layers on the raw per-frame shifts.** REJECTED: with individually
   wrong measurements, a 0.75 px agreement tolerance merges nothing and yields one
   group per layer — 10 layers warped independently, tearing the frame open at every
   shared edge. Grouping on the propagated series gives 9 groups on the kitchen and
   3 on the factory.
5. **Fitting the geometry per candidate.** REJECTED as unfair before it was run: a
   candidate that picks the geometry which flatters it is not being certified.
   Geometry is frozen on the null appearance. Radii and gains are *not* frozen, and
   KAT-2 rung E is the measurement that licenses the asymmetry (absorption 1–2 px,
   residual penalty 3.9x).
6. **The absolute unexplained map as the primary output.** REJECTED for defect
   hunting: `out/certify/kat4_kitchen_absolute.png` is a broad carpet over all
   textured content, and the floor ladder says why (rung 3 p99 47.6 — segmentation
   error is localized and large). The differential is the primary map; the absolute
   one is kept because it is the only one that reports the model's own health.
7. **A yes/no "is the known defect in the top ten".** REJECTED as a threshold
   pretending to be a verdict. Replaced by the cluster **rank** and a detection-limit
   sweep, which is what turned KAT-4c from "no" into "rank 25/35 at the 99th
   percentile, ~7x under the floor".
8. **Tuning the cluster detector until the knob appeared.** NOT DONE. The sweep in
   §7 is characterization, reported in full including the settings that fail; the
   defaults were not moved to make a KAT pass.
9. **Comparing against normalized frames instead of raw ones.** Considered and
   rejected as the default (it removes a nuisance before measuring it), then
   measured anyway: 10.885 vs 10.900, a 0.015-level difference. The default stays
   raw, and the measurement is now the evidence that exposure is not a limit here.

---

## 11. What round B (reconstruction) should know

**Sufficient from pass 1:**

* **The focal ladder.** Worth −2.79 levels on the kitchen, the largest single term,
  and the radius search recovers a known radius exactly (100%) even from a real
  imperfect composite. Defocus is the part of the forward model that works.
* **The global affine.** Carries essentially all the usable motion on the kitchen.
* **Exposure.** A per-(frame, layer) scalar gain absorbs `normalize_exposure`'s
  residual to within 0.015 levels. Solved; do not build for it.
* **The affine/rigid geometry menu.** Already in pass 1, already correct, and the
  certifier reproduces pass 1's own per-scene split (factory rigid, kitchen affine)
  without being told.

**Too crude from pass 1:**

* **The layer masks, by a factor of 3.6 over everything else.** +2.93 of the
  factory's 3.92-level floor, with p99 47.6 — localized and large. They are the
  focus contest's per-pixel winner map, morphologically cleaned; they are a fusion
  *selector*, and F98 already says turning feature groups into pixel regions is the
  open problem. Round B's reconstruction assembles per-LAYER appearance, so it
  inherits this error directly and multiplied. **Fix the decomposition before
  building on it.**
* **Per-layer motion on real scenes.** Worth −0.10 levels beyond the global affine
  on the kitchen, and the raw per-frame fits are individually unreliable away from
  each layer's focal plane. Pass 1 only ever fits the two or three frames it
  elected; a reconstruction pass that needs all N will have to estimate motion it
  does not currently have, and cannot get it by running the existing estimator
  frame-by-frame.
* **Layer ordering.** Provided only as a focal-peak proxy with an unguarded
  direction assumption.

**Operationally:** use the differential map, never the absolute one, to decide
anything about a composite; report certified/boundary/excluded with every number;
and expect ~65% certified coverage on a real handheld sweep. The certifier can
currently arbitrate between two whole composites (KAT-3) and can localize a
silhouette-scale defect (KAT-4a), but it cannot yet see a 3-level localized defect
on a real scene, and it will not be able to until the layer decomposition improves.
