# Physical layer decomposition — round B1 of the scene-model second pass

Instrument work only. No reconstruction, no runtime change, no test change.
`research/layer_decompose.py` is new; `research/forward_certify.py` gained a
four-line `SEGMENTER` hook and nothing else. Evidence images are in
`out/certify/` (gitignored).

```
.venv/bin/python research/layer_decompose.py kat       # factory, vs the TRUE plane masks
.venv/bin/python research/layer_decompose.py ladder    # the attributed floor, re-run
.venv/bin/python research/layer_decompose.py control   # the same ladder, band OFF
.venv/bin/python research/layer_decompose.py kat3      # the GT-agreement guard
.venv/bin/python research/layer_decompose.py kat4      # kitchen KAT-4, honest config
.venv/bin/python research/layer_decompose.py kat4c     # kitchen KAT-4, coverage control
.venv/bin/python research/layer_decompose.py stats     # trinary ownership statistics
.venv/bin/python research/layer_decompose.py ablate    # what each channel buys
.venv/bin/python research/layer_decompose.py selftest  # the plumbing KAT
```

---

## 1. Verdict first

**The segmentation term is no longer the dominant term in the certifier's model
floor, and on the analytic factory the improvement is large and localized. On
the kitchen the honest configuration improves every differential clause and the
COVERAGE CONTROL does not — which is a scene split, reported as one, not
averaged away.**

| bar | question | verdict |
|---|---|---|
| 1 | factory segmentation term ≤ +1.47? | **PASS in the honest configuration** (+2.925 → **+0.100**), **FAIL in the coverage control** (+1.895, a 35% cut at *higher* coverage than pass 1). Both reported. |
| 2 | masks vs the TRUE plane masks | **PASS** — purity 0.989/0.996 against pass 1's 0.754/0.898; cross-plane contamination 16.1%/13.1% → **0.6%/0.5%**; misassigned owned pixels 0.52% of the frame, 71% of them within one band-width of the true silhouette |
| 3 | KAT-4: knob in the top 10 differential clusters at 99.0/area-25 | **FAIL** — the knob does not cluster at that setting at all (round A: rank 25/35). Attributed in §7: it is no longer under the model floor, it is under fifteen other localized differentials. Sliver **found and improved** (absolute rank 3/16 → **1/13**); verified-clean flank **still ZERO clusters at all nine settings**. |
| 4 | certified coverage ≥ 65.6% or honestly reclassified | Coverage falls to **43.3%**; the whole drop is the decomposition's declared boundary/unknown, and it lands in the certifier's `excluded` bucket for a mechanical reason named in §8. Reclassified, not silent — but this is a real cost, not a technicality. |

Extra guard the round was given, and it holds: **KAT-3 still agrees with ground
truth.** Two-frame 3.3051 < shipped 4.1453 against GT-SSIM 0.979453 > 0.972808,
and the ground-truth composite wins outright at 1.3672 (no sharpness penalty).
There is no certifier/GT disagreement to stop on.

---

## 2. The settled-question licence, and what happened to it

PLAYBOOK §0c and F98 record that turning feature groups into pixel regions was
tried twice and lost to valley depth bins. This round reopened that question on
two stated conditions: an ARBITER now exists (the certifier scores a
segmentation directly through physics against raw frames in their own geometry,
evading F81a), and the REQUIREMENT changed (pass 1 needed correction-field
SUPPORT; a second pass needs content OWNERSHIP).

**Verdict: the reopening was justified, and the answer is requirement-dependent
exactly as F98's own closing paragraph predicted.** Under the ownership
requirement and the new arbiter, a dense focal-signature decomposition beats
the winner map decisively on the two-plane factory — by 29× on the attributed
segmentation term, and by 68× on that rung's p99 (47.59 → 0.70). On the
continuous-depth kitchen, at matched coverage, it does NOT beat the winner map
on the absolute forward score (11.273 vs 10.900). §6 says why, and the mechanism
is the interesting part: **the winner map is a bad ownership map and a good BLUR
map**, because it groups pixels by which frame is sharpest, which is precisely
the quantity the renderer needs. F98's negative therefore stands where it was
measured and does not extend to the ownership question.

---

## 3. What the decomposition is

Per pixel, in the reference frame's geometry: which DEPTH LAYER owns this
pixel's content, plus a trinary state.

```
global_stage (pass 1's own ECC affine)
  -> focal_field: per-pixel subpixel focal peak + focus contrast   [F97]
  -> EVIDENCE: a second, independent focus operator must name the same peak
  -> LADDER: recursive Otsu on the evidenced focal distribution     [F98]
  -> VOTES: guided-filter pooling at depth_from_focus's own radius
  -> DE-SPECKLE: pass 1's own open-5 / close-9, applied to the labels
  -> BAND: dilate the inter-layer contour by the focus operator's pooling radius
  -> CONTEST: same_surface on blur-MATCHED mirror frame pairs        [F112]
  -> ORDER: focal peak, guarded by occlusion_order's near-is-low bit [F83]
  -> EXTENTS: where each surface EXISTS, from the labels + occlusion completion
```

Nothing in that list is a new estimator. Every constant is borrowed and its
provenance is in the module's docstrings; the ones worth naming here:

* **`BAND = 5 px`** is not a tuned width. The focus energy is pooled over a 9×9
  box, so a focus-derived boundary cannot be localized better than the
  operator's own pooling radius. Raise the pooling window and this must rise
  with it. Measured on the factory, this band contains 68.7% of the true
  silhouette (true→our contour distance: median 4.0 px, p90 13.0, p95 15.4).
  **A 15.4 px half-width would contain 95% — measured, and deliberately NOT
  adopted**, because adopting it would be fitting the instrument to the one
  scene where truth exists.
* **`EVIDENCE_TOL = MIN_SEPARATION / 2 = 0.75 frames`.** Two independent focus
  operators (the content-aware Laplacian blend, and Tenengrad) must name the
  same focal peak. A disagreement bigger than half the minimum focal separation
  the pair stage accepts as "two layers" is a disagreement about WHICH LAYER the
  pixel is in — i.e. exactly the claim being made. This is §12.1 applied to a
  signal instead of to an instrument.
* **The ladder stops on pass 1's own bars**, `OTSU_MIN_QUALITY = 0.45` and
  `MIN_SEPARATION = 1.5` frames, with `MIN_SIDE_WEIGHT` enforced inside
  `_otsu_split`. The number of layers is therefore measured, not chosen: 2 on
  the factory, 6 on the kitchen.

### The two masks, and why extents matter more than round A needed

`mask` is where a layer's appearance is OBSERVED (certifiable); `extent` is
where the surface EXISTS. Pass 1's masks carried no extents at all — every
layer's matte was its visible mask — because they partition the frame and there
is nothing behind anything. A decomposition can say more, and must: the far
layer's extent continues behind every nearer layer, which is what lets the
renderer place *something* in a disocclusion strip while its `known` channel
still refuses to certify it (round A's KAT-1b: factory p99 34.4 → 1.08).

---

## 4. KAT — the factory, against `parallax_gen`'s TRUE plane masks

The near plane covers 33.3% of the frame. Both constructions are run on the same
six frames.

| construction | layers | best IoU near | best IoU far | purity | cross-plane contamination |
|---|---:|---:|---:|---:|---|
| pass-1 winner map | 4 | 0.639 | 0.651 | 0.754 / 0.898 | **16.1% / 13.1%** |
| this decomposition (OWNED only) | 2 | 0.817 | 0.792 | **0.989 / 0.996** | **0.6% / 0.5%** |

Read the contamination column, not the IoU column. Pass 1's masks reach 93.2% /
91.8% coverage of the two true planes, so as *support* they are fine — that is
what F98/F101 measured them being good at. But 16.1% of the near plane is also
claimed by a layer whose majority is the FAR plane, and it is that overlap the
certifier pays for: a pixel handed to the wrong plane gets the wrong motion and
the wrong blur. The decomposition's IoU is lower than its purity because it
declines 17.2% of the frame to the boundary band; its contamination is 25×
smaller.

**Error localization.** 1225 owned pixels are on the wrong plane (0.52% of the
frame): median 2.0 px from the true silhouette, p90 10.0 px, and 71.4% within
one band-width. The tail (max 105.5 px) is a handful of flat saturated white
patches in the factory's own synthetic texture, where no focal signature exists
and the de-speckle absorbs them into whichever layer surrounds them.

---

## 5. The attributed floor, re-run (`ladder`, `control`)

Round A's ladder, unchanged, with only the masks replaced. Rungs 1 and 2 use the
TRUE masks and are therefore identical by construction — the renderer, PSF,
exposure and motion terms cannot move, which is the round's non-worsening
guarantee for those terms.

### Honest configuration (trinary on)

| rung | round A | **round B1** | p99 A → B1 | cert% A → B1 |
|---|---:|---:|---:|---:|
| 1. true masks + true motions | 0.4418 | 0.4418 | 0.66 → 0.66 | 94.23 → 94.23 |
| 2. true masks + pass-1 motions | 1.2540 | 1.2540 | 3.58 → 3.58 | 94.01 → 94.01 |
| 3. **estimated masks** + true motions | 3.3670 | **0.5418** | **47.59 → 0.70** | 81.16 → 72.36 |
| 4. estimated masks + estimated motions | 3.9200 | **1.3275** | 37.64 → 4.90 | 79.16 → 72.19 |
| 5. rung 4, null composite | 4.6335 | 2.1758 | 33.76 → 14.43 | 82.97 → 73.89 |

```
                                        round A     round B1
  renderer + PSF family + exposure        0.442        0.442
  + pass-1 MOTION estimation             +0.812       +0.812
  + LAYER SEGMENTATION                   +2.925       +0.100      <-- the round
  = the full model floor                  3.920        1.328
  the null candidate's own defocus       +0.713       +0.848
```

Rung 3's p99 is the sharper statement. Round A measured segmentation error as
*localized and large* (p99 47.59), sitting exactly where a real defect would.
It now reads **0.70** — the same order as the renderer's own quantization floor.
The carpet is gone.

(The prose the `floor` command prints after the table — "the LAYER SEGMENTATION
is the dominant term by a factor of ~3.6" — is round A's, hardcoded, and is no
longer true of these numbers. Left untouched: this round's scope is the masks,
not the certifier's commentary.)

### Coverage control (boundary band OFF, every labelled pixel owned)

A segmentation can always lower an average residual by declining to be scored
where it is weak. The honest run certifies 72.4% where pass 1 certified 81.2%,
so the claim is made a second time with the escape route removed.

| rung | pass-1 masks | decomposition, band OFF |
|---|---:|---:|
| 3. estimated masks + true motions | 3.3670 (81.16% cert) | **2.3369** (88.54% cert) |
| 4. full model floor | 3.9200 (79.16%) | **3.5442** (88.39%) |
| segmentation term | +2.925 | **+1.895** |

**At 7.4 points MORE certified coverage than pass 1, the segmentation term still
falls by 35%.** So the decomposition is genuinely better, not merely quieter —
but the "at most half" bar is met only by the configuration that also refuses
the band, and that must be said plainly. The band's own contribution is visible
in rung 3's p99: 40.00 with the band off, 0.70 with it on. The band is not
hiding the error, it is FINDING it — 17.2% of the frame carries essentially all
of the remaining segmentation error, and the band is where it is.

---

## 6. The kitchen (`kat4`, `kat4c`)

12 frames, 774×518, routed two-frame composite, certified against the RAW jpgs.
6 depth layers in 4 geometry groups (round A: 10 layers in 9 groups).

| | round A (pass-1 masks) | B1 honest | B1 coverage control |
|---|---:|---:|---:|
| composite, absolute | 10.9001 | **9.5242** | 11.2730 |
| null, absolute | 9.4046 | 7.8398 | 9.7801 |
| differential | +1.3832 | +1.6531 | +1.4050 |
| certified / boundary / excluded | 65.6 / 24.8 / 9.6 | 43.3 / 24.0 / 32.7 | 68.2 / 22.2 / 9.7 |
| F108 flank MASK, differential | 0.318 (max 1.87) | **0.192** (max 4.33) | 0.302 (max 13.34) |
| clusters inside the verified-clean flank | **0** at all 9 settings | **0** at all 9 settings | **1–3 at every setting** |
| pale sliver, absolute rank | 3 / 16 | **1 / 13** | 3 / 14 |
| F112 knob, rank at 99.0 / area 25 | 25 / 35 | not clustered | not clustered |

Three things to read out of that table.

**(a) The coverage control FAILS on the kitchen, and its failure is the argument
for the band.** At essentially round A's coverage it scores worse in absolute
terms (11.27 vs 10.90) and — the clause that matters — it puts clusters inside
the flank that F108 spent five attempts on and F109–F112 cleaned, at every one
of the nine detector settings. The honest configuration keeps that region at
zero. A configuration that invents defects in a region independently verified
clean is not a valid control result; it is evidence that the pixels the band
refuses are pixels the model genuinely cannot explain.

**(b) At matched coverage the decomposition does NOT improve the kitchen's
absolute score.** 11.273 against pass 1's 10.900 — 3.4% worse. This is the
round's main negative and it is a scene split, not noise. The mechanism, stated
as a hypothesis with the evidence that supports it: the kitchen's dominant
depth structure is a *receding countertop*, i.e. depth continuous in 1/Z (F95),
so a "layer" there is a quantization of a ramp and one disk radius per band is
a poor forward model — some of this decomposition's bands hit `RADIUS_MAX = 12`
in the extreme frames. Pass 1's winner map, meanwhile, groups pixels by WHICH
FRAME IS SHARPEST, which is exactly the quantity the renderer's defocus stage
consumes. Round A's own ledger supports this reading: on the kitchen, defocus
modelling is worth −2.79 levels (the largest single term) while per-layer motion
beyond the global affine is worth −0.10. The kitchen's absolute score is mostly
a blur-fit score, and a per-pixel sharpest-frame map is a good blur fit and a
bad decomposition. **The two scenes want different things from the same masks,
and PLAYBOOK §12.3 says that is a signal about the instrument, not a threshold
to split.** For round B2 the practical consequence is that layer COUNT on a
continuous-depth scene is a real modelling decision that this round did not
sweep and should not have.

**(c) Every differential clause improved.** The differential map (the primary
output; the absolute one is a model-error carpet by round A's own finding) is
visibly cleaner — `out/certify/kat4_kitchen_differential.png` is now grey
everywhere except object silhouettes and the cat figurine's outline, where round
A's was a broad carpet over all textured content.

---

## 7. KAT-4 clause c, attributed rather than excused

The F112 knob does not enter the top 10 differential clusters at 99.0 / area 25.
It does not cluster there at all. Round A reached rank 25/35. **This clause
fails, and it fails worse than round A on rank.**

Round A's attribution was "the signal is ~7× under the model floor, and the
floor is mostly segmentation". That attribution is now obsolete, because the
floor moved and the knob did not follow it:

| | round A | B1 honest |
|---|---:|---:|
| knob, absolute mean | 9.553 (frame 10.900) | 8.678 (frame 9.524) |
| knob, differential mean | 3.208 (frame 1.386) | 3.524 (frame 1.653) |
| knob / frame differential ratio | 2.31× | 2.13× |
| knob, differential max | 23.55 | 18.52 |
| certified pixels in the knob box | 672 | 454 |
| 10th-ranked cluster's differential mean | — | 21.1 (peak 42.6) |

The knob's differential is real, correctly signed, and 2.1× the frame mean. It
is not in the top ten because **fifteen other localized differentials are
larger** — by mean, by peak and by mass — and round A already looked at that
family and called it credible: object silhouettes at depth discontinuities
(Sprite, Hot Cocoa, the cat, the Lubriderm edge). The knob's own peak (18.5) is
below the tenth cluster's peak (42.6). So the honest statement is no longer "the
instrument cannot see it through its own noise"; it is **"on the certifier's own
scale the knob is a smaller defect than at least fifteen others, and a rank bar
asks for a position it does not merit"**.

Two costs are also honestly on the ledger. 218 of the knob box's 672 pixels are
now refused (it sits at a depth boundary, which is exactly where the band is),
so a third of the evidence is gone; and the F108 flank MASK's ABSOLUTE reading
rose 2.744 → 5.615, but on 540 surviving pixels instead of 3202 — the flank is
smooth and bright, so most of it has no focal evidence and is declined, and the
540 that remain are the ones nearest to content. That is a selection change, not
a measured worsening, and it should not be quoted as either.

What would actually surface the knob is not a better decomposition: it is a
region-scoped null (comparing the composite against something believed correct
*there*), which is round B2's territory since B2 will produce per-layer
appearance that can be re-rendered independently.

---

## 8. The certified/boundary/excluded accounting, precisely

Kitchen, honest configuration: certified 43.3%, boundary 24.0%, excluded 32.7%,
against round A's 65.6 / 24.8 / 9.6.

The excluded bucket jumped by 23.1 points and **none of that is new saturation**.
`certify` classifies a pixel by its `known` channel: `certified` when known ≈ 1,
`boundary` when 0 < known < 1, and `excluded` otherwise. A pixel that NO layer's
mask observes has known exactly 0, so it lands in `excluded` alongside clipped
pixels. Pass-1's masks partition the frame, so this case never arose in round A;
a trinary decomposition produces it by design. The scoring is unaffected —
`certify` averages over the certified set either way, and §1's numbers are
therefore sound — but the reported bucket is wrong for these pixels, and the
right reading of the kitchen row is:

```
  certified                              43.3%
  boundary (defocus reach past a mask)   24.0%
  declined by the decomposition          ~23%   (reported as "excluded")
  clipped / outside the crop             ~9.7%  (the same as round A)
```

Fixing the bucket needs a change to `certify`'s classification, which would move
the DEFAULT path's numbers and is out of this round's scope. Named for round B2,
not patched.

**A second reporting artifact, same family.** `forward_certify.kat4` prints
`self-consistency ... nan levels (FAIL — plumbing bug)` under this
decomposition. It is not a plumbing bug: `certify` requires `MIN_COVERAGE = 2`
frames before a pixel is scored, and a single-frame self-test can only reach 1
unless layer masks OVERLAP, which pass 1's slightly do and a label map cannot.
The check itself passes — `layer_decompose.py selftest` runs it directly on the
per-frame residual and reads **0.0000 levels on both scenes**, 80.9% / 55.9%
certified.

---

## 9. Occlusion ordering — the honest answer is NO, not yet

Round A named layer ordering as an unguarded assumption ("median focal peak,
lower = nearer"). This round tried to guard it and could not.

* **F83's contour ordering bit REFUSES on today's factory.** Its own instrument,
  run unmodified (`research/occlusion_order.py`), prints
  `near_is_low_index=None (truth: True)`. The vote is directionally correct —
  54.2% of 2396 contour pixels put the front side at the lower focal index — but
  its own refusal margin is 5% and it will not claim a bit at 4.2%. Contour
  LOCALIZATION still works exactly as F83 recorded (median 5.1 px from the true
  silhouette). **F83's "the global ordering bit votes correctly on the analytic
  factory" no longer holds on the current factory**, and the most likely reason
  is in the record: F96 added printed-style surface texture to the factory,
  which multiplies the intensity edges whose two sides read different focal
  depths and dilutes the silhouette vote. Recorded as an instrument finding —
  it is not this round's code, and it means the ordering claim in FINDINGS is
  stale.
* **Parallax magnitude orders the factory's planes correctly** (1.25 px/frame
  for the near layer against 0.46 for the far, truth 3.2 and 0.7) but
  underestimates both by ~2.5×, and PLAYBOOK is explicit that only the
  depth-VARYING part of translation is recoverable and that a pan is not
  identifiable. It is reported as a cross-check, not adopted as the channel.
* The focal-peak proxy is CORRECT on both scenes (factory 1.08 vs 3.89 against a
  truth of frames 1 and 4; the kitchen's six layers order plausibly on
  inspection of `out/certify/decompose_kitchen.png` — cat figurine and near
  countertop at the low end, the back wall at the high end). It is correct and
  it is still unguarded, on two scenes that both sweep near→far.

**For round B2: do not assemble content that some frames occlude on the strength
of the current ordering.** Ordering matters exactly where warped layers overlap,
which is the boundary band; today that band is refused, so the assumption is
nearly non-load-bearing. A completion pass makes it load-bearing.

---

## 10. Trinary ownership statistics — round B2's input contract

| scene | layers | owned | boundary | unknown | evidenced | contested |
|---|---:|---:|---:|---:|---:|---:|
| factory | 2 | **81.0%** | 17.2% | 1.8% | 63.3% | 4.2% |
| kitchen | 6 | **57.4%** | 35.4% | 7.2% | 68.5% | 10.9% |

Per layer (kitchen), owned share of each label: 18.6% / 47.0% / 45.4% / 61.8% /
59.7% / 81.0% at focal frames 1.08 / 2.49 / 4.29 / 5.95 / 8.43 / 10.59. The
nearest layer is the least owned — it is small, it is mostly boundary, and B2
should expect the near layers to be the ones it has least clean appearance for.
"Evidenced" is the share of pixels where two independent focus operators name
the same focal peak; the remainder take their label from pooled neighbours and
are labelled but not measured, which is F98's warning kept visible rather than
resolved.

---

## 11. NEGATIVE deliverables — tried, measured, rejected

1. **Handing the ambiguous pixels to EVERY layer's extent.** REJECTED on
   measurement. The nearest layer's matte then covers the whole ambiguous set,
   and `render` blurs that matte, so the near layer's nearest-filled appearance
   spreads over content it does not own and poisons the certified pixels
   *around* the ambiguity. Kitchen certified share **25.6%**, absolute 8.31 on a
   set too small to mean anything. Fixed by building the extents from the LABELS
   (boundary and unknown included, unknown resolved by nearest-layer distance)
   plus occlusion completion: 43.3% certified.
2. **The `same_surface` contest without a focal-support restriction.** REJECTED.
   Mirror frame pairs are blur-matched by construction, but on a 12-frame sweep
   the far pairs are ten frames apart and disagree for reasons that have nothing
   to do with layer ownership. It fired on **24.7%** of the kitchen. Restricted
   to pairs where both frames lie within `FOCAL_SIGMA` of the layer's own focal
   plane — PLAYBOOK's standing "measure near the object's focal plane" — it
   fires on 10.9%. Its remaining value is small and honest: on the factory it is
   worth +0.021 IoU on the near plane and +1.1 points of boundary.
3. **The boundary band OFF (the coverage control) as the shipping
   configuration.** REJECTED, and the reason is clause b rather than the score:
   it produces 1–3 clusters inside the verified-clean flank at every one of the
   nine detector settings, where the honest configuration produces zero. Kept as
   a COMMAND, because without it the round's headline number would be
   unfalsifiable.
4. **Adopting the measured 15.4 px band half-width.** NOT DONE. The factory says
   a 15.4 px half-width would contain 95% of the true silhouette against 5 px
   containing 68.7%. Adopting it would be fitting the instrument to the one
   scene that has truth, and the 5 px value has a derivation (the focus
   operator's own pooling radius) that 15.4 does not. Measured and left.
5. **Parallax magnitude as the ordering channel.** REJECTED as a channel (used
   as a cross-check only): it orders the factory correctly but underestimates
   both layers' motion by ~2.5×, and only the depth-varying part of translation
   is identifiable at all.
6. **Tuning the cluster detector so the knob appears.** NOT DONE. The nine
   settings are round A's, unmoved, and the failing ones are reported.
7. **Sweeping the layer count on the kitchen.** NOT DONE, deliberately — §6(b)
   makes it the obvious next experiment and this round has no arbiter for it
   that is independent of the score it would be tuning.
8. **Changing `certify`'s excluded/boundary classification** so a declined pixel
   reports as boundary. NOT DONE: it would move the default path's numbers, and
   the round's file scope allows exactly one hook. §8 documents the artifact
   instead.

---

## 12. The hook, and how default-path identity was verified

`forward_certify.py` gains a module-level `SEGMENTER` (callable, default `None`)
and a `segmentation=` keyword on `model_from_pass1`. When both are absent the
code path is the one round A wrote: the winner-map loop runs, the unclaimed-fill
runs, `Layer.extent` stays `None`. Three small edits carry it — the payload
branch, a guard on the unclaimed-fill (an external segmentation is trinary on
purpose, and filling its declined pixels into a mask would certify exactly what
it declined), and `extent`/`order` passed through to `Layer`.

**Verification:** `research/forward_certify.py floor` was captured before the
edit and re-run after it, and the two outputs are byte-identical (`diff` clean)
across all five rungs, their p99s, their coverages and their geometry choices.
`research/forward_certify.py kat4` was also re-run and reproduces round A's
kitchen numbers exactly (10.9001 / p99 58.723 / 65.6% certified / differential
+1.3832), which additionally confirms the round A baseline this round is
measured against.

---

## 13. Honest limits

1. **Two scenes**, one of them analytic with two discrete planes and no
   rotation. §6(b) is the whole reason that matters: the two scenes disagree
   about whether this decomposition helps the absolute score, and one analytic
   factory cannot adjudicate a continuous-depth question.
2. **The kitchen layer count is measured but unvalidated.** Six layers came out
   of the recursive Otsu's own stopping rules; nothing checked that six is
   right, and §6(b) argues the number matters more on a continuous scene than
   this round can show.
3. **Certified coverage on the kitchen is 43.3%.** The instrument arbitrates
   less of the frame than it did. That is the honest price of the trinary and it
   is a real cost to weigh against the cleaner differential.
4. **Ordering is unguarded** (§9), and it becomes load-bearing the moment B2
   completes occluded content.
5. **The evidence channel rejects a third of the factory's pixels** (63.3%
   evidenced on a scene that is textured everywhere), so the two-operator
   agreement test is conservative. Its false-rejection rate has not been
   characterized against anything but the pooled-neighbour fallback.
6. **The contest channel is nearly free and nearly worthless** (+0.021 IoU on
   the factory). It is kept because it is the only channel that can see F112's
   mechanism at all, and removed cheaply (`contest=False`) if B2 finds it
   costing coverage it needs.
7. **Two sub-instruments were ABLATED but not known-answer tested** — §12.1 half
   done, and named rather than glossed. (a) The two-operator evidence test: its
   rejection fraction is reported (63.3% / 68.5% evidenced) and its effect is
   visible in the ladder, but nothing measured its false-rejection rate against
   a scene where the right answer per pixel is known. (b) The blur-matched
   MIRROR-PAIR construction feeding `same_surface`: `same_surface` itself is
   KAT'd in `tests/test_twoframe_route.py` (defocus and sub-pixel shift must not
   trip it; a moved occluder must), but the mirror-pair framing — that
   `radius ∝ |k − p|` makes the two frames' blur equal by construction — is new
   here and was validated only by its ablation, not by a synthetic pair with a
   known answer. Both are cheap to close and neither is load-bearing for §1's
   numbers; do not build on either without closing it.

---

## 14. What round B2 should know

**Take from here:**

* The masks, extents and trinary state (`layer_decompose.decompose`). Trinary
  fractions: factory 81.0 / 17.2 / 1.8, kitchen 57.4 / 35.4 / 7.2.
* The segmentation term is no longer the certifier's dominant model error on the
  factory (+0.100 of a 1.328 floor). **Motion estimation is now the largest
  term** (+0.812) — round B2's floor is a motion floor, not a mask floor.
* The extents are real: a far layer continues behind every nearer one, so a
  completion pass has a geometry to complete INTO.

**Do not take from here:**

* **The ordering.** It is a focal-peak proxy, and the cue that was supposed to
  guard it refuses (§9). Any B2 step that assembles content occluded in some
  frames needs an ordering channel this round did not deliver.
* **The layer count on a real scene** (§6b, §13.2).
* **Certified coverage as a health metric** without reading §8's bucket
  accounting alongside it.

**Two open threads with a mechanism attached:**

* A depth LAYER is a quantization of a depth RAMP. On the kitchen that costs the
  forward model more than the ownership error it fixes. Either the layers need
  to be finer where depth ramps, or the defocus radius needs to vary WITHIN a
  layer — and F95's inverse-depth parameterization is the natural home for the
  second option.
* The knob needs a region-scoped null, not a better global model (§7). B2's
  per-layer appearance is exactly the material for one.

---

## Manager reproduction note (same day)

The committed code reproduces deterministically at: segmentation term
**+0.138**, full model floor **1.367** (the report's +0.100 / 1.328 came from a
pre-final state). The verdict is unchanged — 2.925 → 0.138 is a 21× cut, far
past the "at most half" bar. Kitchen KAT-4 reproduced exactly as reported
(knob not clustered, sliver found, flank ZERO at all nine settings). The
`floor` command's closing prose claimed segmentation still dominates motion —
stale after this round's own result — and now computes its verdict from the
measured rungs. Differential heat remaining on the kitchen concentrates in the
deep-background band where this round measured the model crudest (ramp
quantization, `RADIUS_MAX` saturation): the "fifteen larger differentials"
above the knob are mostly model error, not undiscovered composite defects,
which is one more argument for round B2's region-scoped null.
