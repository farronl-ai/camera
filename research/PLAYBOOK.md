# focusstack — Expert Playbook (LOADME)

**This file is the project's primary intellectual property: the domain theory, the
conditions under which each concept applies, and the conclusions that were paid for
with failed experiments. Read §0 before doing any technical work here.**

How to read it. §0 is what is TRUE and when it applies — start there, act on it, and
do not re-derive it. §0b says which tool to reach for in which situation. Everything
after that is PROVENANCE: how each conclusion was established, kept because a rule
without its conditions becomes a trap, and because knowing *why* a negative failed is
what lets you recognise when its conditions no longer hold. `FINDINGS.md` is the
dated log; this file is the distillate.

The standing danger this structure exists to prevent: a session that re-runs an
already-settled experiment, or that applies a true rule outside the regime where it
was shown true.

---

## §0. What is true, and when it applies

Each line is load-bearing, measured, and has an anchor in `FINDINGS.md`. The
*condition* matters as much as the claim.

### Focus measurement
- Focus is high-frequency energy; defocus is a low-pass. **Always.**
- Signed Laplacian cancels on smooth low-contrast surfaces; `mod_laplacian`
  (`|I_xx|+|I_yy|`) does not. **Route by local contrast** (`content_aware`) rather
  than choosing globally — but expect only ~+0.002 GT-SSIM from operator routing.
  The visible wins are structural, not operator choice.
- Pool responses over a window: single-pixel focus is sparse and noise-sensitive.

### Fusion
- The decision must be **multi-scale, edge-aware, and hardened**; reconstruction must
  be **multi-band**. `fuse_perband` has all four and is the default at every
  resolution tested.
- Scale-adaptivity belongs in the STRUCTURE, not in a tuned number: decide per pyramid
  band with a fixed small window, and the effective scale grows with the band
  automatically. **Applies whenever a window size seems to need tuning by resolution.**
- Cap any window to its band's size (a window ≈ the band degenerates the guided filter
  to a global mean), and never plain-average the base band (it imports low-frequency
  defocus spread).
- `harden` is one mechanism for two problems — defocus-spread rejection and
  thin-structure preservation. **Use where one frame is confidently sharpest; keep
  soft blending where ambiguous.**

### Registration and scene geometry (the newest and least-explored area)
- **Parallax and focus breathing are superimposed and cannot be fitted by one warp.**
  A hand pivots the device, not the entrance pupil, so displacement scales with
  inverse depth; refocusing independently changes magnification (14% measured on a
  phone macro sweep). A global affine compromises between them. **Applies to every
  handheld stack; not to a rail or a locked-down camera.**
- **Focus breathing is real but small, and the global affine removes it** (F91).
  Measured on the kitchen sweep: ~1.5% raw magnification by frame 10, and ~1.000
  after the affine; confirmed on a GT factory where an applied 2.5%/frame breathing
  leaves residual scale 1.0013…0.9994. Earlier claims of 14% breathing and of
  depth-dependent residual magnification were measurement artifacts (F91 retires
  F87/F88/F90 on this point).
- **A near object's residual motion is dominated by TRANSLATION**, not scale: the
  kitchen bottle measures scale 1.003–1.008 with a validated estimator while needing
  ~+19 px of shift. Do not add machinery for a scale term without first measuring
  that the scale exists.
- **Scale and translation are ill-conditioned for a compact off-centre object.** Its
  edges span a narrow radius range that never approaches zero, so `s·(x−c)` is
  nearly constant across the object and trades against translation. Measure on the
  axis where the object straddles the optical centre, where the radius changes sign.
- **Rigidity is a free two-axis consistency test.** A rigid object must scale
  equally in both axes; a claim that it grew 14% horizontally and 1.5% vertically is
  a broken measurement, not a discovery. Costs one extra measurement and would have
  saved three findings.
- **Fit rigid motion on MATERIAL edges (texture, print, corners), never on the
  silhouette of a curved object** (F92). A cylinder's limb lies where its surface
  turns away from the camera, so it slides across the material surface as the
  viewpoint moves: it is view-dependent by construction. On the kitchen bottle, nine
  printed edges agreed to ~1 px while the left limb read half their value. This is
  why interior edges (F89) were the arc's strongest instrument — not because there
  were more of them, but because they were the right kind.
- Corollary: a non-zero rigidity differential on a curved object may be legitimate
  limb motion, not error. State the rigidity test over material features only.
- Corollary: curved subjects (bottles, cans, jars) bias silhouette-based motion
  systematically; flat-faced subjects will never reveal it. Most of this project's
  real data is curved.
- **A depth bin is a range, not an object.** ECC over a region follows its majority.
  Group by measured motion; use depth only as a seed; cut bin edges at
  depth-histogram valleys, never quantiles.
- **Object integrity is a MERGE rule**: regions whose fitted motion agrees across the
  sweep are one object. Merging past ~2 px merges real depth planes and collapses.
- **Blend coordinates, not warped images** — one source location per output pixel, so
  a multi-stage correction still costs one interpolation. A field may transport
  content freely; it must never stretch it.
- **Read camera motion off ALL edges at once** (F94): each component has a distinct
  spatial signature (radial for breathing and forward translation, uniform for pan
  and lateral translation), the quadrant sign pattern separates radial from uniform
  with no depth at all, and depth then splits each pair because only the
  translational components scale with 1/Z. With ~200 edges this is heavily
  over-determined.
- **Carry depth explicitly in any motion decomposition** (F95). The components are
  not depth-independent; they are characteristically different from each other, and
  depth-dependence is part of that difference. Parameterize by inverse depth ρ, and
  exploit that ρ is a SCENE property shared by every frame while motion is per-frame
  — that makes it hugely over-determined and is solvable by alternating linear
  halves. On real data this explains ~69% of observed displacement, against 25–50%
  for the same idea fed by tiles.
- **ρ is recoverable only up to an affine reparameterization**, because a pan can
  absorb any constant added to it. Use only its depth-VARYING part; its offset and
  scale are gauge. Do not expect monotone ρ to validate a fit — bins at similar true
  depths order arbitrarily within noise.
- **A pan is NOT identifiable.** A uniform shift is indistinguishable from every
  depth translating equally; only the depth-VARYING part of translation is
  recoverable. Fitting both a uniform term and per-depth translations makes the
  system singular and silently halves the answer. Define breathing as the radial
  component the depth bins share and forward translation as the part that varies.
- **Textureless interiors take their motion from edges.** Correlate gradient profiles,
  integrate along the edge, trust only the normal component, and measure near the
  object's focal plane then propagate along the sweep.
- **Interior edges make "is this one object?" falsifiable.** Two edges are exactly
  determined — solvable, never testable.
- **Key a parallax correction on MOTION GROUPS, never on depth** (F99). Content at
  the same depth VALUE can have different motion, so any depth-keyed fit — binned or
  continuous — averages the target away. Measured: a depth-keyed curve never exceeded
  ±5 px at the kitchen bottle's depth in any frame, while the bottle's own features
  gave a clean monotone series reaching ~+19 px. Continuity does not cure it because
  it was never a quantization problem.
- **Match confidence does not detect defocus bias.** A blurred profile correlates
  CONFIDENTLY against a sharp one at about zero shift, so degraded features vote
  "no motion" with high confidence on exactly the objects that moved most. Key trust
  on each feature's own focal distance, never on its match score.
- **Grouping FEATURES is solved; turning groups into pixel REGIONS is not** (F98).
  Features carry evidence — a focal curve, a normal displacement, a confidence — and
  group cleanly. Most pixels carry none, so any region mask is filled in by a
  propagation rule that is doing the real work while looking like plumbing. Sparse
  seeds plus guided propagation cost the factory 0.066 GT-SSIM; dense per-pixel focal
  peaks tie the shipped bins but still leave the target object at 18% IoU. Prefer
  evaluating a per-feature motion model through the field it implies over committing
  to hard region boundaries.
- **Cluster a 1-D signature with Otsu, not with gaps.** A bimodal distribution whose
  tails meet has no large consecutive gap, so single-linkage — absolute or
  median-relative — chains through it into one group every time.
- **Group depths by the FOCAL SIGNATURE, not by motion** (F97). Each material
  feature's own sharpness curve across the sweep peaks at its depth; measured focal
  frames land within 0.1 frame of truth, and groups separate by 2–3 frames against a
  within-group spread of 0.3–0.6. This is orthogonal to motion — features at one
  depth blur together whatever they are doing — and it works across scenes with no
  threshold, where no motion-residual threshold does (each scene wants a value the
  other cannot use). Let motion confirm rigidity WITHIN a depth instead of asking it
  to discover depth.
- **A grouping model must be one that CANNOT explain a depth difference** (F96). A
  similarity model absorbed a whole two-plane scene into one consensus at 0.92 px
  residual — right number, wrong structure — because a radial term imitates two
  separated regions translating differently. Use translation-only for grouping. And
  do not select a consensus by SIZE: that rewards the compromise fit that counts both
  planes as inliers.
- **Define an object as a maximal feature set admitting ONE rigid motion across all
  frames** (F93). Each material edge gives one scalar per frame (its normal
  displacement); a greedy consensus from spatially local seeds keeps a bottle whole
  where tile clustering fragmented it (20 of 21 grouped features inside the object).
  Grouping and motion-parameterization are separate problems — solving the first does
  not give you the second.
- **Fit scale only where it is identifiable.** The radius term must change sign across
  the object's features; a compact off-centre object trades scale against translation
  and will absorb pure translation into a rising scale. Check the conditioning before
  choosing the model, and prefer translation-only when in doubt.
- **Propagate motion from frames near the target, not from every reliable frame.**
  Handheld drift is not linear across a whole sweep: all-frames propagation gave
  +16.07 px where adjacent-frame propagation gave +18.88 against +19.2 truth. Frames
  should be near in both senses — near the focal plane for evidence, near the target
  for extrapolation.

### Occlusion and boundaries
- **Disocclusion earns a hard mask; veiling does not.** If the observation does not
  exist (parallax uncovered the scene), refuse the pixel. If it exists but is a
  mixture (a defocused occluder spreading over its surround), down-weight it softly —
  a hard mask loses in every regime tested, including the one built to favour it.
  **This distinction generalises: ask which of the two any new boundary evidence
  describes before choosing a mechanism.**
- Near a defocused silhouette BOTH sides are compromised, by different mechanisms:
  missing correspondence behind, matte mixing in front.
- Foreground blur spreads foreground outward; it never pulls hidden rear detail
  inward. Ownership is discrete and ordered, never a symmetric blend.
- Absence of foreground evidence is not positive rear visibility.

### Evaluation (where most self-deception happens)
- **With true GT, GT-referenced fidelity is the verdict.** Without it, no-reference
  metrics are weak instruments with named blind spots:
  - **globally**: usable for ranking (`0.3·Q_ABF + 0.7·Q_SSIM`, Spearman +0.72);
  - **per-tile/region**: Q_ABF *anti-correlates* — use Q_SSIM alone;
  - **for an alignment change**: unusable, because they score against sources that
    alignment itself moves;
  - **for synthesis/restoration**: unusable, because success means deviating from
    every source;
  - **on smooth content**: blind.
- A metric's exclusion filter is a blind spot in disguise; build its complement.
- Aggregates hide localized defects; a global mean rated three fusion methods equal
  while one had a visible halo.
- **Validate a measuring instrument against a known answer before believing it** — a
  broken instrument does not look broken, it looks like data.
- Clean/near-ceiling data is a mirage; re-check on the method's designed weakness.
- Test a stage with input only it must handle: alignment cannot be tested on frames
  that differ by one global transform.

### Physical modelling
- Real defocus is a DISK (circle of confusion), not a Gaussian. Gaussian synthetic
  defocus is too easy and hides scene-dependence.
- Every resample softens; compose transforms and resample once.
- Solve the formation, not the fused image. Formation quantities come from the
  original observations, never from a fused base.
- A forward fit cannot detect a support error shared by every inverse model.

## §0b. Which tool, when

| Situation | Reach for | Not for |
|---|---|---|
| Edge-aware smoothing without a detector | guided filter (`out=a·guide+b`) | cases where the guide is noisier than the signal (weak-guide trap) |
| Smooth, low-contrast content | `mod_laplacian` | textured content (use Laplacian) |
| Any resolution, general fusion | `fuse_perband` | — |
| Brightness-varying frames, needs texture | ECC | textureless regions (guard on gradient) |
| Shift measurement, subpixel, cheap | phase correlation | shifts beyond ~¼ of the patch — it saturates silently |
| Object motion where interiors are flat | edge-integrated gradient-profile correlation | intensity profiles (defocus biases them outward) |
| Testing alignment | analytic parallax factory (`parallax_gen.py`) | any stack differing by one global transform |
| Ranking whole outputs, no GT | composite `0.3·Q_ABF+0.7·Q_SSIM` | per-tile decisions, alignment A/B, synthesis audit |
| Local/per-region no-GT scoring | Q_SSIM alone | the composite |
| Deciding a boundary mechanism | "is the observation absent, or mixed?" | applying one mechanism to both |
| Unknown-size discovery | oracle ladder first, estimators second | building estimators before the ceiling is known |

## §0c. Settled — do not re-run without a changed condition

Each of these cost real experiments. They are closed, not forgotten: the *condition*
column is what would make one a live hypothesis again. If you find yourself designing
an experiment that appears here, read the provenance section first and then either
skip it or state which condition has changed.

| Question | Verdict | Reopens if |
|---|---|---|
| Q_MI as a fusion quality metric | Anti-correlates with truth; rewards ghosting/speckle | never, for this purpose |
| Correction applied after fusion | Cannot recover identifiable layer state; retired | never — solve the layer equations jointly instead |
| Symmetric foreground/background blending at an occlusion | Violates ordered visibility | never |
| A more flexible single global warp for depth-dependent parallax | Mathematically cannot fit near and far together | never — the two motions are separable only with depth |
| Raising the per-bin correction acceptance cap alone | Does nothing; the correction is never *proposed* | only alongside a splitting scheme that proposes it |
| Parametric depth→displacement (linear in the depth proxy) | Loses to nonparametric bins (0.9313 vs 0.9565) — the focus index is monotone but not affine in inverse depth | a calibrated inverse-depth map, not the raw index |
| Joint motion/depth/calibration estimator as the default | Best registration of any variant, but invents motion on a still stack | a better observation model (its rigid+depth model explains only 25–50% of tile shifts) |
| One-sided disocclusion refusal (spare the occluder) | Loses to two-sided; a defocused occluder's own matte mixes background onto its boundary | a mechanism that separates matte mixing from disocclusion on the front side |
| Veiling as a hard mask | Loses in every regime including the one built to favour it | never as a hard mask; it is legitimate as a soft weight |
| Median-stabilizing the depth map before the step test | Worsened silhouette concentration (2.81× → 2.07×) | a different stabilizer with its own evidence |
| Connected-coherence gating of region splits | No effect; the spurious regions are coherent, not confetti | never for this purpose |
| Tuning the tile-confidence floor to serve both scenes | No value serves both (0.05 vs 0.35 trade directly) | replace with the physical test (residual proportional to frame motion) |
| Adding a scale term to the region model | Unproven — the object it was built for measures scale ~1.005 and needs pure translation (F91) | a region whose scale is first MEASURED with a known-answer-validated estimator |
| Contrast-over-gradient as a blur estimator | Saturates by 2 px of blur, swamped by texture | never — use distance from the object's focal frame |
| Pseudo-GT or source similarity as latent-scene truth | Ceiling-limited and blind to correct synthesis | real captured latent truth |
| Blind generative / diffusion de-occlusion fill | Not remnant-auditable | never under the current mission |
| Semantic models on synthetic blob content | Pastiche fragments under SAM; bokeh spoofs objectness | benchmarks built from real objects |

**Standing exception, and it matters:** a rigorous negative is a statement about its
TEST CONDITIONS, not about the idea. At each checkpoint, sweep this table for
conditions that have since changed — pipelines, factories and metrics evolve, and
F27 was correctly reopened exactly this way.

---

---

## I. Methodology (generalizes beyond images)

1. **Metrics are hypotheses, not verdicts.** Validate any objective against ground
   truth (rank-correlation) before trusting it; re-validate when you change how it's
   used. A metric great globally can be *backwards* locally.
2. **Look at the pixels — systematically.** The eye catches what metrics miss: ±0.001
   metric deltas can be huge visually (grey vs crisp bright structures), and aggregate
   means hide localized defects (a halo is <1% of pixels). But don't hand-pick crop
   locations (that biases what you see) and don't trust the unaided eye on fidelity
   (a "cleaner-looking" result can be less faithful — see 9b). **Eye-analysis 2.0**
   (`eyetool.py`): crop where the methods *disagree most* (box-filtered |A−B|, greedy
   non-overlapping maxima) and add an amplified-difference view (5× signed diff on
   mid-grey), plus GT when available. Point the eye at the informative pixels.
3. **Clean / near-ceiling data is a mirage.** A change "free" on clean data can wreck
   hard cases. Re-check on the method's designed weakness (focus boundaries, thin/
   bright structures, real optical defocus). This trap recurred repeatedly.
4. **Profile before optimizing** — but profiling finds cost, not quality traps. Pair
   every optimization with a hard-data + visual quality check.
5. **Isolate to test.** End-to-end green hides dead stages (alignment tested only on
   already-aligned frames looks perfect even if broken). Give each stage adversarial
   input only it must handle.
6. **Don't get stuck; don't thrash.** Blocked path (flaky download, no GPU) → document
   the wall, use what you have, move on.
7. **Don't overclaim.** Report the honest quality-safe number, not the tempting one
   that silently sacrifices quality.
8. **Don't throw the baby out with the bathwater.** An outlier failing (a stray hair)
   doesn't condemn a method excellent elsewhere — add an intermediate layer / isolate
   the outlier.
9. **Scene/content-dependence is first-class.** Fine detail vs smooth gradient vs
   specular vs hard edges each want different operators/tunes — route by content.
9b. **With true GT, GT-referenced fidelity is the verdict** — not your eye's sense of
   "clean." A cleaner-looking result can be LESS faithful (softer/displaced) than a
   sharper one with a faint halo. "Look, don't trust the metric" targets NO-REFERENCE
   metrics + aggregates hiding LOCAL defects; it does NOT override GT fidelity when you
   have GT. Use both: hunt artifacts by eye, but respect GT-SSIM as truth.
9c. **Check standard methods first — then steal WHY they win.** The local/multi-scale
   principle was right, and the Laplacian PYRAMID already embodied it intrinsically,
   beating a hand-built local-scale retrofit (don't fall in love with your own code).
   But the sequel matters: understanding *why* pyramid won (multi-scale DECISION) and
   why it still failed (no edge-awareness → halos) is what produced `perband` — the
   standard method's winning property grafted into the edge-aware framework, which
   then beat both parents everywhere. Standard method → diagnose its winning property
   → transplant the property, not the method.
10. **Theory first, then verify** empirically + visually.
10b. **Scale-adaptivity belongs IN THE STRUCTURE, not in a number.** The full arc,
   resolved: fixed windows fail across resolutions (a low-res-tuned radius is tiny vs
   a high-res CoC → ranking reversals). A globally resolution-scaled window helps only
   object-scale depth splits and destroys fine details at fine-scale depth boundaries.
   A measured per-pixel structure-scale map is better but is a retrofit with an extra
   estimation step. The clean answer: make the DECISION per pyramid band with a fixed
   SMALL window — coarse bands are downsampled, so the effective scale grows with the
   band automatically. That is local scale at every location, structurally, with no
   magic numbers and no estimation (`fuse_perband`). Two correctness corollaries: cap
   any window to its band's size (a window ≈ the whole band degenerates to a global
   mean), and never plain-average the base band (it imports low-frequency defocus
   spread — weight it with the coarsest decision). Fixed-pixel operators in METRICS
   have the same disease (Q_ABF's 3×3 Sobel collapses at high-res; Q_SSIM strengthens)
   — re-validate the metric at each resolution regime.
11. Commit per milestone; keep FINDINGS.md; keep a live report; background heavy compute.

## II. MFIF domain theory

- **Pipeline:** align (ECC; global warp + depth-aware pass) → focus measure → fusion.
  Registration has its own theory section below (II-b); it is not a solved preamble.
- **Focus = high-frequency energy; defocus = low-pass.** Focus measures detect what
  defocus destroys.
- **Operators:** Laplacian (texture); **modified Laplacian** `|I_xx|+|I_yy|` (no sign
  cancellation → smooth low-contrast surfaces); gradient/Tenengrad. None universally
  best → **content_aware** routing by local contrast. Pool responses over a window.
- **Fusion ladder:** `max` (crisp, speckly) → `pyramid` (seamless, HALOS at high-
  contrast boundaries — coarse bands select defocused spread in a ring) → `decision`
  (guided-filter-cleaned selection; crisp+clean, single-scale) → `blend` (guided
  weight per pyramid band = Burt–Adelson; halo-free + multiscale) → `perband`
  (multi-scale decision + edge-aware reconstruction; default).
- **Guided filter = edge-aware smoothing without edge detection**: local linear fit
  `out=a·guide+b`, `a=cov/(var+eps)`; edge-awareness emerges from local variance;
  ~6 box filters, O(1) in radius; `eps` = contrast-that-counts-as-an-edge.
- **Resampling softens** (interpolation = low-pass) → resample ONCE, compose warps;
  don't align already-registered frames.
- **Pyramids:** pyrDown = blur (anti-alias) then decimate; `L_i=G_i−pyrUp(G_{i+1})`;
  collapse = reverse; decide per-band on a per-pixel scalar summed over channels.
- **Real defocus PSF ≈ DISK, not Gaussian** → **defocus spread** (bright OOF object
  bleeds as a disk). Gaussian synthetic defocus is too easy; build hard benchmarks
  (disk PSF + spread + thin structures + noise) or scene-dependence won't show.
- **`harden` (confidence-hardening)** unifies spread-rejection AND thin-hair
  preservation: hard-select where one frame is confidently sharpest; soft-blend where
  ambiguous. Metric-tiny, visually-large win.
- **Operator/param routing has a low ceiling (~+0.002).** Big visible wins are
  STRUCTURAL (spread/hair), not operator choice.

## II-b. Registration geometry (the alignment arc, F81–F89)

**Two motions are superimposed and they cannot be fitted together.**
- *Parallax.* A handheld rotation pivots the DEVICE, not the lens entrance pupil, so
  the camera centre translates and image displacement scales with inverse depth. Near
  content moves 2–2.5× as far as far content — measured on every moving phone sweep.
  A single affine or homography splits the difference and leaves the near plane wrong.
- *Focus breathing.* Refocusing changes magnification. Not a rounding effect: 14% on a
  12-frame phone macro sweep (bottle width 138 → 160 px), monotone.
- Both are radial-ish and a global affine fitted with both present COMPROMISES between
  them. Breathing is the depth-independent half, parallax the depth-varying half; the
  small-motion flow decomposition (rotation + breathing depth-independent, translation
  scaled by 1/Z) is what separates them in principle.
- **Fix breathing first.** It is upstream, it is a clean monotone signal, and residual
  scale sabotages every downstream object-grouping step (F88).

**Region models.**
- A depth bin is a RANGE, not an object. ECC over a region follows its majority: the
  kitchen bottle occupied 8.7% of a bin covering 55% of the frame and received +2.3 px
  where it needed +19.2. Raising acceptance caps does nothing — the correction is never
  proposed.
- Bin edges belong at depth-histogram VALLEYS, not quantiles. An equal-population edge
  cuts through one physical object and puts a many-pixel step across its own surface.
- Group by MEASURED MOTION, not depth; depth is only a seed. "Wants the same
  correction" is the operational definition of an object here.
- **Object integrity is a MERGE rule, not a shape rule.** Subdividing a rigid surface
  gives each piece its own noisier fit, so one object gets transported by different
  amounts in different places. Regions whose fitted motion agrees across the sweep are
  one object. Merge tolerance is sharply bounded above (≤2 px here): past it, genuinely
  different depth planes merge and one compromise fit is far worse than over-splitting.
- Per-bin TRANSLATION is the most constrained model that can express parallax — and it
  cannot express residual breathing at all, since a scale error moves an off-centre
  object's two edges by different amounts.

**Fields and resampling.**
- Blend the COORDINATES, not warped images: one source location per output pixel, so a
  multi-stage correction still costs a single interpolation. Compose iterations into the
  field, not the pixels.
- A sampling field may TRANSPORT content freely; it must not STRETCH it. Relax
  displacement wherever its local gradient exceeds ~0.1 — on the factory this RAISED
  GT-SSIM, proving the stretch was pure damage. Membership width cannot substitute:
  narrowing it makes stretch worse, since the same jump crosses fewer pixels.

**Occlusion at boundaries.**
- *Disocclusion* (parallax uncovers scene): the observation does not exist → hard
  per-pixel refusal is correct, and worth as much as the alignment fix itself
  (0.9660 → 0.9785 GT-SSIM on the factory). Derive the ribbon from the MEASURED
  per-region displacement, never the smoothed applied field, and size it by the step:
  a foreground moving Q px uncovers a Q px strip, so test at a ladder of radii where
  radius r requires a step of at least r. A single-scale test condemned 38% of a frame.
- *Veiling* (a defocused occluder spreads over its surround): the observation exists but
  is a MIXTURE → a hard mask loses in every regime, including one built to favour it,
  because it forces fusion onto frames where the background itself is defocused. The
  soft down-weight `harden` already applies is the right expression of the same physics.
- Both sides of a defocused silhouette are compromised, by different mechanisms: missing
  correspondence behind, matte mixing in front. Refusing only the background side loses
  to refusing both (0.9714 vs 0.9785).
- Occlusion-edge blur names the OCCLUDER's focal frame — the boundary is the near
  object's own silhouette, so its sharpness follows the foreground (Marshall, JOSA A
  1996). ~77% reliable per contour, which is useless per-pixel but decisive as ONE
  global bit (near = low index or high) voted across thousands of contours.
- Depth maps cannot LOCATE a contour: their steps sit ~32 px from the true silhouette.
  Intensity edges whose two sides — sampled ~10 px out along the normal — differ by
  ≥1.5 focal frames land within ~5 px.

**Measuring object motion.**
- Textureless interiors have nothing to correlate; EDGES carry the motion, and a rigid
  object's flat interior inherits it. Integrate along the edge's length to turn a weak
  local match into a strong 1-D measurement.
- Correlate GRADIENT profiles, not intensity: defocus spreads a bright object over its
  surround and biases the apparent edge outward on both sides.
- Aperture problem: trust only the component along each edge's normal, and combine
  differently-oriented edges around one outline for full 2-D motion.
- Phase correlation resolves shifts up to roughly a QUARTER of the patch. A 32 px patch
  silently saturates at ~8 px, reporting a confident wrong number.
- **Interior edges make "one object?" falsifiable.** Under magnification a rigid
  object's edge displacement is LINEAR IN X, so interior edges over-determine the fit:
  11 edges agreeing to 0.33 px rms is positive proof of one object, and translation and
  magnification come out separately identified. Two edges alone can never test this.
- Measure near an object's focal plane where interior detail survives, then propagate
  along the sweep — linear extrapolation reached +18.88 px against +19.2 truth where
  direct measurement in the blurred frame was hopeless.

**Evaluation caveat specific to registration.**
- No-reference fusion metrics CANNOT adjudicate an alignment change: they score the
  fused image against its aligned sources, and alignment changes what the sources are.
  Cross-scoring collapses both variants to ~0.72–0.82. Judge alignment by per-depth-
  region registration residual, an analytic parallax factory, and disagreement crops.
- Alignment cannot be tested on frames differing by one global transform — a global
  aligner is exactly right on those. The test stack must have near and far content
  moving by DIFFERENT amounts (`research/parallax_gen.py`).

## III. GT-free evaluation

- No-reference metrics: **Q_ABF** (gradient transfer), **Q_SSIM** (to sharpest source;
  best single), Q_CB; **REJECT Q_MI** (anti-correlates — rewards ghosting/speckle).
  Calibrate composite weights vs GT by rank-correlation (`0.3·Q_ABF+0.7·Q_SSIM`,
  Spearman +0.72).
- **Global-calibrated composite is WRONG for per-region decisions** (Q_ABF
  anti-correlates per-tile). Use Q_SSIM / content-routed metric locally.
- Per-tile no-GT discrimination is inherently hard → learn from GT dev-labels, deploy
  feature-only (no answer key at inference). Metrics go blind on smooth content.

## IV. ML strategy

- Classical foundation FIRST; learning rests on and initially trails it.
- Learned per-tile routing (small numpy MLP) matches the oracle even at modest
  accuracy (tunes near-equivalent where confused).
- Self-supervised CNN (gradient-retention loss, no GT) feasible but trails classical.
- **Distillation matches classical quality in one pass; speed is a GPU story** (CPU
  opencv engine is already fast).

## V. Environment (verify each session)

- Py3.14: only numpy + opencv-headless wheels; scipy/sklearn/skimage/torch absent →
  numpy self-impl, or `uv` → py3.12 + CPU torch (`.venv312`). No GPU.
- Data: Real-MFF (710 GT, gdown, ships RAR → build unrar from source if no sudo);
  Lytro (real optical defocus, yuliu316316/MFIF); MFFW/UHD-MFF not cleanly available.
  **REAL deep/handheld/photographic stacks — the recurring bottleneck — are now
  cataloged in `research/REAL_DATA.md` + fetched by `research/realdata.py`.**
  `mobiledepth` (13 real phone sweeps, N=12–41, no AiF GT) is IN-TREE now; `iphone12`
  (Learn2Refocus, N=9, 4K, +pseudo-GT), `learn2af` (N=49, real, 870 GB), `araujo`
  (raw bursts +pseudo-GT) are scripted/documented there. `research/data/` is gitignored,
  so a pull persists across branch checkouts in this working tree.
- High-res speed: subscale ONLY the guided-filter (`weight_scale`), keep focus/
  confidence full-res or thin structures grey out. `--fast` ≈ 1.5x but costs
  roughly 0.005–0.025 GT-SSIM versus the perband default (CPU tradeoff, not a
  quality-safe path).

## VI. Traps hit (skip them)

1. End-to-end green hid untested alignment (fed already-aligned frames).
2. Global sharpness rated all fusion methods equal; halo invisible in the mean.
3. Q_MI anti-correlated with truth.
4. `content_aware` default broke max/pyramid (needs cross-frame; per-image focus
   raises) → `_focus_maps` branch.
5. Naive weight-subscale: clean data "free", hard scene greyed wires. Subscale guided
   smoothing only; keep focus/conf/decision full-res.
6. `inspect.py` shadows stdlib `inspect` → numpy crash. Don't name scripts after stdlib.
7. `multiprocessing` from a heredoc (`<stdin>`) fails → use a script file.
8. numpy 2.x removed `ndarray.ptp()` → `np.ptp(x)`.
9. A 1-D correlator zero-padded in the TIME domain reports shift/pad-factor with the
   sign flipped — and looks entirely plausible. Known-answer test every correlator.
10. `cv2.medianBlur` needs uint8 for large kernels; `phaseCorrelate` needs contiguous
   float64 and a matching Hanning window.
11. Locating an edge by contrast/gradient saturates by ~2 px of blur and is swamped by
   texture on real frames — it is not a blur estimator. Use focal distance instead.

## VII. Fast start

Default engine: `perband` + `content_aware` + `harden=0.5`; `--harden 0` disables
spread rejection; `--fast` trades measured quality for speed. `.venv` (3.14) for
the engine, `.venv312` for torch. Before any
"it works": isolate-test the stage, check hard data + eyes, distrust clean-data and
global-metric verdicts, use Q_SSIM (not the composite) for local decisions.

---

# Provenance — how each conclusion was established

Below is kept deliberately. A rule without its conditions becomes a trap, and a
negative whose conditions have changed is a hypothesis again (see the closing note).
Read for the *why* behind a §0 line, or before reopening anything settled.

## E-phase additions (boundary/reconstruction arc)
- **Oracle ladders turn mystery negatives into understood ones** — each rung removes
  one explanation (estimation → noise → PSF → model). Run the ladder BEFORE building
  estimators; the ceiling decides whether estimators are worth building at all.
- **Mixed-oracle rungs decompose two-factor failures in one experiment** (true-value/
  est-mask vs est-value/true-mask isolated the matte SUPPORT as the killer, F35).
- **Matte thinness must come from CONTENT (difference vs estimated plate), not
  detector support width** — energy support is always fatter than the structure.
- **A reconstruction that assumes a formation model must gate on evidence the model
  holds** — focus dominance ≠ occlusion (every in-focus region dominates); demand veil
  evidence. Benchmark-matched wins do not transfer; cross-generator gates are the
  honest test (F36).
- **Orthogonal evidence channels only** — a new detector fed by the same signals your
  engine already senses adds correlation, not information (parallel vectors). Semantic
  priors (monocular depth, learned segmentation) see what local math cannot: through
  defocus, through camouflage, closed topology (F31/F32).

## Gate-building playbook (specialist routing arc, F43–F47)
1. **The unified specialist-gate recipe**: (a) candidates from REGIME-MATCHED matting;
   (b) features must include matte-edge quality (transition-shell sharpness,
   silhouette-on-edge alignment) — the safety signal for anything that stamps edges;
   (c) ridge-regress the ACTUAL outcome (delta global) from factory GT — never a
   quality proxy (a small mean matte error can hide a misplaced edge); (d) fire margin
   chosen ON TRAIN as (worst harmful prediction + eps), verified on held. This
   locked contour reconstruction; the veil gate was later retired by F54 because
   its factory/labels omitted the failure axis.
2. **Predict outcomes, not intermediates** — third confirmation of the pattern
   (metric calibration, gate labels, threshold selection). If the factory holds GT,
   outcome labels are free; use them.
3. **Per-candidate granularity multiplies statistics**: 8 scene-level labels became
   366 candidate-level labels from the same data. Gate at the unit you fire at.
4. **Label factories**: generators with GT are unlimited-label machines — scale the
   factory before tuning the model (42→100→120 scenes fixed what thresholds couldn't).
5. **Regime-matched mattes** (F46): edge-stamping needs pixel precision; smooth-field
   subtraction tolerates region precision. Match specialist ↔ regime ↔ matte class;
   cross-regime firing without a precision feature is harmful (worst −0.086).
6. **TRAP: no-reference source-similarity metrics cannot audit synthesis** (F45) —
   a correction whose success means deviating from every source (de-hazing) is
   scored as damage by q_ssim-family checks; they revert GT-verified wins.
7. **TRAP: semantic models need natural content** (F43) — pastiche/synthetic blobs
   fragment under SAM; photo bokeh spoofs objectness. Benchmarks judging semantic
   components must be built FROM real objects (objects-as-occluders pattern:
   model-generated cutouts double as GT silhouettes).
8. **Cross-regime tests are adversarial theory checks** (F46): running a specialist
   outside its regime and watching it fail EXACTLY as theory predicts is
   confirmation, not waste.
9. **Margins price recall in units of effect size**: small-effect specialists fire
   rarely under the same safety bar — expected physics, not gate failure. Report
   coverage per regime so correct refusal isn't misread as timidity.
10. **Ops for long labelers**: flush caches and print progress PER UNIT, not at the
    end — monitorability and interruption-safety are part of the method.

## Scene-recovery arc additions (F51–F56)
1. **Do not infer a scene deficit after fusion unless every contributing transfer
   function is modeled.** The historical veil law `coef = G − w_far` corrected the
   far remnant but ignored scale-dependent background evidence already admitted from
   other frames; it double-restored realistic structure. Correction-after-fusion is
   retired. Solve the multi-frame layer equations jointly or refuse.
2. **Gain laws are measured, not assumed.** Post-correction attenuation ≠ the forward
   model's (subtraction already partially restores; fusion dilutes) — measure the
   residual curve on factory GT and invert THAT. Assumed 1/(1−ab) overshoots 3-5x.
3. **Amplified-remnant denoising wants the ANALYTIC threshold.** When the gain field
   and sensor σ are known, per-band expected noise is computed, not estimated —
   soft-threshold at m·σ·c_k·coef (c_k calibrated once through the actual pyramid).
   Guided filtering with the degraded signal as its own guide LOSES (weak-guide trap):
   the guide's authority must exceed the noise being removed.
4. **TRAP: FFT deconvolution without replicate-padding** → circular-wrap stripes at
   borders (eye-confirmed; fixing it RAISED the score — the artifact sat inside the
   eval band). Pad ~4r, crop back. Applies to RL and Wiener alike.
5. **TRAP: selectors comparing across regularization strengths are degenerate without
   per-strength calibration** — re-blur residual picks under-deconvolution always
   (reblur∘deconv → identity as r→0); Levin 2007's learned per-scale weights exist
   for exactly this. Outcome regression (F47) is our native fix.
6. **Median metrics are blind to sparse outliers** — the eye caught speckle the
   per-band median called calibrated (H4 trigger). Pair every median with a
   worst-case or a look before declaring a band clean.
6b. **A metric's EXCLUSION FILTER is a blind spot in disguise** (F53): contrast_ratio
   excluded textureless-GT pixels — exactly where hallucinated texture lands; a
   user's eye caught what the whole bench missed. For every selective metric, build
   its complement (the false-texture index measures precisely the excluded pixels).
   Corollary: the forward model must match the RENDER (chromatic per-channel radii)
   — a channel-shared model turns aberration into amplifiable residual.
6c. **An oracle win is conditional on the FACTORY CLASS, not just oracle inputs**
   (F54). True matte/radius on four blob scenes did not license realistic objects.
   Before promotion: cross model families, run native resolution, report tails, pair
   every selective metric with its complement, and include identity/refusal as a
   candidate. A learned gate cannot rescue a negative realistic oracle ceiling.
6d. **Forward fit cannot detect support errors shared by every inverse model** (F56).
   A semantic matte that omits true foreground can still achieve excellent
   re-degradation by explaining those pixels as background inside blur's null space.
   Partition metric disagreements spatially by physical ownership. Then add an
   independent observation: focus dominance vetoes recovery where the captured stack
   decisively assigns the pixel to the foreground owner. Regularizer/PSF consensus
   handles solver uncertainty; it does not substitute for support uncertainty.
6e. **Metric disagreement is a localization request, not a referendum.** P8's
   SSIM-only dissent concealed direct-error wins; P9's holdout MSE dissent exposed a
   real foreground-support leak. Keep every vote, map where each error changes, repair
   the mechanism if the tail has physical structure, and rerun the entire lattice.
6f. **Global metric unanimity can still hide a consistent physical-partition
   regression** (F57). The owner lab found foreground MAE worsening on all five
   diagnostic fires even though SSIM, global MAE/MSE, PSNR, and fringe L1 all
   improve. Every synthesis checkpoint must report foreground, target fringe,
   and far-background partitions; changed support outside each; and the
   maximum-regression crop. “All metrics pass” is not closure until the metrics
   cover the physical regions the operator can damage.
6g. **When one frame observes foreground and another reveals background, soft
   fusion is the worst ownership decision** (F58). Do not ask a downstream
   restoration specialist to reinterpret an already mixed pixel. Inspect the
   sharp owner observation for missing semantic fragments, license each fragment
   by re-rendering the captured frames, then make the narrow discrete choice:
   copy observed owner foreground and veto background recovery there. Freeze the
   physical-fit margin before generating the validation extension; otherwise the
   “holdout” is only another development set.
6h. **A containing owner-frame mask is an ordering observation, not a duplicate
   matte** (F59). If a sharp-frame silhouette contains nearly all of the mixed-base
   seed and adds a nearby tail, the tail is evidence that the sharp object stands
   in front of what another focal frame reveals. Test that parent hypothesis in the
   captured-frame formation model, require a stronger relative gain than a small
   satellite, and hard-select only the novel support. Never feed a known opaque
   owner core to a veil specialist and ask regularization to rediscover ownership.
   Freeze the containment/IoU/relative-fit rule before generating its extension,
   and include a causal support-off comparison: pooled benchmark wins alone do not
   establish that the mechanism fixed its intended failure.
6i. **Sharp alpha is not frame-specific optical coverage, and a named PSF is not
   evidence that the implementation uses it** (F60). For layered defocus, save
   the blurred coverage for every frame and grade complete core, inner partial
   occlusion, outer veil, and far background separately. Unit-test the renderer
   against an aperture average; inspect its large-radius branch; clean cutout
   radiance independently of its matte. A benchmark that hides coverage can make
   valid partial occlusion look like bad transparency, while a box shortcut can
   make invalid inputs look like disk optics. When the judge changes, prior
   promotion scores become provisional even if the mechanism still looks sound.
6j. **Absence of foreground evidence is not positive rear visibility** (F61).
   Encode occlusion ordering as two asymmetric observations: a focused front
   owner vetoes rear recovery; a defocused front permits it only where another
   frame positively observes focused rear structure. Intersect that gate with
   conservative cross-PSF coverage and decay to identity near model disagreement.
   For a small proposed support fragment, keep an absolute whole-frame
   re-degradation margin but measure relative improvement on its PSF-dilated
   optical influence neighborhood. A whole-image percentage is the wrong unit
   for a local hypothesis. Close spatial ownership and false-texture tails as
   separate properties; a clean partition lattice does not waive a fine-band
   complement warning.
6k. **Repair front geometry before asking a rear specialist to invert a mixed
   pixel** (F62). A matte segmented from the fused base can overextend or erode
   because the failure is already baked into its input. Let a strongly
   overlapping sharp-owner silhouette replace that matte only when the captured
   formation model improves by an absolute margin. Then hard-select the observed
   front layer only in its eroded interior where independent PSFs agree the
   other focal frame is partially covered; run rear recovery afterward. Do not
   reuse a license for a mask's novel tail as permission for its overlapping
   interior—the hypotheses have different support and need different evidence.
   Always generate another post-rule split: S16's fresh-split fire falsified the
   older rear stage even though the new front mask had zero pixels there, which
   prevented a wrong causal conclusion. Because that counterexample then shaped
   F62, it became development evidence; S19 was generated only after the final
   rule froze and supplied the actual post-final validation.
6l. **A rear-focused boundary observation is not a clean rear plate** (F78).
   Near a foreground veil it contains additive foreground radiance *and*
   multiplicatively attenuated background. Recover the direct low-frequency
   rear estimate as `(O_rear - predicted_front_veil) / T_rear` only where
   transmission is usable. Select among plausible PSFs by the local two-frame
   forward residual; averaging a good disk inverse with a bad box inverse is
   not conservatism—it preserves the wrong subtraction. Center integration on
   a meaningful aperture-coverage/rear-transmission contour, not on the farthest
   nonzero PSF support. Support answers “can the lens contribute here?”; the
   contour answers “where does the recovered field need integration?”
6m. **Confidence is component-specific in an inverse problem** (F78). High
   confidence in the baseline under-veil background does not imply confidence
   in one weak dark transition. Test the proposed component itself: forward it
   through the veil and compare its observation-domain magnitude with the local
   noise floor. If it should have remained visible, absence vetoes inference;
   only a censored component may borrow an exterior trend. Extrapolation is a
   below-noise fallback, never a general background replacement. Apply seam
   regularization to the relative correction field along the local contour
   normal. Circular absolute-image filters and unconstrained normal-slope
   continuation are shape-blind band-aids.
6n. **Formation coefficients are upstream state, not post-fusion estimates**
   (F78). Solve `V_front`, `T_rear`, PSF choice/weight, forward residual, and
   detection floor while the original focal observations and geometry coexist.
   Pass those maps to the integrator. Recomputing them inside a recovery function
   is acceptable only while that function still owns the originals and the same
   formation model; inferring them from the fused base is structurally
   underdetermined and recreates the retired correction-after-fusion failure.
7. **Early stopping IS a regularizer with a measurable turnover** — RL fidelity
   peaks then falls while contrast still rises (k=40 worst < 0): a rising internal
   number while GT fidelity falls is the noise-fitting signature.

## Negatives are conditional (Farron, after reopening F27)
A rigorous negative is a statement about its TEST CONDITIONS, not about the idea:
record the conditions with the verdict (F27 did: 8-bit, thin structures — and wrote
its own revisit clause). At every checkpoint, sweep standing negatives for triggered
revisit clauses — pipelines, factories, and metrics evolve, and a negative whose
conditions changed is a hypothesis again. Corollary: when the mission is to recover
the TRUE SCENE (beyond the best mix of what the camera saw), the evaluation apparatus
itself is part of the method — an idea can only be as good as the bench's ability to
credit it (17a: pseudo-GT ceiling; F45: no-ref blindness).
