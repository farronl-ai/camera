# focusstack — Expert Playbook (LOADME)

Distilled from building this engine end-to-end. Read this + `FINDINGS.md` (F1–F16)
+ the root `README.md` to start at expert level. This is methodology and hard-won
nuance; `FINDINGS.md` is the dated experimental log.

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

- **Pipeline:** align (ECC; correct focus-breathing) → focus measure → fusion.
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

## VII. Fast start

Default engine: `perband` + `content_aware` + `harden=0.5`; `--harden 0` disables
spread rejection; `--fast` trades measured quality for speed. `.venv` (3.14) for
the engine, `.venv312` for torch. Before any
"it works": isolate-test the stage, check hard data + eyes, distrust clean-data and
global-metric verdicts, use Q_SSIM (not the composite) for local decisions.

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
