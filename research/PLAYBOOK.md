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
  weight per pyramid band = Burt–Adelson; halo-free + multiscale; default).
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
  confidence full-res or thin structures grey out. `--fast` ≈ 1.5x quality-safe (CPU ceiling).

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

Default engine: `blend` + `content_aware`; `--harden 0.5` for bright/thin structures;
`--fast` for high-res. `.venv` (3.14) for the engine, `.venv312` for torch. Before any
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
