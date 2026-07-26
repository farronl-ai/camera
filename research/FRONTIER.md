# Frontier — living inventory of unexplored directions

The antidote to closing in after a convergence: every direction we have NOT yet
explored, with status. Maintained like FINDINGS.md — update when a direction is
probed, promoted, or retired. If this file hasn't changed in a while, that itself
is a warning sign.

Status: ⬜ unexplored · 🔶 probing · ✅ resolved/promoted · ❌ blocked (documented)

| # | Direction | Why it matters | Status |
|---|-----------|----------------|--------|
| 1 | **N-frame stacks (N>2)** | Nearly all validation was 2-frame; real stacking = 5–50 frames. | ✅ F24: NOT a defect — quality RISES with N (denoising); "dilution" is beneficial, top-K hurts; perband best ∀N. Sub-case 1b still open. |
| 1b | **Extreme defocus-spread across a deep stack** | Bright point sources at depth. | ✅ F29: no defect — N=8 matches N=2 under maximal spread stress; harden persists (mildly attenuated). |
| 2 | **Real N-frame optical data** (BBBC006 microscopy z-stacks) | Real optical defocus. | ✅ F25: got 4 real 3-plane stacks; perband visibly sharpest, pyramid softens/halos nuclei. Microscopy domain covered; macro/photographic real still open (2b). |
| 2b | **Real photographic/macro deep stacks** | Microscopy ≠ everyday photography content; the photographic real-optical gap persists (MFFW/UHD blocked). | 🔶 DATA IN: `research/REAL_DATA.md`. `mobiledepth` in-tree (13 real phone sweeps, N=12–41, no AiF GT); `iphone12`/Learn2Refocus (N=9, 4K, +pseudo-GT) documented; `araujo` scripted. Photographic gap NARROWED; true macro/product still open. |
| 3 | **Occlusion-boundary physics (α-matte)** | Real depth edges mix fg/bg semi-transparently. | ✅ F25: built layered α-matte defocus generator; re-ranked — perband crown WIDENS (blend worst under honest occlusion). Synthetic conclusions not artifacts. Fusion itself is still occlusion-UNAWARE (3b). |
| 3b | **Occlusion-AWARE scene recovery** | Joint layer inversion (de-veiling beyond camera-mix). | ✅ F76 NARROW SHIP: exactly two-frame, validated-size one-sided opaque recovery is enabled in `--enhance auto`. Frozen post-rule S29 formation passes 12/12; blind geometry licenses 11/refuses one with mean IoU 0.96566 and zero protected-region rear application. All 11 fires improve MAE/MSE and every physical partition with exact far identity; GT-side artifact inspection clears the finest-band diagnostic dissent. N-frame/general-CoC/transmission remain separate frontier rows. |
| 3c | **Transmissive-layer recovery** | A transparent foreground and rear scene transform differently across focus, preserving separability beyond ordinary fusion. | 🔶 F63 design: add distinct geometric-coverage and extinction fields, save both latent layers, establish an oracle scalar-transmission ceiling, then attempt blind formation routing and opacity/layer estimation. Do not weaken opaque complete-core ownership to approximate transmission. |
| 4 | **Per-band Q_ABF metric** | Fix Q_ABF's high-res collapse structurally. | ✅ F26: q_abf_ms (mean-pool) recovers +0.11→+0.78 at high-res; new composite best at BOTH regimes (+0.785/+0.869). Adopted. |
| 5 | **Depth map byproduct** | Free feature from the fusion decision. | ✅ F26: --depth-out shipped; r=0.59 on textured pixels (texture-only observability — documented limitation). |
| 6 | **Exposure/WB drift between frames** | Real capture drifts brightness/color. | ✅ F28: drift costs −0.025 SSIM; per-frame gain to stack-median means (mean is blur-invariant) recovers to −0.002; near-identity gate passed → default-ON (--no-normalize-exposure). lit-scan 2026-07: EDMF (Sensors 2024) = 1000 REAL phone pairs with genuine exposure differences, public — the in-the-wild test F28 never had (→ L9). |
| 7 | **Alignment robustness on real handheld deep stacks** | ECC is a local optimizer aligned to the middle frame; feature-based init / chained alignment for large displacement. | ⬜ DATA IN: `mobiledepth` `Figure6/{large,small,zero}motion` are a ready handheld align-stress set (real optical, N=14), + all 13 real phone sweeps. See `research/REAL_DATA.md`. lit-scan 2026-07: WHU-MFM gives the first LABELED misaligned-stack factory; HDR+-style robust merge + TAF per-stack flow are the two method levers (→ L3/L4/L5). |
| 8 | **Focus breathing across deep stacks** | Scale change accumulates over 10s of frames; interacts with #7. | ⬜ lit-scan 2026-07: TAF (SIGGRAPH Asia 2023) models breathing explicitly as a per-stack neural flow field fitted by re-rendering the stack — the concrete design spec for this row (→ L5). |
| 9 | **Noise-adaptive fusion** | Low-light stacks: focus energy vs noise energy confusion; denoise-aware weighting. | ⬜ |
| 10 | **Learned/semantic segmentation layer** | Object-aware regions beyond structural watershed (torch env exists; CPU-heavy). | ⬜ deferred |
| 11 | **GPU distillation speed** | Distilled CNN matches quality (F14); the speed win needs hardware absent here. | ❌ no GPU |
| 12 | **Learned no-reference metric** | Train a per-tile quality predictor on GT dev-labels to close the 19% no-GT labeling gap (F6) — the metric-side analogue of M3. | ⬜ |
| 13 | **Real camera capture protocol** | Farron shooting an actual bracketed stack (phone/camera + rail) would give first-party real data end-to-end. | ⬜ needs user — but LESS urgent now: real phone/iPhone stacks are cataloged in `research/REAL_DATA.md` (`mobiledepth` in-tree; `iphone12`/`learn2af` documented). First-party still uniquely fills the macro/product gap. |
| 14 | **Video / temporal focus stacking** | Fuse focus sweeps from video; temporal coherence constraints. | ⬜ lit-scan 2026-07: UNBLOCKED on the data side — VF-Bench (NeurIPS 2025 Spotlight) is the first multi-focus VIDEO fusion benchmark, with temporal-consistency metrics + a flow-warped baseline (→ L10). |
| 15 | **UHD-MFF & MFFW datasets** | Real+hard benchmarks; not publicly downloadable at last check. Re-probe occasionally. | ❌ lit-scan 2026-07 re-probe: UHD-MFF paper now published (ECCV 2026, learnable-LUT method; 150 real 4K pairs + 1800 synthetic) and official repo `github.com/zyb5/UHD-MFF` EXISTS but is a code stub — no data yet (MIT, author contact in README; watch it). MFFW still ResearchGate-only. Meanwhile three downloadable alternates landed: WHU-MFM (L3), MattingMFIF 4K (L6), EDMF (L9). |

| 16 | **Boundary Engine (E-phase)** | True object boundaries + near-side occlusion tags from stack evidence ∪ learned appearance; additive integration into guided/perband/harden. THE current push — plan: NEXT_STEPS_boundary.md. Relates to 3b/10/12. | 🔶 E4 done as rigorous negative (F33): decision-side integration cannot fix boundary error — it is coefficient-contamination (reconstruction physics). Boundary engine (fused F=0.55) stands as data product. Successor lever: matte-aware / supersampled boundary RECONSTRUCTION (16b). |
| 16b | **Matte-aware boundary reconstruction** | The hard-lines lever per F33. | 🔶 F34/F35/F36: ceiling −22%; buildable C3 −16% + global WIN on matte-model data; regresses off-model → mechanism ships behind the contour gate; ungated `--reconstruct-boundaries` is deprecated reproduction-only. |
| 16d | **Weight-scaled in-loop veil correction** | F54 overturned the narrow-factory validation: realistic object scenes fail even with oracle matte/radius; native fires are microscopic and fail fringe/false-texture tails. Operator retired; successor is joint two-layer inversion, not correction-after-fusion. | ❌ |
| 16e | **Semantic matte input** | F62: the sharp-owner bank is now allowed to replace an overextended mixed-base matte after same-object association and an absolute forward-fit win. Its partial-coverage interior is directly reconstructed before rear recovery. The prior hierarchical/child-mask gap flips an independently harmful S16 fire positive on every partition; 35/36 refuse. | ✅ narrow mechanism validated; runtime path still disabled with 3b |
| 16c | **Reconstruction applicability gating** | Fire only where the matte model holds. | 🔶 F38: plane-step ribbon (raw-winner median + energy floor) shipped — strongest on-model result (−16% e2, +global, all scenes), real-data neutral + fire moved to silhouettes. REMAINING: F39 killed veil-side licensing (thin structures have no strong veil — category mismatch); next license is SILHOUETTE-side (owner-frame contour sharpness + plane gap); iphone12 GT verdicts gate promotion. |

Near-term execution order: 1 → 2 (parallel) → 3 → 4 → 5 (plan: NEXT_STEPS_breadth.md).

## E-arc close ledger (2026-07-20)
- **16c reconstruction gate: ✅ LOCKED** (F47) — home regime (thin + C3 mattes), held
  5/30 fired all-positive mean +0.0083. Real-data promotion awaits iphone12 GT.
- **16d correction-after-fusion veil correction: ❌ RETIRED** (F54) — the earlier
  held 2/2 property was a benchmark blind spot. Native expanded audit breaks the
  global/fringe/false-texture property; auto enhancement never calls that operator.
- **19f joint-layer successor: ✅ NARROW SHIP** (F56) — exactly two frames, licensed
  giant CoC, max side <=1600. It solves the captured layers jointly and is a distinct
  operator, not a re-enable of 16d.
- **16e semantic matte: 🔶 resolved by regime-matching** (F46) — mask mattes serve the
  veil specialist; C3 serves reconstruction; SAM-quality matting for GENERAL scenes
  remains the recall-growth lever.
- **16f: ✅ SHIPPED, RESCOPED (F48/F54/F76)** — `--enhance auto` retains gated
  contour reconstruction. F54 removed the old correction-after-fusion veil
  path; F76 promotes the separately validated one-sided joint-layer specialist
  only for two-frame auto enhancement, with identity on every nonlicensed path.
- **17a (Farron, 2026-07-22): the pseudo-GT ceiling** — external "GT" benchmarks
  (iphone12 = Helicon output) inherit the toolmaker's ceiling: selection-made
  references CONTAIN the veil, so they score veil removal as ERROR. Consequence:
  the iphone12 gate validates the generalist + never-harm property, but can NEVER
  credit synthesis corrections — only our analytic factories and the observation-
  domain audit (17/L1) can. Head-to-head vs Helicon/Zerene must use factory GT +
  eye, never their pseudo-GT, on veil scenes.
- **17 (NEW, born from F45): synthesis-aware no-reference metric** — a no-ref signal
  that can audit corrections which deviate from every source (current q_ssim-family
  structurally cannot). Would unlock runtime self-audit + recall growth safely.
  lit-scan 2026-07: two concrete candidates found — forward-model re-degradation
  audit (L1) and inpainting-difficulty scoring (L8).
- **19 (Farron, 2026-07-22, updated): the division revisit — scene recovery, float NOT a prerequisite.**
  MISSION REFRAME: the goal is to produce the REAL OBJECTS being captured, not the
  perfect mix of camera-received focuses. Under that goal, F27's bench tested the
  wrong thing — raw inversion alone — when the question is the best scene-recovery
  SYSTEM we can engineer. Subtraction-with-estimated-inputs has a structural ceiling
  (removes additive haze; cannot re-amplify surviving detail contrast, still scaled
  by 1−α_blur); only multiplication restores amplitude. The amplification noise that
  killed raw division is STRUCTURED and SURROUNDED BY CLEAN EVIDENCE — addressable
  even at 8-bit with our existing toolkit:
  (a) per-band SNR-weighted amplification (full gain at coarse/mid bands where
      averaging gives headroom; tapered at fine bands — the F40 in-loop idiom);
  (b) surround-informed denoising of the amplified band: guided filtering with the
      un-veiled neighboring content and/or the owner frame as guide (the geometry of
      the surviving detail is known from the attenuated signal + the matte);
  (c) fringe-clamped, gate-protected application (the 16d safety stack unchanged).
  POST-GAIN CLEANUP (Farron): amplified quantization/noise has a DETECTABLE
  signature — a sudden excess of very-fine-grain HF that does not match the
  should-exist HF implied by the true object's structure. Two analytic levers:
  (1) the gain field is OURS (1/(1−α_blur)) ⇒ the expected noise amplitude map is
  computed, not estimated — per-pixel-σ shrinkage (R4, upgraded) with zero blind
  estimation; (2) cross-scale coherence: real edges carry aligned energy across
  bands; artificial grain is finest-band-isolated ⇒ shrink incoherent fine-band
  energy only, guided by mid-band/owner structure. Simple-first per the ladder;
  ML/AI admitted if simple saturates. Goal: make the 8-bit case truly clean.
  EXPERIMENT (giant-CoC wideocc factory, GT-credited, 8-bit first, float as a
  second condition): 16d subtraction alone vs hybrid (subtract + regularized,
  surround-denoised contrast restoration). Success = strong-veil-band detail
  contrast approaches GT with no off-band harm; gates retrain on hybrid outcomes.
  If the hybrid wins at 8-bit, F27's epitaph reads: the idea was right, the system
  around it was missing. FIRST STEP (per MISSION ladder): a focused litscan on
  CLASSICAL restoration under structured amplification — delegated, in flight.
  **STATUS 2026-07-25: ❌ NARROW-FACTORY RESULT, GENERAL CLAIM WITHDRAWN (F54).**
  Historical F51 result: hybrid = deficit-form
  clamped gain (coef = G − w_far, sq law) + analytic shrink (σ known, c_k calibrated).
  Float, home regime: +0.0019 mean vs subtraction, ALL 10 backgrounds positive, contrast
  ratio →1.000, off-band clean — the full FRONTIER-19 success criterion. 8-bit: mean
  +0.0007, worst −0.0015 (3/10 small losses; all flip positive in float). F27's epitaph
  written at the time. F54's objects-as-occluders audit then found semantic-matte +
  true-radius mean −0.00319/worst −0.03041, and true-matte + true-radius spot checks
  still lose to −0.032. The operator assumes the far remnant is the only surviving
  background evidence and double-restores structure already contributed by other
  frames. Auto subtraction was separately retired because its four native
  fires are microscopic and fail expanded fringe/false-texture tails.
  Retired/redirected sub-frontiers:
  - **19b: gate retrain on HYBRID outcomes** (F47 recipe) — required for ship: off-regime
    (moderate CoC) ungated harm −0.0012 mean (F46's third rhyme). Needs a ~100-scene
    wide-occluder outcome factory; merge with #18 recall work.
  - **19c: input-side structured quantization** (the residual 8-bit wall) — H5 decomposition
    exonerated output bit depth; R5/MAP-AC debanding is the open conditional rung; the
    gate (19b) is the nearer path for the −0.0015 worst case.
  - **19d: blind-matte hybrid** — all F51 results are oracle-alpha; the semantic-matte
    chain (16e/F42) gates real-data promotion, as for every specialist.
  - **19e (from F53, user-caught): chromatic veil model + false-texture instrument** —
    per-channel D and pm-residual removal fix real terms but do not cure the
    model-class error. The false-texture index permanently joins the bench.
  - **19f (F54 replacement, ✅ F56 NARROW SHIP): joint two-layer inversion** — solve all focal-frame
    formation equations simultaneously for foreground premultiplication and sharp
    background, then render the all-focus scene. Full P1 is the first positive
    realistic-object ceiling: mean +0.00408 GT-SSIM, 99/100 positive, worst
    −0.000067, mean fringe error −7.48 gray. P2's regularization-consensus
    projection closes the global tail (100/100 positive, mean +0.00383, worst
    +0.000053) and improves the false-texture tail to +0.065 gray, but one
    +0.35-gray fringe miss remains. It proves the model class is materially
    better than correction-after-fusion, not that it is safe. P3 semantic-matte
    + true-radius is a hard negative (mean −0.00455, worst −0.0430; only 16/98
    positive): current mattes are too boundary/owner-inaccurate. Matte-displacement
    consensus is also negative. P4 physical reranking finds 53/98 good-alpha
    candidates versus 28/98 semantic top-1, but only 24 outcomes are positive.
    A four-feature development license isolates 7 all-positive cases; thresholds
    survive a fresh 25-scene holdout (2/2 fires positive after consensus:
    +0.00303/+0.00075 GT-SSIM). Broad radius consensus is not safe. P6's fixed
    3.5% giant hypothesis plus unchanged gate fires 9/9 positive at 512 and
    refuses all 54 moderate scenes. Native scale-aware P7 keeps fresh holdout
    2/2 positive and all 9 fringe errors improved, but two development global
    SSIM tails remain. P8 cross-PSF consensus keeps fresh native 2/2 positive;
    the two dissenting dev SSIM rows improve global MAE/MSE/PSNR and changed-
    pixel fidelity, with no visual pattern extension. F56 then localized a fresh
    holdout MSE tail to true foreground omitted by the semantic matte: forward fit
    had absorbed the support error into blur's null space. A smooth focus-ownership
    veto closes that leak. The exact package is now 7/7 development, 2/2 first
    holdout, and 1/1 second untouched holdout positive on SSIM, MAE, MSE/PSNR, and
    fringe L1; it refuses all 66 moderate scenes. Shipped scope is two frames,
    3.5%-image giant CoC, max side <=1600, bridge + licensed candidate; everything
    else is identity. F58 repairs detached owner-frame fragments; F59 adds the
    asymmetric-occlusion parent case. A parent silhouette must contain >=90% of
    the licensed seed at >=0.80 IoU and improve the captured-frame fit by >5%
    (and >0.01 absolute) before only its novel observed pixels can override mixed
    fusion. All ten established fires retain positive direct outcomes. On the
    post-rule scenes 175–199, both licensed cases accept a parent; `scene_178`
    flips from harmful on SSIM/MAE/MSE/fringe to positive on all four.
  - **19g (spawned by F56): generalize without weakening refusal** — N>2 joint layer
    equations, a bounded CoC bank wider than the fixed giant model, >1600 px tiled/
    multigrid solving, and multiple disjoint occluders. Each axis needs its own
    factory/holdout; do not extrapolate the narrow license.
  - **19h (spawned by F56): real optical truth for recovery** — first-party macro/
    product capture with physical target geometry or controlled occluder removal.
    Real stacks without latent-scene truth can audit refusal/artifacts and forward
    fit, but cannot certify recovered texture.
  - **19i (spawned by F57, ✅ SPATIAL TAIL CLOSED F62; FINE-BAND OPEN): support-tail projection** — two discrete
    support failures are repaired. Detached satellites use the >0.01 absolute
    observed-fit license; high-overlap parent silhouettes additionally require
    >=90% seed containment, >=0.80 IoU, and >5% relative fit improvement. Both
    hard-select captured foreground and veto veil correction only on novel
    support. F60 reset the judge with exact-disk V2 and four optical partitions.
    F61 separates on-focal front veto from positive non-focal rear visibility,
    moves the 5% parent test to the fragment's PSF-dilated neighborhood, and
    requires conservative cross-PSF coverage. F62 then fixes cascade order:
    forward-winning focused-owner geometry may replace the already-mixed base
    matte; its eroded cross-PSF partial-coverage interior is hard-selected before
    rear recovery. S16 first falsified F61 on its only fire; F62 repairs it:
    ΔSSIM +0.000582, inner −2.167, outer −1.507, core/far identity, with
    35 refusals. A genuinely post-final S19 split then yields 3/72 fires, all
    positive on every physical partition, with 69 exact refusals. Auto remains
    disabled because all seven current fires retain +0.0014…+0.0096
    finest-band error on smooth GT veil pixels; S15 must localize and remove
    that causal tail.
- **20 (NEW, opened by the MISSION framework): stack-gap recovery** — regions where
  NO frame is sharp (focus gaps in the sweep): the best selection is still blurred
  there; scene recovery admits mild remnant-anchored deconvolution (known defocus
  scale from depth-from-focus; surround + factory-GT audited). Previously out of
  scope by definition; in scope under the mission.
  **STATUS 2026-07-22: ✅ FIRST PASS PROVEN (F52, gapfill.py).** Wiener one-shot at the
  known scale: gap-SSIM +0.0544 mean / +0.0340 worst, off-gap 0.0000 everywhere;
  ±15% radius error keeps most of the win (graceful degradation). Lucy turnover
  measured (RL k=40 worst goes negative); wrap-pad pitfall confirmed & fixed by eye.
  Sub-frontiers:
  - **20b: scale from DFF + calibrated selection** — naive re-blur-residual selection
    is degenerate toward under-deconvolution (measured); needs Levin-style per-scale
    calibration or the gate's outcome regression; then real DFF radii (--depth-out).
  - **20c: prior rungs + harder gaps** — TV/hyper-Laplacian (R8 α<2), asymmetric gaps
    (d≠0.5: per-frame radii differ), continuous depth ramps (L7 family), giant-CoC
    gaps (approx-kernel regime breaks the exact-PSF assumption — measure the cost).
  - **20d: gating + composition** — gap detector (where does NO frame win decisively?
    the F26 decisiveness floor inverted), outcome gate per F47, composition with
    --enhance; real handheld stacks (mobiledepth) once 20b lands.
- **18 (RESCOPED by F54): gate recall growth** — more features, bigger factories,
  nonlinear gates. A gate cannot rescue an operator with a negative realistic
  oracle ceiling; this row applies only to contour reconstruction and to future
  recovery operators after their model class passes.
  lit-scan 2026-07: conformal risk control (L2) may buy recall back from the
  worst-case+eps margin with a formal guarantee; ramp-family factory (L7) attacks
  the two feature-invisible outliers.

## Literature scan (2026-07-22)

Sweep of MFIF/focal-stack literature + adjacent fields (2023–2026), filtered hard
against what we have already built, proven, or refuted. Every citation below was
verified by opening the paper/page, not just a search snippet. Entry format:
what it is → why it matters HERE (named finding/wall) → redundancy honesty →
first experiment in our factory/gate idiom.

### L1 — Forward-model re-degradation audit (the #17 candidate)
**What:** HallAssess — no-reference hallucination assessment for AI-reconstructed
microscopy (Yan, Ma, Tan, Sun, Fu; Research Square, Aug 2025,
doi:10.21203/rs.3.rs-7026761/v1). Trick: re-DEGRADE the enhanced output with the
known degradation model and compare against the observed low-quality INPUT —
converting an impossible no-ref problem into a full-ref one in the degraded
domain. Siblings: sFRC (patchwise Fourier ring correlation for restoration
hallucinations, FDA/arXiv 2603.04673); DEReD (CVPR 2023, arXiv 2303.10752) is the
training-time analogue (optical re-rendering validates depth+AiF against the stack).
**Why here:** This is a direct attack on the F45 wall (ledger #17). F45 proved
output-domain source-similarity CANNOT audit synthesis corrections (the de-hazed
output is SUPPOSED to deviate from every source). But our specialists carry an
explicit forward model (disk PSF + alpha veil) — so re-blur the corrected output
through that model and compare to each OBSERVED frame: a correct correction
re-degrades back INTO the sources; a hallucination doesn't. The comparison
happens where sources are valid references again.
**Redundancy:** Adjacent to F45's failed q_ssim self-check — the non-redundant
part is precisely the domain flip (compare after re-degradation, not before).
**First experiment:** On the 75-scene composed-verdict set (GT verdicts known):
for every fired correction, synthesize per-frame re-degraded outputs using the
same PSF/alpha the specialist assumed, score q_ssim vs each source, and test
whether the audit separates GT-verified wins from the two feature-invisible
outliers and scene_46's twin. If it ranks, it becomes both a runtime self-audit
and a new gate feature (recall growth per #18).

### L2 — Conformal risk control for gate margins
**What:** "Conformal Risk Control" (Angelopoulos, Bates, Fisch, Lei, Schuster;
ICLR 2024, arXiv 2208.02814): distribution-free control of the EXPECTED value of
any monotone loss via a calibration-split quantile, generalizing conformal
prediction beyond coverage.
**Why here:** The F47 unified gate recipe sets fire margins as train-worst-harm
+ eps — a worst-case heuristic that "eats small-effect recall first" (veil gate
17→4 scenes). CRC replaces it with a chosen risk level: fire whenever predicted
gain clears a threshold calibrated so E[clipped harm] ≤ δ, with a finite-sample
guarantee. Directly targets #18 (recall) without abandoning the every-scene
property — it reframes it as a controllable risk dial.
**Redundancy:** Our recipe already IS split calibration in spirit; non-redundant
parts are the formal guarantee, the explicit δ dial, and expected-loss (not
worst-case) semantics.
**First experiment:** Retrofit CRC onto the locked recon gate's 320-scene factory:
calibrate λ on the train split for E[max(0, −dg)] ≤ 5e-4, measure held-out recall
vs the F47 margin (23/240 fired) and verify the held worst case doesn't blow past
the F48 bound (−0.0033). Pure re-analysis of existing caches — zero new compute.

### L3 — Misaligned-stack fusion with labels: DSAF-Net + WHU-MFM
**What:** "A defocus and similarity attention-based cascaded network for
multi-focus and misaligned image fusion" (Chen, Jiang, Li, Yao; Information
Fusion 103:102125, 2024; code github.com/PeimingCHEN/DSAF-Net). Defocus-Net +
OpticalFlow-Net warp sources to one view, then similarity-attention down-weights
still-mismatched content before fusing. Companion dataset **WHU-MFM**
(github.com/PeimingCHEN/WHU-MFM-Dataset): 3000 Blender scenes, 5-frame stacks
WITH camera deviations, 960×720 + 480×360, per-frame AiF GT + depth + defocus
maps + masks + camera matrices (Baidu Yun hosting).
**Why here:** F37 (blend beats perband under handheld motion — ghosting) is our
top practical wall and we have ZERO labeled misalignment data (mobiledepth has
no GT). WHU-MFM is exactly the missing label factory: GT under KNOWN, graded
camera deviation. Their architecture is also the reference answer to "warp
first, then down-weight residual mismatch" — the deep mirror of the
misalignment-robust perband variant F37 calls for.
**Redundancy:** Synthetic Blender misalignment ≠ real handheld (the F37/F36
transfer lesson stands); iphone12 GT remains the real-data verdict. Non-redundant
part: labels for gate/feature training that no real set provides.
**First experiment:** Pull WHU-MFM (Baidu may need a manual hop; fallback:
replicate the recipe in our factory — inject known homography jitter + exposure
into existing GT scenes). Grade perband vs blend vs perband+robustness (L4) as a
function of deviation magnitude; train the first ROUTING gate (perband→blend when
residual-motion features exceed margin) with the F47 recipe.

### L4 — Tile-wise robust merge, the industry answer to ghosting
**What:** HDR+ burst pipeline (Hasinoff, Sharlet, Geiss, Adams, Barron, Kainz,
Chen, Levoy; ACM TOG / SIGGRAPH Asia 2016, doi:10.1145/2980179.2980254;
hdrplusdata.org). Production-proven merge: per-tile, frequency-domain, pairwise
Wiener-style shrinkage of each alternate frame toward the reference tile — where
frames disagree (motion/misalignment), the merge gracefully degrades to
reference content instead of ghosting.
**Why here:** F37's eye-confirmed mechanism is per-band fine decisions flipping
on offset content (double-edge ghosting). HDR+'s robustness weight is the
principled per-tile version of "trust cross-frame content only where it agrees
with the owner frame" — graftable INSIDE our band loop: scale each frame's
per-band weight by similarity of its band content to the owner frame's, so
perband keeps its sharpness where aligned and falls back to single-frame content
where not. A decade of production mileage on handheld phone bursts.
**Redundancy:** blend already achieves ghost-freeness via ONE coarse decision —
at a sharpness cost. The non-redundant target is ghost-freeness WITHOUT giving
up per-band sharpness. Note this is a decision-side change and F33's negative
does not apply (we are suppressing misregistered content, not trying to unmix
boundary physics).
**First experiment:** Add per-band robustness scaling w ← w·exp(−d²/σ²), d =
local |candidate band − owner band| (owner = raw winner), to fuse_perband;
A/B on mobiledepth kitchen/largemotion with eyetool double-edge crops + q_ssim
ordering vs blend; then quantify on the L3 labeled factory.

### L5 — Per-stack implicit fitting with breathing-aware flow (TAF)
**What:** "An Implicit Neural Representation for the Image Stack: Depth, All in
Focus, and High Dynamic Range" (Wang, Serrano, Pan, Wolski, Chen, Myszkowski,
Seidel, Theobalt, Leimkühler; ACM TOG / SIGGRAPH Asia 2023; taf.mpi-inf.mpg.de;
code github.com/Hans1984/TAF). Fits neural fields per stack that jointly output
AiF + depth + HDR, with a neural FLOW field explicitly absorbing lens-breathing
misalignment, optimized by re-rendering the observed stack.
**Why here:** Rows 7/8 in one design: breathing is modeled as a smooth learned
flow rather than a global scale/warp — the concrete spec FRONTIER 8 lacks. Even
without adopting INRs in-pipeline (per-stack optimization, torch, no GPU here →
offline only), TAF is a pseudo-GT MACHINE for real handheld stacks: fit slowly
once, grade fast classical variants against its output — an independent
alternative to iphone12's Helicon pseudo-GT for the F37 decider.
**Redundancy:** Joint AiF+depth is what the classical engine already produces;
the non-redundant parts are self-calibrating alignment/breathing and per-stack
test-time optimization.
**First experiment:** Run released TAF on mobiledepth Figure6
{zero,small,large}motion (N=14 fits their regime); eyetool TAF-AiF vs
perband/blend under motion. If TAF is visibly cleaner, adopt as offline
reference for L3/L4 grading on real data.

### L6 — MattingMFIF: an external 4K matte-composited benchmark (VAEEDOF)
**What:** "Addressing the Depth-of-Field Constraint: A New Paradigm for High
Resolution Multi-Focus Image Fusion" (arXiv 2510.19581, Oct 2025): fusion in a
distilled-VAE latent space, up to 7 frames at once; introduces **MattingMFIF**,
a synthetic 4K dataset built by matte-compositing realistic DOF onto real
photographs; code + weights stated available.
**Why here:** Our high-res claims (F18–F26 arc, perband 0.9021) rest on OUR
generator. F36's lesson — benchmark-matched wins don't transfer; cross-generator
tests are the honest ones — means an independently built matte-composited 4K set
is exactly the external validation our high-res regime lacks, and the first
venue where our numbers are directly comparable to published SOTA.
**Redundancy:** Their data-generation idea IS our factory idea, published
independently — corroboration, not novelty. Latent-VAE fusion itself is a GPU
story (F14 territory) — not actionable on this box.
**First experiment:** Download MattingMFIF test split; run perband and
--enhance auto; report GT metrics next to their published table. Divergence
between our-factory and their-factory rankings would itself be a finding
(generator bias made visible).

### L7 — Continuous-depth ("ramp") factory family, from the StackMFF line
**What:** StackMFF (Applied Intelligence 2025, doi:10.1007/s10489-025-06383-8)
fuses whole N-frame stacks end-to-end and — key idea — synthesizes unlimited
training stacks from any AiF image via monocular-depth-driven refocusing;
StackMFF-V2 (Eng. Appl. of AI 2025) reformulates stack fusion as focal-plane
DEPTH REGRESSION; series repo github.com/Xinzhe99/StackMFF-Series.
**Why here:** All our factories are planar/layered (matte composites, discrete
planes). Real scenes are continuous depth ramps — and F48's two persistent
outliers are "mixed" scenes that survived three distribution extensions. The
F46/F48 lesson (gates must be trained on every family they will meet, including
never-fire families) has one missing family: continuous depth. The
monodepth-refocus recipe builds it from assets already in-tree (DA-V2 bridge +
disk PSF). V2's depth-regression view is also the learned mirror of our
winner-index depth byproduct (#5) — same equivalence, opposite direction.
**Redundancy:** The generator machinery is ~already ours; non-redundant parts
are the FAMILY itself and their pretrained N-frame baselines as comparators.
**First experiment:** Factory family "ramp": DA-V2 depth on real photos →
quantize to K focal planes → per-plane disk defocus + alpha compositing (exact
GT retained) → re-run gate training with ramp added; check whether the two
feature-invisible outliers become separable, and that recon correctly refuses
to fire on ramps (the objocc refusal lesson, one family further).

### L8 — Inpainting-difficulty as a GT-free fusion signal (Fusion2Void)
**What:** "Fusion2Void: Unsupervised multi-focus image fusion guided by image
inpainting" (TCSVT 2024; code github.com/LYL1015/Fusion2Void). Randomly drop
patches from sources; an inpainting net must restore them GIVEN the fused image.
Focused content is harder to hallucinate than defocused — so restoration error
scores focus preservation with no GT.
**Why here:** Ledger #17 and row 12 both need a no-ref signal that is NOT
source-similarity. Inpainting difficulty is structurally orthogonal: it measures
whether the fused image CARRIES the information needed to reconstruct each
source's sharp content, rather than whether it resembles the sources — so it
does not automatically score a synthesis correction as damage.
**Redundancy:** We use inpainting generatively (C3 background plates); they use
it evaluatively — different role, no overlap. The F45 trap applies in full: any
such proxy must be validated against factory GT verdicts before being trusted
for anything.
**First experiment:** Patch-drop/reconstruct scoring over the 75-scene composed
set (cv2.inpaint first; LaMa via the torch bridge if classical inpainting is too
weak to show the asymmetry); Spearman vs GT verdicts, head-to-head with the
composite and with L1's re-degradation audit. Adopt whichever (or the pair) ranks.

### L9 — EDMF: real exposure-difference multi-focus pairs
**What:** "EDMF: A New Benchmark for Multi-Focus Images with the Challenge of
Exposure Difference" (Sensors 24(22):7287, Nov 2024; data
github.com/stywmy/EDMF, CC BY 4.0): 1000 REAL smartphone-captured multi-focus
pairs (4 phone models, indoor/outdoor, day/night) with genuine exposure
differences between the frames; no GT (their method is unsupervised).
**Why here:** F28's exposure normalization shipped default-ON validated on
INJECTED drift only. EDMF is the in-the-wild test: real AE/WB behavior, night
scenes, and clipped highlights — the exact documented caveat of the
mean-invariant gain (clipping breaks mean preservation).
**Redundancy:** 2-frame pairs, not deep stacks (doesn't advance 2b); no GT →
F25 protocol (no-ref orderings + eye), not fidelity numbers.
**First experiment:** Run the pipeline with/without --no-normalize-exposure on
the 500 test pairs; check ordering + eyetool crops on the highest-drift and
night pairs, hunting specifically for clipped-highlight gain failures; log the
first real-data F28 verdict.

### L10 — Multi-focus VIDEO fusion is now benchmarked (VF-Bench / UniVF)
**What:** "A Unified Solution to Video Fusion: From Multi-Frame Learning to
Benchmarking" (NeurIPS 2025 Spotlight, arXiv 2505.19858): UniVF (flow-warped
multi-frame fusion) + **VF-Bench**, the first video-fusion benchmark covering
multi-focus among four tasks, with aligned pairs and unified spatial+TEMPORAL
consistency metrics.
**Why here:** Unblocks FRONTIER 14, which had no data and no metric. The
temporal-consistency metric is the missing yardstick for "does frame-independent
perband flicker?" — the question that decides whether row 14 needs temporal
machinery at all.
**Redundancy:** None in-tree; this is a data+metric unblock, not a method claim.
**First experiment:** Run per-frame perband on VF-Bench multi-focus clips and
score their temporal-consistency metric — the baseline number row 14 starts
from. If flicker is real, the cheapest fix to test first is temporal smoothing
of the DECISION maps (weights), not the pixels — decision/reconstruction split,
applied to time.

### L11 — Dedicated defocus-matting networks for general scenes
**What:** "Robust multi-focus image fusion using focus property detection and
deep image matting" (Expert Systems with Applications 237:121389, 2024 —
verified via Semantic Scholar DOI record): treats focus/defocus-boundary fusion
AS an image-matting problem — focus detection yields a trimap, a matting network
regresses the alpha. Successor: SpSwin autoencoder-based matting (ESWA 2025,
S0957417425006025). Lineage: MMF-Net's alpha-matte boundary defocus model (TIP
2020, arXiv 1910.13136).
**Why here:** The 16e close ledger names "SAM-quality matting for GENERAL
scenes" as the recall-growth lever, and F46 proved the two specialists need
different matte classes. A trimap-driven matting net is a third matte source
with precision plausibly between C3 difference mattes (px-precise, thin-only)
and FastSAM masks (region-precise, object-only) — potentially serving both.
**Redundancy:** MMF-Net's alpha-matte training-data generation is our factory
idea, six years earlier — good company, not news. F43's trap governs the eval:
matting nets expect natural-image statistics, so they must be judged on the
objects-as-occluders benchmark, never pastiche.
**First experiment:** Trimap from our focus-dominance ribbon → pretrained
matting net (torch bridge) → score alpha error + the F47 matte-edge features vs
C3 and FastSAM mattes on the objocc factory; if precision lands between the two
classes, register it as a third matte class in gate training and measure recall.

### Deliberately excluded (redundant, surpassed, or wall-blind)
- **Diffusion/generative MFIF** (ReDiffuse, arXiv 2603.21129; GMFF/StackMFF-V4,
  arXiv 2512.21495): synthesis without audit — F45 says nothing we run can
  certify their hallucinations, and their evals use no-GT benchmarks scored by
  the very source-similarity metrics F45 discredited for synthesis. Revisit
  only after L1 lands (then: audit a released model's outputs as the experiment).
- **2-frame deep MFIF nets** (SwinMFF, KCUNET, coupled neural-P systems,
  double-branch encoders, …): compete with the generalist exactly where perband
  sits at 0.99 GT-SSIM ceilings; none addresses a named wall; all GPU.
- **Learned guided filters** (Deep Attentional GIF, TNNLS; CVIU 2025 survey):
  would swap a validated O(1) CPU primitive for a GPU net; the one property we
  actually miss (robustness to misalignment) is delivered cheaper by L4.
- **Text-guided / controllable fusion** (arXiv 2512.20556; Conditional
  Controllable Fusion): control modality orthogonal to every fidelity wall.
- **General AIGC no-ref IQA** (Zoom-IQA, VisualQuality-R1, AIGCIQA2023):
  perceptual "quality" ≠ fidelity-to-scene; for auditing corrections our
  forward model makes L1 the domain-correct instrument.
- **ReFusion meta-learned fusion loss** (arXiv 2312.07943): re-treads the
  self-supervised-loss territory F13/F14 closed (feasible, trails classical).
- **DC-NeRF all-in-focus NeRF** (dual-camera) and **neuromorphic focal stacks**
  (event cameras): hardware assumptions we don't have.
- **Minimal-focal-stack SFF** (CVPRW 2026, arXiv 2604.01603): post-hoc frame
  pruning is refuted here (F24: top-K monotonically hurts — broad weights
  denoise); capture-time focus-position placement is real but blocked on #13
  (first-party capture).
- **Learn2Refocus** (SIGGRAPH Asia 2025): already cataloged as the iphone12
  dataset source (REAL_DATA.md); the video-diffusion method itself falls under
  the generative exclusion above.
- **MFI-WHU / Road-MF**: easy-synthetic (Gaussian) or small no-GT benchmarks —
  the regime PLAYBOOK explicitly warns against optimizing on.

### Lit scan: classical restoration for 19/20 (2026-07-22)

Focused scan for the MISSION ladder step 1: SIMPLE/CLASSICAL methods for (A)
contrast restoration under a semi-transparent veil with structured amplification
noise (frontier 19) and (B) mild anchored deconvolution for stack gaps (frontier
20). Every citation verified by reading the actual paper (full PDFs for all nine).
Entry format: what → mapping onto our machinery → failure modes vs our gates →
first experiment. Doctrine check: all nine are remnant-guided (they operate on
degraded observations of the true content, never synthesize from external
priors); drift risks flagged inline. The task's "regularized/Wiener inversion"
bullet is covered structurally inside R4 (subband SNR gain = Wiener-shaped
attenuation) and R8 (its FFT quadratic sub-step IS the Tikhonov/Wiener inverse).

#### A — veil-contrast restoration under amplification noise

**R1 (A) — Dark channel prior dehazing with lower-bounded transmission.** He,
Sun, Tang, "Single Image Haze Removal Using Dark Channel Prior," TPAMI
33(12):2341–2353, 2011 (doi:10.1109/TPAMI.2010.168).
*What:* haze model I = J·t + A(1−t); radiance recovery J = (I−A)/max(t,t0) + A
with t0 ≈ 0.1, plus ω = 0.95 deliberately keeping a little haze. The paper
itself notes (Eq. 13–14) that the model generalizes to "veiling luminance" and
is IDENTICAL in form to the matting equation I = Fα + B(1−α) — single-image
dehazing is literally our matte-veil problem with t = 1−α.
*Mapping:* frontier 19's division revisit IS radiance recovery with t known
from our matte (better than DCP's estimate). DCP's two safety devices are the
field's standard answer to amplification noise at 8-bit: the gain clamp
max(t,t0) bounds amplification at 1/t0, and ω < 1 under-corrects on purpose.
Both drop directly into the 16d in-loop idiom.
*Failure modes:* objects similar to airlight without shadow → t underestimated
→ over-amplification (their Fig. 18 marble); halos from patch-constant t; noise
at low t. We don't inherit the prior's failures (our t comes from the matte);
over-amplification is exactly what factory GT + fringe clamps + outcome gates
measure.
*First experiment:* in the 16d loop, replace subtraction with clamped division
J = (I − A·α)/max(1−α, t0); sweep t0 ∈ {0.1…0.5} × ω ∈ {0.9, 0.95, 1.0} on the
giant-CoC wideocc factory, 8-bit; GT-credit strong-veil band contrast + off-band
harm; retrain gates on hybrid outcomes.

**R2 (A, also B) — Guided image filtering as gain-field refinement and
structure-transfer denoiser.** He, Sun, Tang, ECCV 2010 / TPAMI
35(6):1397–1409, 2013 (doi:10.1109/TPAMI.2012.213).
*What:* our existing primitive (q = aI + b local linear model). The literature
gives it two roles we haven't used: refining the transmission/gain map so the
amplification boundary hugs image edges (its flagship dehazing application),
and joint filtering with a CLEAN guide (flash/no-flash) — filter the amplified
noisy band using the un-veiled surround / owner frame as guide.
*Mapping:* frontier 19(b) verbatim; zero new code — guided_filter is in-tree.
*Failure modes:* halos when the guide lacks structural correspondence; eps is
"how much contrast counts as an edge" — mis-set eps smooths the recovered
detail back out (undoing the amplification we just paid noise for). Both
GT-measurable per band.
*First experiment:* after per-band amplification, guided-filter each amplified
band with the owner-frame band as guide; sweep (radius, eps) per band vs
unguided amplification on factory GT.

**R3 (A) — Flash/no-flash: joint bilateral denoising + ratio detail transfer
with mask-gated fallback.** Petschnigg, Szeliski, Agrawala, Cohen, Hoppe,
Toyama, "Digital Photography with Flash and No-Flash Image Pairs," SIGGRAPH
2004 (ToG 23(3):664–672); cross-bilateral variant: Eisemann & Durand, SIGGRAPH
2004.
*What:* denoise a noisy image with edge-stopping weights computed from the
clean paired image (joint bilateral, their Eq. 4); transfer detail as a
multiplicative ratio layer F_Detail = (F+ε)/(F_Base+ε), ε = 0.02 (Eq. 6);
detect regions where the guide is untrustworthy (flash shadows/specularities)
and fall back: A_Final = (1−M)·A_NR·F_Detail + M·A_Base (Eq. 7).
*Mapping:* the exact template for "surround-informed denoising of the
amplified band" (19(b)): our amplified veil band = their noisy ambient; our
owner frame / clean surround = their flash. Their mask-gated fallback is our
gate idiom, published 2004. The multiplicative detail-layer idiom is a
ringing-free way to re-inject contrast.
*Failure modes (stated in the paper):* guide detail that does not exist in the
target gets transferred (their shallow-angle flash texture; for us:
owner-frame content leaking across the matte boundary = misattribution); halos
when bilateral widths grow too wide. Fringe clamps + factory GT catch the
first; eye + L1 re-degradation the second. Doctrine flag: detail transfer is
remnant-guided ONLY when the guide is the same scene's clean content (owner
frame / surround) — with any external exemplar it becomes blind import;
enforce guide provenance.
*First experiment:* build the amplified band's edge-stopping from the owner
frame; per-pixel mask from matte confidence falls back to subtraction-only
where guide trust is low; factory GT + outcome gates.

**R4 (A) — Per-subband SNR-adaptive shrinkage (BayesShrink).** Chang, Yu,
Vetterli, "Adaptive Wavelet Thresholding for Image Denoising and Compression,"
IEEE TIP 9(9):1532–1546, 2000.
*What:* soft-threshold each subband with the closed-form T_B = σ̂²/σ̂_X (noise
variance over signal std); σ̂ = Median|Y|/0.6745 from the finest subband; σ̂_X
= sqrt(max(σ̂_Y² − σ̂², 0)); within 5% of the optimal soft-threshold, beats
SureShrink most of the time.
*Mapping:* the closed-form recipe for 19(a) per-band SNR-weighted amplification
in our Laplacian band loop: after multiplying a band by 1/(1−α) the noise σ
scales identically, so T_B rises exactly where amplification hurt SNR —
shrink-after-gain = per-band, per-region gain tapering with no tuning. Same
SNR-driven shape as the Wiener gain, without FFTs.
*Failure modes:* MAD noise estimation assumes the finest band is mostly noise —
ours is spatially STRUCTURED (amplified only inside the veil), so σ must be
estimated per region (in-veil vs out) or the threshold is wrong on both sides;
soft-thresholding biases amplitudes down (systematic contrast under-recovery —
GT-measurable).
*First experiment:* amplify per band, in-veil MAD noise estimate, soft-threshold
with T_B; GT-score band contrast + off-band harm; head-to-head vs R2/R3 guided
denoising — competing implementations of the same slot in the hybrid.

**R5 (A) — Dequantization with a hard re-quantization constraint (MAP-AC
bit-depth enhancement).** Wan, Cheung, Florencio, Zhang, Au, "Image Bit-Depth
Enhancement via Maximum-A-Posteriori Estimation of AC Signal," IEEE TIP 25(6),
2016.
*What:* recover the high-bit-depth image from its quantized version; the
feasible set is every signal that RE-QUANTIZES to the observation (their
Eq. 5); a graph-Laplacian smoothness prior with edge weights from observed
gradients picks within the bin (MAP for AC, closed-form MMSE for DC) — removes
false contours/banding without blurring true edges.
*Mapping:* our 8-bit failure in its purest form — after division by 1−α,
quantization steps become visible bands. Two portable ideas: (i) the
quantization-bin constraint is the L1 observation-domain audit BUILT INTO the
estimator (re-degradation consistency by construction, not post-hoc); (ii)
gradient-derived edge weights are guided-filter-style structure awareness, so
debanding never fights detail recovery.
*Failure modes:* edge weights from the quantized observation can read genuine
sub-bin gradients as flat (over-smooths textureless ramps); block seams
(half-overlapped blocks mitigate). Factory GT + eye catch both.
*First experiment:* cheapest version first — after amplification, guided-filter
each band, then PROJECT the result back into the per-pixel quantization bin
implied by the observed 8-bit value pushed through the forward model. Measures
whether bin-projection alone kills the banding noise F27's bench saw.

#### B — anchored deconvolution for stack gaps

**R6 (B) — Richardson–Lucy with early stopping as the regularizer.**
Richardson, JOSA 62(1):55–59, 1972; Lucy, "An iterative technique for the
rectification of observed distributions," AJ 79(6):745–754, 1974 (read in
full).
*What:* multiplicative ratio updates ψ^{r+1} = ψ^r·[(φ̂/φ^r) ⊛ PSF]; conserves
non-negativity and flux; likelihood rises monotonically but Lucy's own Sec. IV:
"no attempt should be made to achieve convergence" — past a few iterations the
gains only fit noise (his Fig. 1: r=15 visibly worse than r=3). His stopping
rule: stop when residuals against the OBSERVED data become ascribable to noise
(χ² with P > 0.05). His Sec. V(iv) "model testing": if iterations cannot reduce
residuals, the assumed kernel is wrong.
*Mapping:* safest first tool for gap deblur with our known disk PSF — one knob
(iteration count). Lucy's stopping rule IS the L1 audit, published 1974:
re-blur the estimate, compare to the observed gap, stop at noise-level
residuals. His model test doubles as a free mis-estimated-CoC detector — a
gate feature.
*Failure modes:* ringing at strong edges grows with iterations (Gibbs;
confirmed independently in Yuan 2007 and Levin 2007 Fig. 7); noise
amplification past the stopping point. The factory measures the
iteration-vs-artifact curve directly; outcome gates learn the safe count.
*First experiment:* build the GAP FACTORY (matte generator variant with a focus
gap: a depth band where no frame is sharp — GT sharp everywhere); RL with the
DFF-estimated disk PSF, k ∈ {2…15}; GT-credit sharpness vs ringing; gate on
predicted delta-global with the L1 residual as a feature.

**R7 (B) — Residual + gain-controlled RL with guided detail add-back.** Yuan,
Sun, Quan, Shum, "Image Deblurring with Blurred/Noisy Image Pairs," SIGGRAPH
2007 (ToG 26(3)) (read in full).
*What:* anchored deconvolution: write I = N_D + ΔI (denoised anchor + detail
residual) and deconvolve only the residual ΔB = B − N_D⊗K — ringing is
proportional to the magnitude of the deconvolved signal, and ΔB is small
(their Fig. 5). Remaining ringing suppressed by a gain map I_Gain = (1−α) +
α·Σ_l‖∇N_D^l‖ (α = 0.2, pyramid gradients of the CLEAN anchor) multiplying each
RL iterate; fine detail lost to gain control re-added as a joint/cross-bilateral
detail layer.
*Mapping:* THE anchored-deconvolution template for stack gaps, and our anchor is
better than theirs: the best-mix image (sharp surround + mildly blurred gap)
plays N_D; deconvolve only the gap residual; gain map from surround gradients;
the F40 fringe-clamp is a gain map. Every piece (pyramid gradients, joint
filtering) is already in-tree.
*Failure modes (stated):* gain control suppresses some true fine detail (hence
their add-back — GT-measurable); assumes one spatially-invariant kernel (their
limitation; our per-region CoC from depth-from-focus is the fix).
*First experiment (the centerpiece B experiment):* gap factory; plain RL (R6)
vs residual RL anchored on the best-mix vs + gain map; hypothesis per their
Fig. 5: residual anchoring cuts ringing at equal recovered sharpness.

**R8 (B) — Fast non-blind deconvolution with a hyper-Laplacian gradient
prior.** Krishnan, Fergus, NIPS 2009 (read in full).
*What:* min λ/2‖x⊗k − y‖² + Σ|∇x|^α with α ∈ [0.5, 0.8] (α = 2/3 fits natural
gradient statistics); half-quadratic splitting alternates an FFT-solvable
quadratic step (3 FFTs — a Tikhonov/Wiener-regularized inverse) with a
per-pixel shrinkage solved by LUT or analytic cubic/quartic roots (α = 1/2,
2/3); ~3 s/megapixel on a 2009 CPU vs ~20 min IRLS at comparable quality. TV
deconvolution (Wang, Yang, Yin, Zhang, SIAM J. Imaging Sci. 1(3), 2008) is the
α = 1 special case with plain shrinkage.
*Mapping:* the one-shot alternative to iterative RL for the gap — numpy-only
(FFT + LUT), CPU-friendly, fits our no-scipy env. One implementation yields
three rungs of the comparison ladder: α = 2 (pure regularized Wiener baseline,
single FFT solve), α = 1 (TV), α = 2/3 (hyper-Laplacian).
*Failure modes:* circular-boundary wrap artifacts (pad/taper); the sparse prior
favors piecewise-flat — cartoonifies texture at strong λ; ringing under kernel
error. All GT-measurable; λ swept on the factory.
*First experiment:* gap factory; Wiener vs TV vs α = 2/3 at matched runtime;
full-image vs per-band application; chart the artifact-safety frontier against
R6/R7.

**R9 (B) — Depth-indexed deconvolution with reconstruction-error scale
selection.** Levin, Fergus, Durand, Freeman, "Image and Depth from a
Conventional Camera with a Coded Aperture," SIGGRAPH 2007
(doi:10.1145/1275808.1276464) (read in full).
*What:* defocus PSF = the aperture shape scaled by depth (y = f_k∗x, our disk
model exactly); deconvolve with a BANK of scaled PSFs; the WRONG scale produces
ringing, so a local reconstruction-error energy Ê_k with learned per-scale
weights λ_k selects depth per window (their Eq. 13–15); the all-focus image is
assembled per pixel from the correctly-scaled deconvolution. Works with a
conventional circular aperture, but adjacent scales share overlapping frequency
zeros — weaker discrimination (their motivation for coding the aperture).
*Mapping:* the focal-stack-specific answer to "defocus scale approximately
known": don't trust the DFF scale point-estimate — deconvolve at 3–5 CoC
candidates bracketing it and let LOCAL reconstruction error pick; that selector
is an in-situ L1 audit (wrong scale = high re-blur residual). Also a calibrated
warning: with our disk PSF expect a shallow error valley between adjacent
scales, so gate margins carry the safety.
*Failure modes:* textureless windows give unreliable scale (they patch with MRF
+ user strokes; we instead REFUSE — the recon-gate energy-floor idiom); ringing
bleeds across depth discontinuities (deconvolve per matte region).
*First experiment:* gap factory with CoC deliberately mis-estimated ±30%;
verify reconstruction-error selection recovers the true scale where texture
exists and the gate refuses where it doesn't.

#### Excluded from this scan
- **Learned dehazing / deblurring / bit-depth expansion** (DehazeNet, AOD-Net;
  DeblurGAN-class; BitNet 2019): not necessarily blind generation, but ladder
  step 2 by definition — admissible later, only where R1–R9 measurably plateau
  below the oracle ceiling on factory GT.
- **Blind deconvolution** (Fergus et al. 2006 kernel estimation and kin): our
  defocus scale is approximately known from depth-from-focus and R9 selects
  among candidates; blind kernel estimation adds an unanchored failure axis for
  zero benefit here.
- **Diffusion/GAN restoration and inpainting** (diffusion posterior sampling,
  generative deblur): blind generation — doctrine-excluded (consistent with the
  existing generative-MFIF exclusion above).
- **Example-based / hallucinating detail synthesis** (exemplar super-resolution
  and kin): imports texture from EXTERNAL exemplars rather than the scene's own
  remnants — doctrine-excluded. Contrast with R3, which is admissible precisely
  because its guide is the same scene's clean content.
- **Histogram equalization / CLAHE / unsharp masking**: He et al. (TPAMI 2011,
  Fig. 17) demonstrate they cannot determine the spatially-variant correction
  the physics demands; no forward model → the L1 audit cannot even be posed.

### Lit scan: restoration hallucination and model uncertainty (2026-07-25)

This scan follows the user-caught F53 failure: false structure can improve a
global restoration score and can even survive re-degradation. It does not admit
generative restoration; it strengthens the audit and refusal machinery for the
remnant-guided classical operator.

**R10 — Generalized measurement/null decomposition and hallucination maps.**
Bhadra, Kelkar, Brooks, Anastasio, "On Hallucinations in Tomographic Image
Reconstruction," IEEE TMI 40(11), 2021 (arXiv:2012.00646).
*What:* decomposes a reconstruction into generalized measurement and null
components to expose structures introduced by a regularizer/prior that the
measurement cannot determine; proposes hallucination maps for reconstruction
analysis.
*Mapping:* our forward residual audits the measurement component but is blind to
detail in the blur null space. Radius-bank disagreement is a practical local
null/uncertainty proxy: admit only correction components stable across plausible
forward models, and label the rejected component rather than rewarding its
sharpness.
*Failure modes / scope:* the paper studies stylized tomography and has an
explicit linear operator; our occlusion/blur model is spatially varying and
matte-estimated. We borrow the decomposition discipline, not its numerical
operator.
*First experiment:* per Laplacian band, compute the median correction across the
radius bank and its MAD; compare full candidate selection against
agreement-weighted consensus on GT false texture, fringe error, and worst-scene
SSIM.

**R11 — Learned fidelity under inaccurate degradation models.** Ren, Zuo,
Zhang, Zhang, Yang, "Simultaneous Fidelity and Regularization Learning for
Image Restoration," ECCV 2018 (arXiv:1804.04522).
*What:* treats the residual from a partially known or inaccurate degradation
model as spatially dependent and complexly distributed; learns a task-driven
fidelity term from degraded/GT pairs rather than assuming a precise kernel.
*Mapping:* the raw forward residual's radius bias is expected, not an
implementation accident. Keep per-radius residual, fit margin, matte evidence,
and disagreement as outcome features calibrated on the analytic factory. Do
not make raw residual the radius selector.
*Failure modes / scope:* their learned regularizer is beyond our current
classical rung and could inject a learned prior. Only the fidelity-learning
lesson is admitted now; scene synthesis remains the analytic remnant operator.
*First experiment:* scene-disjoint ridge/monotone gate over bank evidence,
compared with raw-min-residual selection and a refusal-only baseline.

**R12 — Inherent uncertainty versus perceptual quality.** Cohen, Kligvasser,
Rivlin, Freedman, "Looks Too Good To Be True: An Information-Theoretic Analysis
of Hallucinations in Generative Restoration Models," NeurIPS 2024
(arXiv:2405.16475).
*What:* formalizes information irrecoverably lost by a non-invertible
degradation and proves an uncertainty/perception tradeoff: increasingly
natural-looking restoration cannot simultaneously erase the inverse problem's
inherent uncertainty.
*Mapping:* the product must expose uncertainty as identity/refusal, not spend it
on plausible texture. Bank disagreement and low observed remnant SNR reduce the
admitted gain even if the stronger output looks more natural.
*Failure modes / scope:* the theory concerns distributions and generative
estimators, not our deterministic focus-stack operator. It supplies a safety
bound and evaluation stance, not a runtime formula.
*First experiment:* chart recovered contrast against bank disagreement and
false texture; choose the conservative knee, then require nonnegative held-out
worst-case outcome.

**R13 — Hallucination-specific evaluation beyond ordinary IQA.** Kim,
Tregidgo, Jin, Figini, Alexander, "HalluGen: Synthesizing Realistic and
Controllable Hallucinations for Evaluating Image Restoration," CVPR 2026
(arXiv:2512.03345).
*What:* constructs patch-labeled restoration hallucinations and reports weak
hallucination sensitivity for ordinary pixel/feature quality metrics; a
reference-free detector consumes both the restored prediction and measurement.
*Mapping:* `false_texture` must remain a separate GT label, not be folded into
SSIM, and the runtime proxy must compare correction with observations. Hard
cases are mined by disagreement/patch location, not average score alone.
*Failure modes / scope:* HalluGen is diffusion-based low-field MRI restoration,
so neither its generator nor detector transfers directly to focus stacks. The
portable result is metric insufficiency and measurement-conditioned detection.
*First experiment:* evaluate the unsupported-texture feature and bank
disagreement against GT `false_texture` on a scene-disjoint split; report
AUC/false-negative rate beside restoration gain.

## Literature scan (2026-07-26) — asymmetric visibility at occlusion boundaries

Focused follow-up for F60/S12. Every source below was read at the paper or
publisher page, then filtered against F25, F55–F60, the earlier scans, and the
measured forward-fit/null-space failures. The non-redundant conclusion is that
our remaining application mask collapses two physically different visibility
terms: a focused front surface must block the rear layer, while a defocused
front surface can partially reveal it through the finite aperture.

### V1 — Dr.Bokeh separates on-focal occlusion from non-focal visibility

**What:** Sheng et al., “Dr.Bokeh: DiffeRentiable Occlusion-aware Bokeh
Rendering,” CVPR 2024 (arXiv:2308.08843), derives a layered renderer with two
separate terms. `O_l(y,x)` excludes rear radiance intercepted by a layer on the
focal plane; `V_l(x)` integrates alpha over the projected CoC and determines
how much a defocused layer and every layer behind it contribute.

**Why here:** this is the exact missing invariant behind the user’s
scene-114/122 reports. F58/F59 established discrete front ownership, while F60
made frame coverage visible; V1 says the application rule itself must also be
asymmetric. A foreground-focused observation is a hard ordering constraint.
Rear recovery belongs only to non-focal visibility for which the rear-focused
frame contains positive rear evidence—not to every pixel outside an estimated
matte.

**Redundancy:** F60’s V2 renderer already implements the two-layer special case
of non-focal coverage. New here is the explicit split between *on-focal
occlusion* and *non-focal visibility* as two runtime gates rather than one alpha
threshold.

**First experiment:** replace `_fringe_mask * (1-owner_veto)` with
`predicted_nonfocal_coverage * positive_rear_observation *
on_focal_foreground_exclusion`. Grade complete core, inner partial occlusion,
outer veil, and far background independently on V2 dev, untouched holdout, and
a post-freeze extension.

### V2 — ordered attenuation, not symmetric blending, is the multilayer law

**What:** Liu, Narasimhan, and Dubrawski, “Matting and Depth Recovery of Thin
Structures Using a Focal Stack,” CVPR 2017, models each layer with an occlusion
index. A layer’s radiance is attenuated only by layers with smaller (nearer)
indices; in the two-layer case the rear coefficient is the product of
`1 - blurred_near_matte`.

**Why here:** a soft pixelwise competition between near and rear is not merely
suboptimal—it discards the causal direction. The foreground-focused frame
observes foreground radiance directly; the background-focused frame can reveal
rear radiance through aperture coverage but cannot revoke the already observed
front surface. This supports a front-first ownership projection before any
rear-layer correction.

**Redundancy:** `solve_layers` already uses the two-layer formation equation.
The missing piece is carrying its ordered attenuation into *support completion
and application*, where the code still treats uncertain ownership
symmetrically.

**First experiment:** build two explicit observation masks from the stack:
sharp-owner silhouette support (front, copy/veto) and decisive rear-focus
structure (rear, correction license). Identity wins wherever neither
observation is positive; no “absence of foreground estimate” license remains.

### V3 — a finite aperture reveals rear information, but only where it was measured

**What:** Favaro and Soatto, “Seeing Beyond Occlusions (and other marvels of a
finite lens aperture),” CVPR 2003, reconstructs occluding shape and radiance
from a defocused sequence. Its central result is that a finite aperture exposes
portions of the rear surface hidden in a pinhole view, with observability
depending on aperture, focal length, occluder size, and layer separation.

**Why here:** the work supports scene recovery beyond any one source frame, but
also bounds it. “The lens can see behind” is not permission to edit the whole
estimated fringe: rear recovery is licensed only on the spatial support where
the captured aperture actually conveyed rear information. This is S12’s
positive-evidence requirement in classical inverse-imaging form.

**Redundancy:** F55 already performs joint layer inversion and forward
re-rendering. New here is treating local rear observability as a support
condition, rather than assuming a globally licensed candidate makes every
predicted fringe pixel observable.

**First experiment:** compute a soft rear-observation density from decisive
rear-frame focus evidence, propagate it only over a resolution-scaled local
neighborhood, and multiply the correction by it. Report recovered outer-veil
credit against rejected inner/far damage and preserve the rejected correction
as an uncertainty map.

### V4 — occlusion-edge blur carries ordinal front/back information

**What:** Marshall et al., “Occlusion edge blur: a cue to relative visual
depth,” JOSA A 13(4), 1996, demonstrates that the sharpness of the shared
boundary between a focused and defocused region resolves the near/far
ambiguity inherent in depth-from-focus.

**Why here:** focus magnitude alone is symmetric in depth; boundary ownership
is not. The sharp side’s boundary continuation is positive ordinal evidence
that it is in front, matching the user’s “quick trick.” It suggests an analytic
ordering feature based on which frame carries the sharp boundary, not another
semantic score or global image-quality threshold.

**Redundancy:** F56 uses interior focus dominance and F59 uses semantic
containment. The boundary *asymmetry* is a third, local ordering channel that
can cover low-texture interiors where focus energy is weak and semantic masks
are hierarchical.

**First experiment:** on V2 dev, compare owner-frame versus rear-frame
transition-shell gradients along each candidate contour, then propagate the
winning orientation inward/outward separately. Test whether it catches missed
inner support without suppressing true outer veil; freeze before a new split.

### V5 — exact disks are a controlled rung, not the final camera model

**What:** Lee, Kim, and Cho, “Realistic Compound-Lens Defocus Blur Synthesis,”
July 2026 (arXiv:2607.05837), combines wave-optics PSFs across 700 compound
lenses, depth-aware occlusion compositing, radiometrically linear blur, noise,
and ISP simulation. It reports complex asymmetric off-axis PSFs and evaluation
bias from imperfect captured references.

**Why here:** F60 corrected a mislabeled box kernel, but a perfect circular
disk in sRGB is still only the analytic rung. V5 independently identifies the
next factory axes: lens/field-dependent PSF, linear-light formation, and ISP.
These belong after S12 closes on exact optics, before any broad camera claim.

**Redundancy:** S14 already asks for real aperture calibration. New here is a
current, reproducible synthetic bridge between the exact-disk factory and
first-party capture, plus the warning that off-axis asymmetry and ISP are
separate axes rather than “noise around a disk.”

**First experiment:** port one on-axis and one off-axis CLDefocus PSF into the
V2 renderer, move compositing to linear light, and rerun the frozen S12 gate.
Treat each PSF/field/ISP combination as a distinct family; a disk result does
not transfer by default.

### Deliberately excluded from this scan

- **Generative bokeh/de-occlusion networks** (MPIB, BokehFlow, MagicBokeh):
  useful renderers but not remnant-auditable scene recovery; they do not solve
  F60’s ownership invariant.
- **Light-field partial-view focal stacks** (Strecke et al., CVPR 2017):
  compelling visibility selection, but require sub-aperture views unavailable
  in an ordinary focal bracket. Revisit only for light-field inputs.
- **RealBokeh/Bokehlicious as latent truth:** valuable for appearance and ISP
  realism, but aperture-varied photographs do not expose foreground/background
  layers or occluder-removed scene truth. They cannot certify S12.
- **Another learned gate on global GT deltas:** F57/F60 already show why this
  misses coherent partition damage. The next label is local ordered visibility,
  not another pooled score.
