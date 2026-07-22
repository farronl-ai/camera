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
| 3b | **Occlusion-AWARE fusion** | Matte inversion (de-veiling). | ❌ F27: rigorous negative — even oracle+noiseless+exact-PSF loses to perband (thin-structure haze is small; fringe error is decision-boundary, not haze; division amplifies quantization). Revisit only for LARGE occluders + float pipeline + regularized unmixing. |
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
| 16b | **Matte-aware boundary reconstruction** | The hard-lines lever per F33. | 🔶 F34/F35/F36: ceiling −22%; buildable C3 −16% + global WIN on matte-model data; regresses off-model → shipped default-OFF experimental (--reconstruct-boundaries). |
| 16d | **Weight-scaled in-loop veil correction** | F40/F41: mechanism VALIDATED (O1b exact-D: fringe −18..−37%, global up everywhere; O2 true-alpha: positive on all 8 scenes). Band-limitation premise refuted (subtraction-not-division is the F27 evasion). BLOCKED ON: wide-occluder matte — classical fails on textureless ambiguity; semantic alpha (DA-V2 two-pass) is the designed input. | 🔶 |
| 16e | **Semantic matte input** | F42: stack-seeded semantic matting proves the blind chain (1 scene fully wins, no oracle) but lands only ~2/8; no self-gate signal separates. F43: mask matting v1 blocked by benchmark pathology (pastiche blobs aren't objects; bokeh spoofs objectness) — NEXT: objects-as-occluders generator (FastSAM cutouts = occluders with true silhouette GT), then re-run selection + per-component confidence + iphone12 GT (manual download required — Sync.com blocks scripts). | 🔶 |
| 16c | **Reconstruction applicability gating** | Fire only where the matte model holds. | 🔶 F38: plane-step ribbon (raw-winner median + energy floor) shipped — strongest on-model result (−16% e2, +global, all scenes), real-data neutral + fire moved to silhouettes. REMAINING: F39 killed veil-side licensing (thin structures have no strong veil — category mismatch); next license is SILHOUETTE-side (owner-frame contour sharpness + plane gap); iphone12 GT verdicts gate promotion. |

Near-term execution order: 1 → 2 (parallel) → 3 → 4 → 5 (plan: NEXT_STEPS_breadth.md).

## E-arc close ledger (2026-07-20)
- **16c reconstruction gate: ✅ LOCKED** (F47) — home regime (thin + C3 mattes), held
  5/30 fired all-positive mean +0.0083. Real-data promotion awaits iphone12 GT.
- **16d veil correction: ✅ mechanism + gate locked** (F40/F41/F44/F47) — property
  holds (held 2/2 good); recall ~4% (small effects price high under safety margin).
- **16e semantic matte: 🔶 resolved by regime-matching** (F46) — mask mattes serve the
  veil specialist; C3 serves reconstruction; SAM-quality matting for GENERAL scenes
  remains the recall-growth lever.
- **16f: ✅ SHIPPED (F48)** — composed --enhance auto in pipeline; 74/75 unseen scenes ≥ −0.0024, real data 13/14 identity + 1 tiny fence fire. Residual: 2 mixed outliers (feature-invisible), fence-fire eyetool check, iphone12 GT. Was: composed --enhance stage — both gates firing per stack; two-pass
  packaging over shipped bridge plumbing; identity when bridgeless. NEXT SESSION.
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
  EXPERIMENT (giant-CoC wideocc factory, GT-credited, 8-bit first, float as a
  second condition): 16d subtraction alone vs hybrid (subtract + regularized,
  surround-denoised contrast restoration). Success = strong-veil-band detail
  contrast approaches GT with no off-band harm; gates retrain on hybrid outcomes.
  If the hybrid wins at 8-bit, F27's epitaph reads: the idea was right, the system
  around it was missing.
- **18 (NEW): gate recall growth** — more features (per-candidate veil evidence,
  D-magnitude stats), bigger factories, nonlinear gate models; recall is the only
  thing between "property-safe" and "most images benefit."
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
