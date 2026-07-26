# NEXT STEPS — scene-recovery arc, phase 1: hybrid veil recovery (FRONTIER 19)

The MISSION's first constructive experiment: revisit F27's division negative INSIDE
the restoration system. 16d subtraction removes additive haze but leaves surviving
detail amplitude-scaled; only a clamped, denoised, fringe-masked gain restores it.
Bench: giant-CoC wideocc factory, GT-credited, oracle alpha, 8-bit first + float
second. Script: `veilgain.py` (subcommands = rungs). Baseline to beat: 16d
subtraction alone (`omega=0` = corr_multi-equivalent, byte-verified).

## Physics (why the gain law is measured, not assumed)
D is built from the OBSERVED far frame, so subtraction already partially restores
amplitude: on the background side post-subtraction detail ≈ (1−ab²)·true, noise
≈ (1+ab)·n. Candidate laws {1−ab, 1−ab², P0-empirical}; P0's measured curve is
the authority. Correction noise std is analytic: (1+ab)·σ·c_k·(G−1), σ=3 known,
c_k calibrated — zero blind estimation (FRONTIER 19 post-gain lever 1).

## Rungs (one variable each; success = cr_mid ↑ AND fringe not worse AND
## |dg_off| ≤ 5e-4 AND dg ≥ dg_sub − 5e-4; ties ±5e-4 are ties)
- [x] **P0** ✅ identities byte-pass; GT/GT=1.000; headroom real (cr_sub 0.877 mean;
      subtraction barely moves contrast 0.919→0.923). Empirical g(ab) measured:
      much shallower than 1−ab (0.64 vs 0.20 @ ab=0.8) → theoretical laws over-gain.
- [x] **H1** ✅ every config beats sub (dg to +0.0009, off-band 0.0000, fringe −1.4)
      but plateaus at cr 0.900 — w_far dilution diagnosed. **H1b** (deficit form,
      coef = G−w_far, sq law): cr = 1.000 EXACT at cost dg −0.0006 (noise rides in).
      **H1a** ✅ full-image placement decisively worse (dg −0.005..−0.010) — F40
      idiom confirmed 3rd time. lin-deficit overshoots (cr 1.07-1.20): ruled out.
- [x] **H2** ✅ analytic shrink m=2: dg_vs_sub +0.0004, cr 0.956 — converts amplitude
      into net GT-SSIM gain. **H3** ✅ NEGATIVE at every (r,β): guide too weak at the
      amplified scale (dg −0.001..−0.002, cr crushed). Analytic threshold wins the slot.
- [x] **H4** ✅ (eye-triggered: speckle in recovered dark regions) coherence gate =
      near-null at σ=3 (no losers flip, ±0.0004 best). Kept OFF.
- [x] **H5** ✅ decomposition: float OUTPUTS don't rescue 8-bit losers → wall is
      INPUT-side structured quantization, output bit depth exonerated. R5/MAP-AC
      debanding = open conditional rung; nearer path is the outcome gate.
- [x] **FINAL** ✅ 10 backgrounds × 2 coc × 2 dtypes (veilgain_final.json):
      coc0.04/float m2 +0.0019 mean, worst +0.0006 — ALL 10 positive, F27 idea
      vindicated. coc0.04/8bit +0.0007 mean, 3/10 small negatives (quantization).
      coc0.012 ungated HARM (−0.0012 mean) → regime gating required (F46 3rd rhyme).
      Eye pass ✅: corrections hug the fringe, contrast visibly restored toward GT,
      no ringing/halos/banding; mild speckle in recovered dark regions (measured).
- [x] FINDINGS F51 + FRONTIER 19 status (19b/c/d spawned) + F27 epitaph. ✅ CLOSED.

## FRONTIER 20 first pass (same session) — ✅ CLOSED (F52)
- [x] Gap factory (gapfill.py): 3 wavy depth bands, r_gap=6.5px both frames,
      exact-kernel regime, gap-eval eroded from seams.
- [x] P0: deficit real (gap contrast 0.51 mean), oracle RL ceiling +0.051 → proceed.
- [x] R6 RL: monotone to k=15; Lucy turnover at k=40 (worst −0.0088) — early
      stopping IS the regularizer, measured. R7 gain-controlled ≈ plain RL.
- [x] R8 Wiener one-shot λ=0.05 WINS: +0.0544/+0.0340 worst, off-gap 0.0000;
      wrap-pad pitfall eye-confirmed and fixed (replicate-pad 4r).
- [x] R9: ±15% radius error → most of win kept; naive residual scale-selection
      DEGENERATE (under-deconv bias) → 20b.
- Next phase (see FRONTIER 20b/c/d): DFF radii + calibrated selection, TV/α<2/3
  rungs, asymmetric/ramp/giant gaps, gap detector + outcome gate, real stacks.

## Follow-up phases (open only on a FINAL win)
1. Gate retrain on hybrid outcomes (F47 recipe) — needs a ~100-scene wide-occluder
   factory (objects-as-occluders per F43); current t2 gate trains on objocc.
2. Estimated-matte rung (F41 pattern) + R3 trust-mask fallback.
3. L1 re-degradation audit as gate feature / runtime self-audit (FRONTIER 17).

## P11–P14 support-ordering checkpoint — ✅ PARTIAL (F58/F59)
- [x] Detached owner-frame satellites: <=2% area, <20% seed overlap, >=90%
      inside a 1.5-CoC neighborhood, and >0.01 captured-frame fit improvement.
- [x] High-overlap parent silhouettes: >=90% seed containment, >=0.80 IoU,
      <=2% novel area, the same neighborhood constraint, and both >0.01 absolute
      and >5% relative captured-frame fit improvement.
- [x] Hard-select only novel observed support; veil application is zero there.
- [x] Freeze the parent rule before scenes 175–199. Fresh `scene_178` flips from
      harmful on SSIM/MAE/MSE/fringe with support off to positive on all four;
      `scene_184` improves further. Exact composed P14 matches direct P13.
- [x] Regenerate the five-case owner lab with fixed scene-114 and scene-122
      coordinate crops. The scene-122 point error falls 6.33→2.33.
- [ ] **S10: continuous outer-support evidence.** Separate opaque owner core from
      uncertain blur spread using positive revealed-background observations; do
      not infer this from the mixed base or tune it against SSIM.
- [ ] **S11: older candidate-license tail.** Localize `scene_172`'s −0.000416
      SSIM dissent (direct MAE/MSE/fringe still improve) and decide whether
      identity or a spatial veto is the physically correct candidate.

## F60 optical-foundation reset — 🔶 ACTIVE
- [x] Replace the V1 >12 px box shortcut with exact circular-aperture convolution.
- [x] Save frame-specific coverage and separate solid/mixed/thin optical strata.
- [x] Clean cutout boundary radiance; brute-aperture equivalence test passes.
- [x] Freeze V2 dev/holdout/extension before judging the corresponding bank.
- [x] Unchanged runtime: 0/24 initial V2 fires (identity). Oracle ceiling:
      dev 9/9 fired positives; holdout 7/8 SSIM+MAE positive, 8/8 MSE/core-safe.
- [x] Frozen 36-scene extension: 2 fires improve global direct errors; parent
      support independently survives, but one worsens inner-partial foreground.
- [x] Safety-disable giant-veil auto; keep contour reconstruction live.
- [x] **S12: positive background observation gate.** ✅ CLOSED (F61). Focused
      foreground is a veto; defocused foreground licenses rear recovery only
      where another frame positively observes rear structure. Cross-PSF coverage
      returns far background to identity. Both diagnosed extension fires and the
      single fire on a fresh 36-scene post-rule split improve all four optical
      partitions; 35/36 fresh scenes refuse.
- [x] **S13/S16: owner-frame geometry replacement + front reconstruction.**
      ✅ CLOSED (F62). A same-object focused-owner silhouette may replace the
      mixed-base matte only after an absolute forward-fit win. Its eroded,
      cross-PSF partial-coverage interior is copied from the focused owner before
      rear recovery. The only fire on the S16 development split flips from
      globally/partition harmful to all-partition positive; 35/36 refuse. A
      genuinely post-final S19 split then produces 3/72 fires, all positive on
      SSIM/MAE/MSE and all four physical partitions; 69/72 refuse exactly.
- [ ] **S14: real aperture calibration.** Compare V2 disk coverage/PSF against a
      controlled first-party macro occluder capture; exact synthetic optics are
      still not real optics.
- [ ] **S15: fine-band causal localization.** The seven current research fires
      improve global/direct and every physical-partition MAE but raise the
      GT-credited smooth-veil fine-band error. F64's consensus-qualified
      front-first ordering reduces the seven current tails to
      +0.0011…+0.0042 but does not close them. Map those pixels against coverage
      slope, support boundary, quantization, channel, and solver disagreement.
      Prefer local identity or analytic shrinkage over thresholding the aggregate
      metric. Require a new post-rule fire with a non-positive complement tail
      before considering auto re-enable.
      **PAUSED / subordinate to input repair:** a single cached F64 attribution
      on `s23_007` found front copy `+0.000229`, float rear correction another
      `+0.000620`, and uint8 truncation another `+0.000276` false-texture error.
      An inward-only support taper reduced but did not close the tail. Do not
      promote that experiment or continue S15 until the one-sided opaque
      formation contract in `OCCLUSION_FORMATION.md` is implemented and F64 is
      regraded.
- [ ] **S16: formation taxonomy and transmission.** 🔶 ACTIVE; S15 runtime
      promotion is paused. Preserve V2's brute-aperture-verified opaque renderer,
      but stop pooling substantial-core, slender/all-veil, truly transmissive,
      and malformed-overlay inputs. Add explicit material/optical metadata,
      make substantial-core opaque scenes the primary validation cohort, keep
      all-veil opaque geometry as a named stress cohort, and add a separate
      scalar-transmission factory with saved foreground/background/opacity
      latents. Re-evaluate F62 by regime before further gate tuning. Design and
      invariants: `OCCLUSION_FORMATION.md`.
- [x] **S17: local opaque-owner confidence.** ✅ F64. A whole-mask forward win
      cannot certify each pixel. On the new 36-primary/24-boundary/12-all-veil
      S23 cohort, one primary fire exposed 18,331 far-background changes despite
      0.940 alpha IoU. Comparable owner proposals now require a 75% local
      supermajority for hard front copy and independently for their PSF fringe;
      parent novel support is clamped, satellites remain separate licensed
      hypotheses. S23 rerun: 7/7 SSIM/MAE/MSE and all-partition positive,
      far background exact for every fire, 65 refusals identity. S23 is
      development evidence; do not spend another large split until S15 freezes.

## Guardrails (doctrine)
No steering by q_abf_ms/q_ssim composite or any no-ref source-similarity (F45);
factory GT + region metrics + eye only. Base band never gains (DC). Gain field
uses pyrDown pyramids of full-res ab/mask (never disk_blur of downsampled alpha).
Package (`src/`) untouched until gates retrain on hybrid outcomes.
