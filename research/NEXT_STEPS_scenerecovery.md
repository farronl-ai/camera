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

## Guardrails (doctrine)
No steering by q_abf_ms/q_ssim composite or any no-ref source-similarity (F45);
factory GT + region metrics + eye only. Base band never gains (DC). Gain field
uses pyrDown pyramids of full-res ab/mask (never disk_blur of downsampled alpha).
Package (`src/`) untouched until gates retrain on hybrid outcomes.
