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
- [ ] **P0** kill switch: c_k calibration, byte-identities (no-D == fuse_perband;
      omega=0 == corr_multi), GT/GT contrast ratio = 1, empirical g(ab) curve,
      headroom (proceed iff post-subtraction cr_mid < ~0.9).
- [ ] **H1** clamped in-loop gain sweep: t0 ∈ {.10,.15,.25,.40} × ω ∈ {.9,.95,1} ×
      law ∈ {lin, sq, emp}. **H1a** full-image control (F27 placement) — expect worse.
- [ ] **H2** analytic shrink-after-gain (m ∈ {1,2,3}) vs **H3** guided denoise of the
      correction, guide = far frame's own gray band (remnant provenance; NEVER the
      owner frame — it carries the occluder), eps = β·(σ·c_k)², r ∈ {2,4,8}.
      Head-to-head on H1 winner; compose if both positive.
- [ ] **H4** (conditional: finest-band grain excess) cross-scale coherence shrink.
- [ ] **H5** (conditional: 8-bit trails float > 2e-3 or banding by eye)
      quantization-bin projection through the forward model.
- [ ] **FINAL** winning stack × 10 backgrounds × coc {0.04, 0.012 off-regime
      no-harm} × {8-bit, float same-seed}. Eye pass: clamp-edge + max-disagreement
      crops, GT alongside (artifact detection only).
- [ ] FINDINGS entry + FRONTIER 19 status + F27 epitaph; commits per milestone.

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
