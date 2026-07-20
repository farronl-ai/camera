# Plan: breadth phase — frontier inventory + the N-frame blind spot

## Context

The engine just converged (perband default, consolidated SYNTHESIS theory, F1–F23).
Farron's directive: the expert style is right, but **keep breadth of mind — this is
the beginning of a leading-edge project, not the end**. A clean theory is exactly
when blind spots calcify. So this phase makes breadth structural: a living frontier
inventory in the repo, plus immediate probes of the biggest blind spots.

**The biggest blind spot (B1):** nearly ALL validation used 2-frame stacks; real
focus stacking uses 5–50 frames (macro rails, microscopy z-stacks). Specific,
testable theory risks at N>2:
  - `harden` confidence = (top1−top2)/top1 focus energy: with many focus planes,
    ADJACENT planes have near-equal energy → confidence collapses → hardening
    silently disables exactly when stacks get deep.
  - Soft weights normalize over N frames → weight mass spreads across N−1 blurred
    frames → contamination grows with N.
  - ECC alignment to the middle frame: distant frames have larger displacement →
    convergence risk grows with stack depth.

**Second (B2):** real optical multi-frame data exists and is public-domain-friendly:
microscopy z-stacks (Broad Bioimage Benchmark Collection, Cell Image Library) —
REAL optical defocus + many frames + high-res, killing the two standing gaps at
once. Best-effort acquisition (don't-thrash rule applies).

**Also inventoried (B3–B5):** occlusion-boundary physics, per-band metric, depth-map
byproduct — see milestones.

## Milestones

### B0 — living frontier doc + memory + plan-in-repo
`research/FRONTIER.md`: the breadth inventory (below), maintained like FINDINGS.md.
**Commit THIS plan into the repo** as `research/NEXT_STEPS_breadth.md` (per Farron:
the plan itself must live in git for reference, like NEXT_STEPS_hires.md before it).
Update marathon-method memory with the "don't close in after converging" guidance
(re-read the file first; last edit failed on stale content).

### B1 — N-frame stress test (immediate, synthetic, hypothesis-driven)
Extend generation to N∈{2,4,8} frame stacks (hires_gen/`frames` param + hardbench
scenes at moderate res for compute). Measure per method (perband/blend/pyramid,
harden on/off): GT-SSIM vs N, and DIRECT probes of the two hypotheses:
  - conf-collapse: distribution of harden confidence vs N (does it →0?);
  - weight dilution: mean weight mass on the true-sharpest frame vs N.
If confirmed, design the fix from the mechanism (candidates, chosen by data): conf
vs the max over NON-adjacent planes; or per-pixel top-K frame preselection before
normalization; or pairwise cascade fusion. Verify: quality vs N flat-or-rising,
2-frame results byte-stable or non-regressing. Promote only if non-regressing.

### B2 — real N-frame optical data (best-effort)
Probe BBBC / Cell Image Library / light-field focal stacks for downloadable real
z-stacks. If obtained: align + fuse with the engine, evaluate no-ref (q_ssim) +
eye-analysis 2.0 (no GT expected), catalog real-optical failure modes (this is the
standing "real high-res optical" gap too). If blocked: document, move on.

### B3 — occlusion-aware generator realism
Real defocus at depth edges has α-matte partial occlusion (blurred foreground edge
semi-transparently overlays sharp background); my generator uses hard per-pixel
depth indexing. Upgrade defocus_ca to matte-composite near/far layers (soft alpha
from the blurred near-mask). Then RE-CHECK the perband-vs-blend-vs-pyramid ranking
on the more honest benchmark — conclusions that flip here were synthetic artifacts.

### B4 — per-band Q_ABF metric (transplant the perband lesson into the metric)
Q_ABF computed per pyramid band, combined across bands → scale-robust gradient
transfer (fixes F17's high-res collapse structurally, same way perband fixed the
engine). Re-validate against GT per regime (low-res, high-res); if it transfers,
re-calibrate the composite and revisit per-region metric guidance (F12).

### B5 — depth map as a free byproduct (feature)
The N-frame fusion decision ≈ depth-from-focus. Emit an optional depth map
(winner-index smoothed/guided) via `--depth-out`. Cheap, useful, and feeds future
occlusion/object reasoning.

## Reuse
hires_gen (frames param, defocus_ca), hardbench scenes, mixed_gen, metrics (+maps),
eyetool.compare, fuse_* family, align_stack, FINDINGS/PLAYBOOK/report machinery.

## Verification
- B1: hypothesis plots/tables (conf vs N, dilution vs N), GT-SSIM vs N per method;
  any fix validated at N=2 (non-regression) AND N=8; pytest green; eyetool crops.
- B2: real z-stack fused; q_ssim + disagreement-guided visual review.
- B3: ranking re-checked on matte-composited benchmark; note any flips in FINDINGS.
- B4: per-band Q_ABF Spearman vs GT at both resolutions vs current Q_ABF.
- B5: depth map visual sanity on N-frame stacks + test.
- Commit per milestone; FINDINGS entries; report + republish at phase end.

## Order
B0 → B1 (core of this phase) → B2 in background alongside → B3 → B4 → B5.
Each stays honest: hypothesis → measure → look (eyetool) → fix from mechanism →
non-regression gate → promote.
