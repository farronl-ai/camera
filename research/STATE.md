# Project state — start here after `MISSION.md`

Last compacted: 2026-07-26.

## Current answer

**Recovery (F78).** The narrow two-frame one-sided opaque path is enabled in
`--enhance auto`. Preserve the output at commit `bf99365`: on the inspection page,
the right side of `s29_010` is the user-validated transmission-boundary result. The
core fix is formation-aware rear inversion,

```text
B_direct = (O_rear - V_front) / T_rear
```

with local two-frame PSF selection, a 10% aperture-coverage contour,
component-specific below-noise extrapolation and contour-relative integration. Do
not replace it with generic seam smoothing.

**N-frame fusion (F79).** Fragmented, low-confidence focus ownership routes to one
shared guided decision snapped to coherent hard frame regions before multiband
reconstruction; stable stacks keep full per-band selection. Two-frame stacks and the
F78 path are unaffected. This remains the safe fusion fallback.

**Alignment (F80–F82), shipped and default on.** Three separate contracts, and they
must stay separate:
- only the scene footprint observed by every frame may reach fusion — validity masks
  warped, intersected, cropped to the largest all-valid rectangle (F80);
- a depth-aware pass after the global warp — bins at depth-histogram VALLEYS, one
  translation-only ECC correction each, blended into one dense field, relaxed where
  it would stretch content, resampled exactly once (F81);
- per-pixel refusal of scene that parallax uncovered, carried into fusion as a
  `usable` mask, derived from MEASURED per-bin displacement and sized by a radius
  ladder (F82).

Analytic factory: 0.8754 → 0.9660 → 0.9785 GT-SSIM. Near-content residual on the
real sweeps roughly halves to quarters; the zero-motion sentinel is left alone.

## What is not solved, and the order to attack it

**Breathing first.** Focus breathing is a real 14% magnification on the kitchen
sweep (bottle width 138 → 160 px raw, still 140 → 154 after the global affine). Per-
bin TRANSLATION cannot express it, since a scale error moves an off-centre object's
two edges by different amounts. It is upstream of everything else: the region
machinery separates objects correctly when the geometry is representable (factory
IoU 79%, object in one region) and fragments the kitchen bottle only because a
magnifying object needs different translations across its extent. A crude 40%
breathing removal already lifts bottle IoU 14.8% → 23.8% and coverage 22% → 42%
(F88). Fix it at the global stage — its per-frame magnification is a clean monotone
signal — then re-measure before changing the region machinery further.

**Then regions.** Residual-driven splitting recovers the bottle (+2.5 → +18.8 px,
artifact visibly gone) but regresses the factory, and no tile-confidence floor serves
both scenes. Do not tune that threshold — replace it with the physical test: a real
object's residual scales with each frame's motion and holds its direction, a bad
phase correlation does not. Promotion also needs `_REFINE_MIN_BIN_FRACTION` and the
acceptance cap relaxed together, since a small region asking a large correction is
exactly what they reject. Object integrity is a MERGE rule; keep the tolerance ≤2 px
or genuinely different depth planes merge and the result collapses.

**Then the per-region two-frame architecture** (user proposal, not yet built):
per region pick the frame where its foreground is sharpest and the frame where its
background is sharpest, align that pair, run it through the base processor, stitch.
It plays to the engine's most-validated path. Two cautions: the stitch reintroduces
exactly the boundary problems this arc fought and needs its own validated stage
(F79 already found hard region-copy seams unacceptable); and `--enhance` is licensed
for exactly two frames ≤1600 px (F56), so a stitched composite is outside it.

## Standing rules for this area

- Textureless interiors take their motion from EDGES, correlating GRADIENT profiles
  (intensity correlation reports a rigid object widening by 17 px). Trust only the
  normal component. Measure near an object's focal plane, then propagate.
- Interior edges make "one object?" falsifiable; two edges alone can only be solved,
  never tested.
- No-reference metrics cannot adjudicate an alignment change — they score against
  sources that alignment itself moves.
- Do not turn veiling into a hard mask: it loses in every regime including the one
  built to favour it. A veiled pixel is a mixture, not an absence.
- Do not revisit one-sided disocclusion refusal: the ordering cue works and the
  conclusion still fails, because a defocused occluder's own matte mixes background
  onto its boundary.
- Do not test alignment on frames differing by one global transform — use
  `research/parallax_gen.py`.
- Known-answer test every new measuring instrument before believing it.

## Read order

1. `MISSION.md` — post-doctrine objective and evidence rules.
2. `STATE.md` — current checkpoint and commands.
3. `FINDINGS.md` — load-bearing scientific conclusions and retired approaches.
4. `OCCLUSION_FORMATION.md` — renderer/inverse physical contract.
5. `FRONTIER.md` — prioritized open work and literature anchors.
6. `DEVSTYLE.md` when planning a longer research loop.

`PLAYBOOK.md` §0 (what is true / which tool when / what is settled) is not optional
for technical work — it is the distilled domain theory, and §0c lists experiments
already run and closed. Reading it is cheaper than repeating them.

## Live code

- Runtime: `src/focusstack/`
- Recovery implementation: `src/focusstack/veil_layers.py`
- Auto routing/composition: `src/focusstack/enhance.py`
- S29 factory: `research/objocc_v2_gen.py`
- S29 evaluator: `research/objocc_v2_eval.py`
- Focused-owner proposals: `research/owner_candidates.py`
- Inspector generator: `research/make_showcase_specialists.py`
- Inspector HTML builder: `research/make_showcase_html.py`
- Current evidence: `research/objocc_v2_s29_{manifest,formation_audit,geometry_audit,ordered_visibility}.json`
- N-frame routing: `selection_instability_score` and
  `stack_consistency_route` in `src/focusstack/fusion.py`
- Alignment: `src/focusstack/align.py` (global warp + depth-aware pass + `usable`
  masks); fusion honours `usable` in `fuse_perband`/`fuse_coherent`
- Alignment instruments (research only, none in the runtime path):
  `parallax_gen.py` analytic parallax factory with GT · `adaptive_bins.py` motion
  splitting and merge · `edge_motion.py` edge-driven object motion ·
  `boundary_probe.py` veiling and bin homogeneity · `occlusion_order.py` front/back
  ordering

Historical scripts, phase plans, generated reports, caches, and legacy inspector
cohorts were removed from the working tree. Recover them from Git only if a
specific old result must be reproduced.

## Fast working loop

Keep the early loop SMALL — the alignment arc resolved nine findings on one
analytic factory plus one real sweep, with no benchmark run. Scale data when the
mechanism stabilizes, not while finding it.

For alignment/geometry work:

```bash
.venv/bin/python research/parallax_gen.py      # GT factory: is the mechanism right?
.venv/bin/python research/adaptive_bins.py     # kitchen: does it move the bottle?
```

For recovery work, the same three visual sentinels as before:

```bash
.venv/bin/python research/make_showcase_specialists.py inspection
.venv/bin/python research/make_showcase_html.py inspection
```

The current quick cohort is `s29_002`, `s29_007`, and `s29_010`. Update only
their assets for visual iteration when asked. Do not launch a full split merely
to inspect one mechanism.

Before a real promotion:

```bash
.venv/bin/pytest
```

Then freeze parameters, generate a fresh post-rule family, run formation and
geometry audits before expensive solves, run the complete evaluator, inspect
the failures, update the current four evidence JSONs, and regenerate the
inspector once.

## Immediate next move

Remove focus breathing at the global stage, then re-measure object separation
before touching the region machinery (see "What is not solved" above). The
magnification signal is clean and monotone; the current global affine compromises
between it and depth-varying parallax, which F81 established cannot be co-fitted.

Still queued behind it, unchanged: refactor formation-state estimation upstream
without changing pixels (`V_front`, `T_rear`, PSF weights/choice, forward residual,
detection floor computed once while both original observations are available and
passed explicitly into recovery; byte-compare against F78 on the quick cohort),
then freeze a fresh cross-family validation split. Do not tune further on `_010`.

## Invariants

- Opaque hard ownership has zero rear throughput.
- Foreground blur spreads foreground outward; it does not pull hidden rear
  detail inward.
- Focused foreground is a hard front-order observation.
- Rear recovery requires positive non-focal visibility evidence.
- Failed gates return the base byte-for-byte.
- Far identity and all protected regions receive zero rear application.
- Formation quantities come from original observations, never the fused base.
- Extrapolation is component-specific and permitted only below the modeled
  observation detection floor.
- Global metrics inform but do not overrule physical partitions and visible
  coherent artifacts.
- A hard mask is only for observations that do not exist; partial contamination
  takes a soft weight.
- A measuring instrument is not trusted until it has returned a known answer.

## Git checkpoint

Commit and push logical checkpoints without AI attribution trailers. Keep the
tree clean and do not accumulate generated experiment ledgers in the repository;
only current frozen evidence belongs in Git.
