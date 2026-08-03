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

**The blocker is the region grab-bag, as originally diagnosed (F84/F85), not
scale.** The apparent magnification that F87/F88/F90 chased was a broken left-edge
width measurement; the bottle does not grow vertically, real breathing is ~1.5% raw
and the affine removes it, and a known-answer-validated estimator measures the
bottle at scale 1.003–1.008 with ~+19 px of pure translation (F91). Do not build a
scale term until a region's scale has been measured with a validated instrument.

**SHIPPED (F102): motion-group override is in the runtime**, default on, in
`src/focusstack/motion_groups.py`. The kitchen bottle's residual falls 20.14 -> 2.51
px and its ghost band is gone; the factory, zero-motion and small-motion sweeps are
bit-identical because nothing disagrees there. Open: alignment now costs 23-67 s
because every material edge is profile-matched against every frame, and the
still-stack gate meant to avoid that does not fire (a bin shift exceeds 1 px even on
zero-motion). Large-motion changes 28% of pixels for -0.004 Q_SSIM and has not been
looked at.

**F100 corrects the bottle: residual 19.97 -> 1.47 px, ghost band visibly gone**
(`research/group_align.py`). Motion-group alignment — material edges, consensus
grouping, focal-weighted motion with temporal propagation, support from each group's
convex hull. Two painting bugs mattered more than any estimator: claim strength is a
GATE not a scale factor (a fully-owned pixel was losing 30-42% of its correction),
and support must be the hull of a group's features, not their neighbourhoods (the
bottle's top and bottom got nothing). Both of those are now done (F101): focal-seeded grouping engages on the factory at
97% purity, and refusal carries through. But the group path REGRESSES the factory
(0.900 vs 0.971) because clean depth separation is exactly where bins are ideal.
Integrate as a TARGETED OVERRIDE: keep the shipped depth-bin path and apply a motion
group's correction only where it demonstrably disagrees with its bin — a bin fitted
to +2.3 px containing a group measured at +18.5 px is not a marginal call. Watch the
purity/coverage trade: finer groups are purer but their hulls cover less, and
coverage is what makes a correction land.

**Its ingredients were the ones F99 found already solved and disconnected.**
F89 measures that object at +18.88 vs +19.2 truth by using frames near the reference
and propagating; F93 identifies which features are it (92.9-100% pure). Do NOT key
the correction on depth — a depth-keyed curve never exceeds +/-5 px at the bottle's
depth in any frame, because other content shares its depth value with different
motion. Build spatial support from the motion group's own features, which is easy
because the group is compact, and propagate temporally where the object is blurred.

**Grouping works at FEATURE level; pixel regions are the open problem (F98).** Do
not spend more effort converting feature groups into region masks by propagation —
two constructions were tried and neither beats the shipped valley depth bins. The
promising unexplored route is to skip regions: evaluate the per-feature motion model
through the dense field it implies, with no hard boundary.

**Object grouping is solved (F93), motion estimation is not.** Defining an object as
a maximal feature set admitting one rigid motion keeps the kitchen bottle whole (20
of 21 features inside it) where every earlier grouping fragmented it. What remains is
accuracy: translation-only propagation from nearby frames reaches ~+19 px against
+19.2 truth, all-frame propagation only +16.07. Wire grouping into the region model
before revisiting the split gate.

**Then regions.** Residual-driven splitting recovers the bottle (+2.5 → +18.8 px,
artifact visibly gone) but regresses the factory, and no tile-confidence floor serves
both scenes. Do not tune that threshold — replace it with the physical test: a real
object's residual scales with each frame's motion and holds its direction, a bad
phase correlation does not. Promotion also needs `_REFINE_MIN_BIN_FRACTION` and the
acceptance cap relaxed together, since a small region asking a large correction is
exactly what they reject. Object integrity is a MERGE rule; keep the tolerance ≤2 px
or genuinely different depth planes merge and the result collapses.

**The per-region two-frame architecture (user design) is BUILT and SHIPPED
(F109–F112)** — `src/focusstack/twoframe.py`, routed in `pipeline.run` (default
on, `--no-twoframe-route`). Route rule: engage iff the motion-group override
fired AND every elected layer sits within the refinement licence (1.5% of the
diagonal); factory/zero/small stay shipped byte-identical, large-motion is
honestly vetoed. As of F112 (pair fusion one-hot via `fuse_coherent` + the
`same_surface` precondition) it BEATS the shipped path on the analytic factory
(0.9795 vs 0.9728) and the user's four marked kitchen defects are closed.
`--enhance` is correctly skipped on the routed path (F56's licence does not
cover a stitched composite).

## Standing rules for this area

- Depth grouping comes from the per-feature FOCAL SIGNATURE (F97), not from a motion
  threshold — no motion threshold serves two scenes. Motion confirms rigidity within
  a depth. Next orthogonal signal, unbuilt: veil vs defocus, one-sided attenuation
  versus symmetric spread.
- Before diagnosing a stack, RUN `research/motion_components.py`: it reports
  breathing, forward translation, rotation and lateral parallax with a residual, in
  one pass over all material edges. On kitchen it says breathing ~1.000, rotation
  ±0.5deg monotone, forward translation up to 4.3%, lateral parallax dominant.
- Fit object motion on MATERIAL edges (print, texture, corners); a curved object's
  silhouette is a limb that slides with the viewpoint and is not a rigid feature
  (F92). Nine printed edges on the bottle agree to ~1 px; its left limb reads half.
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

The scene-model second pass (FRONTIER §7b), in managed rounds. DONE: round A
(F113, the certifier), round B1 (F114, focal-signature decomposition), round
B2 + two corrections (F115, the assembly — factory 0.982061 beats every prior
path; knob repaired; `SURFACE_SIGMA` retired by cross-convolution physics, not
yet ported to `twoframe.same_surface`). Research-only so far: the second pass
lives in `research/scene_model.py` and rewrites the routed composite behind
strict-subset vetoes.

DONE also — B3a (F116, contour continuity), round C (F117, runtime
retirements: cross-convolution + pooling in the default path, pair-aware
refusal; licence-before-render DROPPED, still open), and aligner rounds 1–2
(F118): `research/aligner.py` implements the FRONTIER contract end to end —
transform solved, wall-smear class structurally impossible, kitchen run
end-to-end through unmodified `fuse_perband`.

NEXT — the aligner promotion arc, in order (full 8-item list in
`aligner_NOTES.md` Round 2):
1. Replace the merge tolerance with a fit-uncertainty (model-selection)
   test — BOTH scenes' residual segmentation error is this one decision.
2. Something must PROPOSE cuts from motion (the merge only removes; the
   kitchen's objects were never seeded apart from the counter).
3. Close the zero-motion anchor (identity snap + half-pixel border slack —
   currently a 1 px off-footprint rim, zero interior gaps).
4. `travel` → displacement at the piece centroid (occlusion order rests on
   it); isolate the K2b gate (a live defect since the merge cleaned the cut
   set); per-piece photometric veto (27% whole-frame withdrawal needs a
   diagnosis); certifier on the kitchen aligner output.
Then F101 routing into the pipeline (byte-identical sentinels), and the
cascade re-verification under heavy masking per FRONTIER's integration
expectation. The user's eyes on the inspector layers (1 routed / 4
scene-model / 5 aligner) remain the standing audit.

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
