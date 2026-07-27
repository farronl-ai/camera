# Project state — start here after `MISSION.md`

Last compacted: 2026-07-26.

## Current answer

The narrow two-frame one-sided opaque recovery path is enabled in
`--enhance auto`. Preserve the F78 output at commit `bf99365`: on the inspection
page, the right side of `s29_010` is the user-validated near-perfect
transmission-boundary result.

The core fix is formation-aware rear inversion:

```text
B_direct = (O_rear - V_front) / T_rear
```

with local two-frame PSF selection, a 10% aperture-coverage contour,
component-specific below-noise extrapolation, and contour-relative integration.
Do not replace it with generic seam smoothing.

For ordinary N-frame stacks, F79 now detects fragmented, low-confidence local
focus ownership. Unstable stacks snap one shared guided decision to coherent
hard frame regions before multiband reconstruction; stable stacks retain full
per-band selection. Fine details are never softly double-imaged, while coarse
decision transitions remain feathered. This does not affect the two-frame F78
path.

F80 makes the alignment contract explicit: only the scene footprint observed
by every frame may reach fusion. `align_stack` warps validity masks, intersects
them across all N frames, and crops all aligned images to the largest all-valid
rectangle. Never reintroduce reflected/replicated warp borders as image data.

F81 answers the remaining kitchen geometry — true depth-dependent parallax,
because handheld rotation pivots the device rather than the lens entrance pupil
and the camera centre therefore translates. `align_stack` now runs a depth-aware
second pass after the global warp: depth bins cut at depth-histogram valleys,
one translation-only ECC correction per bin, blended into a single dense field,
relaxed wherever it would stretch rather than transport content, and resampled
exactly once. Do not answer this with a more flexible single global warp, and do
not replace valley edges with quantiles — that puts a seam through the middle of
an object. F79's coherent source route stays as the safe fusion fallback.

F82 adds the second half: parallax uncovers scene, so `align_stack` also
returns per-pixel `usable` masks and fusion refuses those pixels. This is
separate from F80's rectangular crop and must stay separate. Derive the ribbon
from the MEASURED per-bin displacement, never from the smoothed applied field,
and keep the radius ladder — a single-scale test condemned 38.6% of a frame.

Three alignment negatives are load-bearing (F81b, F82b): a model linear in the depth
proxy loses to nonparametric bins, and the alternating motion/depth/calibration
estimator (`depth_model="joint"`), while achieving the best registration of any
variant, invents motion on the zero-motion sentinel and is not promotable. Do
not enable it by default without fixing its observation model first.

No-reference metrics cannot compare two alignments (F81a): they score against
the aligned sources, which the alignment itself changes. Judge alignment by
registration residual per depth region, the GT parallax factory, and
disagreement-guided crops.

## Read order

1. `MISSION.md` — post-doctrine objective and evidence rules.
2. `STATE.md` — current checkpoint and commands.
3. `FINDINGS.md` — load-bearing scientific conclusions and retired approaches.
4. `OCCLUSION_FORMATION.md` — renderer/inverse physical contract.
5. `FRONTIER.md` — prioritized open work and literature anchors.
6. `DEVSTYLE.md` and `PLAYBOOK.md` only when planning a longer research loop.

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

Historical scripts, phase plans, generated reports, caches, and legacy inspector
cohorts were removed from the working tree. Recover them from Git only if a
specific old result must be reproduced.

## Fast working loop

Use compact causal probes and the same three visual sentinels until the user
returns the project to full validation:

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

Refactor formation-state estimation upstream without changing pixels:
`V_front`, `T_rear`, PSF weights/choice, forward residual, and detection floor
should be computed once while both original observations are available and
passed explicitly into recovery. Byte-compare the refactor against F78 on the
quick cohort.

After that, freeze a fresh cross-family validation split. Do not tune further
on `_010`.

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

## Git checkpoint

Commit and push logical checkpoints without AI attribution trailers. Keep the
tree clean and do not accumulate generated experiment ledgers in the repository;
only current frozen evidence belongs in Git.
