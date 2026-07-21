# Plan: E-phase — the Boundary Engine (two half-marathons: build, then integrate)

## Context

The engine's residual artifacts concentrate at **object boundaries** — the hard-lines
test still shows visible defects at the best progress point, and F27 proved the fringe
error is *decision-boundary* error. The missing data product is a **true boundary/
object-structure map**: real edges and object boundaries (not just local contrast),
including nested objects and partial occlusion (half-in-front/half-behind), plus the
layer adjustments to consume that data. Per Farron: ML-based or complex-algorithmic;
**complementary to every existing layer, never a replacement** (replacement wins some
cases and leaves holes our too-clean synthetic data won't reveal); a half-marathon to
build, another to integrate. Discipline and the expert flow (DEVSTYLE/PLAYBOOK) apply
maximally: hypothesis → measure → look → A/B → all-regime non-regression gates.

## Architecture (the design bet, stated up front)

**A boundary is a depth discontinuity — and a focus stack SEES depth.** Single-image
detectors know appearance; our stack additionally knows: (a) edges that appear/vanish
across focal planes (an object contour is sharpest in its own plane — max-over-frames
edge response is defocus-robust), (b) winner-map discontinuities, (c) depth-from-focus
jumps. So the Boundary Engine fuses THREE ORTHOGONAL evidence channels — orthogonality is
the design requirement (Farron: parallel vectors add no information; our engine
already senses every gradient-family signal, so additions must carry information
local math cannot derive from this scene):
  1. **Stack channel** (physical): depth measured by the stack itself — defocus-robust
     max-over-frames edges, winner discontinuities, focus-depth jumps. Metric, but
     texture-only.
  2. **Semantic channel** (learned priors from millions of OTHER images): monocular
     depth (Depth-Anything-V2-Small, 2024) — dense object-level depth everywhere,
     relative not metric; its discontinuities are object boundaries WITH near-side
     everywhere. Plus segmentation masks (MobileSAM/FastSAM-class) — closed regions +
     containment hierarchy (objects-within-objects, topology local operators never
     compute). The two depth sources CALIBRATE each other: focus-depth anchors the
     monocular scale at textured pixels; monocular fills the textureless holes.
  3. **Perceptual-boundary prior** (lightweight fallback only): PiDiNet/Structured
     Forests — boundary-trained but gradient-fed, hence only partially orthogonal;
     used when the semantic channel is unavailable.
Producing:
- `B(x,y) ∈ [0,1]` — soft true-boundary strength;
- **near-side tags** per boundary pixel (which side is closer — from depth/winner):
  this encodes occlusion ("half in front, half behind"; occluding contours are OWNED
  by the foreground object);
- (stretch) a nested region hierarchy (multi-threshold watershed over B) — objects
  within objects.

Integration is **additive**: B feeds the existing guided decisions (guide enrichment +
ε modulation), the per-band machinery (B-pyramid per level), and harden (occlusion-side
ownership at boundary pixels). No layer is removed; every injection is independently
ablatable and gated.

## Environment & feasibility (verified)

- `opencv-contrib-python-headless` 5.0.0.93 ships **abi3 manylinux wheels** → installs
  on Py3.14 → `cv2.ximgproc.createStructuredEdgeDetection` (learned structured-forest
  edges; model ~60MB download) in the MAIN env. Contrib must replace/co-install with
  current headless package — verify import + no regression of existing cv2 usage.
- PiDiNet (deep, <1M params, CPU-fast, BSDS-trained) via the existing `.venv312` torch
  env as a **subprocess bridge** (png in → .npy out). Torch stays OPTIONAL: the package
  boundary engine must degrade gracefully (stack+forest, then stack-only).
- No GPU. Heavy runs → background jobs. Datasets/models gitignored (`research/data/`,
  add `research/models/`).

## Half-marathon 1 — build the Boundary Engine

### E0 — baseline quantification + scaffolding (the metric this phase optimizes)
- Commit this plan as `research/NEXT_STEPS_boundary.md`; FRONTIER entries (new #16
  boundary engine; link 3b/10/12 relationships).
- **Quantify today's boundary artifacts** precisely: define **boundary-band error**
  (mean |err| and SSIM within ±k px of GT boundaries; k∈{2,5,10}) — the aggregate-
  hides-local lesson says global SSIM won't see these gains, so this metric is the
  phase's objective. Baselines on: hard_edges (hires 09), fine-structure benchmark
  (hires_mixed), occ benchmark, + eyetool crops on fence (no GT). Set the numeric
  target AFTER baselining (aim: close ≥50% of gap to GT in-band; near-perfect on
  hard-lines synthetic).
- Feasibility installs: contrib wheel import test; structured-forest model download;
  PiDiNet bridge smoke test (env exists from M4).

### E1 — layered/nested scene generator (boundary GT for free)
`research/layers_gen.py`: K-layer α-matte compositor (2–4 depth layers) over real
Wikimedia photos: nested objects (object cut from a photo placed atop another layer),
objects spanning depth (half-in-front geometry via per-layer masks), thin structures,
per-channel CA defocus (reuse `occ_gen`/`hires_gen` machinery). Emits frames + GT +
**true boundary map + near-side labels** (from layer alphas/depth order) — training/
eval GT for free. Also extract boundary-GT from existing benchmarks (alpha edges,
depth discontinuities). Eye-check scenes for realism/complexity (Farron's warning:
our data is usually too clean — make these deliberately messy).

### E2 — Boundary Engine v1: stack-aware classical-complex (no new deps)
`research/boundary.py`: per-frame multi-scale edges → **max-across-frames**
(defocus-robust) → scale/space coherence filtering → fuse with winner-map gradient +
depth-map discontinuity evidence → soft B + near-side tags (sign of depth step across
the boundary normal). Evaluate: boundary P/R/F vs E1 GT (tolerance px), versus Canny
baseline; eyetool on fence + microscopy + a busy real photo.

### E3 — Boundary Engine v2: the SEMANTIC channel (the orthogonal ML piece)
- `.venv312` bridge (subprocess, png in → .npy out; torch strictly optional):
  **Depth-Anything-V2-Small** monocular depth (probe HF/GitHub weights; CPU ~secs at
  ~518px, fine for offline fusion). Calibrate against focus-depth: robust monotonic
  fit (rank/affine per image) on high-texture pixels where focus-depth is trustworthy
  → a dense, metrically-anchored depth map → boundaries = calibrated-depth
  discontinuities, near-side = depth order, EVERYWHERE (not just textured regions).
- **Segmentation masks** (MobileSAM or FastSAM, whichever bridges cleanly on CPU):
  closed object boundaries + mask-containment tree = nesting hierarchy. Boundary map
  gains closure/topology no local operator produces.
- Fuse channels (stack ∪ semantic; PiDiNet/forest fallback when bridge absent) —
  per 9c transplant-the-property. Eval per channel AND fused on E1 GT (P/R/F) +
  ablation: does the semantic channel find boundaries the stack channel misses
  (textureless, iso-contrast) — the orthogonality test, reported explicitly.
- Fallback chain: semantic→perceptual-prior→stack-only; package guards + tests.

## Half-marathon 2 — integrate without replacing

### E4 — boundary-aware guided decisions (the main event)
Injections, each SEPARATELY ablated:
  (a) guide enrichment: guide' = luminance ⊕ λ·B (B as a second guide channel via
      two-pass guided filtering or B-modulated guide contrast);
  (b) ε-modulation: ε(x) small at high B (preserve true boundaries), larger elsewhere
      (smooth harder within objects — kills within-object speckle without softening
      real contours);
  (c) per-band: downsample B (B-pyramid) and apply (a)/(b) inside `fuse_perband`'s
      per-band decisions.
Gates: boundary-band error improves on E0 baselines; ZERO regression on Real-MFF,
hires_mixed overall, occ, N-frame, microscopy (eye), fence (eyetool), drift; 19+ tests
green; params default-off or provably identity-when-B-absent so the engine never
requires the boundary stage.

### E5 — occlusion-side ownership + cross-boundary blocking
- At boundary pixels, ownership rule: occluding contour belongs to the NEAR side →
  bias decision toward the near frame at B-adjacent pixels (near-side now available
  EVERYWHERE via calibrated semantic depth, not just textured regions);
  implemented as a harden-style confidence injection (reuses that machinery).
- Cross-boundary weight blocking: suppress weight diffusion across high-B lines
  (the remaining hard-lines leak). A/B on hard-lines + nested E1 scenes + occ.
- Same all-regime gates as E4.

### E6 — consolidate + ship
Full regression sweep (all benchmarks + real data + eyetool set), promote defaults
only where evidence is regime-spanning, FINDINGS entries per milestone + SYNTHESIS
update, PLAYBOOK/DEVSTYLE lessons, SHOWCASE addendum (before/after hard-lines),
FRONTIER updates, report republish. CLI: `--boundary [auto|off|stack|forest|deep]`.

## Reuse
`fuse_perband`/`_guided_weights`/`guided_filter`/harden (injection points);
`depth_from_focus`, `content_aware_energies` (stack evidence); `occ_gen`/`hires_gen`/
`disk_blur`/CA machinery (E1); `eyetool`, boundary-band variant of `_ssim_map` eval;
`metrics` (q_ssim local rule); Wikimedia fetcher (`hires_gen.search`); `.venv312`
bridge pattern (M4 scripts); FINDINGS/FRONTIER/report cadence.

## Verification
- E0 defines the objective; every later milestone reports against it + all-regime
  non-regression (the explicit gate list in E4).
- Boundary quality: P/R/F vs E1 GT; integration quality: boundary-band error + global
  metrics + eyetool visual on real images (fence, microscopy, busy photo).
- pytest green throughout; new tests: boundary fallbacks, identity-when-off, ownership
  rule on a synthetic occlusion fixture.
- Commit per milestone (no trailer); plan lives in repo; background jobs for heavy
  compute; findings logged as F30+.

## Honesty/risks (stated now)
- Learned single-image detectors may underwhelm on defocused frames (trained on sharp
  photos) — that's WHY stack evidence is first-class, and why E3 fuses rather than
  replaces (9c).
- Synthetic nested scenes still aren't reality — every milestone includes real-image
  eyetool checks, and the phase does NOT claim victory beyond its validated regimes.
- If an integration point regresses any regime, it ships default-off with findings
  documenting why (F27 precedent: negatives are results).
