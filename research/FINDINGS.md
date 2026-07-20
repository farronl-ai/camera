# Adaptive focus-stacking — running findings log

Persistent notes from the autonomous marathon. Newest first. Pairs metric numbers
with conceptual reasoning and visual inspection (metrics guide, don't decide).

---

## SYNTHESIS — current best understanding (read this instead of F1–F23 in sequence)

**The engine.** Every fusion method = a DECISION (which frame is sharpest, where) +
a RECONSTRUCTION (combine without artifacts). Four properties determine quality:

1. **Edge-aware decision** (guided filter) — else speckle/seams (`max`'s failure).
2. **Confidence-hardened decision** (`harden`) — else defocus *spread* bleeds in and
   thin structures (hairs/wires) grey out. One mechanism, both failure modes.
3. **Multi-scale decision** — else the method fails at whichever scale its window
   wasn't sized for (the cause of every low↔high-res ranking reversal we hit).
4. **Multi-band reconstruction** — else seams at region transitions.

`perband` (the default) is the only method with all four → best all-rounder,
GT-validated at low res (0.9918) and high res (0.9021). `blend` lacks (3) — one
single-scale decision broadcast to all bands. `pyramid` lacks (1)+(2) — halos.
The scale-theory arc (global magic number F19 → measured local-scale map F20 →
per-band F22) resolved cleanly: **put scale-adaptivity in the band structure
itself** — a fixed small window per band IS local scale, at every location, with
no scale-estimation step and no magic numbers.

**The metric.** No single number is trustworthy everywhere. Composite
(0.3·Q_ABF+0.7·Q_SSIM) for *global low-res* ranking; **Q_SSIM alone** at high-res
(Q_ABF's fixed 3×3 Sobel collapses there) and for *all per-region/local* decisions
(Q_ABF anti-correlates locally); Q_MI rejected outright (anti-correlates with
truth). With true GT available, **GT-SSIM is the verdict** — over both no-ref
metrics and the unaided eye's sense of "clean."

**The eye.** Aggregate metrics hide localized artifacts (halos <1% of pixels); the
unaided eye misjudges fidelity (F21). **Eye-analysis 2.0** (`eyetool.py`): crop
where methods *disagree most* + amplified-difference views (+GT when available) —
point the eye at the informative pixels instead of guessing crop locations.

**Structure & operators.** `harden` (confidence-hardening) unifies spread-rejection
and thin-structure preservation. `content_aware` routes laplacian↔mod_laplacian by
local contrast. Operator choice is LOW-leverage; structural/scale handling is where
the quality lives.

**Learning.** Classical foundation first. Learned per-tile routing matches the
oracle; distillation matches classical quality in one pass (speed win needs a GPU);
per-tile no-GT labels are unreliable (~19% GT agreement) → train on GT dev-labels,
deploy feature-only. No answer key at inference — fully achieved.

---

## F23 — perband refined (correctness fixes) + promoted to DEFAULT
Fresh-eyes review found two defects in fuse_perband: (a) base band was a plain MEAN
(imports the defocused frame's low-frequency spread) → now blended with the coarsest
detail band's weights propagated down; (b) coarse-band windows exceeded the band size
(a radius ≈ the whole band degenerates the guided filter into a global mean, ~50/50
blending) → radius/energy_ksize now capped per band. A/B (v1 vs refined v2):
high-res 0.9014→0.9021, low-res 0.9919≈0.9918, fence composite unchanged but output
differs (mean|diff| 0.55). Refined perband now leads EVERY GT-measured regime:
Real-MFF 0.9918 > blend 0.9915 > pyramid 0.9913; high-res 0.9021 > pyramid 0.8926 >
blend 0.8704. On the one no-GT case (fence) the composite prefers blend (-0.005) but
**eye-analysis 2.0 (disagreement-guided crops + amplified diff) favors perband** —
sharper wires, no halo; the composite deficit corresponds to no visible defect.
DEFAULT switched to perband (pipeline+CLI; --fast now maps perband→decision).

## F22 — PER-BAND edge-aware fusion = best-of-both at ALL resolutions (Farron's insight)
Farron's point: blend's pyramid is only in the RECONSTRUCTION; its DECISION is
single-scale (one guided weight broadcast to all bands), whereas pyramid decides
per-band (multi-scale). Fix: make the decision per-band too — at EACH Laplacian band,
decide from that band's energy AND refine with a guided filter (guided by that band's
Gaussian image). A FIXED small radius per band => effective full-res radius grows with
scale automatically (the pyramid "starts at finest pixels and moves up") — multi-scale
by construction, NO magic number.
Result (`fuse_perband`): BEST at high-res (0.9014 > pyramid 0.8926 > blend 0.8704) AND
low-res Real-MFF (0.9926 ≥ blend 0.9923 ≥ pyramid 0.9920); nearly halo-free on the
Lytro fence (composite 0.9091, between blend 0.9143 and pyramid 0.9060; visually mostly
clean vs pyramid's clear halo). So it inherits pyramid's multi-scale strength AND
blend's halo-freeness — dominates pyramid everywhere, beats/ties blend except the
hardest low-res halo case (fence, -0.005 vs blend). Promoted as `--method perband`
(recommended for high-res). Kept `blend` default (still the fence halo-champion).
Lesson: give the DECISION the multi-scale structure, not just the reconstruction.

## F21 — Course-correction: PYRAMID is the best realization of the local-scale principle
Adding pyramid to the fine-structure comparison (with TRUE GT) flips my earlier
visual read: on the fine near structures pyramid is MOST faithful (near-SSIM
0.71–0.77) vs my local (0.55) and global (0.50); pyramid also wins aggregate
(0.8926). Content-routing (local at fine boundaries, pyramid elsewhere) gained only
+0.001 overall and HURT the structures (I routed toward the worse option there per GT).
Resolution crossover: low-res clean Real-MFF blend≈pyramid (0.9923≈0.9922); low-res
HARD boundaries (Lytro fence) blend>pyramid (halo); high-res pyramid>blend (+0.022).
Two lessons (added to PLAYBOOK):
  (a) **With true GT, GT-referenced fidelity is the verdict** — a "cleaner-looking"
      result (my local) can be LESS faithful than a sharper one with a faint halo
      (pyramid). The "look, don't trust the metric" rule targets NO-REFERENCE metrics
      + aggregates hiding LOCAL defects; it does NOT override GT-referenced fidelity.
  (b) **The best realization of a principle may be an existing algorithm, not your
      bespoke retrofit.** Farron's local/multi-scale thesis is CORRECT — and pyramid
      (intrinsically multi-scale) embodies it better than my explicit content-measured
      local-scale guided-blend. Don't fall in love with your own mechanism.
Decision: keep `blend` the default (halo-safe all-rounder; high-res data is synthetic-
only and pyramid halos on hard boundaries). Method choice is resolution/content-
dependent — a selector (blend low-res / pyramid high-res) is the principled next step,
pending REAL high-res optical-defocus data to confirm pyramid's high-res win isn't a
synthetic-benchmark artifact. analyze.py/local_fuse.py retained as validated research.

## F20 — CONTENT-MEASURED LOCAL SCALE beats the global magic number (Farron's thesis, confirmed)
The F19 global scale (0.012·max_dim) only helps object-scale depth splits; it DESTROYS
fine details at FINE-SCALE depth boundaries (thin near structures over a far
background) because the window is coarser than the boundary. Built the right benchmark
(hires_mixed: thin near wires/dots over real far photos, GT) and a content-analysis
stage (analyze.py): Canny edges + a per-pixel LOCAL STRUCTURE-SCALE map measured from
fine-detail energy (small on detail: 17–26px, large on smooth: 31–34px — MEASURED, not
assumed). local_fuse.py sets the guided scale per-pixel from that map (interp over K
discrete scales + multiband).
Result on fine near structures: LOCAL >> global (+0.04 to +0.14 SSIM; e.g. 08:
0.070→0.206, 09: 0.248→0.363) and **visually best — sharp AND clean**. Global blurs
them; the thesis is confirmed: measure local scale, no magic number.
Nuance (honest, and the recurring lesson): **pyramid wins AGGREGATE** (0.8926 vs local
0.8787 vs global 0.8704) because it is intrinsically multi-scale — it already embodies
the local-scale principle — BUT it HALOS on the fine high-contrast structures (visible
in the crop, hidden by the mean). So LOCAL is visually cleanest on structures; pyramid
is strongest on detailed backgrounds. Best-of-both = CONTENT-ROUTE between local-guided
(clean fine boundaries) and pyramid (multi-scale detail) — the L4 next step. Lesson to
PLAYBOOK: fusion scale must be MEASURED locally from structure; and pyramid's aggregate
again hid a boundary halo — look at the structures, not the mean.

## F19 — FIX: resolution-adaptive guided params — blend beats pyramid at high-res, no low-res regression
Root cause of F18 confirmed by experiment: scaling the guided radius + focus pool
with resolution monotonically improves high-res blend (default 0.788 → 0.80). Fix:
`radius`/`smooth_ksize` default to None → auto = `max(8/9, round(0.012·max_dim))`
(≈ the CoC). Results: **byte-identical to the old fixed 8/9 at low-res** (40/40
Real-MFF pairs — the floor guarantees no regression) and **default blend now 0.7998
> pyramid 0.7933** at 3072px, visually recovering the soft foliage detail. Promoted
(it's the package default now). Nuance: pyramid still edges it on a few extreme
large-smooth-defocus stacks (05 foliage, 08 texture) — a future content-routing
opportunity, not chased now. Lesson (added to PLAYBOOK): **every fixed-pixel-size
operator must scale with resolution/CoC**; a param tuned at one resolution silently
mismatches another. (Metric analogue F17 — scale-aware Q_ABF — noted as follow-up.)

## F18 — HIGH-RES METHOD REVERSAL: pyramid beats guided-blend at high-res (root: fixed-scale params)
On 10 high-res (3072px) GT stacks (real Wikimedia photos + depth-dependent disk
defocus + chromatic aberration + noise, CoC~37px), `pyramid` WINS overall
(GT-SSIM 0.7933, worst-tile 0.4734) vs the guided-blend family (~0.788, worst ~0.44)
— a REVERSAL of the low-res finding (where blend/decision beat pyramid's halos).
Visually confirmed on foliage: blend is softer (fine detail smeared), pyramid keeps
detail. **Root cause:** the guided-blend's FIXED-pixel params (guided radius 8, focus
pool 9) are tiny vs a 37px CoC, so the weight is mismatched to large-CoC structure
and blends in blur; pyramid is inherently multi-scale so it adapts. Lesson: the
low-res-optimal default is NOT high-res-optimal; fixed-pixel operators must SCALE
with resolution/CoC. (Testing scale-aware params next.) This is exactly the
"high-res is where it differs" the user predicted.

## F17 — The metric ALSO doesn't transfer to high-res (q_abf collapses)
Re-validating per-stack Spearman vs GT-SSIM at 3072px: q_abf +0.12 (vs ~+0.30
low-res — its fixed 3×3 Sobel only sees the finest scale), q_ssim +0.874 (STRONGER
than low-res), composite (0.3 q_abf+0.7 q_ssim) +0.714 — now DRAGGED DOWN by q_abf.
**q_ssim alone beats the composite at high-res.** Same root cause as F18 (fixed-scale
operators). Fix: scale-aware/multi-scale Q_ABF, or drop its weight at high-res.
Reinforces PLAYBOOK: don't assume the metric transfers; re-validate at each regime.

## F16 — Real-data validation: MFFW blocked; validated on real Lytro optical defocus
MFFW (real + hard defocus-spread) has no accessible download (ResearchGate page
only) — a wall, not a blocker. But Lytro (in standard/) is REAL light-field optical
defocus. On 20 Lytro pairs, `--fast` (decision + weight_scale 0.5) composite 0.9275
vs full blend 0.9285 (-0.001) and is VISUALLY INDISTINGUISHABLE (fence crop). So the
engine + speed path hold on real optical defocus. Overall validation coverage: real
content + GT (Real-MFF 710), real optical defocus (Lytro 20), synthetic + hard + GT
(benchmark). The remaining gap is real+hard-together (MFFW/UHD) — download-blocked.

## F15 — High-res CPU speedup: quality-safe ceiling ~1.5x (profiled, not assumed)
Profiled fuse_blend at 1K/2K/4K: weight pipeline (focus energies + guided filter)
~55%, multiband blend+pyramids ~42%, weight pyramids ~0%. Since weights are smooth,
subscaling helps — BUT naive full-subscale (weight_scale on everything) greyed thin
bright structures (defeats harden: -0.003 GT-SSIM on defocus_spread, visibly grey
wires) even though Real-MFF said -0.0001 (clean-data mirage again — LOOK). Fix:
subscale ONLY the guided-filter smoothing; keep focus/confidence/decision full-res
so harden still hard-selects thin structures. Result (quality-safe):
- `blend`, weight_scale 0.5: ~1.3x, Real-MFF -0.0002.
- `decision` (image-space, skips pyramids): with harden it BEATS blend on structural
  scenes (spread 0.9923 vs 0.9894) and ties on clean; ~1.2x.
- **`--fast` = decision + weight_scale 0.5: ~1.5x, Real-MFF -0.0003, better on hard.**
The dramatic 2-3x needs the quality-sacrificing full-subscale or a GPU. On CPU the
opencv engine is already fast, so ~1.5x is the honest quality-safe ceiling. Exposed
`--weight-scale` + `--fast`; weight_scale=1.0 is byte-identical (test-asserted).

## F14 — Distillation MATCHES classical quality in one pass; speed is a GPU story
Distilling the classical engine (content_aware+harden) into the same tiny FCN:
distilled CNN 0.9885 held-out GT-SSIM vs classical 0.9888 — a quality MATCH in one
forward pass (the engine IS distillable). But on CPU the CNN is 74.8 ms/img vs the
classical 34.9 ms/img (0.5x — SLOWER): the hand-optimized numpy/opencv engine
(integral-image guided filter, pyramids) is very fast on CPU, so the CNN's speed
win is a GPU/high-res-batch story this box can't show. Honest end-state: learned
fusion can reproduce the engine's quality; the "faster" half of "faster + excellent"
needs a GPU. Fulfills the staged plan's M4 and its version-bridge intent.

## F13 — M4 version-bridge works; self-supervised CNN < classical (as expected)
Provisioned py3.12 + CPU torch 2.13 via uv (torch has no 3.14 wheel) — the version
bridge works. A tiny FCN trained PURELY self-supervised (gradient-retention +
smooth-weight loss, no GT) reaches 0.9711 held-out Real-MFF GT-SSIM vs classical
0.9888. So from-scratch self-supervised learning is feasible but does NOT beat the
mature classical engine (guided edge-aware weights + multiband + content-routing +
hardening) — expected for a small CPU net vs a domain-knowledge-rich pipeline. The
route to "fast AND excellent" is DISTILLATION (train the CNN to reproduce the
classical engine in one forward pass) — testing next. Confirms the staged plan: AI
-learning rests on, and currently trails, the conceptual/algorithmic foundation.

## F12 — The GLOBAL composite must NOT be used for per-tile/region decisions
30-scene per-tile study (mean per-tile Spearman vs GT-SSIM over 6 candidate tunes):
q_abf ANTI-correlates per-tile (-0.241; -0.388 smooth), so the global composite
(0.3 q_abf + 0.7 q_ssim, great for GLOBAL ranking) is actively BAD per-tile
(-0.239; -0.642 smooth). Plain q_ssim is the best simple per-tile metric (+0.217);
a content-ROUTED metric (lowfreq on smooth, q_ssim+gradcons on textured) is best
(+0.278) — the metric-level analogue of the content_aware operator. But even the
best per-tile no-ref metric is MODEST: per-tile no-GT discrimination is inherently
hard. Consequences: (1) retroactively validates M3 (train on GT dev-labels, deploy
feature-only — don't rely on a no-ref metric to label per-tile); (2) per-region
SELECTION should use q_ssim, never the q_abf-laden global composite; (3) the "no
answer key even for per-tile training" ideal is only partly reachable now.

## F11 — Recommended engine (content_aware + harden) is non-regressing and scales
Real-MFF (200 clean pairs): content_aware+harden0.5 = 0.9914 vs default 0.9913
(neutral, worse on only 17/200) -> promoted content_aware to the DEFAULT operator;
harden stays opt-in (recommend 0.5 for bright/thin/spread scenes). High-res (2048)
hard scenes: recommended beats baseline on both (defocus_spread 0.9914 vs 0.9907;
gradient_metal 0.9615 vs 0.9582), on global AND worst-tile, and visibly restores
bright wires baseline greys out. Structural wins scale with resolution.

## F10 — Confidence-hardening UNIFIES spread-rejection AND thin-structure ("hair") preservation
The same `harden` mechanism that rejects defocus spread also preserves thin 1px
hairs: both are "hard-select where one frame is confidently the unique sharp
source." On a 1px-hair scene: baseline 0.9847 -> harden0.8 0.9874, and visually the
hairs go from faint/greyed (baseline blends toward the frame where the hair is
absent) to dark+crisp. So NO separate hair-isolation layer is needed — one
theory-grounded mechanism covers both structural failure modes. (adaptive 0.9887
still edges global SSIM but does NOT preserve the hairs; the real best is adaptive
WITH harden.) Don't-over-engineer corollary to F9.

## F9 — Defocus-spread rejection (confidence-hardening): orthogonal, stacks, VISIBLE
Where one frame is confidently sharpest (thin/bright structures), push the guided
weight back toward hard one-hot selection so the other frame's defocus SPREAD (dim
wide blob) can't bleed in; keep soft blending where ambiguous. On a harsh-spread
scene (bright near bars/dots, CoC~26px): baseline 0.9853, adaptive 0.9880,
spread_reject 0.9870, **adaptive+spread_reject 0.9888** (best). Crucially VISUAL:
baseline AND adaptive both render bright dots/bars GRAY (spread bleed); hardening
restores them bright+crisp — a big visual difference the metric barely registers
(+0.001), exactly the "metric-blind, eye-obvious" case. It's ORTHOGONAL to operator
routing and STACKS with it. Promoted as `harden` (0..1, default 0 = off) on
fuse_blend/fuse_decision + --harden CLI flag. Don't-throw-baby-out: kept adaptive,
added the layer, combined.

## F8 — M3 learned per-tile routing beats best-single-tune, MATCHES the oracle
Numpy MLP (8 content features -> best of 6 tunes), trained on GT-supervised dev
labels, deployed feature-only (no GT at inference). Held-out hard collage scenes:
learned routing 0.8580 > best-single-tune 0.8559 > content_aware 0.8554, and it
EQUALS the per-tile oracle upper bound (0.8580). Tile-accuracy is only 54% but
tunes are near-equivalent where confused, so fusion quality saturates near oracle.
Read: per-tile operator/param routing works and is near-optimal, but its CEILING
is small (+0.002 here). Bigger headroom is structural (defocus-spread rejection,
thin-structure isolation), not operator choice -> next focus.

## F7 — Content-aware operator routing = non-regressing best-of-both (promoted)
`content_aware` focus op (blend laplacian/mod_laplacian per pixel by local contrast,
c=1-exp(-ref/tau)): matches laplacian exactly on Real-MFF (0.9913, no regression) and
on defocus_spread, improves smooth-gradient metal (0.9621 vs lap 0.9586; mod 0.9675 is
the ceiling). Hand-tuned tau is a floor for M3 to beat. Committed to the package.

## F6 — Per-tile tune diversity is real; no-GT per-tile labeling is NOT yet reliable
Per-tile best-tune over collage scenes: all 6 candidate tunes win some tiles (GT dist
~[34,22,40,16,25,55]) — strong justification for learned per-region routing. BUT the
composite metric's per-tile best-tune agrees with GT's only ~15% of the time. So the
no-reference metric is fine for GLOBAL ranking (F1) yet too weak to LABEL per-tile tune
choice without GT. Implication: train M3 on GT-supervised dev labels, deploy feature-only
(no GT at inference); closing the no-GT-labeling gap needs metric refinement (future).

## F5 — Composite metric is BLIND (even backwards) on smooth low-texture content
On the `gradient_metal` hard scene (smooth brushed surface, gradual focus), GT-SSIM
ranks **mod_laplacian best (0.9675)** vs laplacian/baseline 0.9586 — a real, distributed
win (14% lower mean error, closer to GT on 57% of pixels). But the **composite metric
ranks baseline ABOVE mod_laplacian** (0.7637 vs 0.7507) — backwards. Cause: q_ssim/q_abf
need structure/edges to grade; on smooth surfaces they can't see the difference.
Consequences: (a) operator selection must be **content-aware**, not purely metric-driven,
for smooth regions; (b) the metric needs a smooth-content term or a fallback. Visual
check: the win is perceptually subtle here (both look near-GT), so this is more a
metric-trust lesson than a big visual gain — but it's exactly the "don't trust the
metric alone" case.

## F4 — Scene-dependence is REAL on hard (disk-defocus) data
Hard benchmark (disk/bokeh PSF, defocus spread, noise, 1024px):
- `gradient_metal`: mod_laplacian > adaptive > baseline (see F5).
- `defocus_spread`: **adaptive best** (GT-SSIM 0.9904 vs 0.9883 baseline), and the
  advantage is larger on the **worst local tile** (0.9719 vs 0.9616, +0.010) — adaptive
  helps most exactly in the hardest region. Local (tiled) metrics reveal wins the global
  mean shrinks.
On clean-Gaussian data (Real-MFF + synth archetypes) operators nearly tie — scene-
dependence only manifests under realistic defocus.

## F3 — Real-MFF is near-ceiling; global tuning has tiny headroom
M1 global search: composite +0.0002, held-out GT-SSIM +0.0003 over baseline (positive,
transfers — objective validated end-to-end, but tiny). laplacian dominates 198/200 scenes.

## F2 — Region-adaptive > global > baseline on GT (monotonic, small)
Held-out Real-MFF: adaptive 0.9897 > global 0.9893 > baseline 0.9889. Premise validated;
magnitude limited by near-ceiling data.

## F1 — Composite objective calibrated & validated (M0)
0.3·Q_ABF + 0.7·Q_SSIM; mean per-pair Spearman vs true GT-SSIM = +0.72. Q_MI was
**rejected** (anti-correlated, +rewards ghosting/speckle). Sharpness alone unreliable.

## Open threads / next
- HIGH-RES: re-run hard benchmark at 2K/4K — do NOT assume 1024 conclusions transfer;
  defocus-spread & thin structures scale with resolution and may reward complex algorithms.
- Content-aware operator selection (smooth -> mod_laplacian) independent of the blind metric.
- Thin-structure ("hair/wire") isolation layer so outliers don't break region partitioning.
- Defocus-spread rejection (consistency check vs sharpest source).
- Metric refinement for smooth/low-texture regions.
