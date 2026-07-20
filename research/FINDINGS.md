# Adaptive focus-stacking — running findings log

Persistent notes from the autonomous marathon. Newest first. Pairs metric numbers
with conceptual reasoning and visual inspection (metrics guide, don't decide).

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
