# Active frontier

This file contains only work that can change the current method. Historical
frontier ledgers and completed scans are in Git history. Status:
`ACTIVE`, `NEXT`, `LATER`, or `BLOCKED`.

## Priority order

| Priority | Direction | Status | Concrete next evidence |
|---:|---|---|---|
| 1 | Fresh validation of F78 transmission-boundary inversion | NEXT | Freeze the current parameters; generate a new scene-disjoint family and require protected-region identity, physical-partition improvement, forward consistency, and no visible contour tail. |
| 2 | Explicit formation-state handoff | NEXT | Compute `V_front`, `T_rear`, PSF choice, residual, and detection floor once while both focal inputs are available; pass them to recovery without changing F78 output. |
| 3 | PSF/ISP family breadth | ACTIVE | Cross disk, box, and compound-lens PSFs; move formation to linear light; add field dependence, noise, demosaic/sharpen/quantization families. |
| 4 | Transparent/transmissive foreground | NEXT | Separate factory with saved foreground, background, opacity/extinction, coverage, and both clean observations. Never relax opaque complete-core ownership. |
| 5 | First-party real optical truth | BLOCKED on capture | Controlled macro/product bracket with removable occluder or known target, RAW if possible, aperture/focus metadata, and an occluder-free latent reference. |
| 6 | N-frame and multiple-occluder recovery | LATER | Ordered multilayer formation state, per-layer observability, identity fallback when ownership is ambiguous. |
| 7 | Alignment: two-frame follow-ups | ACTIVE | Arc complete through F112 (two-frame route shipped, beats shipped path on the factory, user-validated on kitchen). Remaining, ranked: replace `SURFACE_SIGMA`'s two-scene compromise with the physical per-pixel low-pass keyed on `\|frame − peak\|` (designed in `twoframe_NOTES.md`); pair-aware refusal preferring a present-but-defocused member (repairs large-motion, moves the route boundary); licence check before the discarded-composite render. |
| 7b | Scene-model second pass (analysis-by-synthesis) | NEXT | User concept, 2026-08-02 — see section below. Pass 1's outputs (per-frame motions, layer elections, occlusion ordering, same-surface provenance) initialize an inverse-rendering pass that assembles per-LAYER appearance from every frame after detransform and re-renders the composite from the scene model, certified by forward-render consistency against the RAW frames. First build: the certifier alone (§12 — instrument before mechanism). |
| 8 | Stack-gap recovery | LATER | Mild remnant-anchored deconvolution only where no frame is sharp; estimate scale without the known under-deconvolution bias. |
| 9 | Noise-adaptive focal evidence | LATER | Separate focus energy from sensor/ISP high-frequency energy, especially in low light. |

## Scene-model second pass (user concept, recorded 2026-08-02)

The current engine decomposes frames into regions and bands with orthogonal
extraction strategies and reconstructs piece by piece. Those pass-1 results are
precisely the initialization a second, physically-grounded pass needs: by then
we KNOW how each frame moved (lateral, forward/back, rotation), how focus
stepped through the scene, which pieces stand in front of and behind which
corners, and which observations survive detransform. The second pass inverts
the viewpoint: instead of warping frames to the reference and holding a
per-pixel contest, it assembles a SCENE — per-layer appearance built from every
frame's usable observations in the LAYER's own coordinates — and renders the
composite from the model once. Fusion disappears as a concept; frames stop
being the units, so seams between frames cannot exist. Refusal upgrades from
"fall back to the reference" to physical completion: content occluded in some
frames is visible in others, and the model may keep it from where it was seen.

Sketch: (1) scene model from pass-1 provenance — layers, per-layer per-frame
rigid motions, focal ladder, occlusion order; (2) per-layer appearance from
detransformed, visibility- and `same_surface`-gated observations; (3) composite
in depth order with formation-model mattes at boundaries (the
`OCCLUSION_FORMATION` contract already models how defocused occluder edges
mix); (4) certify by forward-rendering every source frame from the model
(transform + per-layer defocus) and comparing to the actual frame — iterate
where the residual exceeds noise, refuse where iteration cannot close.

The certifier is valuable before any reconstruction exists, and it evades
F81a's trap: it scores against the RAW frames in their own geometry, which
alignment never touches — the first no-reference-safe arbiter for real scenes.
Its KAT: on the routed kitchen output it must light up exactly the known
residuals (the F112 knob, the pale sliver) and nothing else; on the factory it
must reproduce the GT-SSIM ranking.

**THE OUTPUT CONTRACT (user directive, 2026-08-02, supersedes the composite-
rewrite framing B2 implemented).** Pass 2's deliverable is not a better
composite — it is the N input frames PERFECTLY ALIGNED into reference
geometry, each with honest GAPS. The scene model (layers, per-layer motions,
focal radii, occlusion ordering) parameterizes, per frame, a piecewise
transformation plus a per-pixel validity mask: a pixel is either OBSERVED by
that frame (brought over, one resample, defocused-as-it-was) or NOT OBSERVED
— occluded behind a nearer object, inside a defocused occluder's matte
(dilate silhouettes by the modeled defocus radius, F83), rotated out of view,
off-frame — in which case that frame carries a GAP there, and other frames
supply the pixel. The existing validated machinery (fuse_perband /
fuse_coherent with `usable` masks, the F80/F82 contracts, refusal, veils,
specialty reconstructions) then consumes aligned-frames-plus-gaps exactly as
it consumes `align_stack` output today. Under this contract INVENTION IS
STRUCTURALLY IMPOSSIBLE — every output pixel is a real observation from a
real frame or a fusion of frames that all observed it; there is no
cross-frame appearance-synthesis step for fabricated content to enter
through. (B2's per-layer appearance assembly WAS such a step, and every
artifact found in inspection leaked in through it — the F116 wall smear was
background assembled from frames where the wall was occluded.) What the
contract does not remove is MISPLACEMENT — a wrong transform still puts real
content in the wrong place — so the certifier (F113) and contour continuity
(F116) remain the alignment auditors.

**The viewpoint caveat (user, same directive).** Different angles see
different pieces behind corners, and the output CANNOT include everything: it
is committed to ONE viewpoint, the reference angle, whose depth ordering
defines per pixel WHICH surface is visible. A frame contributes at p iff it
observed THAT surface, uncontaminated. Content another angle saw around a
corner is real and is DISCARDED anyway — at the reference angle it has no
pixel; forcing it in is the pot-in-front-of-bottle class. This is F82's dual:
F82 refuses reference pixels a moved frame lacks; this withholds frame
content the reference viewpoint cannot legally receive. Both directions of
`usable` are computable from the scene model (ownership at reference; layer
silhouettes warped per frame, dilated by that frame's modeled defocus
radius). The irreducible residue is the reference's OWN defocused-edge matte
band — a few mixed pixels that stay trinary (matte or refuse) forever.

**DIAGNOSIS OF THE REFERENCE-COLLAPSE (user + manager, 2026-08-03 — read
this before building anything).** The user inspected all layers and is
right: every current output is essentially the reference plus a sliver of
sharpening (default focus energy 43.5 vs reference 42.7; aligner gaps 57.2%
of pixel-frames; box energies within 1–2% of reference). This is NOT the
cascade failing — it is every refusal gate working correctly on wrong
upstream geometry: content lands only where the model is provably right, so
crude geometry ⇒ copy of the reference. DO NOT loosen the gates; feed them
geometry that opens them on evidence. Two named root causes:
1. **No organ discovers objects on real scenes.** The merge only removes
   cuts; nothing proposes them from motion. The kitchen's objects stayed
   fused to the counter mega-piece, inherited its wrong affine, and the veto
   rightly gapped them in exactly the sharp (most-moved) frames. The proven
   organ EXISTS and is unwired: F93/F100 motion-consensus grouping finds the
   kitchen bottle at 92–100% feature purity with its ~19 px motion. Cuts
   must be SEEDED by pass-1's motion groups, refined by the merge.
2. **The piece-equivalence test checks the wrong invariant.** Depth
   dependence has TWO axes — lateral translation ⇒ per-depth SHIFT
   (t_xy/Z), forward translation ⇒ per-depth SCALE (t_z/Z) — and
   `motion_components` measured this scene at up to 4.3% forward
   translation. The 2-DoF (translation-only) decision merges
   different-depth objects that translate alike while scaling differently.
   The decision residual must be the PAIR (Δtranslation, Δscale), compared
   under fit uncertainty, not `GATE_TOL`.
Second-order, after objects exist: a ~10 cm-deep object at ~1 m carries
1–2 px internal parallax no affine removes — but F110–F112 achieved verified
sub-pixel fits on the correctly-delineated bottle, so affine-per-TRUE-object
suffices to beat the reference; the within-piece smooth 1/Z-linear
refinement is the extension, legal under the charter (smooth where the
surface is smooth). Also downstream, in order: per-piece photometric veto
with per-piece measured c (27% whole-frame withdrawal must become a
diagnosis), and the zero-motion identity snap.

**Division of labor (user, 2026-08-03 — the moral of the whole arc).**
Everything downstream of alignment already exists and is PROVEN: the focus
cascade (`fuse_perband`/`fuse_coherent`, harden, stack-consistency routing)
and the specialists (veil recovery, `--enhance`, the refusal net). Pass 2 is
an ALIGNER. Every time it tried to do the stack's job it re-derived fusion
machinery badly and paid full price to re-learn recorded lessons: B2's
"never average across a focus disagreement" is F79; its local veto is
`fuse_coherent`'s one-hot; its admission rules are `harden`. The controlled
demonstration is F117 vs F115: the same cross-convolution physics scored
0.9821 grafted into the bespoke second-pass composite (with artifact
whack-a-mole) and **0.9845 the moment it was handed to the existing cascade**
(zero correction rounds). The aligner aligns; the cascade fuses; the
specialists specialize. No future round builds fusion logic inside the
aligner.

**Integration expectation (user, same exchange): light-touch, because F82
already built the interface.** Fusion honours per-pixel `usable` today; the
aligner hands it heavier cargo, not a new capability. What changes is the
SCALE AND TOPOLOGY of absence — thin disocclusion ribbons become large
connected holes (a frame may miss 20–40% of the reference view) — so the
integration round RE-VERIFIES rather than redesigns: guided-filter windows
and per-band mask downsampling at gap boundaries (a gap edge is a cliff in
the weight domain — halo risk); the stack-consistency instability threshold
(calibrated with all frames present); exposure gains computed over
mutually-usable pixels. Structural gifts: the reference frame is gap-free at
its own viewpoint (zero-observers never happens; fallback always
terminates), and the zero-motion sentinel produces zero gaps, so the cascade
must be byte-identical there — the standing F101 anchor. The specialists
already gate to identity when inputs are absent.

**The transform, designed (2026-08-02, follows from the contract).** The
per-frame transform is NON-STANDARD by necessity: a backward sampling field,
PIECEWISE-SMOOTH — one affine per scene piece (a plane under small camera
motion induces an affine flow, so rigid objects and planar ramps are each one
piece; the counter's receding parallax is linear in image coordinates) —
DISCONTINUOUS exactly at occluding contours (near content jumps relative to
far; the F81 blend that smoothed this jump is the soft geometry F106
outlawed), with GAPS computed from the model (nearer pieces' silhouettes
warped by their own transforms, dilated by their defocus radius per frame,
F83) and vetoed by photometry (cross-convolution): geometry proposes,
photometry vetoes, nothing un-vetoes. CORRECTION TO B1: pieces must be cut at
DEPTH DISCONTINUITIES (occluding contours), never at depth values — ramps
stay whole or fake step-seams appear inside smooth surfaces. The two hard
sub-problems: (1) silhouette-exact registration — each frame's near-piece
transform must land its contour on the reference contour within the matte
band (F116's instrument becomes a per-frame alignment GATE, not an auditor);
(2) per-piece motion where the piece is defocused — the temporally-coherent
motion SERIES (B3b, now load-bearing): fit on sharp frames, interpolate
through blurred ones, pass-1 as prior. Known honest limits: transparency
(the glass pitcher) breaks one-surface-per-pixel and self-vetoes to
near-frame fallback; speculars are viewpoint-attached and gap per frame.
Application stays one resample (`remap` accepts a discontinuous field);
`fuse_perband` + `usable` consumes the output unchanged.

**Pass 1 is a GUIDE, not a constraint (user principle, 2026-08-02).** The
whole reason pass 2 exists is that pass 1 is imperfect — so no pass-1 quantity
(motion, masks, focal ladder, ownership) is gospel to the second pass. Each is
a prior with more than enough information to initialize the most likely
physical scenario; where forward-render consistency contradicts a pass-1
estimate, the second pass may revise it, with the certifier as the likelihood
and pass 1 as the prior. Concretely (post-F114): motion is the largest
remaining model term, and a temporally-coherent per-layer motion series
refined against render residual — initialized from, not clamped to, pass-1
fits — is the designated round B3.

Standing cautions that bind this pass: F81's joint estimator registered best
and INVENTED motion on the zero-motion sentinel — revision under the principle
above must therefore be routed (F101) and trinary (F106), engaging only where
pass-1 provenance leaves something unexplained, sentinels byte-identical by
construction, and every revision must beat the prior on held-out physics (the
raw frames), never merely on the quantity it optimized. And F56's licence
discipline applies to completion: the model may only place content some frame
actually observed, never synthesize it.

## F78 validation contract

Do not use `s29_010` as a tuning target again. It is a sentinel and visual
checkpoint. A promotion run must:

1. freeze implementation and thresholds before creating the split;
2. vary source content, ownership topology, foreground/rear contrast sign,
   aperture radius, PSF family, noise, and quantization;
3. report hard ownership, opaque core, soft edge, ordinary boundary, outer
   veil, and far background independently;
4. assert zero rear application in protected regions and exact identity where
   the gate refuses;
5. compare direct error, changed-closer/worse counts, false-texture diagnostics,
   and forward re-render residuals;
6. inspect disagreement-guided crops and formation inputs, not only outputs;
7. preserve negative and metric-dissent cases instead of tuning them away.

A large run is warranted only after compact causal probes and a small frozen
carpet pass. Waiting for jobs is cheaper than repeatedly loading huge outputs
into the session.

## Formation-state architecture

The desired narrow data flow is:

```text
original focal frames
        ↓
alignment / radiometric normalization
        ↓
owner geometry + local focal ordering
        ↓
formation state
  V_front, T_rear, PSF/model weights,
  forward residuals, detection floors
        ↓
safe base fusion + discrete front projection
        ↓
licensed rear inversion / censored completion
        ↓
contour-relative integration + identity fallback
```

This is not a request for another monolithic stage. The point is to keep
identifiable observation-domain quantities alive instead of recomputing them
from a fused image.

## Transparent foreground

Transparency is physically valid, but harder—not an excuse for malformed
opaque inputs. Across focus, foreground and rear layers transform differently,
which can make them separable. The first factory rung should use scalar,
spatially constant transmission with known disk PSFs and linear radiance:

```text
O_k = H_front,k(F) + T · H_rear,k(B) + noise
```

Then add spatial opacity, colored extinction, partial coverage, and ISP as
separate axes. Save every latent. Establish an oracle ceiling before blind
estimation. Evaluate foreground, rear, and re-rendering separately.

## Literature anchors that still affect the design

- Sheng et al., *Dr.Bokeh: Differentiable Occlusion-aware Bokeh Rendering*,
  CVPR 2024, arXiv:2308.08843. Separates on-focal occlusion from non-focal
  aperture visibility; supports distinct front veto and rear license.
- Liu, Narasimhan, and Dubrawski, *Matting and Depth Recovery of Thin
  Structures Using a Focal Stack*, CVPR 2017. Uses ordered attenuation:
  nearer layers attenuate rear layers, not symmetric pixel competition.
- Favaro and Soatto, *Seeing Beyond Occlusions*, CVPR 2003. A finite aperture
  can reveal rear information, but only on the support actually conveyed by
  the captured aperture.
- Marshall et al., *Occlusion edge blur: a cue to relative visual depth*,
  JOSA A 1996. Boundary sharpness provides ordinal front/back evidence that
  focus magnitude alone lacks.
- Lee, Kim, and Cho, *Realistic Compound-Lens Defocus Blur Synthesis*,
  arXiv:2607.05837 (2026). Motivates field-dependent compound PSFs,
  radiometrically linear formation, sensor noise, and ISP as distinct axes.
- Angelopoulos et al., *Conformal Risk Control*, ICLR 2024,
  arXiv:2208.02814. Relevant if learned gates return: calibrate explicit harm
  risk rather than optimizing average recall.
- Forward-model re-degradation from inverse-imaging hallucination assessment
  remains the right runtime-audit direction: compare a proposed latent scene
  only after pushing it back into each observation domain.

Before a new literature scan, read this list and the negative results in
`FINDINGS.md`; novelty is a mechanism that changes a named wall, not another
paper using a different network.

## Deliberately excluded

- Blind generative de-occlusion or diffusion fill: not remnant-auditable.
- Another global quality gate for a local ownership failure.
- Source-similarity as a recovery verdict.
- Helicon/Zerene-style pseudo-GT as latent truth.
- More tuning on legacy V1/S12/S23 cohorts.
- Reintroducing absolute circular seam filters.
- Treating disk PSF validation as a broad real-camera claim.

## Data opportunities

`REAL_DATA.md` tracks downloadable real stacks and capture gaps. The most useful
next data is not another easy two-frame blend benchmark. It is:

- real handheld sweeps with measured motion/breathing;
- controlled occlusion boundaries with known front/back order;
- RAW or minimally processed focal brackets;
- an occluder-free latent rear reference;
- transmissive material with independently captured layers; and
- compound-lens/off-axis PSF calibration.
