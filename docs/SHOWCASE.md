# focusstack — Progress Showcase

*A visual tour of the engine as it stands today: what it does, how it thinks, the
mathematics underneath, the evidence behind every claim — and where it goes next.
This note is a milestone, not a destination.*

---

## The problem, and the result

A lens can only focus at one distance. Anything nearer or farther is blurred by the
optics — the deeper the scene, the worse the compromise. **Focus stacking** takes
several photographs of the same scene focused at different depths and fuses them
into one image that is sharp everywhere.

Below: two real light-field captures — one focused on the chain-link fence, one on
the players behind it — and the engine's fusion. Both the fence *and* the gym are
sharp, with no halo around the wires:

![Two differently-focused sources and the fused all-in-focus result](img/hero_fence.jpg)
*Left: near focus (fence sharp, gym blurred). Middle: far focus (gym sharp, fence a
soft smear). Right: `focusstack` output — everything sharp.*

This is a genuinely hard case: the fence is a *thin, high-contrast structure over a
sharp background*, the exact geometry where classical fusion methods leave bright
halos or grey, ghosted wires.

---

## How it thinks

The engine's reasoning is visible. For the fence pair, left to right: where it
measures sharp detail, which frame wins each pixel, the cleaned-up weight it
actually uses, and the fused result:

![Pipeline anatomy: focus energy, winner map, guided weight, fusion](img/anatomy.jpg)
*Left: focus-energy map (bright = sharp detail present). Second: raw per-pixel
winner (blue = fence frame, red = background frame). Third: the guided weight for
the fence frame — the raw decision cleaned into a smooth, edge-snapped lattice.
Right: the fusion.*

That strip is the whole philosophy in one image: **measure sharpness, decide per
location, clean the decision so it follows real object edges, then reconstruct
without seams.** Everything below is the mathematics of doing each step right.

---

## The five ideas

### 1. Focus is high-frequency energy that survived

Defocus is a physical low-pass filter: a point at depth $z$ images to a disk (the
*circle of confusion*) with radius proportional to its distance from the focal
plane, $r \propto |z - z_f|$. Blur with a disk kernel erases fine detail — so
**sharpness is simply the high-frequency energy that survived**. We measure it with
second-derivative operators, pooled over a small window for noise robustness:

$$E(x,y) = \big|\nabla^2 I\big| * w \quad\text{(Laplacian energy, box-pooled)}$$

One measured subtlety: on smooth, low-contrast surfaces (polished metal, gradual
shading) the signed Laplacian $I_{xx} + I_{yy}$ can *cancel* — curvature in $x$
offsetting curvature in $y$ — hiding real focus. The **modified Laplacian**
$|I_{xx}| + |I_{yy}|$ can't cancel, and measurably wins there. The engine routes
between the two per pixel by local contrast (`content_aware`), so textured and
smooth regions each get the operator that suits them.

### 2. Decide, then clean the decision — with no edge detector

The raw per-pixel winner map is speckled (noise flips ties) and ragged at object
boundaries. We clean it with a **guided filter**: inside each local window, fit the
weight map $p$ as a linear function of the image $I$ itself:

$$q = aI + b, \qquad a = \frac{\mathrm{cov}(I, p)}{\mathrm{var}(I) + \epsilon}$$

Where the image is flat, $\mathrm{var}(I)$ is small, $a \to 0$, and the decision is
smoothed. Where the image has a real edge, $\mathrm{var}(I)$ is large, $a \to 1$,
and the decision *snaps to that edge*. **Edge-awareness emerges from local variance
— there is no explicit edge detector to tune, and the whole filter is a handful of
box filters (O(1) per pixel, any radius).**

### 3. Decide per scale — the centerpiece

The deepest lesson of this project: **a single decision scale cannot be right.** A
window tuned for 500-pixel images is far smaller than the blur disks in a 4K image;
scale it up globally and it destroys exactly the fine details high resolution
exists to capture. Measuring a local scale map helps, but there is a cleaner
answer.

Decompose each frame into a **Laplacian pyramid** — a stack of frequency bands:

$$L_i = G_i - \mathrm{up}(G_{i+1}), \qquad G_{i+1} = \mathrm{down}(G_i)$$

where each band holds exactly the detail the next blur-and-halve removed
(reconstruction is exact: collapse by $\mathrm{up}$-and-add). Now make the
**edge-aware decision of idea 2 independently at every band**, with one fixed
small window. Because band $i$ is downsampled $2^i\times$, that fixed window
*automatically* covers $2^i\times$ the area at full resolution:

> **A fixed small window per band *is* local scale — at every location,
> structurally, with zero magic numbers and no scale-estimation step.**

This method — `perband`, the engine's default — is the unique combination of four
properties, each of which we watched fail when absent:

| Property | Without it |
|---|---|
| Edge-aware decision (idea 2) | speckle and seams (`max`) |
| Confidence-hardened decision (idea 4) | greyed thin structures, defocus-spread bleed |
| **Multi-scale decision** (idea 3) | ranking reversals between low-res and high-res |
| Multi-band reconstruction | visible seams at region transitions |

Classical Laplacian-pyramid fusion has the multi-scale decision but no
edge-awareness — it **halos**. Guided multiband blending has edge-awareness but
one single-scale decision — it **softens fine detail at high resolution**. Look:

![Fence wires: pyramid halos, blend is clean, perband is clean and sharper](img/zoom_halo.jpg)
*3× zoom on the fence wires. Left — classical pyramid fusion: note the bright halo
hugging the wire. Middle — guided blend: clean but slightly soft. Right — `perband`:
clean **and** the sharpest wire edges.*

![High-res fine structures: GT, blend soft, perband close](img/zoom_hires.jpg)
*High-resolution benchmark with thin structures at a different depth than the
background (ground truth on the left). Middle — blend's single-scale decision
smears the fine lines wide and grey. Right — `perband` keeps them thin. Not yet
perfect — this is the hardest regime we test — but clearly closest to truth.*

### 4. Confidence hardening — one formula, two rescues

Soft blending has a failure mode: where one frame is *decisively* the sharp one (a
hair, a wire, a bright point), blending in the other frame's contribution imports
its defocus — thin structures fuse grey, and bright out-of-focus disks ("defocus
spread") bleed over sharp backgrounds. The fix is to measure decision confidence
from the top-two focus energies,

$$c = \frac{E_{(1)} - E_{(2)}}{E_{(1)}}, \qquad
w = (1 - c)\,w_{\text{soft}} + c\,\mathbb{1}_{\text{winner}}$$

and push the weight toward a hard selection exactly where confidence is high,
keeping soft blending where focus is genuinely ambiguous. One mechanism, two
failure modes cured:

![Bright structure: GT white, un-hardened grey, hardened near-white](img/harden.jpg)
*Zoom on a bright structure over noise texture. Left — ground truth: pure white.
Middle — fusion without hardening: the structure fuses dimmed and hazy. Right —
with hardening: restored to near-white.*

### 5. Physical invariants make it robust for free

Real capture wobbles: auto-exposure drifts brightness and white balance between
frames, which measurably breaks fusion (−0.025 SSIM in our drift benchmark). The
fix rests on a small theorem: **blur preserves the mean** — a defocused frame has
(essentially) the same channel means as a sharp one. So *any* difference in frame
means within a stack is exposure, never focus, and a per-frame scalar gain toward
the stack median removes drift without touching focus content. It is provably
near-identity on undrifted stacks, so it runs by default:

![Exposure drift: wobbling frames, and fusion without/with normalization](img/drift.jpg)
*Top: four frames of one stack with injected exposure/white-balance wobble (note
the brightness differences). Bottom: fused without (left) and with (right)
normalization. The visual difference is subtle at this size — the measured
difference is −0.023 SSIM — which is itself a lesson we keep: metrics catch what
the eye misses, and vice versa.*

And a byproduct falls out for free: in an $N$-frame stack, *which frame wins* each
pixel encodes depth — the winner index is a coarse **depth-from-focus** map
(`--depth-out`):

![Depth from focus: image and its depth map](img/depth.jpg)
*An 8-frame stack over a left-to-right depth ramp. The textured subject reads the
ramp cleanly (blue → orange); the featureless background is speckle — depth from
focus is only observable where there is texture, a documented physical limitation,
not a bug.*

---

## Real optics, not just synthetic

Everything above must survive contact with *real* defocus — actual lens physics,
not simulated disks. On real fluorescence-microscopy z-stacks (BBBC006, genuine
optical defocus, three focal planes):

![Real microscopy z-stack: defocused source vs fused](img/real_micro.jpg)

![Nuclei zoom: soft blobs become defined nuclei with internal texture](img/real_micro_zoom.jpg)
*3× zoom. Left: one source plane — nuclei are soft blobs. Right: the fusion —
boundaries tighten and internal chromatin speckle becomes visible. On this real
data the classical pyramid method visibly softens and halos; `perband` is the
sharpest, confirming the synthetic conclusions on genuine optics.*

## The specialist layer — routing and gates

The newest architectural layer (F43–F48). The generalist engine above is what every
stack gets. On top of it sit **specialists**: narrow mechanisms that fix physics the
generalist provably cannot (a lens mixes both sides of a depth boundary into the
captured pixels — no weighting scheme can unmix a contour, and no weighting can
subtract a wide occluder's veil). Two are live:

- **Contour reconstruction** (thin occluders — wires, stems, hairs): re-renders the
  contaminated boundary band as a fresh composite from a pixel-precise difference
  matte. Purely classical; no extra dependencies.
- **Veil correction** (wide occluders — a branch or finger near the lens): subtracts
  a forward-modeled haze field inside the per-band fusion, weight-scaled by how much
  hazy content each band actually admitted. Uses a semantic bridge (monocular depth
  + segmentation in a separate torch environment) when available.

Neither fires freely. Every specialist sits behind an **outcome-trained gate**: its
candidates are scored by a model trained on thousands of factory-generated scenes
where the ground truth is known, predicting *the actual quality change of firing* —
and the fire threshold is set so that, on held-out scenes, firing never lost. The
composed stage shipped as the pipeline default (`--enhance auto`) with this evidence:

| Check | Result |
|---|---|
| Held-out gate fires (recon) | 18/19 positive, mean +0.007 |
| Composed pass, 75 unseen scenes | wins to +0.020; worst case −0.0033 (2 outliers, documented) |
| Real photographs (14 cases through the shipped path) | 13 byte-identical; 1 fire — a fence wire, eye-verified benign |
| Bridge or gates unavailable | byte-identical to the base engine, by construction |

The design rule this arc proved: **a specialist must be paired with its regime and
its matte class, and its gate must be trained on every regime it will meet —
including the ones where the correct answer is "never fire."**

## The evidence, in one table

Every number is GT-referenced SSIM against a true all-in-focus reference unless
noted; each row is a different validation *regime*, because the project's core
discipline is that **a verdict in one regime is a hypothesis in every other**.

| Regime | Result | Honest caveat |
|---|---|---|
| Low-res, real content + GT (Real-MFF, 710 pairs) | perband **0.9918** > blend 0.9915 > pyramid 0.9913 | near-ceiling data; differences small but consistent |
| High-res (3072px) fine-depth benchmark | perband **0.9021** > pyramid 0.8926 > blend 0.8704 | synthetic (disk-PSF) defocus on real photos |
| Occlusion-honest (α-matte layered defocus) | perband **0.9434** > pyramid 0.9308 > blend 0.9283 | perband's lead *widens* under more honest physics |
| Real optical defocus (microscopy z-stacks) | perband sharpest — eye-confirmed | no GT exists; no-ref metric ordering confirmed visually |
| Deep stacks (N = 2 → 8) | quality **rises** with N for every method | broad soft weights act as multi-frame denoising |
| Exposure drift (±12% + WB tilt) | −0.025 SSIM broken → **−0.002** with default-on fix | clipped highlights slightly weaken the mean invariant |

The metric itself got the same treatment as the engine: the standard gradient-transfer
metric (Q<sup>AB/F</sup>) collapses at high resolution for the same fixed-window reason —
so it, too, was made multi-scale (per pyramid level, mean-pooled), and the resulting
composite is the best GT-predictor at **both** resolutions (+0.785 / +0.869 Spearman).

## Run it

```bash
focusstack shots/*.jpg -o sharp.png                    # perband default, drift-corrected
focusstack shots/*.jpg -o sharp.png --harden 0.5       # bright/thin-structure scenes
focusstack big/*.png  -o sharp.png --fast              # ~1.5x for high-res, quality-safe
focusstack shots/*.jpg -o sharp.png --depth-out z.png  # + depth-from-focus map
```

---

## This is only the beginning

The engine is the *foundation* of a leading-edge system, not the finished one. The
frontier is tracked live in [`research/FRONTIER.md`](../research/FRONTIER.md) —
probed items spawn new sub-frontiers, and several were **closed as rigorous
negatives** (occlusion de-veiling loses even with oracle knowledge; deep-stack
spread import doesn't materialize), which is the discipline working: settled
questions, not lingering doubts.

**Near-term, already scoped:**
- **Real photographic deep stacks** — real handheld sweeps are now in
  (`mobiledepth`, 13 sweeps: they exposed misalignment robustness as a new regime
  axis, F37); the iPhone-12 GT dataset is the pending promotion gate.
- **Gate recall growth** — the gates never harm; making them fire on more of their
  wins (more features, bigger factories, better models) is the path from "never
  harms" to "most images benefit somewhere."
- **A synthesis-aware no-reference metric** — today's no-ref metrics score any
  deviation from the sources as damage, so they cannot audit corrections that
  *improve on* every source; a metric that can would unlock runtime self-auditing.
- **Deep-stack alignment** — chained/feature-initialized registration and
  focus-breathing compensation for 10–50 frame handheld rails.
- **Noise-adaptive fusion** — low-light stacks where sensor noise mimics focus energy.
- **A learned no-reference metric** — close the remaining gap between no-GT scoring
  and truth for per-region decisions.

**Bigger swings, with groundwork already laid:**
- **One-pass neural fusion at classical quality** — distillation already *matches*
  the classical engine's quality (0.9885 vs 0.9888); the speed win awaits GPU
  hardware. The self-supervised training loss (no answer key) is designed and tested.
- **Object-aware fusion** — semantic segmentation as a routing layer above the
  structural one; the torch environment and per-region machinery already exist.
- ~~Occlusion-aware fusion~~ — superseded: shipped as the two gated specialists
  (contour reconstruction + veil correction) rather than global unmixing.
- **Temporal focus stacking** — fusing focus sweeps from video with temporal
  coherence.

The deeper record: [`research/FINDINGS.md`](../research/FINDINGS.md) (48 dated
findings + a living synthesis), [`research/PLAYBOOK.md`](../research/PLAYBOOK.md)
(domain knowledge and every trap we hit), and
[`research/DEVSTYLE.md`](../research/DEVSTYLE.md) (the working method that produced
all of it — hypothesis → measure → look → A/B → gate — so any future session starts
where this one leaves off).

*Every figure in this document was regenerated from the current engine by
`research/make_showcase.py` and visually verified before its caption was written.*
