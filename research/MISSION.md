# MISSION — scene recovery, not camera-mix (the framework, 2026-07-22)

**The goal: produce the image true to the PHYSICAL OBJECTS captured — not the
perfect mix of what the camera received.** Selection among frames is the floor;
model-based synthesis is the ceiling being built. Read this before DEVSTYLE.

## The doctrine: remnant-guided synthesis, never blind generation

The dividing line for every synthesis method admitted into this engine:

- **NOT admitted**: blind generation — "continue the image plausibly beyond what
  was observed" (diffusion-style inpainting/hallucination). Plausible ≠ true.
- **Admitted**: remnant-guided recovery — the degraded remnants of the true
  content (attenuated detail under a veil, quantized survivors of amplification,
  clean content adjacent to a contaminated band) are the ANCHOR; algorithms — up
  to ML/AI-learning level when fitting — recover what the physical object must
  look like there. The ceiling: the perfect natural result as if the corruption
  (veil, contamination, quantization) never existed.
- Every admitted method must be auditable — by the three instruments that survive
  the doctrine shift, NOT the ones that enforce the old one. The distinction is a
  direction of comparison:
    - OLD doctrine (barred from steering): `output ≈ sources` — no-ref
      source-similarity metrics (F45) and pseudo-GT (17a). Under these, scene
      recovery scores as damage.
    - Admitted: (1) **analytic-factory GT** — truth-to-SCENE, never contained the
      corruption, so removing it scores as improvement; (2) **observation-domain
      re-degradation** — `forward_model(output) ≈ sources`: the claimed scene,
      pushed back through the physics, must reproduce what the camera saw. It
      punishes deviation from anything the sources could have COME FROM, not
      deviation from the sources — hallucination fails it, exceeding every source
      passes it; (3) **the eye**, judging physical plausibility / GT, never
      source-resemblance.
  Caveat: re-degradation is NECESSARY, not sufficient — blur has a null space
  (several scenes explain the same observations), and inside it the audit cannot
  tell truth from invention. Hence the lattice: re-degradation everywhere,
  factory GT where it exists, the eye as arbiter beyond both — plus the
  remnant-anchoring requirement itself, which is what confines synthesis to the
  observable-rooted part of the null space.

## The escalation ladder (thoroughness-first)

1. **Simple/classical, exhaustively**: research the restoration literature at this
   level FIRST (regularized inversion, guided/joint filtering with clean-surround
   anchors, per-band SNR weighting, priors). Simple strategies get a thorough
   pass — factory-measured — before anything learned.
2. **ML where classical saturates**: small learned components (per our gate/routing
   precedent) trained on factory GT, only where step 1 measurably plateaus below
   the oracle ceiling.
3. **AI-learning where fitting**: full learned restoration, still remnant-anchored
   and audit-gated. Not feared, not default.

## The reinvestigation register — what the framework changes

| Prior conclusion / area | Status under the new goal |
|---|---|
| F27: de-veiling by inversion loses | **REOPENED** (FRONTIER 19) — conditional on 8-bit raw inversion without the restoration system; hybrid specified, viable at 8-bit |
| F33: selection cannot fix boundary contamination | **STRENGTHENED** — the theorem that motivates the whole framework |
| F45: no-ref metrics cannot audit synthesis | stands; the constructive path is the observation-domain audit (L1) |
| Metric composite (q_abf_ms+q_ssim) | **RESCOPED** — valid for the selection floor only; must never steer or veto synthesis work |
| Pseudo-GT datasets (iphone12) | **RESCOPED** (17a) — validate the floor + never-harm; structurally blind to synthesis gains |
| Eye discipline / eyetool | stands — already GT-anchored where GT exists; on real data the eye compares to physical plausibility, not to sources |
| F24: broad-weight "dilution" benign | stands (multi-frame denoising serves scene truth) |
| Diffusion/generative MFIF exclusion (lit-scan) | **REFINED** — excluded as blind generation; remnant-anchored learned restoration is admissible via the ladder |
| Regions where NO frame is sharp (stack gaps) | **NEWLY IN-SCOPE** (FRONTIER 20) — the best mix is still blurred there; scene recovery says: mild anchored deconvolution |
| M4 distillation / classical-first learning stance | stands — it IS the ladder, already practiced |
| Market differentiation story | RESCOPED upward — "recovers the scene behind the photographs" is the claim no competitor can make; update at next /market-refresh |

**Standing directive**: at every /checkpoint-close, sweep standing conclusions and
frontier statuses against this framework (extends PLAYBOOK "negatives are
conditional") — the framework reshapes almost everything, a little or a lot.
