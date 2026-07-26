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
      passes it; (3) **the eye** — scoped tightly: it is an ARTIFACT DETECTOR and EDIT
      LOCATOR (ringing, halos, banding, ghosting; where changes landed —
      amplified diffs read as locations, never severities), and a verifier of
      visible structure WHEN a GT panel is alongside. It is NEVER a truth
      certifier for synthesized detail: F21 measured its fidelity judgments
      failing against GT, and under this mission its worst mode is being TOO
      GENEROUS — plausible hallucination looks good; the eye pays out in the
      exact currency blind generation optimizes. In the null space, eye
      approval of fine detail is not evidence of truth.
  Caveat: re-degradation is NECESSARY, not sufficient — blur has a null space
  (several scenes explain the same observations), and inside it the audit cannot
  tell truth from invention. Hence the lattice: re-degradation everywhere,
  factory GT where it exists, the eye for artifacts/targeting beyond both (never
  for certifying synthesized detail) — plus the
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
| F27: de-veiling by inversion loses | **CONDITIONALLY REOPENED / RESEARCH, AUTO DISABLED** (F64) — F60 replaced the invalid V1 judge; F61/F62 established asymmetric rear visibility and front-first geometry. F63 split primary opaque, slender/all-veil, transmissive, and malformed formation regimes. On the reweighted S23 development cohort, F64's local owner-mask supermajority repairs a far-background leak and gives 7/7 all-metric/all-partition-positive primary fires with 65 exact refusals. The model remains runtime-disabled until its still-positive fine-band tail is causally removed and a genuinely post-freeze split passes. |
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

## Attestation
Any agent (main or sub) working under this framework closes its report with a
DOCTRINE line per DEVSTYLE §10: the single most binding rule here for the task
at hand, and how the work complied or why it deviated. Compose it BEFORE finalizing — if writing it reveals a compliance gap, go back
and close the gap first; the attestation is a mirror, not a signature. A missing
or generic line means this document was not truly considered — reviewers audit
accordingly.
