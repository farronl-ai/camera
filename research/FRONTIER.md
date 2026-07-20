# Frontier — living inventory of unexplored directions

The antidote to closing in after a convergence: every direction we have NOT yet
explored, with status. Maintained like FINDINGS.md — update when a direction is
probed, promoted, or retired. If this file hasn't changed in a while, that itself
is a warning sign.

Status: ⬜ unexplored · 🔶 probing · ✅ resolved/promoted · ❌ blocked (documented)

| # | Direction | Why it matters | Status |
|---|-----------|----------------|--------|
| 1 | **N-frame stacks (N>2)** | Nearly all validation was 2-frame; real stacking = 5–50 frames. | ✅ F24: NOT a defect — quality RISES with N (denoising); "dilution" is beneficial, top-K hurts; perband best ∀N. Sub-case 1b still open. |
| 1b | **Extreme defocus-spread across a deep stack** | Bright point sources: distant-frame leakage could import bokeh spread (F24 scenes didn't stress it). harden's domain. | ⬜ |
| 2 | **Real N-frame optical data** (BBBC006 microscopy z-stacks) | Real optical defocus. | ✅ F25: got 4 real 3-plane stacks; perband visibly sharpest, pyramid softens/halos nuclei. Microscopy domain covered; macro/photographic real still open (2b). |
| 2b | **Real photographic/macro deep stacks** | Microscopy ≠ everyday photography content; the photographic real-optical gap persists (MFFW/UHD blocked). | ⬜ re-probe |
| 3 | **Occlusion-boundary physics (α-matte)** | Real depth edges mix fg/bg semi-transparently. | ✅ F25: built layered α-matte defocus generator; re-ranked — perband crown WIDENS (blend worst under honest occlusion). Synthetic conclusions not artifacts. Fusion itself is still occlusion-UNAWARE (3b). |
| 3b | **Occlusion-AWARE fusion** (not just generator) | The engine doesn't model matting when fusing; a matte-aware weight could beat perband at veiled edges. | ⬜ |
| 4 | **Per-band Q_ABF metric** | Transplant the perband lesson into the metric: fixes Q_ABF's high-res collapse (F17) structurally. Then re-calibrate composite per regime. | 🔶 B4 |
| 5 | **Depth map byproduct** | N-frame fusion decision ≈ depth-from-focus; free feature, enables occlusion/object reasoning later. | 🔶 B5 |
| 6 | **Exposure/WB drift between frames** | Real capture drifts brightness/color across a stack; engine assumes constant. Needs per-frame gain/WB normalization before fusion. | ⬜ |
| 7 | **Alignment robustness on real handheld deep stacks** | ECC is a local optimizer aligned to the middle frame; feature-based init / chained alignment for large displacement. | ⬜ |
| 8 | **Focus breathing across deep stacks** | Scale change accumulates over 10s of frames; interacts with #7. | ⬜ |
| 9 | **Noise-adaptive fusion** | Low-light stacks: focus energy vs noise energy confusion; denoise-aware weighting. | ⬜ |
| 10 | **Learned/semantic segmentation layer** | Object-aware regions beyond structural watershed (torch env exists; CPU-heavy). | ⬜ deferred |
| 11 | **GPU distillation speed** | Distilled CNN matches quality (F14); the speed win needs hardware absent here. | ❌ no GPU |
| 12 | **Learned no-reference metric** | Train a per-tile quality predictor on GT dev-labels to close the 19% no-GT labeling gap (F6) — the metric-side analogue of M3. | ⬜ |
| 13 | **Real camera capture protocol** | Farron shooting an actual bracketed stack (phone/camera + rail) would give first-party real data end-to-end. | ⬜ needs user |
| 14 | **Video / temporal focus stacking** | Fuse focus sweeps from video; temporal coherence constraints. | ⬜ |
| 15 | **UHD-MFF & MFFW datasets** | Real+hard benchmarks; not publicly downloadable at last check. Re-probe occasionally. | ❌ re-check later |

Near-term execution order: 1 → 2 (parallel) → 3 → 4 → 5 (plan: NEXT_STEPS_breadth.md).
