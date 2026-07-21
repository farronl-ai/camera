# Frontier — living inventory of unexplored directions

The antidote to closing in after a convergence: every direction we have NOT yet
explored, with status. Maintained like FINDINGS.md — update when a direction is
probed, promoted, or retired. If this file hasn't changed in a while, that itself
is a warning sign.

Status: ⬜ unexplored · 🔶 probing · ✅ resolved/promoted · ❌ blocked (documented)

| # | Direction | Why it matters | Status |
|---|-----------|----------------|--------|
| 1 | **N-frame stacks (N>2)** | Nearly all validation was 2-frame; real stacking = 5–50 frames. | ✅ F24: NOT a defect — quality RISES with N (denoising); "dilution" is beneficial, top-K hurts; perband best ∀N. Sub-case 1b still open. |
| 1b | **Extreme defocus-spread across a deep stack** | Bright point sources at depth. | ✅ F29: no defect — N=8 matches N=2 under maximal spread stress; harden persists (mildly attenuated). |
| 2 | **Real N-frame optical data** (BBBC006 microscopy z-stacks) | Real optical defocus. | ✅ F25: got 4 real 3-plane stacks; perband visibly sharpest, pyramid softens/halos nuclei. Microscopy domain covered; macro/photographic real still open (2b). |
| 2b | **Real photographic/macro deep stacks** | Microscopy ≠ everyday photography content; the photographic real-optical gap persists (MFFW/UHD blocked). | ⬜ re-probe |
| 3 | **Occlusion-boundary physics (α-matte)** | Real depth edges mix fg/bg semi-transparently. | ✅ F25: built layered α-matte defocus generator; re-ranked — perband crown WIDENS (blend worst under honest occlusion). Synthetic conclusions not artifacts. Fusion itself is still occlusion-UNAWARE (3b). |
| 3b | **Occlusion-AWARE fusion** | Matte inversion (de-veiling). | ❌ F27: rigorous negative — even oracle+noiseless+exact-PSF loses to perband (thin-structure haze is small; fringe error is decision-boundary, not haze; division amplifies quantization). Revisit only for LARGE occluders + float pipeline + regularized unmixing. |
| 4 | **Per-band Q_ABF metric** | Fix Q_ABF's high-res collapse structurally. | ✅ F26: q_abf_ms (mean-pool) recovers +0.11→+0.78 at high-res; new composite best at BOTH regimes (+0.785/+0.869). Adopted. |
| 5 | **Depth map byproduct** | Free feature from the fusion decision. | ✅ F26: --depth-out shipped; r=0.59 on textured pixels (texture-only observability — documented limitation). |
| 6 | **Exposure/WB drift between frames** | Real capture drifts brightness/color. | ✅ F28: drift costs −0.025 SSIM; per-frame gain to stack-median means (mean is blur-invariant) recovers to −0.002; near-identity gate passed → default-ON (--no-normalize-exposure). |
| 7 | **Alignment robustness on real handheld deep stacks** | ECC is a local optimizer aligned to the middle frame; feature-based init / chained alignment for large displacement. | ⬜ |
| 8 | **Focus breathing across deep stacks** | Scale change accumulates over 10s of frames; interacts with #7. | ⬜ |
| 9 | **Noise-adaptive fusion** | Low-light stacks: focus energy vs noise energy confusion; denoise-aware weighting. | ⬜ |
| 10 | **Learned/semantic segmentation layer** | Object-aware regions beyond structural watershed (torch env exists; CPU-heavy). | ⬜ deferred |
| 11 | **GPU distillation speed** | Distilled CNN matches quality (F14); the speed win needs hardware absent here. | ❌ no GPU |
| 12 | **Learned no-reference metric** | Train a per-tile quality predictor on GT dev-labels to close the 19% no-GT labeling gap (F6) — the metric-side analogue of M3. | ⬜ |
| 13 | **Real camera capture protocol** | Farron shooting an actual bracketed stack (phone/camera + rail) would give first-party real data end-to-end. | ⬜ needs user |
| 14 | **Video / temporal focus stacking** | Fuse focus sweeps from video; temporal coherence constraints. | ⬜ |
| 15 | **UHD-MFF & MFFW datasets** | Real+hard benchmarks; not publicly downloadable at last check. Re-probe occasionally. | ❌ re-check later |

| 16 | **Boundary Engine (E-phase)** | True object boundaries + near-side occlusion tags from stack evidence ∪ learned appearance; additive integration into guided/perband/harden. THE current push — plan: NEXT_STEPS_boundary.md. Relates to 3b/10/12. | 🔶 E4 done as rigorous negative (F33): decision-side integration cannot fix boundary error — it is coefficient-contamination (reconstruction physics). Boundary engine (fused F=0.55) stands as data product. Successor lever: matte-aware / supersampled boundary RECONSTRUCTION (16b). |
| 16b | **Matte-aware boundary reconstruction** | The real hard-lines lever per F33: render boundary pixels from layer model (near-owned coefficients + estimated matte), not blended bands. | ⬜ |

Near-term execution order: 1 → 2 (parallel) → 3 → 4 → 5 (plan: NEXT_STEPS_breadth.md).
