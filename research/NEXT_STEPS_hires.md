# Next steps — high-res data (obtain/generate) + expert-flow validation & refinement

Planned but not yet executed. Run with the methodology in `PLAYBOOK.md` and the
experimental log in `FINDINGS.md`.

## Context

The genuine remaining gap is **high-resolution** data. All current data is low-res:
Real-MFF (433×625, real content + GT, synthetic defocus), Lytro (520², real optical
defocus, no GT), synthetic hard benchmark (up to 4K but synthetic textures). Goal:
explore avenues to **obtain/generate high-res data**, then apply the **same expert
flow** (`PLAYBOOK.md`): metric + concept + eyes, validate where weakest, don't trust
clean/low-res verdicts.

**Chosen primary avenue (controllable, reliable, GT):** real high-res photos as
all-in-focus GT + **physically-realistic depth-dependent disk defocus** (CoC ∝
|depth−focus|, + chromatic aberration + sensor noise). This is exactly how Real-MFF
was constructed, but at a resolution and content-diversity we control. Source:
**Wikimedia Commons** freely-licensed featured/quality photos (verified: direct
8000px+ URLs, `iiurlwidth` gives scaled versions; stable, not GitHub-rate-limited).
**Secondary (best-effort):** a real optical focal stack / light-field set if reliably
downloadable; else documented and skipped (lesson: don't thrash on flaky downloads).

## Environment / constraints

Py3.14 numpy+opencv engine; `.venv312` torch; **no GPU**. Downloads via urllib/gdown.
Research/data gitignored (never commit third-party photos). High-res compute is heavy
→ background jobs. Disk defocus uses the fast large-radius approx already in hardbench.

## Reuse (all exist)

`hardbench.disk_blur / defocus_disk / add_noise` (`research/hardbench.py`); `metrics`
(`ref_ssim`, `ref_psnr`, `composite`, `q_abf/q_ssim`, per-tile maps) + the calibration
pattern in `validate_metrics.py` (spearman); `fuse_blend/fuse_decision` with
`harden`/`weight_scale`; `report.py` + `FINDINGS.md` + `PLAYBOOK.md`.

## H1 — high-res GT stack generator (`research/hires_gen.py`)

1. **Acquire** a diverse high-res photo set from Wikimedia Commons via its API
   (`generator=categorymembers` / `search`), pulling `iiurlwidth`-scaled versions at
   a target width (e.g. 2560/3072/4096). Cover the content types that matter for
   scene-dependence: fine multicolor detail (birds/feathers/foliage), smooth+specular
   (metal/machinery), hard edges (text/signs), organic texture. Record license + dims.
2. **Generate stacks with GT:** for each photo (= GT all-in-focus), synthesize a depth
   map (continuous gradient/radial AND a two-plane fg/bg mask variant), then render an
   N-frame stack with depth-dependent **disk** defocus + **chromatic aberration**
   (small per-channel CoC difference) + noise. Reuse `defocus_disk`/`add_noise`.
3. Register in `research/data/hires/` with a manifest (id, dims, content-type, depth-type).

## H2 — expert-flow validation at high-res (`research/hires_eval.py`)

1. **Re-validate the metric at high-res (do NOT assume it transfers).** Recompute the
   composite's rank-correlation vs GT-SSIM on high-res stacks; Q_ABF's fixed 3×3 Sobel
   is scale-sensitive, so check whether weights/scale need adjusting at 2K–4K.
2. **Engine comparison on high-res GT:** baseline vs `content_aware` vs `--harden 0.5`
   vs `--fast` (decision+weight_scale) vs `pyramid`/`max`. Report **GT-SSIM + PSNR +
   tiled-local worst-tile + timing**, per content-type.
3. **VISUAL failure-mode hunt (the core):** render worst-error crops at high-res and
   inspect — look for high-res-specific effects the metric may miss: larger-CoC spread,
   **chromatic fringing** at defocus boundaries, thin structures at high pixel counts,
   halo scale, `--fast` softening. Catalog new failure modes.

## H3 — refine (theory-backed), re-test, promote

Address the top high-res failure mode(s) found in H2 (e.g. chromatic-aberration-aware
fusion; scale-aware focus/metric; resolution-adaptive `weight_scale`). Each fix:
predict from theory → implement → verify GT-SSIM non-regression on low-res (Real-MFF)
AND improvement on high-res AND visual crops. Promote to the package only if
non-regressing everywhere. Update `PLAYBOOK.md`/`FINDINGS.md` with any new lesson.

## Verification

- `pytest` green for any package change; add tests for new params (identity at
  defaults; non-regression).
- H1: manifest + a montage (GT | a defocused frame | depth) inspected visually.
- H2: metric-vs-GT correlation table at high-res; engine speed/quality table by
  content-type; worst-crop montages read and diagnosed.
- H3: before/after crops + metrics on low-res and high-res; commit per milestone
  (no author trailer); rebuild + republish `research/report.html`.

## Deferred (documented)

- Real optical high-res focal stacks / light fields: best-effort only (download risk).
- GPU-dependent speed (distilled CNN): unchanged, still needs hardware absent here.
