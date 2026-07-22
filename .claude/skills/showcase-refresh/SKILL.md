---
name: showcase-refresh
description: Incrementally update the showcase (figures + SHOWCASE.md + SHOWCASE.html) to reflect capabilities shipped since the last refresh. Never regenerates from scratch.
---

# Showcase refresh (incremental)

Goal: bring the showcase up to date with what shipped SINCE THE LAST ITERATION —
extend, don't redo. Suitable for delegation to a subagent (brief it per DEVSTYLE §10).

1. **Diff against last iteration**: `git log --oneline -- docs/SHOWCASE.md docs/img
   research/showcase_template.html` to find the last refresh commit; read FINDINGS.md
   entries newer than it (headlines + the SYNTHESIS section). List capabilities that
   are NEW/changed since then. If nothing material changed, say so and STOP.
2. **Figures**: keep all existing figures. For each new capability, generate figures
   via `research/make_showcase_specialists.py` (extend it — it is the re-runnable
   generator; add functions, don't fork a new script). Prefer scenes where SHIPPED
   gates/features actually fire (check via focusstack.gates). Output to docs/img/.
   NON-NEGOTIABLE: Read each image and verify the caption's claim is visible in the
   pixels before writing it. Disclose oracle inputs in captions when used.
3. **SHOWCASE.md**: insert/extend the relevant section in the document's voice;
   update stale counts (findings N), roadmap cards (done items move or get struck),
   evidence table rows if a regime was added.
4. **SHOWCASE.html**: edit `research/showcase_template.html` (NOT the generated file)
   — keep section kickers numbered contiguously; then rebuild with
   `.venv/bin/python research/make_showcase_html.py` and verify the new content +
   equation counts in the output.
5. **Ship**: pytest green; commit (no author trailers), pull --rebase, push.
