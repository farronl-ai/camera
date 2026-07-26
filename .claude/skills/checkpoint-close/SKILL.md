---
name: checkpoint-close
description: The end-of-checkpoint ritual — distribute lessons to the md files, refresh report + artifacts, update memory, audit git state. Run at every phase/arc close.
---

# Checkpoint close (the ritual)

Run when a phase/arc/half-marathon closes, or at end-of-day on request. Steps are
idempotent — each updates from current state, never regenerates.

1. **Distribute the gold** (main session, not delegated): scan the checkpoint's
   findings and place each lesson where it lives — FINDINGS SYNTHESIS (theory
   changes), PLAYBOOK (transferable methods + traps), DEVSTYLE (working-style
   lessons), FRONTIER (statuses + newly-spawned directions; probing must SPAWN).
2. **Refresh outward faces** as warranted: /showcase-refresh (if capabilities
   shipped), regenerate the current inspector, and /market-refresh if our
   capability cells changed materially.
3. **Memory**: update project memory state snapshot; add durable lessons to the
   playbook memory. Convert relative dates to absolute.
4. **Drift sweep (F49/F50-class)** — the shipped product must match the evidence:
   - DEFAULTS: any config value that appeared in every experiment's invocation this
     checkpoint (a de facto part of the method) must BE the default — check CLI,
     pipeline, AND library signatures (F49: pipeline fixed, library still stale).
   - EVIDENCE CLAIMS in user-facing text: when a default/method changed, every
     comparative claim in --help, presets, and docs is invalidated — re-measure or
     restate (F50: --fast's "quality-neutral" predated the perband default).
   - PROMISED-BUT-UNWIRED: grep plans/task ledgers for shipped-in-plan features
     absent from the package (F50: --boundary-out); wire or explicitly re-defer.
   - SUPERSEDED SURFACES: flags/paths replaced by newer mechanisms must say so
     (deprecation with the honest reason), not linger as "experimental."
5. **Audit** (per DEVSTYLE §10, mandatory if subagents ran): clean tree, no stashes,
   no rebase leftovers, local == origin, linear history, no duplicate sections in
   shared files, pytest green ON THE COMPOSED STATE.
6. **Compact state**: update `research/STATE.md`; keep only current frozen evidence
   in Git. Historical ledgers, generated reports, and caches belong in Git history.
7. **Ship**: commit per logical unit (no trailers), pull --rebase, push.
