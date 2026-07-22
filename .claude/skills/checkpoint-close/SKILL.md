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
   shipped), report.py + regenerate + republish the live-status artifact (same URL),
   /market-refresh (if our capability cells changed materially).
3. **Memory**: update project memory state snapshot; add durable lessons to the
   playbook memory. Convert relative dates to absolute.
4. **Audit** (per DEVSTYLE §10, mandatory if subagents ran): clean tree, no stashes,
   no rebase leftovers, local == origin, linear history, no duplicate sections in
   shared files, pytest green ON THE COMPOSED STATE.
5. **Ship**: commit per logical unit (no trailers), pull --rebase, push. Close/annotate
   task-list entries with what the next session inherits (needs-doing, clearly marked).
