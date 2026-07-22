---
name: frontier-litscan
description: Literature/innovation scan that appends a dated section to FRONTIER.md — strictly non-redundant with all prior scans, our findings, and our measured negatives.
---

# Frontier literature scan (incremental)

Goal: add genuinely NEW directions since the previous scan. Best run BY a subagent
(general-purpose, WebSearch/WebFetch) briefed per DEVSTYLE §10.

1. **Load the redundancy baseline**: research/FRONTIER.md IN FULL (all rows, ledgers,
   and every previous "## Literature scan (date)" section — prior entries are
   excluded territory); FINDINGS.md SYNTHESIS + headlines; PLAYBOOK traps (so
   measured negatives — e.g. full-band matte inversion, no-ref audit of synthesis —
   are never re-imported as novelty).
2. **Scan window**: focus on work published/updated since the previous scan's date;
   re-probe rows marked ❌ for availability changes.
3. **Verify on the actual source** (paper/repo/page), never search snippets.
4. **Write**: new dated "## Literature scan (YYYY-MM-DD)" section, 5-12 entries max,
   each with citation, why-it-matters tied to a named finding/wall, redundancy note,
   and a concrete first experiment in our factory/gate idiom. AMEND existing rows
   in place when findings change their status (marked "lit-scan DATE: ...").
   Include the EXCLUSION list with one-line reasons — it is a deliverable.
5. **Ship**: commit only FRONTIER.md (no trailers), pull --rebase, push.
