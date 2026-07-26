---
name: frontier-litscan
description: Literature/innovation scan that updates the compact active frontier without rebuilding a chronological ledger.
---

# Frontier literature scan (incremental)

Goal: add genuinely NEW directions since the previous scan. Best run BY a subagent
(general-purpose, WebSearch/WebFetch) briefed per DEVSTYLE §10.

1. **Load the redundancy baseline**: `STATE.md`, `FRONTIER.md`, `FINDINGS.md`, and
   the relevant PLAYBOOK traps. Consult Git history only when a candidate appears
   suspiciously familiar.
2. **Scan window**: focus on work published/updated since the previous scan's date;
   re-probe rows marked ❌ for availability changes.
3. **Verify on the actual source** (paper/repo/page), never search snippets.
4. **Write compactly**: amend an active direction or add at most 3 new literature
   anchors, each tied to a named wall and concrete experiment. Replace stale
   anchors instead of appending a dated chronicle. Update the exclusion list when
   a tempting but inapplicable family appears.
5. **Ship**: commit only FRONTIER.md (no trailers), pull --rebase, push.
