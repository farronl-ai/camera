---
name: market-refresh
description: Update docs/market_analysis.html in place — re-verify competitor facts/prices, sync our-status cells with current capabilities, keep the candid-analyst voice.
---

# Market analysis refresh (incremental)

Goal: keep the analysis TRUE, not rewrite it. Suitable for a subagent (needs web).

1. **Diff our side first**: read FINDINGS SYNTHESIS + SHOWCASE since the doc's
   stated date; list our cells that changed (gaps closed, e.g. 16-bit/RAW/alignment;
   new validated regimes; recall numbers). Update ONLY those cells/paragraphs;
   red cells stay red until evidence says otherwise.
2. **Re-verify the market side**: every price/feature cell older than ~1 quarter —
   check the linked vendor page; update numbers + access dates. Re-check the "most
   important missing datum" note (head-to-head vs Helicon/Zerene) — if that benchmark
   has since been run, replace the caveat with the result.
3. **Keep the contract**: fully self-contained HTML (inline CSS/SVG, no external
   assets, light+dark), no invented market-size numbers, candid tone, all claims
   linked. Update the date in title/footer.
4. **Ship**: commit only the html (no trailers), pull --rebase, push. Then the MAIN
   session republishes the artifact (same URL) from the body-only derivative.
