#!/usr/bin/env python3
"""Build docs/SHOWCASE.html from research/showcase_template.html.

The template is the fully styled page with the LaTeX math left as tokens:
`§§...§§` for inline math, `§§§...§§§` for display math. This script renders each
token to native MathML so the page needs no JavaScript, CDN, or web fonts —
it opens offline in any modern browser, sitting next to docs/img/.

Companion to make_showcase.py (which regenerates the figures themselves);
run that first if the images are stale.

Run:  python research/make_showcase_html.py
Needs: pip install latex2mathml   (or: pip install -e .[docs])
"""
from __future__ import annotations
import os
import re
import sys
import json

try:
    import latex2mathml.converter as l2m
except ImportError:
    sys.exit("latex2mathml is required: pip install latex2mathml  (or: pip install -e .[docs])")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TEMPLATE = os.path.join(HERE, "showcase_template.html")
OUT = os.path.join(REPO, "docs", "SHOWCASE.html")
INSPECTION_TEMPLATE = os.path.join(HERE, "inspection_template.html")
INSPECTION_MANIFEST = os.path.join(REPO, "docs", "inspection_manifest.json")
INSPECTION_OUT = os.path.join(REPO, "docs", "INSPECTION.html")


def main() -> None:
    with open(TEMPLATE, encoding="utf-8") as f:
        src = f.read()

    counts = {"display": 0, "inline": 0}

    def render(kind: str):
        def sub(m: re.Match) -> str:
            counts[kind] += 1
            latex = " ".join(m.group(1).split())
            if kind == "display":
                return l2m.convert(latex, display="block")
            return l2m.convert(latex)
        return sub

    out = re.sub(r"§§§(.+?)§§§", render("display"), src, flags=re.S)
    out = re.sub(r"§§(.+?)§§", render("inline"), out, flags=re.S)

    if "§§" in out:
        sys.exit("unbalanced §§ math tokens remain in template — aborting")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"wrote {os.path.relpath(OUT, REPO)}  "
          f"({counts['display']} display, {counts['inline']} inline equations)")

    if not os.path.exists(INSPECTION_MANIFEST):
        print("inspection manifest absent; run make_showcase_specialists.py inspection")
        return
    with open(INSPECTION_TEMPLATE, encoding="utf-8") as f:
        inspection = f.read()
    with open(INSPECTION_MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)
    embedded = json.dumps(manifest, separators=(",", ":")).replace("</", "<\\/")
    token = "__INSPECTION_MANIFEST__"
    if inspection.count(token) != 1:
        sys.exit("inspection template must contain exactly one manifest token")
    inspection = inspection.replace(token, embedded)
    with open(INSPECTION_OUT, "w", encoding="utf-8") as f:
        f.write(inspection)
    print(
        f"wrote {os.path.relpath(INSPECTION_OUT, REPO)} "
        f"({len(manifest['cases'])} deep cases, {len(manifest['ledger'])} ledger rows)"
    )


if __name__ == "__main__":
    main()
