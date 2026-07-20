#!/usr/bin/env python3
"""Build research/report.html — the live status of the adaptive-engine program.

Reads manifest.json / metric_weights.json / best_global.json and embeds a few
key images. Rebuilt after each milestone and published as an artifact.
"""

from __future__ import annotations

import base64
import glob
import json
import os

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, default=None):
    p = os.path.join(HERE, name)
    return json.load(open(p)) if os.path.exists(p) else default


def img_tag(path, max_w=1100, q=82):
    im = cv2.imread(path)
    if im is None:
        return "<p class='muted'>[missing image]</p>"
    h, w = im.shape[:2]
    if w > max_w:
        im = cv2.resize(im, (max_w, round(h * max_w / w)), interpolation=cv2.INTER_AREA)
    _ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(buf).decode()}">'


def main():
    weights = load("metric_weights.json", {})
    manifest = load("manifest.json", [])
    registry = load(os.path.join("data", "registry.json"), {})

    def latest(stage):
        xs = [m for m in manifest if m.get("stage") == stage]
        return xs[-1] if xs else None

    m1 = latest("M1_global")
    m2 = latest("M2_adaptive")

    data_rows = "".join(
        f"<tr><td class='mono'>{k}</td><td class='mono num'>{v['count']}</td>"
        f"<td class='mono'>{'×'.join(map(str, v['sample_shape'][:2])) if v.get('sample_shape') else '?'}</td></tr>"
        for k, v in (registry or {}).items()
    )

    def metric_block():
        if not weights:
            return "<p class='muted'>not yet calibrated</p>"
        ind = weights.get("individual", {})
        rows = "".join(f"<tr><td class='mono'>{k}</td><td class='mono num'>{v:+.3f}</td></tr>"
                       for k, v in sorted(ind.items(), key=lambda t: -t[1]))
        w = weights.get("weights", {})
        return (f"<p>Composite = " + " + ".join(f"{c:.2f}·{n}" for n, c in w.items() if c > 0)
                + f" &nbsp; (mean per-pair Spearman vs GT-SSIM = <b>{weights.get('spearman', 0):+.3f}</b>)</p>"
                f"<table><thead><tr><th>metric</th><th>Spearman vs GT</th></tr></thead><tbody>{rows}</tbody></table>")

    def ssim_row(m, keys):
        if not m:
            return "<p class='muted'>pending</p>"
        return "".join(f"<tr><td>{lbl}</td><td class='mono num'>{m[k]:.4f}</td></tr>" for lbl, k in keys)

    winner_img = ""
    wm = os.path.join(HERE, "c05_winner_map.png")
    if os.path.exists(wm):
        winner_img = f"<figure><figcaption>Region detection — which tune wins where [ source · winner-map · adaptive ]</figcaption><div class='frame'>{img_tag(wm, 1400)}</div></figure>"

    html = f"""<style>
:root{{--bg:#0e1113;--surface:#161c1f;--ink:#e7edef;--muted:#8fa0a6;--line:#28343a;--accent:#39d3bf;--sans:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}}
@media(prefers-color-scheme:light){{:root{{--bg:#eef1f2;--surface:#fff;--ink:#141b1d;--muted:#5a686e;--line:#d3dbde;--accent:#0f9488}}}}
:root[data-theme=light]{{--bg:#eef1f2;--surface:#fff;--ink:#141b1d;--muted:#5a686e;--line:#d3dbde;--accent:#0f9488}}
:root[data-theme=dark]{{--bg:#0e1113;--surface:#161c1f;--ink:#e7edef;--muted:#8fa0a6;--line:#28343a;--accent:#39d3bf}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.6}}
.wrap{{max-width:920px;margin:0 auto;padding:clamp(1.2rem,4vw,3rem) 1.2rem}}
.eyebrow{{font-family:var(--mono);font-size:.72rem;letter-spacing:.2em;text-transform:uppercase;color:var(--accent);margin:0 0 .6rem}}
h1{{font-size:clamp(1.7rem,4vw,2.5rem);line-height:1.1;margin:0 0 .5rem;letter-spacing:-.02em}}
h2{{font-size:1.25rem;margin:2.4rem 0 .5rem;letter-spacing:-.01em}}
.deck{{color:var(--muted);max-width:62ch}}
table{{border-collapse:collapse;width:100%;font-size:.9rem;margin:.5rem 0}}
th,td{{text-align:right;padding:.4rem .7rem;border-bottom:1px solid var(--line)}}th:first-child,td:first-child{{text-align:left}}
thead th{{font-family:var(--mono);font-size:.7rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}}
.mono{{font-family:var(--mono)}}.num{{font-variant-numeric:tabular-nums}}.muted{{color:var(--muted)}}
.card{{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:1rem 1.3rem;margin:1rem 0}}
.milestones{{display:flex;flex-wrap:wrap;gap:.5rem;margin:1rem 0}}
.pill{{font-family:var(--mono);font-size:.75rem;padding:.25rem .7rem;border-radius:999px;border:1px solid var(--line)}}
.done{{border-color:var(--accent);color:var(--accent)}}.wip{{color:var(--ink)}}.todo{{color:var(--muted)}}
figure{{margin:1rem 0}}figcaption{{font-size:.85rem;color:var(--muted);margin-bottom:.4rem}}
.frame{{overflow-x:auto;border:1px solid var(--line);border-radius:10px}}img{{display:block;max-width:100%;height:auto}}
</style>
<div class="wrap">
  <p class="eyebrow">focusstack · adaptive engine · live status</p>
  <h1>Self-tuning, region-aware focus stacking</h1>
  <p class="deck">An autonomous test→refine program: validate a ground-truth-free quality metric, then use it to auto-tune globally, adapt tunes per region, and learn a fast mapping — so a new high-res stack needs no answer key.</p>
  <div class="milestones">
    <span class="pill done">M0 metric ✓</span>
    <span class="pill {'done' if m1 else 'wip'}">M1 global {'✓' if m1 else '…'}</span>
    <span class="pill {'done' if m2 else 'wip' if m1 else 'todo'}">M2 region {'✓' if m2 else '…'}</span>
    <span class="pill todo">M3 learned</span>
    <span class="pill todo">M4 deep</span>
  </div>

  <h2>Data</h2>
  <div class="card"><table><thead><tr><th>dataset</th><th>files</th><th>resolution</th></tr></thead><tbody>{data_rows}</tbody></table></div>

  <h2>M0 — the validated GT-free objective</h2>
  <div class="card">{metric_block()}
  <p class="muted">Only place ground truth is used: to prove the no-reference objective tracks true quality. Inference never needs it.</p></div>

  <h2>M1/M2 — quality on held-out Real-MFF ground truth</h2>
  <div class="card"><table><thead><tr><th>method</th><th>GT-SSIM</th></tr></thead><tbody>
    {ssim_row(m2, [("baseline (blend default)", "baseline_gt_ssim"), ("global auto-tuned (M1)", "global_gt_ssim"), ("region-adaptive (M2)", "adaptive_gt_ssim")]) if m2 else ssim_row(m1, [("baseline", "baseline_gt_ssim"), ("global tuned", "tuned_gt_ssim")])}
  </tbody></table></div>

  {winner_img}
</div>"""
    out = os.path.join(HERE, "report.html")
    open(out, "w").write(html)
    print(f"wrote {out} ({os.path.getsize(out)//1024} KB)")


if __name__ == "__main__":
    main()
