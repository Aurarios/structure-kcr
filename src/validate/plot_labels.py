"""Overlay V3 label JSON onto its page image so you can eyeball box + class quality.

Draws every block's box color-coded by detector class, with a small readable class tag (table cells
are colored only, to avoid clutter), writes one annotated JPG per page, a class-color legend, and an
HTML contact sheet grouped by layout type.

  python -m src.validate.plot_labels --src E:/kcr-v3-smoke --out E:/kcr-v3-smoke/_overlays
  python -m src.validate.plot_labels --src E:/kcr-v3 --max-per-layout 8     # sample the full set
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2

from ..detect.data.build_det_targets import CLASSES

# fixed, distinct BGR per class (stable across runs so colors are memorable)
COLORS = {
    "title": (255, 140, 0), "heading": (220, 90, 0), "subheading": (180, 120, 40),
    "text": (90, 90, 90), "list_item": (140, 70, 200), "caption": (110, 110, 110),
    "table_head": (0, 120, 255), "table_cell": (0, 175, 0),
    "image": (0, 0, 230), "signature": (230, 0, 230), "hand_drawing": (0, 220, 220),
    "formula": (200, 0, 200), "form_label": (0, 165, 215), "form_value": (60, 60, 230),
    "background": (200, 200, 200),
}
NO_TAG = {"table_cell"}            # too many per page -> color only, no text tag


def _tag(img, x, y, text, color):
    fs, th = 0.42, 1
    (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    yt = max(tht + 3, y)
    cv2.rectangle(img, (x, yt - tht - 3), (x + tw + 4, yt + 2), color, -1)
    cv2.putText(img, text, (x + 2, yt - 1), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)


def draw_label(label_path: str, out_path: str) -> tuple[dict, str]:
    d = json.loads(Path(label_path).read_text(encoding="utf-8"))
    img = cv2.imread(d["image"])
    if img is None:
        return {}, ""
    counts: dict[str, int] = defaultdict(int)
    for b in d["blocks"]:
        c = b["block_type"]
        col = COLORS.get(c, (128, 128, 128))
        # draw the PER-LINE boxes (the actual detector targets), not the block bbox; non-text
        # regions have a single whole-region line box. Tag only the first line to avoid clutter.
        lines = b.get("lines") or [b["bbox"]]
        for i, ln in enumerate(lines):
            x1, y1, x2, y2 = (int(v) for v in ln)
            counts[c] += 1
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            if i == 0 and c not in NO_TAG:
                _tag(img, x1, y1, c, col)
    cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return counts, d.get("markdown", "")


def make_legend(out_path: str) -> None:
    import numpy as np
    rowh, w = 30, 240
    img = np.full((rowh * len(CLASSES) + 10, w, 3), 255, np.uint8)
    for i, c in enumerate(CLASSES):
        y = 10 + i * rowh
        cv2.rectangle(img, (8, y), (34, y + 20), COLORS.get(c, (128, 128, 128)), -1)
        cv2.putText(img, c, (42, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, img)


def main() -> None:
    ap = argparse.ArgumentParser(description="overlay V3 labels on page images")
    ap.add_argument("--src", default="E:/kcr-v3-smoke", help="dataset root (has {layout}/labels/*.json)")
    ap.add_argument("--out", default=None, help="output dir (default <src>/_overlays)")
    ap.add_argument("--max-per-layout", type=int, default=0, help="cap images per layout (0=all)")
    args = ap.parse_args()

    out = Path(args.out or (Path(args.src) / "_overlays"))
    out.mkdir(parents=True, exist_ok=True)
    make_legend(str(out / "_legend.png"))

    by_layout: dict[str, list[str]] = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(args.src, "*", "labels", "*.json"))):
        lt = Path(f).parts[-3]
        by_layout[lt].append(f)

    total = defaultdict(int)
    rows_html: list[str] = []
    for lt in sorted(by_layout):
        files = by_layout[lt]
        if args.max_per_layout:
            files = files[:args.max_per_layout]
        rows_html.append(f"<h2>{html.escape(lt)} ({len(files)})</h2>")
        for f in files:
            name = Path(f).stem + ".jpg"
            counts, md = draw_label(f, str(out / name))
            for k, v in counts.items():
                total[k] += v
            # each page: box-overlay image | structured markdown (raw) | rendered HTML preview
            rows_html.append(
                f'<div class="pair"><div class="cap">{html.escape(Path(f).stem)}</div>'
                f'<div class="cols"><img src="{name}">'
                f'<pre class="raw">{html.escape(md)}</pre>'
                f'<div class="rendered"></div></div></div>')

    # marked.js (GFM: headings/lists/tables/images/bold) + MathJax ($$..$$) render the structured
    # markdown to HTML live, so you verify the md->html structure conversion. Falls back to the raw
    # <pre> if offline (no CDN).
    sheet = f"""<!doctype html><meta charset=utf-8><title>V3 label + structure</title>
<style>
 body{{font-family:sans-serif;background:#1e1e1e;color:#ddd;margin:16px}}
 h2{{border-bottom:1px solid #555;margin-top:28px}}
 .pair{{border:1px solid #333;border-radius:6px;margin:10px 0;padding:8px;background:#262626}}
 .cap{{font:12px monospace;color:#8cf;margin-bottom:6px}}
 .cols{{display:grid;grid-template-columns:380px 360px 1fr;gap:12px;align-items:start}}
 img{{width:380px;border:1px solid #444}}
 pre.raw{{white-space:pre-wrap;font:11px/1.4 monospace;background:#111;color:#bdf;
   padding:8px;border-radius:4px;max-height:560px;overflow:auto;margin:0}}
 .rendered{{background:#fff;color:#111;padding:10px 14px;border-radius:4px;max-height:560px;overflow:auto}}
 .rendered img{{max-width:120px;border:1px dashed #999}}
 .rendered table{{border-collapse:collapse}} .rendered th,.rendered td{{border:1px solid #999;padding:2px 6px}}
</style>
<h1>V3 label overlays + structured Markdown→HTML — {sum(len(v) for v in by_layout.values())} pages</h1>
<p>Columns per page: <b>box overlay</b> · <b>structured markdown</b> · <b>rendered HTML</b>.
<img src="_legend.png" style="width:240px;vertical-align:top"></p>
{''.join(rows_html)}
<script>
window.MathJax={{tex:{{inlineMath:[['$','$']],displayMath:[['$$','$$']]}}}};
</script>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<script>
document.querySelectorAll('.pair').forEach(p=>{{
  const raw=p.querySelector('.raw').textContent, r=p.querySelector('.rendered');
  if(window.marked){{r.innerHTML=marked.parse(raw);}} else {{r.textContent='(offline: see raw)';}}
}});
if(window.MathJax&&MathJax.typesetPromise){{MathJax.typesetPromise();}}
</script>"""
    (out / "contact_sheet.html").write_text(sheet, encoding="utf-8")

    print(f"[plot] wrote overlays + contact_sheet.html to {out}")
    print("[plot] class instance counts:", dict(sorted(total.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
