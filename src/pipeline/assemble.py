"""Assemble detector boxes + recognizer text into reading order and grounded markdown.

Deterministic geometry (no training): column clustering -> top-to-bottom / left-to-right reading
order -> optional line->block merge -> DeepSeek-OCR grounded-markdown emission (same format as
src/render/to_grounded_markdown.py, so output is comparable to the AR baseline and scorable by
src/train/eval/metrics.page_metrics).
"""
from __future__ import annotations

import re

from src.detect.data.build_det_targets import CLASSES

REF_OPEN, REF_CLOSE = "<|ref|>", "<|/ref|>"
DET_OPEN, DET_CLOSE = "<|det|>", "<|/det|>"

Line = dict   # {"box":[x1,y1,x2,y2] (pixel), "text":str, "cls":int}

# region-type -> markdown rendering of a block's text (canonical V3 mapping; matches
# src/render/to_grounded_markdown.py so the model's output converts to HTML the same way labels do).
_MD = {
    "title": lambda t: f"# {t}",
    "heading": lambda t: f"## {t}",
    "subheading": lambda t: f"### {t}",
    "section_header": lambda t: f"## {t}",      # back-compat with frozen V1/V2 output
    "list_item": lambda t: f"- {t}",
    "caption": lambda t: f"*{t}*",
    "form_label": lambda t: f"**{t}**",
    "form_value": lambda t: t,
    "text": lambda t: t,
}
_NONTEXT = ("image", "signature", "hand_drawing")   # rendered as markdown <img> placeholders


def _norm_box(b, w, h):
    x1, y1, x2, y2 = b
    return [max(0, min(999, round(x1 / w * 999))), max(0, min(999, round(y1 / h * 999))),
            max(0, min(999, round(x2 / w * 999))), max(0, min(999, round(y2 / h * 999)))]


def _cls_id(name: str, classes: list = CLASSES) -> int:
    return classes.index(name) if name in classes else classes.index("text")


def refine_structure(lines: list[Line], page_w: float, classes: list = CLASSES) -> list[Line]:
    """Light geometry fallback that TRUSTS the (V2) learned class.

    V2 trains the class head on diverse layouts so its predictions are reliable for production
    structure — therefore we no longer override a confident learned class. Geometry is used only to
    disambiguate the generic ``text`` default: an unmarked line that is genuinely centred + short (a
    heading the head left as text) is promoted. Everything else, including title / section_header /
    list_item / table_cell / form_* / image / formula, is kept exactly as predicted.
    """
    if not lines:
        return lines
    heights = sorted(l["box"][3] - l["box"][1] for l in lines)
    med_h = heights[len(heights) // 2] or 1

    for l in lines:
        if classes[l.get("cls", 0)] != "text":
            continue                                   # trust the learned class
        x1, y1, x2, y2 = l["box"]
        w, h = x2 - x1, y2 - y1
        cx = (x1 + x2) / 2
        center_off = abs(cx - page_w / 2) / page_w
        left_margin, right_margin = x1 / page_w, (page_w - x2) / page_w
        width_frac = w / page_w
        tall = h >= med_h * 1.15
        # genuinely centred + short = a heading the head defaulted to text (symmetric margins reject
        # indented body lines that merely balance near the page centre).
        centered = (center_off < 0.10 and abs(left_margin - right_margin) < 0.10
                    and left_margin > 0.12 and width_frac < 0.55)
        if centered and tall:
            l["cls"] = _cls_id("title", classes)
        elif centered:
            l["cls"] = _cls_id("heading", classes)
        elif tall and width_frac < 0.45 and left_margin < 0.2:
            l["cls"] = _cls_id("heading", classes)   # short left marker (e.g. មាត្រា១)
    return lines


def cluster_columns(lines: list[Line], page_w: float) -> list[int]:
    """Assign each line a column index via gap-based 1-D clustering on box x-centers."""
    if not lines:
        return []
    centers = sorted((((l["box"][0] + l["box"][2]) / 2), i) for i, l in enumerate(lines))
    gap_thresh = page_w * 0.18      # a column break is a horizontal gap > 18% of page width
    col_of = [0] * len(lines)
    col = 0
    prev_c = centers[0][0]
    # assign in x-center order, bumping column when a large gap appears
    order = [i for _, i in centers]
    cx = [c for c, _ in centers]
    for k in range(1, len(order)):
        if cx[k] - cx[k - 1] > gap_thresh:
            col += 1
        col_of[order[k]] = col
    # but a single big gap from one outlier shouldn't fragment; cap columns at 3
    if col > 2:
        # fall back to single column if clustering is noisy
        return [0] * len(lines)
    return col_of


def reading_order(lines: list[Line], page_w: float) -> list[int]:
    """Return line indices in reading order: columns L->R, rows T->B, ties L->R."""
    if not lines:
        return []
    cols = cluster_columns(lines, page_w)
    heights = sorted((l["box"][3] - l["box"][1]) for l in lines)
    med_h = heights[len(heights) // 2] or 1
    y_tol = med_h * 0.6

    def col_x(c):
        xs = [(l["box"][0] + l["box"][2]) / 2 for l, cc in zip(lines, cols) if cc == c]
        return sum(xs) / len(xs) if xs else 0

    order: list[int] = []
    for c in sorted(set(cols), key=col_x):
        idxs = [i for i in range(len(lines)) if cols[i] == c]
        idxs.sort(key=lambda i: lines[i]["box"][1])       # by top
        # within a y-tolerance band, order left-to-right
        i = 0
        while i < len(idxs):
            j = i + 1
            top = lines[idxs[i]]["box"][1]
            while j < len(idxs) and lines[idxs[j]]["box"][1] - top <= y_tol:
                j += 1
            band = sorted(idxs[i:j], key=lambda k: lines[k]["box"][0])
            order.extend(band)
            i = j
    return order


_NUMBERED_START = re.compile(r"^\s*(?:[០-៩]+|\d+)[.)]|^\s*[ក-អ][.)]\s|^\s*[-•]\s")


def merge_lines_to_blocks(ordered: list[Line], page_w: float, classes: list = CLASSES) -> list[Line]:
    """Merge consecutive lines with small vertical gap + similar left edge into block units.

    Class-aware: only merges lines of the SAME region-type and only for flowing types (text,
    list_item, form_value, caption). Titles, headers, and table cells stay as their own units so
    document structure is preserved. A line that STARTS with numbering (១. / 1. / ក) / -) begins a
    new unit — otherwise numbered questions collapse into one run-on paragraph.
    """
    if not ordered:
        return []
    blocks: list[Line] = []
    heights = sorted((l["box"][3] - l["box"][1]) for l in ordered)
    med_h = heights[len(heights) // 2] or 1
    cur = None
    for ln in ordered:
        x1, y1, x2, y2 = ln["box"]
        cls = ln.get("cls", 0)
        if cur is None:
            cur = {**ln, "box": list(ln["box"])}
            continue
        cx1, cy1, cx2, cy2 = cur["box"]
        v_gap = y1 - cy2
        same_left = abs(x1 - cx1) < med_h * 1.5
        flowing = classes[cls] in ("text", "list_item", "form_value", "caption")
        if cls == cur["cls"] and flowing and -med_h * 0.3 <= v_gap <= med_h * 1.3 and same_left \
                and not _NUMBERED_START.match(ln.get("text", "")):
            cur["box"] = [min(cx1, x1), min(cy1, y1), max(cx2, x2), max(cy2, y2)]
            cur["text"] += ln["text"]        # Khmer flows without spaces across line breaks
        else:
            blocks.append(cur)
            cur = {**ln, "box": list(ln["box"])}    # preserve extra keys (e.g. fig_index)
    if cur:
        blocks.append(cur)
    return blocks


def _reconstruct_table(cells: list[Line]) -> str:
    """Group table_cell units into a markdown table by row (y-overlap) then column (x order)."""
    if not cells:
        return ""
    cells = sorted(cells, key=lambda c: c["box"][1])
    heights = sorted(c["box"][3] - c["box"][1] for c in cells)
    tol = (heights[len(heights) // 2] or 10) * 0.7
    rows: list[list[Line]] = []
    for c in cells:
        if rows and abs(c["box"][1] - rows[-1][0]["box"][1]) <= tol:
            rows[-1].append(c)
        else:
            rows.append([c])
    ncol = max(len(r) for r in rows)
    md = []
    for ri, r in enumerate(rows):
        r = sorted(r, key=lambda c: c["box"][0])
        texts = [c["text"] for c in r] + [""] * (ncol - len(r))
        md.append("| " + " | ".join(texts) + " |")
        if ri == 0:
            md.append("| " + " | ".join(["---"] * ncol) + " |")
    return "\n".join(md)


def build_markdown(blocks: list[Line], classes: list = CLASSES) -> str:
    """Render assembled blocks to structured markdown via region types (headings/lists/tables)."""
    out: list[str] = []
    i = 0
    while i < len(blocks):
        name = classes[blocks[i].get("cls", 0)]
        if name in ("table_head", "table_cell"):
            run = []
            while i < len(blocks) and classes[blocks[i].get("cls", 0)] in ("table_head", "table_cell"):
                run.append(blocks[i]); i += 1
            out.append(_reconstruct_table(run))
            continue
        if name in _NONTEXT:
            fi = blocks[i].get("fig_index")
            out.append(f"![{name}](figure_{fi})" if fi is not None else f"![{name}]()")
            i += 1; continue
        if name == "formula":
            t = blocks[i]["text"].strip()
            out.append(f"$$ {t} $$" if t else "$$ [រូបមន្ត] $$")
            i += 1; continue
        if (name == "form_label" and i + 1 < len(blocks)
                and classes[blocks[i + 1].get("cls", 0)] == "form_value"):
            out.append(f"**{blocks[i]['text']}** {blocks[i + 1]['text']}".strip())
            i += 2; continue
        out.append(_MD.get(name, lambda t: t)(blocks[i]["text"]))
        i += 1
    return "\n\n".join(out)


def _table_rows(cells: list[Line]) -> list[list[str]]:
    """Group table_cell units into rows (y-overlap) x columns (x order). Shared by md + html."""
    cells = sorted(cells, key=lambda c: c["box"][1])
    heights = sorted(c["box"][3] - c["box"][1] for c in cells)
    tol = (heights[len(heights) // 2] or 10) * 0.7
    rows: list[list[Line]] = []
    for c in cells:
        if rows and abs(c["box"][1] - rows[-1][0]["box"][1]) <= tol:
            rows[-1].append(c)
        else:
            rows.append([c])
    ncol = max(len(r) for r in rows)
    return [[c["text"] for c in sorted(r, key=lambda c: c["box"][0])] + [""] * (ncol - len(r))
            for r in rows]


def _esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_html(blocks: list[Line], classes: list = CLASSES) -> str:
    """Render assembled blocks to structured HTML (headings, lists, <table> like OCR studios)."""
    out: list[str] = []
    i = 0
    while i < len(blocks):
        name = classes[blocks[i].get("cls", 0)]
        if name in ("table_head", "table_cell"):
            run = []
            while i < len(blocks) and classes[blocks[i].get("cls", 0)] in ("table_head", "table_cell"):
                run.append(blocks[i]); i += 1
            rows = _table_rows(run)
            if not rows:
                continue
            head, *body = rows
            html = ["<table>", "  <thead>", "    <tr>"]
            html += [f"      <th>{_esc(c)}</th>" for c in head]
            html += ["    </tr>", "  </thead>", "  <tbody>"]
            for r in body:
                html.append("    <tr>")
                html += [f"      <td>{_esc(c)}</td>" for c in r]
                html.append("    </tr>")
            html += ["  </tbody>", "</table>"]
            out.append("\n".join(html))
            continue
        if name in _NONTEXT:
            fi = blocks[i].get("fig_index")
            # data-figure lets the viewer swap in the actual cropped image (data URL)
            out.append(f'<img class="{name}" data-figure="{fi}" alt="{name}">' if fi is not None
                       else f'<div class="figure-ph {name}">[{name}]</div>')
            i += 1; continue
        if name == "formula":
            t = _esc(blocks[i]["text"].strip())
            out.append(f'<pre class="formula">{t or "[រូបមន្ត]"}</pre>')
            i += 1; continue
        if (name == "form_label" and i + 1 < len(blocks)
                and classes[blocks[i + 1].get("cls", 0)] == "form_value"):
            out.append(f"<p><strong>{_esc(blocks[i]['text'])}</strong> {_esc(blocks[i + 1]['text'])}</p>")
            i += 2; continue
        t = _esc(blocks[i]["text"])
        tag = {"title": f"<h1>{t}</h1>", "heading": f"<h2>{t}</h2>",
               "subheading": f"<h3>{t}</h3>", "section_header": f"<h2>{t}</h2>",
               "list_item": f"<li>{t}</li>", "caption": f"<figcaption>{t}</figcaption>",
               "form_label": f"<strong>{t}</strong>"}.get(name, f"<p>{t}</p>")
        out.append(tag)
        i += 1
    return "\n".join(out)


def to_grounded(units: list[Line], page_w: float, page_h: float) -> str:
    """Emit DeepSeek-OCR grounded markdown from ordered units (lines or blocks)."""
    parts = []
    for u in units:
        nb = _norm_box(u["box"], page_w, page_h)
        parts.append(f"{REF_OPEN}{u['text']}{REF_CLOSE}{DET_OPEN}[[{nb[0]}, {nb[1]}, "
                     f"{nb[2]}, {nb[3]}]]{DET_CLOSE}")
    return "\n".join(parts)


def assemble(lines: list[Line], page_w: float, page_h: float, merge_blocks: bool = True,
             refine: bool = True, classes: list = CLASSES) -> dict:
    """Full assembly. Returns {order, line_units, block_units, grounded_lines, grounded_blocks}.

    `classes` is the detector's class list (default V3 CLASSES); pass CLASSES_V4 for the V4 detector
    so cls ids resolve to the right region names.
    """
    if refine:
        lines = refine_structure(lines, page_w, classes)
    order = reading_order(lines, page_w)
    ordered = [lines[i] for i in order]
    blocks = merge_lines_to_blocks(ordered, page_w, classes) if merge_blocks else ordered
    return {
        "line_units": ordered,
        "block_units": blocks,
        "grounded_lines": to_grounded(ordered, page_w, page_h),
        "grounded_blocks": to_grounded(blocks, page_w, page_h),
        "markdown": build_markdown(blocks, classes),   # structured: headings, lists, tables
        "html": build_html(blocks, classes),           # HTML with <table> (OCR-studio style)
        "text": "\n".join(b["text"] for b in blocks),   # plain text, reading order
    }
