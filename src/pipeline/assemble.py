"""Assemble detector boxes + recognizer text into reading order and grounded markdown.

Deterministic geometry (no training): column clustering -> top-to-bottom / left-to-right reading
order -> optional line->block merge -> DeepSeek-OCR grounded-markdown emission (same format as
src/render/to_grounded_markdown.py, so output is comparable to the AR baseline and scorable by
src/train/eval/metrics.page_metrics).
"""
from __future__ import annotations

from src.detect.data.build_det_targets import CLASSES

REF_OPEN, REF_CLOSE = "<|ref|>", "<|/ref|>"
DET_OPEN, DET_CLOSE = "<|det|>", "<|/det|>"

Line = dict   # {"box":[x1,y1,x2,y2] (pixel), "text":str, "cls":int}

# region-type id -> markdown rendering of a block's text
_MD = {
    "title": lambda t: f"# {t}",
    "section_header": lambda t: f"## {t}",
    "list_item": lambda t: f"- {t}",
    "caption": lambda t: f"*{t}*",
    "form_label": lambda t: f"**{t}**",
    "form_value": lambda t: t,
    "text": lambda t: t,
}


def _norm_box(b, w, h):
    x1, y1, x2, y2 = b
    return [max(0, min(999, round(x1 / w * 999))), max(0, min(999, round(y1 / h * 999))),
            max(0, min(999, round(x2 / w * 999))), max(0, min(999, round(y2 / h * 999)))]


def _cls_id(name: str) -> int:
    return CLASSES.index(name) if name in CLASSES else CLASSES.index("text")


def refine_structure(lines: list[Line], page_w: float) -> list[Line]:
    """Reconcile the learned class with domain-invariant geometry cues.

    The detector's class head overfits synthetic heading *appearance* and collapses real-doc
    headings to 'text'. Layout geometry (centering, relative height, width) transfers across domains
    far better, so we use it to recover headings:
      - centered + not full-width  -> title (large) / section_header (smaller)
      - left-margin + short + taller-than-body -> section_header (e.g. article markers)
    The learned class is kept for table_cell / form_* / list_item, where geometry alone is weak;
    geometry only overrides the text<->heading confusion.
    """
    if not lines:
        return lines
    heights = sorted(l["box"][3] - l["box"][1] for l in lines)
    med_h = heights[len(heights) // 2] or 1
    widths = [l["box"][2] - l["box"][0] for l in lines]
    med_w = sorted(widths)[len(widths) // 2] or 1

    for l in lines:
        x1, y1, x2, y2 = l["box"]
        w, h = x2 - x1, y2 - y1
        cx = (x1 + x2) / 2
        learned = CLASSES[l.get("cls", 0)]
        # keep structural classes the geometry can't judge well
        if learned in ("table_cell", "form_label", "form_value", "list_item"):
            l["cls"] = l.get("cls", 0)
            continue
        center_off = abs(cx - page_w / 2) / page_w          # 0 = perfectly centered
        left_margin = x1 / page_w
        width_frac = w / page_w
        tall = h >= med_h * 1.15
        centered = center_off < 0.12 and width_frac < 0.85   # centered and not full-width
        if centered and tall:
            l["cls"] = _cls_id("title")
        elif centered:
            l["cls"] = _cls_id("section_header")
        elif tall and width_frac < 0.45 and left_margin < 0.2:
            # short, left, larger-than-body line = a heading/marker (e.g. មាត្រា១)
            l["cls"] = _cls_id("section_header")
        else:
            l["cls"] = _cls_id("text")
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


def merge_lines_to_blocks(ordered: list[Line], page_w: float) -> list[Line]:
    """Merge consecutive lines with small vertical gap + similar left edge into block units.

    Class-aware: only merges lines of the SAME region-type and only for flowing types (text,
    list_item, form_value, caption). Titles, headers, and table cells stay as their own units so
    document structure is preserved.
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
            cur = {"box": list(ln["box"]), "text": ln["text"], "cls": cls}
            continue
        cx1, cy1, cx2, cy2 = cur["box"]
        v_gap = y1 - cy2
        same_left = abs(x1 - cx1) < med_h * 1.5
        flowing = CLASSES[cls] in ("text", "list_item", "form_value", "caption")
        if cls == cur["cls"] and flowing and -med_h * 0.3 <= v_gap <= med_h * 1.3 and same_left:
            cur["box"] = [min(cx1, x1), min(cy1, y1), max(cx2, x2), max(cy2, y2)]
            cur["text"] += ln["text"]        # Khmer flows without spaces across line breaks
        else:
            blocks.append(cur)
            cur = {"box": list(ln["box"]), "text": ln["text"], "cls": cls}
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


def build_markdown(blocks: list[Line]) -> str:
    """Render assembled blocks to structured markdown via region types (headings/lists/tables)."""
    out: list[str] = []
    i = 0
    while i < len(blocks):
        name = CLASSES[blocks[i].get("cls", 0)]
        if name == "table_cell":
            run = []
            while i < len(blocks) and CLASSES[blocks[i].get("cls", 0)] == "table_cell":
                run.append(blocks[i]); i += 1
            out.append(_reconstruct_table(run))
            continue
        out.append(_MD.get(name, lambda t: t)(blocks[i]["text"]))
        i += 1
    return "\n\n".join(out)


def to_grounded(units: list[Line], page_w: float, page_h: float) -> str:
    """Emit DeepSeek-OCR grounded markdown from ordered units (lines or blocks)."""
    parts = []
    for u in units:
        nb = _norm_box(u["box"], page_w, page_h)
        parts.append(f"{REF_OPEN}{u['text']}{REF_CLOSE}{DET_OPEN}[[{nb[0]}, {nb[1]}, "
                     f"{nb[2]}, {nb[3]}]]{DET_CLOSE}")
    return "\n".join(parts)


def assemble(lines: list[Line], page_w: float, page_h: float, merge_blocks: bool = True,
             refine: bool = True) -> dict:
    """Full assembly. Returns {order, line_units, block_units, grounded_lines, grounded_blocks}."""
    if refine:
        lines = refine_structure(lines, page_w)
    order = reading_order(lines, page_w)
    ordered = [lines[i] for i in order]
    blocks = merge_lines_to_blocks(ordered, page_w) if merge_blocks else ordered
    return {
        "line_units": ordered,
        "block_units": blocks,
        "grounded_lines": to_grounded(ordered, page_w, page_h),
        "grounded_blocks": to_grounded(blocks, page_w, page_h),
        "markdown": build_markdown(blocks),   # structured: headings, lists, tables
    }
