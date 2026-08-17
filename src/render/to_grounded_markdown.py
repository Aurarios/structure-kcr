"""Convert DOM leaf boxes into (a) clean markdown and (b) a DeepSeek-OCR-style grounded target.

Grounded format (one ref per leaf, in reading/DOM order), coordinates normalized to [0,999]:

    <|ref|># ចំណងជើង<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>
    <|ref|>កថាខណ្ឌ...<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>

The ref/det tag strings are constants so they can be swapped to match the model's exact special
tokens at fine-tune time. Tables are reconstructed into a markdown table for the plain-markdown
target and grounded per-cell for the grounded target.
"""
from __future__ import annotations

from .. import khmer_utils as ku

REF_OPEN, REF_CLOSE = "<|ref|>", "<|/ref|>"
DET_OPEN, DET_CLOSE = "<|det|>", "<|/det|>"

# block_type -> markdown prefix for the leaf's text. (V3 uses heading/subheading; section_header is
# kept so the frozen V1/V2 labels still convert.)
_MD_PREFIX = {
    "title": "# ",
    "heading": "## ",
    "subheading": "### ",
    "section_header": "## ",
    "caption": "*",          # closed below
    "text": "",
    "list_item": "- ",
    "word_bank": "",         # one rounded container holding a whole row of words -> plain line
    "table_head": "",
    "form_label": "**",      # closed below
    "form_value": "",
}

# non-text regions (cropped at inference) -> VALID markdown image syntax `![alt](url)` so it
# converts to <img> (a bare `![alt]` is literal text in CommonMark, not an image). The alt is the
# class name so downstream HTML can style/identify it; the empty () is the placeholder destination.
_NONTEXT_MD = {"image": "![image]()", "chart": "![chart]()", "signature": "![signature]()",
               "hand_drawing": "![hand_drawing]()"}


def _norm_box(bbox: list[float], w: float, h: float) -> list[int]:
    x1, y1, x2, y2 = bbox
    return [
        max(0, min(999, round(x1 / w * 999))),
        max(0, min(999, round(y1 / h * 999))),
        max(0, min(999, round(x2 / w * 999))),
        max(0, min(999, round(y2 / h * 999))),
    ]


def _md_text(leaf: dict) -> str:
    t = leaf["text"]
    bt = leaf["block_type"]
    if bt == "caption":
        return f"*{t}*"
    if bt == "form_label":
        return f"**{t}**"
    if bt in _NONTEXT_MD:
        return _NONTEXT_MD[bt]      # non-text region placeholder (cropped at inference)
    if bt == "formula":
        return f"$$ {t} $$"         # display math -> converts to a math block in HTML
    return _MD_PREFIX.get(bt, "") + t


def build_grounded(leaves: list[dict], css_w: float, css_h: float) -> str:
    """One grounded ref per leaf, in DOM order, normalized coords."""
    lines = []
    for lf in leaves:
        box = _norm_box(lf["bbox"], css_w, css_h)
        text = _md_text(lf)
        coords = "[[" + ", ".join(str(v) for v in box) + "]]"
        lines.append(f"{REF_OPEN}{text}{REF_CLOSE}{DET_OPEN}{coords}{DET_CLOSE}")
    return "\n".join(lines)


def build_markdown(leaves: list[dict]) -> str:
    """Clean markdown reconstruction (no boxes). Groups table cells back into tables."""
    out: list[str] = []
    i = 0
    n = len(leaves)
    while i < n:
        lf = leaves[i]
        if lf["block_type"] in ("table_head", "table_cell"):
            j = i
            tid = lf.get("tbl")
            cells = []
            while (j < n and leaves[j]["block_type"] in ("table_head", "table_cell")
                   and leaves[j].get("tbl") == tid):
                cells.append(leaves[j])
                j += 1
            out.append(_render_md_table(cells))
            i = j
        elif (lf["block_type"] == "form_label" and i + 1 < n
              and leaves[i + 1]["block_type"] == "form_value"):
            # key:value field -> one line `**label** value` (converts to <strong>label</strong> value)
            out.append(f"**{lf['text']}** {leaves[i + 1]['text']}".strip())
            i += 2
        else:
            out.append(_md_text(lf))
            i += 1
    return "\n\n".join(out)


def _render_md_table(cells: list[dict]) -> str:
    rows: dict[int, dict[int, str]] = {}
    for c in cells:
        r = int(c.get("row") or 0)
        col = int(c.get("col") or 0)
        rows.setdefault(r, {})[col] = c["text"]
    ncols = max((max(cols) for cols in rows.values()), default=0) + 1
    lines = []
    for r in sorted(rows):
        cells_txt = [rows[r].get(col, "") for col in range(ncols)]
        lines.append("| " + " | ".join(cells_txt) + " |")
        if r == 0:
            lines.append("| " + " | ".join(["---"] * ncols) + " |")
    return "\n".join(lines)


def label_text_for_roundtrip(leaves: list[dict]) -> str:
    """All leaf text concatenated + normalized — used by roundtrip_check to compare against source."""
    return ku.normalize("".join(lf["text"] for lf in leaves))
