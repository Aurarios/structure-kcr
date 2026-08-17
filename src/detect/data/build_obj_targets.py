"""V4 object-detection layout targets — region/block boxes + classes from existing V3 labels.

Unlike DBNet (segmentation, per-text-line), the V4 detector is a box-regression object detector that
works at REGION/BLOCK level and allows overlapping/nested boxes. This module turns a V3 label's
per-leaf `bbox` (block, image-pixel space) into `(box, class_id)` detection targets, and synthesizes
one enclosing `table` region box per table (union of its cells) — impossible to represent with the
DBNet prob-map but native to an object detector.

Boxes are returned in IMAGE-PIXEL space (the obj dataset scales them through the letterbox, exactly
like `build_det_targets.collect_line_boxes`).
"""
from __future__ import annotations

from typing import Any

# V4 taxonomy (object detection — NO background class; "no detection" = background).
# `table` is the enclosing region; `chart` is split from `image` (we fetch ChartQA separately).
CLASSES_V4 = [
    "title", "heading", "subheading", "text", "list_item", "caption",
    "table", "table_head", "table_cell",
    "image", "chart", "signature", "hand_drawing", "formula",
    "form_label", "form_value",
]
CLS2ID_V4 = {c: i for i, c in enumerate(CLASSES_V4)}
NUM_CLASSES_V4 = len(CLASSES_V4)

# region types cropped at inference (not sent to the text recognizer)
NONTEXT_V4 = {"image", "chart", "signature", "hand_drawing", "formula"}

# Text-bearing classes detected at LINE level (one box per visual line, from the label's `lines`).
# Block-level boxes fuse across small gaps (a text paragraph swallows the centered heading below it)
# and force the recognizer to read multi-line crops. Detecting lines directly fixes both. Everything
# else (figures/formulas/table cells) stays a REGION box — those are genuinely areas, not lines.
LINE_LEVEL_V4 = {"title", "heading", "subheading", "text", "list_item", "caption",
                 "form_label", "form_value"}

# legacy / synonym block_type -> canonical V4 class (covers frozen V1/V2/V3 labels too)
_ALIAS_V4 = {
    "section_header": "heading", "byline": "caption", "form": "form_value", "list": "list_item",
    "figure": "image", "photo": "image", "logo": "image", "stamp": "image", "seal": "image",
    "table_region": "table",
}

# --- V6 taxonomy (additive; V4 ids 0..15 are unchanged so V4/V5 checkpoints still load) ---------
# `word_bank` splits the SINGLE-CONTAINER word row (one rounded rect holding many space-separated
# words) away from `list_item` (one rounded bubble per word). Both rendered identically as
# `list_item` in V5, which taught the detector two contradictory targets for one visual pattern:
# measured on real worksheets it hedged at 0.11-0.59 confidence and emitted competing whole-row
# boxes. The ambiguity is WITHIN a layout (8/30 sampled v5 worksheet pages contain both patterns,
# sometimes on the same page), so no amount of layout conditioning resolves it — only distinct
# labels do.
CLASSES_V6 = CLASSES_V4 + ["word_bank"]
CLS2ID_V6 = {c: i for i, c in enumerate(CLASSES_V6)}
NUM_CLASSES_V6 = len(CLASSES_V6)
NONTEXT_V6 = set(NONTEXT_V4)
LINE_LEVEL_V6 = LINE_LEVEL_V4 | {"word_bank"}
_ALIAS_V6 = dict(_ALIAS_V4)

# Page-level layout vocabulary for the detector's AUXILIARY classification head (multi-task, not a
# cascade): the head shapes shared backbone features with page context at zero inference cost and
# with no error propagation. `layout_type` is already carried on every manifest row.
LAYOUTS = ["book_page", "business_report", "certificate", "composed", "contract_legal",
           "exam_paper", "financial_statement", "form_generic", "id_card", "letter_memo",
           "magazine_multicol", "news_article", "passport_mrz", "receipt_invoice",
           "scientific_paper", "textbook_figures", "worksheet"]
LAYOUT2ID = {n: i for i, n in enumerate(LAYOUTS)}
NUM_LAYOUTS = len(LAYOUTS)

_TAXONOMIES = {
    "v4": (CLASSES_V4, CLS2ID_V4, LINE_LEVEL_V4, _ALIAS_V4),
    "v6": (CLASSES_V6, CLS2ID_V6, LINE_LEVEL_V6, _ALIAS_V6),
}

Box = tuple[float, float, float, float]


def class_id(block_type: str | None) -> int | None:
    """V4 class id for a block_type, or None if it should be skipped (unknown / no box)."""
    bt = _ALIAS_V4.get(block_type or "", block_type or "")
    return CLS2ID_V4.get(bt)


def _valid(box: Box) -> bool:
    return box[2] - box[0] >= 2 and box[3] - box[1] >= 2


def collect_obj_boxes(label: dict[str, Any], line_level: bool = True,
                      taxonomy: str = "v4") -> list[tuple[Box, int]]:
    """(box, class_id) detection targets in image-pixel space from a V3 label.

    line_level=True (default): text-bearing classes (LINE_LEVEL_*) emit one box per visual line
    from the block's `lines` (already stored in the label, image-pixel space) — clean single-line
    detection, no block fusion. Region classes (figures/formulas/table cells) keep their `bbox`.
    line_level=False: legacy block-level (every class uses `bbox`) — what the first V4 detector used.

    taxonomy: "v4" (16 classes, frozen — V4/V5 checkpoints) or "v6" (adds `word_bank`). A v5 label
    carrying `word_bank` blocks read under "v4" simply drops them (class_id -> None), so old
    pipelines stay valid against new data.

    A `table` region box is synthesized per `tbl` group (union of its cells) either way.
    """
    classes, cls2id, line_level_set, alias = _TAXONOMIES[taxonomy]
    out: list[tuple[Box, int]] = []
    tables: dict[Any, list[Box]] = {}
    for blk in label.get("blocks", []):
        bt = blk.get("block_type")
        cid = cls2id.get(alias.get(bt or "", bt or ""))
        bb = blk.get("bbox")
        if cid is None or not bb or len(bb) != 4:
            continue
        block_box = tuple(float(v) for v in bb)  # type: ignore[assignment]
        if not _valid(block_box):
            continue
        canon = alias.get(bt or "", bt or "")
        lines = blk.get("lines") or []
        if line_level and canon in line_level_set and lines:
            # one detection box per visual line (already in image-pixel space)
            for ln in lines:
                if not ln or len(ln) != 4:
                    continue
                lb = tuple(float(v) for v in ln)  # type: ignore[assignment]
                if _valid(lb):
                    out.append((lb, cid))
        else:
            out.append((block_box, cid))         # region class (or single-line / no `lines`)
        if bt in ("table_head", "table_cell"):
            tables.setdefault(blk.get("tbl"), []).append(block_box)
    # one enclosing `table` region per table (union of its cells)
    for cells in tables.values():
        if len(cells) < 2:
            continue
        rx1 = min(c[0] for c in cells); ry1 = min(c[1] for c in cells)
        rx2 = max(c[2] for c in cells); ry2 = max(c[3] for c in cells)
        if rx2 - rx1 >= 2 and ry2 - ry1 >= 2:
            out.append(((rx1, ry1, rx2, ry2), cls2id["table"]))
    return out


def layout_id(layout_type: str | None) -> int:
    """Layout index for the auxiliary head; unknown/missing -> `composed` (the catch-all mix)."""
    return LAYOUT2ID.get(layout_type or "", LAYOUT2ID["composed"])


if __name__ == "__main__":   # quick self-test on a rendered label
    import glob, json
    from collections import Counter
    files = glob.glob("E:/kcr-v3-smoke/*/labels/*.json")
    print(f"NUM_CLASSES_V4={NUM_CLASSES_V4} {CLASSES_V4}")
    cnt = Counter(); npages = 0; ntables = 0
    for f in files:
        d = json.load(open(f, encoding="utf-8")); npages += 1
        for box, cid in collect_obj_boxes(d):
            cnt[CLASSES_V4[cid]] += 1
            if cid == CLS2ID_V4["table"]:
                ntables += 1
    print(f"pages={npages} table-regions={ntables}")
    print("class box counts:", dict(sorted(cnt.items(), key=lambda x: -x[1])))
