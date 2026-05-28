"""Sample a page: choose a template, font, typography, and assemble corpus text into a page model.

The page model is a plain dict consumed by render_playwright. Text comes from the LM corpus
(data/corpus/lm_corpus, else filtered/clean/raw); if no corpus exists yet, a small built-in Khmer
pool keeps the render smoke test self-contained.
"""
from __future__ import annotations

import random
from pathlib import Path

from ..corpus.common import (CLEAN_DIR, DATA, FILTERED_DIR, LM_DIR, RAW_DIR,
                             iter_jsonl_dir, load_yaml)
from .. import khmer_utils as ku

# fallback Khmer text so rendering works before any corpus is collected
_FALLBACK = [
    "ប្រទេសកម្ពុជាមានទីតាំងស្ថិតនៅភូមិភាគអាស៊ីអាគ្នេយ៍។",
    "ភាសាខ្មែរគឺជាភាសាផ្លូវការរបស់ប្រទេសកម្ពុជា។",
    "ទីក្រុងភ្នំពេញគឺជារាជធានីនៃព្រះរាជាណាចក្រកម្ពុជា។",
    "ប្រាសាទអង្គរវត្តគឺជាសំណង់ប្រវត្តិសាស្ត្រដ៏ល្បីល្បាញ។",
    "ការអប់រំមានសារៈសំខាន់សម្រាប់ការអភិវឌ្ឍន៍ប្រទេសជាតិ។",
    "សេដ្ឋកិច្ចកម្ពុជាកំពុងមានកំណើនជារៀងរាល់ឆ្នាំ។",
    "បច្ចេកវិទ្យាព័ត៌មានបានផ្លាស់ប្ដូររបៀបរស់នៅរបស់មនុស្ស។",
    "ទេសចរណ៍គឺជាវិស័យសំខាន់មួយក្នុងការរកចំណូលរបស់ប្រទេស។",
]

_LABELS = ["ឈ្មោះ", "អាសយដ្ឋាន", "ថ្ងៃខែឆ្នាំកំណើត", "លេខទូរស័ព្ទ", "ភេទ",
           "សញ្ជាតិ", "មុខរបរ", "លេខអត្តសញ្ញាណប័ណ្ណ"]
_VALUES = ["សុខ សំណាង", "ភ្នំពេញ", "០១/០១/២០០០", "០១២ ៣៤៥ ៦៧៨", "ប្រុស",
           "ខ្មែរ", "គ្រូបង្រៀន", "១២៣៤៥៦៧៨៩"]


class CorpusText:
    """Yields normalized Khmer sentences/paragraphs from the best available corpus stage."""

    def __init__(self, seed: int = 0, quiet: bool = False, max_docs: int = 20000):
        self.rng = random.Random(seed)
        self.pool: list[str] = []
        for d in (LM_DIR, FILTERED_DIR, CLEAN_DIR, RAW_DIR):
            if d.exists() and any(d.glob("*.jsonl")):
                for row in iter_jsonl_dir(d):
                    t = row.get("text", "")
                    if t:
                        self.pool.append(t)
                    if len(self.pool) >= max_docs:
                        break
                if self.pool:
                    if not quiet:
                        print(f"[sampler] text pool from {d.name}: {len(self.pool)} docs")
                    break
        if not self.pool:
            if not quiet:
                print("[sampler] no corpus found; using built-in fallback Khmer pool")
            self.pool = list(_FALLBACK)

    def sentence(self) -> str:
        doc = self.rng.choice(self.pool)
        sents = ku.split_sentences(doc) or [doc]
        return self.rng.choice(sents)

    def paragraph(self, n_sent: int) -> str:
        return "".join(self.sentence() for _ in range(n_sent))


def _weighted_choice(rng: random.Random, items: list, weights: list[float]):
    return rng.choices(items, weights=weights, k=1)[0]


def sample_page(rng: random.Random, corpus: CorpusText, layouts: dict,
                fonts: list[dict]) -> dict:
    tmpls = layouts["templates"]
    template = _weighted_choice(rng, [t["name"] for t in tmpls], [t["weight"] for t in tmpls])
    font = _weighted_choice(rng, fonts, [f.get("weight", 1.0) for f in fonts])

    typ = layouts["typography"]
    body_px = rng.randint(*typ["body_font_px"])
    scale = rng.uniform(*typ["heading_scale"])
    pg = layouts["page"]
    margin = rng.randint(*pg["margin_px"])
    white = rng.random() < pg["background"]["white_prob"]
    page_bg = "#ffffff" if white else rng.choice(["#fbf7ee", "#f4f4f4", "#fcfbf7"])

    page = {
        "template": template,
        "font_name": font["name"],
        "font_path": str((DATA.parent / font["path"]).resolve()) if "path" in font else None,
        "width": pg["width_px"],
        "margin": margin,
        "bg": "#dcdcdc" if not white else "#ffffff",
        "page_bg": page_bg,
        "body_px": body_px,
        "title_px": int(body_px * scale),
        "header_px": int(body_px * (1 + (scale - 1) * 0.6)),
        "line_height": round(rng.uniform(*typ["line_height"]), 2),
        "letter_spacing": round(rng.uniform(*typ["letter_spacing_px"]), 2),
        "color": rng.choice(typ["text_color_dark"]),
        "align": rng.choice(typ["align"]),
        "columns": 2 if template == "article_multicol" else 1,
        "blocks": [],
    }

    c = layouts["content"]
    blocks: list[dict] = []
    if template in ("article_single", "article_multicol", "mixed_km_en"):
        blocks.append({"type": "title", "text": corpus.sentence()})
        blocks.append({"type": "byline", "text": "ដោយ អ្នកនិពន្ធ · " + rng.choice(["ភ្នំពេញប៉ុស្តិ៍", "ព័ត៌មាន", "សារព័ត៌មាន"])})
        n_par = rng.randint(*c["paragraphs_per_article"])
        for i in range(n_par):
            if i > 0 and rng.random() < 0.4:
                blocks.append({"type": "section_header", "text": corpus.sentence()})
            text = corpus.paragraph(rng.randint(*c["sentences_per_paragraph"]))
            if template == "mixed_km_en" and rng.random() < 0.6:
                text += f" ({rng.choice(['GDP', 'COVID-19', 'AI', 'USD', '2026'])})"
            blocks.append({"type": "text", "text": text})
        if rng.random() < 0.3:
            blocks.append({"type": "list", "items": [corpus.sentence() for _ in range(rng.randint(3, 5))]})
    elif template == "doc_with_table":
        blocks.append({"type": "title", "text": corpus.sentence()})
        blocks.append({"type": "text", "text": corpus.paragraph(rng.randint(2, 4))})
        n_rows = rng.randint(*c["table"]["rows"])
        n_cols = rng.randint(*c["table"]["cols"])
        header = [corpus.sentence()[:12] for _ in range(n_cols)] if rng.random() < c["table"]["header_prob"] else None
        rows = [[corpus.sentence()[:14] for _ in range(n_cols)] for _ in range(n_rows)]
        blocks.append({"type": "table", "tid": 0, "header": header, "rows": rows})
        blocks.append({"type": "text", "text": corpus.paragraph(rng.randint(2, 4))})
    elif template == "form_label_value":
        blocks.append({"type": "title", "text": rng.choice(["ពាក្យសុំ", "ប័ណ្ណសម្គាល់ខ្លួន", "ទម្រង់ព័ត៌មាន"])})
        n = rng.randint(*c["form"]["fields"])
        idx = list(range(len(_LABELS)))
        rng.shuffle(idx)
        fields = [{"label": _LABELS[i % len(_LABELS)], "value": rng.choice(_VALUES)} for i in idx[:n]]
        blocks.append({"type": "form", "fields": fields})

    page["blocks"] = blocks
    return page


def expected_leaf_texts(page: dict) -> list[str]:
    """Leaf texts in the exact order the template emits them (== DOM order).

    Used by build_dataset to store source_text for the round-trip check.
    """
    out: list[str] = []
    for b in page["blocks"]:
        t = b["type"]
        if t in ("title", "byline", "section_header", "text"):
            out.append(b["text"])
        elif t == "list":
            out.extend(b["items"])
        elif t == "table":
            if b.get("header"):
                out.extend(b["header"])
            for row in b["rows"]:
                out.extend(row)
        elif t == "form":
            for f in b["fields"]:
                out.append(f["label"])
                out.append(f["value"])
    return out
