# V3 Document Layout Specifications

These specs are the **source of truth** for V3 synthetic-document generation. Each layout type
below has its own spec file describing how content is arranged, which detector classes it exercises,
and which real-image categories it embeds. The synthetic generator (`src/render/layout_sampler.py`)
implements these specs, and the dataset is written to **per-layout-type directories** so each type
can be inspected, debugged, and scaled independently:

```
E:\kcr-v3\
  news_article\        images\*.jpg   labels\*.json
  magazine_multicol\   images\*.jpg   labels\*.json
  ...                  (one dir per layout type below)
```

## Why V3 exists

V1/V2 detectors converged in 1–3 epochs then failed on real documents. Root cause was the dataset:
"images" were CSS gradients (the detector learned "smooth blob = image", which does not transfer to
real photos), and layouts were a dozen near-identical hardcoded branches (trivially memorized). V3
fixes both: **real images** are embedded into figure/signature/stamp regions, and layouts are
diverse and spec-driven to mimic ~99% of real-world documents.

## Detector class taxonomy (15 detector classes incl. background; table_region via assembly)

The detector's golden rule is **precise structured output** — every region gets an accurate box and
the right class so the recognizer and assembler can rebuild structure.

| id | class | meaning | sent to recognizer? |
|----|-------|---------|---------------------|
| 0 | `background` | non-content | — |
| 1 | `title` | overall document title | yes |
| 2 | `heading` | section header (≈ H2) | yes |
| 3 | `subheading` | sub-section header (≈ H3) | yes |
| 4 | `text` | paragraph / body line | yes |
| 5 | `list_item` | bullet / numbered item / word-bank bubble | yes |
| 6 | `caption` | figure/table caption, byline | yes |
| 7 | `table_head` | header-row cell | yes |
| 8 | `table_cell` | body cell | yes |
| 9 | `image` | real photo / figure / logo / flag / chart | **no — cropped** |
| 10 | `signature` | handwritten signature region | **no — cropped** |
| 11 | `hand_drawing` | sketch / hand-drawn diagram | **no — cropped** |
| 12 | `formula` | math / equation region | **no — cropped** |
| 13 | `form_label` | key in a key:value field (eKYC/forms) | yes |
| 14 | `form_value` | value in a key:value field | yes |

`NONTEXT_CLASSES = {image, signature, hand_drawing, formula}` — at inference these are cropped and
returned as figures, not passed to the text recognizer.

**`table_region` ("table layout")** is NOT a detector pixel class. DBNet detects boxes from
connected components in the text-probability map; filling a whole-table region into that map would
bridge the gaps between cells and collapse the entire table into a single box, destroying cell
detection. Instead, `table_head`/`table_cell` carry `data-tbl/row/col`, and the table-layout
bounding box is reconstructed at **assembly time** as the union of cells sharing the same `data-tbl`
— giving a precise table-layout coordinate without sabotaging cell detection.

## Structured output format: class → Markdown → HTML

Every class maps to a **standardized Markdown construct** so labels (and the model's output) convert
losslessly to HTML via any GFM + math parser. Implemented in `src/render/to_grounded_markdown.py`
(labels) and mirrored in `src/pipeline/assemble.py` (inference output).

| class | Markdown | HTML |
|-------|----------|------|
| `title` | `# text` | `<h1>` |
| `heading` | `## text` | `<h2>` |
| `subheading` | `### text` | `<h3>` |
| `text` | `text` (paragraph) | `<p>` |
| `list_item` | `- text` | `<li>` inside `<ul>` |
| `caption` | `*text*` | `<figcaption>` / `<em>` |
| `table_head` / `table_cell` | GFM table (`\| … \|` + `\| --- \|`) | `<table><thead><th>` / `<tbody><td>` |
| `image` / `signature` / `hand_drawing` | `![<class>]()` (valid image syntax) | `<img class="<class>">` |
| `formula` | `$$ … $$` (display math) | math block (MathJax/KaTeX) |
| `form_label` + `form_value` | `**label** value` (paired) | `<strong>label</strong> value` |

Notes: a bare `![alt]` is literal text in CommonMark — image placeholders MUST include `()` to
become `<img>`. The grounded target (`<\|ref\|>…<\|/ref\|><\|det\|>[[box]]<\|/det\|>`) uses the same
per-leaf Markdown so detection + structure are learned together; the clean `markdown` field pairs
form fields and groups tables for direct HTML conversion.

## Real-image categories (from `E:\kcr-assets`, see `src/assets/fetch_assets.py`)

`photos` (article/book figures) · `portraits` (ID/eKYC faces) · `signatures` · `stamps` (seals) ·
`drawings` (→ `hand_drawing`) · `logos` (logos/flags/emblems) · `charts` (graphs/diagrams).
All license-clean (CC0 / public-domain / CC-BY) with provenance in `E:\kcr-assets\assets.jsonl`.

## Layout types (sampling weights tuned toward production pain points)

| layout | weight | one-line description |
|--------|:------:|----------------------|
| [news_article](news_article.md) | 2.0 | single/2-col news with headline, byline, body, inline figure |
| [magazine_multicol](magazine_multicol.md) | 1.5 | 2–3 column magazine, pull-quotes, multiple figures |
| [scientific_paper](scientific_paper.md) | 1.5 | abstract, numbered sections, figures, equations, references |
| [business_report](business_report.md) | 1.5 | headings, body, charts, tables, KPI callouts |
| [financial_statement](financial_statement.md) | 1.5 | dense multi-table financial grids, totals |
| [form_generic](form_generic.md) | 1.5 | label:value fields, checkboxes, signature line |
| [id_card](id_card.md) | 2.5 | eKYC: portrait, logo/flag, bilingual fields, signature, MRZ |
| [passport_mrz](passport_mrz.md) | 1.5 | passport page: portrait, fields, 2-line MRZ |
| [book_page](book_page.md) | 2.0 | chapter/heading, body, figure, page number/footer |
| [textbook_figures](textbook_figures.md) | 1.5 | dense text + multiple captioned figures + formulas |
| [worksheet](worksheet.md) | 2.0 | numbered questions, dotted fill-ins, word banks, picture prompts |
| [exam_paper](exam_paper.md) | 1.5 | header block, numbered questions, multiple-choice, marks |
| [receipt_invoice](receipt_invoice.md) | 1.5 | key:value header, items table, totals, stamp |
| [certificate](certificate.md) | 1.5 | centered titles, body, signature(s), official seal |
| [letter_memo](letter_memo.md) | 1.0 | letterhead/logo, date, salutation, body, signature |
| [contract_legal](contract_legal.md) | 1.5 | numbered clauses, sub-clauses, signature blocks, stamp |

Spec file conventions: each lists **Page geometry**, **Block sequence** (ordered, with classes),
**Real images used**, **Diversity knobs** (what is randomized), and **Sampling weight**.
