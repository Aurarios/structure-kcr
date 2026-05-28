# Khmer OCR Dataset Pipeline

Builds two datasets on this Mac (CPU only — **no training here**):

1. **Reusable Khmer LM text corpus** — broadly sourced, normalized, deduped, quality-filtered,
   license-tracked. Usable to train a future Khmer LLM, and as the text source for OCR rendering.
2. **Structured-layout OCR dataset** — synthetic document images rendered HTML/CSS → Playwright
   (Chromium, which shapes Khmer correctly), with bounding boxes read from the DOM and emitted as
   **grounded-markdown** labels in DeepSeek-OCR-2's `<|grounding|>` format, plus a small hand-labeled
   **real** evaluation set.

Training / fine-tuning DeepSeek-OCR-2 is a **separate later effort** on a Linux box (3× A5000 24GB).

See `khmer-ocr-findings.html` for the research writeup and `~/.claude/plans/` for the full plan.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
```

## Pipeline

```text
config/sources.yaml ──> src/corpus/registry.py ──> fetch_* ──> data/corpus/raw/
                                                  clean_normalize ──> data/corpus/clean/
data/corpus/clean ──> dedup ──> quality_filter ──> package_lm ──> data/corpus/lm_corpus/  [LM DATASET]
data/corpus/lm_corpus ──> render (sample text + fonts + layout) ──> data/synthetic/{images,labels}/
data/synthetic ──> validate gates ──> build_dataset.py ──> data/manifests/{train,val,test}.jsonl  [OCR DATASET]
data/real (hand-labeled) ──────────────────────────────────────┘ (test split)
```

## Commands

```bash
# Corpus
python -m src.corpus.registry --list                 # show configured sources + licenses
python -m src.corpus.registry --fetch --smoke        # small fetch into data/corpus/raw/
python -m src.corpus.clean_normalize                 # -> data/corpus/clean/
python -m src.corpus.dedup                            # exact + MinHash near-dup
python -m src.corpus.quality_filter                  # LM-grade heuristics
python -m src.corpus.package_lm --smoke              # -> data/corpus/lm_corpus/ + dataset card

# Fonts
python -m src.fonts.fetch_fonts                       # download Khmer fonts -> data/fonts/
python -m src.fonts.validate_coverage                 # coverage_manifest.json + auto-weights

# Synthetic OCR data
python src/build_dataset.py --smoke --n 200          # render 200 pages + manifests

# Validation gates (run before any large generation)
python -m src.validate.roundtrip_check
python -m src.validate.box_sanity
python -m src.validate.visual_qa                      # -> data/manifests/contact_sheet.html
python -m src.validate.stats_report
```

## Khmer correctness — two non-negotiable gates

- **Unicode/coeng ordering** (`roundtrip_check.py`): the text fed into a page must equal the text decoded
  from its label after normalization. Catches malformed ordering that silently produces wrong labels.
- **Shaping** (`visual_qa.py`): a contact sheet over every font × template, eyeballed for correct
  subscript placement and zero `.notdef` boxes. Plain bitmap renderers do **not** shape Khmer — that is
  why we render through Chromium.

## Layout / scope

Clean printed + rich structured docs (tables, forms, multi-column, mixed Khmer/English). Boxes are
**line-level** (word boxes are ill-defined in Khmer — no inter-word spaces). Configure everything in
`config/*.yaml`.
