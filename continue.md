# CONTINUE.md — Khmer OCR dataset, session handoff

_Last updated 2026-05-28. Read `CLAUDE.md` first for architecture + commands + gotchas._

## TL;DR — where we are

The **dataset-prep pipeline is fully built and verified end-to-end**, and a **full-scale run has been
produced**. Two deliverables exist on this Mac. Training is still out of scope (separate Linux A5000 box).
The main open item is the **real test set** (needs documents the user supplies).

## What's done (verified, all gates green)

- **Pipeline code** complete in `src/` (corpus chain, fonts, render, validate, eval orchestrator) +
  `config/*.yaml`. Dependencies installed in `.venv/`; Chromium installed.
- **Deliverable 1 — Khmer LM corpus** → `data/corpus/lm_corpus/` (680 MB, 10 shards + `dataset_card.md`)
  - **151,044 docs · 116.2M chars · ~38.7M tokens**. Sources: Wikipedia-km (73K), FineWeb-2-km (40K),
    Glot500-km. Normalized, deduped, quality-filtered.
- **Deliverable 2 — Synthetic OCR dataset** → `data/synthetic/` (7 GB) + `data/manifests/`
  - **15,000 pages** (JPEG, dsf=1) + grounded-markdown labels (~217K boxes), 15 fonts (hard fonts
    up-weighted), 5 templates. Manifests: **train 14,250 / val 750 / test 0**.
- **Gates:** roundtrip 100% (15000/15000), box_sanity 100% clean, stats OK,
  visual QA → `data/manifests/contact_sheet.html`.
- Disk: ~12 GiB free.

## Current state caveats (important before re-running)

- Corpus **intermediates were cleared to reclaim disk**: `data/corpus/{clean,dedup,filtered}/` are
  **empty**. `raw/` (385 MB) and `lm_corpus/` (the deliverable) are intact. To re-run
  `quality_filter`/`package_lm` you must first re-run `clean_normalize` → `dedup` from `raw/`
  (or just re-run the whole `./run_corpus.sh`).
- `data/real/` is **empty** — no real test set yet (`test.jsonl` has 0 rows; this is expected).
- Two gated sources (OSCAR, CulturaX) are disabled in `config/sources.yaml` — no HF token configured.

## Open items / next steps (priority order)

1. **Build the real eval/test set** (highest value — it's the only trusted accuracy metric).
   - User drops Khmer docs (`.png/.jpg/.pdf`) into `data/real/raw_documents/`.
   - `./.venv/bin/python -m src.eval.collect_real --ingest`  (PDF→image needs `pip install pymupdf`)
   - Label in Label Studio per `src/eval/labeling_guide.md` (line-level boxes; block types).
   - `./.venv/bin/python -m src.eval.collect_real --from-export export.json`  → `data/real/labels/`
   - Re-run `build_dataset.py` (or just `write_manifests`) so `test.jsonl` picks them up.

2. **Scale up if disk allows.** Current run was capped by ~10 GiB free disk (15K pages / 40K docs-per-src).
   To go bigger: free disk or point output to an external drive, then
   `./.venv/bin/python src/build_dataset.py --n 100000 --workers 6 --image-format jpg --dsf 1 --min-free-gb 3`.
   For more corpus: `huggingface-cli login`, re-enable OSCAR/CulturaX in `sources.yaml`, raise `--limit`,
   re-run `./run_corpus.sh`.

3. **Optional corpus breadth:** `leipzig_khm_news` and `khpos` didn't produce shards last run (download/
   load failed) — investigate their fetchers if more sources are wanted. Add Khmer news scraping by
   putting real RSS/feed URLs in the `news_rss` source and enabling it (honor robots.txt).

4. **Hand off to training (separate effort, Linux 3× A5000).** Copy `data/corpus/lm_corpus/` and
   `data/manifests/` + `data/synthetic/` to the GPU box. There, run tokenizer-fertility + zero-shot
   baseline on DeepSeek-OCR-2, then LoRA fine-tune via Unsloth. See plan §"Out of scope" and
   `khmer-ocr-findings.html`. **Do not add training code on this Mac (no GPU).**

## Things to NOT redo / known-good decisions

- Render via HTML/CSS→Playwright(Chromium) with base64-embedded fonts; labels = DeepSeek `<|grounding|>`
  grounded markdown, coords [0,999], line-level boxes. (See CLAUDE.md.)
- Only Parquet HF datasets load (`datasets>=4` dropped scripts) — FineWeb-2/Glot500/Wikipedia are the
  working web sources.
- Always JPEG + `--dsf 1` + disk guard; launch long jobs *as* the background command (not `nohup &`).

## Key references

- `CLAUDE.md` — architecture, commands, gotchas.
- `~/.claude/plans/can-u-make-a-linked-papert.md` — the approved plan.
- `khmer-ocr-findings.html` — research writeup (model choice, Khmer pitfalls, dataset recipe).
- `run_corpus.sh` / `run_render.sh` — full-pipeline runbook scripts.
- Memory: `khmer-ocr-project`, `khmer-ocr-decisions` (project context + locked decisions).
