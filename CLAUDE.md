# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A **dataset-preparation pipeline** for Khmer OCR. It produces two deliverables, both on this Mac (CPU only):

1. **Khmer LM text corpus** (`data/corpus/lm_corpus/`) — broadly sourced, normalized, deduped,
   quality-filtered, license-tracked. Reusable to train a future Khmer LLM, and the text source for #2.
2. **Structured-layout OCR dataset** (`data/synthetic/` + `data/manifests/`) — synthetic document
   images with **grounded-markdown** labels in DeepSeek-OCR-2's `<|grounding|>` format, for fine-tuning.

**Training/fine-tuning is explicitly OUT OF SCOPE here** — it happens later on a separate Linux box
(3× A5000). There is no GPU on this machine; don't add training code. `khmer-ocr-findings.html` is the
research writeup behind the design decisions.

## Environment & commands

Always use the venv interpreter and run modules with `-m` from the repo root:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && python -m playwright install chromium

# --- Corpus pipeline (chain; each reads the previous stage's output dir) ---
./.venv/bin/python -m src.corpus.registry --list                 # show sources + licenses + enabled
./.venv/bin/python -m src.corpus.registry --fetch --limit 40000  # --smoke = cap 200/src
./.venv/bin/python -m src.corpus.clean_normalize                 # raw/ -> clean/
./.venv/bin/python -m src.corpus.dedup                            # clean/ -> dedup/
./.venv/bin/python -m src.corpus.quality_filter                  # dedup/ -> filtered/ (+ rejected audit)
./.venv/bin/python -m src.corpus.package_lm                      # filtered/ -> lm_corpus/ + dataset_card.md
# run_corpus.sh chains all of the above.

# --- Fonts (prereq for rendering) ---
./.venv/bin/python -m src.fonts.fetch_fonts                       # -> data/fonts/
./.venv/bin/python -m src.fonts.validate_coverage                 # -> coverage_manifest.json (+ usable flag)

# --- Render synthetic OCR set (samples text from lm_corpus automatically) ---
./.venv/bin/python src/build_dataset.py --n 15000 --workers 6 --image-format jpg --dsf 1 --min-free-gb 3
# run_render.sh wraps the full-scale invocation.

# --- Validation gates (run after build; these are the QA story) ---
./.venv/bin/python -m src.validate.roundtrip_check   # text fidelity (MUST be 100%)
./.venv/bin/python -m src.validate.box_sanity        # boxes in-bounds / ordered
./.venv/bin/python -m src.validate.stats_report      # font/template/blocktype distributions
./.venv/bin/python -m src.validate.visual_qa         # -> data/manifests/contact_sheet.html (human eyeball)
./.venv/bin/python -m src.khmer_utils                # self-test the Khmer normalizer
```

There is **no test framework**; correctness is verified by the validation gates above (they run over the
generated labels) and by the `--smoke`/`--n 8` small runs.

## Architecture (the big picture)

Two pipelines joined by `data/corpus/lm_corpus/`:

```
sources.yaml ─▶ registry.py ─▶ fetch_hf / fetch_dumps / scrape ─▶ raw/
   ─▶ clean_normalize ─▶ dedup ─▶ quality_filter ─▶ package_lm ─▶ lm_corpus/   [LM dataset]
lm_corpus/ ─▶ layout_sampler (text) + fonts/ + layouts.yaml
   ─▶ render_playwright ─▶ augment ─▶ to_grounded_markdown ─▶ synthetic/ + manifests/   [OCR dataset]
```

- **Everything is config-driven** by `config/*.yaml` (`sources`, `fonts`, `layouts`, `augment`). Change
  behavior there, not in code. `fonts.yaml` deliberately **up-weights hard decorative fonts** (Bayon,
  Dangrek, iSeth First…) because that's where OCR fails.
- **Provenance is carried end-to-end** via the `Doc` dataclass and JSONL shards (`src/corpus/common.py`:
  `Doc`, `ShardWriter`, `iter_jsonl_dir`, `doc_from_row`, path constants). Each corpus stage writes a new
  dir and reads the previous one; intermediates (`clean/ dedup/ filtered/`) are regenerable — only
  `raw/` and `lm_corpus/` are worth keeping.
- **`src/khmer_utils.py` is correctness-critical.** `normalize()` canonicalizes Khmer combining-mark
  order (coeng/vowels) and strips zero-width controls; it must stay **idempotent** and is applied to
  **both** rendered text and labels so the round-trip holds. `khmer_ratio()` drives script-purity
  filtering.
- **Rendering = HTML/CSS → Playwright(Chromium) → DOM boxes** (`render_playwright.py`). Chromium is used
  *specifically because it shapes Khmer correctly* (HarfBuzz); bitmap renderers (plain Pillow) do not.
  The chosen font is **base64-embedded as an `@font-face` data URL** so it's guaranteed to apply
  regardless of page origin. Bounding boxes are read from the DOM (`getBoundingClientRect` + Range line
  rects), giving perfect image↔label alignment with zero manual annotation. **Reading order = DOM order.**
- **Label format** (`to_grounded_markdown.py`): one grounded ref per leaf, in DOM order —
  `<|ref|>{markdown text}<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>`, coords normalized to **[0,999]**. This
  is the single training-target format and matches DeepSeek-OCR-2's grounding prompt
  `<image>\n<|grounding|>Convert the document to markdown.` Tables are reconstructed from `data-tbl/row/col`
  attributes. Box granularity is **line/block-level** (word boxes are ill-defined in spaceless Khmer).
- **The two mandatory QA gates exist for specific failure modes:** `roundtrip_check` catches
  Unicode-ordering bugs and text loss (fed-in text must equal text decoded from the label after
  normalize); `visual_qa` catches **shaping bugs and `.notdef` boxes** that loss curves never reveal.
  Run both before any large generation.

## Non-obvious gotchas (these have already bitten)

- **`datasets>=4` dropped loading-script support.** Script-based HF datasets (cc100, mc4, MADLAD-400,
  OSCAR-2301) **cannot load** — only Parquet sources work (FineWeb-2 `khm_Khmr`, Glot500 `khm_Khmr`,
  `wikimedia/wikipedia`). Gated sets (OSCAR, CulturaX) need `huggingface-cli login`. `sources.yaml`
  records which are disabled and why.
- **Disk is the binding constraint** (this machine runs ~10 GiB free). Always render as **JPEG with
  `--dsf 1`** (PNG at dsf=2 is ~10× larger) and keep the `--min-free-gb` guard, which hard-stops
  generation before filling the disk.
- **Launching long background jobs:** run the script *as* the background command (e.g.
  `./run_render.sh > log 2>&1`). Do **not** `nohup … &` *inside* a backgrounded launcher — the detached
  child gets reaped when the launcher task completes. Set `PYTHONUNBUFFERED=1` or logs won't flush.
- **Jinja templates:** a dict key named `items` collides with `dict.items` — use `b['items']`, not
  `b.items`, in `templates/page.html.j2`.
- **macOS specifics:** no `timeout` command; `fetch_fonts.py` seeds the system Khmer font
  (`Khmer Sangam MN`) as a guaranteed fallback so rendering works offline.
- **Parallel render** (`build_dataset.py`) uses `ProcessPoolExecutor` (spawn); each worker owns its own
  Chromium and a contiguous index range. The worker fn and all args must stay picklable.
- `source_text` in each label is stripped **per leaf** to match the DOM's `textContent.trim()` — keep
  these two in sync or `roundtrip_check` will flag harmless edge-whitespace as failures.
