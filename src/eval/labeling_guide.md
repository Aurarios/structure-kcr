# Real Khmer OCR Eval Set — Labeling Guide

This set is the **only trusted measure** of real-world accuracy and is **held out** from all
training. Target **200–500 pages** spanning the in-scope doc types (news, articles, tables, forms,
mixed Khmer/English). Quality over quantity — a few hundred carefully labeled pages beat thousands
of sloppy ones.

## Workflow

1. Put source files in `data/real/raw_documents/` (`.png/.jpg/.pdf`).
2. `python -m src.eval.collect_real --ingest` → page images in `data/real/images/` +
   `data/real/labelstudio_tasks.json`.
3. Label in [Label Studio](https://labelstud.io) using the config below.
4. Export as **JSON**, then `python -m src.eval.collect_real --from-export export.json`
   → `data/real/labels/*.json`.
5. `python src/build_dataset.py ...` picks these up automatically as the **test** split.

## Label Studio config

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="bbox" toName="image">
    <Label value="title"/>
    <Label value="section_header"/>
    <Label value="text"/>
    <Label value="list_item"/>
    <Label value="table_cell"/>
    <Label value="form_label"/>
    <Label value="form_value"/>
    <Label value="caption"/>
  </RectangleLabels>
  <TextArea name="text" toName="image" perRegion="true" editable="true"
            placeholder="Transcribe Khmer text for this box"/>
</View>
```

Each region = one rectangle (block type) + its transcription. The converter pairs them by region id
and emits boxes normalized to `[0,999]`.

## Boxing rules

- **Line-level boxes** for flowing text. Khmer has **no inter-word spaces**, so do **not** draw word
  boxes — box whole lines (or logical blocks for titles/headers).
- One box per table cell (`table_cell`); one box per form label and per form value.
- Tight boxes: include all stacked/subscript glyphs (coeng sits *below* the base — don't clip it).
- Reading order is recovered automatically (top-to-bottom, then left-to-right); for multi-column
  pages, label down one column then the next.

## Transcription rules

- Transcribe **exactly what is printed**, including English words/numbers and Khmer digits
  (០១២៣៤៥៦៧៨៩).
- Type in correct Khmer encoding order (base → subscripts → vowels → signs). The pipeline
  re-normalizes, but enter clean text.
- Do not silently "correct" the document's spelling; transcribe as printed.
- Mark illegible spans with `[?]` and skip the region if a whole line is unreadable.

## Output format (per page → `data/real/labels/<id>.json`)

```json
{
  "id": "real_0007",
  "image": "data/real/images/real_0007.png",
  "blocks": [{"block_type": "title", "text": "…", "bbox": [x1, y1, x2, y2]}],
  "grounded": "<|ref|># …<|/ref|><|det|>[[..]]<|/det|> …",
  "markdown": "# …",
  "source": "real"
}
```
`bbox` is normalized to `[0,999]`. `grounded` is the DeepSeek-OCR-2-style fine-tune target.
