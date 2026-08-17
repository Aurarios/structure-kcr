# textbook_figures

School/university textbook page: dense body interleaved with **multiple** captioned figures,
diagrams and formulas. Trains many `image`/`hand_drawing`/`formula` regions per page alongside text.

**Page geometry**
- Width ~1150px; margins 70–120px; 1 column (65%) or 2 columns (35%).
- Body 15–20px.

**Block sequence**
1. `heading` — lesson/unit title; optional `subheading`.
2. Repeating: `text` paragraph → (often) a figure block:
   - real `image` (photo/diagram) **or** `hand_drawing` (sketch) + `caption`.
3. 1–3 `formula` lines.
4. 0–1 `table_region` (data table) with `table_head`+`table_cell`.
5. Optional `list_item` "exercise" bullets.

**Real images used:** `drawings` (→ `hand_drawing`), `photos`, `charts`, occasional `portraits`.

**Diversity knobs:** figure count (1–4), drawing vs photo mix, formula count, 1/2 column,
figure float left/right vs full width.

**Sampling weight:** 1.5
