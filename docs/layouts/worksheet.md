# worksheet

Primary-school worksheet: numbered questions with dotted fill-in leaders, word-bank bubbles, and
captioned picture prompts. A known V1/V2 failure case (front-word cutoff, image detection).

**Page geometry**
- Width ~1200px; margins 60–110px; 1 column.
- Body 18–26px (large, child-friendly); generous line-height.

**Block sequence**
1. `title` — "លំហាត់ / មេរៀនទី N".
2. Repeating sections, each:
   - `heading` "ក. …" item-group header.
   - Optional word bank: a `list_item` row of rounded **bubbles** (or one centered long bubble).
   - 2–4 numbered `text` questions, each ending with a dotted fill-in leader ("…………… ។").
3. Optional 1–2 picture prompts: real `image` (or `hand_drawing`) + a `caption` with a fill-in.

**Real images used:** `photos`, `drawings` (→ `hand_drawing`), simple object photos.

**Diversity knobs:** bubble row vs long bubble, question count, leader style, picture-prompt
presence/count, large decorative title font.

**Sampling weight:** 2.0
