# exam_paper

Formal exam / quiz paper: header block (school, subject, time, marks), numbered questions, multiple
-choice options, mark allocations. Denser and more formal than `worksheet`.

**Page geometry**
- Width ~1200px; margins 60–100px; 1 column.
- Body 16–22px.

**Block sequence**
1. `title` — institution / exam name; `subheading` — subject + duration + total marks.
2. Optional `form_label`/`form_value` candidate fields (Name, ID, Class) + a fill-in line.
3. `heading` — "ផ្នែកទី … (… ពិន្ទុ)" section header.
4. Numbered `text` questions ("N. …  (M ពិន្ទុ)"), each optionally followed by `list_item`
   choices ("ក) … ខ) … គ) …").
5. Occasional `formula`, `image` (a diagram the question refers to), or small `table_region`.

**Real images used:** `drawings` (→ `hand_drawing`), `charts`, `photos`.

**Diversity knobs:** section count, MCQ vs open questions, marks shown, diagram presence, candidate
field block presence.

**Sampling weight:** 1.5
