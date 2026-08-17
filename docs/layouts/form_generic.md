# form_generic

Generic administrative form: label:value fields, checkboxes, fill-in lines, a signature line.
Trains `form_label`/`form_value` and `signature`.

**Page geometry**
- Width ~1240px; margins 60–110px; 1 column (occasional 2-column field grid).
- Body 15–20px.

**Block sequence**
1. `title` — form name ("ពាក្យសុំ…").
2. Optional `text` instruction line.
3. 5–12 `form_label` + `form_value` pairs (value may be a dotted fill-in leader or boxed).
4. Optional small `table_region` (e.g. dependents list).
5. Checkbox rows rendered as `list_item` ("☐ បាទ/ចាស  ☐ ទេ").
6. A `form_label` "ហត្ថលេខា" + a real `signature` image at the bottom; optional `stamp` → `image`.

**Real images used:** `signatures` (→ `signature`), `stamps` (→ `image`), `logos` (header).

**Diversity knobs:** field count, 1 vs 2 column field grid, dotted vs boxed values, checkbox rows,
signature/stamp presence and position.

**Sampling weight:** 1.5
