# financial_statement

Dense, table-heavy financial page (balance sheet / income statement). The hardest `table_region` /
`table_head` / `table_cell` case: many rows, right-aligned numbers, total rows, ruled lines.

**Page geometry**
- Width ~1240px; margins 50–90px; 1 column.
- Body 13–17px (dense); tight line-height (1.3–1.5).

**Block sequence**
1. `title` — statement name ("របាយការណ៍លំហូរសាច់ប្រាក់").
2. `caption` — period/date line.
3. 1–3 `table_region`, each: a `table_head` row + many `table_cell` rows (3–15 rows, 2–6 cols),
   numeric right-aligned cells, occasional bold "សរុប/Total" row.
4. Short `text` notes / footnotes between tables.

**Real images used:** rarely — `logos` (letterhead) only.

**Diversity knobs:** number of tables, row/col counts, ruled vs zebra vs borderless tables,
Khmer vs Arabic digits, currency symbol (៛/$), nested subtotal rows.

**Sampling weight:** 1.5
