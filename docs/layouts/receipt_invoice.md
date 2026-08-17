# receipt_invoice

Receipt / invoice: key:value header, an items table, totals, and often a stamp. Trains compact
`table_region` + `form_label`/`form_value` + `image`(stamp/logo) together.

**Page geometry**
- Narrow (width ~700–900px, receipt-like) or A4 invoice (width ~1100px); margins 30–80px;
  1 column.
- Body 14–20px; monospace-ish allowed.

**Block sequence**
1. Optional `image` — shop `logo` (top, centered).
2. `title` — "វិក្កយបត្រ / បង្កាន់ដៃ" + shop name.
3. 2–4 `form_label`/`form_value` header rows (invoice no, date, customer).
4. `table_region`: `table_head` ("ទំនិញ | ចំនួន | តម្លៃ") + 3–10 `table_cell` rows.
5. `form_label`/`form_value` totals ("សរុប … ៛", "ពន្ធ", "សរុបចុងក្រោយ").
6. Optional `signature` and/or real `stamp` → `image`; `caption` thank-you footer.

**Real images used:** `logos`, `stamps`→`image`, `signatures`.

**Diversity knobs:** receipt vs invoice width, item count, tax/discount rows, stamp/signature
presence, Khmer vs Arabic digits, currency.

**Sampling weight:** 1.5
