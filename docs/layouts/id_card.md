# id_card

eKYC / national ID / driving licence card. Highest-priority production layout. Trains real
`image` (portrait + logo/flag), `signature`, bilingual `form_label`/`form_value`, and an MRZ
`text` line — on a small, tightly-packed card canvas.

**Page geometry**
- Card aspect (landscape ~1.585:1); width ~1000px; small margins (24–48px); colored/patterned
  background common (not pure white).

**Block sequence**
1. `image` — issuer logo / national flag (top-left), small.
2. `title` — bilingual issuer line ("ព្រះរាជាណាចក្រកម្ពុជា / KINGDOM OF CAMBODIA").
3. `image` — real **portrait** photo (left), card-portrait sized.
4. 5–9 bilingual `form_label` + `form_value` rows (Name, Sex, DOB, Address, ID No, Expiry…),
   labels often "ខ្មែរ English", values mixed Khmer/Latin/digits.
5. `signature` — real signature image (bottom of field area).
6. `image` — QR / barcode block (right).
7. `text` — 1–2 MRZ lines (monospace uppercase + chevrons).

**Real images used:** `portraits` (face), `logos` (flag/emblem), `signatures`, occasional
`stamps`→`image`.

**Diversity knobs:** card vs licence vs student-ID title, field subset/order, background color/
pattern, portrait position (left/right), QR presence, MRZ 1 vs 2 lines, slight rotation/skew
(photographed-card realism).

**Sampling weight:** 2.5
