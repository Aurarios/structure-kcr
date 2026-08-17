# passport_mrz

Passport data page. Like `id_card` but portrait-orientation page with a prominent two-line MRZ at
the bottom and a watermark-style background.

**Page geometry**
- Page aspect (portrait ~1:1.4); width ~1000px; margins 30–60px; tinted/guilloché background.

**Block sequence**
1. `title` — "លិខិតឆ្លងដែន / PASSPORT" + country.
2. `image` — real **portrait** photo (left).
3. 6–10 bilingual `form_label`/`form_value` (Type, Country Code, Passport No, Surname, Given
   names, Nationality, DOB, Sex, Place of birth, Date of issue/expiry, Authority).
4. `signature` — holder signature image.
5. `image` — small national emblem.
6. `text` — **two** MRZ lines (44 chars, uppercase + `<` fillers).

**Real images used:** `portraits`, `logos` (emblem), `signatures`.

**Diversity knobs:** field subset, background guilloché color, portrait box size, MRZ content,
slight skew.

**Sampling weight:** 1.5
