# news_article

Single- or two-column news/blog article. The most common "wall of text + one figure" page.

**Page geometry**
- Width ~1240px (A4 @150 DPI); margins 60–120px; 1 column (70%) or 2 columns (30%).
- Body 16–22px; generous line-height (1.5–1.9) for Khmer stacking.

**Block sequence**
1. `title` — headline (1 line, large).
2. `caption` — byline ("ដោយ … · <publication>") + optional date line.
3. 3–9 paragraphs of `text`; every few paragraphs a `heading` may break the flow.
4. 0–2 inline figures: real `image` + a `caption` below it ("រូបភាព៖ …").
5. Occasional `list_item` block (3–5 bullets).

**Real images used:** `photos` (charts occasionally) for the inline figure.

**Diversity knobs:** column count, figure count/position (top/inline/float-right), justified vs
left, presence of subheads, decorative font for headline, paper tint.

**Sampling weight:** 2.0
