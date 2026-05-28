"""Validate Khmer glyph coverage of every font in data/fonts/ and write coverage_manifest.json.

Coverage of the base Khmer codepoints (consonants, vowels, signs, COENG) is *necessary* but not
*sufficient* — correct shaping (GSUB) is verified visually by validate/visual_qa.py. Fonts below a
coverage threshold are marked unusable so the sampler skips them.

    python -m src.fonts.validate_coverage [--min-coverage 0.95]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from ..corpus.common import CONFIG_DIR, PROJECT_ROOT

FONTS_DIR = PROJECT_ROOT / "data" / "fonts"
MANIFEST = FONTS_DIR / "coverage_manifest.json"

# the codepoints a usable Khmer font must cover (consonants, dep. vowels, key signs, coeng)
REQUIRED = (
    list(range(0x1780, 0x17A3))   # consonants
    + list(range(0x17B6, 0x17C6)) # dependent vowels
    + [0x17C6, 0x17C7, 0x17C8, 0x17C9, 0x17CA, 0x17CB, 0x17CC, 0x17D2]  # signs + coeng
    + list(range(0x17E0, 0x17EA)) # digits
)


def _covered_codepoints(font_path: Path) -> set[int]:
    from fontTools.ttLib import TTFont, TTCollection
    try:
        if font_path.suffix.lower() == ".ttc":
            fonts = TTCollection(str(font_path)).fonts
        else:
            fonts = [TTFont(str(font_path), fontNumber=0)]
    except Exception:
        return set()
    cps: set[int] = set()
    for ft in fonts:
        try:
            cps |= set(ft.getBestCmap().keys())
        except Exception:
            pass
    return cps


def validate(min_coverage: float) -> dict:
    cfg = yaml.safe_load((CONFIG_DIR / "fonts.yaml").read_text(encoding="utf-8"))
    weights = {f["name"]: f for f in cfg.get("fonts", [])}

    entries = []
    font_files = sorted(p for p in FONTS_DIR.glob("*.tt*") if p.suffix.lower() in (".ttf", ".otf", ".ttc"))
    if not font_files:
        print(f"No fonts in {FONTS_DIR}. Run: python -m src.fonts.fetch_fonts")
        return {}

    for fp in font_files:
        cps = _covered_codepoints(fp)
        covered = sum(1 for c in REQUIRED if c in cps)
        coverage = covered / len(REQUIRED)
        meta = weights.get(fp.stem, {})
        usable = coverage >= min_coverage
        entries.append({
            "name": fp.stem,
            "path": str(fp.relative_to(PROJECT_ROOT)),
            "family": meta.get("family", fp.stem),
            "difficulty": meta.get("difficulty", "unknown"),
            "weight": float(meta.get("weight", 1.0)),
            "coverage": round(coverage, 4),
            "usable": usable,
        })

    usable_n = sum(e["usable"] for e in entries)
    MANIFEST.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'FONT':28} {'DIFF':8} {'COV':>6} {'WEIGHT':>7} USABLE")
    print("-" * 64)
    for e in entries:
        print(f"{e['name']:28} {e['difficulty']:8} {e['coverage']*100:5.1f}% {e['weight']:7.2f} "
              f"{'yes' if e['usable'] else 'NO'}")
    print(f"\n{usable_n}/{len(entries)} fonts usable (>= {min_coverage:.0%}). -> {MANIFEST}")
    return {"entries": entries, "usable": usable_n}


def load_usable_fonts() -> list[dict]:
    """Used by the layout sampler. Falls back to raw font files if no manifest yet."""
    if MANIFEST.exists():
        entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
        return [e for e in entries if e.get("usable")]
    files = sorted(p for p in FONTS_DIR.glob("*.tt*") if p.suffix.lower() in (".ttf", ".otf", ".ttc"))
    return [{"name": p.stem, "path": str(p.relative_to(PROJECT_ROOT)), "family": p.stem,
             "difficulty": "unknown", "weight": 1.0, "coverage": None, "usable": True} for p in files]


def main() -> None:
    ap = argparse.ArgumentParser(description="validate Khmer font coverage")
    ap.add_argument("--min-coverage", type=float, default=0.95)
    args = ap.parse_args()
    validate(args.min_coverage)


if __name__ == "__main__":
    main()
