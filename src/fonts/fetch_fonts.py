"""Download Khmer fonts listed in config/fonts.yaml into data/fonts/.

Primary source: the google/fonts GitHub repo (OFL). Filenames are guessed from the family name;
failures are reported, not fatal. System Khmer fonts (macOS) are always copied in as a guaranteed
fallback so the render pipeline works even with no network.

    python -m src.fonts.fetch_fonts
"""
from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

import yaml

from ..corpus.common import CONFIG_DIR, PROJECT_ROOT

FONTS_DIR = PROJECT_ROOT / "data" / "fonts"
_UA = "khmer-ocr-corpus/0.1 (research)"

# macOS system Khmer fonts — reliable fallback
SYSTEM_KHMER_FONTS = [
    "/System/Library/Fonts/Supplemental/Khmer Sangam MN.ttf",
]

_GH_RAW = "https://raw.githubusercontent.com/google/fonts/main"


def _gh_candidates(family: str) -> list[str]:
    """Best-effort google/fonts paths for a family (OFL, then APACHE)."""
    slug = family.replace(" ", "").lower()
    file_base = family.replace(" ", "")
    names = [
        f"{file_base}-Regular.ttf",       # static
        f"{file_base}[wght].ttf",         # variable (weight axis)
        f"{file_base}[wdth,wght].ttf",    # variable (width+weight, e.g. Noto Sans Khmer)
        f"{file_base}.ttf",
    ]
    paths = []
    for lic in ("ofl", "apache"):
        for n in names:
            paths.append(f"{_GH_RAW}/{lic}/{slug}/{n}")
    return paths


def _download(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                return False
            data = resp.read()
        if len(data) < 1000:  # not a real font
            return False
        dest.write_bytes(data)
        return True
    except Exception:
        return False


def fetch() -> dict:
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((CONFIG_DIR / "fonts.yaml").read_text(encoding="utf-8"))
    got, failed = [], []

    for f in cfg.get("fonts", []):
        name = f["name"]
        dest = FONTS_DIR / f"{name}.ttf"
        if dest.exists():
            got.append(name)
            continue
        if f.get("source") == "local":
            src = PROJECT_ROOT / f["path"]
            if src.exists():
                shutil.copy(src, dest)
                got.append(name)
            else:
                failed.append(f"{name} (local missing: {f['path']})")
            continue
        family = f.get("google_family", f.get("family", name))
        ok = any(_download(url, dest) for url in _gh_candidates(family))
        (got if ok else failed).append(name)

    # always seed system Khmer fonts as fallback
    for sysf in SYSTEM_KHMER_FONTS:
        p = Path(sysf)
        if p.exists():
            dest = FONTS_DIR / (p.stem.replace(" ", "") + ".ttf")
            if not dest.exists():
                shutil.copy(p, dest)
                got.append(dest.stem + " (system)")

    print(f"[fonts] downloaded/present: {len(got)}")
    for n in got:
        print(f"   ✓ {n}")
    if failed:
        print(f"[fonts] failed ({len(failed)}): {', '.join(failed)}")
        print("   (run validate_coverage; pipeline still works with whatever is present)")
    return {"got": got, "failed": failed}


if __name__ == "__main__":
    fetch()
