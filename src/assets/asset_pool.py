"""Serve downloaded real images to the layout sampler.

`AssetPool` loads the per-category file lists from the asset root (default E:\\kcr-assets) and yields
random local image paths. Categories with no downloaded assets return None so the sampler can fall
back to a synthetic gradient (keeps rendering working before/without a full fetch).

Used inside `build_dataset` workers (ProcessPoolExecutor, spawn): construct once per worker, then
call `.get(category)`. Cheap to build (just reads file lists).
"""
from __future__ import annotations

import random
from pathlib import Path

DEFAULT_ROOT = Path("E:/kcr-assets")

# logical category -> asset directory. `flag`/`hand_drawing` reuse existing pools.
CATEGORIES = ["photos", "portraits", "signatures", "stamps", "drawings", "logos", "charts"]


class AssetPool:
    def __init__(self, root: Path | str = DEFAULT_ROOT, seed: int = 0, quiet: bool = False):
        self.root = Path(root)
        self.rng = random.Random(seed)
        self._cat: dict[str, list[Path]] = {}
        total = 0
        for cat in CATEGORIES:
            d = self.root / cat
            files = sorted(d.glob("*.jpg")) if d.exists() else []
            self._cat[cat] = files
            total += len(files)
        self.total = total
        if not quiet:
            counts = ", ".join(f"{c}:{len(self._cat[c])}" for c in CATEGORIES)
            print(f"[assets] pool from {self.root}: {total} images ({counts})")

    def has(self, category: str) -> bool:
        return bool(self._cat.get(category))

    def get(self, category: str) -> Path | None:
        """Random asset path for a category, or None if that category is empty."""
        files = self._cat.get(category)
        if not files:
            return None
        return self.rng.choice(files)

    def get_any(self, categories: list[str]) -> Path | None:
        """First non-empty category among `categories` (in order), random pick from it."""
        for c in categories:
            if self._cat.get(c):
                return self.rng.choice(self._cat[c])
        return None
