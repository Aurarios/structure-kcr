"""Shared corpus helpers: project paths, JSONL shard I/O, source-config loading."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA = PROJECT_ROOT / "data"
RAW_DIR = DATA / "corpus" / "raw"
CLEAN_DIR = DATA / "corpus" / "clean"
DEDUP_DIR = DATA / "corpus" / "dedup"
FILTERED_DIR = DATA / "corpus" / "filtered"
FILTERED_REJECTED_DIR = DATA / "corpus" / "filtered_rejected"
LM_DIR = DATA / "corpus" / "lm_corpus"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Doc:
    """One corpus document, with provenance carried end-to-end."""
    text: str
    source: str
    license: str = "unknown"
    url: str | None = None
    lang_mix: float | None = None       # khmer_ratio, filled during cleaning
    n_chars: int | None = None
    non_commercial: bool = False
    fetched_at: str = field(default_factory=now_iso)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


_DOC_FIELDS = ("text", "source", "license", "url", "lang_mix", "n_chars",
               "non_commercial", "fetched_at", "extra")


def doc_from_row(row: dict) -> "Doc":
    """Rebuild a Doc from a jsonl row, ignoring any unknown keys."""
    kwargs = {k: row[k] for k in _DOC_FIELDS if k in row and row[k] is not None}
    return Doc(**kwargs)


def load_sources() -> dict[str, Any]:
    with open(CONFIG_DIR / "sources.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_yaml(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


# --- JSONL shard I/O --------------------------------------------------------

def iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_jsonl_dir(d: Path, pattern: str = "*.jsonl") -> Iterator[dict]:
    for p in sorted(d.glob(pattern)):
        yield from iter_jsonl(p)


def write_jsonl(path: Path, rows: Iterable[dict | Doc]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            s = row.to_json() if isinstance(row, Doc) else json.dumps(row, ensure_ascii=False)
            f.write(s + "\n")
            n += 1
    return n


class ShardWriter:
    """Writes Docs to size-capped jsonl shards: <prefix>-00000.jsonl, ..."""

    def __init__(self, out_dir: Path, prefix: str, max_lines: int = 50_000):
        self.out_dir = out_dir
        self.prefix = prefix
        self.max_lines = max_lines
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._idx = 0
        self._count = 0
        self._total = 0
        self._fh = None
        self._open()

    def _open(self):
        if self._fh:
            self._fh.close()
        path = self.out_dir / f"{self.prefix}-{self._idx:05d}.jsonl"
        self._fh = open(path, "w", encoding="utf-8")
        self._count = 0

    def write(self, doc: dict | Doc):
        if self._count >= self.max_lines:
            self._idx += 1
            self._open()
        s = doc.to_json() if isinstance(doc, Doc) else json.dumps(doc, ensure_ascii=False)
        self._fh.write(s + "\n")
        self._count += 1
        self._total += 1

    def close(self) -> int:
        if self._fh:
            self._fh.close()
        return self._total

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
