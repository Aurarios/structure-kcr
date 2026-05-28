"""Fetch Khmer text from HuggingFace `datasets` (OSCAR, CC100, mC4, CulturaX, MADLAD, khPOS...).

Uses streaming so we never download a multi-GB shard in full just to sample. Writes raw Docs to
data/corpus/raw/<source>-*.jsonl with provenance.
"""
from __future__ import annotations

from typing import Any

from tqdm import tqdm

from .common import RAW_DIR, Doc, ShardWriter


def fetch_hf_source(src: dict[str, Any], limit: int | None = None) -> int:
    """Stream one HF-typed source entry from sources.yaml into raw shards.

    `limit` caps documents (use for --smoke). Returns number written.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise SystemExit("pip install datasets") from e

    name = src["name"]
    path = src["hf_path"]
    config = src.get("hf_config")
    text_field = src.get("text_field", "text")
    split = src.get("hf_split", "train")

    print(f"[hf] {name}: streaming {path}" + (f":{config}" if config else ""))
    try:
        ds = load_dataset(path, config, split=split, streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"[hf] {name}: FAILED to load ({e}). Skipping.")
        return 0

    written = 0
    with ShardWriter(RAW_DIR, name) as w:
        it = iter(ds)
        pbar = tqdm(it, desc=name, unit="doc")
        for row in pbar:
            text = row.get(text_field) or ""
            if not isinstance(text, str) or not text.strip():
                continue
            w.write(Doc(
                text=text,
                source=name,
                license=src.get("license", "unknown"),
                url=row.get("url") or row.get("meta", {}).get("url") if isinstance(row.get("meta"), dict) else row.get("url"),
                non_commercial=bool(src.get("non_commercial", False)),
                extra={"hf_path": path, "hf_config": config},
            ))
            written += 1
            if limit and written >= limit:
                break
            if written % 1000 == 0:
                pbar.set_postfix(written=written)
    print(f"[hf] {name}: wrote {written} docs")
    return written
