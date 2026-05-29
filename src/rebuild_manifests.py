"""Rebuild data/manifests/{train,val}.jsonl from data/synthetic/labels/*.json.

Use this if the manifests were lost or overwritten. Scans every label JSON, emits manifest rows
in the same shape build_dataset.py uses, and writes them via the standard deterministic
val-by-hash split so val membership matches what training expects.

  python -m src.rebuild_manifests [--val-frac 0.05]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.build_dataset import PROMPT, write_manifests
from src.corpus.common import DATA

LABELS = DATA / "synthetic" / "labels"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.05)
    args = ap.parse_args()

    rows: list[dict] = []
    n = 0
    for lp in sorted(LABELS.glob("syn_*.json")):
        try:
            d = json.loads(lp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[!] {lp.name}: {e}")
            continue
        grounded = d.get("grounded") or d.get("answer")
        img_path = d.get("image")
        if not (grounded and img_path):
            continue
        rows.append({
            "image": img_path,
            "prompt": PROMPT,
            "answer": grounded,
            "source": "synthetic",
            "label": str(lp.relative_to(DATA.parent)),
        })
        n += 1
        if n % 50000 == 0:
            print(f"  scanned {n}")
    print(f"[rebuild] {n} labels -> manifests")
    write_manifests(rows, args.val_frac, merge_existing=False)


if __name__ == "__main__":
    main()
