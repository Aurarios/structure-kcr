"""Merge per-worker line-crop manifests into train/val splits.

build_line_crops.py writes a per-worker manifest_wNN.jsonl as it runs, but only merges them into
lines_train.jsonl / lines_val.jsonl at the very end. This standalone merger lets us stop the build
early (or recover after a crash) and still produce proper train/val manifests, using the same
deterministic hash split.

  python -m src.recognize.data.merge_line_manifests --out-dir E:/kcr-recognize/lines --val-frac 0.02
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-frac", type=float, default=0.02)
    args = ap.parse_args()

    out = Path(args.out_dir)
    worker_manifests = sorted(out.glob("manifest_w*.jsonl"))
    if not worker_manifests:
        raise SystemExit(f"no manifest_w*.jsonl in {out}")

    bucket = int(round(1 / args.val_frac))
    total = 0
    train_p, val_p = out / "lines_train.jsonl", out / "lines_val.jsonl"
    with open(train_p, "w", encoding="utf-8") as tr, open(val_p, "w", encoding="utf-8") as va:
        for wm in worker_manifests:
            with open(wm, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    (va if hash(row["crop"]) % bucket == 0 else tr).write(
                        json.dumps(row, ensure_ascii=False) + "\n")
                    total += 1
    print(f"[merge] {len(worker_manifests)} worker manifests -> {total} crops")
    print(f"  -> {train_p}")
    print(f"  -> {val_p}")


if __name__ == "__main__":
    main()
