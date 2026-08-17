"""Round-trip gate: text fed into a page must equal text decoded from its label (after normalize).

This catches Unicode/coeng ordering bugs and any text loss in the render -> DOM-extraction path. It
compares the stored `source_text` (fed in) against the normalized concatenation of the label's block
texts (decoded from the rendered DOM). Mismatches are reported with a small diff.

    python -m src.validate.roundtrip_check [--max-show 10] [--root E:/kcr-v5]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import khmer_utils as ku
from ..corpus.common import DATA

LABELS = DATA / "synthetic" / "labels"


def _label_files(root: Path | None) -> list[Path]:
    """Flat V1 layout (data/synthetic/labels) or a versioned per-layout root (<root>/*/labels)."""
    if root is None:
        return sorted(LABELS.glob("*.json")) or sorted(
            (DATA / "synthetic").glob("*/labels/*.json"))
    return sorted(Path(root).glob("*/labels/*.json"))


def _decoded_text(label: dict) -> str:
    return ku.normalize("".join(b["text"] for b in label["blocks"]))


def check(max_show: int, root: Path | None = None) -> dict:
    files = _label_files(root)
    if not files:
        print(f"No labels in {root or LABELS}. Run build_dataset first.")
        return {"total": 0, "pass": 0, "fail": 0}

    total = passed = 0
    shown = 0
    for f in files:
        label = json.loads(f.read_text(encoding="utf-8"))
        total += 1
        expected = ku.normalize(label.get("source_text", ""))
        got = _decoded_text(label)
        if expected == got:
            passed += 1
        elif shown < max_show:
            shown += 1
            print(f"\n[MISMATCH] {f.name}")
            print(f"   fed in : {expected[:80]!r}")
            print(f"   decoded: {got[:80]!r}")
    fail = total - passed
    rate = passed / total * 100 if total else 0
    print(f"\nroundtrip: {passed}/{total} pass ({rate:.1f}%), {fail} fail")
    if fail == 0:
        print("[GATE PASS]")
    else:
        print("[GATE FAIL] fix normalization/rendering before scaling generation")
    return {"total": total, "pass": passed, "fail": fail}


def main() -> None:
    ap = argparse.ArgumentParser(description="round-trip text gate")
    ap.add_argument("--max-show", type=int, default=10)
    ap.add_argument("--root", type=Path, default=None,
                    help="versioned dataset root with <layout>/labels/*.json (e.g. E:/kcr-v5)")
    args = ap.parse_args()
    res = check(args.max_show, root=args.root)
    raise SystemExit(0 if res["fail"] == 0 else 1)


if __name__ == "__main__":
    main()
