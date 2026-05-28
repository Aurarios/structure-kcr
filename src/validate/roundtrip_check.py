"""Round-trip gate: text fed into a page must equal text decoded from its label (after normalize).

This catches Unicode/coeng ordering bugs and any text loss in the render -> DOM-extraction path. It
compares the stored `source_text` (fed in) against the normalized concatenation of the label's block
texts (decoded from the rendered DOM). Mismatches are reported with a small diff.

    python -m src.validate.roundtrip_check [--max-show 10]
"""
from __future__ import annotations

import argparse
import json

from .. import khmer_utils as ku
from ..corpus.common import DATA

LABELS = DATA / "synthetic" / "labels"


def _decoded_text(label: dict) -> str:
    return ku.normalize("".join(b["text"] for b in label["blocks"]))


def check(max_show: int) -> dict:
    files = sorted(LABELS.glob("*.json"))
    if not files:
        print(f"No labels in {LABELS}. Run build_dataset first.")
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
            print(f"\n✗ MISMATCH {f.name}")
            print(f"   fed in : {expected[:80]!r}")
            print(f"   decoded: {got[:80]!r}")
    fail = total - passed
    rate = passed / total * 100 if total else 0
    print(f"\nroundtrip: {passed}/{total} pass ({rate:.1f}%), {fail} fail")
    if fail == 0:
        print("✓ GATE PASS")
    else:
        print("✗ GATE FAIL — fix normalization/rendering before scaling generation")
    return {"total": total, "pass": passed, "fail": fail}


def main() -> None:
    ap = argparse.ArgumentParser(description="round-trip text gate")
    ap.add_argument("--max-show", type=int, default=10)
    args = ap.parse_args()
    res = check(args.max_show)
    raise SystemExit(0 if res["fail"] == 0 else 1)


if __name__ == "__main__":
    main()
