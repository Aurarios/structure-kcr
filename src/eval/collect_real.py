"""Ingest real Khmer documents for the held-out OCR test set, and convert labels.

Two modes:

1) Ingest:  drop images/PDFs into data/real/raw_documents/, then:
       python -m src.eval.collect_real --ingest
   Copies images into data/real/images/ (real_XXXX.png), converts PDF pages if PyMuPDF is
   installed, and writes data/real/labelstudio_tasks.json for import into Label Studio.

2) Convert a Label Studio export into our label format (-> data/real/labels/*.json):
       python -m src.eval.collect_real --from-export export.json

The held-out test set is sacred: never render synthetic versions of these pages.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ..corpus.common import DATA
from ..render import to_grounded_markdown as gm

RAW = DATA / "real" / "raw_documents"
IMAGES = DATA / "real" / "images"
LABELS = DATA / "real" / "labels"
LS_TASKS = DATA / "real" / "labelstudio_tasks.json"

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def ingest() -> dict:
    IMAGES.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    n = 0
    tasks = []
    for src in sorted(RAW.iterdir()):
        if src.suffix.lower() in IMG_EXT:
            dest = IMAGES / f"real_{n:04d}{src.suffix.lower()}"
            shutil.copy(src, dest)
            tasks.append({"data": {"image": str(dest.relative_to(DATA.parent))}})
            n += 1
        elif src.suffix.lower() == ".pdf":
            n = _ingest_pdf(src, n, tasks)
    LS_TASKS.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[real] ingested {n} page images -> {IMAGES}")
    print(f"[real] Label Studio tasks -> {LS_TASKS}")
    if n == 0:
        print("   (drop .png/.jpg/.pdf files into data/real/raw_documents/ first)")
    return {"pages": n}


def _ingest_pdf(pdf: Path, start: int, tasks: list) -> int:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print(f"[real] {pdf.name}: skipped (pip install pymupdf to convert PDFs)")
        return start
    doc = fitz.open(pdf)
    n = start
    for page in doc:
        pix = page.get_pixmap(dpi=150)
        dest = IMAGES / f"real_{n:04d}.png"
        pix.save(str(dest))
        tasks.append({"data": {"image": str(dest.relative_to(DATA.parent))}})
        n += 1
    return n


def from_export(export_path: Path) -> dict:
    """Convert a Label Studio export (RectangleLabels + per-region TextArea) into label JSONs."""
    LABELS.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(export_path).read_text(encoding="utf-8"))
    written = 0
    for task in data:
        image = (task.get("data") or {}).get("image", "")
        leaves = _parse_ls_task(task)
        if not image or not leaves:
            continue
        # reading order: top-to-bottom, then left-to-right
        leaves.sort(key=lambda l: (round(l["bbox"][1] / 20), l["bbox"][0]))
        grounded = gm.build_grounded(leaves, 999, 999)   # boxes already in 0-999
        markdown = gm.build_markdown(leaves)
        stem = Path(image).stem
        label = {"id": stem, "image": image, "blocks": leaves,
                 "grounded": grounded, "markdown": markdown, "source": "real"}
        (LABELS / f"{stem}.json").write_text(json.dumps(label, ensure_ascii=False), encoding="utf-8")
        written += 1
    print(f"[real] converted {written} labeled pages -> {LABELS}")
    return {"labels": written}


def _parse_ls_task(task: dict) -> list[dict]:
    """Extract {block_type, text, bbox(0-999)} regions from one LS task's annotations."""
    anns = task.get("annotations") or task.get("completions") or []
    if not anns:
        return []
    results = anns[0].get("result", [])
    boxes: dict[str, dict] = {}
    for r in results:
        rid = r.get("id")
        val = r.get("value", {})
        if "rectanglelabels" in val or all(k in val for k in ("x", "y", "width", "height")):
            x, y, w, h = val.get("x", 0), val.get("y", 0), val.get("width", 0), val.get("height", 0)
            bt = (val.get("rectanglelabels") or ["text"])[0]
            boxes.setdefault(rid, {})["bbox"] = [
                round(x / 100 * 999), round(y / 100 * 999),
                round((x + w) / 100 * 999), round((y + h) / 100 * 999),
            ]
            boxes[rid]["block_type"] = bt
        if "text" in val:
            txt = val["text"]
            boxes.setdefault(rid, {})["text"] = txt[0] if isinstance(txt, list) else txt
    return [b for b in boxes.values() if "bbox" in b and b.get("text")]


def main() -> None:
    ap = argparse.ArgumentParser(description="collect + convert real Khmer eval documents")
    ap.add_argument("--ingest", action="store_true", help="ingest raw_documents/ -> images + LS tasks")
    ap.add_argument("--from-export", type=str, default=None, help="Label Studio export json -> labels/")
    args = ap.parse_args()
    if args.from_export:
        from_export(Path(args.from_export))
    else:
        ingest()


if __name__ == "__main__":
    main()
