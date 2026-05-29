"""Orchestrator: render synthetic OCR pages (parallel) and assemble train/val/test manifests.

  python src/build_dataset.py --smoke --n 200
  python src/build_dataset.py --n 10000 --workers 6 --image-format jpg

Per page it writes:
  data/synthetic/images/syn_XXXXXXX.(jpg|png)
  data/synthetic/labels/syn_XXXXXXX.json   (blocks w/ image-px + normalized boxes, markdown, grounded)
And manifests:
  data/manifests/train.jsonl   synthetic (minus val)
  data/manifests/val.jsonl     held-out synthetic
  data/manifests/test.jsonl    real labeled pages from data/real/labels/ (if any)

Rendering runs across multiple processes (each its own Chromium). A disk guard stops generation
before free space drops below --min-free-gb so a full-scale run can never fill the disk.
Manifest rows match DeepSeek-OCR-2 fine-tuning shape: {image, prompt, answer, source, label}.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import khmer_utils as ku  # noqa: E402
from src.corpus.common import DATA, load_yaml  # noqa: E402
from src.fonts.validate_coverage import load_usable_fonts  # noqa: E402
from src.render import augment as aug  # noqa: E402
from src.render import overlays as ov  # noqa: E402
from src.render import to_grounded_markdown as gm  # noqa: E402
from src.render.layout_sampler import CorpusText, expected_leaf_texts, sample_page  # noqa: E402
from src.render.render_playwright import PageRenderer  # noqa: E402

PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
IMAGES = DATA / "synthetic" / "images"
LABELS = DATA / "synthetic" / "labels"
MANIFESTS = DATA / "manifests"
REAL_LABELS = DATA / "real" / "labels"


def _scale_leaves(leaves, sx, sy):
    for lf in leaves:
        x1, y1, x2, y2 = lf["bbox"]
        lf["bbox"] = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
        if lf.get("lines"):
            lf["lines"] = [[a * sx, b * sy, c * sx, d * sy] for a, b, c, d in lf["lines"]]
    return leaves


def _normalize_leaves(leaves, w, h):
    for lf in leaves:
        x1, y1, x2, y2 = lf["bbox"]
        lf["bbox_norm"] = [
            max(0, min(999, round(x1 / w * 999))), max(0, min(999, round(y1 / h * 999))),
            max(0, min(999, round(x2 / w * 999))), max(0, min(999, round(y2 / h * 999))),
        ]
    return leaves


def _encode(img, fmt: str, quality: int) -> bytes:
    if fmt == "jpg":
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def _render_chunk(task: dict) -> list[dict]:
    """Worker: render indices [start, start+count) in its own process/browser."""
    wid = task["worker_id"]
    layouts, aug_cfg, fonts = task["layouts"], task["aug_cfg"], task["fonts"]
    fmt, quality, dsf = task["fmt"], task["quality"], task["dsf"]
    augment_on, min_free = task["augment_on"], task["min_free_bytes"]
    overlay_prob = float(task.get("overlay_prob", 0.0))

    rng = random.Random(task["seed"] + wid * 1_000_003)
    corpus = CorpusText(seed=task["seed"] + wid, quiet=True)
    rows: list[dict] = []
    done = 0
    with PageRenderer(device_scale_factor=dsf) as renderer:
        for j in range(task["count"]):
            if done % 200 == 0 and shutil.disk_usage(str(DATA)).free < min_free:
                print(f"[w{wid}] disk guard hit, stopping at {done}")
                break
            i = task["start"] + j
            page = sample_page(rng, corpus, layouts, fonts)
            try:
                res = renderer.render(page)
            except Exception as e:
                print(f"[w{wid}] render failed page {i} ({page['template']}/{page['font_name']}): {e}")
                continue
            if not res.leaves:
                continue

            img = cv2.imdecode(np.frombuffer(res.image_png, np.uint8), cv2.IMREAD_COLOR)
            ih, iw = img.shape[:2]
            sx, sy = iw / res.css_width, ih / res.css_height
            leaves = _scale_leaves(res.leaves, sx, sy)

            if augment_on:
                png, leaves = aug.augment(res.image_png, leaves, aug_cfg, rng)
                img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
                ih, iw = img.shape[:2]

            # overlay augmentation (seals / stamps / signatures / highlights).
            # Labels are intentionally NOT updated — ground truth is the underlying text;
            # the overlay is visual noise the model must learn to read through.
            if overlay_prob > 0.0 and rng.random() < overlay_prob:
                try:
                    img = ov.apply_overlays(img, rng, fonts)
                except Exception as e:
                    print(f"[w{wid}] overlay failed page {i}: {e}")

            leaves = _normalize_leaves(leaves, iw, ih)
            grounded = gm.build_grounded(leaves, iw, ih)
            markdown = gm.build_markdown(leaves)

            pid = f"syn_{i:07d}"
            img_path = IMAGES / f"{pid}.{fmt}"
            img_path.write_bytes(_encode(img, fmt, quality))
            label = {
                "id": pid, "image": str(img_path.relative_to(DATA.parent)),
                "template": page["template"], "font": page["font_name"],
                "image_width": iw, "image_height": ih,
                "blocks": leaves, "markdown": markdown, "grounded": grounded,
                "source_text": ku.normalize("".join(t.strip() for t in expected_leaf_texts(page))),
            }
            label_path = LABELS / f"{pid}.json"
            label_path.write_text(json.dumps(label, ensure_ascii=False), encoding="utf-8")
            rows.append({"image": label["image"], "prompt": PROMPT, "answer": grounded,
                         "source": "synthetic", "label": str(label_path.relative_to(DATA.parent))})
            done += 1
            if done % 1000 == 0:
                print(f"[w{wid}] {done}/{task['count']}")
    print(f"[w{wid}] finished: {len(rows)} pages")
    return rows


def generate(n: int, seed: int, augment_on: bool, workers: int, fmt: str,
             quality: int, dsf: int, min_free_gb: float,
             start_index: int = 0, overlay_prob: float = 0.0) -> list[dict]:
    IMAGES.mkdir(parents=True, exist_ok=True)
    LABELS.mkdir(parents=True, exist_ok=True)
    layouts = load_yaml("layouts.yaml")
    aug_cfg = load_yaml("augment.yaml")
    fonts = load_usable_fonts()
    if not fonts:
        from src.fonts.fetch_fonts import fetch
        print("[build] no usable fonts; fetching...")
        fetch()
        fonts = load_usable_fonts()
    if not fonts:
        raise SystemExit("No usable Khmer fonts. Run: python -m src.fonts.fetch_fonts")

    free_gb = shutil.disk_usage(str(DATA)).free / 1e9
    print(f"[build] {len(fonts)} fonts | {n} pages | {workers} workers | {fmt} q{quality} | "
          f"dsf={dsf} | free disk {free_gb:.1f}GB (guard {min_free_gb}GB) | "
          f"start_index={start_index} overlay_prob={overlay_prob}")

    # split contiguous index ranges across workers (offset by start_index so we don't
    # overwrite earlier-rendered pages with the same filenames)
    per = n // workers
    rem = n % workers
    tasks, start = [], start_index
    for w in range(workers):
        count = per + (1 if w < rem else 0)
        if count == 0:
            continue
        tasks.append({"worker_id": w, "start": start, "count": count,
                      "seed": seed + start_index,    # vary seed so reruns from different start
                                                     # indices don't sample identical pages
                      "layouts": layouts, "aug_cfg": aug_cfg, "fonts": fonts,
                      "augment_on": augment_on, "fmt": fmt, "quality": quality, "dsf": dsf,
                      "min_free_bytes": int(min_free_gb * 1e9),
                      "overlay_prob": overlay_prob})
        start += count

    rows: list[dict] = []
    if workers == 1:
        rows = _render_chunk(tasks[0])
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_render_chunk, t) for t in tasks]
            for f in as_completed(futs):
                rows.extend(f.result())
    return rows


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_manifests(rows: list[dict], val_frac: float, merge_existing: bool = False) -> None:
    """Write train/val/test manifests.

    If ``merge_existing`` is True, load any existing train.jsonl + val.jsonl and union them
    with ``rows`` (dedupe by image path, last write wins). val/train split is then computed
    deterministically from a hash of the image path so re-runs are stable.
    """
    MANIFESTS.mkdir(parents=True, exist_ok=True)

    if merge_existing:
        existing = _read_jsonl(MANIFESTS / "train.jsonl") + _read_jsonl(MANIFESTS / "val.jsonl")
        by_image: dict[str, dict] = {r["image"]: r for r in existing}
        for r in rows:
            by_image[r["image"]] = r        # new rows replace any stale entry for the same image
        all_rows = list(by_image.values())
    else:
        all_rows = list(rows)

    all_rows.sort(key=lambda r: r["image"])
    # deterministic split: hash the image path modulo bucket size; keeps val membership stable
    # across re-runs even as the dataset grows.
    bucket = max(1, int(round(1.0 / val_frac)))
    val: list[dict] = []
    train: list[dict] = []
    for r in all_rows:
        if hash(r["image"]) % bucket == 0:
            val.append(r)
        else:
            train.append(r)

    test = []
    if REAL_LABELS.exists():
        for lp in sorted(REAL_LABELS.glob("*.json")):
            d = json.loads(lp.read_text(encoding="utf-8"))
            ans = d.get("grounded") or d.get("answer")
            if d.get("image") and ans:
                test.append({"image": d["image"], "prompt": PROMPT, "answer": ans,
                             "source": "real", "label": str(lp.relative_to(DATA.parent))})

    for name, split in (("train", train), ("val", val), ("test", test)):
        path = MANIFESTS / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in split:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {name:5} {len(split):>7} -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="render synthetic OCR dataset + manifests")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--smoke", action="store_true", help="cap n at 200")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--image-format", choices=["jpg", "png"], default="jpg")
    ap.add_argument("--jpeg-quality", type=int, default=85)
    ap.add_argument("--dsf", type=int, default=2, help="device scale factor (1 = smaller files)")
    ap.add_argument("--min-free-gb", type=float, default=2.0, help="stop if free disk drops below this")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--start-index", type=int, default=0,
                    help="first syn_NNNNNNN index to write; use > 0 to extend an existing dataset "
                         "without overwriting earlier files")
    ap.add_argument("--overlay-prob", type=float, default=0.0,
                    help="per-page probability of compositing seals/stamps/signatures/highlights "
                         "(visual noise only; labels unchanged)")
    ap.add_argument("--merge-manifest", action="store_true",
                    help="merge new rows with the existing train/val manifests instead of overwriting")
    args = ap.parse_args()

    n = min(args.n, 200) if args.smoke else args.n
    rows = generate(n, args.seed, not args.no_augment, args.workers, args.image_format,
                    args.jpeg_quality, args.dsf, args.min_free_gb,
                    start_index=args.start_index, overlay_prob=args.overlay_prob)
    print(f"\n[build] rendered {len(rows)} pages")
    write_manifests(rows, args.val_frac, merge_existing=args.merge_manifest)


if __name__ == "__main__":
    main()
