"""Generate real in-context line crops + text for the recognizer.

Renders pages fresh (Chromium, correct Khmer shaping), extracts per-line (text, box) via
src/render/render_lines.py, crops each line from the page screenshot, and persists ONLY the small
crops (sharded dirs) + a JSONL manifest of {crop, text}. The full-resolution page images are never
written, so even a full-scale run stays disk-frugal (~30 GB of crops vs ~150 GB of pages).

Crops are CLEAN (no augmentation) — line-level augmentation is applied at recognizer train time.
Labels are ku.normalize'd so they match the tokenizer's normalized space.

  python -m src.recognize.data.build_line_crops --pages 100000 --workers 12 --start-seed 0
  python -m src.recognize.data.build_line_crops --pages 200 --workers 2   # smoke
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src import khmer_utils as ku  # noqa: E402
from src.corpus.common import DATA, load_yaml  # noqa: E402
from src.fonts.validate_coverage import load_usable_fonts  # noqa: E402
from src.render.layout_sampler import CorpusText, sample_page  # noqa: E402
from src.render.render_lines import LineRenderer  # noqa: E402

OUT_DIR = DATA / "recognize" / "lines"
CROPS_PER_SHARD = 2000
MIN_LINE_H = 8        # css px; skip sub-line fragments
PAD = 2               # px padding around each crop
CROP_H = 48           # save crops at recognizer input height (disk + no train-time resize)
CROP_MAX_W = 1200     # cap width so a long line can't blow up seq length / disk


def _resize_crop(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape[:2]
    if h < 1:
        return crop
    nw = max(1, int(round(w * CROP_H / h)))
    nw = min(nw, CROP_MAX_W)
    return cv2.resize(crop, (nw, CROP_H), interpolation=cv2.INTER_AREA)


def _crop_worker(task: dict) -> dict:
    wid = task["worker_id"]
    layouts = task["layouts"]
    fonts = task["fonts"]
    start, count, seed = task["start"], task["count"], task["seed"]
    dsf = task["dsf"]
    quality = task["quality"]
    out_dir = Path(task["out_dir"])

    rng = random.Random(seed + wid * 1_000_003)
    corpus = CorpusText(seed=seed + wid, quiet=True)
    shard_base = out_dir / f"w{wid:02d}"
    manifest_path = out_dir / f"manifest_w{wid:02d}.jsonl"
    written = 0
    shard_idx = -1
    cur_dir = None
    mf = open(manifest_path, "w", encoding="utf-8")

    with LineRenderer(device_scale_factor=dsf) as renderer:
        for j in range(count):
            page = sample_page(rng, corpus, layouts, fonts)
            try:
                res = renderer.render(page)
            except Exception as e:
                print(f"[w{wid}] render failed page {start+j}: {e}")
                continue
            if not res.lines:
                continue
            img = cv2.imdecode(np.frombuffer(res.image_png, np.uint8), cv2.IMREAD_COLOR)
            ih, iw = img.shape[:2]
            sx, sy = iw / res.css_width, ih / res.css_height
            for ln in res.lines:
                text = ku.normalize((ln.get("text") or "").strip())
                if not text:
                    continue
                x1, y1, x2, y2 = ln["box"]
                if (y2 - y1) < MIN_LINE_H or (x2 - x1) < MIN_LINE_H:
                    continue
                # to image px + padding, clipped
                px1 = max(0, int(x1 * sx) - PAD); py1 = max(0, int(y1 * sy) - PAD)
                px2 = min(iw, int(x2 * sx) + PAD); py2 = min(ih, int(y2 * sy) + PAD)
                if px2 - px1 < 4 or py2 - py1 < 4:
                    continue
                crop = _resize_crop(img[py1:py2, px1:px2])
                # new shard dir every CROPS_PER_SHARD crops
                if written % CROPS_PER_SHARD == 0:
                    shard_idx += 1
                    cur_dir = shard_base / f"shard_{shard_idx:05d}"
                    cur_dir.mkdir(parents=True, exist_ok=True)
                rel = f"w{wid:02d}/shard_{shard_idx:05d}/ln_{written:08d}.jpg"
                ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if not ok:
                    continue
                (out_dir / rel).write_bytes(buf.tobytes())
                mf.write(json.dumps({"crop": rel, "text": text}, ensure_ascii=False) + "\n")
                written += 1
            if (j + 1) % 200 == 0:
                print(f"[w{wid}] pages {j+1}/{count} crops {written}")
    mf.close()
    print(f"[w{wid}] done: {written} crops")
    return {"worker_id": wid, "manifest": str(manifest_path), "crops": written}


def main() -> None:
    ap = argparse.ArgumentParser(description="build recognizer line crops")
    ap.add_argument("--pages", type=int, default=200)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--dsf", type=int, default=2)
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR),
                    help="where to write crops + manifests (use a drive with space)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layouts = load_yaml("layouts.yaml")
    fonts = load_usable_fonts()
    if not fonts:
        raise SystemExit("no usable fonts; run src.fonts.fetch_fonts")

    per = args.pages // args.workers
    rem = args.pages % args.workers
    tasks, start = [], 0
    for w in range(args.workers):
        count = per + (1 if w < rem else 0)
        if count == 0:
            continue
        tasks.append({"worker_id": w, "start": start, "count": count,
                      "seed": args.start_seed, "layouts": layouts, "fonts": fonts,
                      "dsf": args.dsf, "quality": args.jpeg_quality, "out_dir": str(out_dir)})
        start += count

    results = []
    if args.workers == 1:
        results = [_crop_worker(tasks[0])]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_crop_worker, t) for t in tasks]
            for f in as_completed(futs):
                results.append(f.result())

    # merge per-worker manifests into train/val with a deterministic split
    total = 0
    train_p = out_dir / "lines_train.jsonl"
    val_p = out_dir / "lines_val.jsonl"
    with open(train_p, "w", encoding="utf-8") as tr, open(val_p, "w", encoding="utf-8") as va:
        for r in sorted(results, key=lambda x: x["worker_id"]):
            with open(r["manifest"], encoding="utf-8") as mf:
                for line in mf:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    (va if hash(row["crop"]) % int(round(1 / args.val_frac)) == 0 else tr).write(
                        json.dumps(row, ensure_ascii=False) + "\n")
                    total += 1
    print(f"\n[lines] total crops: {total}")
    print(f"  -> {train_p}")
    print(f"  -> {val_p}")


if __name__ == "__main__":
    main()
