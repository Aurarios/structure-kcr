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
from src.assets.asset_pool import AssetPool  # noqa: E402

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


def _store_path(p: Path) -> str:
    """Manifest path: relative to the repo when under it (V1), else absolute (e.g. on E:\\ for V2)."""
    try:
        return str(p.relative_to(DATA.parent))
    except ValueError:
        return str(p)


def _render_chunk(task: dict) -> list[dict]:
    """Worker: render indices [start, start+count) in its own process/browser."""
    wid = task["worker_id"]
    layouts, aug_cfg, fonts = task["layouts"], task["aug_cfg"], task["fonts"]
    fmt, quality, dsf = task["fmt"], task["quality"], task["dsf"]
    augment_on, min_free = task["augment_on"], task["min_free_bytes"]
    overlay_prob = float(task.get("overlay_prob", 0.0))

    # V3: outputs grouped per layout type under base_out/{layout_type}/{images,labels}
    base_out = Path(task["base_out"])
    _dir_cache: dict[str, tuple[Path, Path]] = {}

    def _dirs_for(layout_type: str) -> tuple[Path, Path]:
        if layout_type not in _dir_cache:
            im = base_out / layout_type / "images"
            lb = base_out / layout_type / "labels"
            im.mkdir(parents=True, exist_ok=True)
            lb.mkdir(parents=True, exist_ok=True)
            _dir_cache[layout_type] = (im, lb)
        return _dir_cache[layout_type]

    rng = random.Random(task["seed"] + wid * 1_000_003)
    corpus = CorpusText(seed=task["seed"] + wid, quiet=True)
    assets = AssetPool(task["assets_dir"], seed=task["seed"] + wid * 7919, quiet=True)
    rows: list[dict] = []
    done = 0
    with PageRenderer(device_scale_factor=dsf) as renderer:
        for j in range(task["count"]):
            if done % 200 == 0 and shutil.disk_usage(task.get("disk_path") or str(DATA)).free < min_free:
                print(f"[w{wid}] disk guard hit, stopping at {done}")
                break
            i = task["start"] + j
            page = sample_page(rng, corpus, layouts, fonts, assets)
            layout_type = page["layout_type"]
            try:
                res = renderer.render(page)
            except Exception as e:
                print(f"[w{wid}] render failed page {i} ({layout_type}/{page['font_name']}): {e}")
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

            # NOTE: figure/image regions now contain REAL embedded images (from AssetPool), so the
            # old gradient-painting (_paint_figures) is gone — that fake texture was why V1/V2
            # overfit. overlay augmentation (seals / stamps / signatures / highlights):
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

            images_dir, labels_dir = _dirs_for(layout_type)
            pid = f"{layout_type}_{i:07d}"
            img_path = images_dir / f"{pid}.{fmt}"
            img_path.write_bytes(_encode(img, fmt, quality))
            # per-page provenance/style record: lets dataset_tools browse/stats query by style and
            # makes every page attributable to its generator + sampling run (V5 debugging story)
            meta = {
                "engine": page.get("engine", "curated"),
                "font_body": page["font_name"], "font_title": page.get("title_font_name",
                                                                       page["font_name"]),
                "body_px": page.get("body_px"), "color": page.get("color"),
                "bg": page.get("page_bg"), "align": page.get("align"),
                "columns": page.get("columns", 1),
                "run_seed": task["seed"], "worker": wid, "page_index": i,
            }
            label = {
                "id": pid, "image": _store_path(img_path),
                "layout_type": layout_type, "template": layout_type, "font": page["font_name"],
                "image_width": iw, "image_height": ih, "meta": meta,
                "blocks": leaves, "markdown": markdown, "grounded": grounded,
                "source_text": ku.normalize("".join(t.strip() for t in expected_leaf_texts(page))),
            }
            label_path = labels_dir / f"{pid}.json"
            label_path.write_text(json.dumps(label, ensure_ascii=False), encoding="utf-8")
            rows.append({"image": label["image"], "prompt": PROMPT, "answer": grounded,
                         "source": "synthetic", "layout_type": layout_type,
                         "label": _store_path(label_path),
                         "classes": sorted({b.get("block_type", "?") for b in leaves}),
                         "engine": meta["engine"]})
            done += 1
            if done % 1000 == 0:
                print(f"[w{wid}] {done}/{task['count']}")
    print(f"[w{wid}] finished: {len(rows)} pages")
    return rows


def generate(n: int, seed: int, augment_on: bool, workers: int, fmt: str,
             quality: int, dsf: int, min_free_gb: float,
             start_index: int = 0, overlay_prob: float = 0.0,
             out_dir: Path | None = None, assets_dir: str = "E:/kcr-assets") -> list[dict]:
    base_out = out_dir if out_dir else (DATA / "synthetic")   # per-layout dirs created per worker
    base_out.mkdir(parents=True, exist_ok=True)
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

    # Free-space guard must watch the OUTPUT filesystem. Path.anchor is wrong on POSIX
    # (every absolute path's anchor is "/", so it'd poll the root drive, not the output
    # mount like /mnt/DATA_1). Resolve to the nearest existing dir on the output path instead.
    _probe = base_out
    while not _probe.exists() and _probe != _probe.parent:
        _probe = _probe.parent
    disk_path = str(_probe)
    free_gb = shutil.disk_usage(disk_path).free / 1e9
    assets = AssetPool(assets_dir, quiet=True)
    print(f"[build] {len(fonts)} fonts | {assets.total} real images | {n} pages | {workers} workers "
          f"| {fmt} q{quality} | dsf={dsf} | out={base_out} | free {free_gb:.1f}GB "
          f"(guard {min_free_gb}GB) | start_index={start_index} overlay_prob={overlay_prob}")
    if assets.total == 0:
        print("[build] WARNING: asset pool empty — image regions will use gradient fallback. "
              "Run: python -m src.assets.fetch_assets")

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
                      "min_free_bytes": int(min_free_gb * 1e9), "overlay_prob": overlay_prob,
                      "base_out": str(base_out), "assets_dir": str(assets_dir),
                      "disk_path": disk_path})
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


def write_manifests(rows: list[dict], val_frac: float, merge_existing: bool = False,
                    manifest_dir: Path | None = None) -> None:
    """Write train/val/test manifests.

    ``manifest_dir`` lets V2 write to a SEPARATE directory (e.g. data/manifests_v2) so the baked V1
    manifests are never overwritten. If ``merge_existing`` is True, load any existing train.jsonl +
    val.jsonl and union them with ``rows`` (dedupe by image path, last write wins). val/train split
    is computed deterministically from a hash of the image path so re-runs are stable.
    """
    global MANIFESTS
    MANIFESTS = manifest_dir or MANIFESTS
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


def write_dataset_meta(base_out: Path, rows: list[dict], args) -> None:
    """Versioned-dataset bookkeeping under the output root: meta.json (run provenance, appended per
    run), index.jsonl (one row per page: id/layout/engine/classes/paths), DATASET_CARD.md summary."""
    import subprocess
    from collections import Counter
    from datetime import datetime

    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=DATA.parent,
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        git = "?"
    run = {"date": datetime.now().isoformat(timespec="seconds"), "git": git,
           "pages": len(rows), "args": {k: str(v) for k, v in vars(args).items()}}
    meta_p = base_out / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {"runs": []}
    meta["runs"].append(run)
    meta_p.write_text(json.dumps(meta, indent=1), encoding="utf-8")

    mode = "a" if (args.start_index or 0) > 0 else "w"
    with open(base_out / "index.jsonl", mode, encoding="utf-8") as f:
        for r in rows:
            pid = Path(r["label"]).stem
            f.write(json.dumps({"id": pid, "layout": r["layout_type"],
                                "engine": r.get("engine", "curated"),
                                "classes": r.get("classes", []),
                                "image": r["image"], "label": r["label"]},
                               ensure_ascii=False) + "\n")

    index = _read_jsonl(base_out / "index.jsonl")
    lay = Counter(r["layout"] for r in index)
    eng = Counter(r.get("engine", "curated") for r in index)
    cls = Counter(c for r in index for c in r.get("classes", []))
    card = [f"# Dataset card — {base_out.name}", "",
            f"Pages: **{len(index)}** · engines: " +
            ", ".join(f"{k} {v}" for k, v in eng.most_common()),
            f"Last run: {run['date']} (git {git}, {run['pages']} pages this run)", "",
            "## Pages per layout", ""]
    card += [f"- {k}: {v}" for k, v in lay.most_common()]
    card += ["", "## Block-type page coverage (pages containing >=1)", ""]
    card += [f"- {k}: {v}" for k, v in cls.most_common()]
    card += ["", "Full det-target distribution: "
             f"`python -m src.dataset_tools.stats --root {base_out} --gate`", ""]
    (base_out / "DATASET_CARD.md").write_text("\n".join(card), encoding="utf-8")
    print(f"[build] dataset meta -> {meta_p}, index.jsonl ({len(index)} pages), DATASET_CARD.md")


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
    ap.add_argument("--manifest-dir", default=None,
                    help="write manifests to this dir instead of data/manifests (e.g. manifests_v2 "
                         "to keep the baked V1 manifests untouched)")
    ap.add_argument("--out-dir", default=None,
                    help="write per-layout images/labels under this dir (e.g. E:/kcr-v3) instead of "
                         "data/synthetic; manifests store absolute paths so any drive works")
    ap.add_argument("--assets-dir", default="E:/kcr-assets",
                    help="real-image pool root (from src.assets.fetch_assets); empty -> gradients")
    args = ap.parse_args()

    n = min(args.n, 200) if args.smoke else args.n
    out_dir = Path(args.out_dir) if args.out_dir else None
    rows = generate(n, args.seed, not args.no_augment, args.workers, args.image_format,
                    args.jpeg_quality, args.dsf, args.min_free_gb,
                    start_index=args.start_index, overlay_prob=args.overlay_prob, out_dir=out_dir,
                    assets_dir=args.assets_dir)
    print(f"\n[build] rendered {len(rows)} pages")
    if out_dir and rows:
        write_dataset_meta(out_dir, rows, args)
    mdir = (DATA / args.manifest_dir) if args.manifest_dir else None
    write_manifests(rows, args.val_frac, merge_existing=args.merge_manifest, manifest_dir=mdir)


if __name__ == "__main__":
    main()
