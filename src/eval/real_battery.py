"""Real-document battery: fixed set of real Khmer pages -> versioned end-to-end reports.

The synthetic val set can't measure real-world generalization, so this runs the FULL pipeline
(detect -> line split -> recognize -> assemble) on the pages in data/real_eval/pages/ and writes a
self-contained HTML report per --tag (one tag per model version). Comparing two tags side-by-side
is how we judge whether a data/model change actually helped on real documents.

  python -m src.eval.real_battery --tag v4_parallel \
      --det-ckpt data/checkpoints/detector_obj_v4_parallel_line/step_00055404 --det-profile parallel \
      --det-size 1280 --rec-ckpt data/checkpoints/recognizer_parallel_v3/step_REC --rec-profile parallel
  python -m src.eval.real_battery --compare v4_parallel v5_parallel   # build diff HTML, no models
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import date
from html import escape
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src.corpus.common import PROJECT_ROOT
from src.detect.data.build_obj_targets import CLASSES_V4

PAGES = PROJECT_ROOT / "data" / "real_eval" / "pages"
REPORTS = PROJECT_ROOT / "data" / "real_eval" / "reports"

# stable class -> BGR color for overlays (text-ish warm, structure cool, figures green)
_COLORS = {"title": (60, 76, 231), "heading": (18, 156, 243), "subheading": (133, 160, 22),
           "text": (80, 62, 44), "list_item": (182, 89, 155), "caption": (156, 188, 26),
           "table": (211, 0, 148), "table_head": (219, 152, 52), "table_cell": (173, 68, 142),
           "image": (113, 204, 46), "chart": (96, 174, 39), "signature": (0, 84, 211),
           "hand_drawing": (43, 57, 192), "formula": (94, 73, 52),
           "form_label": (15, 196, 241), "form_value": (133, 187, 101)}


def _overlay(img_bgr: np.ndarray, units: list[dict]) -> np.ndarray:
    im = img_bgr.copy()
    for u in units:
        x1, y1, x2, y2 = (int(v) for v in u["box"])
        name = CLASSES_V4[u["cls"]] if isinstance(u.get("cls"), int) else str(u.get("cls"))
        c = _COLORS.get(name, (128, 128, 128))
        cv2.rectangle(im, (x1, y1), (x2, y2), c, 2)
        cv2.putText(im, name, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1,
                    cv2.LINE_AA)
    return im


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "?"


def run_battery(args) -> None:
    from src.pipeline.run_ocr import load_obj_models, run_ocr_v4   # heavy import; lazy

    device = args.device
    det, rec, enc = load_obj_models(
        _ckpt(args.det_ckpt), _ckpt(args.rec_ckpt), args.det_profile, args.rec_profile, device)
    per_cls = {}
    if args.thresholds and Path(args.thresholds).exists():
        sweep = json.loads(Path(args.thresholds).read_text(encoding="utf-8")).get("sweep") or {}
        per_cls = {k: max(args.thresh_floor, v["best_thresh"]) for k, v in sweep.items()}
        print(f"[battery] per-class thresholds from {args.thresholds} "
              f"(floored at {args.thresh_floor})")
    det_cfg = {"image_size": args.det_size, "score_thresh": args.score_thresh,
               "score_thresh_per_class": per_cls, "deskew": args.deskew,
               "rec_max_w": args.rec_max_w or (1200 if args.rec_profile == "parallel" else 800),
               "gray_crops": True, "decode": args.decode, "upscale_width": 2400,
               "agnostic_nms": True}

    out_dir = REPORTS / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = sorted(p for p in PAGES.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not pages:
        raise SystemExit(f"no battery pages in {PAGES}")

    results = []
    for p in pages:
        t0 = time.time()
        res = run_ocr_v4(Image.open(p), det, rec, enc, det_cfg, device)
        dt = time.time() - t0
        norm = cv2.cvtColor(np.asarray(res["norm_image"].convert("RGB")), cv2.COLOR_RGB2BGR)
        ov = _overlay(norm, res["line_units"])
        scale = min(1.0, 1400 / ov.shape[1])           # keep report images a sane size
        if scale < 1.0:
            ov = cv2.resize(ov, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / f"{p.stem}_overlay.jpg"), ov,
                    [cv2.IMWRITE_JPEG_QUALITY, 82])
        (out_dir / f"{p.stem}.md").write_text(res["markdown"], encoding="utf-8")
        (out_dir / f"{p.stem}.json").write_text(json.dumps({
            "page": p.name, "seconds": round(dt, 2), "n_lines": len(res["line_units"]),
            "n_blocks": len(res["block_units"]), "n_figures": len(res["figures"]),
            "units": [{"box": [round(v, 1) for v in u["box"]],
                       "cls": CLASSES_V4[u["cls"]] if isinstance(u.get("cls"), int) else u.get("cls"),
                       "text": u.get("text", "")} for u in res["line_units"]],
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        results.append({"page": p.name, "stem": p.stem, "seconds": round(dt, 2),
                        "n_lines": len(res["line_units"]), "n_blocks": len(res["block_units"]),
                        "n_figures": len(res["figures"]), "markdown": res["markdown"],
                        "html": res["html"]})
        print(f"[battery] {p.name}: {len(res['line_units'])} lines, "
              f"{len(res['block_units'])} blocks, {len(res['figures'])} figures, {dt:.1f}s")

    meta = {"tag": args.tag, "date": str(date.today()), "git": _git_commit(),
            "det_ckpt": str(args.det_ckpt), "rec_ckpt": str(args.rec_ckpt),
            "det_profile": args.det_profile, "rec_profile": args.rec_profile,
            "det_cfg": {k: v for k, v in det_cfg.items()}}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    _write_report(out_dir, meta, results)
    print(f"\n[battery] report -> {out_dir / 'report.html'}")


_CSS = """
 :root{--bg:#262624;--panel:#1f1e1d;--line:#3a3733;--ink:#ece9e3;--mut:#a8a29a;--accent:#d97757}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);
   font-family:ui-sans-serif,system-ui,'Segoe UI','Khmer OS',sans-serif;line-height:1.55}
 header{padding:26px 36px 10px;max-width:1500px;margin:0 auto}
 h1{margin:0;font-size:24px} .sub{color:var(--mut);font-size:13px;margin-top:4px}
 .page{max-width:1500px;margin:18px auto;padding:0 36px}
 .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}
 .hd{padding:10px 16px;border-bottom:1px solid var(--line);display:flex;gap:14px;
   align-items:baseline} .hd b{font-size:15px} .hd span{color:var(--mut);font-size:12px}
 .cols{display:grid;grid-template-columns:1fr 1fr;gap:0}
 .cols>div{padding:12px;min-width:0} .cols img{width:100%;border-radius:6px}
 pre{white-space:pre-wrap;word-break:break-word;background:#191817;border:1px solid var(--line);
   border-radius:8px;padding:12px;font-size:13px;max-height:760px;overflow:auto;margin:0}
 .lbl{font:600 11px ui-monospace,monospace;color:var(--accent);margin:0 0 6px}
"""


def _write_report(out_dir: Path, meta: dict, results: list[dict]) -> None:
    secs = []
    for r in results:
        secs.append(f"""<div class="page"><div class="card">
  <div class="hd"><b>{escape(r['page'])}</b>
    <span>{r['n_lines']} lines · {r['n_blocks']} blocks · {r['n_figures']} figures · {r['seconds']}s</span></div>
  <div class="cols">
    <div><p class="lbl">DETECTIONS</p><img src="{r['stem']}_overlay.jpg" loading="lazy"></div>
    <div><p class="lbl">ASSEMBLED MARKDOWN</p><pre>{escape(r['markdown'])}</pre></div>
  </div></div></div>""")
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>real battery — {escape(meta['tag'])}</title><style>{_CSS}</style></head><body>
<header><h1>Real-document battery — <span style="color:var(--accent)">{escape(meta['tag'])}</span></h1>
<div class="sub">{meta['date']} · git {meta['git']} · det {escape(Path(meta['det_ckpt']).name)}
({meta['det_profile']}) · rec {escape(Path(meta['rec_ckpt']).name)} ({meta['rec_profile']})
· thresh {meta['det_cfg']['score_thresh']} · size {meta['det_cfg']['image_size']}</div></header>
{''.join(secs)}</body></html>"""
    (out_dir / "report.html").write_text(html, encoding="utf-8")


def compare(tag_a: str, tag_b: str) -> None:
    a, b = REPORTS / tag_a, REPORTS / tag_b
    for d in (a, b):
        if not (d / "meta.json").exists():
            raise SystemExit(f"missing report dir: {d} (run the battery for that tag first)")
    stems = sorted({p.stem.replace("_overlay", "") for p in a.glob("*_overlay.jpg")} &
                   {p.stem.replace("_overlay", "") for p in b.glob("*_overlay.jpg")})
    rows = []
    for s in stems:
        ja = json.loads((a / f"{s}.json").read_text(encoding="utf-8"))
        jb = json.loads((b / f"{s}.json").read_text(encoding="utf-8"))
        md_a = (a / f"{s}.md").read_text(encoding="utf-8")
        md_b = (b / f"{s}.md").read_text(encoding="utf-8")
        rows.append(f"""<div class="page"><div class="card">
  <div class="hd"><b>{escape(s)}</b><span>{tag_a}: {ja['n_lines']} lines · {tag_b}: {jb['n_lines']} lines</span></div>
  <div class="cols">
    <div><p class="lbl">{escape(tag_a.upper())}</p><img src="{tag_a}/{s}_overlay.jpg" loading="lazy">
      <pre>{escape(md_a)}</pre></div>
    <div><p class="lbl">{escape(tag_b.upper())}</p><img src="{tag_b}/{s}_overlay.jpg" loading="lazy">
      <pre>{escape(md_b)}</pre></div>
  </div></div></div>""")
    out = REPORTS / f"compare_{tag_a}_vs_{tag_b}.html"
    out.write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<title>{tag_a} vs {tag_b}</title><style>{_CSS}</style></head><body>
<header><h1>{escape(tag_a)} <span style="color:var(--accent)">vs</span> {escape(tag_b)}</h1>
<div class="sub">real-document battery comparison · overlays + assembled markdown</div></header>
{''.join(rows)}</body></html>""", encoding="utf-8")
    print(f"[battery] comparison -> {out}")


def _ckpt(p: Path) -> Path:
    return p / "model.pt" if p.is_dir() else p


def main() -> None:
    ap = argparse.ArgumentParser(description="Real-document end-to-end battery")
    ap.add_argument("--tag", help="report tag, e.g. v4_parallel")
    ap.add_argument("--det-ckpt", type=Path)
    ap.add_argument("--rec-ckpt", type=Path)
    ap.add_argument("--det-profile", default="single")
    ap.add_argument("--rec-profile", default="single")
    ap.add_argument("--det-size", type=int, default=1024)
    ap.add_argument("--score-thresh", type=float, default=0.3)
    ap.add_argument("--rec-max-w", type=int, default=0, help="0 = auto from rec profile")
    ap.add_argument("--decode", choices=["attn", "ctc"], default="attn")
    ap.add_argument("--thresholds", default="data/real_eval/det_eval_v5_sweep.json",
                    help="det_eval --sweep JSON for per-class thresholds ('' to disable)")
    ap.add_argument("--thresh-floor", type=float, default=0.15,
                    help="clamp swept thresholds up to this (see app.py._load_sweep_thresholds)")
    ap.add_argument("--deskew", action="store_true",
                    help="perspective-correct photographed pages before OCR")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--compare", nargs=2, metavar=("TAG_A", "TAG_B"),
                    help="build comparison HTML from two existing report tags")
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return
    if not (args.tag and args.det_ckpt and args.rec_ckpt):
        ap.error("--tag, --det-ckpt and --rec-ckpt are required (or use --compare)")
    run_battery(args)


if __name__ == "__main__":
    main()
