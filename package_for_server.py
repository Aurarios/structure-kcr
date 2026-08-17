"""Bundle everything needed to run the V4 A5000 best-accuracy pipeline into one zip for transfer.

Includes: source tree + the small data payload (lm_corpus, fonts, tokenizer model) + the 204MB
real-image asset pool. Excludes: venv, .git, caches, checkpoints, manifests, the 750MB tokenizer
train dump, and scratch logs/images. See LINUX_A5000_RUNBOOK.md.
"""
import os, sys, zipfile
from pathlib import Path

REPO = Path(r"C:\Users\USER\Desktop\KCR\structure-kcr")
ASSETS = Path("E:/kcr-assets")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("E:/kcr_server_transfer.zip")

SKIP_PARTS = {".venv", ".git", "__pycache__", "node_modules", ".pytest_cache", ".mypy_cache"}
SKIP_DIRS_REL = {"audit_real_doc", "audit_struct_test", "inference_v1", "data/checkpoints",
                 "data/synthetic", "data/real", "data/recognize"}
# under data/, allow ONLY these (everything else under data/ is regenerable or huge)
DATA_ALLOW_PREFIX = ("data/corpus/lm_corpus/", "data/fonts/")
DATA_ALLOW_EXACT = {"data/tokenizer/khmer_ocr.model", "data/tokenizer/khmer_ocr.vocab"}


def keep(rel: str) -> bool:
    parts = rel.split("/")
    if any(p in SKIP_PARTS for p in parts):
        return False
    if any(rel == d or rel.startswith(d + "/") for d in SKIP_DIRS_REL):
        return False
    if rel.startswith("data/"):
        return rel.startswith(DATA_ALLOW_PREFIX) or rel in DATA_ALLOW_EXACT
    # root-level scratch
    if "/" not in rel:
        if rel.endswith((".log", ".jpg", ".jpeg", ".png", ".zip")):
            return False
        if rel.startswith("_"):           # _v4_infer_demo.py, _cer_render.log, etc.
            return False
    if "manifests" in parts[0:2] and rel.startswith("data/"):
        return False
    return True


def main():
    n, total = 0, 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for root, dirs, files in os.walk(REPO):
            rel_root = Path(root).relative_to(REPO).as_posix()
            dirs[:] = [d for d in dirs if (d not in SKIP_PARTS)]
            for f in files:
                rel = f if rel_root == "." else f"{rel_root}/{f}"
                if not keep(rel):
                    continue
                fp = Path(root) / f
                try:
                    sz = fp.stat().st_size
                except OSError:
                    continue
                z.write(fp, f"structure-kcr/{rel}")
                n += 1; total += sz
        # bundle the asset pool at top level -> extract to /mnt/DATA_3/kcr/assets on the box
        if ASSETS.exists():
            for root, dirs, files in os.walk(ASSETS):
                rr = Path(root).relative_to(ASSETS).as_posix()
                for f in files:
                    if f.endswith(".log"):
                        continue
                    rel = f if rr == "." else f"{rr}/{f}"
                    fp = Path(root) / f
                    z.write(fp, f"kcr-assets/{rel}")
                    n += 1; total += fp.stat().st_size
    print(f"wrote {OUT}  ({n} files, {total/1e9:.2f} GB uncompressed, "
          f"{OUT.stat().st_size/1e9:.2f} GB zipped)")


if __name__ == "__main__":
    main()
