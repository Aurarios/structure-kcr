"""Download + parse non-HF dumps: MediaWiki XML (Wikipedia/Wiktionary) and Leipzig corpora.

Kept dependency-light: MediaWiki parsing uses mwparserfromhell to strip markup; Leipzig packages
are simple <id>\t<sentence> .tar.gz archives.
"""
from __future__ import annotations

import bz2
import io
import re
import tarfile
import urllib.request
from typing import Any, Iterator

from tqdm import tqdm

from .common import RAW_DIR, Doc, ShardWriter

_UA = "khmer-ocr-corpus/0.1 (research; contact: phengsamnanggit@gmail.com)"


def _download(url: str, dest) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        bar = tqdm(total=total, unit="B", unit_scale=True, desc="download")
        while chunk := resp.read(1 << 16):
            f.write(chunk)
            bar.update(len(chunk))
        bar.close()


# --- MediaWiki XML dump -----------------------------------------------------

def _iter_mediawiki_pages(bz2_path) -> Iterator[str]:
    """Yield raw wikitext for each <page> in a *-pages-articles.xml.bz2 dump."""
    text_re = re.compile(r"<text[^>]*>(.*?)</text>", re.DOTALL)
    with bz2.open(bz2_path, "rt", encoding="utf-8") as f:
        buf = []
        for line in f:
            buf.append(line)
            if "</page>" in line:
                page = "".join(buf)
                buf = []
                m = text_re.search(page)
                if m:
                    yield m.group(1)


def _strip_wikitext(wikitext: str) -> str:
    try:
        import mwparserfromhell
        return mwparserfromhell.parse(wikitext).strip_code()
    except Exception:
        # crude fallback
        t = re.sub(r"\{\{.*?\}\}", "", wikitext, flags=re.DOTALL)
        t = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", t)
        t = re.sub(r"<[^>]+>", "", t)
        return t


def fetch_mediawiki(src: dict[str, Any], limit: int | None = None) -> int:
    name = src["name"]
    url = src["url"]
    dest = RAW_DIR / f"{name}.xml.bz2"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"[dump] {name}: downloading {url}")
        _download(url, dest)
    written = 0
    with ShardWriter(RAW_DIR, name) as w:
        for wikitext in tqdm(_iter_mediawiki_pages(dest), desc=name, unit="page"):
            text = _strip_wikitext(wikitext).strip()
            if not text:
                continue
            w.write(Doc(text=text, source=name, license=src.get("license", "CC BY-SA 4.0"), url=url))
            written += 1
            if limit and written >= limit:
                break
    print(f"[dump] {name}: wrote {written} docs")
    return written


# --- Leipzig corpora --------------------------------------------------------

def fetch_leipzig(src: dict[str, Any], limit: int | None = None) -> int:
    name = src["name"]
    url = src["url"]
    dest = RAW_DIR / f"{name}.tar.gz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"[dump] {name}: downloading {url}")
        _download(url, dest)
    written = 0
    with ShardWriter(RAW_DIR, name) as w, tarfile.open(dest, "r:gz") as tar:
        member = next((m for m in tar.getmembers() if m.name.endswith("-sentences.txt")), None)
        if member is None:
            print(f"[dump] {name}: no *-sentences.txt in archive")
            return 0
        fh = tar.extractfile(member)
        for raw in tqdm(io.TextIOWrapper(fh, encoding="utf-8"), desc=name, unit="sent"):
            # format: "<id>\t<sentence>"
            parts = raw.rstrip("\n").split("\t", 1)
            if len(parts) != 2 or not parts[1].strip():
                continue
            w.write(Doc(text=parts[1].strip(), source=name, license=src.get("license", "CC BY 4.0"), url=url))
            written += 1
            if limit and written >= limit:
                break
    print(f"[dump] {name}: wrote {written} docs")
    return written


def fetch_dump_source(src: dict[str, Any], limit: int | None = None) -> int:
    parser = src.get("parser")
    if parser == "mediawiki":
        return fetch_mediawiki(src, limit)
    if parser == "leipzig_sentences":
        return fetch_leipzig(src, limit)
    print(f"[dump] {src['name']}: unknown parser {parser!r}, skipping")
    return 0
