"""Targeted scraper for Khmer news/blog/gov sites.

Etiquette is mandatory: robots.txt is always honored, requests are rate-limited per host, and we
store extracted text only (never redistribute raw articles). Each `scrape`-typed source in
sources.yaml lists seed RSS/index URLs; we expand them to article URLs and extract main text with
trafilatura.
"""
from __future__ import annotations

import time
import urllib.robotparser
import xml.etree.ElementTree as ET
from typing import Any, Iterable
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from .common import RAW_DIR, Doc, ShardWriter

_UA = "khmer-ocr-corpus/0.1 (research; +https://example.org/contact)"
_robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}


def _robots(url: str) -> urllib.robotparser.RobotFileParser:
    base = "{0.scheme}://{0.netloc}".format(urlparse(url))
    if base not in _robots_cache:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(base + "/robots.txt")
        try:
            rp.read()
        except Exception:
            pass  # if robots is unreachable, default-allow but still rate-limit
        _robots_cache[base] = rp
    return _robots_cache[base]


def _allowed(url: str) -> bool:
    return _robots(url).can_fetch(_UA, url)


def _get(url: str, timeout: int = 20) -> str | None:
    try:
        r = requests.get(url, headers={"User-Agent": _UA}, timeout=timeout)
        if r.status_code == 200 and r.text:
            return r.text
    except requests.RequestException:
        return None
    return None


def _rss_links(feed_xml: str) -> list[str]:
    """Extract item/entry links from an RSS or Atom feed."""
    links: list[str] = []
    try:
        root = ET.fromstring(feed_xml)
    except ET.ParseError:
        return links
    # RSS: channel/item/link ; Atom: entry/link[@href]
    for item in root.iter():
        tag = item.tag.split("}")[-1]
        if tag == "item":
            link = item.find("link")
            if link is not None and link.text:
                links.append(link.text.strip())
        elif tag == "entry":
            for link in item.findall("{*}link"):
                href = link.get("href")
                if href:
                    links.append(href.strip())
    return links


def scrape_source(src: dict[str, Any], limit: int | None = None) -> int:
    name = src["name"]
    seeds: Iterable[str] = src.get("seeds", [])
    rate = float(src.get("rate_limit_seconds", 2.0))
    max_per_seed = int(src.get("max_pages_per_seed", 200))
    written = 0

    with ShardWriter(RAW_DIR, name) as w:
        for seed in seeds:
            if not _allowed(seed):
                print(f"[scrape] {name}: robots.txt disallows {seed}, skipping")
                continue
            feed = _get(seed)
            time.sleep(rate)
            if not feed:
                print(f"[scrape] {name}: could not fetch seed {seed}")
                continue
            article_urls = _rss_links(feed)[:max_per_seed]
            print(f"[scrape] {name}: {len(article_urls)} article links from {seed}")
            for url in tqdm(article_urls, desc=name, unit="art"):
                if not _allowed(url):
                    continue
                html = _get(url)
                time.sleep(rate)
                if not html:
                    continue
                text = _extract_main(html, url)
                if not text:
                    continue
                w.write(Doc(text=text, source=name, license=src.get("license", "scraped"), url=url))
                written += 1
                if limit and written >= limit:
                    print(f"[scrape] {name}: hit limit {limit}")
                    return written
    print(f"[scrape] {name}: wrote {written} docs")
    return written


def _extract_main(html: str, url: str) -> str | None:
    try:
        import trafilatura
        text = trafilatura.extract(html, url=url, favor_recall=True)
        return text.strip() if text else None
    except Exception:
        # fallback: strip tags with BeautifulSoup
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "lxml")
            for t in soup(["script", "style", "nav", "header", "footer"]):
                t.decompose()
            return soup.get_text("\n").strip() or None
        except Exception:
            return None
