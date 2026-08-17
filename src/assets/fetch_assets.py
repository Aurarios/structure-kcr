r"""Download license-clean real images for synthetic-document figure/signature/stamp regions.

Why: V1/V2 painted CSS gradients as "images", so the detector learned "smooth blob = image" — a
cue that does not transfer to real photos. V3 embeds REAL images.

Sources (default `--source hybrid`):
  * **HuggingFace datasets** (reliable, CDN, no rate limit) for the bulk categories — photos
    (COCO/Flickr), portraits (CelebA), logos, charts (ChartQA), drawings (WikiArt). Streamed, so
    only consumed images download. Licenses are mixed (some research/non-commercial) and recorded
    per asset; the user opted into HF as a source.
  * **Wikimedia Commons** (https://commons.wikimedia.org) — all-free media (CC0/PD/CC-BY/CC-BY-SA),
    license-cleanest — for categories HF doesn't cover (signatures, stamps) and to supplement.
    The upload server hard-throttles bursts (HTTP 429); we honor it and fail-fast per category.
Also `--source hf`, `--source wikimedia`, `--source openverse` (needs OPENVERSE_CLIENT_ID/SECRET).
Provenance (source, license, url) is recorded for every asset in `assets.jsonl`.

Output (default root E:/kcr-assets):
    {root}/{category}/{sha12}.jpg         downloaded, RGB, long-side <= --max-px
    {root}/{category}/_index.json         list of sha in this category (fast pool load)
    {root}/assets.jsonl                   one provenance row per saved asset

Idempotent: re-runs skip categories already at --limit and never re-download a known url/sha.

Usage:
    python -m src.assets.fetch_assets --limit 1500              # full pool per category
    python -m src.assets.fetch_assets --smoke                   # ~40 per category (wire-up test)
    python -m src.assets.fetch_assets --only photos,portraits --limit 500
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
from pathlib import Path

import requests
from PIL import Image, ImageOps

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
OPENVERSE_API = "https://api.openverse.org/v1/images/"
# Honest identification (NOT a spoofed browser UA). Wikimedia requires a descriptive UA.
USER_AGENT = "structure-kcr-ocr/0.3 (Khmer OCR research dataset; contact via project repo)"

DEFAULT_ROOT = Path("E:/kcr-assets")

# Per-category search queries. Each category collects a varied pool the sampler reuses (with
# augmentation) across the 350K pages, so a few hundred-to-thousand per category is plenty.
CATEGORY_QUERIES: dict[str, list[str]] = {
    "photos": ["landscape", "city street", "people working", "computer technology",
               "food meal", "wildlife animal", "building architecture", "market scene",
               "agriculture rice field", "classroom students", "traffic vehicle", "factory machine"],
    "portraits": ["portrait man", "portrait woman", "headshot face person",
                  "passport style portrait", "studio portrait person"],
    "signatures": ["signature", "autograph", "handwritten signature ink", "signature document"],
    "stamps": ["rubber stamp", "official seal", "wax seal", "postal cancellation stamp",
               "ink stamp impression"],
    "drawings": ["pencil sketch", "ink drawing illustration", "technical drawing diagram",
                 "hand drawn map", "engineering blueprint"],
    "logos": ["logo", "coat of arms", "flag", "emblem badge", "monogram"],
    "charts": ["bar chart", "line chart graph", "pie chart", "statistical diagram",
               "flow chart diagram"],
}


# HuggingFace sources per category (Parquet datasets only — datasets>=4 dropped script loading).
# Streamed, so only the consumed images are downloaded. License recorded per dataset (mixed:
# COCO/Flickr photos, CelebA faces = research/non-commercial; wikiart ~PD; logos = generated). The
# user opted into HF as a source; provenance keeps every asset's origin auditable.
CATEGORY_HF: dict[str, list[tuple[str, str, str, str]]] = {
    "photos":    [("detection-datasets/coco", "train", "image", "flickr-cc/research"),
                  ("jxie/flickr8k", "train", "image", "flickr/research")],
    "portraits": [("tpremoli/CelebA-attrs", "train", "image", "celeba-research-noncommercial")],
    "logos":     [("logo-wizard/modern-logo-dataset", "train", "image", "generated"),
                  ("iamkaikai/amazing_logos_v4", "train", "image", "generated")],
    "charts":    [("HuggingFaceM4/ChartQA", "train", "image", "research")],
    "drawings":  [("huggan/wikiart", "train", "image", "public-domain/fair-use")],
}


# HF detection datasets we crop region-of-interest from (category_id -> our category). Signatures
# have no cropped-image dataset on HF, but Francesco/signatures-xc8up has signature bounding boxes
# in document images, so we crop them out to get real signature images.
CATEGORY_HF_CROP: dict[str, list[tuple[str, str, int]]] = {
    "signatures": [("Francesco/signatures-xc8up", "train", 1)],   # category 1 = signature
}


def _is_free_license(lic: str | None) -> bool:
    """Accept CC0 / public-domain / CC-BY / CC-BY-SA (all free, commercial-OK); reject the rest."""
    if not lic:
        return False
    l = lic.lower()
    return l.startswith(("cc0", "cc-by", "pd", "pdm")) or "public domain" in l


# ---------------------------------------------------------------------------- sources

def _session(source: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    if source == "openverse":
        tok = _openverse_token()
        if tok:
            s.headers["Authorization"] = f"Bearer {tok}"
    return s


def _openverse_token() -> str | None:
    cid, secret = os.environ.get("OPENVERSE_CLIENT_ID"), os.environ.get("OPENVERSE_CLIENT_SECRET")
    if not (cid and secret):
        print("[fetch] OPENVERSE_CLIENT_ID/SECRET unset; anonymous Openverse now returns 401")
        return None
    try:
        r = requests.post("https://api.openverse.org/v1/auth_tokens/token/",
                          data={"grant_type": "client_credentials", "client_id": cid,
                                "client_secret": secret}, timeout=30)
        if r.status_code == 200:
            print("[fetch] using authenticated Openverse token")
            return r.json().get("access_token")
    except requests.RequestException:
        pass
    return None


def _search_wikimedia(sess: requests.Session, query: str, offset: int, limit: int) -> list[dict]:
    """One Commons File-namespace search page -> list of candidate {url,license,source,...}."""
    params = {"action": "query", "format": "json", "generator": "search",
              "gsrsearch": f"filetype:bitmap {query}", "gsrnamespace": "6",
              "gsrlimit": str(limit), "gsroffset": str(offset),
              "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": "1024"}
    for attempt in range(4):
        try:
            r = sess.get(WIKIMEDIA_API, params=params, timeout=30)
        except requests.RequestException as e:
            print(f"    [warn] network ({e}); retry"); time.sleep(2 * (attempt + 1)); continue
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 5 * (attempt + 1)))
            print(f"    [rate-limit] waiting {wait}s"); time.sleep(wait); continue
        if r.status_code != 200:
            print(f"    [warn] HTTP {r.status_code} for '{query}'"); return []
        try:
            data = r.json()
        except ValueError:
            return []
        pages = (data.get("query") or {}).get("pages") or {}
        cands = []
        for pg in pages.values():
            ii = (pg.get("imageinfo") or [{}])[0]
            em = ii.get("extmetadata") or {}
            lic = (em.get("License") or {}).get("value") or \
                  (em.get("LicenseShortName") or {}).get("value")
            if not _is_free_license(lic):
                continue
            url = ii.get("thumburl") or ii.get("url")
            if not url:
                continue
            cands.append({"url": url, "license": lic, "source": "wikimedia_commons",
                          "title": pg.get("title"), "creator": (em.get("Artist") or {}).get("value"),
                          "landing": ii.get("descriptionurl"), "query": query})
        return cands
    return []


def _search_openverse(sess: requests.Session, query: str, page: int, limit: int) -> list[dict]:
    params = {"q": query, "license": "cc0,pdm,by", "page": page, "page_size": limit,
              "mature": "false"}
    try:
        r = sess.get(OPENVERSE_API, params=params, timeout=30)
    except requests.RequestException:
        return []
    if r.status_code != 200:
        print(f"    [warn] openverse HTTP {r.status_code}"); return []
    res = (r.json().get("results") or [])
    return [{"url": x.get("url"), "license": x.get("license"), "source": x.get("source"),
             "title": x.get("title"), "creator": x.get("creator"),
             "landing": x.get("foreign_landing_url"), "query": query} for x in res if x.get("url")]


def search(sess: requests.Session, source: str, query: str, page: int, limit: int) -> list[dict]:
    if source == "openverse":
        return _search_openverse(sess, query, page, limit)
    return _search_wikimedia(sess, query, offset=(page - 1) * limit, limit=limit)


# ---------------------------------------------------------------------------- download

def _download_image(sess: requests.Session, url: str, max_px: int) -> bytes | None:
    """Fetch + validate + normalize (RGB, long-side<=max_px) -> JPEG bytes, or None if unusable.

    Honors 429 (the upload server rate-limits bursts and returns a tiny HTML error page that would
    otherwise pass as 'content') with a short backoff, and rejects non-image content-types.
    """
    cooldowns = [10, 25, 50]                       # upload server hard-throttles bursts; real cooldown
    for attempt in range(3):
        try:
            r = sess.get(url, timeout=45)
        except requests.RequestException:
            return None
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", cooldowns[min(attempt, 2)]))
            print(f"    [dl 429] cooldown {wait}s")
            time.sleep(wait)
            continue
        if r.status_code != 200:
            return None
        if not (r.headers.get("Content-Type", "").startswith("image/")):
            return None                                # HTML error page / redirect, not an image
        if len(r.content) < 1024:
            return None
        try:
            im = ImageOps.exif_transpose(Image.open(io.BytesIO(r.content))).convert("RGB")
        except Exception:
            return None
        break
    else:
        return None
    w, h = im.size
    if min(w, h) < 64:
        return None
    if max(w, h) > max_px:
        s = max_px / max(w, h)
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return buf.getvalue()


def _load_seen(root: Path) -> tuple[set[str], set[str]]:
    urls, shas = set(), set()
    jl = root / "assets.jsonl"
    if jl.exists():
        for line in jl.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("url"):
                urls.add(row["url"])
            if row.get("sha"):
                shas.add(row["sha"])
    return urls, shas


def _category_count(root: Path, category: str) -> int:
    d = root / category
    return len(list(d.glob("*.jpg"))) if d.exists() else 0


def fetch_category(sess: requests.Session, source: str, root: Path, category: str,
                   queries: list[str], limit: int, max_px: int, seen_urls: set[str],
                   seen_shas: set[str], throttle: float, prov_fp) -> int:
    """Download up to `limit` images for one category; returns count of NEW images saved."""
    out_dir = root / category
    out_dir.mkdir(parents=True, exist_ok=True)
    have = _category_count(root, category)
    if have >= limit:
        print(f"  [{category}] already have {have} >= {limit}; skip")
        return 0
    target = limit - have
    saved = 0
    page = 1
    empty_pages = 0
    consec_fail = 0                                 # fail-fast if the host is hard-throttling us
    while saved < target and empty_pages < 3:
        got_any = False
        for query in queries:
            if saved >= target or consec_fail >= 12:
                break
            cands = search(sess, source, query, page, limit=40)
            if not cands:
                continue
            got_any = True
            for c in cands:
                if saved >= target or consec_fail >= 12:
                    break
                url = c["url"]
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                blob = _download_image(sess, url, max_px)
                time.sleep(throttle)
                if blob is None:
                    consec_fail += 1
                    continue
                consec_fail = 0
                sha = hashlib.sha256(blob).hexdigest()[:16]
                if sha in seen_shas:
                    continue
                seen_shas.add(sha)
                (out_dir / f"{sha[:12]}.jpg").write_bytes(blob)
                prov = {"category": category, "sha": sha, **c}
                prov_fp.write(json.dumps(prov, ensure_ascii=False) + "\n")
                prov_fp.flush()
                saved += 1
                if saved % 25 == 0:
                    print(f"  [{category}] +{saved}/{target} (total {have + saved})")
        if consec_fail >= 12:
            print(f"  [{category}] host throttling ({consec_fail} fails); skipping (try later)")
            break
        empty_pages = empty_pages + 1 if not got_any else 0
        page += 1
    print(f"  [{category}] done: +{saved} new (total {have + saved})")
    return saved


def _pil_to_jpeg(im, max_px: int) -> bytes | None:
    """A PIL image (already in memory, e.g. from a HF dataset) -> normalized JPEG bytes."""
    try:
        im = ImageOps.exif_transpose(im).convert("RGB")
    except Exception:
        return None
    w, h = im.size
    if min(w, h) < 48:
        return None
    if max(w, h) > max_px:
        s = max_px / max(w, h)
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return buf.getvalue()


def fetch_hf_category(root: Path, category: str, sources: list[tuple[str, str, str, str]],
                      limit: int, max_px: int, seen_shas: set[str], prov_fp) -> int:
    """Stream HuggingFace dataset(s) for a category until `limit` images are saved. Reliable (CDN,
    no rate limit). Returns count of NEW images saved."""
    try:
        from datasets import load_dataset
    except ImportError:
        print(f"  [{category}] datasets lib missing; skip HF"); return 0
    out_dir = root / category
    out_dir.mkdir(parents=True, exist_ok=True)
    have = _category_count(root, category)
    if have >= limit:
        print(f"  [{category}] already have {have} >= {limit}; skip HF")
        return 0
    target = limit - have
    saved = 0
    for dsid, split, key, lic in sources:
        if saved >= target:
            break
        try:
            ds = load_dataset(dsid, split=split, streaming=True)
        except Exception as e:
            print(f"  [{category}] HF {dsid} load failed: {type(e).__name__}; next"); continue
        print(f"  [{category}] streaming {dsid} ...")
        for ex in ds:
            if saved >= target:
                break
            im = ex.get(key)
            if im is None:
                im = next((v for v in ex.values() if hasattr(v, "size") and hasattr(v, "convert")), None)
            if im is None:
                continue
            blob = _pil_to_jpeg(im, max_px)
            if blob is None:
                continue
            sha = hashlib.sha256(blob).hexdigest()[:16]
            if sha in seen_shas:
                continue
            seen_shas.add(sha)
            (out_dir / f"{sha[:12]}.jpg").write_bytes(blob)
            prov_fp.write(json.dumps({"category": category, "sha": sha, "source": f"hf:{dsid}",
                                      "license": lic, "url": None, "query": split},
                                     ensure_ascii=False) + "\n")
            prov_fp.flush()
            saved += 1
            if saved % 50 == 0:
                print(f"  [{category}] +{saved}/{target} (total {have + saved})")
    print(f"  [{category}] HF done: +{saved} new (total {have + saved})")
    return saved


def fetch_hf_cropped(root: Path, category: str, sources: list[tuple[str, str, int]], limit: int,
                     max_px: int, seen_shas: set[str], prov_fp) -> int:
    """Crop region-of-interest (e.g. signatures) out of HF detection datasets -> real cropped assets."""
    try:
        from datasets import load_dataset
    except ImportError:
        return 0
    out_dir = root / category
    out_dir.mkdir(parents=True, exist_ok=True)
    have = _category_count(root, category)
    if have >= limit:
        print(f"  [{category}] already have {have} >= {limit}; skip HF-crop")
        return 0
    target = limit - have
    saved = 0
    for dsid, split, want_cat in sources:
        if saved >= target:
            break
        try:
            ds = load_dataset(dsid, split=split, streaming=True)
        except Exception as e:
            print(f"  [{category}] HF-crop {dsid} load failed: {type(e).__name__}; next"); continue
        print(f"  [{category}] cropping ROIs from {dsid} ...")
        for ex in ds:
            if saved >= target:
                break
            im = ex.get("image")
            obj = ex.get("objects") or {}
            bboxes, cats = obj.get("bbox") or [], obj.get("category") or []
            if im is None or not bboxes:
                continue
            W, H = im.size
            for bb, cat in zip(bboxes, cats):
                if saved >= target:
                    break
                if want_cat is not None and cat != want_cat:
                    continue
                x, y, w, h = bb                         # COCO xywh
                if w < 24 or h < 12:
                    continue
                px = max(2, int(w * 0.06)); py = max(2, int(h * 0.12))
                crop = im.crop((max(0, x - px), max(0, y - py),
                                min(W, x + w + px), min(H, y + h + py)))
                blob = _pil_to_jpeg(crop, max_px)
                if blob is None:
                    continue
                sha = hashlib.sha256(blob).hexdigest()[:16]
                if sha in seen_shas:
                    continue
                seen_shas.add(sha)
                (out_dir / f"{sha[:12]}.jpg").write_bytes(blob)
                prov_fp.write(json.dumps({"category": category, "sha": sha,
                                          "source": f"hf-crop:{dsid}", "license": "see-card",
                                          "url": None, "query": f"cat{want_cat}"},
                                         ensure_ascii=False) + "\n")
                prov_fp.flush()
                saved += 1
                if saved % 50 == 0:
                    print(f"  [{category}] +{saved}/{target} (total {have + saved})")
    print(f"  [{category}] HF-crop done: +{saved} new (total {have + saved})")
    return saved


def write_index(root: Path, category: str) -> int:
    d = root / category
    shas = sorted(p.stem for p in d.glob("*.jpg")) if d.exists() else []
    (d / "_index.json").write_text(json.dumps(shas), encoding="utf-8")
    return len(shas)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch license-clean real images for V3 synth docs")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--limit", type=int, default=1500, help="target images PER category")
    ap.add_argument("--only", default="", help="comma list of categories (default: all)")
    ap.add_argument("--source", default="hybrid",
                    choices=["hybrid", "hf", "wikimedia", "openverse"],
                    help="hybrid = HuggingFace for covered categories + Wikimedia for the rest")
    ap.add_argument("--max-px", type=int, default=1024, help="downscale long side to this")
    ap.add_argument("--throttle", type=float, default=0.2, help="seconds between web downloads")
    ap.add_argument("--smoke", action="store_true", help="~40 per category (wire-up test)")
    args = ap.parse_args()

    limit = 40 if args.smoke else args.limit
    cats = [c.strip() for c in args.only.split(",") if c.strip()] or list(CATEGORY_QUERIES)
    bad = [c for c in cats if c not in CATEGORY_QUERIES]
    if bad:
        ap.error(f"unknown categories: {bad}; valid: {list(CATEGORY_QUERIES)}")

    args.root.mkdir(parents=True, exist_ok=True)
    seen_urls, seen_shas = _load_seen(args.root)
    print(f"[fetch] source={args.source} root={args.root} limit={limit}/cat categories={cats}\n"
          f"[fetch] known: {len(seen_urls)} urls, {len(seen_shas)} shas")

    use_hf = args.source in ("hf", "hybrid")
    use_web = args.source in ("wikimedia", "openverse", "hybrid")
    web_source = "openverse" if args.source == "openverse" else "wikimedia"
    sess = _session(web_source) if use_web else None
    total_new = 0
    with open(args.root / "assets.jsonl", "a", encoding="utf-8") as prov_fp:
        # 1) HuggingFace first (reliable) for the categories it covers
        if use_hf:
            for cat in cats:
                if cat in CATEGORY_HF:
                    total_new += fetch_hf_category(args.root, cat, CATEGORY_HF[cat], limit,
                                                   args.max_px, seen_shas, prov_fp)
                if cat in CATEGORY_HF_CROP:
                    total_new += fetch_hf_cropped(args.root, cat, CATEGORY_HF_CROP[cat], limit,
                                                  args.max_px, seen_shas, prov_fp)
        # 2) Web (Wikimedia/Openverse) for categories still short of target (e.g. signatures, stamps),
        #    license-cleanest; fail-fast if the host is throttling us.
        if use_web:
            for cat in cats:
                if _category_count(args.root, cat) >= limit:
                    continue
                total_new += fetch_category(sess, web_source, args.root, cat, CATEGORY_QUERIES[cat],
                                            limit, args.max_px, seen_urls, seen_shas, args.throttle,
                                            prov_fp)

    print("\n[fetch] index per category:")
    for cat in cats:
        print(f"  {cat:12s} {write_index(args.root, cat):6d} images")
    print(f"[fetch] total new this run: {total_new}")


if __name__ == "__main__":
    main()
