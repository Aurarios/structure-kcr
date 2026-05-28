"""Corpus source registry + fetch dispatcher.

    python -m src.corpus.registry --list
    python -m src.corpus.registry --fetch [--smoke] [--only name1,name2]

Reads config/sources.yaml and routes each enabled source to the right fetcher by `type`.
"""
from __future__ import annotations

import argparse

from .common import load_sources


def cmd_list() -> None:
    cfg = load_sources()
    rows = cfg.get("sources", [])
    print(f"{'NAME':28} {'TYPE':8} {'ON':3} {'NC':3} LICENSE")
    print("-" * 90)
    for s in rows:
        nc = "yes" if s.get("non_commercial") else ""
        on = "yes" if s.get("enabled") else "no"
        print(f"{s['name']:28} {s.get('type',''):8} {on:3} {nc:3} {s.get('license','')}")
    enabled = [s for s in rows if s.get("enabled")]
    print(f"\n{len(enabled)}/{len(rows)} sources enabled.")


def cmd_fetch(smoke: bool, only: set[str] | None, limit_override: int | None) -> None:
    from . import fetch_dumps, fetch_hf, scrape

    cfg = load_sources()
    limit = 200 if smoke else limit_override
    total = 0
    for s in cfg.get("sources", []):
        if not s.get("enabled"):
            continue
        if only and s["name"] not in only:
            continue
        t = s.get("type")
        try:
            if t == "hf":
                total += fetch_hf.fetch_hf_source(s, limit)
            elif t == "dump":
                total += fetch_dumps.fetch_dump_source(s, limit)
            elif t == "scrape":
                total += scrape.scrape_source(s, limit)
            elif t == "opus":
                print(f"[opus] {s['name']}: OPUS fetcher not yet implemented, skipping")
            else:
                print(f"[?] {s['name']}: unknown type {t!r}, skipping")
        except Exception as e:
            print(f"[!] {s['name']}: error {e}")
    print(f"\nTotal docs written to data/corpus/raw/: {total}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Khmer corpus source registry")
    ap.add_argument("--list", action="store_true", help="list configured sources")
    ap.add_argument("--fetch", action="store_true", help="fetch enabled sources")
    ap.add_argument("--smoke", action="store_true", help="cap each source at 200 docs")
    ap.add_argument("--limit", type=int, default=None, help="cap each source at N docs")
    ap.add_argument("--only", type=str, default=None, help="comma-separated source names")
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    if args.list or not (args.fetch):
        cmd_list()
    if args.fetch:
        cmd_fetch(args.smoke, only, args.limit)


if __name__ == "__main__":
    main()
