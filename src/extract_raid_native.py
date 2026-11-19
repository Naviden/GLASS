#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract English, unattacked RAID machine generations as additional generator panels.

Some generator families overlap with evaluation corpora; use these panels only in labelled ablations."""
from __future__ import annotations
import argparse, json, math, collections
from pathlib import Path
from datasets import load_dataset

ENGLISH = {"abstracts", "books", "news", "poetry", "recipes", "reddit", "reviews", "wiki"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="liamdugan/raid")
    ap.add_argument("--split", default="train")
    ap.add_argument("--cap-per-gen", type=int, default=1600, help="docs per generator (stratified over 8 domains)")
    ap.add_argument("--clean-only", action="store_true", default=True, help="keep attack=none rows only")
    ap.add_argument("--out", default="results_v2/raid_native_en/per_doc.jsonl")
    ap.add_argument("--max-rows", type=int, default=12_000_000)
    ap.add_argument("--progress", type=int, default=500_000)
    args = ap.parse_args()

    per_dom_cap = max(1, math.ceil(args.cap_per_gen / len(ENGLISH)))
    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    count = collections.Counter()          # (model, domain) -> kept
    seen = set()                           # (model, title, domain) dedup
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fout = open(args.out, "w", encoding="utf-8")
    n = kept = 0
    for r in ds:
        n += 1
        if n > args.max_rows: break
        if args.progress and n % args.progress == 0:
            print(f"  [stream] {n:,} rows; kept={kept}; gens={len(set(m for m,_ in count))}", flush=True)
        dom = str(r.get("domain") or "")
        if dom not in ENGLISH: continue
        if args.clean_only and str(r.get("attack") or "none").lower() != "none": continue
        m = str(r.get("model") or "")
        if m in ("", "human"): continue
        if count[(m, dom)] >= per_dom_cap: continue
        gen = (r.get("generation") or "").strip()
        if len(gen) < 20: continue
        title = str(r.get("title") or "")
        key = (m, title, dom)
        if key in seen: continue
        seen.add(key); count[(m, dom)] += 1; kept += 1
        fout.write(json.dumps({"model": m, "title": title, "response": gen, "domain": dom}) + "\n")
    fout.close()

    by_gen = collections.Counter()
    for (m, d), c in count.items(): by_gen[m] += c
    print(f"\n[done] scanned {n:,} rows -> {kept} docs across {len(by_gen)} generators")
    for g in sorted(by_gen): print(f"    {g:22s} {by_gen[g]}")
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
