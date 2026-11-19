#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract English RAID human documents and associated prompts by domain and title."""
from __future__ import annotations
import argparse
import collections
from pathlib import Path

import pandas as pd
from datasets import load_dataset

ENGLISH = {"abstracts", "books", "news", "poetry",
           "recipes", "reddit", "reviews", "wiki"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="liamdugan/raid")
    ap.add_argument("--split", default="train")
    ap.add_argument("--cap-per-domain", type=int, default=5000)
    ap.add_argument("--out-humans", default="results_v2/raid_en_humans.parquet")
    ap.add_argument("--out-prompts", default="results_v2/raid_en_prompts.parquet")
    ap.add_argument("--progress", type=int, default=500_000)
    args = ap.parse_args()

    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    humans = []
    seen_titles = collections.defaultdict(set)   # domain -> {title}
    per_dom = collections.Counter()
    prompt_by_key = {}                           # (domain, title) -> real prompt
    closed = set()
    prev_dom = None
    n = 0

    for r in ds:
        n += 1
        if args.progress and n % args.progress == 0:
            cov = len(prompt_by_key)
            print(f"  [stream] {n:,} rows; humans={dict(per_dom)}; "
                  f"prompts_seen={cov}; closed={sorted(closed)}", flush=True)

        dom = str(r.get("domain") or "")
        if dom != prev_dom:
            if prev_dom in ENGLISH and per_dom[prev_dom] > 0:
                closed.add(prev_dom)
            prev_dom = dom
            if ENGLISH.issubset(closed):
                print(f"  [stream] all English domains passed at row {n:,}")
                break

        if dom not in ENGLISH:
            continue

        title = str(r.get("title") or "")

        # capture RAID's genuine prompt for this (domain, title) from any row
        # that carries one (machine rows do)
        p = r.get("prompt")
        if title and p not in (None, "") and (dom, title) not in prompt_by_key:
            prompt_by_key[(dom, title)] = str(p)

        if r.get("model") != "human":
            continue
        if per_dom[dom] >= args.cap_per_domain:
            continue
        if not title or title in seen_titles[dom]:
            continue

        seen_titles[dom].add(title)
        per_dom[dom] += 1
        humans.append({
            "id":        r.get("id"),
            "source_id": r.get("source_id"),
            "domain":    dom,
            "title":     title,
            "prompt":    None,   # filled from prompt_by_key after the loop
            "text":      r.get("generation"),
        })

    # attach the real RAID prompt to each human document
    for h in humans:
        h["prompt"] = prompt_by_key.get((h["domain"], h["title"]))

    print(f"[done] scanned {n:,} rows; humans per domain: {dict(per_dom)}")
    if not humans:
        raise SystemExit("no English-domain humans found; check split/domains")

    cov = sum(1 for h in humans if h["prompt"])
    print(f"[prompts] {cov}/{len(humans)} humans matched a REAL RAID prompt "
          f"({100*cov/max(len(humans),1):.1f}%)")
    if cov:
        ex = next(h for h in humans if h["prompt"])
        print(f"[prompts] example ({ex['domain']}): {ex['prompt'][:200]}")

    hd = pd.DataFrame(humans)
    Path(args.out_humans).parent.mkdir(parents=True, exist_ok=True)
    hd.to_parquet(args.out_humans, index=False)

    pp = (hd[["domain", "title", "prompt"]]
          .dropna(subset=["title"])
          .drop_duplicates()
          .reset_index(drop=True))
    pp.to_parquet(args.out_prompts, index=False)

    print(f"[out] {args.out_humans}  ({len(hd)} humans)")
    print(f"[out] {args.out_prompts}  ({len(pp)} unique prompts)")
    print("\nprompt pool per domain:")
    print(pp["domain"].value_counts())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
