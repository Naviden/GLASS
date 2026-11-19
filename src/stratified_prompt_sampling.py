#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sample RAID prompts within domains using equal or proportional allocations."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd


# Reuse openrouter_query's template fallback when RAID has no verbatim prompt.
# (Imported there already; duplicated here so this file runs standalone.)
DOMAIN_TEMPLATES = {
    "abstracts": "Write a scientific abstract about: {title}",
    "books":     "Write the opening paragraphs of a book chapter titled: {title}",
    "news":      "Write a news article with the headline: {title}",
    "poetry":    "Write a poem on the theme: {title}",
    "recipes":   "Write a detailed recipe for: {title}",
    "reddit":    "Write a Reddit post about: {title}",
    "reviews":   "Write a product or media review of: {title}",
    "wiki":      "Write the opening section of a Wikipedia article about: {title}",
    "czech_news":  "Write a news article (in English) about: {title}",
    "german_news": "Write a news article (in English) about: {title}",
    "_default":  "Write a {domain} entry titled: {title}",
}


def _synthesise_prompt(title: str, domain: str) -> str:
    tpl = DOMAIN_TEMPLATES.get(domain.strip().lower(), DOMAIN_TEMPLATES["_default"])
    return tpl.format(title=title.strip(), domain=domain.strip() or "article")


def allocate_budget(domain_sizes: Dict[str, int], N: int, mode: str) -> Dict[str, int]:
    """Split N across domains, capped by availability, redistributing any
    leftover (from undersized domains) round-robin to domains with spare room.
    Pure integer arithmetic; deterministic given inputs."""
    keys = sorted(domain_sizes)
    if mode == "proportional":
        total = sum(domain_sizes.values())
        quota = {d: int(round(N * domain_sizes[d] / total)) for d in keys}
    else:  # equal
        base = N // len(keys)
        quota = {d: base for d in keys}
    # cap by availability
    quota = {d: min(quota[d], domain_sizes[d]) for d in keys}

    def short() -> int:
        return N - sum(quota.values())

    # fill remaining budget round-robin among domains that still have capacity
    guard = 0
    while short() > 0 and guard < 10_000:
        guard += 1
        progressed = False
        for d in keys:
            if short() == 0:
                break
            if quota[d] < domain_sizes[d]:
                quota[d] += 1
                progressed = True
        if not progressed:
            break  # not enough unique prompts anywhere to reach N
    return quota


def load_prompts_from_raid(args) -> List[dict]:
    """Domain-stratified prompt sampler. Same return shape as the original:
    a list of dicts with keys prompt_idx, title, domain, prompt_text."""
    meta = pd.read_parquet(args.metadata_path)
    have_prompt = "prompt" in meta.columns

    # one row per unique prompt/title
    group_col = "title" if "title" in meta.columns else "doc_id"
    grouped = meta.groupby(group_col).first().reset_index()
    if "title" not in grouped.columns or "domain" not in grouped.columns:
        raise SystemExit("metadata needs (title, domain) columns to stratify")
    # Keep rows with a usable title. The prompt is used verbatim when present
    # and long enough, otherwise synthesised from (title, domain) in the output
    # loop below -- RAID human rows carry no prompt, so synthesis is the norm.
    grouped = grouped[grouped["title"].notna() & (grouped["title"].str.len() > 3)]
    grouped = grouped.reset_index(drop=True)
    if len(grouped) == 0:
        raise SystemExit("no usable prompts after filtering")

    rng = np.random.default_rng(args.seed)
    N = min(args.n_prompts, len(grouped))

    # bucket row positions by (lowercased) domain
    buckets: Dict[str, List[int]] = {}
    doms = grouped["domain"].astype(str).str.strip().str.lower()
    for ridx, dom in enumerate(doms):
        buckets.setdefault(dom, []).append(ridx)

    quota = allocate_budget({d: len(v) for d, v in buckets.items()}, N, args.alloc)

    chosen: List[int] = []
    for d in sorted(buckets):
        idxs = np.asarray(buckets[d])
        k = quota.get(d, 0)
        if k <= 0:
            continue
        pick = rng.choice(len(idxs), size=k, replace=False)
        chosen.extend(idxs[pick].tolist())
    rng.shuffle(chosen)  # avoid domain-blocked ordering

    out: List[dict] = []
    for i in chosen:
        r = grouped.iloc[int(i)]
        title = str(r.get("title", ""))
        domain = str(r.get("domain", ""))
        if have_prompt and isinstance(r.get("prompt"), str) and len(r["prompt"]) > 20:
            ptext = str(r["prompt"])
        else:
            ptext = _synthesise_prompt(title, domain)
        out.append({"prompt_idx": int(i), "title": title,
                    "domain": domain, "prompt_text": ptext})

    realized = Counter(o["domain"].strip().lower() for o in out)
    print(f"[stratified seed={args.seed} alloc={args.alloc}] {len(out)} prompts; "
          f"per-domain: {dict(sorted(realized.items()))}")
    return out


def add_args(ap: argparse.ArgumentParser) -> None:
    """The two flags to add to openrouter_query.py's parser (option A)."""
    ap.add_argument("--stratify", action="store_true",
                    help="use domain-stratified sampling instead of a flat draw")
    ap.add_argument("--alloc", choices=["equal", "proportional"], default="equal",
                    help="how to split the prompt budget across domains")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-path", default="results_v2/raid_metadata.parquet")
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("prompts_stratified.parquet"))
    add_args(ap)
    args = ap.parse_args()
    rows = load_prompts_from_raid(args)
    df = pd.DataFrame(rows)
    df.to_parquet(args.out, index=False)
    df.to_json(args.out.with_suffix(".json"), orient="records", indent=2)
    print(f"[out] {args.out}  ({len(df)} prompts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
