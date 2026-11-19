#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filter cached documents and aligned embeddings by minimum word count."""
from __future__ import annotations
import argparse, json, collections
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default=str(Path.home() / "glass/.cache/pipeline"))
ap.add_argument("--src", default="frontier_en_clean")
ap.add_argument("--dst", default="frontier_en_long")
ap.add_argument("--alias", default="deberta-v3-base")
ap.add_argument("--min-words", type=int, default=200)
args = ap.parse_args()

cache = Path(args.cache)
info = json.loads((cache / f"corpus_{args.src}.json").read_text())
texts = info["texts"]
labels = np.asarray(info["labels"]); gens = np.asarray(info["gens"])
X = np.load(cache / f"emb_{args.src}_{args.alias}.npy")
assert len(texts) == len(labels) == X.shape[0], "cache misaligned"

wc = np.array([len(t.split()) for t in texts])
keep = wc >= args.min_words
print(f"[filter] {int(keep.sum())}/{len(texts)} docs >= {args.min_words} words "
      f"(median len kept: {int(np.median(wc[keep])) if keep.any() else 0})")

ai_keep = keep & (labels == 1)
cnt = collections.Counter(gens[ai_keep].tolist())
print(f"[per-generator AI docs >= {args.min_words} words]")
for g, c in sorted(cnt.items(), key=lambda x: -x[1]):
    flag = "" if c >= 50 else "   <-- below min-per-gen(50)!"
    print(f"    {g:42s} {c}{flag}")

out = {"name": args.dst,
       "labels": labels[keep].astype(int).tolist(),
       "gens": gens[keep].tolist(),
       "texts": [t for t, k in zip(texts, keep) if k]}
(cache / f"corpus_{args.dst}.json").write_text(json.dumps(out), encoding="utf-8")
np.save(cache / f"emb_{args.dst}_{args.alias}.npy", X[keep])
print(f"[done] wrote corpus_{args.dst}.json + emb_{args.dst}_{args.alias}.npy  (X={X[keep].shape})")
