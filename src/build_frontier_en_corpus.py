#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cache English generation panels, human documents and encoder embeddings for attribution."""
from __future__ import annotations
import argparse, glob, json, collections
from pathlib import Path
import numpy as np
import pandas as pd


def encode(texts, model_name, batch=32, device=None):
    import torch
    from transformers import AutoTokenizer, AutoModel
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True,
                      max_length=512, return_tensors="pt").to(device)
            h = mdl(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1e-9)).cpu().numpy())
            print(f"  encode {min(i+batch,len(texts))}/{len(texts)}", end="\r")
    print(); del mdl
    return np.vstack(out).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default="results_v2/openrouter_en_s*/per_doc.jsonl")
    ap.add_argument("--humans", default="results_v2/raid_en_humans.parquet")
    ap.add_argument("--encoder", default="microsoft/deberta-v3-base")
    ap.add_argument("--alias", default="deberta-v3-base")
    ap.add_argument("--cache", default=".cache/pipeline")
    ap.add_argument("--max-humans", type=int, default=13250)
    ap.add_argument("--min-per-gen", type=int, default=50)
    ap.add_argument("--name", default="frontier_en",
                    help="corpus name (use frontier_en_clean for the filtered cache)")
    ap.add_argument("--clean-text", action="store_true",
                    help="drop non-prose junk (min 40 words, matches v3_english_clean bundle)")
    ap.add_argument("--min-words", type=int, default=40)
    args = ap.parse_args()
    keep = (lambda t: True)
    if args.clean_text:
        from text_quality import is_clean
        keep = lambda t: is_clean(t, min_words=args.min_words)

    # ---- pool AI across panels, dedup by (model, title) ----
    seen = set(); per_gen = collections.defaultdict(list); dropped = 0
    files = sorted(glob.glob(args.panels))
    if not files:
        raise SystemExit(f"no panels matched {args.panels!r}")
    for f in files:
        for line in open(f, encoding="utf-8"):
            try: d = json.loads(line)
            except Exception: continue
            if d.get("error"): continue
            r = (d.get("response") or "").strip()
            if len(r) < 20: continue
            if not keep(r): dropped += 1; continue
            key = (str(d["model"]), str(d["title"]))
            if key in seen: continue
            seen.add(key); per_gen[str(d["model"])].append(r)
    if args.clean_text: print(f"[clean] dropped {dropped} junk AI docs")
    # drop generators with too few docs (gas needs >= min_per_gen)
    per_gen = {g: v for g, v in per_gen.items() if len(v) >= args.min_per_gen}
    gens_sorted = sorted(per_gen)
    n_ai = sum(len(v) for v in per_gen.values())
    print(f"[ai] {len(files)} panels -> {n_ai} docs across {len(gens_sorted)} generators")
    for g in gens_sorted:
        print(f"    {g:42s} {len(per_gen[g])}")

    ai_texts, ai_gens = [], []
    for g in gens_sorted:
        ai_texts += per_gen[g]; ai_gens += [g] * len(per_gen[g])

    # ---- humans ----
    H = pd.read_parquet(args.humans)
    hum = H["text"].astype(str).tolist()
    if args.clean_text:
        n0 = len(hum); hum = [t for t in hum if keep(t)]
        print(f"[clean] dropped {n0 - len(hum)} junk human docs")
    if len(hum) > args.max_humans:
        hum = list(pd.Series(hum).sample(n=args.max_humans, random_state=0))
    print(f"[human] {len(hum)} English RAID humans")

    # ---- assemble (humans first, then AI) ----
    texts = list(hum) + ai_texts
    labels = [0] * len(hum) + [1] * len(ai_texts)
    gens = ["Human"] * len(hum) + ai_gens

    # ---- encode ----
    print(f"[encode] {len(texts)} docs with {args.encoder}")
    X = encode(texts, args.encoder)

    # ---- write gas.py cache ----
    cache = Path(args.cache); cache.mkdir(parents=True, exist_ok=True)
    corpus = {"name": args.name, "labels": labels, "gens": gens, "texts": texts}
    (cache / f"corpus_{args.name}.json").write_text(
        json.dumps(corpus), encoding="utf-8")
    np.save(cache / f"emb_{args.name}_{args.alias}.npy", X)
    print(f"[done] wrote {cache}/corpus_{args.name}.json "
          f"and emb_{args.name}_{args.alias}.npy  (X={X.shape})")
    print(f"       AI generators (attribution classes): {len(gens_sorted)}")
    print(f"\nNext: python3 gas.py --corpus frontier_en --alias {args.alias}")


if __name__ == "__main__":
    main()
