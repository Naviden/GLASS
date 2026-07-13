#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_subspace_v2_english.py
============================
Build a NEW GLASS subspace bundle (default version 'v2_english') whose AI side
is fit on the EIGHT English RAID domains, pooling the five English stratified
panels' generations, and whose human side S_H is fit on the English RAID humans.

Mirrors glass_production_fit.py's bundle layout so product_eval.py loads it as a
drop-in:
    <out-root>/<version>/
        S_H_universal.npz        (mean, basis [d x K_max], svals)   -- raw/unwhitened
        whitener.npz             (mean, W [d x d])                  -- ZCA
        subspaces/S_A_<gen>.npz  per generator                      -- raw/unwhitened
        meta.json

Run (turing, ~/glass/paper, GPU):
    python3 build_subspace_v2_english.py \
        --panels 'results_v2/openrouter_en_s*/per_doc.jsonl' \
        --humans results_v2/raid_en_humans.parquet
"""
from __future__ import annotations
import argparse, glob, json, re, time, collections
from pathlib import Path
import numpy as np
import pandas as pd


def fit_subspace(X, k):
    mean = X.mean(0); Xc = X - mean
    _, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(k, Vt.shape[0])
    return mean, Vt[:k].T, sv[:k]


def whiten_fit(X, ridge=1e-3):
    """Cholesky/Mahalanobis whitener (mean, W=L^-1), matching the headline
    glass_eval_panel scoring. Subspaces are fit in this whitened space."""
    mean = X.mean(0); Xc = X - mean
    cov = (Xc.T @ Xc) / max(1, len(Xc) - 1)
    cov = cov + ridge * (np.trace(cov) / cov.shape[0]) * np.eye(cov.shape[0])
    L = np.linalg.cholesky(cov)
    return mean, np.linalg.solve(L, np.eye(L.shape[0]))


def whiten_apply(X, mean, W):
    return (X - mean) @ W.T


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
            e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            out.append(e.cpu().numpy())
            print(f"  encode {min(i + batch, len(texts))}/{len(texts)}", end="\r")
    print()
    return np.vstack(out).astype(np.float64)


def safe(name):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default="results_v2/openrouter_en_s*/per_doc.jsonl")
    ap.add_argument("--humans", default="results_v2/raid_en_humans.parquet")
    ap.add_argument("--encoder", default="microsoft/deberta-v3-base")
    ap.add_argument("--alias", default="deberta-v3-base")
    ap.add_argument("--K-max", type=int, default=128)
    ap.add_argument("--min-train-per-gen", type=int, default=50)
    ap.add_argument("--sh-humans", type=int, default=8000)
    ap.add_argument("--version", default="v2_english")
    ap.add_argument("--out-root", default=".cache/glass_production")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean-text", action="store_true",
                    help="drop non-prose junk (short/corrupted/boilerplate) from the fitting data")
    args = ap.parse_args()
    keep = (lambda t: True)
    if args.clean_text:
        from text_quality import is_clean as keep

    # ---- pool AI docs across the panels, dedup by (model, title) ----
    seen = set(); per_gen = collections.defaultdict(list); dropped = 0
    files = sorted(glob.glob(args.panels))
    if not files:
        raise SystemExit(f"no panel files matched {args.panels!r}")
    for f in files:
        for line in open(f):
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
    n_ai = sum(len(v) for v in per_gen.values())
    print(f"[ai] {len(files)} panels -> {n_ai} unique docs across {len(per_gen)} generators")
    for g in sorted(per_gen):
        print(f"    {g:42s} {len(per_gen[g])}")

    # ---- human pool for S_H ----
    H = pd.read_parquet(args.humans)
    hum = H["text"].astype(str)
    if args.clean_text:
        before = len(hum); hum = hum[hum.map(keep)]
        print(f"[clean] dropped {before - len(hum)} junk human docs")
    if len(hum) > args.sh_humans:
        hum = hum.sample(n=args.sh_humans, random_state=args.seed)
    hum = list(hum)
    print(f"[human] {len(hum)} English RAID humans for S_H")

    # ---- encode everything once ----
    gens = sorted(per_gen)
    offs = {}; o = 0; ai_texts = []
    for g in gens:
        offs[g] = (o, o + len(per_gen[g])); o += len(per_gen[g]); ai_texts += per_gen[g]
    X = encode(hum + ai_texts, args.encoder)
    Xh = X[:len(hum)]; Xai = X[len(hum):]

    base = Path(args.out_root) / args.version
    (base / "subspaces").mkdir(parents=True, exist_ok=True)

    # ---- whitener (Cholesky on human pool); subspaces fit in whitened space ----
    wm, W = whiten_fit(Xh)
    np.savez(base / "whitener.npz", mean=wm, W=W)
    Xhw = whiten_apply(Xh, wm, W)
    Xaiw = whiten_apply(Xai, wm, W)

    # ---- S_H (whitened) ----
    m, b, s = fit_subspace(Xhw, args.K_max)
    np.savez(base / "S_H_universal.npz", mean=m, basis=b, svals=s)

    # ---- per-generator AI subspaces (whitened) ----
    kept = []
    for g in gens:
        a, bnd = offs[g]; Xg = Xaiw[a:bnd]
        if len(Xg) < args.min_train_per_gen:
            print(f"  skip {g} ({len(Xg)} < min {args.min_train_per_gen})"); continue
        mm, bb, sv = fit_subspace(Xg, args.K_max)
        np.savez(base / "subspaces" / f"S_A_{safe(g)}.npz", mean=mm, basis=bb, svals=sv)
        kept.append(g)

    meta = {
        "version": args.version, "alias": args.alias, "K_max": args.K_max,
        "build_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "human_pool": {"corpora": ["raid_en_humans"], "n_total": len(hum),
                       "domains": "8 English RAID domains "
                                  "(abstracts,books,news,poetry,recipes,reddit,reviews,wiki)"},
        "S_A_library": {"corpora": ["frontier_english_panels"], "generators": kept,
                        "n_per_gen": {g: len(per_gen[g]) for g in kept},
                        "min_train_per_gen": args.min_train_per_gen,
                        "source": "pooled English stratified panels, 8 RAID domains"},
        "split_seed": args.seed,
    }
    json.dump(meta, open(base / "meta.json", "w"), indent=2)
    print(f"\n[done] bundle written: {base}")
    print(f"       S_H ({len(hum)} humans) + {len(kept)} English per-generator subspaces")
    print(f"       generators: {kept}")


if __name__ == "__main__":
    main()
