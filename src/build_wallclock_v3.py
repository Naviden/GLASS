#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_wallclock_v3.py
=====================
Measure GLASS per-document scoring throughput with the v3_english bundle
(18 subspaces), to back the GLASS row of tab_wallclock. Reports ms/doc and
docs/sec for: one DeBERTa-v3-base encoder pass + whitening + (|L|+1)
low-rank residual projections. Also reports the one-time fit cost (whitener
+ all SVDs).

Competitor rows in the table are inference-cost estimates (number of LLM
forward passes per document x measured throughput), documented as such.

Run (turing, ~/glass/paper, venv, GPU; pin a free GPU):
    CUDA_VISIBLE_DEVICES=1 PYTHONUTF8=1 python3 build_wallclock_v3.py
"""
from __future__ import annotations
import argparse, json, glob, time
from pathlib import Path
import numpy as np


def residual_sq(X, mean, B):
    Xc = X - mean
    return np.sum((Xc - Xc @ B @ B.T) ** 2, axis=1)


def load_bundle(base, K):
    base = Path(base)
    sh = np.load(base / "S_H_universal.npz"); SH = (sh["mean"], sh["basis"][:, :K])
    SA = {p.stem[4:]: (np.load(p)["mean"], np.load(p)["basis"][:, :K])
          for p in sorted((base / "subspaces").glob("S_A_*.npz"))}
    wh = np.load(base / "whitener.npz"); WH = (wh["mean"], wh["W"])
    return SH, SA, WH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default="results_v2/openrouter_en_s*/per_doc.jsonl")
    ap.add_argument("--bundle", default=".cache/glass_production/v3_english")
    ap.add_argument("--encoder", default="microsoft/deberta-v3-base")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--n", type=int, default=2000, help="docs to time")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # sample N docs
    texts = []
    for f in sorted(glob.glob(args.panels)):
        for line in open(f, encoding="utf-8"):
            try: d = json.loads(line)
            except Exception: continue
            r = (d.get("response") or "").strip()
            if len(r) >= 20: texts.append(r)
            if len(texts) >= args.n: break
        if len(texts) >= args.n: break
    texts = texts[:args.n]
    print(f"[timing] {len(texts)} docs, bundle {args.bundle}")

    SH, SA, WH = load_bundle(args.bundle, args.K)
    print(f"[bundle] {len(SA)} per-generator subspaces (K={args.K})")

    tok = AutoTokenizer.from_pretrained(args.encoder)
    mdl = AutoModel.from_pretrained(args.encoder).to(dev).eval()

    # warmup
    with torch.no_grad():
        e = tok(texts[:args.batch], padding=True, truncation=True, max_length=512,
                return_tensors="pt").to(dev)
        _ = mdl(**e).last_hidden_state
    if dev == "cuda": torch.cuda.synchronize()

    t0 = time.time()
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), args.batch):
            e = tok(texts[i:i + args.batch], padding=True, truncation=True,
                    max_length=512, return_tensors="pt").to(dev)
            h = mdl(**e).last_hidden_state
            m = e["attention_mask"].unsqueeze(-1).float()
            embs.append(((h * m).sum(1) / m.sum(1).clamp(min=1e-9)).cpu().numpy())
    if dev == "cuda": torch.cuda.synchronize()
    t_enc = time.time() - t0
    X = np.vstack(embs).astype(np.float64)

    t1 = time.time()
    Xw = (X - WH[0]) @ WH[1].T
    rh = residual_sq(Xw, SH[0], SH[1])
    allr = np.stack([residual_sq(Xw, m, b) for m, b in SA.values()], 0)
    _ = rh - allr.min(0)
    t_score = time.time() - t1

    total = t_enc + t_score
    ms_doc = 1000.0 * total / len(texts)
    dps = len(texts) / total
    print(f"\n[result] encode {t_enc:.2f}s + score {t_score:.3f}s = {total:.2f}s")
    print(f"         {ms_doc:.3f} ms/doc   {dps:.0f} docs/sec   "
          f"(GLASS, v3_english, K={args.K}, {len(SA)} subspaces)")
    print(f"[paste] GLASS (v3\\_english, $K{{=}}10$, whitened, best-of) & "
          f"{ms_doc:.3f} & {dps:.0f} & $1\\times$ (reference) \\\\")


if __name__ == "__main__":
    main()
