#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fit and evaluate GLASS on an English panel using a title-based train/test split."""
from __future__ import annotations
import argparse, json
import numpy as np
import pandas as pd


# ---- GLASS primitives (identical to glass_variants.py) ----------------------
def fit_subspace(X, k):
    mean = X.mean(0); Xc = X - mean
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return mean, Vt[: min(k, Vt.shape[0])].T


def residual_sq(X, mean, B):
    Xc = X - mean
    proj = Xc @ B @ B.T
    return np.sum((Xc - proj) ** 2, axis=1)


def whiten_fit(X, ridge=1e-3):
    mean = X.mean(0); Xc = X - mean
    cov = (Xc.T @ Xc) / max(1, len(Xc) - 1)
    cov = cov + ridge * (np.trace(cov) / cov.shape[0]) * np.eye(cov.shape[0])
    L = np.linalg.cholesky(cov)
    W = np.linalg.solve(L, np.eye(L.shape[0]))
    return mean, W


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
            emb = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            out.append(emb.cpu().numpy())
            print(f"  encode {min(i + batch, len(texts))}/{len(texts)}", end="\r")
    print()
    return np.vstack(out).astype(np.float64)


def auc_tpr(scores, y):
    from sklearn.metrics import roc_auc_score, roc_curve
    auc = float(roc_auc_score(y, scores))
    fpr, tpr, _ = roc_curve(y, scores)
    tpr1 = float(np.interp(0.01, fpr, tpr))
    return auc, tpr1


def bootstrap(scores, y, n=1000, seed=0):
    from sklearn.metrics import roc_auc_score, roc_curve
    rng = np.random.default_rng(seed); N = len(scores)
    a, t = [], []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        yy = y[idx]
        if yy.sum() == 0 or (1 - yy).sum() == 0:
            continue
        ss = scores[idx]
        a.append(roc_auc_score(yy, ss))
        fpr, tpr, _ = roc_curve(yy, ss)
        t.append(np.interp(0.01, fpr, tpr))
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)),
            float(np.percentile(t, 2.5)), float(np.percentile(t, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--humans", default="results_v2/raid_en_humans.parquet")
    ap.add_argument("--encoder", default="microsoft/deberta-v3-base")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--sh-humans", type=int, default=4000,
                    help="background humans (disjoint from panel) to fit S_H")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--clean-text", action="store_true",
                    help="drop non-prose junk from fitting+eval (min 40 words, all 8 domains kept)")
    ap.add_argument("--min-words", type=int, default=40)
    args = ap.parse_args()
    keep = (lambda t: True)
    if args.clean_text:
        from text_quality import is_clean
        keep = lambda t: is_clean(t, min_words=args.min_words)

    # AI generations
    ai = []
    for line in open(args.jsonl):
        try: d = json.loads(line)
        except Exception: continue
        if d.get("error"): continue
        r = (d.get("response") or "").strip()
        if len(r) < 20: continue
        ai.append({"text": r, "gen": str(d["model"]),
                   "title": str(d["title"]), "domain": str(d.get("domain", ""))})
    ai = pd.DataFrame(ai)
    titles = sorted(ai["title"].unique())
    rng = np.random.default_rng(args.seed)
    n_te = max(1, int(round(len(titles) * args.test_frac)))
    test_titles = set(rng.choice(titles, size=n_te, replace=False))
    ai_tr = ai[~ai["title"].isin(test_titles)].reset_index(drop=True)
    ai_te = ai[ai["title"].isin(test_titles)].reset_index(drop=True)
    # clean-text filters the FITTING split only; the held-out test set is left
    # identical to what the competitor scorers see, so the comparison stays fair.
    if args.clean_text:
        n0a = len(ai_tr); ai_tr = ai_tr[ai_tr["text"].map(keep)].reset_index(drop=True)
        print(f"[clean] AI fit split kept {len(ai_tr)}/{n0a} (test set left unfiltered)")
    print(f"[data] {len(ai)} AI docs, {ai['gen'].nunique()} generators, "
          f"{len(titles)} titles ({len(test_titles)} test)")

    # humans
    H = pd.read_parquet(args.humans)
    htext = dict(zip(H["title"].astype(str), H["text"].astype(str)))
    panel_titles = set(titles)
    test_hum = [htext[t] for t in test_titles if t in htext]   # test set: NOT filtered
    bg = H[~H["title"].astype(str).isin(panel_titles)]
    if args.clean_text:                                        # S_H fit pool: filtered
        n0h = len(bg); bg = bg[bg["text"].astype(str).map(keep)]
        print(f"[clean] S_H background humans kept {len(bg)}/{n0h} (test humans left unfiltered)")
    bg = bg.sample(n=min(args.sh_humans, len(bg)), random_state=args.seed)
    print(f"[data] {len(test_hum)} test humans, {len(bg)} background humans (S_H)")

    # encode once
    texts = list(bg["text"].astype(str)) + list(ai_tr["text"]) \
        + list(ai_te["text"]) + list(test_hum)
    X = encode(texts, args.encoder)
    n0 = len(bg); n1 = n0 + len(ai_tr); n2 = n1 + len(ai_te)
    Xbg, Xtr, Xte_ai, Xte_hu = X[:n0], X[n0:n1], X[n1:n2], X[n2:]

    # whiten on background humans; fit S_H and per-gen S_A in whitened space
    wm, W = whiten_fit(Xbg)
    Wbg = whiten_apply(Xbg, wm, W)
    hm, hb = fit_subspace(Wbg, args.k)
    gen_tr = ai_tr["gen"].to_numpy()
    SA = {}
    Wtr = whiten_apply(Xtr, wm, W)
    for g in sorted(set(gen_tr)):
        idx = np.where(gen_tr == g)[0]
        if len(idx) >= args.k + 1:
            SA[g] = fit_subspace(Wtr[idx], args.k)
    print(f"[fit] S_H + {len(SA)} per-generator subspaces (k={args.k})")

    def score(Xraw):
        Xw = whiten_apply(Xraw, wm, W)
        rh = residual_sq(Xw, hm, hb)
        allr = np.stack([residual_sq(Xw, m, b) for (m, b) in SA.values()], 0)
        return rh - allr.min(0)

    s = np.concatenate([score(Xte_ai), score(Xte_hu)])
    y = np.concatenate([np.ones(len(Xte_ai)), np.zeros(len(Xte_hu))])
    auc, tpr1 = auc_tpr(s, y)
    alo, ahi, tlo, thi = bootstrap(s, y)
    print(f"\n[RESULT] panel={args.jsonl}")
    print(f"  AUC = {auc:.4f}  [95% CI {alo:.4f}, {ahi:.4f}]")
    print(f"  TPR@1%FPR = {tpr1:.4f}  [95% CI {tlo:.4f}, {thi:.4f}]")
    print(f"  test: {int(y.sum())} AI vs {int((1-y).sum())} human; "
          f"{len(SA)} generators in library")

    res = {"jsonl": args.jsonl, "auc": auc, "auc_ci": [alo, ahi],
           "tpr_at_1": tpr1, "tpr_ci": [tlo, thi],
           "n_ai_test": int(y.sum()), "n_human_test": int((1 - y).sum()),
           "n_generators": len(SA), "k": args.k}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
