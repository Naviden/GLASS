#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_ghostbuster_attr_v3.py
============================
RE-DO of tab_ghostbuster on English (frontier_en): Ghostbuster-style multi-LM
log-probability features + multinomial logistic regression for N-way generator
attribution, vs closed-form GAS on the same split.

Features per doc: mean / std / min / slope of token log-probs under gpt2 and
distilgpt2 (the same multi-LM primitive the paper's Ghostbuster baseline uses),
concatenated.

Writes tables_v2/tab_ghostbuster_v3.tex.
Run: python3 build_ghostbuster_attr_v3.py     (~1-2 h GPU)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import Counter
import numpy as np


def lm_feats(texts, model_id, dev, maxlen=256, log_every=500):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(model_id).to(dev).eval()
    feats = []
    with torch.no_grad():
        for i, t in enumerate(texts):
            enc = tok(t, return_tensors="pt", truncation=True, max_length=maxlen).to(dev)
            if enc.input_ids.size(1) < 2:
                feats.append([0, 0, 0, 0]); continue
            lp = torch.log_softmax(mdl(**enc).logits[:, :-1, :], -1)
            tgt = enc.input_ids[:, 1:]
            tlp = lp.gather(2, tgt.unsqueeze(-1)).squeeze().float().cpu().numpy()
            tlp = np.atleast_1d(tlp)
            sl = np.polyfit(np.arange(len(tlp)), tlp, 1)[0] if len(tlp) > 1 else 0.0
            feats.append([tlp.mean(), tlp.std(), tlp.min(), float(sl)])
            if (i + 1) % log_every == 0: print(f"    {model_id} {i+1}/{len(texts)}")
    del mdl
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return np.array(feats, dtype=np.float64)


class GAS:
    def __init__(self, k=10): self.k = k; self.subs = {}; self.order = []
    def fit(self, X, g):
        for gg in sorted(set(g.tolist())):
            Xg = X[g == gg]
            if len(Xg) < self.k + 5: continue
            m = Xg.mean(0); Xc = Xg - m
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            self.subs[gg] = (m, Vt[:self.k].T)
        self.order = sorted(self.subs); return self
    def predict(self, X):
        S = np.empty((len(X), len(self.order)))
        for i, gg in enumerate(self.order):
            m, B = self.subs[gg]; Xc = X - m
            S[:, i] = -np.sum((Xc - Xc @ B @ B.T) ** 2, axis=1)
        return np.array(self.order)[S.argmax(1)]


def main():
    here = Path(__file__).resolve().parent
    root = here if (here / "results").exists() else here.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(root / ".cache" / "pipeline"))
    ap.add_argument("--corpus", default="frontier_en")
    ap.add_argument("--alias", default="deberta-v3-base")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nways", default="4,18")
    ap.add_argument("--min-per-gen", type=int, default=50)
    ap.add_argument("--cap", type=int, default=6000, help="cap docs for LM features")
    args = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    cache = Path(args.cache)
    info = json.loads((cache / f"corpus_{args.corpus}.json").read_text())
    y = np.asarray(info["labels"], int); gens = np.asarray(info["gens"])
    texts = info["texts"]; Xemb = np.load(cache / f"emb_{args.corpus}_{args.alias}.npy")
    ai = np.where(y == 1)[0]
    counts = Counter(gens[ai].tolist())
    ranked = sorted([g for g, c in counts.items() if c >= args.min_per_gen],
                    key=lambda g: -counts[g])

    # subsample AI docs (cap) balanced across the full set for LM features
    rng = np.random.default_rng(0)
    full = ranked
    sel = np.concatenate([rng.permutation(np.where(gens == g)[0])[:args.cap // len(full)]
                          for g in full])
    print(f"[ghostbuster] LM features on {len(sel)} docs (gpt2+distilgpt2)")
    F = np.hstack([lm_feats([texts[i] for i in sel], "gpt2", dev),
                   lm_feats([texts[i] for i in sel], "distilgpt2", dev)])
    F = np.nan_to_num(F)
    gsel = gens[sel]; Xsel = Xemb[sel]

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    nways = [len(ranked) if x == "all" else int(x) for x in args.nways.split(",")]
    rows = []
    for N in nways:
        keep = ranked[:N]; m = np.isin(gsel, keep)
        Fk, Xk, gk = F[m], Xsel[m], gsel[m]
        ri = np.random.default_rng(1); idx = ri.permutation(len(Fk))
        h = int(len(idx) * 0.8); tr, te = idx[:h], idx[h:]
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(StandardScaler().fit_transform(Fk[tr]), gk[tr])
        sc = StandardScaler().fit(Fk[tr])
        gb_acc = float(np.mean(clf.predict(sc.transform(Fk[te])) == gk[te]))
        g = GAS(args.k).fit(Xk[tr], gk[tr])
        gas_acc = float(np.mean(g.predict(Xk[te]) == gk[te]))
        rows.append((N, gas_acc, gb_acc, 1.0 / N))
        print(f"  N={N}: GAS={gas_acc:.3f}  Ghostbuster={gb_acc:.3f}  chance={1.0/N:.3f}")

    Path("tables_v2").mkdir(exist_ok=True)
    L = ["% RE-DO Ghostbuster vs GAS attribution (English frontier_en).",
         "\\begin{table}[t]\\centering\\small",
         "\\caption{Top-1 generator attribution on the English panels "
         "(\\texttt{frontier\\_en}): closed-form GAS vs a Ghostbuster-style "
         "multi-LM log-probability classifier.}",
         "\\label{tab:ghostbuster-v3}",
         "\\begin{tabular}{lccc}", "\\toprule",
         "Classes $N$ & GAS & Ghostbuster & Chance \\\\", "\\midrule"]
    for N, ga, gb, ch in rows:
        L.append(f"{N} & {ga:.3f} & {gb:.3f} & {ch:.3f} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    Path("tables_v2/tab_ghostbuster_v3.tex").write_text("\n".join(L) + "\n")
    print("\n[out] tables_v2/tab_ghostbuster_v3.tex")


if __name__ == "__main__":
    main()
