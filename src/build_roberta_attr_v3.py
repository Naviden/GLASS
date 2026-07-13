#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_roberta_attr_v3.py
========================
RE-DO of tab_gas_vs_roberta + attribution_headtohead on English (frontier_en):
fine-tuned RoBERTa-base N-way generator attribution vs closed-form GAS, same
train/test split.

Writes:
  tables_v2/tab_gas_vs_roberta_v3.tex   (GAS vs fine-tuned RoBERTa top-1)
  tables_v2/attribution_headtohead_v3.tex (GAS vs RoBERTa, no-training note)

Reads gas cache: corpus_frontier_en.json (texts/labels/gens) +
emb_frontier_en_<alias>.npy (for GAS).

Run: python3 build_roberta_attr_v3.py        (~30-60 min GPU)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import Counter
import numpy as np


# ---------------- GAS (closed form) ----------------
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


def split_idx(gens, keep, seed=42, test_frac=0.2):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for g in keep:
        idx = np.where(gens == g)[0]; rng.shuffle(idx)
        n = max(1, int(len(idx) * test_frac))
        te += idx[:n].tolist(); tr += idx[n:].tolist()
    return np.array(tr), np.array(te)


def roberta_attr(texts, y, tr, te, classes, epochs=3, bs=16, maxlen=256, dev="cuda"):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    c2i = {c: i for i, c in enumerate(classes)}
    tok = AutoTokenizer.from_pretrained("roberta-base")
    def enc(ix):
        e = tok([texts[i] for i in ix], padding=True, truncation=True,
                max_length=maxlen, return_tensors="pt")
        lab = torch.tensor([c2i[y[i]] for i in ix])
        return TensorDataset(e["input_ids"], e["attention_mask"], lab)
    dl = DataLoader(enc(tr), batch_size=bs, shuffle=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=len(classes)).to(dev)
    opt = torch.optim.AdamW(mdl.parameters(), lr=2e-5)
    mdl.train()
    for ep in range(epochs):
        for ii, am, lb in dl:
            opt.zero_grad()
            out = mdl(input_ids=ii.to(dev), attention_mask=am.to(dev), labels=lb.to(dev))
            out.loss.backward(); opt.step()
        print(f"    roberta epoch {ep+1}/{epochs} done")
    mdl.eval(); preds = []
    dlt = DataLoader(enc(te), batch_size=64)
    with torch.no_grad():
        for ii, am, lb in dlt:
            logit = mdl(input_ids=ii.to(dev), attention_mask=am.to(dev)).logits
            preds += logit.argmax(1).cpu().tolist()
    del mdl
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gold = [c2i[y[i]] for i in te]
    return float(np.mean(np.array(preds) == np.array(gold)))


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
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    cache = Path(args.cache)
    info = json.loads((cache / f"corpus_{args.corpus}.json").read_text())
    y = np.asarray(info["labels"], int); gens = np.asarray(info["gens"])
    texts = info["texts"]
    X = np.load(cache / f"emb_{args.corpus}_{args.alias}.npy")
    ai = np.where(y == 1)[0]
    counts = Counter(gens[ai].tolist())
    ranked = sorted([g for g, c in counts.items() if c >= args.min_per_gen],
                    key=lambda g: -counts[g])
    nways = [len(ranked) if x == "all" else int(x) for x in args.nways.split(",")]

    rows = []
    for N in nways:
        keep = ranked[:N]
        sel = np.isin(gens, keep) & (y == 1)
        gsel = gens[sel]
        tr_l, te_l = split_idx(gens, keep)  # global indices restricted to keep
        tr = [i for i in tr_l if y[i] == 1]; te = [i for i in te_l if y[i] == 1]
        # GAS
        g = GAS(args.k).fit(X[tr], gens[np.array(tr)])
        pg = g.predict(X[np.array(te)])
        gas_acc = float(np.mean(pg == gens[np.array(te)]))
        # RoBERTa
        rob_acc = roberta_attr(texts, gens, tr, te, keep, epochs=args.epochs, dev=dev)
        rows.append((N, gas_acc, rob_acc, 1.0 / N))
        print(f"  N={N}: GAS={gas_acc:.3f}  RoBERTa={rob_acc:.3f}  chance={1.0/N:.3f}")

    Path("tables_v2").mkdir(exist_ok=True)
    L = ["% RE-DO GAS vs fine-tuned RoBERTa attribution (English frontier_en).",
         "\\begin{table}[t]\\centering\\small",
         "\\caption{Top-1 generator attribution on the English panels "
         "(\\texttt{frontier\\_en}): closed-form GAS (no training) vs fine-tuned "
         "RoBERTa-base.}",
         "\\label{tab:gas-vs-roberta-v3}",
         "\\begin{tabular}{lccc}", "\\toprule",
         "Classes $N$ & GAS (no train) & RoBERTa (fine-tuned) & Chance \\\\", "\\midrule"]
    for N, ga, ra, ch in rows:
        L.append(f"{N} & {ga:.3f} & {ra:.3f} & {ch:.3f} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    Path("tables_v2/tab_gas_vs_roberta_v3.tex").write_text("\n".join(L) + "\n")
    # head-to-head = same data, framed as no-training vs full fine-tune
    Path("tables_v2/attribution_headtohead_v3.tex").write_text(
        "\n".join(L).replace("gas-vs-roberta-v3", "attribution-headtohead-v3") + "\n")
    print("\n[out] tab_gas_vs_roberta_v3.tex, attribution_headtohead_v3.tex")


if __name__ == "__main__":
    main()
