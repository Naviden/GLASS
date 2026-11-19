#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate generator attribution and kernel-bandwidth calibration."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import Counter
import numpy as np


class GAS:
    def __init__(self, k=10, bw_q=0.5):
        self.k = k; self.bw_q = bw_q; self.subs = {}; self.bw = 0.0; self.order = []

    def fit(self, X, gens):
        res = []
        for g in sorted(set(gens.tolist())):
            Xg = X[gens == g]
            if len(Xg) < self.k + 5: continue
            mean = Xg.mean(0); Xc = Xg - mean
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            B = Vt[:self.k].T
            self.subs[g] = (mean, B)
            res.extend(np.linalg.norm(Xc - Xc @ B @ B.T, axis=1).tolist())
        self.bw = float(np.quantile(res, self.bw_q)) if res else 1.0
        self.order = sorted(self.subs); return self

    def _logscores(self, X):
        S = np.empty((len(X), len(self.order)))
        for i, g in enumerate(self.order):
            m, B = self.subs[g]; Xc = X - m
            S[:, i] = -np.sum((Xc - Xc @ B @ B.T) ** 2, axis=1) / (self.bw ** 2 + 1e-12)
        return S

    def predict(self, X):
        return np.array(self.order)[self._logscores(X).argmax(1)]

    def predict_proba(self, X):
        S = self._logscores(X); S -= S.max(1, keepdims=True)
        P = np.exp(S); P /= P.sum(1, keepdims=True)
        idx = P.argmax(1)
        return np.array(self.order)[idx], P[np.arange(len(P)), idx]


def cv_top1(X, gens, keep, k=10, folds=5, seed=42):
    mask = np.isin(gens, keep); X = X[mask]; gens = gens[mask]
    rng = np.random.default_rng(seed); idx = rng.permutation(len(X))
    parts = np.array_split(idx, folds); accs = []
    for f in range(folds):
        te = parts[f]; tr = np.concatenate([parts[j] for j in range(folds) if j != f])
        g = GAS(k).fit(X[tr], gens[tr])
        if not g.order: continue
        pred = g.predict(X[te]); ok = np.isin(gens[te], g.order)
        if ok.sum() == 0: continue
        accs.append(float((pred[ok] == gens[te][ok]).mean()))
    return (float(np.mean(accs)), float(np.std(accs))) if accs else (float("nan"), 0.0)


def ece(conf, correct, nbins=10):
    bins = np.linspace(0, 1, nbins + 1); e = 0.0; n = len(conf)
    for b in range(nbins):
        m = (conf >= bins[b]) & (conf < bins[b + 1]) if b < nbins - 1 else \
            (conf >= bins[b]) & (conf <= bins[b + 1])
        if m.sum() == 0: continue
        e += (m.sum() / n) * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def cv_ece(X, gens, keep, k=10, bw_q=0.5, folds=5, seed=42):
    mask = np.isin(gens, keep); X = X[mask]; gens = gens[mask]
    rng = np.random.default_rng(seed); idx = rng.permutation(len(X))
    parts = np.array_split(idx, folds); conf = []; corr = []
    for f in range(folds):
        te = parts[f]; tr = np.concatenate([parts[j] for j in range(folds) if j != f])
        g = GAS(k, bw_q).fit(X[tr], gens[tr])
        if not g.order: continue
        pred, p = g.predict_proba(X[te]); ok = np.isin(gens[te], g.order)
        conf += p[ok].tolist(); corr += (pred[ok] == gens[te][ok]).astype(float).tolist()
    if not conf: return float("nan")
    return ece(np.array(conf), np.array(corr))


def main():
    here = Path(__file__).resolve().parent
    root = here if (here / "results").exists() else here.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(root / ".cache" / "pipeline"))
    ap.add_argument("--corpus", default="frontier_en")
    ap.add_argument("--alias", default="deberta-v3-base")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nways", default="4,8,all")
    ap.add_argument("--sigmas", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--min-per-gen", type=int, default=50)
    args = ap.parse_args()

    cache = Path(args.cache)
    info = json.loads((cache / f"corpus_{args.corpus}.json").read_text())
    y = np.asarray(info["labels"], int); gens = np.asarray(info["gens"])
    X = np.load(cache / f"emb_{args.corpus}_{args.alias}.npy")
    ai = y == 1; Xai = X[ai]; gai = gens[ai]
    counts = Counter(gai.tolist())
    ranked = sorted([g for g, c in counts.items() if c >= args.min_per_gen],
                    key=lambda g: -counts[g])
    print(f"[attr] {len(ranked)} generators >= {args.min_per_gen} docs, n_ai={int(ai.sum())}")
    Path("tables_v2").mkdir(exist_ok=True)

    # ---- N-way attribution ----
    nways = [len(ranked) if x == "all" else int(x) for x in args.nways.split(",")]
    L = ["% RE-DO N-way GAS attribution (English frontier_en).",
         "\\begin{table}[t]\\centering\\small",
         "\\caption{Closed-form GAS top-1 attribution accuracy among $N$ English "
         "frontier generators (\\texttt{frontier\\_en}, 5-fold CV).}",
         "\\label{tab:frontier-4way-v3}",
         "\\begin{tabular}{lccc}", "\\toprule",
         "Classes $N$ & Top-1 & Chance & $\\times$chance \\\\", "\\midrule"]
    for N in nways:
        m, s = cv_top1(Xai, gai, ranked[:N], args.k); ch = 1.0 / N
        L.append(f"{N} & {m:.3f}$\\pm${s:.3f} & {ch:.3f} & {m/ch:.1f} \\\\")
        print(f"  N={N:>3}  top1={m:.3f}+-{s:.3f}  x{m/ch:.1f}")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    Path("tables_v2/tab_frontier_4way_v3.tex").write_text("\n".join(L) + "\n")

    # ---- bandwidth (sigma) sweep: ECE (top-1 is sigma-invariant) ----
    sigs = [float(x) for x in args.sigmas.split(",")]
    L = ["% RE-DO GAS bandwidth (sigma) sweep on attribution CALIBRATION.",
         "% Top-1 is bandwidth-invariant; we report 10-bin ECE of max-prob.",
         "\\begin{table}[t]\\centering\\small",
         "\\caption{Calibration of GAS attribution probabilities vs the kernel "
         "bandwidth quantile $q$ on the full English panel (\\texttt{frontier\\_en}, "
         "5-fold CV): 10-bin ECE of the top-class probability. (Top-1 accuracy is "
         "invariant to $q$.)}",
         "\\label{tab:calibration-sigma-v3}",
         "\\begin{tabular}{l" + "c" * len(sigs) + "}", "\\toprule",
         "Bandwidth $q$ & " + " & ".join(f"{q:g}" for q in sigs) + " \\\\", "\\midrule"]
    cells = []
    for q in sigs:
        e = cv_ece(Xai, gai, ranked, args.k, bw_q=q); cells.append(f"{e:.3f}")
        print(f"  sigma_q={q}  ECE={e:.3f}")
    L.append("ECE $\\downarrow$ & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    Path("tables_v2/tab_calibration_sigma_v3.tex").write_text("\n".join(L) + "\n")
    print("\n[out] tables_v2/tab_frontier_4way_v3.tex, tables_v2/tab_calibration_sigma_v3.tex")


if __name__ == "__main__":
    main()
