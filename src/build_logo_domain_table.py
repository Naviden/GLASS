#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarise leave-one-generator-out detection by RAID domain."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DET_ORDER = ["GLASS", "Fast-DetectGPT", "Binoculars", "Ghostbuster", "log-rank", "RoBERTa-OpenAI"]
DET_SHORT = {"GLASS": "\\textbf{GLASS}", "Fast-DetectGPT": "Fast-DetectGPT",
             "Binoculars": "Binoculars", "Ghostbuster": "Ghostbuster",
             "log-rank": "log-rank", "RoBERTa-OpenAI": "RoBERTa"}
DOM_PRETTY = {"abstracts": "Abstracts", "books": "Books", "news": "News",
              "poetry": "Poetry", "recipes": "Recipes", "reddit": "Reddit",
              "reviews": "Reviews", "wiki": "Wikipedia"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overall", default="results_v2/logo_detection.parquet")
    ap.add_argument("--bydomain", default="results_v2/logo_detection_bydomain.parquet")
    ap.add_argument("--metric", default="auc", choices=["auc", "tpr"])
    ap.add_argument("--out-tex", default="tables/tab_logo_by_domain.tex")
    args = ap.parse_args()

    ov = pd.read_parquet(args.overall)
    bd = pd.read_parquet(args.bydomain)
    dets = [d for d in DET_ORDER if f"{d}_{args.metric}" in ov.columns]
    n_seeds = ov["seed"].nunique() if "seed" in ov.columns else "?"

    # Overall row: mean over (seed, generator)
    rows = [("Overall", {d: ov[f"{d}_{args.metric}"].mean() for d in dets})]
    # per-domain rows: mean over (seed, generator) within domain
    for dom in sorted(bd["domain"].unique()):
        sub = bd[bd["domain"] == dom]
        vals = {}
        for d in dets:
            v = sub[sub["detector"] == d][args.metric]
            vals[d] = v.mean() if len(v) else np.nan
        rows.append((DOM_PRETTY.get(dom, dom.capitalize()), vals))

    metric_name = "AUC" if args.metric == "auc" else "TPR@$1\\%$FPR"
    col = "l" + "c" * len(dets)
    L = ["% Per-domain leave-one-generator-out detection (build_logo_domain_table.py).",
         "\\begin{table}[t]\\centering\\small",
         "\\setlength{\\tabcolsep}{4pt}",
         "\\caption{\\textbf{Unseen-generator detection by domain "
         "(leave-one-generator-out).} Each cell is " + metric_name + ", averaged "
         "over the $18$ held-out generators $\\times$ " + str(n_seeds) +
         " domain-stratified seeds; every generator is held out of every detector "
         "in turn, and all detectors are scored on identical held-out documents. "
         "\\textbf{Bold} = best per row.}",
         "\\label{tab:logo-by-domain}",
         "\\begin{tabular}{" + col + "}", "\\toprule",
         "Domain & " + " & ".join(DET_SHORT[d] for d in dets) + " \\\\", "\\midrule"]
    for name, vals in rows:
        best = max((d for d in dets if not np.isnan(vals[d])), key=lambda d: vals[d])
        cells = []
        for d in dets:
            x = vals[d]
            c = "n/a" if np.isnan(x) else f"{x:.3f}"
            if d == best and not np.isnan(x): c = f"\\textbf{{{c}}}"
            cells.append(c)
        sep = "\\midrule\n" if name == "Overall" else ""
        L.append(f"{name} & " + " & ".join(cells) + " \\\\")
        if name == "Overall": L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    Path(args.out_tex).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_tex).write_text("\n".join(L) + "\n")
    print(f"[out] {args.out_tex}")
    for name, vals in rows:
        print(f"  {name:12s} " + "  ".join(f"{d}={vals[d]:.3f}" for d in dets if not np.isnan(vals[d])))


if __name__ == "__main__":
    main()
