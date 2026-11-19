#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare detector rankings at alternative minimum document lengths."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()

ORDER = ["GLASS", "RADAR", "Binoculars", "Fast-DetectGPT", "Ghostbuster",
         "OpenAI-large", "RoBERTa-OpenAI", "log-rank"]


def auc(scores, y):
    s = np.asarray(scores, float); y = np.asarray(y)
    ok = np.isfinite(s); s, y = s[ok], y[ok]
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos < 20 or nneg < 20:
        return np.nan
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); avg = (csum - cnt + csum + 1) / 2.0
    r = avg[inv]
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perdoc", default="results_v2/length_effect_perdoc.parquet")
    ap.add_argument("--thresholds", default="0,25,50,100,200,400")
    ap.add_argument("--out-tex", default="tables/tab_threshold_sweep.tex")
    ap.add_argument("--out-csv", default="results_v2/threshold_sweep.csv")
    args = ap.parse_args()

    d = pd.read_parquet(args.perdoc)
    dets = [c for c in ORDER if c in d.columns]
    corpora = sorted(d["corpus"].unique())
    thresholds = [int(t) for t in args.thresholds.split(",")]
    console.print(f"[bold]{len(d)} documents[/]  |  detectors: {len(dets)}  |  corpora: {corpora}")

    rows = []
    for T in thresholds:
        sub = d[d.wlen >= T]
        for det in dets:
            per = {c: auc(sub[sub.corpus == c][det], sub[sub.corpus == c]["label"]) for c in corpora}
            vals = [v for v in per.values() if np.isfinite(v)]
            if not vals:
                continue
            rows.append({"threshold": T, "detector": det,
                         **{f"auc_{c}": per[c] for c in corpora},
                         "mean": float(np.mean(vals)), "worst": float(np.min(vals)),
                         "n_docs": len(sub)})
    R = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(args.out_csv, index=False)

    # ---- the answer: does GLASS's rank survive the threshold? ----
    summary = Table(title="Cross-corpus ranking vs the word-count threshold "
                          "(sensitivity to minimum document length)")
    for c in ["threshold", "n docs", "detector", "mean AUC", "rank(mean)",
              "worst AUC", "rank(worst)"]:
        summary.add_column(c, justify="right")
    verdict = {}
    for T in thresholds:
        sub = R[R.threshold == T].copy()
        if sub.empty:
            continue
        sub["rank_mean"] = sub["mean"].rank(ascending=False).astype(int)
        sub["rank_worst"] = sub["worst"].rank(ascending=False).astype(int)
        sub = sub.sort_values("mean", ascending=False)
        g = sub[sub.detector == "GLASS"].iloc[0]
        verdict[T] = (int(g.rank_mean), int(g.rank_worst), float(g["mean"]), float(g["worst"]))
        for _, r in sub.iterrows():
            hi = "[bold]" if r.detector == "GLASS" else ""
            end = "[/]" if r.detector == "GLASS" else ""
            summary.add_row(f"{hi}>={T}{end}", f"{hi}{int(r.n_docs)}{end}",
                            f"{hi}{r.detector}{end}", f"{hi}{r['mean']:.3f}{end}",
                            f"{hi}{int(r.rank_mean)}{end}", f"{hi}{r['worst']:.3f}{end}",
                            f"{hi}{int(r.rank_worst)}{end}")
        summary.add_section()
    console.print(summary)

    console.print("\n[bold]GLASS rank as the threshold moves[/]")
    v = Table()
    for c in ["threshold", "GLASS mean", "rank by mean", "GLASS worst", "rank by worst"]:
        v.add_column(c, justify="right")
    for T, (rm, rw, mn, wo) in verdict.items():
        col = "green" if (rm == 1 and rw == 1) else ("yellow" if rm <= 2 else "red")
        v.add_row(f">={T}", f"{mn:.3f}", f"[{col}]{rm}[/]", f"{wo:.3f}", f"[{col}]{rw}[/]")
    console.print(v)

    robust = all(rm == 1 for rm, rw, _, _ in verdict.values() if _ is not None)
    r100 = verdict.get(100, (None,))[0]
    console.print()
    for T, (rm, rw, mn, wo) in verdict.items():
        console.print(f"RESULT[T={T}] GLASS rank_mean={rm} rank_worst={rw} "
                      f"mean={mn:.3f} worst={wo:.3f}")
    console.print("\n[bold]VERDICT[/]: "
                  + ("threshold-ROBUST (GLASS first by mean at every threshold)"
                     if all(x[0] == 1 for x in verdict.values())
                     else "threshold-DEPENDENT -- the claim must be stated as conditional "
                          "on the document-length regime. Report this honestly."))

    L = ["% Cross-corpus ranking as a function of the minimum-word-count threshold.",
         "% Detector rankings across minimum document lengths.",
         "% Reproduced by: build_threshold_sweep.py (CPU, uses length_effect_perdoc.parquet)",
         "\\begin{table}[!htbp]\\centering\\small",
         "\\caption{\\textbf{Sensitivity of the cross-corpus ranking to the document-length "
         "threshold.} Mean and worst-case AUC over the five out-of-distribution corpora, and "
         "\\glass's rank on each, as the minimum document length is varied. All detectors are "
         "re-scored on the identical filtered sets. The $\\geq 100$-word row is the one reported "
         "in Table~\\ref{tab:thirdparty}.}",
         "\\label{tab:threshold-sweep}",
         "\\begin{tabular}{lccccc}", "\\toprule",
         "Min.\\ words & Documents & \\glass\\ mean & Rank & \\glass\\ worst & Rank \\\\",
         "\\midrule"]
    for T, (rm, rw, mn, wo) in verdict.items():
        n = int(R[R.threshold == T]["n_docs"].iloc[0])
        star = "\\textbf{" if T == 100 else ""
        end = "}" if T == 100 else ""
        L.append(f"{star}$\\geq${T}{end} & {n} & {star}{mn:.3f}{end} & {star}{rm}{end} & "
                 f"{star}{wo:.3f}{end} & {star}{rw}{end} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    Path(args.out_tex).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_tex).write_text("\n".join(L) + "\n")
    console.print(f"\n[green]done[/] -> {args.out_tex}, {args.out_csv}")


if __name__ == "__main__":
    main()
