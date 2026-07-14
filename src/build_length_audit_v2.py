#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_length_audit_v2.py   (rich UI, CPU-only, ~10 s)
=====================================================
Fixes a REAL BUG found while answering reviewer 4.

THE BUG
-------
build_length_effect.py macro-averaged per-corpus AUCs within each length bin and
claimed this "controls for corpus composition". It does not, because a corpus with
too few AI (or human) documents in a bin is silently dropped from that bin's average.
In this benchmark suite document length is nearly COLLINEAR with corpus:

    HC3          : no documents  >= 400 words        (0 AI docs)
    NYT          : almost no documents < 200 words
    Pile         : 16 AI docs >= 200 words, 0 >= 400
    TuringBench  : 0 documents >= 400 words

so the bins average over DIFFERENT corpora:
    25-50 w   -> MAGE alone            (GLASS 0.539  <- the "collapse on short text")
    >=400 w   -> MAGE + NYT            (GLASS 0.800)
The apparent "AUC rises with length" trend is therefore partly the easy corpora
(HC3, NYT ~0.99) ENTERING the average as bins get longer. Same for the cumulative
threshold sweep. Any number that averages across a changing corpus set is not
comparable to the number in the row above it.

THE FIX (two analyses, both composition-safe)
---------------------------------------------
A) THRESHOLD SWEEP (reviewer 4): only compare thresholds over the corpora that
   remain valid at EVERY threshold compared. Corpora that drop out are named, and
   thresholds where the fixed set collapses are refused rather than reported.

B) LENGTH EFFECT: abandon the cross-corpus macro-average entirely. Report the
   WITHIN-CORPUS length curve for each detector: does AUC rise or fall with length
   INSIDE a fixed corpus? That is the causal question, and it is immune to
   composition. Aggregate it as a within-corpus slope (Kendall tau over the bins a
   corpus actually spans), so no average is ever taken across different corpus sets.

Run:  python3 build_length_audit_v2.py
"""
from __future__ import annotations
import argparse, itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from rich.console import Console
from rich.table import Table

console = Console()
ORDER = ["GLASS", "RADAR", "Binoculars", "Fast-DetectGPT", "Ghostbuster",
         "OpenAI-large", "RoBERTa-OpenAI", "log-rank"]
BIN_LABEL = {0: "<25", 1: "25-50", 2: "50-100", 3: "100-200", 4: "200-400", 5: ">=400"}
MIN_N = 20                       # per class, per cell


def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y)
    ok = np.isfinite(s); s, y = s[ok], y[ok]
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos < MIN_N or nneg < MIN_N:
        return np.nan
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); r = ((csum - cnt + csum + 1) / 2.0)[inv]
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def valid(sub):
    """Is this (corpus, slice) cell estimable at all?"""
    return int((sub.label == 1).sum()) >= MIN_N and int((sub.label == 0).sum()) >= MIN_N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perdoc", default="results_v2/length_effect_perdoc.parquet")
    ap.add_argument("--thresholds", default="0,25,50,100,200,400")
    ap.add_argument("--outdir", default="tables")
    args = ap.parse_args()

    d = pd.read_parquet(args.perdoc)
    dets = [c for c in ORDER if c in d.columns]
    corpora = sorted(d.corpus.unique())
    TT = [int(t) for t in args.thresholds.split(",")]
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    console.rule("[bold]Composition audit")

    # ---------------------------------------------------------------- coverage --
    cov = Table(title="Corpus x length-slice coverage (AI docs / human docs). "
                      "'--' = not estimable (<20 per class) => SILENTLY DROPPED before")
    cov.add_column("corpus")
    for T in TT:
        cov.add_column(f">={T}", justify="right")
    for b in sorted(d["bin"].unique()):
        cov.add_column(BIN_LABEL[b], justify="right")
    ok_thr = {T: [] for T in TT}
    ok_bin = {b: [] for b in sorted(d["bin"].unique())}
    for c in corpora:
        row = [c]
        for T in TT:
            s = d[(d.corpus == c) & (d.wlen >= T)]
            v = valid(s)
            ok_thr[T].append(c) if v else None
            row.append(f"{int((s.label==1).sum())}/{int((s.label==0).sum())}" if v
                       else f"[red]--[/]")
        for b in sorted(d["bin"].unique()):
            s = d[(d.corpus == c) & (d["bin"] == b)]
            v = valid(s)
            ok_bin[b].append(c) if v else None
            row.append(f"{int((s.label==1).sum())}/{int((s.label==0).sum())}" if v
                       else f"[red]--[/]")
        cov.add_row(*row)
    console.print(cov)

    # ==========================================================================
    # A) THRESHOLD SWEEP ON A FIXED CORPUS SET
    # ==========================================================================
    console.rule("[bold]A. Threshold sweep, fixed corpus set (reviewer 4)")
    # largest run of thresholds sharing a non-trivial corpus set
    comparable = [T for T in TT if len(ok_thr[T]) >= 4]
    fixed = sorted(set.intersection(*[set(ok_thr[T]) for T in comparable]))
    dropped = sorted(set(corpora) - set(fixed))
    refused = [T for T in TT if T not in comparable]
    console.print(f"[green]comparable thresholds[/]: {comparable}")
    console.print(f"[green]fixed corpus set[/]     : {fixed}")
    console.print(f"[yellow]corpora dropped[/]      : {dropped or 'none'} "
                  f"(not estimable at every comparable threshold)")
    console.print(f"[red]thresholds refused[/]   : {refused} "
                  f"(fewer than 4 corpora survive -> not comparable, will NOT be "
                  f"reported as a mean)")

    rows = []
    for T in comparable:
        sub = d[d.wlen >= T]
        for det in dets:
            per = {c: auc(sub[sub.corpus == c][det], sub[sub.corpus == c].label) for c in fixed}
            v = [x for x in per.values() if np.isfinite(x)]
            rows.append({"threshold": T, "detector": det, **{f"auc_{c}": per[c] for c in fixed},
                         "mean": float(np.mean(v)), "worst": float(np.min(v)),
                         "n_docs": int(len(sub))})
    R = pd.DataFrame(rows)
    R.to_csv("results_v2/threshold_sweep_fixed.csv", index=False)

    sweep = Table(title=f"Cross-corpus ranking vs length threshold "
                        f"-- FIXED corpus set {fixed}")
    for c in ["min words", "detector", "mean AUC", "rank", "worst AUC", "rank"]:
        sweep.add_column(c, justify="right")
    glass_track = {}
    for T in comparable:
        s = R[R.threshold == T].copy()
        s["rm"] = s["mean"].rank(ascending=False).astype(int)
        s["rw"] = s["worst"].rank(ascending=False).astype(int)
        g = s[s.detector == "GLASS"].iloc[0]
        glass_track[T] = (int(g.rm), int(g.rw), float(g["mean"]), float(g["worst"]))
        for _, r in s.sort_values("mean", ascending=False).iterrows():
            b = r.detector == "GLASS"
            f = (lambda x: f"[bold]{x}[/]") if b else (lambda x: str(x))
            sweep.add_row(f(f">={T}"), f(r.detector), f(f"{r['mean']:.3f}"), f(int(r.rm)),
                          f(f"{r['worst']:.3f}"), f(int(r.rw)))
        sweep.add_section()
    console.print(sweep)

    # who is top-2 on worst-case at how many thresholds?
    stab = Table(title="Stability: how often is each detector top-2 by WORST-CASE AUC?")
    for c in ["detector", "mean-rank range", "worst-rank range", "top-2 worst (of "
              f"{len(comparable)})"]:
        stab.add_column(c, justify="right")
    srows = []
    for det in dets:
        rm, rw = [], []
        for T in comparable:
            s = R[R.threshold == T].copy()
            s["rm"] = s["mean"].rank(ascending=False).astype(int)
            s["rw"] = s["worst"].rank(ascending=False).astype(int)
            rm.append(int(s[s.detector == det].rm.iloc[0]))
            rw.append(int(s[s.detector == det].rw.iloc[0]))
        srows.append((det, min(rm), max(rm), min(rw), max(rw), sum(1 for x in rw if x <= 2)))
    for det, a, b, c_, dd, n2 in sorted(srows, key=lambda r: -r[5]):
        h = (lambda x: f"[bold]{x}[/]") if det == "GLASS" else (lambda x: str(x))
        stab.add_row(h(det), h(f"{a}-{b}"), h(f"{c_}-{dd}"), h(n2))
    console.print(stab)
    for T, (rm, rw, mn, wo) in glass_track.items():
        console.print(f"RESULT[sweep T={T}] GLASS mean={mn:.4f} rank={rm} "
                      f"worst={wo:.4f} rank={rw}")

    # ==========================================================================
    # B) WITHIN-CORPUS LENGTH EFFECT (replaces the flawed macro-average)
    # ==========================================================================
    console.rule("[bold]B. Within-corpus length effect (replaces Table 17 / Fig 6)")
    bins = sorted(d["bin"].unique())
    long_rows = []
    for c in corpora:
        for det in dets:
            for b in bins:
                s = d[(d.corpus == c) & (d["bin"] == b)]
                a = auc(s[det], s.label)
                if np.isfinite(a):
                    long_rows.append({"corpus": c, "detector": det, "bin": b,
                                      "bin_label": BIN_LABEL[b], "auc": a, "n": len(s)})
    W = pd.DataFrame(long_rows)
    W.to_csv("results_v2/length_within_corpus.csv", index=False)

    for c in corpora:
        sub = W[W.corpus == c]
        if sub.empty:
            console.print(f"[red]{c}: no estimable length bin at all[/]")
            continue
        bs = sorted(sub.bin.unique())
        t = Table(title=f"{c}: AUC by length bin (within-corpus -- no composition confound)")
        t.add_column("detector")
        for b in bs:
            t.add_column(BIN_LABEL[b], justify="right")
        t.add_column("trend (tau)", justify="right")
        for det in dets:
            r = sub[sub.detector == det].sort_values("bin")
            if len(r) < 1:
                continue
            cells = []
            for b in bs:
                x = r[r.bin == b]
                cells.append(f"{x.auc.iloc[0]:.3f}" if len(x) else "--")
            if len(r) >= 3:
                tau = kendalltau(r.bin.values, r.auc.values).statistic
                col = "green" if tau > 0.3 else ("red" if tau < -0.3 else "yellow")
                trend = f"[{col}]{tau:+.2f}[/]"
            else:
                trend = "[dim]n/a[/]"
            h = "[bold]" if det == "GLASS" else ""
            e = "[/]" if det == "GLASS" else ""
            t.add_row(f"{h}{det}{e}", *[f"{h}{x}{e}" for x in cells], trend)
        console.print(t)

    # the composition-safe headline: sign of the within-corpus trend
    console.rule("[bold]Composition-safe length claim")
    hd = Table(title="Within-corpus trend of AUC with document length (Kendall tau, "
                     "computed only where a corpus spans >=3 estimable bins)")
    hd.add_column("detector"); hd.add_column("corpora with >=3 bins", justify="right")
    hd.add_column("tau per corpus"); hd.add_column("sign", justify="right")
    for det in dets:
        taus = {}
        for c in corpora:
            r = W[(W.corpus == c) & (W.detector == det)].sort_values("bin")
            if len(r) >= 3:
                taus[c] = kendalltau(r.bin.values, r.auc.values).statistic
        if not taus:
            continue
        pos = sum(1 for v in taus.values() if v > 0)
        neg = sum(1 for v in taus.values() if v < 0)
        sign = ("[green]improves[/]" if neg == 0 and pos > 0 else
                "[red]degrades[/]" if pos == 0 and neg > 0 else "[yellow]mixed[/]")
        h = "[bold]" if det == "GLASS" else ""
        e = "[/]" if det == "GLASS" else ""
        hd.add_row(f"{h}{det}{e}", f"{h}{len(taus)}{e}",
                   f"{h}" + "  ".join(f"{c}:{v:+.2f}" for c, v in taus.items()) + f"{e}", sign)
        console.print(f"RESULT[len {det}] " +
                      " ".join(f"{c}={v:+.3f}" for c, v in taus.items()))
    console.print(hd)
    console.print("\n[green]done[/] -> results_v2/threshold_sweep_fixed.csv, "
                  "results_v2/length_within_corpus.csv")


if __name__ == "__main__":
    main()
