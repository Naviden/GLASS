#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare PROFILER and GAS generator attribution.

Requires profiler_on_our_data.py and an external PROFILER checkout."""
from __future__ import annotations
import os, warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")
import argparse, json, sys, collections
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from rich.console import Console

sys.path.insert(0, ".")
from profiler_on_our_data import extract_features_for_llm, PROFILER_BASE_MODELS
from build_attribution_v3 import GAS

console = Console()


def main():
    here = Path(__file__).resolve().parent
    root = here if (here / "results").exists() else here.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiler-root", required=True, type=Path)
    ap.add_argument("--cache", default=str(root / ".cache" / "pipeline"))
    ap.add_argument("--corpus", default="frontier_en_long")
    ap.add_argument("--alias", default="deberta-v3-base")
    ap.add_argument("--nways", default="4,18")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--min-per-gen", type=int, default=50)
    ap.add_argument("--cap-per-gen", type=int, default=300,
                    help="cap docs/generator fed to the 6-LLM feature extractor (cost control)")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--feature-cache", default="results_v2/profiler_attr_features")
    ap.add_argument("--sample-clip", type=int, default=4000)
    ap.add_argument("--context-window", type=int, default=6)
    ap.add_argument("--rf-n-estimators", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-tex", default="tables_v2/tab_gas_vs_profiler_v3.tex")
    args = ap.parse_args()

    cache = Path(args.cache)
    info = json.loads((cache / f"corpus_{args.corpus}.json").read_text())
    y = np.asarray(info["labels"], int); gens = np.asarray(info["gens"]); texts = info["texts"]
    X = np.load(cache / f"emb_{args.corpus}_{args.alias}.npy")

    ai = np.where(y == 1)[0]
    counts = collections.Counter(gens[ai].tolist())
    ranked = sorted([g for g, c in counts.items() if c >= args.min_per_gen], key=lambda g: -counts[g])
    console.print(f"[attr] {len(ranked)} generators >= {args.min_per_gen} docs (corpus={args.corpus})")

    # cap docs/generator for the widest N (superset); extract PROFILER features ONCE
    rng = np.random.default_rng(args.seed)
    sel = []
    for g in ranked:
        gi = ai[gens[ai] == g]
        if len(gi) > args.cap_per_gen:
            gi = rng.choice(gi, args.cap_per_gen, replace=False)
        sel.extend(gi.tolist())
    sel = np.array(sorted(sel))
    texts_sel = [texts[int(i)] for i in sel]
    gens_sel = gens[sel]; Xemb = X[sel]
    console.print(f"[features] PROFILER 6-LLM features for {len(sel)} docs (resumable) ...")

    feat_dir = Path(args.feature_cache); feat_dir.mkdir(parents=True, exist_ok=True)
    pargs = SimpleNamespace(sample_clip=args.sample_clip, context_window=args.context_window)
    per = {}
    for bm in PROFILER_BASE_MODELS:
        console.print(f"  ===== {bm} =====")
        f = extract_features_for_llm(args.profiler_root, bm, texts_sel, pargs,
                                     feat_dir / f"{bm}_{args.corpus}.pkl")
        if f is None:
            console.print(f"  [DROP] {bm}"); continue
        per[bm] = f
    if not per:
        raise SystemExit("no LLM produced features (check HF auth for gated repos)")
    avail = list(per)
    console.print(f"[ensemble] {len(avail)}/{len(PROFILER_BASE_MODELS)}: {avail}")
    san = lambda A: np.clip(np.nan_to_num(np.asarray(A, float), nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)
    F = san(np.hstack([per[m] for m in avail]))

    from sklearn.ensemble import RandomForestClassifier
    rows = []
    nways = [len(ranked) if x == "all" else int(x) for x in args.nways.split(",")]
    for N in nways:
        keep = ranked[:N]
        idx = np.where(np.isin(gens_sel, keep))[0]
        gsub = gens_sel[idx]
        tr, te = [], []
        for g in keep:                                  # stratified split per generator
            gi = rng.permutation(idx[gsub == g])
            ncut = max(1, int(round(len(gi) * args.test_frac)))
            te.extend(gi[:ncut].tolist()); tr.extend(gi[ncut:].tolist())
        tr = np.array(tr); te = np.array(te)
        # GAS (training-free)
        g = GAS(args.k).fit(Xemb[tr], gens_sel[tr])
        gas_acc = float(np.mean(g.predict(Xemb[te]) == gens_sel[te]))
        # PROFILER (its 6-LLM features + N-way RF)
        clf = RandomForestClassifier(n_estimators=args.rf_n_estimators, random_state=args.seed,
                                     n_jobs=-1, class_weight="balanced").fit(F[tr], gens_sel[tr])
        prof_acc = float(np.mean(clf.predict(F[te]) == gens_sel[te]))
        rows.append((N, gas_acc, prof_acc, 1.0 / N))
        console.print(f"  N={N}: GAS={gas_acc:.3f}  PROFILER={prof_acc:.3f}  chance={1.0/N:.3f}")

    L = ["% GAS (training-free) vs PROFILER (6-LLM features + N-way RF) attribution, >=200-word frontier.",
         "\\begin{table}[t]\\centering\\small",
         "\\caption{Top-1 generator attribution on the length-restricted English frontier panel "
         "($\\geq 200$ words): closed-form GAS (no training) vs PROFILER (its 6-LLM context-loss "
         "features with an $N$-way random forest). Both fit on the same generators and evaluated "
         "on the same held-out documents. \\textbf{Bold} = best.}",
         "\\label{tab:gas-vs-profiler}",
         "\\begin{tabular}{lccc}", "\\toprule",
         "Classes $N$ & GAS (no train) & PROFILER & Chance \\\\", "\\midrule"]
    for N, ga, pa, ch in rows:
        g_s = f"\\textbf{{{ga:.3f}}}" if ga >= pa else f"{ga:.3f}"
        p_s = f"\\textbf{{{pa:.3f}}}" if pa > ga else f"{pa:.3f}"
        L.append(f"{N} & {g_s} & {p_s} & {ch:.3f} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    Path(args.out_tex).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_tex).write_text("\n".join(L) + "\n")
    console.print(f"[green]done[/] -> {args.out_tex}")


if __name__ == "__main__":
    main()
