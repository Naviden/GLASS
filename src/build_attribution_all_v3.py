#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_attribution_all_v3.py   (rich UI)
=======================================
ONE table, ONE shared split: GAS vs fine-tuned RoBERTa vs Ghostbuster for N-way
generator attribution on the >=200-word frontier panel (frontier_en_long).

Every method is fit and scored on the IDENTICAL stratified train/test split, so
GAS reports a single number that reconciles across the whole comparison (fixes
the earlier issue where each script rolled its own split and GAS came out
differently). Reuses the exact implementations already in the repo:
  GAS          -> build_ghostbuster_attr_v3.GAS   (closed-form, no training)
  RoBERTa      -> build_roberta_attr_v3.roberta_attr (fine-tuned roberta-base)
  Ghostbuster  -> build_ghostbuster_attr_v3.lm_feats (gpt2+distilgpt2) + LR

Run (turing, ~/glass/paper, venv, GPU):
    CUDA_VISIBLE_DEVICES=1 PYTHONUTF8=1 TRANSFORMERS_VERBOSITY=error \\
    TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore \\
    python3 build_attribution_all_v3.py --corpus frontier_en_long
~35 min (RoBERTa fine-tunes per N; Ghostbuster LM features once).
"""
from __future__ import annotations
import os, warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")
import argparse, json, sys, collections
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.table import Table

sys.path.insert(0, ".")
from build_ghostbuster_attr_v3 import lm_feats, GAS
from build_roberta_attr_v3 import roberta_attr

console = Console()


def main():
    here = Path(__file__).resolve().parent
    root = here if (here / "results").exists() else here.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(root / ".cache" / "pipeline"))
    ap.add_argument("--corpus", default="frontier_en_long")
    ap.add_argument("--alias", default="deberta-v3-base")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nways", default="4,18")
    ap.add_argument("--min-per-gen", type=int, default=50)
    ap.add_argument("--cap-per-gen", type=int, default=300)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-tex", default="tables_v2/tab_attribution_all_v3.tex")
    args = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    cache = Path(args.cache)
    info = json.loads((cache / f"corpus_{args.corpus}.json").read_text())
    y = np.asarray(info["labels"], int); gens = np.asarray(info["gens"]); texts = info["texts"]
    Xemb = np.load(cache / f"emb_{args.corpus}_{args.alias}.npy")
    ai = np.where(y == 1)[0]
    counts = collections.Counter(gens[ai].tolist())
    ranked = sorted([g for g, c in counts.items() if c >= args.min_per_gen], key=lambda g: -counts[g])
    console.print(f"[attr] {len(ranked)} generators >= {args.min_per_gen} docs (corpus={args.corpus})")

    # select capped docs/generator (superset for widest N); extract LM features ONCE
    rng = np.random.default_rng(args.seed)
    sel = []
    for g in ranked:
        gi = ai[gens[ai] == g]
        if len(gi) > args.cap_per_gen:
            gi = rng.choice(gi, args.cap_per_gen, replace=False)
        sel.extend(gi.tolist())
    sel = np.array(sorted(sel))
    texts_s = [texts[int(i)] for i in sel]; gens_s = gens[sel]; Xs = Xemb[sel]
    console.print(f"[ghostbuster] LM features on {len(sel)} docs (gpt2 + distilgpt2) ...")
    F = np.nan_to_num(np.hstack([lm_feats(texts_s, "gpt2", dev),
                                 lm_feats(texts_s, "distilgpt2", dev)]))

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    nways = [len(ranked) if x == "all" else int(x) for x in args.nways.split(",")]
    rows = []
    for N in nways:
        keep = ranked[:N]
        pos = np.where(np.isin(gens_s, keep))[0]           # positions in sel-space
        gp = gens_s[pos]
        # ONE stratified-by-generator split, shared by all three methods
        srng = np.random.default_rng(args.seed + N)
        tr, te = [], []
        for g in keep:
            gi = srng.permutation(pos[gp == g])
            ncut = max(1, int(round(len(gi) * args.test_frac)))
            te.extend(gi[:ncut].tolist()); tr.extend(gi[ncut:].tolist())
        tr = np.array(tr); te = np.array(te)

        # GAS (closed-form, no training)
        gmod = GAS(args.k).fit(Xs[tr], gens_s[tr])
        gas_acc = float(np.mean(gmod.predict(Xs[te]) == gens_s[te]))
        # Ghostbuster (multi-LM log-prob features + multinomial LR)
        sc = StandardScaler().fit(F[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(F[tr]), gens_s[tr])
        gb_acc = float(np.mean(clf.predict(sc.transform(F[te])) == gens_s[te]))
        # RoBERTa (fine-tuned) on the SAME split
        rob_acc = roberta_attr(texts_s, gens_s, tr.tolist(), te.tolist(), keep, epochs=args.epochs, dev=dev)

        rows.append((N, gas_acc, rob_acc, gb_acc, 1.0 / N))
        console.print(f"  N={N}: GAS={gas_acc:.3f}  RoBERTa={rob_acc:.3f}  "
                      f"Ghostbuster={gb_acc:.3f}  chance={1.0/N:.3f}")

    t = Table(title=f"N-way generator attribution -- shared split ({args.corpus})")
    for c in ["N", "GAS (no train)", "RoBERTa (fine-tuned)", "Ghostbuster", "chance"]:
        t.add_column(c, justify="right")
    for N, ga, ra, gb, ch in rows:
        t.add_row(str(N), f"{ga:.3f}", f"{ra:.3f}", f"{gb:.3f}", f"{ch:.3f}")
    console.print(t)

    L = ["% Unified N-way generator attribution, >=200-word frontier panel, ONE shared split.",
         "\\begin{table}[t]\\centering\\small",
         "\\caption{Top-1 generator attribution on the length-restricted English frontier panel "
         "($\\geq 200$ words); all methods fit and scored on the \\emph{same} held-out split: "
         "closed-form GAS (no training), fine-tuned RoBERTa-base, and a Ghostbuster-style "
         "multi-LM log-probability classifier. \\textbf{Bold} = best per row.}",
         "\\label{tab:attribution-all}",
         "\\begin{tabular}{lcccc}", "\\toprule",
         "Classes $N$ & GAS (no train) & RoBERTa & Ghostbuster & Chance \\\\", "\\midrule"]
    for N, ga, ra, gb, ch in rows:
        vals = {"GAS": ga, "RoB": ra, "GB": gb}; best = max(vals, key=vals.get)
        ga_s = f"\\textbf{{{ga:.3f}}}" if best == "GAS" else f"{ga:.3f}"
        ra_s = f"\\textbf{{{ra:.3f}}}" if best == "RoB" else f"{ra:.3f}"
        gb_s = f"\\textbf{{{gb:.3f}}}" if best == "GB" else f"{gb:.3f}"
        L.append(f"{N} & {ga_s} & {ra_s} & {gb_s} & {ch:.3f} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    Path(args.out_tex).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_tex).write_text("\n".join(L) + "\n")
    console.print(f"[green]done[/] -> {args.out_tex}")


if __name__ == "__main__":
    main()
