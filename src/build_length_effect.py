#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure detector AUC across document-length bins.

Bin composition can vary by corpus; use build_length_audit_v2.py for within-corpus analysis."""
from __future__ import annotations
import os, warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

import argparse, re
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from build_thirdparty_attack_v3 import (
    load_bundle, score_all_detectors, load_raid_library,
    load_hc3, load_turingbench, load_mage, load_nyt,
)
from build_thirdparty_competitors_v3 import load_hf_generic
from text_quality import _STOP, _WORD, _BOILER

console = Console()

BIN_EDGES = [0, 25, 50, 100, 200, 400, 10**9]
BIN_LABELS = ["<25", "25-50", "50-100", "100-200", "200-400", ">=400"]


def keep_len(text):
    """Gentle quality gate for the length study: drop boilerplate everywhere;
    apply corruption checks (stopword/uniqueness) only to docs >= 30 words so
    genuine short sentences are not thrown away."""
    low = (text or "").strip().lower()
    if not low:
        return False
    if any(b in low for b in _BOILER):
        return False
    words = _WORD.findall(low)
    if len(words) < 1:
        return False
    if len(words) >= 30:
        if sum(1 for w in words if w in _STOP) / len(words) < 0.15:
            return False
        if len(set(words)) / len(words) < 0.25:
            return False
    return True


def auc(scores, y):
    s = np.asarray(scores, float); y = np.asarray(y)
    ok = np.isfinite(s); s = s[ok]; y = y[ok]
    pos = (y == 1).sum(); neg = (y == 0).sum()
    if pos < 5 or neg < 5:
        return float("nan")
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); avg = (csum - cnt + csum + 1) / 2.0
    ranks = avg[inv]
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def load_models(args, dev):
    import torch
    models = {}
    SH, SA, WH = load_bundle(args.bundle, args.K)
    models["glass"] = (SH, SA, WH, args.encoder)
    console.print(f"[bundle] S_H + {len(SA)} subspaces (K={args.K})")
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                              AutoModelForSequenceClassification)
    if args.with_falcon:
        ftok = AutoTokenizer.from_pretrained(args.falcon)
        m1 = AutoModelForCausalLM.from_pretrained(args.falcon, dtype=torch.float16).to(dev).eval()
        m2 = (AutoModelForCausalLM.from_pretrained(args.performer, dtype=torch.float16).to(dev).eval()
              if args.with_binoculars else None)
        models["falcon"] = (ftok, m1, m2)
    if args.with_roberta:
        rtok = AutoTokenizer.from_pretrained(args.roberta)
        rmdl = AutoModelForSequenceClassification.from_pretrained(args.roberta).to(dev).eval()
        models["roberta"] = (rtok, rmdl)
    if args.with_ghostbuster:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from build_thirdparty_attack_v3 import gb_feats_one
        gmods = []
        for mid in ("gpt2", "distilgpt2"):
            tk = AutoTokenizer.from_pretrained(mid)
            if tk.pad_token is None: tk.pad_token = tk.eos_token
            gmods.append((AutoModelForCausalLM.from_pretrained(mid).to(dev).eval(), tk))
        rng = np.random.default_rng(0)
        tr_ai, tr_hu = load_raid_library(args.panels, args.humans, args.gb_train_cap, rng)
        gbmat = lambda T: np.hstack([np.array([gb_feats_one(t, mm, tk, dev) for t in T]) for mm, tk in gmods])
        Xtr = np.vstack([gbmat(tr_ai), gbmat(tr_hu)])
        ytr = np.array([1] * len(tr_ai) + [0] * len(tr_hu))
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
        models["ghostbuster"] = (gmods, sc, clf)
    if args.with_radar:
        dtok = AutoTokenizer.from_pretrained(args.radar_model)
        dmdl = AutoModelForSequenceClassification.from_pretrained(args.radar_model).to(dev).eval()
        models["radar"] = (dtok, dmdl)
    if args.hf_detector:
        hfd = []
        for spec in args.hf_detector:
            dname, mid = spec.split(":", 1)
            tok = AutoTokenizer.from_pretrained(mid)
            mdl = AutoModelForSequenceClassification.from_pretrained(mid).to(dev).eval()
            hfd.append((dname, tok, mdl))
        models["hf_detector"] = hfd
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=".cache/glass_production/v3_english_clean")
    ap.add_argument("--encoder", default="microsoft/deberta-v3-base")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--n", type=int, default=1500, help="max docs per class per corpus (before length binning)")
    ap.add_argument("--corpora", default="HC3,TuringBench,MAGE,NYT")
    ap.add_argument("--hf-corpus", action="append",
                    default=["Pile:artem9k/ai-text-detection-pile:text:source:ai:human"])
    ap.add_argument("--with-falcon", action="store_true")
    ap.add_argument("--with-binoculars", action="store_true")
    ap.add_argument("--with-roberta", action="store_true")
    ap.add_argument("--with-ghostbuster", action="store_true")
    ap.add_argument("--with-radar", action="store_true")
    ap.add_argument("--radar-model", default="TrustSafeAI/RADAR-Vicuna-7B")
    ap.add_argument("--hf-detector", action="append", default=[])
    ap.add_argument("--falcon", default="tiiuae/falcon-7b")
    ap.add_argument("--performer", default="tiiuae/falcon-7b-instruct")
    ap.add_argument("--roberta", default="openai-community/roberta-base-openai-detector")
    ap.add_argument("--panels", default="results_v2/openrouter_en_s*/per_doc.jsonl")
    ap.add_argument("--humans", default="results_v2/raid_en_humans.parquet")
    ap.add_argument("--gb-train-cap", type=int, default=4000)
    ap.add_argument("--min-per-cell", type=int, default=40, help="min per class per bin, else AUC=NaN")
    ap.add_argument("--out-fig", default="figs/fig_length_effect")
    ap.add_argument("--out-tex", default="tables/tab_length_effect.tex")
    ap.add_argument("--out-csv", default="results_v2/length_effect.csv")
    ap.add_argument("--out-perdoc", default="results_v2/length_effect_perdoc.parquet",
                    help="per-document (corpus,label,wlen,bin,score-per-detector) for instant re-slicing")
    args = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- pool corpora, KEEP short real texts ----
    REG = {"HC3": load_hc3, "TuringBench": load_turingbench, "MAGE": load_mage, "NYT": load_nyt}
    over = 6
    texts, labels, wlen, corp = [], [], [], []
    def add(name, hum, ai):
        h = [t for t in hum if keep_len(t)][:args.n]
        a = [t for t in ai if keep_len(t)][:args.n]
        for t in a: texts.append(t); labels.append(1); wlen.append(len(t.split())); corp.append(name)
        for t in h: texts.append(t); labels.append(0); wlen.append(len(t.split())); corp.append(name)
        console.print(f"  [{name}] kept {len(a)} ai / {len(h)} human")
    for name in [c.strip() for c in args.corpora.split(",") if c.strip()]:
        console.print(f"[{name}] loading ...")
        hum, ai = REG[name](args.n * over)
        add(name, hum, ai)
    for spec in args.hf_corpus:
        p = spec.split(":")
        if len(p) < 6: continue
        nm, did, tcol, lcol, aiv, huv = p[:6]
        cfg = p[6] if len(p) > 6 and p[6] else None
        spl = p[7] if len(p) > 7 and p[7] else "train"
        console.print(f"[{nm}] loading {did} ...")
        try:
            hum, ai = load_hf_generic(args.n * over, did, tcol, lcol, aiv.split(","), huv.split(","), cfg, spl)
            add(nm, hum, ai)
        except Exception as e:
            console.print(f"  [{nm}] load failed: {str(e)[:120]}")

    labels = np.array(labels); wlen = np.array(wlen); corp = np.array(corp)
    console.print(f"\n[pool] {len(texts)} docs  ({int((labels==1).sum())} ai / {int((labels==0).sum())} human)")
    binidx = np.digitize(wlen, BIN_EDGES[1:-1], right=False)  # 0..5
    for b, lab in enumerate(BIN_LABELS):
        m = binidx == b
        console.print(f"    {lab:>8}  n={int(m.sum()):5d}  (ai {int((labels[m]==1).sum())} / hu {int((labels[m]==0).sum())})")

    # ---- load detectors + score every doc once ----
    models = load_models(args, dev)
    console.print("\n[scoring] all detectors on the pooled docs (this is the long part) ...")
    with Progress(SpinnerColumn(), TextColumn("[bold]{task.description}"), BarColumn(),
                  TimeElapsedColumn(), TimeRemainingColumn(), console=console) as prog:
        prog.add_task("scoring", total=None)
        scores = score_all_detectors(texts, models, dev)

    # orient any hf-detector by overall AUC (flip if < 0.5)
    for spec in args.hf_detector:
        dn = spec.split(":", 1)[0]
        if dn in scores and auc(scores[dn], labels) < 0.5:
            scores[dn] = -np.asarray(scores[dn], float)
            console.print(f"  [{dn}] orientation flipped")

    det_order = [d for d in ["GLASS", "Fast-DetectGPT", "Binoculars", "Ghostbuster", "log-rank",
                             "RoBERTa-OpenAI", "RADAR"] + [s.split(":", 1)[0] for s in args.hf_detector]
                 if d in scores]

    # ---- dump per-document scores so any re-slice is instant (no re-scoring) ----
    import pandas as pd
    perdoc = {"corpus": corp, "label": labels, "wlen": wlen, "bin": binidx}
    for d in det_order:
        perdoc[d] = np.asarray(scores[d], float)
    Path(args.out_perdoc).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(perdoc).to_parquet(args.out_perdoc)
    console.print(f"[green]per-doc dump[/] -> {args.out_perdoc}  ({len(labels)} rows)")

    corpora_list = sorted(set(corp.tolist()))
    nb = len(BIN_LABELS)
    mpc = args.min_per_cell

    def cell_auc(mask, d):
        sc = np.asarray(scores[d])[mask]; yy = labels[mask]
        if (yy == 1).sum() < mpc or (yy == 0).sum() < mpc:
            return float("nan")
        return auc(sc, yy)

    # POOLED (count-weighted; confounded by corpus mix -- kept for reference)
    grid_pool = {d: [cell_auc(binidx == b, d) for b in range(nb)] for d in det_order}
    # MACRO over corpora (each bin = mean of per-corpus AUCs -> controls for corpus mix)
    grid_macro = {d: [] for d in det_order}
    contrib = []
    for b in range(nb):
        cs = [c for c in corpora_list
              if ((binidx == b) & (corp == c) & (labels == 1)).sum() >= mpc
              and ((binidx == b) & (corp == c) & (labels == 0)).sum() >= mpc]
        contrib.append(cs)
        for d in det_order:
            vals = [cell_auc((binidx == b) & (corp == c), d) for c in cs]
            vals = [v for v in vals if np.isfinite(v)]
            grid_macro[d].append(float(np.mean(vals)) if vals else float("nan"))

    for b, lab in enumerate(BIN_LABELS):
        console.print(f"    bin {lab:>8}: corpora contributing = {contrib[b] or '--'}")

    def show(grid, title):
        tbl = Table(title=title)
        tbl.add_column("detector")
        for lab in BIN_LABELS: tbl.add_column(lab, justify="right")
        for d in det_order:
            tbl.add_row(d, *[("--" if not np.isfinite(v) else f"{v:.3f}") for v in grid[d]])
        console.print(tbl)
    show(grid_macro, "AUC by length -- MACRO-avg over corpora (corpus-controlled, headline)")
    show(grid_pool, "AUC by length -- pooled (count-weighted, confounded; reference)")

    # ---- CSV (both) ----
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w") as f:
        f.write("method,detector," + ",".join(BIN_LABELS) + "\n")
        for d in det_order:
            f.write("macro," + d + "," + ",".join(f"{v:.4f}" for v in grid_macro[d]) + "\n")
        for d in det_order:
            f.write("pooled," + d + "," + ",".join(f"{v:.4f}" for v in grid_pool[d]) + "\n")

    # ---- plot (macro-averaged, the clean corpus-controlled curve) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.arange(nb)
        fig, axp = plt.subplots(figsize=(7, 4.3))
        for d in det_order:
            y = np.array(grid_macro[d], float)
            lw = 3.0 if d == "GLASS" else 1.6
            z = 5 if d == "GLASS" else 2
            axp.plot(x, y, marker="o", lw=lw, zorder=z, label=d)
        axp.axhline(0.5, ls=":", c="gray", lw=1)
        axp.set_xticks(x); axp.set_xticklabels(BIN_LABELS)
        axp.set_xlabel("document length (words)"); axp.set_ylabel("detection AUC (macro-avg over corpora)")
        axp.set_ylim(0, 1.02); axp.grid(alpha=0.25)
        axp.legend(fontsize=8, ncol=2, loc="lower right")
        axp.set_title("Detection AUC vs document length (corpus-controlled)")
        fig.tight_layout()
        Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_fig + ".pdf"); fig.savefig(args.out_fig + ".png", dpi=150)
        console.print(f"[green]plot[/] -> {args.out_fig}.pdf / .png")
    except Exception as e:
        console.print(f"[yellow]plot skipped:[/] {str(e)[:120]}")

    # ---- LaTeX table (macro-averaged) ----
    L = ["% Detection AUC by document length, macro-averaged over five cross-corpus benchmarks",
         "% (each length bin averages per-corpus AUCs -> controls for corpus composition).",
         "\\begin{table}[t]\\centering\\small",
         "\\caption{Detection AUC as a function of document length (words), macro-averaged "
         "over HC3, TuringBench, MAGE, NYT, and Pile (each bin averages the per-corpus AUCs, "
         "controlling for corpus composition). \\textbf{Bold} = best per column.}",
         "\\label{tab:length-effect}",
         "\\begin{tabular}{l" + "c" * nb + "}", "\\toprule",
         "Detector & " + " & ".join(BIN_LABELS) + " \\\\", "\\midrule"]
    best = [max((d for d in det_order if np.isfinite(grid_macro[d][b])),
                key=lambda d: grid_macro[d][b], default=None) for b in range(nb)]
    for d in det_order:
        cells = []
        for b in range(nb):
            v = grid_macro[d][b]
            c = "--" if not np.isfinite(v) else f"{v:.3f}"
            if d == best[b] and np.isfinite(v): c = f"\\textbf{{{c}}}"
            cells.append(c)
        nm = "\\textbf{GLASS}" if d == "GLASS" else d
        L.append(nm + " & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    Path(args.out_tex).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_tex).write_text("\n".join(L) + "\n")
    console.print(f"[green]done[/] -> {args.out_tex}, {args.out_csv}, {args.out_perdoc}")


if __name__ == "__main__":
    main()
