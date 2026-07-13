#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_logo_boxplot.py  (BBC bbplot-style)
=========================================
Box plots of leave-one-generator-out (unseen-generator) detection, OVERALL and
PER-DOMAIN, styled after the BBC Visual & Data Journalism cookbook (bbplot):
BBC palette, bold left title + subtitle, horizontal gridlines only, no spines or
ticks, clean top legend, source footer. GLASS in BBC red.

Inputs (from build_logo_detection_v3.py):
  results_v2/logo_detection.parquet           # overall: one AUC/TPR per (seed,gen)
  results_v2/logo_detection_bydomain.parquet  # long: (seed,gen,domain,detector,auc,tpr)

Run:
    python3 build_logo_boxplot.py                 # AUC + TPR
    python3 build_logo_boxplot.py --metric auc
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.font_manager as fm

# ---- BBC bbplot palette ----
BBC_TEXT = "#222222"
BBC_GRID = "#cbcbcb"
DET_ORDER = ["GLASS", "Fast-DetectGPT", "Binoculars", "Ghostbuster", "log-rank", "RoBERTa-OpenAI"]
DET_COLOR = {
    "GLASS":          "#BB1919",  # BBC News red
    "Fast-DetectGPT": "#1380A1",  # BBC teal-blue
    "Ghostbuster":    "#588300",  # BBC green
    "log-rank":       "#707070",  # grey
    "RoBERTa-OpenAI": "#FAAB18",  # BBC yellow
    "Binoculars":     "#990099",  # purple
}
DOM_PRETTY = {"abstracts": "Abstracts", "books": "Books", "news": "News",
              "poetry": "Poetry", "recipes": "Recipes", "reddit": "Reddit",
              "reviews": "Reviews", "wiki": "Wikipedia"}


def _font():
    for fam in ("Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"):
        if any(fam.lower() in f.name.lower() for f in fm.fontManager.ttflist):
            return fam
    return "DejaVu Sans"


def long_overall(df):
    rows = []
    for d in DET_ORDER:
        if f"{d}_auc" not in df.columns: continue
        for _, r in df.iterrows():
            rows.append({"domain": "Overall", "detector": d,
                         "auc": r.get(f"{d}_auc", np.nan), "tpr": r.get(f"{d}_tpr", np.nan)})
    return pd.DataFrame(rows)


def panel(ax, data, metric, dets, domains):
    n = len(dets); w = 0.74 / n
    for di, det in enumerate(dets):
        pos, vals = [], []
        for xi, dom in enumerate(domains):
            v = data[(data["domain"] == dom) & (data["detector"] == det)][metric].dropna().values
            if len(v) == 0: continue
            pos.append(xi + (di - (n - 1) / 2) * w); vals.append(v)
        if not vals: continue
        c = DET_COLOR[det]; glass = det == "GLASS"
        bp = ax.boxplot(vals, positions=pos, widths=w * 0.88, patch_artist=True,
                        showfliers=False, zorder=3,
                        medianprops=dict(color="white" if glass else BBC_TEXT, lw=1.4),
                        whiskerprops=dict(color=c, lw=1.1),
                        capprops=dict(color=c, lw=1.1))
        for b in bp["boxes"]:
            b.set(facecolor=c, edgecolor=c, lw=0.8, alpha=1.0 if glass else 0.92)
    # bbplot styling
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels([DOM_PRETTY.get(d, d.capitalize()) if d != "Overall" else "Overall"
                        for d in domains])
    ax.tick_params(axis="both", length=0, labelsize=12, colors=BBC_TEXT)
    ax.axvline(0.5, color=BBC_GRID, lw=1.0, zorder=1)        # split Overall | per-domain
    ax.yaxis.grid(True, color=BBC_GRID, lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_visible(False)
    lo = 0.4 if metric == "auc" else 0.0
    ax.set_ylim(lo, 1.02); ax.set_xlim(-0.6, len(domains) - 0.4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overall", default="results_v2/logo_detection.parquet")
    ap.add_argument("--bydomain", default="results_v2/logo_detection_bydomain.parquet")
    ap.add_argument("--metric", default="both", choices=["auc", "tpr", "both"])
    ap.add_argument("--out", default="figs/fig_logo_boxplot")
    args = ap.parse_args()

    ov = long_overall(pd.read_parquet(args.overall))
    if Path(args.bydomain).exists():
        bd = pd.read_parquet(args.bydomain)[["domain", "detector", "auc", "tpr"]]
        data = pd.concat([ov, bd], ignore_index=True)
    else:
        print(f"[warn] {args.bydomain} not found -> OVERALL-only.")
        data = ov
    dets = [d for d in DET_ORDER if d in set(data["detector"])]
    doms = ["Overall"] + sorted(d for d in data["domain"].unique() if d != "Overall")

    plt.rcParams.update({"font.family": _font(), "text.color": BBC_TEXT,
                         "axes.edgecolor": BBC_TEXT, "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.facecolor": "white"})
    metrics = ["auc", "tpr"] if args.metric == "both" else [args.metric]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(13.5, 4.7 * len(metrics)), squeeze=False)
    sub = {"auc": "Area under the ROC curve (AUC), by domain",
           "tpr": "True-positive rate at 1% false-positive rate, by domain"}
    for i, (ax, m) in enumerate(zip(axes[:, 0], metrics)):
        panel(ax, data, m, dets, doms)
        if i == 0:
            ax.text(0, 1.30, "Detecting unseen generators", transform=ax.transAxes,
                    fontsize=20, fontweight="bold", color=BBC_TEXT, va="bottom")
            ax.text(0, 1.225, "Leave-one-generator-out: every detector is tested only on "
                    "generators it has never seen", transform=ax.transAxes,
                    fontsize=12.5, color="#555555", va="bottom")
        ax.text(0, 1.045, sub[m], transform=ax.transAxes, fontsize=13,
                fontweight="bold", color=BBC_TEXT, va="bottom")

    # clean top legend (no frame), like bbplot
    handles = [Patch(facecolor=DET_COLOR[d], edgecolor=DET_COLOR[d], label=d) for d in dets]
    axes[0, 0].legend(handles=handles, ncol=len(dets), loc="lower left",
                      bbox_to_anchor=(0, 1.40), frameon=False, fontsize=12,
                      handlelength=1.1, handleheight=1.1, columnspacing=1.4, labelcolor=BBC_TEXT)
    # source footer
    fig.text(0.012, 0.012, "Source: frontier-EN panels · leave-one-generator-out "
             "(18 generators held out in turn × 5 domain-stratified seeds). "
             "GLASS in red.", fontsize=9.5, color="#767676")
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    fig.savefig(args.out + ".png", dpi=200, bbox_inches="tight")
    print(f"[out] {args.out}.pdf / .png  ({len(doms)} domains x {len(dets)} detectors)")


if __name__ == "__main__":
    main()
