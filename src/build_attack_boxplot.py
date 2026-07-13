#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_attack_boxplot.py  (BBC bbplot-style)
===========================================
Box plots of cross-corpus detection under attack. Reads the per-(corpus, attack,
detector) AUCs from build_thirdparty_attack_v3.py and draws one panel per attack
stage (Clean / Character / Paraphrase). Because there are only THREE corpora
(HC3, TuringBench, MAGE), each box spans just 3 points -- so the three individual
corpus AUCs are overlaid as labelled dots, and a chance line is drawn at 0.5.
Nothing is aggregated away.

Run:
    python3 build_attack_boxplot.py
    python3 build_attack_boxplot.py --stages clean,character   # drop paraphrase
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

BBC_TEXT = "#222222"; BBC_GRID = "#cbcbcb"
DET_ORDER = ["GLASS", "Fast-DetectGPT", "Binoculars", "Ghostbuster", "log-rank", "RoBERTa-OpenAI"]
DET_COLOR = {"GLASS": "#BB1919", "Fast-DetectGPT": "#1380A1", "Ghostbuster": "#588300",
             "log-rank": "#707070", "RoBERTa-OpenAI": "#FAAB18", "Binoculars": "#990099"}
CORP_MARK = {"HC3": "o", "TuringBench": "s", "MAGE": "^"}
STAGE_PRETTY = {"clean": "Clean", "character": "Character attack", "paraphrase": "Paraphrase"}


def _font():
    for fam in ("Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"):
        if any(fam.lower() in f.name.lower() for f in fm.fontManager.ttflist):
            return fam
    return "DejaVu Sans"


def panel(ax, df, stage, dets):
    sub = df[df["attack"] == stage]
    box_data, positions, colors = [], [], []
    for i, det in enumerate(dets):
        vals = sub[sub["detector"] == det]["auc"].dropna().values
        if len(vals) == 0: continue
        box_data.append(vals); positions.append(i); colors.append(DET_COLOR[det])
    bp = ax.boxplot(box_data, positions=positions, widths=0.55, patch_artist=True,
                    showfliers=False, zorder=2,
                    medianprops=dict(color=BBC_TEXT, lw=1.3),
                    whiskerprops=dict(color="#888888", lw=1.0),
                    capprops=dict(color="#888888", lw=1.0))
    for b, c, det in zip(bp["boxes"], colors, dets):
        b.set(facecolor=c, edgecolor=c, lw=0.8, alpha=0.30 if det != "GLASS" else 0.45)
    # overlay the 3 corpus dots
    for i, det in enumerate(dets):
        ds = sub[sub["detector"] == det]
        for _, r in ds.iterrows():
            ax.scatter(i, r["auc"], marker=CORP_MARK.get(r["corpus"], "o"),
                       s=44, facecolor=DET_COLOR[det], edgecolor="white", lw=0.8, zorder=4)
    ax.axhline(0.5, color="#999999", lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(len(dets) - 0.45, 0.505, "chance", fontsize=9, color="#999999", va="bottom", ha="right")
    ax.set_xticks(range(len(dets)))
    ax.set_xticklabels([d.replace("-OpenAI", "") for d in dets], rotation=18, ha="right")
    ax.tick_params(axis="both", length=0, labelsize=11, colors=BBC_TEXT)
    ax.yaxis.grid(True, color=BBC_GRID, lw=0.9, zorder=0); ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_ylim(0.0, 1.03); ax.set_xlim(-0.6, len(dets) - 0.4)
    ax.set_title(STAGE_PRETTY.get(stage, stage), fontsize=13, fontweight="bold",
                 color=BBC_TEXT, loc="left", pad=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="results_v2/thirdparty_attack.parquet")
    ap.add_argument("--stages", default="clean,character,paraphrase")
    ap.add_argument("--out", default="figs/fig_attack_boxplot")
    args = ap.parse_args()

    p = args.parquet
    df = pd.read_csv(p) if str(p).lower().endswith(".csv") else pd.read_parquet(p)
    stages = [s.strip() for s in args.stages.split(",") if s.strip() in set(df["attack"])]
    dets = [d for d in DET_ORDER if d in set(df["detector"])]

    plt.rcParams.update({"font.family": _font(), "text.color": BBC_TEXT,
                         "axes.edgecolor": BBC_TEXT, "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, len(stages), figsize=(4.7 * len(stages), 5.0), squeeze=False)
    for ax, st in zip(axes[0], stages):
        panel(ax, df, st, dets)
    axes[0][0].set_ylabel("Detection AUC", fontsize=12, color=BBC_TEXT)

    fig.suptitle("Cross-corpus detection under attack", x=0.012, y=1.02,
                 ha="left", fontsize=18, fontweight="bold", color=BBC_TEXT)
    fig.text(0.012, 0.965, "Each box spans the three corpora (HC3 ●, TuringBench ■, MAGE ▲); "
             "AI side attacked, humans clean.", ha="left", fontsize=11, color="#555555")
    # corpus-marker legend
    handles = [plt.Line2D([0], [0], marker=m, color="#555555", linestyle="none",
                          markersize=8, markerfacecolor="#555555", markeredgecolor="white", label=c)
               for c, m in CORP_MARK.items()]
    axes[0][-1].legend(handles=handles, loc="lower left", frameon=False, fontsize=10,
                       title="corpus", title_fontsize=10)
    fig.text(0.012, 0.01, "Source: frozen v3_english bundle on HC3 / TuringBench / MAGE; "
             "character = homoglyph/transposition edit. GLASS in red.",
             fontsize=9, color="#767676")
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    fig.savefig(args.out + ".png", dpi=200, bbox_inches="tight")
    print(f"[out] {args.out}.pdf / .png  ({len(stages)} stages x {len(dets)} detectors)")


if __name__ == "__main__":
    main()
