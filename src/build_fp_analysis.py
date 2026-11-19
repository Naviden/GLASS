#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyse false-positive rates, document length and nearest-generator subspaces."""
from __future__ import annotations
import os, warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

import argparse, collections
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from build_thirdparty_attack_v3 import (
    load_bundle, encode, load_mage, load_hc3, load_turingbench, load_nyt,
)

console = Console()


def residual_sq(X, mean, B):
    Xc = X - mean
    return np.sum((Xc - Xc @ B @ B.T) ** 2, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=".cache/glass_production/v3_diverse_nocohere")
    ap.add_argument("--encoder", default="microsoft/deberta-v3-base")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", default="results_v2/fp_analysis.txt")
    args = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    SH, SA, WH = load_bundle(args.bundle, args.K)
    gens = list(SA.keys())
    wh = lambda X: (X - WH[0]) @ WH[1].T
    console.print(f"[bold cyan]{args.bundle}[/] — {len(SA)} subspaces\n")

    corpora = [("HC3", load_hc3), ("TuringBench", load_turingbench), ("MAGE", load_mage), ("NYT", load_nyt)]
    table = Table(title="GLASS false positives (human misread as AI)")
    for c in ["corpus", "FP rate", "FP len", "OK len", "top subspace catching humans"]:
        table.add_column(c)
    dump = [f"False-positive human texts on {args.bundle}\n"]

    with Progress(SpinnerColumn(), TextColumn("[bold]{task.description}"), BarColumn(),
                  TimeElapsedColumn(), TimeRemainingColumn(), console=console) as prog:
        t = prog.add_task("encoding corpora", total=len(corpora))
        for name, loader in corpora:
            hum, ai = loader(args.n)
            if len(hum) < 50:
                prog.update(t, advance=1); continue
            Xw = wh(encode(hum, args.encoder, device=dev))
            rH = residual_sq(Xw, *SH)
            R = np.stack([residual_sq(Xw, m, b) for (m, b) in SA.values()], 1)
            minA = R.min(1); argmin = R.argmin(1); s = rH - minA
            fp = s > 0
            lens = np.array([len(x) for x in hum])
            cnt = collections.Counter(gens[i] for i in argmin[fp]) if fp.any() else collections.Counter()
            top = ", ".join(f"{g}({c})" for g, c in cnt.most_common(2)) or "-"
            table.add_row(name, f"{fp.mean()*100:.1f}%",
                          f"{np.median(lens[fp]):.0f}" if fp.any() else "-",
                          f"{np.median(lens[~fp]):.0f}" if (~fp).any() else "-", top)
            dump.append(f"\n{'='*70}\n[{name}] FP {fp.sum()}/{len(hum)} ({fp.mean()*100:.1f}%)  "
                        f"top catching: {top}")
            order = [i for i in np.argsort(-s) if fp[i]][:args.top]
            for i in order:
                txt = " ".join(hum[i].split())[:380]
                dump.append(f"  [s={s[i]:.0f} rH={rH[i]:.0f} minA={minA[i]:.0f} -> {gens[argmin[i]]}]\n    {txt}\n")
            prog.update(t, advance=1)

    console.print(table)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(dump), encoding="utf-8")
    console.print(f"\n[green]✓[/] full FP-text dump → [bold]{args.out}[/]  (scp it over so I can read the examples)")


if __name__ == "__main__":
    main()
