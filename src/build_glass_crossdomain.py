#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_glass_crossdomain.py   (rich UI, GLASS-only, fast)
========================================================
GLASS is 1.000 on all 8 in-distribution RAID domains, so its cross-corpus dips
are distribution-shift, not content-genre. This script answers the real
question: OUT of distribution, which content GENRE does GLASS fail on most?

It keeps the per-record genre labels that the pooled loaders throw away:
  * HC3   -> field "source"  (reddit_eli5, open_qa, wiki_csai, medicine, finance)
  * MAGE  -> field "src"     (xsum, yelp, cmv, eli5, wp, squad, sci_gen, ...)

For each genre: GLASS AUC (higher=better) and human false-positive rate, sorted
worst genre first. GLASS-only so it runs in ~5 min (DeBERTa encode, no Falcon).

Run (turing, ~/glass/paper, venv, GPU):
  PYTHONUTF8=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false \
  PYTHONWARNINGS=ignore python3 build_glass_crossdomain.py \
      --bundle .cache/glass_production/v3_clean_pruned [--clean-text]
"""
from __future__ import annotations
import os, warnings, time
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

import argparse, json, collections
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from build_thirdparty_attack_v3 import load_bundle, glass_score, encode

console = Console()


def auc(scores, y):
    """Mann-Whitney AUC (higher score => label 1 = AI)."""
    s = np.asarray(scores, float); y = np.asarray(y)
    ok = np.isfinite(s); s = s[ok]; y = y[ok]
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) < 3 or len(neg) < 3: return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s) + 1)
    # average ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); start = csum - cnt
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    r_pos = ranks[y == 1].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ---------------- genre-labelled loaders (keep the label the pooled ones drop) --
def hc3_by_genre(cap):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("Hello-SimpleAI/HC3", "all.jsonl", repo_type="dataset")
    H = collections.defaultdict(list); A = collections.defaultdict(list)
    for line in open(p, encoding="utf-8"):
        try: d = json.loads(line)
        except Exception: continue
        g = str(d.get("source") or "unknown")
        for h in (d.get("human_answers") or []):
            if isinstance(h, str) and h.strip() and len(H[g]) < cap: H[g].append(h.strip())
        for a in (d.get("chatgpt_answers") or []):
            if isinstance(a, str) and a.strip() and len(A[g]) < cap: A[g].append(a.strip())
    return H, A


def mage_by_genre(cap):
    from datasets import load_dataset
    ds = load_dataset("yaful/MAGE", split="train", streaming=True)
    H = collections.defaultdict(list); A = collections.defaultdict(list)
    for r in ds:
        t = (r.get("text") or "").strip()
        if len(t) < 50: continue
        src = str(r.get("src") or r.get("source") or "unknown")
        g = src.split("_")[0]                      # dataset prefix = genre
        if r["label"] == 1 and len(H[g]) < cap: H[g].append(t)      # 1 = human
        elif r["label"] == 0 and len(A[g]) < cap: A[g].append(t)    # 0 = machine
        if sum(len(v) for v in H.values()) >= 40 * cap and \
           sum(len(v) for v in A.values()) >= 40 * cap: break
    return H, A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=".cache/glass_production/v3_clean_pruned")
    ap.add_argument("--encoder", default="microsoft/deberta-v3-base")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--cap", type=int, default=400, help="max texts per genre per side")
    ap.add_argument("--min-per-side", type=int, default=30)
    ap.add_argument("--clean-text", action="store_true")
    ap.add_argument("--out", default="results_v2/glass_crossdomain.txt")
    args = ap.parse_args()

    keep = (lambda t: True)
    if args.clean_text:
        from text_quality import is_clean as keep

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    SH, SA, WH = load_bundle(args.bundle, args.K)
    console.print(f"[bold cyan]{args.bundle}[/] — {len(SA)} subspaces  (K={args.K})")
    console.print(f"[dim]clean-text filter: {'ON' if args.clean_text else 'off'}[/]\n")

    sources = [("HC3", hc3_by_genre), ("MAGE", mage_by_genre)]
    dump = [f"GLASS cross-corpus per-genre  |  bundle={args.bundle}  clean={args.clean_text}\n"]
    all_rows = []

    with Progress(SpinnerColumn(), TextColumn("[bold]{task.description}"), BarColumn(),
                  TimeElapsedColumn(), TimeRemainingColumn(), console=console) as prog:
        for cname, loader in sources:
            task = prog.add_task(f"loading {cname}", total=None)
            H, A = loader(args.cap)
            genres = sorted(set(H) | set(A))
            rows = []
            for g in genres:
                hum = [t for t in H.get(g, []) if keep(t)]
                ai = [t for t in A.get(g, []) if keep(t)]
                if len(hum) < args.min_per_side or len(ai) < args.min_per_side:
                    continue
                Xh = encode(hum, args.encoder, device=dev)
                Xa = encode(ai, args.encoder, device=dev)
                sh = glass_score(Xh, SH, SA, WH)
                sa = glass_score(Xa, SH, SA, WH)
                s = np.concatenate([sa, sh]); y = np.array([1] * len(sa) + [0] * len(sh))
                a = auc(s, y)
                fp = float((sh > 0).mean())            # humans wrongly called AI
                rows.append((cname, g, len(hum), len(ai), a, fp))
            all_rows += rows
            prog.update(task, completed=1, total=1)

    # sort worst GLASS AUC first
    all_rows.sort(key=lambda r: (np.nan_to_num(r[4], nan=1.0)))
    table = Table(title="GLASS per-genre, out of distribution (worst first)")
    for c, j in [("corpus", "left"), ("genre", "left"), ("n_hum", "right"),
                 ("n_ai", "right"), ("GLASS AUC", "right"), ("human FP", "right")]:
        table.add_column(c, justify=j)
    for cname, g, nh, na, a, fp in all_rows:
        col = "red" if a < 0.75 else ("yellow" if a < 0.9 else "green")
        table.add_row(cname, g, str(nh), str(na),
                      f"[{col}]{a:.3f}[/]", f"{fp*100:.0f}%")
        dump.append(f"{cname:5s} {g:14s} n_hum={nh:4d} n_ai={na:4d}  AUC={a:.3f}  FP={fp*100:.0f}%")
    console.print(table)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(dump) + "\n", encoding="utf-8")
    console.print(f"\n[green]done[/] → [bold]{args.out}[/]")


if __name__ == "__main__":
    main()
