#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_extra_panels.py   (rich UI)
=================================
Generate AI text from ADDITIONAL, eval-disjoint generators on the RAID prompts to
legitimately expand GLASS's subspace library (no eval text / eval generators used).
Writes per_doc.jsonl panels to results_v2/extra_en_<gen>/ (kept in the *en* glob).

    OPENROUTER_API_KEY=sk-... python3 build_extra_panels.py \
        --models "anthropic/claude-sonnet-4.5,google/gemini-2.5-flash,x-ai/grok-4.20,deepseek/deepseek-v3.2" \
        --per-domain 200
"""
from __future__ import annotations
import os, warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

import argparse, json, re, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from rich.console import Console
from rich.progress import (Progress, SpinnerColumn, BarColumn, TextColumn,
                           MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn)

console = Console()


def safe(name): return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def generate(model, prompt, key, max_tokens=600, temperature=0.7, retries=4):
    """Returns (text, error); error is None on success."""
    import requests
    url = "https://openrouter.ai/api/v1/chat/completions"
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "max_tokens": max_tokens}
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=120)
            if r.status_code == 429:
                last = "429 rate-limited"; time.sleep(2 ** attempt + 1); continue
            if r.status_code >= 400:
                return "", f"HTTP {r.status_code}: {r.text[:180]}"
            return (r.json()["choices"][0]["message"]["content"] or "").strip(), None
        except Exception as e:
            last = str(e)[:180]; time.sleep(1.5 * (attempt + 1))
    return "", last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--prompts", default="results_v2/raid_en_prompts.parquet")
    ap.add_argument("--per-domain", type=int, default=200)
    ap.add_argument("--workers", type=int, default=16, help="parallel API requests")
    ap.add_argument("--out-root", default="results_v2")
    args = ap.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        console.print("[bold red]set OPENROUTER_API_KEY[/]"); raise SystemExit(1)

    import pandas as pd
    P = pd.read_parquet(args.prompts).dropna(subset=["prompt", "title", "domain"])
    rng = np.random.default_rng(0)
    S = pd.concat([g.iloc[rng.choice(len(g), min(args.per_domain, len(g)), replace=False)]
                   for _, g in P.groupby("domain")]).reset_index(drop=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    console.print(f"[bold cyan]{len(S)}[/] prompts × [bold cyan]{len(models)}[/] models "
                  f"({S['domain'].nunique()} domains)")

    with Progress(SpinnerColumn(), TextColumn("[bold]{task.description}"), BarColumn(),
                  MofNCompleteColumn(), TextColumn("[green]{task.fields[kept]} kept"),
                  TimeElapsedColumn(), TimeRemainingColumn(), console=console) as prog:
        for model in models:
            outdir = Path(args.out_root) / f"extra_en_{safe(model)}"; outdir.mkdir(parents=True, exist_ok=True)
            fout = open(outdir / "per_doc.jsonl", "w", encoding="utf-8")
            task = prog.add_task(model, total=len(S), kept=0)
            n_ok = fails = 0; last_err = None
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                fut2meta = {ex.submit(generate, model, str(r["prompt"]), key):
                            (str(r["title"]), str(r["domain"])) for _, r in S.iterrows()}
                for fut in as_completed(fut2meta):
                    title, domain = fut2meta[fut]
                    resp, err = fut.result()
                    if len(resp) >= 20:
                        fout.write(json.dumps({"model": model, "title": title,
                                               "response": resp, "domain": domain}) + "\n")
                        n_ok += 1; fails = 0
                    else:
                        fails += 1; last_err = err or last_err
                    prog.update(task, advance=1, kept=n_ok)
                    if n_ok == 0 and fails >= 6:                 # bad key/model/credits -> stop this model
                        for f in fut2meta: f.cancel()
                        console.print(f"[bold red][abort][/] {model}: 6 failures, 0 kept. Last error: {last_err}")
                        break
            fout.close()
            console.print(f"  [green]✓[/] {model} → {outdir}/per_doc.jsonl  ([bold]{n_ok}[/] docs)")


if __name__ == "__main__":
    main()
