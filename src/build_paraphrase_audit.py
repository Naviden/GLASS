#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure truncation and length confounds in the paraphrase evaluation."""
from __future__ import annotations
import argparse, re
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()


def auc(s, y, min_n=10):
    s = np.asarray(s, float); y = np.asarray(y)
    ok = np.isfinite(s); s, y = s[ok], y[ok]
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos < min_n or nneg < min_n:
        return np.nan
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); r = ((csum - cnt + csum + 1) / 2.0)[inv]
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def sym(a):
    if not np.isfinite(a):
        return np.nan, "-"
    return (a, "longer=AI") if a >= 0.5 else (1 - a, "shorter=AI")


_SENT = re.compile(r"(?<=[.!?])\s+")


def load_para(dev):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    name = "humarin/chatgpt_paraphraser_on_T5_base"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(name).to(dev).eval()
    return tok, mdl


def para_published(texts, tok, mdl, dev, batch=8):
    """EXACTLY what the paper ran: input truncated at 256, output capped at 256."""
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = [f"paraphrase: {t}" for t in texts[i:i + batch]]
            enc = tok(chunk, padding=True, truncation=True, max_length=256,
                      return_tensors="pt").to(dev)
            g = mdl.generate(**enc, max_length=256, num_beams=4, do_sample=True,
                             temperature=0.7, top_p=0.95)
            out.extend(tok.batch_decode(g, skip_special_tokens=True))
            console.print(f"  published-paraphrase {min(i+batch,len(texts))}/{len(texts)}",
                          end="\r")
    console.print()
    return out


def para_fixed(texts, tok, mdl, dev, batch=8, win=3):
    """What the attack SHOULD have been: paraphrase the whole document by sliding a
    window of `win` sentences, so nothing is truncated and length is preserved."""
    import torch
    out = []
    for k, t in enumerate(texts):
        sents = [s for s in _SENT.split(t) if s.strip()]
        pieces = [" ".join(sents[i:i + win]) for i in range(0, len(sents), win)]
        res = []
        with torch.no_grad():
            for i in range(0, len(pieces), batch):
                chunk = [f"paraphrase: {p}" for p in pieces[i:i + batch]]
                enc = tok(chunk, padding=True, truncation=True, max_length=256,
                          return_tensors="pt").to(dev)
                g = mdl.generate(**enc, max_length=256, num_beams=4, do_sample=True,
                                 temperature=0.7, top_p=0.95)
                res.extend(tok.batch_decode(g, skip_special_tokens=True))
        out.append(" ".join(res))
        console.print(f"  length-preserving-paraphrase {k+1}/{len(texts)}", end="\r")
    console.print()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="AI docs per corpus")
    ap.add_argument("--corpora", default="HC3,TuringBench,MAGE,NYT")
    ap.add_argument("--min-words", type=int, default=100)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results_v2/paraphrase_audit.csv")
    args = ap.parse_args()

    import torch
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold]device[/] = {dev}  (CPU is fine: T5-base is 220M)")

    from build_thirdparty_competitors_v3 import (load_hc3, load_turingbench,
                                                 load_mage, load_nyt)
    from text_quality import filter_pair
    REG = {"HC3": load_hc3, "TuringBench": load_turingbench,
           "MAGE": load_mage, "NYT": load_nyt}
    rng = np.random.default_rng(0)

    tok, mdl = load_para(dev)
    rows = []
    for name in args.corpora.split(","):
        loader = REG.get(name)
        if loader is None:
            continue
        console.rule(f"[bold]{name}")
        ai, hum = loader(args.n * 8)
        ai, hum, _, _ = filter_pair(ai, hum, args.n, min_words=args.min_words)
        ai, hum = list(ai)[:args.n], list(hum)[:args.n]
        console.print(f"{len(ai)} AI / {len(hum)} human, all >= {args.min_words} words")

        pub = para_published(ai, tok, mdl, dev)
        fix = para_fixed(ai, tok, mdl, dev)

        w_ai = np.array([len(t.split()) for t in ai], float)
        w_hum = np.array([len(t.split()) for t in hum], float)
        w_pub = np.array([len(t.split()) for t in pub], float)
        w_fix = np.array([len(t.split()) for t in fix], float)

        def lenauc(w_attacked):
            s = np.concatenate([w_attacked, w_hum])
            y = np.array([1] * len(w_attacked) + [0] * len(w_hum))
            return sym(auc(s, y))

        a_clean, d_clean = lenauc(w_ai)
        a_pub, d_pub = lenauc(w_pub)
        a_fix, d_fix = lenauc(w_fix)
        kept_pub = float(np.mean(w_pub / np.maximum(w_ai, 1)))
        kept_fix = float(np.mean(w_fix / np.maximum(w_ai, 1)))
        trunc = float(np.mean(w_ai > 190))       # docs long enough to be cut by the cap

        rows.append(dict(corpus=name, n=len(ai),
                         median_ai=np.median(w_ai), median_hum=np.median(w_hum),
                         median_pub=np.median(w_pub), median_fix=np.median(w_fix),
                         frac_text_kept_published=kept_pub, frac_text_kept_fixed=kept_fix,
                         frac_docs_over_cap=trunc,
                         lenauc_clean=a_clean, lenauc_published=a_pub, lenauc_fixed=a_fix))
        console.print(f"  words:  AI {np.median(w_ai):.0f}  ->  published-paraphrase "
                      f"{np.median(w_pub):.0f}  |  length-preserving {np.median(w_fix):.0f}"
                      f"   (humans {np.median(w_hum):.0f}, untouched)")
        console.print(f"  text kept: published {kept_pub:.0%}   fixed {kept_fix:.0%}   "
                      f"({trunc:.0%} of AI docs exceed the 256-token cap)")
        console.print(f"  [bold]word-count-only AUC vs clean humans[/]: "
                      f"clean {a_clean:.3f} ({d_clean})  ->  "
                      f"PUBLISHED ATTACK [red]{a_pub:.3f}[/] ({d_pub})  |  "
                      f"length-preserving [green]{a_fix:.3f}[/] ({d_fix})")
        print(f"RESULT[para {name}] kept={kept_pub:.3f} lenauc_clean={a_clean:.3f} "
              f"lenauc_published={a_pub:.3f} lenauc_fixed={a_fix:.3f}")

    R = pd.DataFrame(rows)
    R.to_csv(args.out, index=False)

    console.rule("[bold]VERDICT")
    t = Table(title="Does OUR paraphrase attack manufacture a length confound?")
    for c in ["corpus", "AI words", "after published attack", "text kept",
              "word-count AUC (clean)", "word-count AUC (published attack)",
              "word-count AUC (length-preserving)"]:
        t.add_column(c, justify="right")
    for _, r in R.iterrows():
        bad = r.lenauc_published >= 0.75
        col = "red" if bad else "green"
        t.add_row(r.corpus, f"{r.median_ai:.0f}", f"{r.median_pub:.0f}",
                  f"{r.frac_text_kept_published:.0%}", f"{r.lenauc_clean:.3f}",
                  f"[{col}]{r.lenauc_published:.3f}[/]", f"{r.lenauc_fixed:.3f}")
    console.print(t)
    mp = float(R.lenauc_published.mean()); mc = float(R.lenauc_clean.mean())
    mf = float(R.lenauc_fixed.mean())
    print(f"RESULT[para MEAN] lenauc_clean={mc:.3f} lenauc_published={mp:.3f} "
          f"lenauc_fixed={mf:.3f}")
    if mp >= 0.75 and mp - mc > 0.15:
        console.print(f"\n[red bold]CONFIRMED[/]: the published paraphrase attack raises the "
                      f"word-count-only AUC from {mc:.3f} to {mp:.3f}. The attack truncates "
                      f"the AI side and leaves humans full length, so the 'paraphrase' column "
                      f"of both attack tables rewards detectors that read length. "
                      f"'GLASS loses paraphrase' is an artifact of our own attack code. "
                      f"The attack must be re-run length-preserving ({mf:.3f}).")
    else:
        console.print(f"\n[green bold]NOT CONFIRMED[/]: word-count AUC {mc:.3f} -> {mp:.3f}. "
                      f"The truncation does not create a usable length shortcut; the "
                      f"paraphrase result stands as published.")


if __name__ == "__main__":
    main()
