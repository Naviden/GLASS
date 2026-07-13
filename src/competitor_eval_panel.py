#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
competitor_eval_panel.py
========================
Run competitor detectors on the SAME English-panel test split that
glass_eval_panel.py uses, so the comparison is head-to-head on identical docs.

Detectors (on the test set = AI docs of held-out titles vs their paired humans):
  - Fast-DetectGPT (analytic, Falcon-7B)      [paper's strongest zero-shot]
  - log-rank (Falcon-7B)
  - RoBERTa-OpenAI (supervised GPT-2 detector)
  - (optional) Binoculars, with --binoculars  [loads a second Falcon]

The Falcon scorer and RoBERTa scorer are byte-for-byte the repo's
score_falcon_extras / roberta score_doc. GLASS's own row is read from the
glass_en_s<seed>.json written by glass_eval_panel.py.

Run (turing, ~/glass/paper, GPU; ~few min for Falcon over the test docs):
    python3 competitor_eval_panel.py \
        --jsonl results_v2/openrouter_en_s0/per_doc.jsonl \
        --humans results_v2/raid_en_humans.parquet \
        --glass-json results_v2/glass_en_s0.json --seed 0
"""
from __future__ import annotations
import argparse, json
import numpy as np
import pandas as pd


def auc_tpr(scores, y):
    """AUC and TPR@1%FPR with 'higher score = more AI' orientation."""
    from sklearn.metrics import roc_auc_score, roc_curve
    ok = np.isfinite(scores)
    s, yy = scores[ok], y[ok]
    auc = float(roc_auc_score(yy, s))
    fpr, tpr, _ = roc_curve(yy, s)
    tpr1 = float(np.interp(0.01, fpr, tpr))
    return auc, tpr1, int(ok.sum())


# ---- repo's exact scorers (inlined) ----------------------------------------
def score_falcon_extras(text, model, tok, device, max_length=512):
    """Returns (log_rank, fast_detectgpt, log_ppl) under one Falcon pass."""
    import torch, torch.nn.functional as F
    enc = tok(text, return_tensors="pt", truncation=True,
              max_length=max_length).to(device)
    if enc.input_ids.size(1) < 2:
        return float("nan"), float("nan"), float("nan")
    with torch.no_grad():
        logits = model(**enc).logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    targets = enc.input_ids[:, 1:]
    target_lps = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    n_higher = (log_probs[0] > target_lps[0].unsqueeze(-1)).sum(dim=-1).float()
    log_rank = float(np.log(n_higher.cpu().numpy() + 1.0).mean())
    p = log_probs.exp()
    mu = (p * log_probs).sum(dim=-1)
    var = (p * log_probs.pow(2)).sum(dim=-1) - mu.pow(2)
    sigma = var.clamp_min(1e-12).sqrt()
    z = (target_lps - mu) / sigma
    log_ppl = float((-target_lps).mean().item())     # mean NLL = log perplexity
    return log_rank, float(z.mean().item()), log_ppl


def score_binoculars(text, m1, m2, tok, device, max_length=512):
    """Binoculars (Hans et al. 2024): log_ppl_m1 / cross-ppl(m1,m2).
    LOWER => more AI. Faithful to binoculars_correct.py."""
    import torch, torch.nn.functional as F
    enc = tok(text, return_tensors="pt", truncation=True,
              max_length=max_length).to(device)
    ids = enc.input_ids
    if ids.size(1) < 2:
        return float("nan")
    with torch.no_grad():
        l1 = m1(**enc).logits
        l2 = m2(**enc).logits
    lp1 = F.log_softmax(l1[:, :-1, :], dim=-1)
    lp2 = F.log_softmax(l2[:, :-1, :], dim=-1)
    tgt = ids[:, 1:]
    log_ppl = float((-lp1.gather(2, tgt.unsqueeze(-1)).squeeze(-1)).mean().item())
    x_ppl = float((-(lp1.exp() * lp2).sum(-1)).mean().item())
    return log_ppl / max(x_ppl, 1e-12)


def score_roberta(text, model, tok, device, max_length=512):
    import torch, torch.nn.functional as F
    enc = tok(text, return_tensors="pt", truncation=True,
              max_length=max_length).to(device)
    if enc.input_ids.size(1) < 2:
        return float("nan")
    with torch.no_grad():
        logits = model(**enc).logits
    probs = F.softmax(logits, dim=-1)
    # roberta-base-openai-detector: id2label {0:'Fake',1:'Real'} -> P(machine)=probs[0,0]
    return float(probs[0, 0].item())


def reproduce_split(jsonl, humans, seed, test_frac):
    """IDENTICAL split logic to glass_eval_panel.py -> same test docs."""
    ai = []
    for line in open(jsonl):
        try: d = json.loads(line)
        except Exception: continue
        if d.get("error"): continue
        r = (d.get("response") or "").strip()
        if len(r) < 20: continue
        ai.append({"text": r, "gen": str(d["model"]), "title": str(d["title"])})
    ai = pd.DataFrame(ai)
    titles = sorted(ai["title"].unique())
    rng = np.random.default_rng(seed)
    n_te = max(1, int(round(len(titles) * test_frac)))
    test_titles = set(rng.choice(titles, size=n_te, replace=False))
    ai_te = ai[ai["title"].isin(test_titles)]
    H = pd.read_parquet(humans)
    htext = dict(zip(H["title"].astype(str), H["text"].astype(str)))
    hum_te = [htext[t] for t in test_titles if t in htext]
    return list(ai_te["text"]), hum_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--humans", default="results_v2/raid_en_humans.parquet")
    ap.add_argument("--glass-json", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--falcon", default="tiiuae/falcon-7b")
    ap.add_argument("--performer", default="tiiuae/falcon-7b-instruct",
                    help="Binoculars performer (second model)")
    ap.add_argument("--no-binoculars", action="store_true",
                    help="skip Binoculars (avoids loading the second 7B model)")
    ap.add_argument("--roberta", default="openai-community/roberta-base-openai-detector")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ai_txt, hu_txt = reproduce_split(args.jsonl, args.humans, args.seed, args.test_frac)
    texts = ai_txt + hu_txt
    y = np.array([1] * len(ai_txt) + [0] * len(hu_txt))
    print(f"[data] test: {len(ai_txt)} AI vs {len(hu_txt)} human")

    import torch
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                              AutoModelForSequenceClassification)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []

    # GLASS row from glass_eval_panel output
    if args.glass_json:
        g = json.load(open(args.glass_json))
        rows.append(("GLASS (whitened, best-of)", g["auc"], g["tpr_at_1"]))

    # Falcon-7B: Fast-DetectGPT + log-rank + log-perplexity (+ Binoculars)
    print(f"[falcon] loading {args.falcon} ...")
    ftok = AutoTokenizer.from_pretrained(args.falcon)
    m1 = AutoModelForCausalLM.from_pretrained(
        args.falcon, dtype=torch.float16).to(dev).eval()
    m2 = None
    if not args.no_binoculars:
        print(f"[falcon] loading performer {args.performer} ...")
        m2 = AutoModelForCausalLM.from_pretrained(
            args.performer, dtype=torch.float16).to(dev).eval()
    lr, fdg, lppl, binoc = [], [], [], []
    for i, t in enumerate(texts):
        a, b, c = score_falcon_extras(t, m1, ftok, dev, args.max_length)
        lr.append(a); fdg.append(b); lppl.append(c)
        if m2 is not None:
            binoc.append(score_binoculars(t, m1, m2, ftok, dev, args.max_length))
        if (i + 1) % 100 == 0: print(f"  falcon {i+1}/{len(texts)}", end="\r")
    print()
    del m1
    if m2 is not None: del m2
    if dev == "cuda": torch.cuda.empty_cache()
    a, t1, _ = auc_tpr(np.array(fdg), y);   rows.append(("Fast-DetectGPT (Falcon-7B)", a, t1))
    a, t1, _ = auc_tpr(-np.array(lppl), y); rows.append(("log-perplexity (Falcon-7B)", a, t1))
    a, t1, _ = auc_tpr(-np.array(lr), y);   rows.append(("log-rank (Falcon-7B)", a, t1))
    if binoc:
        a, t1, _ = auc_tpr(-np.array(binoc), y)
        rows.append(("Binoculars (Falcon-7B pair)", a, t1))

    # RoBERTa-OpenAI
    print(f"[roberta] loading {args.roberta} ...")
    rtok = AutoTokenizer.from_pretrained(args.roberta)
    rmdl = AutoModelForSequenceClassification.from_pretrained(args.roberta).to(dev).eval()
    pm = [score_roberta(t, rmdl, rtok, dev, args.max_length) for t in texts]
    a, t1, _ = auc_tpr(np.array(pm), y);              rows.append(("RoBERTa-OpenAI (supervised)", a, t1))

    # report
    rows.sort(key=lambda r: -r[1])
    print(f"\n{'Detector':32s} {'AUC':>8} {'TPR@1%FPR':>11}")
    print("-" * 53)
    for name, auc, tpr1 in rows:
        print(f"{name:32s} {auc:8.4f} {tpr1:11.4f}")

    if args.out:
        json.dump([{"detector": n, "auc": a, "tpr_at_1": t} for n, a, t in rows],
                  open(args.out, "w"), indent=2)
        print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
