#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate GLASS and baseline detectors separately for each English RAID domain."""
from __future__ import annotations
import argparse, json, glob, collections
from pathlib import Path
import numpy as np
import pandas as pd

DOMAIN_COLS = ["domain", "source", "domain_name", "raid_domain", "category"]


# ---------- GLASS (refit per panel: strict held-out, matches headline) ----------
def residual_sq(X, mean, B):
    Xc = X - mean
    return np.sum((Xc - Xc @ B @ B.T) ** 2, axis=1)


def fit_subspace(X, k):
    mean = X.mean(0); Xc = X - mean
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return mean, Vt[:min(k, Vt.shape[0])].T


def whiten_fit(X, ridge=1e-3):
    mean = X.mean(0); Xc = X - mean
    cov = (Xc.T @ Xc) / max(1, len(Xc) - 1)
    cov = cov + ridge * (np.trace(cov) / cov.shape[0]) * np.eye(cov.shape[0])
    L = np.linalg.cholesky(cov)
    return mean, np.linalg.solve(L, np.eye(L.shape[0]))


def whiten_apply(X, m, W):
    return (X - m) @ W.T


def encode(texts, model_name, dev, batch=32):
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True,
                      max_length=512, return_tensors="pt").to(dev)
            h = mdl(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1e-9)).cpu().numpy())
    del mdl
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return np.vstack(out).astype(np.float64)


# ---------- Falcon: Fast-DetectGPT + log-rank ----------
def falcon_scores(text, model, tok, dev, maxlen=512):
    import torch, torch.nn.functional as F
    enc = tok(text, return_tensors="pt", truncation=True, max_length=maxlen).to(dev)
    if enc.input_ids.size(1) < 2: return float("nan"), float("nan")
    with torch.no_grad():
        logits = model(**enc).logits[:, :-1, :]
    tgt = enc.input_ids[:, 1:]
    lp = F.log_softmax(logits, -1)
    tlp = lp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
    p = lp.exp(); mu = (p * lp).sum(-1); var = (p * lp.pow(2)).sum(-1) - mu.pow(2)
    fdg = float(((tlp - mu) / var.clamp_min(1e-12).sqrt()).mean().item())
    true_logit = logits.gather(2, tgt.unsqueeze(-1))
    ranks = (logits > true_logit).sum(-1).float() + 1.0
    logrank = float(torch.log(ranks).mean().item())
    return fdg, logrank          # AI-score for log-rank is -logrank


# ---------- RoBERTa-OpenAI ----------
def roberta_pmachine(text, model, tok, dev, maxlen=512):
    import torch
    enc = tok(text, return_tensors="pt", truncation=True, max_length=maxlen).to(dev)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, -1)[0]
    return float(probs[0].item())          # label 0 = machine (reversed)


# ---------- Ghostbuster features ----------
def lm_logprob_feats(texts, model_id, dev, maxlen=256):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(model_id).to(dev).eval()
    feats = []
    with torch.no_grad():
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=maxlen).to(dev)
            if enc.input_ids.size(1) < 2:
                feats.append([0, 0, 0]); continue
            lp = torch.log_softmax(mdl(**enc).logits[:, :-1, :], -1)
            tgt = enc.input_ids[:, 1:]
            tlp = lp.gather(2, tgt.unsqueeze(-1)).squeeze(-1).float().cpu().numpy()
            feats.append([float(tlp.mean()), float(tlp.std()), float(tlp.min())])
    del mdl
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return np.array(feats)


def auc(s, y):
    from sklearn.metrics import roc_auc_score
    s = np.asarray(s, float); y = np.asarray(y); ok = np.isfinite(s)
    if len(set(y[ok].tolist())) < 2: return float("nan")
    return float(roc_auc_score(y[ok], s[ok]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default="results_v2/openrouter_en_s*/per_doc.jsonl")
    ap.add_argument("--humans", default="results_v2/raid_en_humans.parquet")
    ap.add_argument("--bundle", default=".cache/glass_production/v3_english")
    ap.add_argument("--encoder", default="microsoft/deberta-v3-base")
    ap.add_argument("--falcon-model", default="tiiuae/falcon-7b")
    ap.add_argument("--roberta", default="openai-community/roberta-base-openai-detector")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--sh-humans", type=int, default=2000)
    args = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    H = pd.read_parquet(args.humans); H.columns = [c.lower() for c in H.columns]
    dcol = next((c for c in DOMAIN_COLS if c in H.columns), None)
    if dcol is None:
        raise SystemExit(f"no domain column; columns={list(H.columns)}")
    H["title"] = H["title"].astype(str); H["text"] = H["text"].astype(str)
    title2dom = dict(zip(H["title"], H[dcol].astype(str)))
    title2h = {}
    for t, x in zip(H["title"], H["text"]):
        title2h.setdefault(t, []).append(x)
    domains = sorted(set(H[dcol].astype(str)))
    print(f"[domains] {domains}")

    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                              AutoModelForSequenceClassification)
    ftok = AutoTokenizer.from_pretrained(args.falcon_model)
    fal = AutoModelForCausalLM.from_pretrained(args.falcon_model, dtype=torch.float16).to(dev).eval()
    rtok = AutoTokenizer.from_pretrained(args.roberta)
    rob = AutoModelForSequenceClassification.from_pretrained(args.roberta).to(dev).eval()

    dets = ["GLASS", "Fast-DetectGPT", "log-rank", "RoBERTa-OpenAI", "Ghostbuster"]
    res = {d: {dm: [] for dm in domains} for d in dets}

    for si, f in enumerate(sorted(glob.glob(args.panels))):
        rows = []
        for line in open(f, encoding="utf-8"):
            try: d = json.loads(line)
            except Exception: continue
            if d.get("error"): continue
            r = (d.get("response") or "").strip()
            if len(r) >= 20: rows.append((str(d["title"]), str(d["model"]), r))
        titles = sorted({t for t, _, _ in rows})
        rng = np.random.default_rng(si)
        test_t = set(rng.choice(titles, size=max(1, int(round(len(titles) * args.test_frac))),
                                replace=False))
        ai_te = [(title2dom.get(t, "?"), r) for t, m, r in rows if t in test_t and title2dom.get(t) in domains]
        hu_te = [(title2dom.get(t, "?"), x) for t in test_t if t in title2h
                 for x in title2h[t] if title2dom.get(t) in domains]
        ai_tr = [r for t, m, r in rows if t not in test_t]   # for Ghostbuster training
        per_gen_tr = collections.defaultdict(list)           # for GLASS refit
        for t, m, r in rows:
            if t not in test_t: per_gen_tr[m].append(r)
        bg = list(H[~H["title"].isin(test_t)]["text"].sample(
            n=min(args.sh_humans, (~H["title"].isin(test_t)).sum()), random_state=si))

        texts_ai = [r for _, r in ai_te]; texts_hu = [x for _, x in hu_te]
        dom_ai = np.array([dm for dm, _ in ai_te]); dom_hu = np.array([dm for dm, _ in hu_te])
        n_ai = len(texts_ai); n_hu = len(texts_hu)
        allt = texts_ai + texts_hu

        # GLASS: refit per panel on the TRAIN split only (strict held-out, no leakage)
        gens_tr = sorted(per_gen_tr)
        offs = {}; o = 0; tr_ai_flat = []
        for g in gens_tr:
            offs[g] = (o, o + len(per_gen_tr[g])); o += len(per_gen_tr[g]); tr_ai_flat += per_gen_tr[g]
        Xall = encode(bg + tr_ai_flat + allt, args.encoder, dev)
        nb = len(bg); nt = nb + len(tr_ai_flat)
        Xbg, Xtr_ai, Xte = Xall[:nb], Xall[nb:nt], Xall[nt:]
        wm, W = whiten_fit(Xbg)
        Sh = fit_subspace(whiten_apply(Xbg, wm, W), args.K)
        Xtr_aiw = whiten_apply(Xtr_ai, wm, W)
        SAr = {}
        for g in gens_tr:
            a, b = offs[g]; Xgi = Xtr_aiw[a:b]
            if len(Xgi) >= args.K + 1: SAr[g] = fit_subspace(Xgi, args.K)
        Xtew = whiten_apply(Xte, wm, W)
        rh = residual_sq(Xtew, Sh[0], Sh[1])
        sg = rh - np.stack([residual_sq(Xtew, m, b) for m, b in SAr.values()]).min(0)
        # Falcon (fdg, log-rank)
        fdg = np.empty(len(allt)); lr = np.empty(len(allt))
        for i, t in enumerate(allt):
            fdg[i], lr[i] = falcon_scores(t, fal, ftok, dev)
        # RoBERTa
        rb = np.array([roberta_pmachine(t, rob, rtok, dev) for t in allt])
        # Ghostbuster (train LR on ai_tr + bg, gpt2+distilgpt2)
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        tr_texts = ai_tr + bg
        ytr = np.array([1] * len(ai_tr) + [0] * len(bg))
        gtr = np.hstack([lm_logprob_feats(tr_texts, "gpt2", dev),
                         lm_logprob_feats(tr_texts, "distilgpt2", dev)])
        gte = np.hstack([lm_logprob_feats(allt, "gpt2", dev),
                         lm_logprob_feats(allt, "distilgpt2", dev)])
        gtr = np.nan_to_num(gtr); gte = np.nan_to_num(gte)
        sc = StandardScaler().fit(gtr)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(gtr), ytr)
        gb = clf.predict_proba(sc.transform(gte))[:, 1]

        scores = {"GLASS": sg, "Fast-DetectGPT": fdg, "log-rank": -lr,
                  "RoBERTa-OpenAI": rb, "Ghostbuster": gb}

        dom_all = np.concatenate([dom_ai, dom_hu]); y_all = np.array([1] * n_ai + [0] * n_hu)
        for dm in domains:
            m = dom_all == dm
            if (y_all[m] == 1).sum() < 5 or (y_all[m] == 0).sum() < 5: continue
            for d in dets:
                res[d][dm].append(auc(scores[d][m], y_all[m]))
        print(f"  [s{si}] done ({n_ai} AI / {n_hu} hu)")

    def ms(v):
        v = [x for x in v if np.isfinite(x)]
        return (np.mean(v), np.std(v)) if v else (float("nan"), 0.0)

    Path("tables_v2").mkdir(exist_ok=True)
    L = ["% Domain-stratified detection AUC (English panels, v3_english).",
         "\\begin{table*}[t]\\centering\\small",
         "\\caption{Domain-stratified detection AUC on the eight English RAID "
         "domains (\\texttt{v3\\_english}), mean$\\pm$std over five panels.}",
         "\\label{tab:per-domain}",
         "\\begin{tabular}{l" + "c" * len(dets) + "}", "\\toprule",
         "Domain & " + " & ".join(dets) + " \\\\", "\\midrule"]
    hdr = f"{'domain':10s} " + " ".join(f"{d:>15s}" for d in dets)
    print("\n" + hdr)
    for dm in domains:
        cells = [f"{ms(res[d][dm])[0]:.3f}$\\pm${ms(res[d][dm])[1]:.3f}" for d in dets]
        L.append(f"{dm} & " + " & ".join(cells) + " \\\\")
        print(f"{dm:10s} " + " ".join(f"{ms(res[d][dm])[0]:>9.3f}" for d in dets))
    over = [f"{ms([x for dm in domains for x in res[d][dm]])[0]:.3f}" +
            f"$\\pm${ms([x for dm in domains for x in res[d][dm]])[1]:.3f}" for d in dets]
    L += ["\\midrule", "\\textbf{All domains} & " + " & ".join(over) + " \\\\",
          "\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    Path("tables_v2/tab_per_domain_v3.tex").write_text("\n".join(L) + "\n")
    print("\n[out] tables_v2/tab_per_domain_v3.tex")


if __name__ == "__main__":
    main()
