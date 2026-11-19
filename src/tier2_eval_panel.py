#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate DetectGPT, DNA-GPT and Ghostbuster on the GLASS panel test split."""
from __future__ import annotations
import argparse, json, re
import numpy as np
import pandas as pd


def auc_tpr(scores, y):
    from sklearn.metrics import roc_auc_score, roc_curve
    s = np.asarray(scores); ok = np.isfinite(s)
    s, yy = s[ok], np.asarray(y)[ok]
    auc = float(roc_auc_score(yy, s))
    fpr, tpr, _ = roc_curve(yy, s)
    return auc, float(np.interp(0.01, fpr, tpr))


# ---------- split (identical to the other panel scripts) --------------------
def load_panel(jsonl):
    rows = []
    for line in open(jsonl):
        try: d = json.loads(line)
        except Exception: continue
        if d.get("error"): continue
        r = (d.get("response") or "").strip()
        if len(r) < 20: continue
        rows.append({"text": r, "gen": str(d["model"]), "title": str(d["title"])})
    return pd.DataFrame(rows)


def split(jsonl, humans, seed, test_frac, n_bg=2000):
    ai = load_panel(jsonl)
    titles = sorted(ai["title"].unique())
    rng = np.random.default_rng(seed)
    n_te = max(1, int(round(len(titles) * test_frac)))
    test_t = set(rng.choice(titles, size=n_te, replace=False))
    ai_tr = ai[~ai["title"].isin(test_t)]
    ai_te = ai[ai["title"].isin(test_t)]
    H = pd.read_parquet(humans); htext = dict(zip(H["title"].astype(str), H["text"].astype(str)))
    hu_te = [htext[t] for t in test_t if t in htext]
    bgdf = H[~H["title"].astype(str).isin(set(titles))]
    bg = bgdf.sample(n=min(n_bg, len(bgdf)), random_state=seed)["text"].astype(str).tolist()
    return (list(ai_tr["text"]), list(ai_te["text"]), hu_te, bg)


# ---------- DetectGPT (detectgpt.py) ----------------------------------------
def make_t5_masked_input(text, mask_pct, span_len, rng):
    tokens = text.split(); n = len(tokens)
    if n < span_len + 2:
        return text
    n_spans = max(1, int(np.ceil(mask_pct * n / span_len)))
    cand = list(range(n - span_len + 1)); rng.shuffle(cand)
    chosen = []; occ = np.zeros(n, dtype=bool)
    for s in cand:
        if any(occ[s:s + span_len]): continue
        occ[s:s + span_len] = True; chosen.append(s)
        if len(chosen) >= n_spans: break
    chosen.sort(); out = []; last = 0
    for k, s in enumerate(chosen):
        out.extend(tokens[last:s]); out.append(f"<extra_id_{k}>"); last = s + span_len
    out.extend(tokens[last:])
    return " ".join(out)


def t5_fill_to_text(filled, masked_input):
    pat = re.compile(r"<extra_id_(\d+)>"); parts = pat.split(filled); fills = {}
    if parts and parts[0] == "":
        for i in range(1, len(parts) - 1, 2):
            try: fills[int(parts[i])] = parts[i + 1].strip()
            except Exception: pass
    return pat.sub(lambda m: fills.get(int(m.group(1)), ""), masked_input)


def doc_logp(text, model, tok, dev, max_len):
    import torch, torch.nn.functional as F
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(dev)
    if enc.input_ids.size(1) < 2: return float("nan")
    with torch.no_grad():
        logits = model(**enc).logits
    lp = F.log_softmax(logits[:, :-1, :], -1)
    tgt = enc.input_ids[:, 1:]
    nll = -lp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)[0]
    return float(-nll.mean().item())


def detectgpt_scores(texts, K, mask_pct, span_len, t5, t5tok, fal, ftok, dev, max_len, seed):
    import torch
    rng = np.random.default_rng(seed); out = []
    for i, text in enumerate(texts):
        lp_x = doc_logp(text, fal, ftok, dev, max_len)
        lps = []
        for _ in range(K):
            masked = make_t5_masked_input(text, mask_pct, span_len, rng)
            if "<extra_id_0>" not in masked:
                lps.append(doc_logp(text, fal, ftok, dev, max_len)); continue
            enc = t5tok(masked, return_tensors="pt", truncation=True, max_length=max_len).to(dev)
            with torch.no_grad():
                g = t5.generate(**enc, max_new_tokens=200, do_sample=True,
                                top_p=0.95, temperature=1.0, num_return_sequences=1)
            filled = t5tok.decode(g[0], skip_special_tokens=False)
            lps.append(doc_logp(t5_fill_to_text(filled, masked), fal, ftok, dev, max_len))
        lps = [v for v in lps if np.isfinite(v)]
        out.append((lp_x - np.mean(lps)) / (np.std(lps) + 1e-6) if lps and np.isfinite(lp_x) else float("nan"))
        if (i + 1) % 50 == 0: print(f"  detectgpt {i+1}/{len(texts)}", end="\r")
    print(); return out


# ---------- DNA-GPT (dna_gpt.py) --------------------------------------------
def ngram_set(tokens, n):
    if len(tokens) < n: return set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def dna_scores(texts, K, n_gram, model, tok, dev, max_len):
    import torch
    pad = tok.pad_token_id or tok.eos_token_id; out = []
    for i, text in enumerate(texts):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(dev)
        ids = enc.input_ids[0]; n_tok = int(ids.shape[0])
        if n_tok < 2 * n_gram + 4: out.append(float("nan")); continue
        mid = n_tok // 2; pre = ids[:mid].unsqueeze(0); suf = ids[mid:]
        sgr = ngram_set(suf.cpu().numpy().tolist(), n_gram)
        if not sgr: out.append(float("nan")); continue
        with torch.no_grad():
            g = model.generate(pre, max_new_tokens=int(suf.shape[0]) + 4, do_sample=True,
                               top_p=0.95, temperature=0.7, num_return_sequences=K, pad_token_id=pad)
        nt = g[:, pre.shape[1]:]; ov = []
        for k in range(nt.shape[0]):
            cg = ngram_set(nt[k].cpu().numpy().tolist(), n_gram)
            if cg: ov.append(len(sgr & cg) / len(sgr))
        out.append(float(np.mean(ov)) if ov else float("nan"))
        if (i + 1) % 50 == 0: print(f"  dna-gpt {i+1}/{len(texts)}", end="\r")
    print(); return out


# ---------- Ghostbuster (ghostbuster_light.py) ------------------------------
def doc_features(lps, lrk):
    f = []
    for arr in (lps, lrk):
        if len(arr) == 0: f += [0.0] * 7; continue
        f += [float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr)),
              float(np.quantile(arr, 0.1)), float(np.quantile(arr, 0.5)), float(np.quantile(arr, 0.9))]
    return f


def gb_feats_one(text, model, tok, dev, max_len):
    import torch, torch.nn.functional as F
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(dev)
    if enc.input_ids.size(1) < 2: return [0.0] * 14
    with torch.no_grad():
        logits = model(**enc).logits
    lp = F.log_softmax(logits[:, :-1, :], -1); tgt = enc.input_ids[:, 1:]
    tlp = lp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)[0]
    nh = (lp[0] > tlp.unsqueeze(-1)).sum(-1).float()
    return doc_features(tlp.cpu().numpy(), np.log(nh.cpu().numpy() + 1.0))


def gb_matrix(texts, models, dev, max_len):
    import numpy as np
    cols = []
    for mdl, tok in models:
        cols.append(np.array([gb_feats_one(t, mdl, tok, dev, max_len) for t in texts]))
    return np.hstack(cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--humans", default="results_v2/raid_en_humans.parquet")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--falcon", default="tiiuae/falcon-7b")
    ap.add_argument("--t5", default="t5-large")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--skip", default="", help="comma list: detectgpt,dna,ghostbuster")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    ai_tr, ai_te, hu_te, bg = split(args.jsonl, args.humans, args.seed, args.test_frac)
    test = ai_te + hu_te
    y = np.array([1] * len(ai_te) + [0] * len(hu_te))
    print(f"[data] test {len(ai_te)} AI / {len(hu_te)} human; train {len(ai_tr)} AI / {len(bg)} human")

    import torch
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                              AutoModelForSeq2SeqLM)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []

    if {"detectgpt", "dna"} - skip:
        print(f"[load] {args.falcon}")
        ftok = AutoTokenizer.from_pretrained(args.falcon)
        if ftok.pad_token is None: ftok.pad_token = ftok.eos_token
        fal = AutoModelForCausalLM.from_pretrained(args.falcon, dtype=torch.float16).to(dev).eval()
        if "detectgpt" not in skip:
            print(f"[load] {args.t5}")
            t5tok = AutoTokenizer.from_pretrained(args.t5)
            t5 = AutoModelForSeq2SeqLM.from_pretrained(args.t5, dtype=torch.float16).to(dev).eval()
            sc = detectgpt_scores(test, 10, 0.15, 2, t5, t5tok, fal, ftok, dev, args.max_length, args.seed)
            rows.append(("DetectGPT (T5+Falcon-7B)", *auc_tpr(sc, y))); del t5
            if dev == "cuda": torch.cuda.empty_cache()
        if "dna" not in skip:
            sc = dna_scores(test, 10, 4, fal, ftok, dev, args.max_length)
            rows.append(("DNA-GPT (Falcon-7B)", *auc_tpr(sc, y)))
        del fal
        if dev == "cuda": torch.cuda.empty_cache()

    if "ghostbuster" not in skip:
        print("[load] gpt2 + distilgpt2 (Ghostbuster)")
        models = []
        for mid in ("gpt2", "distilgpt2"):
            tk = AutoTokenizer.from_pretrained(mid)
            mm = AutoModelForCausalLM.from_pretrained(mid).to(dev).eval()
            models.append((mm, tk))
        Xtr = gb_matrix(ai_tr + bg, models, dev, args.max_length)
        ytr = np.array([1] * len(ai_tr) + [0] * len(bg))
        Xte = gb_matrix(test, models, dev, args.max_length)
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(Xtr, ytr)
        pm = clf.predict_proba(Xte)[:, 1]
        rows.append(("Ghostbuster (in-corpus)", *auc_tpr(pm, y)))

    print(f"\n{'Detector':28s} {'AUC':>8} {'TPR@1%':>8}")
    for n, a, t in rows: print(f"{n:28s} {a:8.4f} {t:8.4f}")
    if args.out:
        json.dump([{"detector": n, "auc": a, "tpr_at_1": t} for n, a, t in rows],
                  open(args.out, "w"), indent=2)
        print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
