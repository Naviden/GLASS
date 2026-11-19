#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate leave-one-generator-out detection across generators and split seeds."""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np


def whiten_fit(H, ridge=1e-3):
    mu = H.mean(0); Hc = H - mu
    cov = (Hc.T @ Hc) / max(len(Hc), 1) + ridge * np.eye(H.shape[1])
    return mu, np.linalg.inv(np.linalg.cholesky(cov))


def fit_subspace(Xw, K):
    mean = Xw.mean(0)
    _, _, Vt = np.linalg.svd(Xw - mean, full_matrices=False)
    return mean, Vt[:K].T


def residual_sq(Xw, mean, B):
    Xc = Xw - mean
    return np.sum((Xc - Xc @ B @ B.T) ** 2, axis=1)


def auc_tpr(scores, y):
    from sklearn.metrics import roc_auc_score, roc_curve
    s = np.asarray(scores, float); y = np.asarray(y); ok = np.isfinite(s)
    if len(np.unique(y[ok])) < 2: return float("nan"), float("nan")
    a = float(roc_auc_score(y[ok], s[ok]))
    fpr, tpr, _ = roc_curve(y[ok], s[ok])
    return a, float(np.interp(0.01, fpr, tpr))


def encode(texts, model_name, batch=32, device=None):
    import torch
    from transformers import AutoTokenizer, AutoModel
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i+batch], padding=True, truncation=True,
                      max_length=512, return_tensors="pt").to(device)
            h = mdl(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((h*m).sum(1)/m.sum(1).clamp(min=1e-9)).cpu().numpy())
            print(f"  encode {min(i+batch,len(texts))}/{len(texts)}", end="\r")
    print(); del mdl
    if device == "cuda": torch.cuda.empty_cache()
    return np.vstack(out).astype(np.float64)


# ---- per-doc competitor scorers (split-independent) ----
def falcon_fdg_lr(text, model, tok, dev, max_len=512):
    import torch, torch.nn.functional as F
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(dev)
    if enc.input_ids.size(1) < 2: return float("nan"), float("nan")
    with torch.no_grad(): logits = model(**enc).logits
    lp = F.log_softmax(logits[:, :-1, :], -1); tgt = enc.input_ids[:, 1:]
    tlp = lp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
    nh = (lp[0] > tlp[0].unsqueeze(-1)).sum(-1).float()
    lrk = float(np.log(nh.cpu().numpy()+1.0).mean())
    p = lp.exp(); mu = (p*lp).sum(-1); var = (p*lp.pow(2)).sum(-1)-mu.pow(2)
    z = (tlp - mu) / var.clamp_min(1e-12).sqrt()
    return float(z.mean().item()), lrk


def falcon_binoc(text, m1, m2, tok, dev, max_len=512):
    import torch, torch.nn.functional as F
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(dev)
    if enc.input_ids.size(1) < 2: return float("nan")
    with torch.no_grad():
        l1 = m1(**enc).logits; l2 = m2(**enc).logits
    lp1 = F.log_softmax(l1[:, :-1, :], -1); lp2 = F.log_softmax(l2[:, :-1, :], -1)
    tgt = enc.input_ids[:, 1:]
    log_ppl = float((-lp1.gather(2, tgt.unsqueeze(-1)).squeeze(-1)).mean().item())
    x_ppl = float((-(lp1.exp()*lp2).sum(-1)).mean().item())
    return log_ppl / max(x_ppl, 1e-12)


def roberta_p(text, model, tok, dev, max_len=512):
    import torch, torch.nn.functional as F
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(dev)
    if enc.input_ids.size(1) < 2: return float("nan")
    with torch.no_grad(): logits = model(**enc).logits
    return float(F.softmax(logits, -1)[0, 0].item())


def gb_doc_features(lps, lrk):
    f = []
    for arr in (lps, lrk):
        if len(arr) == 0: f += [0.0]*7; continue
        f += [float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr)),
              float(np.quantile(arr,0.1)), float(np.quantile(arr,0.5)), float(np.quantile(arr,0.9))]
    return f


def gb_feats_one(text, model, tok, dev, max_len=512):
    import torch, torch.nn.functional as F
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(dev)
    if enc.input_ids.size(1) < 2: return [0.0]*14
    with torch.no_grad(): logits = model(**enc).logits
    lp = F.log_softmax(logits[:, :-1, :], -1); tgt = enc.input_ids[:, 1:]
    tlp = lp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)[0]
    nh = (lp[0] > tlp.unsqueeze(-1)).sum(-1).float()
    return gb_doc_features(tlp.cpu().numpy(), np.log(nh.cpu().numpy()+1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default="results_v2/openrouter_en_s*/per_doc.jsonl")
    ap.add_argument("--humans", default="results_v2/raid_en_humans.parquet")
    ap.add_argument("--encoder", default="microsoft/deberta-v3-base")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--clean-text", action="store_true",
                    help="drop non-prose junk (short/corrupted/boilerplate) from fitting+eval, uniformly")
    ap.add_argument("--min-words", type=int, default=40,
                    help="with --clean-text: min words/doc (40=mild junk removal, keeps all 8 domains)")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--domain-field", default="domain",
                    help="per-doc field giving the RAID domain (for stratified splits)")
    ap.add_argument("--min-docs", type=int, default=100)
    ap.add_argument("--max-humans", type=int, default=6000)
    ap.add_argument("--gb-train-cap", type=int, default=4000)
    ap.add_argument("--with-falcon", action="store_true")
    ap.add_argument("--with-binoculars", action="store_true")
    ap.add_argument("--with-roberta", action="store_true")
    ap.add_argument("--with-ghostbuster", action="store_true")
    ap.add_argument("--falcon", default="tiiuae/falcon-7b")
    ap.add_argument("--performer", default="tiiuae/falcon-7b-instruct")
    ap.add_argument("--roberta", default="openai-community/roberta-base-openai-detector")
    ap.add_argument("--out-tex", default="tables/tab_logo_detection.tex")
    ap.add_argument("--out-parquet", default="results_v2/logo_detection.parquet")
    ap.add_argument("--out-parquet-domain", default="results_v2/logo_detection_bydomain.parquet")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    keep = (lambda t: True)
    if args.clean_text:
        from text_quality import is_clean
        keep = lambda t: is_clean(t, min_words=args.min_words)

    import torch, pandas as pd
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- gather AI per generator (pooled over panels) ----
    by_gen = {}; title2dom = {}
    dom_field = args.domain_field
    for f in sorted(glob.glob(args.panels)):
        for line in open(f, encoding="utf-8"):
            try: d = json.loads(line)
            except Exception: continue
            if d.get("error"): continue
            r = (d.get("response") or "").strip()
            if len(r) < 20: continue
            if not keep(r): continue
            t = str(d["title"])
            by_gen.setdefault(str(d["model"]), []).append((t, r))
            title2dom.setdefault(t, str(d.get(dom_field, "na")))
    by_gen = {g: v for g, v in by_gen.items() if len(v) >= args.min_docs}
    gens = sorted(by_gen)
    print(f"[gens] {len(gens)} generators")
    import collections as _c
    _dc = _c.Counter(title2dom.values())
    print(f"[domains] {dict(_dc)}")
    if set(_dc) == {"na"}:
        print("[warn] no domain field found -> splits will be random, not domain-stratified")

    # flat AI arrays with generator + title
    ai_text, ai_gen, ai_title = [], [], []
    for g in gens:
        for t, r in by_gen[g]:
            ai_text.append(r); ai_gen.append(g); ai_title.append(t)
    ai_gen = np.array(ai_gen); ai_title = np.array(ai_title)

    # humans
    H = pd.read_parquet(args.humans)
    H["title"] = H["title"].astype(str); H["text"] = H["text"].astype(str)
    if args.clean_text:
        n0 = len(H); H = H[H["text"].map(keep)].reset_index(drop=True)
        print(f"[clean] humans: kept {len(H)}/{n0} (min_words={args.min_words})")
    hum = list(zip(H["title"], H["text"]))
    rng0 = np.random.default_rng(12345)
    if len(hum) > args.max_humans:
        hum = [hum[i] for i in rng0.choice(len(hum), args.max_humans, replace=False)]
    hu_title = np.array([t for t, _ in hum]); hu_text = [x for _, x in hum]
    ai_domain = np.array([title2dom.get(t, "na") for t in ai_title])
    hu_domain = np.array([title2dom.get(t, "na") for t in hu_title])

    # ---- encode once (GLASS) ----
    print(f"[encode] {len(ai_text)} AI + {len(hu_text)} humans")
    Xai = encode(ai_text, args.encoder, device=dev)
    Xhu = encode(hu_text, args.encoder, device=dev)

    # ---- per-doc competitor scores ONCE (split-independent) ----
    comp = {}        # name -> (ai_scores, hu_scores), higher=more AI
    gbf_ai = gbf_hu = None
    if args.with_falcon or args.with_roberta or args.with_ghostbuster:
        from transformers import (AutoTokenizer, AutoModelForCausalLM,
                                   AutoModelForSequenceClassification)
        if args.with_falcon:
            ftok = AutoTokenizer.from_pretrained(args.falcon)
            m1 = AutoModelForCausalLM.from_pretrained(args.falcon, dtype=torch.float16).to(dev).eval()
            m2 = AutoModelForCausalLM.from_pretrained(args.performer, dtype=torch.float16).to(dev).eval() if args.with_binoculars else None
            def fal_all(texts, tag):
                fdg, lrk, bino = [], [], []
                for i, t in enumerate(texts):
                    a, b = falcon_fdg_lr(t, m1, ftok, dev); fdg.append(a); lrk.append(b)
                    if m2 is not None: bino.append(falcon_binoc(t, m1, m2, ftok, dev))
                    if (i+1) % 200 == 0: print(f"  falcon[{tag}] {i+1}/{len(texts)}", end="\r")
                print()
                return np.array(fdg), np.array(lrk), (np.array(bino) if m2 is not None else None)
            fa = fal_all(ai_text, "ai"); fh = fal_all(hu_text, "hu")
            comp["Fast-DetectGPT"] = (fa[0], fh[0])
            comp["log-rank"] = (-fa[1], -fh[1])
            if m2 is not None: comp["Binoculars"] = (-fa[2], -fh[2])
            del m1
            if m2 is not None: del m2
            torch.cuda.empty_cache()
        if args.with_roberta:
            rtok = AutoTokenizer.from_pretrained(args.roberta)
            rmdl = AutoModelForSequenceClassification.from_pretrained(args.roberta).to(dev).eval()
            comp["RoBERTa-OpenAI"] = (np.array([roberta_p(t, rmdl, rtok, dev) for t in ai_text]),
                                      np.array([roberta_p(t, rmdl, rtok, dev) for t in hu_text]))
            del rmdl; torch.cuda.empty_cache()
        if args.with_ghostbuster:
            print("[ghostbuster] gpt2 + distilgpt2 features (once)")
            gmods = []
            for mid in ("gpt2", "distilgpt2"):
                tk = AutoTokenizer.from_pretrained(mid)
                if tk.pad_token is None: tk.pad_token = tk.eos_token
                gmods.append((AutoModelForCausalLM.from_pretrained(mid).to(dev).eval(), tk))
            gbf_ai = np.hstack([np.array([gb_feats_one(t, mm, tk, dev) for t in ai_text]) for mm, tk in gmods])
            gbf_hu = np.hstack([np.array([gb_feats_one(t, mm, tk, dev) for t in hu_text]) for mm, tk in gmods])
            for mm, _ in gmods: del mm
            torch.cuda.empty_cache()

    # ---- folds: per seed, per held-out generator ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    det_names = ["GLASS"] + list(comp.keys()) + (["Ghostbuster"] if args.with_ghostbuster else [])
    acc = {d: {"auc": [], "tpr": []} for d in det_names}
    recs = []
    recs_dom = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        # global DOMAIN-STRATIFIED title split, shared by humans + all generators
        all_titles = sorted(set(ai_title.tolist()) | set(hu_title.tolist()))
        by_dom = {}
        for t in all_titles:
            by_dom.setdefault(title2dom.get(t, "na"), []).append(t)
        test_titles = set()
        for dom, ts in by_dom.items():
            ts = sorted(ts)
            k = max(1, int(round(len(ts) * args.test_frac)))
            test_titles |= set(rng.choice(ts, k, replace=False).tolist())
        tt = list(test_titles)
        hu_te_m = np.isin(hu_title, tt); hu_fit_m = ~hu_te_m
        gen_test_mask = {g: (ai_gen == g) & np.isin(ai_title, tt) for g in gens}
        gen_fit_mask  = {g: (ai_gen == g) & ~np.isin(ai_title, tt) for g in gens}

        # whitener + S_H on human fit
        mu, W = whiten_fit(Xhu[hu_fit_m]); wh = lambda M: (M - mu) @ W.T
        SH = fit_subspace(wh(Xhu[hu_fit_m]), args.K)
        Xai_w = wh(Xai); Xhu_te_w = wh(Xhu[hu_te_m])
        SA = {g: fit_subspace(Xai_w[gen_fit_mask[g]], args.K) for g in gens}
        rH_hu_te = residual_sq(Xhu_te_w, *SH)
        rA_hu_te = {g: residual_sq(Xhu_te_w, *SA[g]) for g in gens}

        for g in gens:
            te = gen_test_mask[g]
            n_ai = int(te.sum()); n_hu = int(hu_te_m.sum())
            yg = np.array([1]*n_ai + [0]*n_hu)
            others = [h for h in gens if h != g]
            # ---- per-detector AI/human score vectors for this fold ----
            rH_ai = residual_sq(Xai_w[te], *SH)
            minA_ai = np.min(np.stack([residual_sq(Xai_w[te], *SA[h]) for h in others], 0), 0)
            minA_hu = np.min(np.stack([rA_hu_te[h] for h in others], 0), 0)
            det_scores = {"GLASS": (rH_ai - minA_ai, rH_hu_te - minA_hu)}
            for name, (sa, sh) in comp.items():
                det_scores[name] = (sa[te], sh[hu_te_m])
            if args.with_ghostbuster:
                tr_mask = np.zeros(len(ai_text), bool)
                for h in others: tr_mask |= gen_fit_mask[h]
                Xtr_ai = gbf_ai[tr_mask]; Xtr_hu = gbf_hu[hu_fit_m]
                cap = args.gb_train_cap
                if len(Xtr_ai) > cap: Xtr_ai = Xtr_ai[rng.choice(len(Xtr_ai), cap, replace=False)]
                if len(Xtr_hu) > cap: Xtr_hu = Xtr_hu[rng.choice(len(Xtr_hu), cap, replace=False)]
                Xtr = np.vstack([Xtr_ai, Xtr_hu]); ytr = np.array([1]*len(Xtr_ai)+[0]*len(Xtr_hu))
                sc = StandardScaler().fit(Xtr)
                clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
                det_scores["Ghostbuster"] = (clf.predict_proba(sc.transform(gbf_ai[te]))[:, 1],
                                             clf.predict_proba(sc.transform(gbf_hu[hu_te_m]))[:, 1])
            # ---- overall (per seed, per held-out generator) ----
            rec = {"seed": seed, "generator": g, "n_ai": n_ai}
            for name, (sa, sh) in det_scores.items():
                a2, t2 = auc_tpr(np.concatenate([sa, sh]), yg)
                acc[name]["auc"].append(a2); acc[name]["tpr"].append(t2)
                rec[f"{name}_auc"] = a2; rec[f"{name}_tpr"] = t2
            recs.append(rec)
            # ---- per-domain ----
            ai_dom = ai_domain[te]; hu_dom = hu_domain[hu_te_m]
            for dmn in sorted(set(ai_dom.tolist())):
                aim = ai_dom == dmn; hum = hu_dom == dmn
                if int(aim.sum()) < 5 or int(hum.sum()) < 5: continue
                ydd = np.array([1]*int(aim.sum()) + [0]*int(hum.sum()))
                for name, (sa, sh) in det_scores.items():
                    ad, td = auc_tpr(np.concatenate([sa[aim], sh[hum]]), ydd)
                    recs_dom.append({"seed": seed, "generator": g, "domain": dmn,
                                     "detector": name, "auc": ad, "tpr": td})
        print(f"[seed {seed}] done ({len(gens)} held-out generators)")

    pd.DataFrame(recs).to_parquet(args.out_parquet)
    pd.DataFrame(recs_dom).to_parquet(args.out_parquet_domain)
    print(f"[out] {args.out_parquet_domain}  ({len(recs_dom)} per-domain rows)")
    ms = lambda v: (float(np.nanmean(v)), float(np.nanstd(v)))
    order = ["GLASS", "Fast-DetectGPT", "Binoculars", "Ghostbuster", "log-rank", "RoBERTa-OpenAI"]
    order = [d for d in order if d in acc]

    L = ["% Leave-one-generator-out detection, multi-seed (build_logo_detection_v3.py).",
         f"% {len(gens)} held-out generators x {len(seeds)} seeds; GLASS library excludes the held-out generator.",
         "\\begin{table}[t]\\centering\\small",
         "\\caption{\\textbf{Unseen-generator detection, leave-one-generator-out.} "
         "Each of the $%d$ frontier generators is held out of every detector in turn "
         "(no chosen block): \\glass's library and Ghostbuster's classifier are fit "
         "on the other $%d$ generators, and the zero-shot / GPT-2 baselines train on "
         "no frontier model. AUC and TPR@$1\\%%$FPR, mean$\\pm$std over the $%d$ "
         "held-out generators $\\times$ $%d$ seeds, on identical held-out documents. "
         "\\textbf{Bold} = best.}" % (len(gens), len(gens)-1, len(gens), len(seeds)),
         "\\label{tab:logo-detection}",
         "\\begin{tabular}{lcc}", "\\toprule",
         "Detector & AUC & TPR @ $1\\%$ FPR \\\\", "\\midrule"]
    best = max(order, key=lambda d: np.nanmean(acc[d]["auc"]))
    for d in order:
        (am, asd), (tm, tsd) = ms(acc[d]["auc"]), ms(acc[d]["tpr"])
        nm = f"\\textbf{{{d}}} (leave-one-generator-out)" if d == "GLASS" else d
        if d == best and d != "GLASS": nm = f"\\textbf{{{d}}}"
        a_s = f"\\textbf{{{am:.3f}}}" if d == best else f"{am:.3f}"
        L.append(f"{nm} & {a_s}$\\pm${asd:.3f} & {tm:.3f}$\\pm${tsd:.3f} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    Path(args.out_tex).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_tex).write_text("\n".join(L) + "\n")
    print(f"\n[out] {args.out_tex}\n[out] {args.out_parquet}")
    for d in order:
        (am, asd), (tm, tsd) = ms(acc[d]["auc"]), ms(acc[d]["tpr"])
        print(f"  {d:18s} AUC {am:.4f}±{asd:.4f}  TPR@1 {tm:.4f}±{tsd:.4f}")


if __name__ == "__main__":
    main()
