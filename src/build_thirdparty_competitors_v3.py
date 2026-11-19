#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare frozen-bundle GLASS and baseline detectors on external corpora."""
from __future__ import annotations
import argparse, json, glob, tempfile, zipfile
from pathlib import Path
import numpy as np


# ----------------------------- GLASS -----------------------------------------
def residual_sq(X, mean, B):
    Xc = X - mean
    return np.sum((Xc - Xc @ B @ B.T) ** 2, axis=1)


def load_bundle(base, K):
    base = Path(base)
    sh = np.load(base / "S_H_universal.npz")
    SH = (sh["mean"], sh["basis"][:, :K])
    SA = {}
    for p in sorted((base / "subspaces").glob("S_A_*.npz")):
        z = np.load(p); SA[p.stem[4:]] = (z["mean"], z["basis"][:, :K])
    wh = np.load(base / "whitener.npz"); WH = (wh["mean"], wh["W"])
    return SH, SA, WH


def glass_score(X, SH, SA, WH):
    Xw = (X - WH[0]) @ WH[1].T             # whiten before residual (matches headline)
    rh = residual_sq(Xw, SH[0], SH[1])
    allr = np.stack([residual_sq(Xw, m, b) for (m, b) in SA.values()], 0)
    return rh - allr.min(0)                # higher = AI


# ----------------------------- encoders / scorers ----------------------------
def encode(texts, model_name, batch=32, device=None):
    import torch
    from transformers import AutoTokenizer, AutoModel
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True,
                      max_length=512, return_tensors="pt").to(device)
            h = mdl(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1e-9)).cpu().numpy())
            print(f"  encode {min(i + batch, len(texts))}/{len(texts)}", end="\r")
    print(); del mdl
    if device == "cuda": torch.cuda.empty_cache()
    return np.vstack(out).astype(np.float64)


def falcon_fdg_lr(text, model, tok, dev, max_len=512):
    import torch, torch.nn.functional as F
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(dev)
    if enc.input_ids.size(1) < 2: return float("nan"), float("nan")
    with torch.no_grad(): logits = model(**enc).logits
    lp = F.log_softmax(logits[:, :-1, :], -1); tgt = enc.input_ids[:, 1:]
    tlp = lp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
    nh = (lp[0] > tlp[0].unsqueeze(-1)).sum(-1).float()
    lrk = float(np.log(nh.cpu().numpy() + 1.0).mean())
    p = lp.exp(); mu = (p * lp).sum(-1); var = (p * lp.pow(2)).sum(-1) - mu.pow(2)
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
    x_ppl = float((-(lp1.exp() * lp2).sum(-1)).mean().item())
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
        if len(arr) == 0: f += [0.0] * 7; continue
        f += [float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr)),
              float(np.quantile(arr, 0.1)), float(np.quantile(arr, 0.5)), float(np.quantile(arr, 0.9))]
    return f


def gb_feats_one(text, model, tok, dev, max_len=512):
    import torch, torch.nn.functional as F
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(dev)
    if enc.input_ids.size(1) < 2: return [0.0] * 14
    with torch.no_grad(): logits = model(**enc).logits
    lp = F.log_softmax(logits[:, :-1, :], -1); tgt = enc.input_ids[:, 1:]
    tlp = lp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)[0]
    nh = (lp[0] > tlp.unsqueeze(-1)).sum(-1).float()
    return gb_doc_features(tlp.cpu().numpy(), np.log(nh.cpu().numpy() + 1.0))


def auc_of(scores, y):
    from sklearn.metrics import roc_auc_score
    s = np.asarray(scores, float); y = np.asarray(y); ok = np.isfinite(s)
    if len(np.unique(y[ok])) < 2: return float("nan")
    return float(roc_auc_score(y[ok], s[ok]))


# ----------------------------- corpus loaders --------------------------------
def load_mage(n):
    from datasets import load_dataset
    ds = load_dataset("yaful/MAGE", split="train", streaming=True)
    hum, ai = [], []
    for r in ds:
        t = (r.get("text") or "").strip()
        if len(t) < 50: continue
        if r["label"] == 1 and len(hum) < n: hum.append(t)
        elif r["label"] == 0 and len(ai) < n: ai.append(t)
        if len(hum) >= n and len(ai) >= n: break
    return hum, ai


def load_nyt(n, dataset="gsingh1-py/train"):
    """RECENT-generator corpus: NYT human articles vs. synthetic versions from
    GPT-4o / Qwen-2-72B / Gemma-2-9b / Mistral-7B / Llama-8B / Yi-Large (wide
    format). Cycles generators so the AI side is a mix."""
    from datasets import load_dataset
    ds = load_dataset(dataset, split="train", streaming=True)
    META = {"prompt", "human_story", "id", "url", "title", "index", "abstract", "story"}
    hum, ai = [], []; gens = None
    for r in ds:
        if gens is None:
            gens = [c for c in r.keys() if c.lower() not in META]
        h = str(r.get("Human_story") or r.get("human_story") or "").strip()
        if len(h) >= 50 and len(hum) < n: hum.append(h)
        if gens:
            for j in range(len(gens)):
                c = gens[(len(ai) + j) % len(gens)]
                a = str(r.get(c) or "").strip()
                if len(a) >= 50:
                    if len(ai) < n: ai.append(a)
                    break
        if len(hum) >= n and len(ai) >= n: break
    return hum, ai


def load_hf_generic(n, dataset, text_col, label_col, ai_vals, human_vals, config=None, split="train"):
    """Generic long-format loader: one row = one text + a label. Label matching is
    case-insensitive against comma-separated ai_vals / human_vals. Prints available
    columns if nothing matches, so the spec can be corrected."""
    from datasets import load_dataset
    ds = load_dataset(dataset, config, split=split, streaming=True) if config \
        else load_dataset(dataset, split=split, streaming=True)
    ai_set = {v.strip().lower() for v in ai_vals}; hu_set = {v.strip().lower() for v in human_vals}
    hum, ai = [], []; cols = None
    for r in ds:
        if cols is None: cols = list(r.keys())
        t = str(r.get(text_col) or "").strip()
        if len(t) < 50: continue
        lab = str(r.get(label_col)).strip().lower()
        if lab in ai_set and len(ai) < n: ai.append(t)
        elif lab in hu_set and len(hum) < n: hum.append(t)
        if len(hum) >= n and len(ai) >= n: break
    if (len(ai) < 50 or len(hum) < 50) and cols:
        print(f"  [hf-corpus] only {len(ai)} ai / {len(hum)} human; available columns = {cols}")
    return hum, ai


def load_hc3(n):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("Hello-SimpleAI/HC3", "all.jsonl", repo_type="dataset")
    hum, ai = [], []
    for line in open(p):
        try: d = json.loads(line)
        except Exception: continue
        for h in (d.get("human_answers") or []):
            if isinstance(h, str) and h.strip() and len(hum) < n: hum.append(h.strip())
        for a in (d.get("chatgpt_answers") or []):
            if isinstance(a, str) and a.strip() and len(ai) < n: ai.append(a.strip())
        if len(hum) >= n and len(ai) >= n: break
    return hum, ai


def load_turingbench(n):
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
        p = hf_hub_download("turingbench/TuringBench", "TuringBench.zip", repo_type="dataset")
        d = tempfile.mkdtemp()
        with zipfile.ZipFile(p) as z: z.extractall(d)
        hum, ai = [], []
        for f in glob.glob(d + "/**/*", recursive=True):
            if not f.lower().endswith((".csv", ".tsv")): continue
            try: df = pd.read_csv(f, sep=None, engine="python")
            except Exception: continue
            cols = {c.lower(): c for c in df.columns}
            tc = cols.get("generation") or cols.get("text") or cols.get("content")
            lc = cols.get("label") or cols.get("class")
            if not tc or not lc: continue
            for t, lab in zip(df[tc].astype(str), df[lc].astype(str)):
                if len(t.strip()) < 50: continue
                if "human" in lab.lower():
                    if len(hum) < n: hum.append(t.strip())
                elif len(ai) < n:
                    ai.append(t.strip())
            if len(hum) >= n and len(ai) >= n: break
        return hum, ai
    except Exception as e:
        print(f"  [turingbench] skipped: {str(e)[:120]}")
        return [], []


# ----------------------------- Ghostbuster training pool ---------------------
def load_raid_library(panels_glob, humans_parquet, cap, rng):
    """RAID frontier AI responses + RAID humans -> Ghostbuster LR training pool."""
    import pandas as pd
    ai = []
    for f in sorted(glob.glob(panels_glob)):
        for line in open(f, encoding="utf-8"):
            try: d = json.loads(line)
            except Exception: continue
            if d.get("error"): continue
            r = (d.get("response") or "").strip()
            if len(r) >= 20: ai.append(r)
    H = pd.read_parquet(humans_parquet)
    hum = [str(x) for x in H["text"].tolist() if isinstance(x, str) or x is not None]
    hum = [h for h in hum if len(h.strip()) >= 20]
    if len(ai) > cap: ai = [ai[i] for i in rng.choice(len(ai), cap, replace=False)]
    if len(hum) > cap: hum = [hum[i] for i in rng.choice(len(hum), cap, replace=False)]
    return ai, hum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=".cache/glass_production/v3_english")
    ap.add_argument("--encoder", default="microsoft/deberta-v3-base")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--n", type=int, default=1000, help="per-class cap per corpus")
    ap.add_argument("--corpora", default="HC3,TuringBench,MAGE",
                    help="which built-in corpora to run (from HC3,TuringBench,MAGE,NYT)")
    ap.add_argument("--clean-text", action="store_true",
                    help="drop non-prose junk (short quips, corrupted, boilerplate) from all corpora, uniformly")
    ap.add_argument("--min-words", type=int, default=100,
                    help="with --clean-text: minimum words/document (uniform, all detectors+corpora)")
    ap.add_argument("--oversample", type=int, default=8,
                    help="with --clean-text: load n*this raw docs so enough survive the stricter filter")
    ap.add_argument("--hf-corpus", action="append", default=[],
                    help='add any HF dataset: "Name:dataset_id:text_col:label_col:ai_vals:human_vals[:config[:split]]" '
                         '(ai_vals/human_vals comma-separated). Repeatable.')
    ap.add_argument("--with-falcon", action="store_true")
    ap.add_argument("--with-binoculars", action="store_true")
    ap.add_argument("--with-roberta", action="store_true")
    ap.add_argument("--with-radar", action="store_true",
                    help="RADAR paraphrase-robust detector (Hu et al., NeurIPS 2023)")
    ap.add_argument("--radar-model", default="TrustSafeAI/RADAR-Vicuna-7B")
    ap.add_argument("--with-ghostbuster", action="store_true")
    ap.add_argument("--falcon", default="tiiuae/falcon-7b")
    ap.add_argument("--performer", default="tiiuae/falcon-7b-instruct")
    ap.add_argument("--roberta", default="openai-community/roberta-base-openai-detector")
    ap.add_argument("--panels", default="results_v2/openrouter_en_s*/per_doc.jsonl")
    ap.add_argument("--humans", default="results_v2/raid_en_humans.parquet")
    ap.add_argument("--gb-train-cap", type=int, default=4000)
    ap.add_argument("--out-tex", default="tables/tab_thirdparty_all.tex")
    ap.add_argument("--out-parquet", default="results_v2/thirdparty_all.parquet")
    ap.add_argument("--hf-detector", action="append", default=[],
                    help='"Name:hf/model_id" released classifier, scored on the SAME docs (repeatable)')
    ap.add_argument("--external", action="append", default=[],
                    help='"Name:file.jsonl" per-doc scores from a detector\'s own repo (repeatable)')
    ap.add_argument("--dump-originals", default=None, help="export the test docs for external detectors, then exit")
    args = ap.parse_args()

    import torch, pandas as pd
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)

    # ---- load corpora once: name -> (ai_texts, hum_texts) ----
    REGISTRY = {"HC3": load_hc3, "TuringBench": load_turingbench, "MAGE": load_mage, "NYT": load_nyt}
    selected = [c.strip() for c in args.corpora.split(",") if c.strip()]
    if args.clean_text:
        from text_quality import filter_pair
    corpora = {}
    for name in selected:
        loader = REGISTRY.get(name)
        if loader is None:
            print(f"  [{name}] unknown corpus, skip"); continue
        print(f"[{name}] loading ...")
        hum, ai = loader(args.n * args.oversample if args.clean_text else args.n)   # oversample to survive filtering
        if args.clean_text:
            ai, hum, da, dh = filter_pair(ai, hum, args.n, min_words=args.min_words)
            print(f"  [{name}] cleaned (min_words={args.min_words}): dropped {da} ai / {dh} human junk")
        else:
            ai, hum = ai[:args.n], hum[:args.n]
        if len(hum) < 50 or len(ai) < 50:
            print(f"  [{name}] too few docs ({len(hum)} human / {len(ai)} ai); skip"); continue
        corpora[name] = (ai, hum)
        print(f"  [{name}] {len(ai)} ai / {len(hum)} human")

    # ---- extra corpora from --hf-corpus specs ----
    for spec in args.hf_corpus:
        parts = spec.split(":")
        if len(parts) < 6:
            print(f"  [hf-corpus] bad spec (need Name:id:text:label:ai:human[:config[:split]]): {spec}"); continue
        nm, did, tcol, lcol, aiv, huv = parts[:6]
        cfg = parts[6] if len(parts) > 6 and parts[6] else None
        spl = parts[7] if len(parts) > 7 and parts[7] else "train"
        print(f"[{nm}] loading {did} ...")
        try:
            hum, ai = load_hf_generic(args.n * args.oversample if args.clean_text else args.n,
                                      did, tcol, lcol, aiv.split(","), huv.split(","), cfg, spl)
        except Exception as e:
            print(f"  [{nm}] load failed: {str(e)[:160]}"); continue
        if args.clean_text:
            ai, hum, da, dh = filter_pair(ai, hum, args.n, min_words=args.min_words)
            print(f"  [{nm}] cleaned (min_words={args.min_words}): dropped {da} ai / {dh} human junk")
        else:
            ai, hum = ai[:args.n], hum[:args.n]
        if len(hum) < 50 or len(ai) < 50:
            print(f"  [{nm}] too few docs ({len(hum)} human / {len(ai)} ai); skip"); continue
        corpora[nm] = (ai, hum)
        print(f"  [{nm}] {len(ai)} ai / {len(hum)} human")

    if not corpora:
        print("[fatal] no corpora loaded"); return

    # scores[name][detector] = AUC ; counts[name] = (n_ai, n_hum)
    scores = {name: {} for name in corpora}
    counts = {name: (len(ai), len(hum)) for name, (ai, hum) in corpora.items()}
    y_of = {name: np.array([1] * len(ai) + [0] * len(hum)) for name, (ai, hum) in corpora.items()}
    texts_of = {name: (ai + hum) for name, (ai, hum) in corpora.items()}

    # raw[name][detector] = per-document score, ORIENTED so that higher = AI.
    # Without this the AUCs in the table cannot be given confidence intervals, cannot be
    # compared against a word-count baseline, and cannot be re-thresholded by length --
    # Keep per-document scores for these analyses.
    raw = {name: {} for name in corpora}

    def put(name, det, s, flip=False):
        """Record a detector's per-document scores AND its AUC from the same array."""
        s = np.asarray(s, dtype=float)
        if flip:
            s = -s
        raw[name][det] = s
        scores[name][det] = auc_of(s, y_of[name])
        return scores[name][det]

    # ---- export identical test docs for external detectors, then exit ----
    if args.dump_originals:
        import json as _json
        with open(args.dump_originals, "w", encoding="utf-8") as f:
            for name, (ai, hum) in corpora.items():
                for i, t in enumerate(ai):  f.write(_json.dumps({"corpus": name, "idx": i, "text": t, "label": "ai"}) + "\n")
                for j, t in enumerate(hum): f.write(_json.dumps({"corpus": name, "idx": len(ai)+j, "text": t, "label": "human"}) + "\n")
        print(f"[dump] {args.dump_originals}: score every line (higher=AI), keep corpus+idx -> feed via --external"); return

    # ---- GLASS (frozen v3_english) ----
    SH, SA, WH = load_bundle(args.bundle, args.K)
    print(f"[bundle] S_H + {len(SA)} per-generator subspaces (K={args.K}, whitened)")
    for name in corpora:
        X = encode(texts_of[name], args.encoder, device=dev)
        put(name, "GLASS", glass_score(X, SH, SA, WH))
        print(f"  [GLASS|{name}] AUC = {scores[name]['GLASS']:.4f}")

    # ---- Falcon family: Fast-DetectGPT, log-rank, (Binoculars) ----
    if args.with_falcon:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        ftok = AutoTokenizer.from_pretrained(args.falcon)
        m1 = AutoModelForCausalLM.from_pretrained(args.falcon, dtype=torch.float16).to(dev).eval()
        m2 = (AutoModelForCausalLM.from_pretrained(args.performer, dtype=torch.float16).to(dev).eval()
              if args.with_binoculars else None)
        for name in corpora:
            T = texts_of[name]; fdg, lrk, bino = [], [], []
            for i, t in enumerate(T):
                a, b = falcon_fdg_lr(t, m1, ftok, dev); fdg.append(a); lrk.append(b)
                if m2 is not None: bino.append(falcon_binoc(t, m1, m2, ftok, dev))
                if (i + 1) % 200 == 0: print(f"  falcon[{name}] {i+1}/{len(T)}", end="\r")
            print()
            put(name, "Fast-DetectGPT", fdg)                                   # higher = AI
            put(name, "log-rank", lrk, flip=True)                              # flip -> higher = AI
            if m2 is not None:
                put(name, "Binoculars", bino, flip=True)                       # flip -> higher = AI
            print(f"  [Falcon|{name}] FDGPT={scores[name]['Fast-DetectGPT']:.4f} "
                  f"log-rank={scores[name]['log-rank']:.4f}"
                  + (f" Binoc={scores[name]['Binoculars']:.4f}" if m2 is not None else ""))
        del m1
        if m2 is not None: del m2
        torch.cuda.empty_cache()

    # ---- RoBERTa-OpenAI ----
    if args.with_roberta:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        rtok = AutoTokenizer.from_pretrained(args.roberta)
        rmdl = AutoModelForSequenceClassification.from_pretrained(args.roberta).to(dev).eval()
        for name in corpora:
            put(name, "RoBERTa-OpenAI", [roberta_p(t, rmdl, rtok, dev) for t in texts_of[name]])
            print(f"  [RoBERTa|{name}] AUC = {scores[name]['RoBERTa-OpenAI']:.4f}")
        del rmdl; torch.cuda.empty_cache()

    # ---- RADAR (paraphrase-robust; Hu et al., NeurIPS 2023) ----
    if args.with_radar:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        dtok = AutoTokenizer.from_pretrained(args.radar_model)
        dmdl = AutoModelForSequenceClassification.from_pretrained(args.radar_model).to(dev).eval()
        for name in corpora:
            put(name, "RADAR", [roberta_p(t, dmdl, dtok, dev) for t in texts_of[name]])
            print(f"  [RADAR|{name}] AUC = {scores[name]['RADAR']:.4f}")
        del dmdl; torch.cuda.empty_cache()

    # ---- extra released classifiers (e.g. OpenAI-large), SAME test docs, auto-oriented ----
    if args.hf_detector:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        for spec in args.hf_detector:
            dname, mid = spec.split(":", 1)
            tok = AutoTokenizer.from_pretrained(mid)
            mdl = AutoModelForSequenceClassification.from_pretrained(mid).to(dev).eval()
            s_of = {name: np.array([roberta_p(t, mdl, tok, dev) for t in texts_of[name]])
                    for name in corpora}
            aucs = {name: auc_of(s_of[name], y_of[name]) for name in corpora}
            # Orientation is decided ONCE, globally, from the mean AUC -- and the flip is
            # applied to the per-document scores, not just to the AUC, so the dump and the
            # table can never disagree.
            flip = np.nanmean(list(aucs.values())) < 0.5
            for name in corpora:
                put(name, dname, s_of[name], flip=flip)
                print(f"  [{dname}|{name}] AUC = {scores[name][dname]:.4f}"
                      + ("  (orientation flipped)" if flip else ""))
            del mdl; torch.cuda.empty_cache()

    # ---- external per-doc scores (detector run via its OWN repo), SAME test docs ----
    if args.external:
        import json as _json
        for spec in args.external:
            dname, fpath = spec.split(":", 1)
            by = {name: {} for name in corpora}
            for line in open(fpath, encoding="utf-8"):
                d = _json.loads(line)
                if d["corpus"] in by: by[d["corpus"]][int(d["idx"])] = float(d["score"])
            s_of, aucs = {}, {}
            for name in corpora:
                nt = counts[name][0] + counts[name][1]
                s_of[name] = np.array([by[name].get(i, np.nan) for i in range(nt)])
                aucs[name] = auc_of(s_of[name], y_of[name])
            fin = [a for a in aucs.values() if np.isfinite(a)]
            flip = bool(fin and np.nanmean(fin) < 0.5)
            for name in corpora:
                put(name, dname, s_of[name], flip=flip)
                print(f"  [{dname}|{name}] AUC = {scores[name][dname]:.4f}"
                      + ("  (orientation flipped)" if flip else ""))

    # ---- Ghostbuster: LR trained ONCE on the RAID library, then frozen ----
    if args.with_ghostbuster:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        print("[ghostbuster] gpt2 + distilgpt2 features")
        gmods = []
        for mid in ("gpt2", "distilgpt2"):
            tk = AutoTokenizer.from_pretrained(mid)
            if tk.pad_token is None: tk.pad_token = tk.eos_token
            gmods.append((AutoModelForCausalLM.from_pretrained(mid).to(dev).eval(), tk))

        def gb_matrix(T):
            return np.hstack([np.array([gb_feats_one(t, mm, tk, dev) for t in T]) for mm, tk in gmods])

        tr_ai, tr_hu = load_raid_library(args.panels, args.humans, args.gb_train_cap, rng)
        print(f"  [ghostbuster] train pool: {len(tr_ai)} AI / {len(tr_hu)} human (RAID library)")
        Xtr = np.vstack([gb_matrix(tr_ai), gb_matrix(tr_hu)])
        ytr = np.array([1] * len(tr_ai) + [0] * len(tr_hu))
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
        for name in corpora:
            Xc = gb_matrix(texts_of[name])
            put(name, "Ghostbuster", clf.predict_proba(sc.transform(Xc))[:, 1])
            print(f"  [Ghostbuster|{name}] AUC = {scores[name]['Ghostbuster']:.4f}")
        for mm, _ in gmods: del mm
        torch.cuda.empty_cache()

    # ----------------------------- write table -------------------------------
    extra = [s.split(":", 1)[0] for s in args.hf_detector] + [s.split(":", 1)[0] for s in args.external]
    order = ["GLASS", "Fast-DetectGPT", "Binoculars", "Ghostbuster", "log-rank",
             "RoBERTa-OpenAI", "RADAR"] + extra
    dets = [d for d in order if any(d in scores[name] for name in corpora)]
    short = {"GLASS": "\\textbf{GLASS}", "Fast-DetectGPT": "Fast-DetectGPT",
             "Binoculars": "Binoculars", "Ghostbuster": "Ghostbuster",
             "log-rank": "log-rank", "RoBERTa-OpenAI": "RoBERTa", "RADAR": "RADAR"}

    # flat parquet of AUCs for reproducibility
    rec = []
    for name in corpora:
        for d in dets:
            rec.append({"corpus": name, "detector": d, "auc": scores[name].get(d, np.nan),
                        "n_ai": counts[name][0], "n_hum": counts[name][1]})
    Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rec).to_parquet(args.out_parquet)

    # ---------------- per-document dump: EVERY downstream number comes from here ----------
    # One row per test document: its corpus, label, word count, and each detector's score
    # (already oriented so higher = AI). This is what makes bootstrap CIs, the word-count
    # baseline and the length-threshold sweep derivable from the SAME documents as the
    # table above -- rather than from a second, differently-sampled run.
    pd_rows = []
    for name in corpora:
        T = texts_of[name]
        for i, t in enumerate(T):
            r = {"corpus": name, "idx": i, "label": int(y_of[name][i]),
                 "wlen": len(t.split())}
            for d in dets:
                r[d] = float(raw[name][d][i]) if d in raw[name] else np.nan
            pd_rows.append(r)
    P = pd.DataFrame(pd_rows)
    perdoc_path = args.out_parquet.replace(".parquet", "_perdoc.parquet")
    P.to_parquet(perdoc_path)
    print(f"[perdoc] {len(P)} documents x {len(dets)} detectors -> {perdoc_path}")

    # ---- word-count baseline: is this benchmark measuring provenance, or length? -------
    # A corpus where document length alone separates AI from human cannot discriminate
    # detectors, no matter how good they look on it. Report it next to the detectors.
    print("\n[length-confound audit] AUC of a WORD-COUNT-ONLY 'detector':")
    for name in corpora:
        x = P[P.corpus == name]
        a = auc_of(x.wlen.values.astype(float), x.label.values)
        best = max((scores[name][d] for d in dets if np.isfinite(scores[name].get(d, np.nan))),
                   default=np.nan)
        a2, direction = (a, "longer=AI") if a >= 0.5 else (1.0 - a, "shorter=AI")
        warn = "   <-- CONFOUNDED: length alone rivals the detectors" if a2 >= 0.85 else ""
        print(f"  [word-count|{name}] AUC = {a2:.4f} ({direction})   "
              f"best detector = {best:.4f}{warn}")
        print(f"RESULT[wordcount {name}] auc={a2:.4f} best_detector={best:.4f}")

    col = "l" + "c" * len(dets)
    L = ["% Cross-corpus frozen-bundle generalization, ALL detectors "
         "(build_thirdparty_competitors_v3.py).",
         "\\begin{table}[t]\\centering\\small",
         "\\setlength{\\tabcolsep}{4pt}",
         "\\caption{\\textbf{Frozen-bundle generalization to out-of-distribution "
         "corpora (no refit), all detectors.} The \\texttt{v3\\_english} bundle "
         "(subspaces, whitener, calibrator all fit on RAID) is applied \\emph{unchanged} "
         "to three public benchmarks GLASS never saw, each supplying its own AI and "
         "human text---\\emph{both} the generators \\emph{and} the human sources differ "
         "from RAID: HC3 (ChatGPT vs.\\ human Reddit/StackExchange answers), TuringBench "
         "($\\sim$20 pre-LLaMA generators vs.\\ human news), and MAGE ($\\sim$27 LLMs "
         "across mixed domains). The zero-shot baselines (Fast-DetectGPT, log-rank, "
         "Binoculars, RoBERTa-OpenAI) train on no frontier model; Ghostbuster's "
         "classifier is fit \\emph{once} on the RAID library and then frozen. Each cell "
         "is detection AUC on $n$ AI $+$ $n$ human documents \\emph{as provided} ($n$ as "
         "noted); no component is re-estimated on the target corpus. \\textbf{Bold} = "
         "best per row.}",
         "\\label{tab:thirdparty}",
         "\\begin{tabular}{" + col + "}", "\\toprule",
         "Corpus & " + " & ".join(short.get(d, d) for d in dets) + " \\\\", "\\midrule"]
    for name in corpora:
        present = [d for d in dets if np.isfinite(scores[name].get(d, np.nan))]
        best = max(present, key=lambda d: scores[name][d]) if present else None
        cells = []
        for d in dets:
            x = scores[name].get(d, np.nan)
            c = "n/a" if not np.isfinite(x) else f"{x:.3f}"
            if d == best and np.isfinite(x): c = f"\\textbf{{{c}}}"
            cells.append(c)
        nai, nhu = counts[name]
        L.append(f"{name} ({nai}/{nhu}) & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    Path(args.out_tex).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_tex).write_text("\n".join(L) + "\n")
    print(f"\n[out] {args.out_tex}\n[out] {args.out_parquet}")
    for name in corpora:
        print(f"  {name:12s} " + "  ".join(f"{d}={scores[name].get(d, float('nan')):.3f}" for d in dets))


if __name__ == "__main__":
    main()
