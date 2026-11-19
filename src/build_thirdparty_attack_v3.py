#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate frozen-bundle and baseline detectors under cross-corpus perturbations.

Paraphrasing truncates machine-generated text; human documents remain intact."""
from __future__ import annotations
import argparse, json, glob, tempfile, zipfile, random
from pathlib import Path
import numpy as np


# ----------------------------- GLASS -----------------------------------------
def residual_sq(X, mean, B):
    Xc = X - mean
    return np.sum((Xc - Xc @ B @ B.T) ** 2, axis=1)


def load_bundle(base, K):
    base = Path(base)
    sh = np.load(base / "S_H_universal.npz"); SH = (sh["mean"], sh["basis"][:, :K])
    SA = {}
    for p in sorted((base / "subspaces").glob("S_A_*.npz")):
        z = np.load(p); SA[p.stem[4:]] = (z["mean"], z["basis"][:, :K])
    wh = np.load(base / "whitener.npz"); WH = (wh["mean"], wh["W"])
    return SH, SA, WH


def glass_score(X, SH, SA, WH):
    Xw = (X - WH[0]) @ WH[1].T
    rh = residual_sq(Xw, SH[0], SH[1])
    allr = np.stack([residual_sq(Xw, m, b) for (m, b) in SA.values()], 0)
    return rh - allr.min(0)


# ----------------------------- attacks (verbatim from build_stress_v2.py) ----
HOMO = {"a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у", "i": "і", "s": "ѕ"}
def char_attack(text, rate, rng):
    ch = list(text); n = len(ch); ne = max(1, int(rate * n))
    for _ in range(ne):
        if not ch: break
        i = rng.randrange(len(ch)); op = rng.random(); c = ch[i]
        if op < 0.45 and c in HOMO: ch[i] = HOMO[c]
        elif op < 0.70 and i + 1 < len(ch): ch[i], ch[i + 1] = ch[i + 1], ch[i]
        elif op < 0.85: ch.insert(i, " ")
        elif c == " ": del ch[i]
        else: ch.insert(i, c)
    return "".join(ch)


def humarin_paraphrase(texts, dev, batch=16):
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    name = "humarin/chatgpt_paraphraser_on_T5_base"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(name).to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = [f"paraphrase: {t}" for t in texts[i:i + batch]]
            enc = tok(chunk, padding=True, truncation=True, max_length=256,
                      return_tensors="pt").to(dev)
            g = mdl.generate(**enc, max_length=256, num_beams=4, do_sample=True,
                             temperature=0.7, top_p=0.95)
            out.extend(tok.batch_decode(g, skip_special_tokens=True))
            print(f"  paraphrase {min(i+batch,len(texts))}/{len(texts)}", end="\r")
    print(); del mdl
    if dev == "cuda": torch.cuda.empty_cache()
    return out


# ----------------------------- scorers (identical to competitor script) ------
_SENT_SPLIT = __import__("re").compile(r"(?<=[.!?])\s+")
def _sent_tokenize(text):
    parts = [s for s in _SENT_SPLIT.split(" ".join(text.split())) if s.strip()]
    return parts or [text.strip()]


def dipper_paraphrase_all(ai_by_corpus, dev, lex=60, order=20, sent_interval=3,
                          model_id="kalpeshk2011/dipper-paraphraser-xxl",
                          tok_id="google/t5-v1_1-xxl"):
    """Strong paraphrase attack (DIPPER, Krishna et al. 2023). Loads the 11B model
    ONCE, paraphrases every corpus's AI docs, then frees it before scorers load.
    lex/order in {0,20,40,60,80,100}; higher = more diverse. Faithful to the
    official DipperParaphraser: control codes are (100 - diversity), paragraph
    paraphrased in windows of `sent_interval` sentences with rolling prefix."""
    import torch
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    assert lex in (0, 20, 40, 60, 80, 100) and order in (0, 20, 40, 60, 80, 100)
    lex_code, order_code = 100 - lex, 100 - order
    tok = T5Tokenizer.from_pretrained(tok_id)
    mdl = T5ForConditionalGeneration.from_pretrained(model_id, dtype=torch.float16).to(dev).eval()

    def para_one(text):
        sents = _sent_tokenize(text); prefix = ""; out = ""
        for i in range(0, len(sents), sent_interval):
            window = " ".join(sents[i:i + sent_interval])
            inp = f"lexical = {lex_code}, order = {order_code}"
            if prefix: inp += f" {prefix}"
            inp += f" <sent> {window} </sent>"
            enc = tok([inp], return_tensors="pt", truncation=True, max_length=512).to(dev)
            with torch.inference_mode():
                g = mdl.generate(**enc, do_sample=True, top_p=0.75, max_length=512)
            piece = tok.batch_decode(g, skip_special_tokens=True)[0]
            prefix = (prefix + " " + piece).strip(); out += " " + piece
        return out.strip()

    result = {}
    for name, ai in ai_by_corpus.items():
        print(f"[dipper] {name} ...")
        outs = []
        for j, t in enumerate(ai):
            outs.append(para_one(t))
            if (j + 1) % 50 == 0: print(f"  dipper[{name}] {j+1}/{len(ai)}", end="\r")
        print(); result[name] = outs
    del mdl
    if dev == "cuda": torch.cuda.empty_cache()
    return result


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
    del mdl
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
    GPT-4o / Qwen-2-72B / Gemma-2-9b / Mistral-7B / Llama-8B / Yi-Large. Wide
    format (one human column + one column per generator); we cycle generators so
    the AI side is a mix, like the other corpora."""
    from datasets import load_dataset
    ds = load_dataset(dataset, split="train", streaming=True)
    META = {"prompt", "human_story", "id", "url", "title", "index", "abstract", "story"}
    hum, ai = [], []
    gens = None
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


def load_raid_library(panels_glob, humans_parquet, cap, rng):
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
    hum = [str(x) for x in H["text"].tolist() if x is not None]
    hum = [h for h in hum if len(h.strip()) >= 20]
    if len(ai) > cap: ai = [ai[i] for i in rng.choice(len(ai), cap, replace=False)]
    if len(hum) > cap: hum = [hum[i] for i in rng.choice(len(hum), cap, replace=False)]
    return ai, hum


# ----------------------------- per-detector scoring of an arbitrary text list -
def score_all_detectors(texts, models, dev):
    """Return dict detector -> np.array of scores (higher = AI), for the given texts."""
    out = {}
    # GLASS
    SH, SA, WH, encoder = models["glass"]
    X = encode(texts, encoder, device=dev)
    out["GLASS"] = glass_score(X, SH, SA, WH)
    # Falcon family
    if "falcon" in models:
        ftok, m1, m2 = models["falcon"]
        fdg, lrk, bino = [], [], []
        for i, t in enumerate(texts):
            a, b = falcon_fdg_lr(t, m1, ftok, dev); fdg.append(a); lrk.append(b)
            if m2 is not None: bino.append(falcon_binoc(t, m1, m2, ftok, dev))
        out["Fast-DetectGPT"] = np.array(fdg)
        out["log-rank"] = -np.array(lrk)
        if m2 is not None: out["Binoculars"] = -np.array(bino)
    # RoBERTa
    if "roberta" in models:
        rtok, rmdl = models["roberta"]
        out["RoBERTa-OpenAI"] = np.array([roberta_p(t, rmdl, rtok, dev) for t in texts])
    # Ghostbuster
    if "ghostbuster" in models:
        gmods, sc, clf = models["ghostbuster"]
        Xc = np.hstack([np.array([gb_feats_one(t, mm, tk, dev) for t in texts]) for mm, tk in gmods])
        out["Ghostbuster"] = clf.predict_proba(sc.transform(Xc))[:, 1]
    # RADAR (paraphrase-robust; same P(machine) reader as RoBERTa)
    if "radar" in models:
        rtok, rmdl = models["radar"]
        out["RADAR"] = np.array([roberta_p(t, rmdl, rtok, dev) for t in texts])
    # extra released classifiers (e.g. OpenAI-large); orientation fixed post-hoc in main()
    if "hf_detector" in models:
        for dname, tok, mdl in models["hf_detector"]:
            out[dname] = np.array([roberta_p(t, mdl, tok, dev) for t in texts])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=".cache/glass_production/v3_english")
    ap.add_argument("--encoder", default="microsoft/deberta-v3-base")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--attacks", default="clean,character,paraphrase",
                    help="comma list from {clean,character,paraphrase}")
    ap.add_argument("--char-rate", type=float, default=0.08)
    ap.add_argument("--paraphraser", default="humarin", choices=["humarin", "dipper"],
                    help="paraphrase attack model; dipper = 11B DIPPER (needs ~22GB)")
    ap.add_argument("--dipper-lex", type=int, default=60, help="DIPPER lexical diversity (0..100 by 20)")
    ap.add_argument("--dipper-order", type=int, default=20, help="DIPPER order diversity (0..100 by 20)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with-falcon", action="store_true")
    ap.add_argument("--with-binoculars", action="store_true")
    ap.add_argument("--with-roberta", action="store_true")
    ap.add_argument("--with-ghostbuster", action="store_true")
    ap.add_argument("--with-radar", action="store_true",
                    help="RADAR paraphrase-robust detector (Hu et al., NeurIPS 2023)")
    ap.add_argument("--radar-model", default="TrustSafeAI/RADAR-Vicuna-7B")
    ap.add_argument("--hf-detector", action="append", default=[],
                    help='"Name:hf/model_id" released classifier, scored on the SAME docs (repeatable)')
    ap.add_argument("--clean-text", action="store_true",
                    help="uniform text-quality filter on every corpus (all detectors), like the main table")
    ap.add_argument("--min-words", type=int, default=100,
                    help="with --clean-text: min words/document (uniform)")
    ap.add_argument("--oversample", type=int, default=8,
                    help="with --clean-text: load n*this raw docs so enough survive the filter")
    ap.add_argument("--corpora", default="HC3,TuringBench,MAGE,NYT",
                    help="built-in corpora from HC3,TuringBench,MAGE,NYT")
    ap.add_argument("--hf-corpus", action="append", default=[],
                    help='add any HF dataset: "Name:id:text_col:label_col:ai_vals:human_vals[:config[:split]]"')
    ap.add_argument("--falcon", default="tiiuae/falcon-7b")
    ap.add_argument("--performer", default="tiiuae/falcon-7b-instruct")
    ap.add_argument("--roberta", default="openai-community/roberta-base-openai-detector")
    ap.add_argument("--panels", default="results_v2/openrouter_en_s*/per_doc.jsonl")
    ap.add_argument("--humans", default="results_v2/raid_en_humans.parquet")
    ap.add_argument("--gb-train-cap", type=int, default=4000)
    ap.add_argument("--out-tex", default="tables/tab_thirdparty_attack.tex")
    ap.add_argument("--out-parquet", default="results_v2/thirdparty_attack.parquet")
    args = ap.parse_args()
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]

    import torch, pandas as pd
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng_np = np.random.default_rng(args.seed)
    rng_py = random.Random(args.seed)

    # ---- load corpora (uniform quality filter, same as the main table) ----
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
        hum, ai = loader(args.n * args.oversample if args.clean_text else args.n)
        if args.clean_text:
            ai, hum, da, dh = filter_pair(ai, hum, args.n, min_words=args.min_words)
            print(f"  [{name}] cleaned (min_words={args.min_words}): dropped {da} ai / {dh} human junk")
        else:
            ai, hum = ai[:args.n], hum[:args.n]
        if len(hum) < 50 or len(ai) < 50:
            print(f"  [{name}] too few docs; skip"); continue
        corpora[name] = (ai, hum)
        print(f"  [{name}] {len(ai)} ai / {len(hum)} human")
    # ---- extra corpora from --hf-corpus (e.g. Pile) ----
    if args.hf_corpus:
        from build_thirdparty_competitors_v3 import load_hf_generic
        for spec in args.hf_corpus:
            parts = spec.split(":")
            if len(parts) < 6:
                print(f"  [hf-corpus] bad spec: {spec}"); continue
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
                print(f"  [{nm}] too few docs; skip"); continue
            corpora[nm] = (ai, hum)
            print(f"  [{nm}] {len(ai)} ai / {len(hum)} human")
    if not corpora:
        print("[fatal] no corpora"); return

    # ---- build attacked AI variants per corpus (paraphrase needs T5, loaded once) ----
    variants = {name: {} for name in corpora}   # name -> attack -> ai_texts
    need_para = "paraphrase" in attacks
    for name, (ai, hum) in corpora.items():
        if "clean" in attacks: variants[name]["clean"] = ai
        if "character" in attacks:
            variants[name]["character"] = [char_attack(t, args.char_rate, rng_py) for t in ai]
    if need_para:
        if args.paraphraser == "dipper":
            ai_by_corpus = {name: ai for name, (ai, hum) in corpora.items()}
            dp = dipper_paraphrase_all(ai_by_corpus, dev,
                                       lex=args.dipper_lex, order=args.dipper_order)
            for name in corpora: variants[name]["paraphrase"] = dp[name]
        else:  # humarin (default)
            for name, (ai, hum) in corpora.items():
                print(f"[paraphrase] {name} ...")
                variants[name]["paraphrase"] = humarin_paraphrase(ai, dev)

    # ---- load all detector models ONCE ----
    models = {}
    SH, SA, WH = load_bundle(args.bundle, args.K)
    print(f"[bundle] S_H + {len(SA)} subspaces (K={args.K}, whitened)")
    models["glass"] = (SH, SA, WH, args.encoder)

    if args.with_falcon:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        ftok = AutoTokenizer.from_pretrained(args.falcon)
        m1 = AutoModelForCausalLM.from_pretrained(args.falcon, dtype=torch.float16).to(dev).eval()
        m2 = (AutoModelForCausalLM.from_pretrained(args.performer, dtype=torch.float16).to(dev).eval()
              if args.with_binoculars else None)
        models["falcon"] = (ftok, m1, m2)
    if args.with_roberta:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        rtok = AutoTokenizer.from_pretrained(args.roberta)
        rmdl = AutoModelForSequenceClassification.from_pretrained(args.roberta).to(dev).eval()
        models["roberta"] = (rtok, rmdl)
    if args.with_ghostbuster:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        print("[ghostbuster] training LR once on RAID library ...")
        gmods = []
        for mid in ("gpt2", "distilgpt2"):
            tk = AutoTokenizer.from_pretrained(mid)
            if tk.pad_token is None: tk.pad_token = tk.eos_token
            gmods.append((AutoModelForCausalLM.from_pretrained(mid).to(dev).eval(), tk))
        tr_ai, tr_hu = load_raid_library(args.panels, args.humans, args.gb_train_cap, rng_np)
        gbmat = lambda T: np.hstack([np.array([gb_feats_one(t, mm, tk, dev) for t in T]) for mm, tk in gmods])
        Xtr = np.vstack([gbmat(tr_ai), gbmat(tr_hu)])
        ytr = np.array([1] * len(tr_ai) + [0] * len(tr_hu))
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
        models["ghostbuster"] = (gmods, sc, clf)
        print(f"  [ghostbuster] trained on {len(tr_ai)} AI / {len(tr_hu)} human")
    if args.with_radar:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        dtok = AutoTokenizer.from_pretrained(args.radar_model)
        dmdl = AutoModelForSequenceClassification.from_pretrained(args.radar_model).to(dev).eval()
        models["radar"] = (dtok, dmdl)
        print(f"  [radar] {args.radar_model}")
    if args.hf_detector:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        hfd = []
        for spec in args.hf_detector:
            dname, mid = spec.split(":", 1)
            tok = AutoTokenizer.from_pretrained(mid)
            mdl = AutoModelForSequenceClassification.from_pretrained(mid).to(dev).eval()
            hfd.append((dname, tok, mdl))
            print(f"  [hf-detector] {dname} = {mid}")
        models["hf_detector"] = hfd

    # ---- score: human side once per corpus; AI side per attack ----
    recs = []
    for name, (ai, hum) in corpora.items():
        hum_scores = score_all_detectors(hum, models, dev)        # clean humans, once
        for atk in attacks:
            if atk not in variants[name]: continue
            ai_scores = score_all_detectors(variants[name][atk], models, dev)
            for det in ai_scores:
                sa = ai_scores[det]; sh = hum_scores[det]
                y = np.array([1] * len(sa) + [0] * len(sh))
                a = auc_of(np.concatenate([sa, sh]), y)
                recs.append({"corpus": name, "attack": atk, "detector": det, "auc": a})
                print(f"  [{name}|{atk}|{det}] AUC = {a:.4f}")

    df = pd.DataFrame(recs)
    # auto-orient released hf-detectors by their CLEAN mean AUC (flip if < 0.5),
    # matching build_thirdparty_competitors_v3; flipping scores == 1 - AUC.
    for spec in args.hf_detector:
        dname = spec.split(":", 1)[0]
        cl = df[(df.detector == dname) & (df.attack == "clean")]["auc"]
        if len(cl) and cl.mean() < 0.5:
            df.loc[df.detector == dname, "auc"] = 1.0 - df.loc[df.detector == dname, "auc"]
            print(f"  [{dname}] orientation flipped (clean mean AUC was {cl.mean():.3f})")
    Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_parquet)

    # ---- summary table: detector x attack, mean AUC over corpora (+ drop vs clean) ----
    extra = [s.split(":", 1)[0] for s in args.hf_detector]
    det_order = ["GLASS", "Fast-DetectGPT", "Binoculars", "Ghostbuster", "log-rank",
                 "RoBERTa-OpenAI", "RADAR"] + extra
    dets = [d for d in det_order if d in set(df["detector"])]
    atk_order = [a for a in ["clean", "character", "paraphrase"] if a in set(df["attack"])]
    pretty_atk = {"clean": "Clean", "character": "Char-edit", "paraphrase": "Paraphrase"}
    short = {"GLASS": "\\textbf{GLASS}", "Fast-DetectGPT": "Fast-DetectGPT",
             "Binoculars": "Binoculars", "Ghostbuster": "Ghostbuster",
             "log-rank": "log-rank", "RoBERTa-OpenAI": "RoBERTa", "RADAR": "RADAR"}

    def cell(det, atk):
        v = df[(df.detector == det) & (df.attack == atk)]["auc"]
        return float(v.mean()) if len(v) else float("nan")

    has_drop = "clean" in atk_order and len(atk_order) > 1
    head = ["Detector"] + [pretty_atk[a] for a in atk_order] + (["worst-case drop"] if has_drop else [])
    L = ["% Cross-corpus detection under attack, mean AUC over corpora "
         "(build_thirdparty_attack_v3.py).",
         "\\begin{table}[t]\\centering\\small",
         "\\setlength{\\tabcolsep}{6pt}",
         "\\caption{\\textbf{Cross-corpus detection under attack.} Mean AUC over the "
         "out-of-distribution corpora when the AI side is "
         "perturbed by a character-level edit or a T5 paraphrase (humans left clean); "
         "the same attacks as the in-library stress test. Reported alongside the clean "
         "comparison (Table~\\ref{tab:thirdparty}); all detectors and attacks shown. "
         "``Worst-case drop'' is the largest AUC loss from clean across the two attacks. "
         "\\textbf{Bold} = best per column.}",
         "\\label{tab:thirdparty-attack}",
         "\\begin{tabular}{" + ("l" + "c" * (len(head) - 1)) + "}", "\\toprule",
         " & ".join(head) + " \\\\", "\\midrule"]
    # precompute columns for bolding best per attack
    colvals = {a: {d: cell(d, a) for d in dets} for a in atk_order}
    best_per_atk = {a: max((d for d in dets if np.isfinite(colvals[a][d])),
                           key=lambda d: colvals[a][d], default=None) for a in atk_order}
    # worst-case drop: smallest drop = most robust -> bold the smallest
    drops = {}
    if has_drop:
        for d in dets:
            cl = colvals["clean"][d]
            worst = max((cl - colvals[a][d]) for a in atk_order if a != "clean"
                        and np.isfinite(colvals[a][d]) and np.isfinite(cl))
            drops[d] = worst
        best_drop = min(drops, key=lambda d: drops[d]) if drops else None
    for d in dets:
        cells = []
        for a in atk_order:
            x = colvals[a][d]
            c = "n/a" if not np.isfinite(x) else f"{x:.3f}"
            if d == best_per_atk[a] and np.isfinite(x): c = f"\\textbf{{{c}}}"
            cells.append(c)
        if has_drop:
            dv = drops.get(d, float("nan"))
            if not np.isfinite(dv):
                c = "n/a"
            elif dv >= 0:          # AUC lost under attack
                c = f"$-${dv:.3f}"
            else:                  # AUC gained (rare)
                c = f"$+${abs(dv):.3f}"
            if d == best_drop and np.isfinite(dv): c = f"\\textbf{{{c}}}"
            cells.append(c)
        L.append(f"{short.get(d, d)} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    Path(args.out_tex).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_tex).write_text("\n".join(L) + "\n")
    print(f"\n[out] {args.out_tex}\n[out] {args.out_parquet}")
    # console summary
    for d in dets:
        line = "  ".join(f"{a}={colvals[a][d]:.3f}" for a in atk_order)
        extra = f"   worst-drop={drops[d]:.3f}" if has_drop and d in drops else ""
        print(f"  {d:16s} {line}{extra}")


if __name__ == "__main__":
    main()
