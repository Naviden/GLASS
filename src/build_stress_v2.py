#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_stress_v2.py
==================
RE-DO of tab_stress_profile on English with the v2_english bundle.

Takes an English frontier panel's test AI documents (the generators the
v2_english bundle knows), applies attacks, and scores GLASS (frozen v2_english
best-of residual) and Fast-DetectGPT per stage; AI-under-attack vs clean humans.

Stages: clean | character-edit | paraphrase (humarin) | [dipper, with --dipper].
Emits tables_v2/tab_stress_profile_v2.tex.

Run (turing, ~/glass/paper, GPU, venv):
    python3 build_stress_v2.py --jsonl results_v2/openrouter_en_s0/per_doc.jsonl
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd


# ---- GLASS bundle scoring ----
def residual_sq(X, mean, B):
    Xc = X - mean
    return np.sum((Xc - Xc @ B @ B.T) ** 2, axis=1)


def load_bundle(base, K):
    base = Path(base)
    sh = np.load(base / "S_H_universal.npz"); SH = (sh["mean"], sh["basis"][:, :K])
    SA = {p.stem[4:]: (np.load(p)["mean"], np.load(p)["basis"][:, :K])
          for p in sorted((base / "subspaces").glob("S_A_*.npz"))}
    wh = np.load(base / "whitener.npz"); WH = (wh["mean"], wh["W"])
    return SH, SA, WH


def glass_score(X, SH, SA, WH):
    Xw = (X - WH[0]) @ WH[1].T            # whiten before residual (matches headline)
    rh = residual_sq(Xw, SH[0], SH[1])
    allr = np.stack([residual_sq(Xw, m, b) for m, b in SA.values()], 0)
    return rh - allr.min(0)


# ---- Fast-DetectGPT (Falcon) ----
def fast_detectgpt(text, model, tok, dev, maxlen=512):
    import torch, torch.nn.functional as F
    enc = tok(text, return_tensors="pt", truncation=True, max_length=maxlen).to(dev)
    if enc.input_ids.size(1) < 2: return float("nan")
    with torch.no_grad():
        logits = model(**enc).logits
    lp = F.log_softmax(logits[:, :-1, :], -1); tgt = enc.input_ids[:, 1:]
    tlp = lp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
    p = lp.exp(); mu = (p * lp).sum(-1); var = (p * lp.pow(2)).sum(-1) - mu.pow(2)
    return float(((tlp - mu) / var.clamp_min(1e-12).sqrt()).mean().item())


# ---- attacks ----
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
    return out


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
    return np.vstack(out).astype(np.float64)


def auc(s, y):
    from sklearn.metrics import roc_auc_score
    s = np.asarray(s); ok = np.isfinite(s)
    return float(roc_auc_score(np.asarray(y)[ok], s[ok]))


def split(jsonl, humans, seed, tf):
    ai = []
    for line in open(jsonl):
        try: d = json.loads(line)
        except Exception: continue
        if d.get("error"): continue
        r = (d.get("response") or "").strip()
        if len(r) < 20: continue
        ai.append({"text": r, "title": str(d["title"])})
    ai = pd.DataFrame(ai); titles = sorted(ai["title"].unique())
    rng = np.random.default_rng(seed); n = max(1, int(round(len(titles) * tf)))
    tt = set(rng.choice(titles, size=n, replace=False))
    ai_te = list(ai[ai["title"].isin(tt)]["text"])
    H = pd.read_parquet(humans); ht = dict(zip(H["title"].astype(str), H["text"].astype(str)))
    hu = [ht[t] for t in tt if t in ht]
    return ai_te, hu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--humans", default="results_v2/raid_en_humans.parquet")
    ap.add_argument("--bundle", default=".cache/glass_production/v2_english")
    ap.add_argument("--encoder", default="microsoft/deberta-v3-base")
    ap.add_argument("--falcon", default="tiiuae/falcon-7b")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--char-rate", type=float, default=0.08)
    ap.add_argument("--skip-falcon", action="store_true")
    ap.add_argument("--out-tex", default="tables_v2/tab_stress_profile_v2.tex")
    args = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ai, hu = split(args.jsonl, args.humans, args.seed, args.test_frac)
    print(f"[data] {len(ai)} test AI, {len(hu)} humans")
    rng = random.Random(args.seed)

    stages = {"clean": ai,
              "character-edit": [char_attack(t, args.char_rate, rng) for t in ai]}
    print("[humarin] paraphrasing test AI ...")
    stages["paraphrase (humarin)"] = humarin_paraphrase(ai, dev)
    if dev == "cuda": torch.cuda.empty_cache()

    SH, SA, WH = load_bundle(args.bundle, args.K)
    print(f"[bundle] S_H + {len(SA)} subspaces (whitened)")
    Xhu = encode(hu, args.encoder, device=dev)
    res = {}
    for st, texts in stages.items():
        Xai = encode(texts, args.encoder, device=dev)
        s = np.concatenate([glass_score(Xai, SH, SA, WH), glass_score(Xhu, SH, SA, WH)])
        y = np.array([1] * len(Xai) + [0] * len(Xhu))
        res[st] = {"GLASS": auc(s, y)}
        print(f"  GLASS  {st:22s} AUC {res[st]['GLASS']:.4f}")

    if not args.skip_falcon:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"[falcon] {args.falcon}")
        ftok = AutoTokenizer.from_pretrained(args.falcon)
        fal = AutoModelForCausalLM.from_pretrained(args.falcon, dtype=torch.float16).to(dev).eval()
        hu_f = [fast_detectgpt(t, fal, ftok, dev) for t in hu]
        for st, texts in stages.items():
            ai_f = [fast_detectgpt(t, fal, ftok, dev) for t in texts]
            s = np.array(ai_f + hu_f); y = np.array([1] * len(ai_f) + [0] * len(hu_f))
            res[st]["Fast-DetectGPT"] = auc(s, y)
            print(f"  FDGPT  {st:22s} AUC {res[st]['Fast-DetectGPT']:.4f}")

    Path(args.out_tex).parent.mkdir(parents=True, exist_ok=True)
    has_f = not args.skip_falcon
    L = ["% RE-DO stress profile (English, v2_english bundle).",
         "\\begin{table}[t]\\centering\\small",
         "\\caption{Robustness on English frontier AI under attack "
         "(\\texttt{v2\\_english}). Detection AUC of attacked AI vs clean humans.}",
         "\\label{tab:stress-profile-v2}",
         "\\begin{tabular}{l" + ("cc" if has_f else "c") + "}", "\\toprule",
         "Stage & GLASS" + (" & Fast-DetectGPT" if has_f else "") + " \\\\", "\\midrule"]
    for st in stages:
        g = res[st]["GLASS"]
        row = f"{st} & {g:.3f}"
        if has_f: row += f" & {res[st]['Fast-DetectGPT']:.3f}"
        L.append(row + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    Path(args.out_tex).write_text("\n".join(L) + "\n")
    print(f"\n[out] {args.out_tex}")


if __name__ == "__main__":
    main()
