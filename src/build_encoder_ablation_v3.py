#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_encoder_ablation_v3.py
============================
RE-DO of tab_encoder_ablation on English: GLASS detection AUC across encoder
backbones, re-fit per panel (whitened, K=10) on the English data.

For each encoder and each of the five English panels: hold out 20% of titles
as test; fit the human subspace S_H on background English RAID humans and a
per-generator AI subspace on the panel's train-split generations (whitened,
Cholesky); score test AI vs paired test humans; report AUC.

Encoders: DeBERTa-v3-base (canonical), DeBERTa-v3-large, E5-large-v2.

Writes tables_v2/tab_encoder_ablation_v3.tex (AUC mean$\\pm$std over panels).

Run (turing, ~/glass/paper, venv, GPU):
    PYTHONUTF8=1 python3 build_encoder_ablation_v3.py
"""
from __future__ import annotations
import argparse, json, glob, collections
from pathlib import Path
import numpy as np
import pandas as pd

ENCODERS = [
    ("DeBERTa-v3-base", "microsoft/deberta-v3-base"),
    ("DeBERTa-v3-large", "microsoft/deberta-v3-large"),
    ("E5-large-v2", "intfloat/e5-large-v2"),
]


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


def wapply(X, m, W): return (X - m) @ W.T


def residual_sq(X, mean, B):
    Xc = X - mean
    return np.sum((Xc - Xc @ B @ B.T) ** 2, axis=1)


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
            print(f"  encode {min(i+batch,len(texts))}/{len(texts)}", end="\r")
    print(); del mdl
    import torch as _t
    if _t.cuda.is_available(): _t.cuda.empty_cache()
    return np.vstack(out).astype(np.float64)


def auc(s, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.asarray(y), np.asarray(s)))


def panel_rows(jsonl):
    out = []
    for line in open(jsonl, encoding="utf-8"):
        try: d = json.loads(line)
        except Exception: continue
        if d.get("error"): continue
        r = (d.get("response") or "").strip()
        if len(r) < 20: continue
        out.append((str(d["title"]), str(d["model"]), r))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default="results_v2/openrouter_en_s*/per_doc.jsonl")
    ap.add_argument("--humans", default="results_v2/raid_en_humans.parquet")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--sh-humans", type=int, default=4000)
    ap.add_argument("--min-per-gen", type=int, default=20)
    args = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    H = pd.read_parquet(args.humans)
    H["title"] = H["title"].astype(str); H["text"] = H["text"].astype(str)
    files = sorted(glob.glob(args.panels))

    res = {name: [] for name, _ in ENCODERS}
    for name, model in ENCODERS:
        print(f"\n===== encoder {name} ({model}) =====")
        for si, f in enumerate(files):
            rows = panel_rows(f)
            titles = sorted({t for t, _, _ in rows})
            rng = np.random.default_rng(si)
            n_te = max(1, int(round(len(titles) * args.test_frac)))
            test_t = set(rng.choice(titles, size=n_te, replace=False))

            # train AI per generator (non-test titles), test AI (test titles)
            per_gen_tr = collections.defaultdict(list); ai_te = []
            for t, g, r in rows:
                if t in test_t:
                    ai_te.append(r)
                else:
                    per_gen_tr[g].append(r)
            per_gen_tr = {g: v for g, v in per_gen_tr.items() if len(v) >= args.min_per_gen}
            ht = dict(zip(H["title"], H["text"]))
            hu_te = [ht[t] for t in test_t if t in ht]
            bg = H[~H["title"].isin(test_t)]["text"]
            bg = list(bg.sample(n=min(args.sh_humans, len(bg)), random_state=si))
            if len(ai_te) < 20 or len(hu_te) < 20 or not per_gen_tr:
                print(f"  [s{si}] skip"); continue

            gens = sorted(per_gen_tr)
            offs = {}; o = 0; tr_texts = []
            for g in gens:
                offs[g] = (o, o + len(per_gen_tr[g])); o += len(per_gen_tr[g])
                tr_texts += per_gen_tr[g]
            X = encode(bg + tr_texts + ai_te + hu_te, model, device=dev)
            n0 = len(bg); n1 = n0 + len(tr_texts); n2 = n1 + len(ai_te)
            Xbg, Xtr, Xai, Xhu = X[:n0], X[n0:n1], X[n1:n2], X[n2:]

            wm, W = whiten_fit(Xbg)
            Sh = fit_subspace(wapply(Xbg, wm, W), args.k)
            SA = {}
            Xtrw = wapply(Xtr, wm, W)
            for g in gens:
                a, b = offs[g]
                SA[g] = fit_subspace(Xtrw[a:b], args.k)
            def score(Z):
                Zw = wapply(Z, wm, W)
                rh = residual_sq(Zw, Sh[0], Sh[1])
                allr = np.stack([residual_sq(Zw, m, bb) for m, bb in SA.values()], 0)
                return rh - allr.min(0)
            s = np.concatenate([score(Xai), score(Xhu)])
            y = np.array([1] * len(Xai) + [0] * len(Xhu))
            res[name].append(auc(s, y))
            print(f"  [s{si}] AUC={res[name][-1]:.4f}")

    def ms(v): return (np.mean(v), np.std(v)) if v else (float("nan"), 0.0)
    Path("tables_v2").mkdir(exist_ok=True)
    L = ["% RE-DO encoder ablation (English, refit per panel, whitened K=10).",
         "\\begin{table}[t]\\centering\\small",
         "\\caption{GLASS detection AUC across encoder backbones on the English "
         "panels (refit per panel, whitened, $K=10$), mean$\\pm$std over five "
         "panels.}",
         "\\label{tab:encoder-ablation-v3}",
         "\\begin{tabular}{lc}", "\\toprule",
         "Encoder & GLASS AUC \\\\", "\\midrule"]
    for name, _ in ENCODERS:
        m, s = ms(res[name])
        L.append(f"{name} & {m:.3f}$\\pm${s:.3f} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    Path("tables_v2/tab_encoder_ablation_v3.tex").write_text("\n".join(L) + "\n")
    print("\n[out] tables_v2/tab_encoder_ablation_v3.tex")
    for name, _ in ENCODERS:
        m, s = ms(res[name]); print(f"  {name:18s} AUC {m:.4f}±{s:.4f}")


if __name__ == "__main__":
    main()
