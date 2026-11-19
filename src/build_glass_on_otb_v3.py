#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a frozen bundle on OpenTuringBench AI encodings against RAID human negatives.

The human negatives share their source corpus with the fitted human subspace."""
from __future__ import annotations
import argparse, glob, re
from pathlib import Path
import numpy as np


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
    Xw = (X - WH[0]) @ WH[1].T
    rh = residual_sq(Xw, SH[0], SH[1])
    allr = np.stack([residual_sq(Xw, m, b) for m, b in SA.values()], 0)
    return rh - allr.min(0)


def auc_tpr(s, y):
    from sklearn.metrics import roc_auc_score, roc_curve
    s = np.asarray(s, float); y = np.asarray(y); ok = np.isfinite(s)
    if len(np.unique(y[ok])) < 2:
        return float("nan"), float("nan")
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
            enc = tok(texts[i:i + batch], padding=True, truncation=True,
                      max_length=512, return_tensors="pt").to(device)
            h = mdl(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1e-9)).cpu().numpy())
            print(f"  encode {min(i+batch,len(texts))}/{len(texts)}", end="\r")
    print(); del mdl
    return np.vstack(out).astype(np.float64)


def split_name(fname):
    s = Path(fname).stem
    s = re.sub(r"^emb_", "", s); s = re.sub(r"_deberta.*$", "", s)
    return s.replace("__", " / ").replace("_", "-")


# row order + display names; everything else appended as-is
PRETTY = [
    ("in-domain / test",                       "In-domain test (7 generators)"),
    ("out-of-distribution / unseen-model",     "Unseen-model (OOD, strictly unseen)"),
    ("out-of-distribution / essay",            "Out-of-domain essays"),
    ("in-domain-variations / high-temperature","In-domain, high-temperature decoding"),
    ("in-domain-variations / mid-temperature", "In-domain, mid-temperature decoding"),
    ("in-domain-variations / larger-models",   "In-domain, larger models"),
    ("in-domain-variations / model-rewrite",   "In-domain, model rewrite"),
    ("in-domain-variations / human-continue",  "Human-continue (hybrid)"),
    ("in-domain-variations / human-revise",    "Human-revise (hybrid)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=".cache/glass_production/v3_english")
    ap.add_argument("--encoder", default="microsoft/deberta-v3-base")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--enc-glob",
                    default="results_v2/openturingbench_encodings/emb_*deberta-v3-base.npz")
    ap.add_argument("--humans", default="results_v2/raid_en_humans.parquet")
    ap.add_argument("--human-text-col", default="text")
    ap.add_argument("--max-humans", type=int, default=8000)
    ap.add_argument("--skip-train", action="store_true", default=True)
    ap.add_argument("--out-tex", default="tables/tab_glass_on_otb.tex")
    args = ap.parse_args()

    import torch, pandas as pd
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    SH, SA, WH = load_bundle(args.bundle, args.K)
    print(f"[bundle] S_H + {len(SA)} subspaces (whitened, K={args.K})")

    # shared human negatives (RAID-EN), encoded once
    H = pd.read_parquet(args.humans)
    htxt = H[args.human_text_col].astype(str).tolist()
    if len(htxt) > args.max_humans:
        rng = np.random.default_rng(0)
        htxt = [htxt[i] for i in rng.choice(len(htxt), args.max_humans, replace=False)]
    print(f"[humans] encoding {len(htxt)} RAID-EN humans (shared negatives)...")
    s_hu = glass_score(encode(htxt, args.encoder, device=dev), SH, SA, WH)
    n_hu = len(s_hu)

    # AI positives per split (already encoded)
    results = {}
    for f in sorted(glob.glob(args.enc_glob)):
        nm = split_name(f)
        if args.skip_train and "train" in nm:
            print(f"  skip (train): {nm}"); continue
        z = np.load(f, allow_pickle=True)
        s_ai = glass_score(z["X"].astype(np.float64), SH, SA, WH)
        n_ai = len(s_ai)
        s = np.concatenate([s_ai, s_hu])
        y = np.array([1] * n_ai + [0] * n_hu)
        a, t = auc_tpr(s, y)
        results[nm] = (n_ai, a, t)
        print(f"  [{nm}] AI {n_ai} vs H {n_hu}  AUC {a:.4f}  TPR@1 {t:.4f}")

    # order: known splits first (PRETTY), then any leftovers
    ordered = [(k, lbl) for k, lbl in PRETTY if k in results]
    seen = {k for k, _ in ordered}
    ordered += [(k, k) for k in results if k not in seen]

    def cell(v): return "\\textit{n/a}" if (v != v) else f"{v:.3f}"
    L = ["% GLASS (frozen v3_english bundle) on OpenTuringBench test splits.",
         "% Regenerated by build_glass_on_otb_v3.py. AI from OTB encodings;",
         "% human negatives = RAID-EN pool (OTB ships none). Every cell computed.",
         "\\begin{table}[t]\\centering\\small",
         "\\caption{\\glass\\ as a fixed-bundle detector on OpenTuringBench "
         "(\\cite{la2025openturingbench}): the frozen \\texttt{v3\\_english} bundle "
         "is applied with no refitting; detection AUC and TPR@$1\\%$FPR of machine "
         "vs human per split. \\emph{Bias caveat:} two of OTB's seven in-domain "
         "generators (Llama-3.1-8B, Qwen-2.5-7B) have matching subspaces in the "
         "\\glass\\ library while the other five and the unseen-model split do not, "
         "so the in-domain rows mix in-library and cross-generator detection; the "
         "\\texttt{Unseen-model} row is the only strictly unseen-generator test. "
         "OpenTuringBench provides no parallel human set, so human negatives are the "
         "RAID-EN pool used elsewhere (disclosed, not OTB-native).}",
         "\\label{tab:glass-on-otb}",
         "\\setlength{\\tabcolsep}{6pt}",
         "\\begin{tabular}{lrcc}", "\\toprule",
         f"OTB split & $n_{{\\mathrm{{AI}}}}$ (vs {n_hu} H) & AUC & TPR @ $1\\%$ FPR \\\\",
         "\\midrule"]
    for k, lbl in ordered:
        n_ai, a, t = results[k]
        L.append(f"{lbl} & {n_ai} & {cell(a)} & {cell(t)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    Path(args.out_tex).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_tex).write_text("\n".join(L) + "\n")
    print(f"\n[out] {args.out_tex}  ({len(ordered)} splits)")


if __name__ == "__main__":
    main()
