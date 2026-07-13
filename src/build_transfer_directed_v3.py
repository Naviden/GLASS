#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_transfer_directed_v3.py
=============================
DIRECTED test of Theorem 1 (not the symmetric dual already in the paper).

Theorem 1 (matched filter, isotropic-Gaussian model):
    AUC_tr(i -> j) = Phi( cbar_ij * Phi^{-1}( AUC_ii ) )
where the classifier is the matched filter  w_i = mu_{A_i} / ||mu_{A_i}||
trained on generator i vs human, AUC_ii is its self (in-distribution) AUC,
and cbar_ij is the mean canonical correlation (mean cosine of principal
angles) between the two generator subspaces.

This script MEASURES AUC_tr(i->j) directly for every ordered pair and
compares it to the value PREDICTED from cbar_ij and AUC_ii, reporting
Pearson r, MAE, RMSE, and a calibration slope. Writes a results parquet,
a paste-ready scatter (optional), and tables/tab_transfer_directed.tex.

Everything is whitened with the v3_english whitener so the human class is
~isotropic (the theorem's premise). NOTHING is invented.

Run (turing, ~/glass/paper, venv, GPU; pin a free GPU):
    CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
      PYTHONUTF8=1 python3 build_transfer_directed_v3.py
"""
from __future__ import annotations
import argparse, glob, json, re
from pathlib import Path
import numpy as np


def load_bundle(base, K):
    base = Path(base)
    SA = {p.stem[4:]: np.load(p)["basis"][:, :K]
          for p in sorted((base / "subspaces").glob("S_A_*.npz"))}
    wh = np.load(base / "whitener.npz"); WH = (wh["mean"], wh["W"])
    return SA, WH


def whiten(X, WH):
    return (X - WH[0]) @ WH[1].T


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


def auc_score(scores, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, scores))


def norm_name(s):
    # keep the provider; only / and _ differ between panel and bundle names
    s = s.lower().replace("/", "-").replace("_", "-")
    s = s.replace("instruct", "").replace("-it", "")
    return re.sub(r"[^a-z0-9]", "", s)


def cbar(Bi, Bj):
    # columns orthonormal -> singular values of Bi^T Bj are cosines of principal angles
    s = np.linalg.svd(Bi.T @ Bj, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return float(np.mean(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default="results_v2/openrouter_en_s*/per_doc.jsonl")
    ap.add_argument("--humans", default="results_v2/raid_en_humans.parquet")
    ap.add_argument("--human-text-col", default="text")
    ap.add_argument("--bundle", default=".cache/glass_production/v3_english")
    ap.add_argument("--encoder", default="microsoft/deberta-v3-base")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--min-docs", type=int, default=100, help="min AI docs per generator")
    ap.add_argument("--max-humans", type=int, default=8000)
    ap.add_argument("--out-tex", default="tables/tab_transfer_directed.tex")
    ap.add_argument("--out-parquet", default="results_v2/transfer_directed.parquet")
    args = ap.parse_args()

    from scipy.stats import norm
    import pandas as pd
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    SA, WH = load_bundle(args.bundle, args.K)
    SAn = {norm_name(k): (k, v) for k, v in SA.items()}
    print(f"[bundle] {len(SA)} subspaces, K={args.K}")

    # gather AI docs per generator from the panels
    gens = {}
    for f in sorted(glob.glob(args.panels)):
        for line in open(f, encoding="utf-8"):
            try: d = json.loads(line)
            except Exception: continue
            if d.get("error"): continue
            r = (d.get("response") or "").strip()
            if len(r) < 20: continue
            gens.setdefault(str(d["model"]), []).append(r)
    # keep only generators present in the library, with enough docs
    keep = {}
    for g, txts in gens.items():
        m = SAn.get(norm_name(g))
        if m and len(txts) >= args.min_docs:
            keep[g] = (m[0], m[1], txts)  # libname, basis, texts
    print(f"[gens] {len(keep)} library generators with >= {args.min_docs} docs")
    if len(keep) < 3:
        raise SystemExit("Too few matched generators; check name alignment.")

    # humans (negatives), whitened
    H = pd.read_parquet(args.humans)
    htxt = H[args.human_text_col].astype(str).tolist()
    if len(htxt) > args.max_humans:
        rng = np.random.default_rng(0)
        htxt = [htxt[i] for i in rng.choice(len(htxt), args.max_humans, replace=False)]
    print(f"[humans] encoding {len(htxt)} ...")
    Hw = whiten(encode(htxt, args.encoder, device=dev), WH)
    mu_H = Hw.mean(0)

    # per-generator: whitened embeddings, mean, matched filter, basis
    G = {}
    for g, (libname, basis, txts) in keep.items():
        Xw = whiten(encode(txts, args.encoder, device=dev), WH)
        mu = Xw.mean(0) - mu_H
        nrm = np.linalg.norm(mu)
        w = mu / (nrm + 1e-12)
        G[g] = {"libname": libname, "B": basis, "Xw": Xw, "w": w}
        print(f"  [{g}] n={len(txts)} ||mu||={nrm:.3f}")

    names = list(G.keys())
    yj = lambda nj: np.array([1] * len(G[nj]["Xw"]) + [0] * len(Hw))

    # self AUC_ii
    auc_ii = {}
    for i in names:
        w = G[i]["w"]
        s = np.concatenate([G[i]["Xw"] @ w, Hw @ w])
        auc_ii[i] = auc_score(s, yj(i))

    # directed pairs
    rows = []
    for i in names:
        w = G[i]["w"]; a_ii = auc_ii[i]
        z_ii = norm.ppf(min(max(a_ii, 1e-6), 1 - 1e-6))
        for j in names:
            if i == j: continue
            s = np.concatenate([G[j]["Xw"] @ w, Hw @ w])
            meas = auc_score(s, yj(j))
            c = cbar(G[i]["B"], G[j]["B"])
            pred = float(norm.cdf(c * z_ii))
            rows.append(dict(i=G[i]["libname"], j=G[j]["libname"],
                             cbar=c, auc_ii=a_ii, auc_tr_meas=meas, auc_tr_pred=pred))
    df = pd.DataFrame(rows)
    Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_parquet)

    meas = df["auc_tr_meas"].to_numpy(); pred = df["auc_tr_pred"].to_numpy()
    r = float(np.corrcoef(meas, pred)[0, 1])
    mae = float(np.mean(np.abs(meas - pred)))
    rmse = float(np.sqrt(np.mean((meas - pred) ** 2)))
    # OLS slope of measured on predicted
    A = np.vstack([pred, np.ones_like(pred)]).T
    slope, intercept = np.linalg.lstsq(A, meas, rcond=None)[0]
    print(f"\n[directed Theorem 1 test]  n_pairs={len(df)}  "
          f"Pearson r={r:.3f}  MAE={mae:.3f}  RMSE={rmse:.3f}  "
          f"slope={slope:.3f}  intercept={intercept:.3f}")

    L = ["% Directed test of Theorem 1 (build_transfer_directed_v3.py).",
         "% Measured AUC_tr(i->j) vs predicted Phi(cbar_ij * Phi^{-1}(AUC_ii)).",
         "\\begin{table}[t]\\centering\\small",
         "\\caption{Direct test of the transfer identity (Theorem~\\ref{thm:transfer}). "
         "For every ordered generator pair $(i,j)$ in the \\texttt{v3\\_english} library "
         f"({len(names)} generators, ${len(df)}$ ordered pairs), we train the matched "
         "filter $w_i=\\mu_{A_i}/\\lVert\\mu_{A_i}\\rVert$ on generator $i$ vs human and "
         "measure its directed transfer AUC to generator $j$, then compare to the value "
         "predicted by the identity $\\mathrm{AUC}_{\\mathrm{tr}}(i\\!\\to\\!j)="
         "\\Phi(\\bar c_{ij}\\,\\Phi^{-1}(\\mathrm{AUC}_{ii}))$ from the measured "
         "self-AUC and the subspace mean canonical correlation $\\bar c_{ij}$.}",
         "\\label{tab:transfer-directed}",
         "\\begin{tabular}{lc}", "\\toprule",
         "Quantity & Value \\\\", "\\midrule",
         f"Ordered pairs & {len(df)} \\\\",
         f"Pearson $r$ (measured vs predicted) & {r:.3f} \\\\",
         f"Mean absolute error & {mae:.3f} \\\\",
         f"RMSE & {rmse:.3f} \\\\",
         f"Calibration slope (measured on predicted) & {slope:.3f} \\\\",
         "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    Path(args.out_tex).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_tex).write_text("\n".join(L) + "\n")
    print(f"[out] {args.out_tex}\n[out] {args.out_parquet}")


if __name__ == "__main__":
    main()
