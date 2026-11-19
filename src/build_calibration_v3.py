#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate probability calibration on separate calibration and evaluation splits."""
from __future__ import annotations
import argparse, json, glob
from pathlib import Path
import numpy as np
import pandas as pd


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


def brier(p, y): return float(np.mean((p - y) ** 2))


def ece(p, y, nbins=10):
    bins = np.linspace(0, 1, nbins + 1)
    e = 0.0; n = len(p)
    for b in range(nbins):
        m = (p >= bins[b]) & (p < bins[b + 1]) if b < nbins - 1 else \
            (p >= bins[b]) & (p <= bins[b + 1])
        if m.sum() == 0: continue
        e += (m.sum() / n) * abs(p[m].mean() - y[m].mean())
    return float(e)


def panel_ai(jsonl):
    out = []
    for line in open(jsonl, encoding="utf-8"):
        try: d = json.loads(line)
        except Exception: continue
        if d.get("error"): continue
        r = (d.get("response") or "").strip()
        if len(r) < 20: continue
        out.append((str(d["title"]), r))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default="results_v2/openrouter_en_s*/per_doc.jsonl")
    ap.add_argument("--humans", default="results_v2/raid_en_humans.parquet")
    ap.add_argument("--bundle", default=".cache/glass_production/v3_english")
    ap.add_argument("--encoder", default="microsoft/deberta-v3-base")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--test-frac", type=float, default=0.2)
    args = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression

    H = pd.read_parquet(args.humans)
    H["title"] = H["title"].astype(str); H["text"] = H["text"].astype(str)
    t2h = {}
    for t, x in zip(H["title"], H["text"]):
        t2h.setdefault(t, []).append(x)

    SH, SA, WH = load_bundle(args.bundle, args.K)
    cals = ["Uncalibrated", "Platt", "Isotonic"]
    res = {c: {"brier": [], "ece": []} for c in cals}

    for si, f in enumerate(sorted(glob.glob(args.panels))):
        rows = panel_ai(f)
        titles = sorted({t for t, _ in rows})
        rng = np.random.default_rng(si)
        n_te = max(1, int(round(len(titles) * args.test_frac)))
        test_t = set(rng.choice(titles, size=n_te, replace=False))
        ai = [r for t, r in rows if t in test_t]
        hu = [x for t in test_t if t in t2h for x in t2h[t]]
        if len(ai) < 20 or len(hu) < 20:
            continue
        Xai = encode(ai, args.encoder, device=dev)
        Xhu = encode(hu, args.encoder, device=dev)
        s = np.concatenate([glass_score(Xai, SH, SA, WH), glass_score(Xhu, SH, SA, WH)])
        y = np.array([1] * len(Xai) + [0] * len(Xhu))
        # 50/50 cal/eval split
        idx = rng.permutation(len(s)); h = len(s) // 2
        ci, ei = idx[:h], idx[h:]
        sc, yc = s[ci], y[ci]; se, ye = s[ei], y[ei]

        # Uncalibrated: sigmoid of z-scored score (z fit on cal)
        mu, sd = sc.mean(), sc.std() + 1e-9
        p_un = 1 / (1 + np.exp(-(se - mu) / sd))
        res["Uncalibrated"]["brier"].append(brier(p_un, ye))
        res["Uncalibrated"]["ece"].append(ece(p_un, ye))
        # Platt
        lr = LogisticRegression().fit(sc.reshape(-1, 1), yc)
        p_pl = lr.predict_proba(se.reshape(-1, 1))[:, 1]
        res["Platt"]["brier"].append(brier(p_pl, ye))
        res["Platt"]["ece"].append(ece(p_pl, ye))
        # Isotonic
        ir = IsotonicRegression(out_of_bounds="clip").fit(sc, yc)
        p_is = np.clip(ir.predict(se), 0, 1)
        res["Isotonic"]["brier"].append(brier(p_is, ye))
        res["Isotonic"]["ece"].append(ece(p_is, ye))
        print(f"  [s{si}] Platt Brier={res['Platt']['brier'][-1]:.4f} "
              f"ECE={res['Platt']['ece'][-1]:.4f}")

    def ms(v): return (np.mean(v), np.std(v)) if v else (float("nan"), 0.0)
    Path("tables_v2").mkdir(exist_ok=True)
    L = ["% RE-DO calibration (English, frozen v3_english bundle).",
         "\\begin{table}[t]\\centering\\small",
         "\\caption{Calibration of GLASS detection probabilities on the English "
         "panels (\\texttt{v3\\_english}): Brier score and 10-bin ECE under three "
         "calibrators, mean$\\pm$std over five panels.}",
         "\\label{tab:calibration-v3}",
         "\\begin{tabular}{lcc}", "\\toprule",
         "Calibrator & Brier $\\downarrow$ & ECE $\\downarrow$ \\\\", "\\midrule"]
    for c in cals:
        bm, bs = ms(res[c]["brier"]); em, es = ms(res[c]["ece"])
        L.append(f"{c} & {bm:.3f}$\\pm${bs:.3f} & {em:.3f}$\\pm${es:.3f} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    Path("tables_v2/tab_calibration_v3.tex").write_text("\n".join(L) + "\n")
    print("\n[out] tables_v2/tab_calibration_v3.tex")
    for c in cals:
        bm, bs = ms(res[c]["brier"]); em, es = ms(res[c]["ece"])
        print(f"  {c:14s} Brier {bm:.4f}  ECE {em:.4f}")


if __name__ == "__main__":
    main()
