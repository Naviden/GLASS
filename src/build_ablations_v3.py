#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_ablations_v3.py
=====================
RE-DO of two detection ablations on English (frozen v3_english bundle),
both fast re-scores (no new model encoding):

  tab_k_ablation_v3   : GLASS detection AUC as the subspace rank K varies,
                        showing K=10 is within delta of the best K.
  tab_redundancy_v3   : detection AUC as random generators are dropped from
                        the library (library-redundancy / saturation check).

Test set = the five English panels' held-out 20% of titles (panel AI vs
paired English RAID humans), whitened best-of residual scored against the
frozen v3_english bundle.

Run (turing, ~/glass/paper, venv, GPU):
    PYTHONUTF8=1 python3 build_ablations_v3.py
"""
from __future__ import annotations
import argparse, json, glob
from pathlib import Path
import numpy as np
import pandas as pd


def residual_sq(X, mean, B):
    Xc = X - mean
    return np.sum((Xc - Xc @ B @ B.T) ** 2, axis=1)


def load_bundle(base):
    base = Path(base)
    sh = np.load(base / "S_H_universal.npz")
    SH = (sh["mean"], sh["basis"])                       # full K_max bases
    SA = {p.stem[4:]: (np.load(p)["mean"], np.load(p)["basis"])
          for p in sorted((base / "subspaces").glob("S_A_*.npz"))}
    wh = np.load(base / "whitener.npz"); WH = (wh["mean"], wh["W"])
    return SH, SA, WH


def glass_score(Xw, SH, SA, K, gens=None):
    rh = residual_sq(Xw, SH[0], SH[1][:, :K])
    use = SA if gens is None else {g: SA[g] for g in gens}
    allr = np.stack([residual_sq(Xw, m, b[:, :K]) for m, b in use.values()], 0)
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
            print(f"  encode {min(i+batch,len(texts))}/{len(texts)}", end="\r")
    print(); del mdl
    return np.vstack(out).astype(np.float64)


def auc(s, y):
    from sklearn.metrics import roc_auc_score
    s = np.asarray(s); y = np.asarray(y); ok = np.isfinite(s)
    return float(roc_auc_score(y[ok], s[ok]))


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
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--ks", default="2,5,10,20,30,50,64,128")
    ap.add_argument("--drops", default="0,2,4,6,9,12")
    ap.add_argument("--n-rep", type=int, default=5, help="random drops per level")
    ap.add_argument("--K-fixed", type=int, default=10, help="K for redundancy")
    args = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ks = [int(x) for x in args.ks.split(",")]
    drops = [int(x) for x in args.drops.split(",")]

    H = pd.read_parquet(args.humans)
    H["title"] = H["title"].astype(str); H["text"] = H["text"].astype(str)
    title2htext = {}
    for t, x in zip(H["title"], H["text"]):
        title2htext.setdefault(t, []).append(x)

    SH, SA, WH = load_bundle(args.bundle)
    gen_names = list(SA.keys())
    print(f"[bundle] S_H(K_max={SH[1].shape[1]}) + {len(SA)} subspaces")

    files = sorted(glob.glob(args.panels))
    kauc = {k: [] for k in ks}
    rauc = {d: [] for d in drops}
    for si, f in enumerate(files):
        rows = panel_ai(f)
        titles = sorted({t for t, _ in rows})
        rng = np.random.default_rng(si)
        n_te = max(1, int(round(len(titles) * args.test_frac)))
        test_t = set(rng.choice(titles, size=n_te, replace=False))
        ai = [r for t, r in rows if t in test_t]
        hu = [x for t in test_t if t in title2htext for x in title2htext[t]]
        if len(ai) < 20 or len(hu) < 20:
            print(f"  [s{si}] too few; skip"); continue
        Xai = encode(ai, args.encoder, device=dev)
        Xhu = encode(hu, args.encoder, device=dev)
        Xaiw = (Xai - WH[0]) @ WH[1].T
        Xhuw = (Xhu - WH[0]) @ WH[1].T
        y = np.array([1] * len(Xai) + [0] * len(Xhu))

        for k in ks:
            s = np.concatenate([glass_score(Xaiw, SH, SA, k),
                                glass_score(Xhuw, SH, SA, k)])
            kauc[k].append(auc(s, y))
        rdr = np.random.default_rng(100 + si)
        for d in drops:
            reps = []
            for _ in range(1 if d == 0 else args.n_rep):
                keep = (gen_names if d == 0 else
                        list(rdr.choice(gen_names, len(gen_names) - d, replace=False)))
                s = np.concatenate([glass_score(Xaiw, SH, SA, args.K_fixed, keep),
                                    glass_score(Xhuw, SH, SA, args.K_fixed, keep)])
                reps.append(auc(s, y))
            rauc[d].append(float(np.mean(reps)))
        print(f"  [s{si}] K10={kauc[10][-1]:.4f}  drop0={rauc[0][-1]:.4f}")

    def ms(v): return (np.mean(v), np.std(v)) if v else (float("nan"), 0.0)
    Path("tables_v2").mkdir(exist_ok=True)

    # ---- K ablation ----
    L = ["% RE-DO K ablation (English, frozen v3_english bundle).",
         "\\begin{table}[t]\\centering\\small",
         "\\caption{GLASS detection AUC vs subspace rank $K$ on the English "
         "panels (\\texttt{v3\\_english}), mean$\\pm$std over five panels.}",
         "\\label{tab:k-ablation-v3}",
         "\\begin{tabular}{l" + "c" * len(ks) + "}", "\\toprule",
         "$K$ & " + " & ".join(str(k) for k in ks) + " \\\\", "\\midrule"]
    cells = []
    for k in ks:
        m, s = ms(kauc[k]); cells.append(f"{m:.3f}")
    L.append("AUC & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    Path("tables_v2/tab_k_ablation_v3.tex").write_text("\n".join(L) + "\n")

    # ---- redundancy ----
    L = [f"% RE-DO library redundancy (English, frozen v3_english bundle, K={args.K_fixed}).",
         "\\begin{table}[t]\\centering\\small",
         "\\caption{GLASS detection AUC as random generators are dropped from "
         f"the \\texttt{{v3\\_english}} library ($K={args.K_fixed}$); mean$\\pm$std over five "
         f"panels $\\times$ {args.n_rep} random drops.}}",
         "\\label{tab:redundancy-v3}",
         "\\begin{tabular}{lc}", "\\toprule",
         "Generators dropped & GLASS AUC \\\\", "\\midrule"]
    for d in drops:
        m, s = ms(rauc[d])
        L.append(f"{d} (of {len(gen_names)}) & {m:.3f}$\\pm${s:.3f} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    Path("tables_v2/tab_redundancy_v3.tex").write_text("\n".join(L) + "\n")

    print("\n[out] tables_v2/tab_k_ablation_v3.tex, tables_v2/tab_redundancy_v3.tex")
    print("  K-AUC:", {k: round(ms(kauc[k])[0], 4) for k in ks})
    print("  drop-AUC:", {d: round(ms(rauc[d])[0], 4) for d in drops})


if __name__ == "__main__":
    main()
