#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare GLASS and an OpenTuringBench-trained classifier on English panels."""
from __future__ import annotations
import argparse, json, glob
from pathlib import Path
import numpy as np
import pandas as pd

OTB_GENERATORS = [
    "Intel/neural-chat-7b-v3-3", "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-9b-it", "meta-llama/Llama-3.1-8B-Instruct",
    "microsoft/Phi-3.5-mini-instruct", "mistralai/Mistral-7B-Instruct-v0.3",
    "upstage/SOLAR-10.7B-Instruct-v1.0",
]
HUMAN_LABEL = "Human"


# ---------- GLASS frozen-bundle scoring (whitened best-of) ----------
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


# ---------- encoder ----------
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


def auc_tpr(s, y):
    from sklearn.metrics import roc_auc_score, roc_curve
    s = np.asarray(s); y = np.asarray(y); ok = np.isfinite(s)
    a = float(roc_auc_score(y[ok], s[ok]))
    fpr, tpr, _ = roc_curve(y[ok], s[ok])
    t = float(np.interp(0.01, fpr, tpr))
    return a, t


def panel_titles_ai(jsonl):
    rows = []
    for line in open(jsonl):
        try: d = json.loads(line)
        except Exception: continue
        if d.get("error"): continue
        r = (d.get("response") or "").strip()
        if len(r) < 20: continue
        rows.append((str(d["title"]), str(d["model"]), r))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default="results_v2/openrouter_en_s*/per_doc.jsonl")
    ap.add_argument("--humans", default="results_v2/raid_en_humans.parquet")
    ap.add_argument("--otb-npz", default="results_v2/openturingbench_encodings/"
                                          "emb_in_domain__train_deberta-v3-base.npz")
    ap.add_argument("--bundle", default=".cache/glass_production/v3_english")
    ap.add_argument("--encoder", default="microsoft/deberta-v3-base")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--max-human-train", type=int, default=8000)
    ap.add_argument("--otb-exclude", default="",
                    help="comma-separated substrings; OTB training generators "
                         "matching any are dropped (to avoid train/test overlap "
                         "with the frontier-EN panel)")
    ap.add_argument("--out-tex", default="tables_v2/tab_symmetric_v3.tex")
    args = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # filter OTB training generators (contamination control)
    excl = [s.strip() for s in args.otb_exclude.split(",") if s.strip()]
    otb_gens = [g for g in OTB_GENERATORS if not any(e.lower() in g.lower() for e in excl)]
    dropped = [g for g in OTB_GENERATORS if g not in otb_gens]
    if dropped:
        print(f"[otb] EXCLUDING from OTB training (test-panel overlap): {dropped}")
    print(f"[otb] OTB trained on Human + {len(otb_gens)} OLLMs: {otb_gens}")

    # English RAID humans (title -> text), encode once
    H = pd.read_parquet(args.humans)
    H["title"] = H["title"].astype(str); H["text"] = H["text"].astype(str)
    hum_titles = H["title"].tolist(); hum_texts = H["text"].tolist()
    print(f"[humans] {len(hum_texts)} English RAID humans; encoding...")
    Xhum_all = encode(hum_texts, args.encoder, device=dev)
    title2hum = {}
    for t, x in zip(hum_titles, Xhum_all):
        title2hum.setdefault(t, []).append(x)

    # OTB train AI (already encoded)
    z = np.load(args.otb_npz, allow_pickle=True)
    Xotb = z["X"].astype(np.float64)
    lab_otb = np.array([str(l) for l in z["labels"]])
    print(f"[otb] {len(Xotb)} train-AI rows, {len(set(lab_otb))} labels")

    SH, SA, WH = load_bundle(args.bundle, args.K)
    print(f"[bundle] S_H + {len(SA)} subspaces (whitened)")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    files = sorted(glob.glob(args.panels))
    res = {"GLASS": [], "OTB-supervised": []}
    res_tpr = {"GLASS": [], "OTB-supervised": []}
    for si, f in enumerate(files):
        rows = panel_titles_ai(f)
        titles = sorted({t for t, _, _ in rows})
        rng = np.random.default_rng(si)
        n_te = max(1, int(round(len(titles) * args.test_frac)))
        test_t = set(rng.choice(titles, size=n_te, replace=False))

        ai_te = [r for t, _, r in rows if t in test_t]
        hum_te_X = [x for t in test_t if t in title2hum for x in title2hum[t]]
        if len(ai_te) < 20 or len(hum_te_X) < 20:
            print(f"  [s{si}] too few test docs; skip"); continue
        Xai = encode(ai_te, args.encoder, device=dev)
        Xhu = np.array(hum_te_X)

        # ---- GLASS (frozen v3) ----
        s = np.concatenate([glass_score(Xai, SH, SA, WH), glass_score(Xhu, SH, SA, WH)])
        y = np.array([1] * len(Xai) + [0] * len(Xhu))
        a, t1 = auc_tpr(s, y)
        res["GLASS"].append(a); res_tpr["GLASS"].append(t1)

        # ---- OTB-supervised (train LR; humans = English RAID humans not in test titles) ----
        train_h_idx = [i for i, t in enumerate(hum_titles) if t not in test_t]
        rng2 = np.random.default_rng(1000 + si)
        if len(train_h_idx) > args.max_human_train:
            train_h_idx = rng2.choice(train_h_idx, args.max_human_train, replace=False)
        Xtr = [Xhum_all[i] for i in train_h_idx]
        ytr = [HUMAN_LABEL] * len(train_h_idx)
        for g in otb_gens:
            m = lab_otb == g
            Xg = Xotb[m]
            n = min(len(Xg), args.max_human_train)
            if len(Xg) > n:
                idx = rng2.choice(len(Xg), n, replace=False); Xg = Xg[idx]
            Xtr.extend(list(Xg)); ytr.extend([g] * len(Xg))
        Xtr = np.array(Xtr); ytr = np.array(ytr, dtype=object)
        scaler = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                                 n_jobs=-1, solver="lbfgs")
        clf.fit(scaler.transform(Xtr), ytr)
        classes = list(clf.classes_); hidx = classes.index(HUMAN_LABEL)
        Xte = np.vstack([Xai, Xhu])
        p_ai = 1.0 - clf.predict_proba(scaler.transform(Xte))[:, hidx]
        a, t1 = auc_tpr(p_ai, y)
        res["OTB-supervised"].append(a); res_tpr["OTB-supervised"].append(t1)
        print(f"  [s{si}] GLASS {res['GLASS'][-1]:.4f}  OTB {a:.4f}")

    def ms(v): return (np.mean(v), np.std(v)) if v else (float("nan"), 0.0)
    Path(args.out_tex).parent.mkdir(parents=True, exist_ok=True)
    L = ["% RE-DO symmetric comparison (English, v3_english). GLASS vs OTB-supervised.",
         "% PROFILER row pending profiler_on_our_data.py English run.",
         "\\begin{table}[t]\\centering\\small",
         "\\caption{Detector vs detector on the English frontier panels "
         "(\\texttt{v3\\_english}), identical held-out test docs. AUC and "
         "TPR@1\\%FPR, mean$\\pm$std over five panels.}",
         "\\label{tab:symmetric-v3}",
         "\\begin{tabular}{lcc}", "\\toprule",
         "Detector & AUC & TPR @ 1\\% FPR \\\\", "\\midrule"]
    for d in ["GLASS", "OTB-supervised"]:
        am, asd = ms(res[d]); tm, tsd = ms(res_tpr[d])
        name = "\\textbf{GLASS}" if d == "GLASS" else d
        L.append(f"{name} & {am:.3f}$\\pm${asd:.3f} & {tm:.3f}$\\pm${tsd:.3f} \\\\")
    L.append("PROFILER (6-LLM ensemble) & \\textit{pending} & \\textit{pending} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    Path(args.out_tex).write_text("\n".join(L) + "\n")
    print(f"\n[out] {args.out_tex}")
    for d in ["GLASS", "OTB-supervised"]:
        am, asd = ms(res[d]); tm, tsd = ms(res_tpr[d])
        print(f"  {d:16s} AUC {am:.4f}±{asd:.4f}  TPR@1 {tm:.4f}±{tsd:.4f}")


if __name__ == "__main__":
    main()
