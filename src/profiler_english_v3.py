#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate PROFILER detection on English panels.

Requires profiler_on_our_data.py and an external PROFILER checkout."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, ".")
from profiler_on_our_data import (              # reuse the tested functions
    load_profiler_train, extract_features_for_llm,
    PROFILER_BASE_MODELS, _is_human)


def load_frontier_en_test(cache: Path, n_per_class: int, seed: int = 42):
    info = json.loads((cache / "corpus_frontier_en.json").read_text())
    y = np.asarray(info["labels"], int); texts = info["texts"]
    rng = np.random.default_rng(seed)
    ai = np.where(y == 1)[0]; hu = np.where(y == 0)[0]
    ai = rng.choice(ai, min(n_per_class, len(ai)), replace=False)
    hu = rng.choice(hu, min(n_per_class, len(hu)), replace=False)
    idx = np.concatenate([ai, hu])
    return [texts[int(i)] for i in idx], \
           ["AI" if y[int(i)] == 1 else "Human" for i in idx]


def main():
    here = Path(__file__).resolve().parent
    root = here if (here / "results").exists() else here.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiler-root", required=True, type=Path)
    ap.add_argument("--cache", default=str(root / ".cache" / "pipeline"))
    ap.add_argument("--train-domain", default="Code")
    ap.add_argument("--n-train-per-source", type=int, default=500)
    ap.add_argument("--n-test-per-class", type=int, default=500)
    ap.add_argument("--feature-cache", default="results_v2/profiler_features_en")
    ap.add_argument("--sample-clip", type=int, default=4000)
    ap.add_argument("--context-window", type=int, default=6)
    ap.add_argument("--rf-n-estimators", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cache = Path(args.cache)
    feat_dir = Path(args.feature_cache); feat_dir.mkdir(parents=True, exist_ok=True)
    pargs = SimpleNamespace(sample_clip=args.sample_clip,
                            context_window=args.context_window)

    print("[1/3] load text")
    tr_texts, tr_labels = load_profiler_train(
        args.profiler_root, args.train_domain, args.n_train_per_source, args.seed)
    te_texts, te_labels = load_frontier_en_test(cache, args.n_test_per_class, args.seed)
    print(f"  train {len(tr_texts)} (PROFILER {args.train_domain}), "
          f"test {len(te_texts)} (frontier_en: "
          f"{sum(l=='AI' for l in te_labels)} AI / {sum(l=='Human' for l in te_labels)} H)")

    print("[2/3] 6-LLM feature extraction (resumable)")
    per = {}
    for bm in PROFILER_BASE_MODELS:
        print(f"  ===== {bm} =====")
        ftr = extract_features_for_llm(args.profiler_root, bm, tr_texts, pargs,
                                       feat_dir / f"{bm}_train_{args.train_domain}.pkl")
        if ftr is None:
            print(f"  [DROP] {bm}"); continue
        fte = extract_features_for_llm(args.profiler_root, bm, te_texts, pargs,
                                       feat_dir / f"{bm}_test_frontier_en.pkl")
        if fte is None:
            print(f"  [DROP] {bm} (test)"); continue
        per[bm] = (ftr, fte)
    if not per:
        raise SystemExit("no LLM produced features (check HF auth for gated repos)")
    avail = list(per)
    print(f"  ensemble: {len(avail)}/{len(PROFILER_BASE_MODELS)} -> {avail}")

    def san(X):
        return np.clip(np.nan_to_num(np.asarray(X, float), nan=0.0,
                                     posinf=1e6, neginf=-1e6), -1e6, 1e6)
    Xtr = san(np.hstack([per[m][0] for m in avail]))
    Xte = san(np.hstack([per[m][1] for m in avail]))

    print("[3/3] train RF (human-vs-rest), score frontier_en")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, roc_curve
    ytr = np.array([1 if s == "human" else 0 for s in tr_labels])  # 1 = human
    clf = RandomForestClassifier(n_estimators=args.rf_n_estimators,
                                 random_state=args.seed, n_jobs=-1,
                                 class_weight="balanced").fit(Xtr, ytr)
    p_ai = 1.0 - clf.predict_proba(Xte)[:, 1]
    is_ai = np.array([0 if _is_human(l) else 1 for l in te_labels])
    auc = float(roc_auc_score(is_ai, p_ai))
    fpr, tpr, _ = roc_curve(is_ai, p_ai); tpr1 = float(np.interp(0.01, fpr, tpr))
    print(f"\n[PROFILER on frontier_en]  AUC={auc:.4f}  TPR@1%={tpr1:.4f}")

    out = {"detector": "PROFILER (6-LLM ensemble)", "corpus": "frontier_en",
           "auc": auc, "tpr_at_1": tpr1, "llms_used": avail,
           "n_test": len(te_texts)}
    Path("results_v2").mkdir(exist_ok=True)
    json.dump(out, open("results_v2/profiler_english.json", "w"), indent=2)
    Path("tables_v2").mkdir(exist_ok=True)
    Path("tables_v2/tab_symmetric_v3_PROFILER.txt").write_text(
        f"PROFILER (6-LLM ensemble) & {auc:.3f} & {tpr1:.3f} \\\\\n")
    print("[out] results_v2/profiler_english.json + "
          "tables_v2/tab_symmetric_v3_PROFILER.txt (paste this row into tab_symmetric_v3)")


if __name__ == "__main__":
    main()
