#!/usr/bin/env bash
# =============================================================================
# GLASS — full reproduction, end to end.
#
#   bash run_all.sh 2>&1 | tee run_all.log
#
# Assumes: results_v2/openrouter_en_s{0..4}/per_doc.jsonl  (released generations)
#          results_v2/raid_en_humans.parquet               (src/build_raid_english.py)
# Everything else downloads from the HuggingFace Hub on first use.
#
# ~10-14 GPU-hours. Every step prints the numbers the paper reports.
# See REPRODUCE.md for the table-by-table map and expected values.
# =============================================================================
set -euo pipefail

BUNDLE=.cache/glass_production/v3_english_clean
PANELS='results_v2/openrouter_en_s*/per_doc.jsonl'
HUM=results_v2/raid_en_humans.parquet
PILE="Pile:artem9k/ai-text-detection-pile:text:source:ai:human"
OAIL="OpenAI-large:openai-community/roberta-large-openai-detector"

export PYTHONUTF8=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS=ignore PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# pick the GPU with the most free memory
if command -v nvidia-smi >/dev/null; then
  export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used \
      --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
  echo "### using GPU $CUDA_VISIBLE_DEVICES"
fi
mkdir -p tables figs results_v2

say() { echo; echo "############################################################"; echo "### $*"; echo "############################################################"; }

# ---------------------------------------------------------------- STAGE 1 ----
say "STAGE 1/4  Fit the bundle (everything depends on this)"
python3 src/build_subspace_v2_english.py --clean-text \
    --panels "$PANELS" --humans "$HUM" --version v3_english_clean
# expect: 18 per-generator subspaces + S_H(8000)

# ---------------------------------------------------------------- STAGE 2 ----
say "STAGE 2/4  Main results (Tables 6-17, Figures 5-6)"

echo "--- Tables 6 & 13: headline / strict-FPR  (expect GLASS AUC .9992, TPR .9737)"
for s in 0 1 2 3 4; do
  python3 src/glass_eval_panel.py --clean-text \
      --jsonl results_v2/openrouter_en_s${s}/per_doc.jsonl --humans "$HUM" \
      --seed ${s} --out results_v2/glass_en_clean_s${s}.json
done
python3 - <<'PY'
import json,glob,statistics as st
a=[json.load(open(f)) for f in sorted(glob.glob('results_v2/glass_en_clean_s*.json'))]
A=[x['auc'] for x in a]; T=[x['tpr_at_1'] for x in a]
print("RESULT[Tables 6,13] GLASS AUC %.4f+-%.4f  TPR@1%% %.4f+-%.4f"
      % (st.mean(A), st.pstdev(A), st.mean(T), st.pstdev(T)))
PY
# competitor rows of Table 6 (Fast-DetectGPT, log-rank, RoBERTa-OpenAI, then
# DetectGPT / DNA-GPT / Ghostbuster), then assemble tab_headline + tab_strict_fpr
python3 src/competitor_eval_panel.py
python3 src/tier2_eval_panel.py
python3 src/make_v2_tables.py

echo "--- Tables 7,8,9 + Figure 5: leave-one-generator-out  (expect GLASS .9988/.9951)"
python3 src/build_logo_detection_v3.py --clean-text --min-words 40 \
    --panels "$PANELS" --humans "$HUM" \
    --with-falcon --with-roberta --with-ghostbuster \
    --out-tex tables/tab_logo_detection.tex \
    --out-parquet results_v2/logo_detection_clean.parquet \
    --out-parquet-domain results_v2/logo_detection_bydomain_clean.parquet
python3 src/build_logo_boxplot.py \
    --overall results_v2/logo_detection_clean.parquet \
    --bydomain results_v2/logo_detection_bydomain_clean.parquet \
    --out figs/fig_logo_boxplot
echo "--- per-generator AUC spread (S9 benchmark-methodology claim)"
python3 - <<'PY'
import pandas as pd
d = pd.read_parquet('results_v2/logo_detection_clean.parquet')
for det in [c[:-4] for c in d.columns if c.endswith('_auc')]:
    g = d.groupby('generator')[f'{det}_auc'].mean()
    print(f"RESULT[spread] {det:16s} {g.min():.3f}-{g.max():.3f}  range {g.max()-g.min():.3f}")
PY

echo "--- Table 10: stress profile  (expect GLASS 1.000/1.000/0.991)"
python3 src/build_stress_v2.py --jsonl results_v2/openrouter_en_s0/per_doc.jsonl \
    --bundle "$BUNDLE" --humans "$HUM" --out-tex tables/tab_stress_profile.tex

echo "--- Table 11: cross-corpus, pass 1/2  (expect GLASS mean .848, worst .609)"
python3 src/build_thirdparty_competitors_v3.py --clean-text --min-words 100 \
    --bundle "$BUNDLE" --panels "$PANELS" \
    --corpora HC3,TuringBench,MAGE,NYT --hf-corpus "$PILE" \
    --with-falcon --with-roberta --with-radar --with-ghostbuster \
    --hf-detector "$OAIL" \
    --out-tex tables/tab_thirdparty.tex --out-parquet results_v2/thirdparty.parquet

echo "--- Table 11: cross-corpus, pass 2/2 (Binoculars alone: needs ~28GB)"
python3 src/build_thirdparty_competitors_v3.py --clean-text --min-words 100 \
    --bundle "$BUNDLE" --panels "$PANELS" \
    --corpora HC3,TuringBench,MAGE,NYT --hf-corpus "$PILE" \
    --with-falcon --with-binoculars \
    --out-parquet results_v2/thirdparty_binoc.parquet

echo "--- Table 12: attack, pass 1/2  (expect GLASS worst-drop -0.037)"
python3 src/build_thirdparty_attack_v3.py --clean-text --min-words 100 \
    --attacks clean,character,paraphrase --paraphraser humarin \
    --bundle "$BUNDLE" --panels "$PANELS" --humans "$HUM" \
    --corpora HC3,TuringBench,MAGE,NYT --hf-corpus "$PILE" \
    --with-falcon --with-roberta --with-radar --with-ghostbuster \
    --hf-detector "$OAIL" \
    --out-tex tables/tab_thirdparty_attack.tex --out-parquet results_v2/attack.parquet

echo "--- Table 12: attack, pass 2/2 (Binoculars alone)"
python3 src/build_thirdparty_attack_v3.py --clean-text --min-words 100 \
    --attacks clean,character,paraphrase --paraphraser humarin \
    --bundle "$BUNDLE" --panels "$PANELS" --humans "$HUM" \
    --corpora HC3,TuringBench,MAGE,NYT --hf-corpus "$PILE" \
    --with-falcon --with-binoculars \
    --out-parquet results_v2/attack_binoc.parquet

echo "--- Tables 14 & 16: symmetric  (expect GLASS .9998 / .9930)"
python3 src/build_symmetric_v3.py --bundle "$BUNDLE" \
    --panels "$PANELS" --humans "$HUM" --out-tex tables/tab_symmetric.tex

echo "--- Table 15: frozen bundle on OpenTuringBench  (expect .996)"
python3 src/build_glass_on_otb_v3.py --bundle "$BUNDLE" \
    --out-tex tables/tab_glass_on_otb.tex

echo "--- Table 17 + Figure 6: AUC vs document length  (expect GLASS .539 -> .999)"
python3 src/build_length_effect.py --bundle "$BUNDLE" \
    --panels "$PANELS" --humans "$HUM" \
    --with-falcon --with-binoculars --with-roberta --with-radar --with-ghostbuster \
    --hf-detector "$OAIL"

# ---------------------------------------------------------------- STAGE 3 ----
say "STAGE 3/4  Attribution (Appendix B)"
python3 src/build_frontier_en_corpus.py --clean-text --name frontier_en_clean
python3 src/make_long_cache.py --min-words 200
python3 src/build_attribution_all_v3.py --corpus frontier_en_long
# expect: N=4 GAS .546 / RoBERTa .646 / Ghostbuster .417
#         N=18 GAS .378 / RoBERTa .490 / Ghostbuster .144

# ---------------------------------------------------------------- STAGE 4 ----
say "STAGE 4/4  Appendix (ablations, calibration, cost, transfer)"
python3 src/build_ablations_v3.py --bundle "$BUNDLE" --panels "$PANELS" --humans "$HUM"
python3 src/build_calibration_v3.py --bundle "$BUNDLE" --panels "$PANELS" --humans "$HUM"
python3 src/build_encoder_ablation_v3.py
python3 src/build_wallclock_v3.py
python3 src/build_transfer_directed_v3.py
python3 src/build_attack_boxplot.py

say "DONE — tables/ and figs/ now hold every table and figure in the paper."
echo "Compare the RESULT[...] lines above against the expected values in REPRODUCE.md."
echo
echo "NOT run here (needs the PROFILER repo + gated HF models):"
echo "  python3 src/profiler_english_v3.py     --profiler-root <PROFILER_REPO>"
echo "  python3 src/profiler_attribution_v3.py --profiler-root <PROFILER_REPO> --corpus frontier_en_long"
