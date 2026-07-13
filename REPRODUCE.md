# Reproducing every number in the GLASS paper

Every table and figure in the paper maps to exactly one command below. All GLASS
results come from a **single fitted bundle**, `v3_english_clean`, built in Stage 1.

Conventions used throughout:

```bash
BUNDLE=.cache/glass_production/v3_english_clean
PANELS='results_v2/openrouter_en_s*/per_doc.jsonl'
HUM=results_v2/raid_en_humans.parquet
export PYTHONUTF8=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0        # pin a free GPU
```

Two text-quality thresholds appear below and are **not** interchangeable:

* **`--clean-text` with `--min-words 40`** — used when *fitting* (data hygiene: drops
  corrupted / boilerplate / sub-sentence fragments while keeping all eight RAID domains).
* **`--clean-text --min-words 100`** — used only for the *cross-corpus evaluation*
  (a deployment-regime document filter, fixed in advance, applied identically to every
  detector). §9 / Table 17 reports what happens below that threshold.

---

## Stage 0 — Data

| Artifact | Command |
|---|---|
| English RAID humans + prompts | `python3 src/build_raid_english.py` |
| Domain-stratified prompt panels | `python3 src/stratified_prompt_sampling.py` |
| Frontier-EN generations (18 models) | `python3 src/build_extra_panels.py` (needs `OPENROUTER_API_KEY`) |

> **Note.** Re-running the generation step will *not* reproduce the paper's documents
> byte-for-byte (temperature 0.7 sampling, and provider models drift). To reproduce the
> paper's *numbers*, use the released `results_v2/openrouter_en_s{0..4}/per_doc.jsonl`
> generations, which are published alongside this package. Regenerate only if you want
> a fresh panel.

## Stage 1 — Fit the bundle (everything downstream depends on this)

```bash
python3 src/build_subspace_v2_english.py --clean-text \
    --panels "$PANELS" --humans "$HUM" --version v3_english_clean
```
Produces `.cache/glass_production/v3_english_clean/`: the ZCA whitener, the human
subspace `S_H` (8000 RAID humans), and 18 per-generator subspaces. ~15 min on one GPU.

Expected log: `dropped 423 junk AI docs`, `dropped 130 junk human docs`,
`S_H (8000 humans) + 18 English per-generator subspaces`.

---

## Stage 2 — Main results (§8)

### Tables 6 & 13 — headline detection, strict-FPR
```bash
for s in 0 1 2 3 4; do
  python3 src/glass_eval_panel.py --clean-text \
      --jsonl results_v2/openrouter_en_s${s}/per_doc.jsonl --humans "$HUM" \
      --seed ${s} --out results_v2/glass_en_clean_s${s}.json
done
python3 src/competitor_eval_panel.py     # Fast-DetectGPT, log-rank, RoBERTa-OpenAI
python3 src/tier2_eval_panel.py          # DetectGPT, DNA-GPT, Ghostbuster
python3 src/make_v2_tables.py            # -> tab_headline, tab_strict_fpr
```
**Expected:** GLASS AUC `0.9992 ± 0.0014`, TPR@1%FPR `0.9737 ± 0.0477`.

### Tables 7, 8, 9 + Figure 5 — leave-one-generator-out
```bash
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
```
**Expected:** GLASS `0.9988 / 0.9951`, Fast-DetectGPT `0.9921 / 0.9222`,
Ghostbuster `0.9633 / 0.6345`, log-rank `0.8357 / 0.1065`, RoBERTa `0.7639 / 0.2527`.
GLASS is best AUC in every domain; Fast-DetectGPT edges it on Books TPR by 0.002.

The **per-generator spread** quoted in §9 (benchmark methodology) comes from the same
parquet:
```bash
python3 - <<'PY'
import pandas as pd
d = pd.read_parquet('results_v2/logo_detection_clean.parquet')
for det in [c[:-4] for c in d.columns if c.endswith('_auc')]:
    g = d.groupby('generator')[f'{det}_auc'].mean()
    print(f"{det:16s} {g.min():.3f}-{g.max():.3f}  range {g.max()-g.min():.3f}")
PY
```
**Expected:** GLASS range `0.003`; RoBERTa-OpenAI `0.404`; log-rank `0.182`; Ghostbuster `0.103`.

### Table 10 — in-library stress profile
```bash
python3 src/build_stress_v2.py \
    --jsonl results_v2/openrouter_en_s0/per_doc.jsonl \
    --bundle "$BUNDLE" --humans "$HUM" \
    --out-tex tables/tab_stress_profile.tex
```
**Expected:** GLASS `1.000 / 1.000 / 0.991`; Fast-DetectGPT `0.993 / 0.402 / 0.863`.

### Table 11 — cross-corpus detection (the headline OOD result)
Run in **two passes**: Binoculars holds two Falcon-7Bs (~28 GB) and cannot coexist with
RADAR on one GPU.
```bash
# pass 1 — seven detectors
python3 src/build_thirdparty_competitors_v3.py --clean-text --min-words 100 \
    --bundle "$BUNDLE" --panels "$PANELS" \
    --corpora HC3,TuringBench,MAGE,NYT \
    --hf-corpus "Pile:artem9k/ai-text-detection-pile:text:source:ai:human" \
    --with-falcon --with-roberta --with-radar --with-ghostbuster \
    --hf-detector "OpenAI-large:openai-community/roberta-large-openai-detector" \
    --out-tex tables/tab_thirdparty.tex --out-parquet results_v2/thirdparty.parquet

# pass 2 — Binoculars alone
python3 src/build_thirdparty_competitors_v3.py --clean-text --min-words 100 \
    --bundle "$BUNDLE" --panels "$PANELS" \
    --corpora HC3,TuringBench,MAGE,NYT \
    --hf-corpus "Pile:artem9k/ai-text-detection-pile:text:source:ai:human" \
    --with-falcon --with-binoculars \
    --out-parquet results_v2/thirdparty_binoc.parquet
```
**Expected (GLASS):** HC3 `0.992`, TuringBench `0.707`, MAGE `0.609`, NYT `0.995`,
Pile `0.935` → **mean `0.848`, worst-case `0.609` — best on both of any detector.**

### Table 12 — cross-corpus under attack
Same two-pass structure.
```bash
python3 src/build_thirdparty_attack_v3.py --clean-text --min-words 100 \
    --attacks clean,character,paraphrase --paraphraser humarin \
    --bundle "$BUNDLE" --panels "$PANELS" --humans "$HUM" \
    --corpora HC3,TuringBench,MAGE,NYT \
    --hf-corpus "Pile:artem9k/ai-text-detection-pile:text:source:ai:human" \
    --with-falcon --with-roberta --with-radar --with-ghostbuster \
    --hf-detector "OpenAI-large:openai-community/roberta-large-openai-detector" \
    --out-tex tables/tab_thirdparty_attack.tex --out-parquet results_v2/attack.parquet

python3 src/build_thirdparty_attack_v3.py --clean-text --min-words 100 \
    --attacks clean,character,paraphrase --paraphraser humarin \
    --bundle "$BUNDLE" --panels "$PANELS" --humans "$HUM" \
    --corpora HC3,TuringBench,MAGE,NYT \
    --hf-corpus "Pile:artem9k/ai-text-detection-pile:text:source:ai:human" \
    --with-falcon --with-binoculars \
    --out-parquet results_v2/attack_binoc.parquet
```
**Expected (GLASS):** clean `0.848`, char-edit `0.842`, paraphrase `0.811`,
**worst drop `-0.037`** (next-best `-0.141`). GLASS *loses* the paraphrase column
(7th of 8) — this is reported, not hidden.

### Tables 14 & 16 — symmetric comparison / OTB overlap
```bash
python3 src/build_symmetric_v3.py --bundle "$BUNDLE" \
    --panels "$PANELS" --humans "$HUM" --out-tex tables/tab_symmetric.tex
```
**Expected:** GLASS `0.9998 ± 0.0004` / TPR `0.9930`; OTB (7 gen) `0.9676 / 0.8532`.

### Table 15 — frozen bundle on OpenTuringBench
```bash
python3 src/build_glass_on_otb_v3.py --bundle "$BUNDLE" \
    --out-tex tables/tab_glass_on_otb.tex
```
**Expected:** in-domain `0.996`, unseen-model `0.996`, human-continue `0.920`.

### Table 17 + Figure 6 — detection AUC vs document length
```bash
python3 src/build_length_effect.py --bundle "$BUNDLE" \
    --panels "$PANELS" --humans "$HUM" \
    --with-falcon --with-binoculars --with-roberta --with-radar --with-ghostbuster \
    --hf-detector "OpenAI-large:openai-community/roberta-large-openai-detector"
```
**Expected (GLASS, macro-averaged):** `0.539` (25–50 w) → `0.716` → `0.748` → `0.833`
→ **`0.999`** (≥400 w). Supervised detectors invert (RoBERTa `0.683`, OpenAI-large
`0.586` at ≥400). Writes `figs/fig_length_effect.pdf`, `tables/tab_length_effect.tex`,
and `results_v2/length_effect_perdoc.parquet` (per-document scores, for re-slicing
without re-scoring).

> Use the **macro-averaged** table, not the pooled one. A naive pooled average conflates
> length with which corpus dominates each bin; the script prints both so the difference
> is auditable.

---

## Stage 3 — Attribution (Appendix B)

```bash
# clean attribution cache (18 generators), then restrict to >= 200-word documents
python3 src/build_frontier_en_corpus.py --clean-text --name frontier_en_clean
python3 src/make_long_cache.py --min-words 200      # -> frontier_en_long

python3 src/build_attribution_all_v3.py --corpus frontier_en_long
```
**Expected:** N=4 → GAS `0.546`, RoBERTa `0.646`, Ghostbuster `0.417` (chance 0.250).
N=18 → GAS `0.378`, RoBERTa `0.490`, Ghostbuster `0.144` (chance 0.056).

All three methods share **one** stratified split — this is why a single unified table
replaces the earlier separate ones, which disagreed with each other on GAS.

Attribution is restricted to ≥200-word documents on purpose: the geometric score is
length-sensitive (Table 17), so attributing over random-length text would confound
generator identity with document length.

---

## Stage 4 — Appendix

| Table | Command |
|---|---|
| K-ablation, library redundancy | `python3 src/build_ablations_v3.py --bundle "$BUNDLE" --panels "$PANELS" --humans "$HUM"` |
| Calibration (Brier / ECE) | `python3 src/build_calibration_v3.py --bundle "$BUNDLE" --panels "$PANELS" --humans "$HUM"` |
| Encoder ablation | `python3 src/build_encoder_ablation_v3.py` |
| Wall-clock cost | `python3 src/build_wallclock_v3.py` |
| Directed transfer + principal angles | `python3 src/build_transfer_directed_v3.py` |
| Attack boxplot figure | `python3 src/build_attack_boxplot.py` |
| PROFILER row (symmetric table) | `python3 src/profiler_english_v3.py --profiler-root <PROFILER_REPO>` |
| GAS vs PROFILER attribution | `python3 src/profiler_attribution_v3.py --profiler-root <PROFILER_REPO> --corpus frontier_en_long` |

**Expected:** K=10 within ΔAUC ≤ 0.001 of best K (2–128); detection holds with 12 of 18
generators removed; Isotonic Brier `0.0022` / ECE `0.0026`.

---

## Supporting analyses (not tables, but cited in the text)

| Analysis | Command |
|---|---|
| Which genres GLASS fails on (ROCStories inversion) | `python3 src/build_glass_crossdomain.py --bundle "$BUNDLE" --clean-text` |
| False-positive audit (what the junk actually is) | `python3 src/build_fp_analysis.py --bundle "$BUNDLE"` |

---

## Known gaps in this package — you must add two files

1. **`profiler_on_our_data.py`** — imported by `profiler_english_v3.py` and
   `profiler_attribution_v3.py` (provides `extract_features_for_llm`,
   `PROFILER_BASE_MODELS`). It lives in the working tree, not here. Copy it into `src/`.
2. **Tables 1–2 (intrinsic-dimension gap, K-scaling law)** are produced by the upstream
   geometry pipeline, which is not in this package. Add that script to `src/` and list
   it here before release.

Both are flagged rather than silently omitted. **Do not publish until they are in.**
