# Experiments

Run commands from the repository root after installing `requirements.txt`.
Some baselines require a CUDA GPU and access to gated Hugging Face models.

## Inputs

The fitted bundle is included as `v3_english_clean.zip`. Inference instructions
are in [README.md](README.md).

Fitting and panel experiments additionally require:

- `results_v2/raid_en_humans.parquet`, generated with `python src/build_raid_english.py`.
- `results_v2/openrouter_en_s{0..4}/per_doc.jsonl`, the original generation panels.
  These panels are not included in this repository or its current release.
- Cached encodings and trained baseline files for experiments that use them;
  consult each script's command-line arguments.

`src/build_extra_panels.py` creates additional generations using
`OPENROUTER_API_KEY` and an explicit `--models` list. It does not reconstruct the
original five panels, and regenerated text is not an exact substitute.

The PROFILER wrappers require the absent `profiler_on_our_data.py` helper and an
external PROFILER checkout. The intrinsic-dimension and K-scaling experiments
(Tables 1–2 in the original manuscript) are also absent. This is therefore a
partial reproduction package.

## Commands

```bash
BUNDLE=.cache/glass_production/v3_english_clean
PANELS='results_v2/openrouter_en_s*/per_doc.jsonl'
HUM=results_v2/raid_en_humans.parquet

python src/build_subspace_v2_english.py --clean-text --panels "$PANELS" --humans "$HUM" --version v3_english_clean
python src/glass_eval_panel.py --clean-text --jsonl results_v2/openrouter_en_s0/per_doc.jsonl --humans "$HUM" --seed 0 --out results_v2/glass_en_clean_s0.json
python src/build_logo_detection_v3.py --clean-text --min-words 40 --panels "$PANELS" --humans "$HUM" --with-falcon --with-roberta --with-ghostbuster --out-tex tables/tab_logo_detection.tex --out-parquet results_v2/logo_detection_clean.parquet --out-parquet-domain results_v2/logo_detection_bydomain_clean.parquet
```

| Experiment | Script |
|---|---|
| Panel baselines | `competitor_eval_panel.py`, `tier2_eval_panel.py` |
| Cross-corpus detection | `build_thirdparty_competitors_v3.py` |
| Perturbations | `build_stress_v2.py`, `build_thirdparty_attack_v3.py` |
| Generator attribution | `build_frontier_en_corpus.py`, `make_long_cache.py`, `build_attribution_all_v3.py` |
| Ablations and calibration | `build_ablations_v3.py`, `build_calibration_v3.py`, `build_encoder_ablation_v3.py` |
| Timing and transfer | `build_wallclock_v3.py`, `build_transfer_directed_v3.py` |
| Evaluation checks | `build_paper_audit.py`, `build_paraphrase_audit.py`, `build_length_audit_v2.py`, `build_threshold_sweep.py` |

All scripts are under `src/`. `run_all.sh` records the original experiment
sequence and requires the additional inputs above. Check `--bundle` paths:
several scripts default to older bundle names.

## Evaluation notes

- Text cleaning uses a 40-word minimum for fitting; cross-corpus experiments
  use a 100-word minimum applied to every detector.
- The Pile split has a word-count-only AUC of 0.999 and should be excluded from
  detection aggregates.
- The original paraphrase scripts truncate machine-generated documents while
  leaving human documents intact. Those results cannot isolate paraphrase robustness.
- Length bins can contain different corpora. Analyse length effects within
  each corpus; the pooled and macro-averaged curves do not establish a
  short-text failure threshold.
- The four-corpus comparison does not establish a lead over OpenAI-large
  (mean AUC difference 0.004, 95% CI [-0.012, 0.020]).
- OpenTuringBench experiments use RAID human negatives, also the source used
  to fit the human subspace. Domain-specific LOGO results for wiki and poetry
  have strong word-count baselines.

The scripts retain the original computations; generated captions and historical
expected values should be interpreted with these limitations.
