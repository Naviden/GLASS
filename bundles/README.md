# Fitted bundle

Download `v3_english_clean.zip` from this repository's **Releases** and unzip it here:

```
bundles/v3_english_clean/
    whitener.npz            ZCA whitener (mean, W)
    S_H_universal.npz       human subspace  (8000 RAID humans)
    subspaces/S_A_*.npz     18 per-generator AI subspaces
    meta.json               provenance: encoder, K, generators, build time
```

~20 MB. This is the exact bundle every number in the paper comes from.

To rebuild it from scratch instead (needs the frontier-EN generations and a GPU, ~15 min):

```bash
python3 src/build_subspace_v2_english.py --clean-text \
    --panels 'results_v2/openrouter_en_s*/per_doc.jsonl' \
    --humans results_v2/raid_en_humans.parquet \
    --version v3_english_clean
```
