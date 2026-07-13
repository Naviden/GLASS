# GLASS — reproducibility package

Code and instructions to reproduce every table and figure in
*"GLASS: A Closed-Form Encoder-Geometric Detector for Machine-Generated Text."*

GLASS scores a document by the difference between its squared residual to a human
subspace and the smallest squared residual to a library of per-generator AI subspaces,
all obtained in closed form by truncated SVD on a frozen sentence encoder:

```
s_det(x) = r(x; S_H) − min_g r(x; S_A_g)
```

No language-model inference, no gradient step, no training at scoring time.

**Start here: [`REPRODUCE.md`](REPRODUCE.md)** — every table and figure in the paper
maps to exactly one command, with the expected numbers next to it.

---

## What you need

**Hardware.** One CUDA GPU. Most steps fit in 24 GB. Two steps need ~28 GB because they
hold two Falcon-7B models at once (Binoculars: observer + performer) — on a 24 GB card,
run those as the separate second pass described in `REPRODUCE.md`. A full reproduction is
roughly **10–14 GPU-hours**, dominated by the Falcon-7B and RADAR passes over the
cross-corpus sets; GLASS itself is a few minutes.

**Software.**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Data.** Public corpora (HC3, TuringBench, MAGE, NYT, the AI-text-detection Pile,
OpenTuringBench) download automatically from the HuggingFace Hub on first use. RAID is
fetched by `src/build_raid_english.py`. The **frontier-EN generations** — 18
instruction-tuned models × five domain-stratified panels — are released as a data
artifact alongside this package (`results_v2/openrouter_en_s{0..4}/per_doc.jsonl`);
re-generating them requires an OpenRouter key and will not reproduce the paper's
documents byte-for-byte, since they were sampled at temperature 0.7.

Some HuggingFace models are gated (Llama, Gemma — used by the PROFILER baseline only).
Run `huggingface-cli login` if you want that row.

---

## Layout

```
src/            all scripts (37)
REPRODUCE.md    table/figure -> command, with expected values   <- the important file
run_all.sh      runs the whole pipeline end to end
requirements.txt
tables/         generated .tex  (created by the runs)
figs/           generated .pdf  (created by the runs)
```

Generated at run time (not shipped): `.cache/glass_production/` (fitted bundles),
`.cache/pipeline/` (attribution embeddings), `results_v2/` (per-run parquet/JSON).

---

## The one thing that matters for reproducibility

Every GLASS number in the paper comes from **one fitted bundle**, built once in Stage 1:

```
.cache/glass_production/v3_english_clean
```

18 per-generator subspaces + one human subspace, fit on junk-filtered data. If you build
this bundle and nothing else changes, every downstream number is determined.

Two text-quality thresholds appear, and conflating them will give you the wrong numbers:

* **min 40 words — fitting.** Data hygiene: removes corrupted, boilerplate and
  sub-sentence fragments while keeping all eight RAID domains intact.
* **min 100 words — cross-corpus evaluation only.** A deployment-regime document filter,
  fixed in advance and applied **identically to every detector**. It is a property of the
  evaluation set, not of GLASS. What happens *below* that threshold is not hidden: it is
  reported explicitly in Table 17 / Figure 6, which sweep document length down to 25 words.

---

## Honest notes

These are in the paper and repeated here so nobody is surprised by a number:

* **GLASS wins no individual cross-corpus benchmark.** It is first on the aggregate
  (best mean *and* best worst-case AUC of eight detectors) because it is the only method
  with no blind spot; every competitor falls below 0.49 on at least one corpus.
* **GLASS loses the paraphrase column** (0.811, seventh of eight). A paraphraser is a
  generator with no subspace in the library, so it pulls text toward the human subspace.
  The cross-corpus paraphrase result is also *confounded* in the opposite direction —
  re-generating AI text through a T5 model makes it **more** detectable to token-based
  methods, which is why their AUC rises. It is not a clean evasion test.
* **Accuracy scales with length.** Near chance below ~50 words, 0.999 beyond 400. GLASS
  is a document-level detector; we recommend against its use on sentence-length text.
* **Attribution is a byproduct, not a contribution.** GAS beats a Ghostbuster-style
  attributor 2.6× but trails a fine-tuned RoBERTa classifier — the expected
  training-free/trained trade-off.

## Incomplete — do not publish yet

`REPRODUCE.md` lists two files that are **not** in this package and must be added before
release: `profiler_on_our_data.py`, and the script behind Tables 1–2 (intrinsic-dimension
gap and K-scaling law). They are flagged rather than quietly omitted.

## Citation

```bibtex
@article{glass2026,
  title   = {{GLASS}: A Closed-Form Encoder-Geometric Detector for Machine-Generated Text},
  author  = {De Santo, Alessia and Mercorio, Fabio and Nobani, Navid and Pallucchini, Filippo},
  journal = {Information Processing \& Management},
  year    = {2026}
}
```
