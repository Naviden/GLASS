# GLASS

A closed-form, encoder-geometric detector for machine-generated text.

GLASS scores a document by comparing how far it sits from a *human* subspace against
how far it sits from the nearest *generator* subspace, in the embedding space of a
frozen sentence encoder:

```
s_det(x) = r(x; S_H) − min_g r(x; S_A_g)          s_det > 0  ⇒  machine-generated
```

Both subspaces are obtained in closed form by truncated SVD. **Scoring a document is
one encoder pass plus a few matrix–vector products** — no language-model inference, no
gradient step, no training at scoring time. It takes ~0.2 ms per document and runs
fine on a CPU.

Paper: *"GLASS: A Closed-Form Encoder-Geometric Detector for Machine-Generated Text"*
(Information Processing & Management, 2026).

---

## Use GLASS

No GPU. No training. No API key.

```bash
pip install torch transformers numpy        # that's all GLASS needs
# download the fitted bundle (~20 MB) from this repo's Releases into bundles/
```

```python
from glass import GLASS

g = GLASS.load("bundles/v3_english_clean")

g.is_ai("The proliferation of large language models has fundamentally ...")
# True

g.score(["document one ...", "document two ..."])
# array([ 41.7, -18.2 ])      # > 0 = machine-generated

g.explain("...")
# {'s_det': 41.7, 'is_ai': True, 'nearest_generator': 'openai/gpt-4o',
#  'n_words': 312, 'reliable': True, ...}
```

Or from the command line:

```bash
python glass.py --text "The quick brown fox ..."
python glass.py --file article.txt
cat article.txt | python glass.py
python glass.py --file 'docs/*.txt' --json
```

**Two things to know before you trust a score.**

* **GLASS is a document-level detector.** Accuracy is near chance below ~50 words and
  reaches ~0.999 beyond 400 (paper, Table 17). `explain()` returns
  `reliable: False` and a warning for short inputs — heed it. Do not use GLASS on
  sentence-length snippets.
* **Detection is not limited to the 18 generators in the library.** The score is the
  distance to the *nearest* known generator subspace, so text from models GLASS has
  never seen is still detected — that is the cross-corpus result the paper is about.
  The `nearest_generator` field is a hint, not an attribution claim.

---

## Reproduce the paper

This is the part that needs a GPU — **not because of GLASS, but because of the
baselines** we compare against (Falcon-7B for Fast-DetectGPT / log-rank / DNA-GPT,
two Falcon-7Bs for Binoculars, RADAR, RoBERTa detectors). GLASS itself is minutes.

**[`REPRODUCE.md`](REPRODUCE.md) maps every table and figure in the paper to exactly
one command, and states the number you should get.**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh 2>&1 | tee run_all.log        # ~10–14 GPU-hours end to end
```

Every reproduced value is printed as `RESULT[...]` so it can be diffed against the paper.

**Data.** Public corpora (HC3, TuringBench, MAGE, NYT, the AI-text-detection Pile,
OpenTuringBench) download from the HuggingFace Hub on first use. The **frontier-EN
generations** — 18 instruction-tuned models × five domain-stratified panels — ship as a
data artifact alongside this repo; regenerating them needs an OpenRouter key and will
*not* reproduce the paper's documents byte-for-byte (they were sampled at temperature
0.7), only its numbers.

**One bundle determines everything.** Every GLASS number in the paper comes from
`v3_english_clean`, fit once in Stage 1: 18 per-generator subspaces plus one human
subspace, fit on junk-filtered data. Build that and the rest is determined.

Two text-quality thresholds appear and are **not interchangeable**:

| | |
|---|---|
| **min 40 words** | *fitting* — data hygiene; drops corrupted/boilerplate/sub-sentence fragments while keeping all eight RAID domains |
| **min 100 words** | *cross-corpus evaluation only* — a deployment-regime document filter, fixed in advance, applied identically to **every** detector. It is a property of the evaluation set, not of GLASS. What happens *below* it is not hidden: Table 17 / Figure 6 sweep length down to 25 words. |

---

## Layout

```
glass.py         use GLASS (inference; CPU, no training)   <- start here
REPRODUCE.md     table/figure -> command -> expected value  <- reproduction
run_all.sh       full pipeline, end to end
src/             all 37 scripts behind the paper
bundles/         fitted bundle (download from Releases)
```

## Incomplete — do not publish yet

`REPRODUCE.md` flags two files that must be added before release: `profiler_on_our_data.py`,
and the script behind Tables 1–2 (intrinsic-dimension gap, K-scaling law).
