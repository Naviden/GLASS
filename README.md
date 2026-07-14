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

* **GLASS is a document-level detector.** Its AUC rises with document length inside every
  corpus we tested, so longer inputs are more reliable; `explain()` warns below 100 words.
  We previously claimed accuracy was "near chance below ~50 words" — **that claim was
  wrong** and has been withdrawn (see [`AUDIT.md`](AUDIT.md) §3). Short-text behaviour is
  being re-measured; until it is, treat sub-100-word scores as unvalidated rather than
  as known-bad.
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

## What GLASS does not do

Stated here as plainly as in the paper, so no number surprises you. Several entries here
are corrections to claims we previously made ourselves — see [`AUDIT.md`](AUDIT.md).

* **It wins no individual cross-corpus benchmark**, and on the aggregate it is **tied**
  with a fine-tuned RoBERTa classifier, not ahead of it (mean AUC difference +0.004, 95%
  CI [−0.012, +0.020]). We previously claimed "best mean *and* best worst-case of eight
  detectors"; that is **not statistically supported** and has been withdrawn.
* **What it does do** is dominate every *training-free* detector — the class it belongs
  to — by large, significant margins (worst-case AUC +0.189 over Fast-DetectGPT and
  +0.127 over Binoculars, both P = 1.000), while matching trained classifiers at ~0.2 ms
  per document on a CPU.
* **Its weak corpus is MAGE** (AUC ≈ 0.61), at *every* document length. We previously
  described the weakness as "short or templated text" — that was **wrong**, an artifact
  of averaging over corpora with different length distributions.
* **The paraphrase result is under re-evaluation.** Our published paraphrase attack
  truncated the AI side to ~190 words while leaving humans full length, which rewarded
  detectors that read document length. Until the length-preserving re-run lands, we make
  **no claim** about paraphrase robustness in either direction. See [`AUDIT.md`](AUDIT.md) §2.
* **Attribution is a byproduct, not a contribution.** GAS beats a Ghostbuster-style
  attributor 2.6× but trails a fine-tuned RoBERTa classifier — the expected
  training-free/trained trade-off.

## Benchmark integrity

While auditing our own results we found that **one widely-used cross-corpus split is
separable by document length alone** (word-count-only AUC **0.999**), meaning it cannot
discriminate detectors at all. We report the word-count baseline for every benchmark we
use, and recommend others do the same — it costs nothing and it caught a real problem.
Details and numbers in [`AUDIT.md`](AUDIT.md).
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
