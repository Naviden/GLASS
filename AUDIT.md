# Audit log

This file records errors we found in our **own** results and interpretations, and what
we did about them. It is kept in the repository deliberately: several of these were
claims we had already written into the paper, and a reader deserves to see the
corrections rather than a silently cleaned-up final version.

Every audit below is reproducible from a script in `src/`. No number in this file is
asserted without one.

---

## 1. A length–label confound runs through the cross-corpus benchmarks

`src/build_paper_audit.py`, `src/build_length_audit_v2.py`

We scored every evaluation set with a "detector" that reads **nothing but the document's
word count**. On a benchmark that measures provenance, this should sit at chance.

| corpus | word-count-only AUC, all docs | word-count-only AUC, ≥100 words |
|---|---|---|
| HC3 | 0.823 | 0.501 |
| MAGE | 0.861 | 0.595 |
| NYT | 0.523 | 0.553 |
| TuringBench | 0.603 | 0.647 |
| **Pile** | **1.000** | **0.999** |

**The Pile split is not a detection benchmark.** Its AI documents are short and its human
documents are long, nearly disjointly. A rule that says *"under 200 words ⇒ AI"* beats
every one of the eight detectors we tested on it (0.999 vs best-real 0.996). No number
measured on Pile — including GLASS's — carries information about detecting machine text.
It is **excluded from the headline cross-corpus table**, and reported separately as a
benchmark-integrity finding.

Consequences:

* The `≥100`-word filter is **not** a deployment-regime preference, which is how we
  originally justified it. It is the operation that **removes the confound** (HC3
  0.823 → 0.501; MAGE 0.861 → 0.595). After filtering *and* dropping Pile, word count
  scores exactly **0.500** across the remaining four corpora — the benchmark is clean.
* GLASS's own MAGE AUC *falls* from 0.758 to 0.610 under the filter. GLASS was reading
  length too. So was every detector. Removing the shortcut makes the benchmark harder
  for all of them.
* The detectors that scored best on Pile are the ones that most resemble a length rule
  (word-count 0.999, Binoculars 0.998, log-rank 0.992). GLASS scored **0.939 — the
  lowest of the training-free group — precisely because it does not read length.**

**The flagship leave-one-generator-out result is *not* affected.** LOGO AUC is computed
on a pool where word count scores **0.505** (chance). We checked, because the same failure
would have invalidated it.

## 2. Our own paraphrase attack truncated the AI side

`src/build_paraphrase_audit.py`

Both `build_thirdparty_attack_v3.py` and `build_stress_v2.py` paraphrased with
`truncation=True, max_length=256` on the input **and** `max_length=256` on the output,
while leaving human documents clean and full length. The AI side was therefore not just
paraphrased but **truncated to ~190 words**, manufacturing the very length confound
described above.

This is the only explanation consistent with the published numbers: detectors do not get
*better* under attack, yet RADAR went 0.796 → 0.996, OpenAI-large 0.793 → 0.979 and
RoBERTa-OpenAI 0.714 → 0.952 under "paraphrase". They were rewarded for a
"shorter ⇒ AI" shortcut we introduced ourselves.

GLASS moved 0.826 → 0.803, because it does not read length.

**Therefore the limitation "GLASS loses the paraphrase attack" — which we had written
into the paper, the abstract and this README — is an artifact of our own attack code, not
a property of GLASS.** The attack is being re-run length-preserving (sliding-window
paraphrase, nothing truncated) before any paraphrase claim is made in either direction.

## 3. Cross-corpus length "trends" were composition artifacts

`src/build_length_audit_v2.py`

Our length-effect table macro-averaged per-corpus AUCs within each length bin and claimed
this controlled for corpus composition. It did not: a corpus with too few documents in a
bin is silently dropped from that bin's average, and in this benchmark suite **document
length is nearly collinear with corpus** (HC3 has no documents ≥400 words; NYT has almost
none <200; Pile has 16 AI documents ≥200). The bins therefore averaged over *different*
corpora — the 25–50-word bin was MAGE alone.

Two published claims died:

* **"Supervised detectors collapse on long documents due to 512-token truncation."**
  False. *Within* corpus they **improve** with length (OpenAI-large: HC3 0.974 → 0.997;
  MAGE 0.595 → 0.936). The apparent collapse was NYT — where OpenAI-large is weak at
  *every* length — dominating the long bins. The mechanism was invented to fit an artifact.
* **"GLASS is near chance below ~50 words."** Unsupported. The only sub-50-word evidence
  in the suite is MAGE, where GLASS scores 0.49–0.63 at *every* length. On HC3 at 50–100
  words GLASS scores **0.937**. GLASS's weakness is MAGE, not short text.

What survives, measured **within** corpus and therefore composition-safe: GLASS's AUC
rises with document length in **all four** corpora (Kendall τ > 0 in each).

## 4. The headline comparison is a statistical tie, not a win

`src/build_length_audit_v2.py`, bootstrap over documents

On the confound-free four-corpus benchmark, GLASS vs OpenAI-large:
mean AUC difference **+0.004, 95% CI [−0.012, +0.020]**. That is a tie, not a win. Any
claim of "best mean *and* best worst-case detector" is not supported and has been removed.

What **is** supported, with large and significant margins, is that GLASS dominates every
**training-free** detector — the class it belongs to:

| comparison | mean AUC Δ | worst-case AUC Δ | P(GLASS better) |
|---|---|---|---|
| vs Fast-DetectGPT | +0.020 [+0.003, +0.039] | **+0.189 [+0.122, +0.258]** | 1.000 |
| vs Binoculars | +0.026 [+0.009, +0.044] | **+0.127 [+0.063, +0.192]** | 1.000 |

and that it *matches* trained classifiers (OpenAI-large, RADAR) while doing no training,
no LM inference, ~0.2 ms per document, on a CPU.

## 5. What held up

Audited and unchanged:

* **Leave-one-generator-out detection.** GLASS beats every baseline on all 90 paired
  measurements (18 generators × 5 seeds), Wilcoxon p ≤ 1.5e-13 throughout;
  TPR@1%FPR +0.073 over Fast-DetectGPT, +0.361 over Ghostbuster, +0.889 over log-rank.
  The evaluation pool has word-count-only AUC 0.505 — this result is about provenance.
* **Per-domain LOGO tables.** 25 cells missing (all in `recipes`, uniformly across every
  detector); restricting to the complete sub-grid moves the reported means by 0.0000.
  *Caveat now stated:* within `wiki` (0.927) and `poetry` (0.902) a length rule is
  strong, so those two per-domain cells carry an available shortcut — GLASS scores ~0.999
  there, well above it, but readers should know.
* **Character-edit robustness.** GLASS is #1 on worst-case drop with or without Pile
  (drop 0.023; next best 0.046).

---

## Open

* Paraphrase attack re-run, length-preserving (in progress).
* Tables 1–2 (intrinsic-dimension gap, K-scaling law) have **no script in this
  repository** and are therefore not currently reproducible. They must be either
  reproduced or removed.
* OTB evaluation scores AI against RAID humans while `S_H` was fit on RAID humans —
  an asymmetric bias that must be stated, not just caveated.
