# GLASS

Code and fitted subspaces for detecting machine-generated text with a frozen encoder.

## Usage

Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy torch transformers sentencepiece
python -m zipfile -e v3_english_clean.zip bundles
python glass.py --file article.txt --device cpu --json
```

The encoder is downloaded from Hugging Face on first use.

```python
from glass import GLASS

detector = GLASS.load("bundles/v3_english_clean", device="cpu")
score = detector.score("Text to evaluate.")
```

Positive scores indicate machine-generated text at the default threshold.
The nearest-generator output identifies a subspace, not the source of a document.
Scores for inputs below 100 words are unvalidated; the `reliable` field is only a
length check, not a confidence estimate.

## Experiments

See [REPRODUCE.md](REPRODUCE.md) for commands, required inputs, and evaluation limitations.
Experiment dependencies are listed in `requirements.txt`; scripts are in `src/`.

## Licence

[MIT](LICENSE).
