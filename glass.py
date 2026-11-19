#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GLASS inference with a frozen encoder and fitted subspaces.

The score is the squared residual to the human subspace minus the minimum
squared residual to a generator subspace. Positive values indicate machine-
generated text at the default threshold. See README.md for installation and usage.
"""
from __future__ import annotations
import argparse, glob as _glob, json, sys
from pathlib import Path
import numpy as np

MIN_RELIABLE_WORDS = 100          # length check, not a confidence estimate


def _residual_sq(X, mean, B):
    """Squared residual of rows of X to the affine subspace (mean, span(B))."""
    Xc = X - mean
    return np.sum((Xc - (Xc @ B) @ B.T) ** 2, axis=1)


class GLASS:
    """A fitted GLASS detector: a whitener, a human subspace, and a library of
    per-generator AI subspaces. Everything is loaded from a bundle directory."""

    def __init__(self, whitener, S_H, S_A, encoder_name, k):
        self.w_mean, self.W = whitener
        self.S_H = S_H                      # (mean, basis)
        self.S_A = S_A                      # {generator: (mean, basis)}
        self.encoder_name = encoder_name
        self.k = k
        self._tok = None
        self._model = None
        self._device = None

    # ------------------------------------------------------------------ load --
    @classmethod
    def load(cls, bundle, k=10, encoder=None, device=None):
        """Load a fitted bundle (the one used in the paper is v3_english_clean)."""
        b = Path(bundle)
        if not b.exists():
            raise FileNotFoundError(
                f"bundle not found: {b}\n"
                "Extract v3_english_clean.zip into bundles/, or build it "
                "yourself with src/build_subspace_v2_english.py (see REPRODUCE.md)."
            )
        wh = np.load(b / "whitener.npz")
        sh = np.load(b / "S_H_universal.npz")
        S_H = (sh["mean"], sh["basis"][:, :k])
        S_A = {}
        for p in sorted((b / "subspaces").glob("S_A_*.npz")):
            z = np.load(p)
            S_A[p.stem[4:]] = (z["mean"], z["basis"][:, :k])
        if not S_A:
            raise ValueError(f"no per-generator subspaces found under {b/'subspaces'}")
        meta = {}
        if (b / "meta.json").exists():
            meta = json.loads((b / "meta.json").read_text())
        enc = encoder or meta.get("encoder") or "microsoft/deberta-v3-base"
        g = cls((wh["mean"], wh["W"]), S_H, S_A, enc, k)
        g._device = device
        return g

    # --------------------------------------------------------------- encoder --
    def _ensure_encoder(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModel
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tok = AutoTokenizer.from_pretrained(self.encoder_name)
        self._model = AutoModel.from_pretrained(self.encoder_name).to(self._device).eval()

    def embed(self, texts, batch_size=16, max_length=512):
        """Mean-pooled embeddings from the frozen encoder. CPU is fine."""
        import torch
        self._ensure_encoder()
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self._tok(texts[i:i + batch_size], padding=True, truncation=True,
                                max_length=max_length, return_tensors="pt").to(self._device)
                h = self._model(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).float()
                out.append(((h * m).sum(1) / m.sum(1).clamp(min=1e-9)).cpu().numpy())
        return np.vstack(out).astype(np.float64)

    # ----------------------------------------------------------------- score --
    def score(self, texts, batch_size=16):
        """s_det for each text. Positive => machine-generated. Higher => more so."""
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        X = self.embed(texts, batch_size=batch_size)
        Xw = (X - self.w_mean) @ self.W.T                       # whiten
        r_h = _residual_sq(Xw, *self.S_H)                       # distance to human
        r_a = np.stack([_residual_sq(Xw, m, B) for (m, B) in self.S_A.values()], 0)
        s = r_h - r_a.min(0)                                    # Eq. 2 of the paper
        return float(s[0]) if single else s

    def is_ai(self, texts, threshold=0.0, batch_size=16):
        """Boolean decision at the geometric boundary (s_det = 0)."""
        s = self.score(texts, batch_size=batch_size)
        return bool(s > threshold) if np.isscalar(s) else (s > threshold)

    def explain(self, text):
        """Score one document and say which generator's subspace it sits closest to.

        The nearest subspace does not establish the document's source.
        """
        X = self.embed([text])
        Xw = (X - self.w_mean) @ self.W.T
        r_h = float(_residual_sq(Xw, *self.S_H)[0])
        gens = list(self.S_A)
        r_a = np.array([_residual_sq(Xw, *self.S_A[g])[0] for g in gens])
        j = int(r_a.argmin())
        n_words = len(text.split())
        return {
            "s_det": r_h - float(r_a[j]),
            "is_ai": bool(r_h - float(r_a[j]) > 0),
            "residual_to_human": r_h,
            "residual_to_nearest_generator": float(r_a[j]),
            "nearest_generator": gens[j],
            "n_words": n_words,
            "reliable": n_words >= MIN_RELIABLE_WORDS,
            "warning": (None if n_words >= MIN_RELIABLE_WORDS else
                        f"Input has {n_words} words; scores below "
                        f"{MIN_RELIABLE_WORDS} words are unvalidated."),
        }

    @property
    def generators(self):
        """Return the generator identifiers represented in the fitted bundle."""
        return list(self.S_A)


# ------------------------------------------------------------------------ CLI --
def main():
    ap = argparse.ArgumentParser(
        description="GLASS: score text using a frozen encoder and fitted subspaces.")
    ap.add_argument("--bundle", default="bundles/v3_english_clean",
                    help="fitted bundle directory (default: bundles/v3_english_clean)")
    ap.add_argument("--text", help="score this string")
    ap.add_argument("--file", nargs="+", help="score these files (globs allowed)")
    ap.add_argument("--k", type=int, default=10, help="subspace rank (paper uses 10)")
    ap.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args()

    docs = []                                    # (label, text)
    if args.text:
        docs.append(("<--text>", args.text))
    for pat in (args.file or []):
        for f in sorted(_glob.glob(pat)):
            docs.append((f, Path(f).read_text(encoding="utf-8", errors="replace")))
    if not docs and not sys.stdin.isatty():
        docs.append(("<stdin>", sys.stdin.read()))
    if not docs:
        ap.error("give --text, --file, or pipe text on stdin")

    g = GLASS.load(args.bundle, k=args.k, device=args.device)
    results = []
    for label, text in docs:
        r = g.explain(text)
        r["source"] = label
        results.append(r)

    if args.json:
        print(json.dumps(results, indent=2))
        return
    for r in results:
        verdict = "MACHINE-GENERATED" if r["is_ai"] else "human"
        print(f"\n{r['source']}")
        print(f"  verdict     : {verdict}")
        print(f"  s_det       : {r['s_det']:+.2f}   (>0 = machine)")
        print(f"  looks like  : {r['nearest_generator']}")
        print(f"  length      : {r['n_words']} words")
        if r["warning"]:
            print(f"  WARNING     : {r['warning']}")
    print()


if __name__ == "__main__":
    main()
