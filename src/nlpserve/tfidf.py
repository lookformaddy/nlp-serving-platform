"""TF-IDF vectorizer, implemented from scratch on sparse dicts.

Design:
- Sublinear term frequency: ``tf = 1 + log(count)`` for count > 0, which
  damps the effect of a word repeated many times in one short message.
- Smoothed IDF: ``idf(t) = log((1 + N) / (1 + df(t))) + 1``, so terms
  seen in every document still carry a small positive weight and unseen
  terms map to a fixed ``max_idf`` instead of exploding.
- Optional n-gram range (unigrams + bigrams by default) — bigrams catch
  phrases like "refund request" that unigrams split apart.
- ``max_features`` caps vocabulary size by document frequency, keeping
  the model small and the serving path fast.
- Vectors are L2-normalized sparse dicts ``{feature_index: weight}``.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from nlpserve.text import tokenize

__all__ = ["TfidfVectorizer"]


@dataclass
class TfidfVectorizer:
    ngram_range: tuple[int, int] = (1, 2)
    max_features: int | None = 5000
    min_df: int = 2
    sublinear_tf: bool = True

    vocabulary_: dict[str, int] = field(default_factory=dict, init=False)
    idf_: dict[int, float] = field(default_factory=dict, init=False)
    max_idf_: float = field(default=0.0, init=False)
    n_docs_: int = field(default=0, init=False)

    # -- fitting ---------------------------------------------------------
    def _ngrams(self, tokens: list[str]) -> list[str]:
        lo, hi = self.ngram_range
        feats: list[str] = []
        for n in range(lo, hi + 1):
            if n == 1:
                feats.extend(tokens)
            else:
                feats.extend(
                    " ".join(tokens[i:i + n])
                    for i in range(len(tokens) - n + 1)
                )
        return feats

    def fit(self, documents: list[str]) -> "TfidfVectorizer":
        df: Counter[str] = Counter()
        self.n_docs_ = len(documents)
        for doc in documents:
            feats = set(self._ngrams(tokenize(doc)))
            df.update(feats)
        # Keep terms frequent enough to be reliable; rank by df so the
        # cap keeps the most broadly useful vocabulary.
        kept = [t for t, c in df.items() if c >= self.min_df]
        kept.sort(key=lambda t: (-df[t], t))
        if self.max_features is not None:
            kept = kept[: self.max_features]
        self.vocabulary_ = {t: i for i, t in enumerate(kept)}
        n = self.n_docs_
        self.idf_ = {
            idx: math.log((1 + n) / (1 + df[term])) + 1.0
            for term, idx in self.vocabulary_.items()
        }
        self.max_idf_ = math.log(1 + n) + 1.0  # df == 0 term
        return self

    def fit_transform(self, documents: list[str]) -> list[dict[int, float]]:
        self.fit(documents)
        return self.transform(documents)

    # -- transforming ----------------------------------------------------
    def transform(self, documents: list[str]) -> list[dict[int, float]]:
        if not self.vocabulary_:
            raise ValueError("vectorizer is not fitted")
        out: list[dict[int, float]] = []
        for doc in documents:
            counts = Counter(self._ngrams(tokenize(doc)))
            vec: dict[int, float] = {}
            for term, count in counts.items():
                idx = self.vocabulary_.get(term)
                if idx is None:
                    continue
                tf = 1.0 + math.log(count) if self.sublinear_tf else float(count)
                vec[idx] = tf * self.idf_[idx]
            out.append(self._l2_normalize(vec))
        return out

    @staticmethod
    def _l2_normalize(vec: dict[int, float]) -> dict[int, float]:
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm == 0.0:
            return {}
        return {k: v / norm for k, v in vec.items()}

    # -- introspection ---------------------------------------------------
    def feature_names(self) -> list[str]:
        return [term for term, _ in sorted(self.vocabulary_.items(),
                                           key=lambda kv: kv[1])]

    def oov_rate(self, documents: list[str]) -> float:
        """Fraction of tokens in ``documents`` absent from the vocabulary."""
        total = 0
        oov = 0
        for doc in documents:
            for feat in self._ngrams(tokenize(doc)):
                total += 1
                if feat not in self.vocabulary_:
                    oov += 1
        return (oov / total) if total else 0.0
