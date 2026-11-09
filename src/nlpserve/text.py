"""Text preprocessing: normalization and tokenization.

Deliberately small and dependency-free. Normalization lowercases, strips
punctuation, collapses whitespace, and expands a handful of common
contractions so the same intent phrased two ways lands on the same tokens.
Stopwords are removed only when the caller asks — the classifier keeps
them out by default because intent text is short and stopwords carry
almost no signal, but the raw tokenizer is available for debugging.
"""
from __future__ import annotations

import re
import unicodedata

__all__ = ["normalize", "tokenize", "STOPWORDS"]

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_WS_RE = re.compile(r"\s+")

_CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'ve": " have",
    "'ll": " will",
    "'d": " would",
    "'m": " am",
    "i'm": "i am",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "here's": "here is",
    "what's": "what is",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "hasn't": "has not",
    "haven't": "have not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "wouldn't": "would not",
}

STOPWORDS = frozenset(
    """
    a an the and or but if then else for of to in on at by with from as is am
    are was were be been being have has had do does did will would shall
    should can could may might must i you he she it we they me him her us
    them my your his its our their this that these those there here what
    when where which who whom how why not no yes please just very really
    get got getting make made take took
    """.split()
)


def normalize(text: str) -> str:
    """Lowercase, de-accent, expand contractions, collapse whitespace."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    for contraction, expansion in _CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return _WS_RE.sub(" ", text).strip()


def tokenize(text: str, *, remove_stopwords: bool = True,
             min_len: int = 2) -> list[str]:
    """Tokenize normalized text into word tokens.

    Keeps tokens of at least ``min_len`` characters; optionally drops
    stopwords. Returns an empty list for empty/whitespace-only input
    instead of raising, so the serving layer can fall back cleanly.
    """
    normalized = normalize(text)
    tokens = _TOKEN_RE.findall(normalized)
    tokens = [t for t in tokens if len(t) >= min_len]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens
