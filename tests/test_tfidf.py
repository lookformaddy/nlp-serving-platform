"""Tests for the TF-IDF vectorizer."""
import math

import pytest

from nlpserve.tfidf import TfidfVectorizer

DOCS = [
    "refund my order please refund",
    "refund for the faulty item",
    "my password is not working",
    "cannot log in password reset",
]


def test_fit_builds_vocabulary():
    vec = TfidfVectorizer(min_df=1)
    vec.fit(DOCS)
    assert "refund" in vec.vocabulary_
    assert "password" in vec.vocabulary_
    assert len(vec.vocabulary_) == len(set(vec.vocabulary_))


def test_transform_vectors_are_l2_normalized():
    vec = TfidfVectorizer(min_df=1)
    vec.fit(DOCS)
    out = vec.transform(["refund my order please"])
    assert len(out) == 1
    norm = math.sqrt(sum(v * v for v in out[0].values()))
    assert norm == pytest.approx(1.0)


def test_sublinear_tf_damps_repetition():
    # Sublinear TF (1 + log count) damps repetition *relative to linear
    # TF*: the repeated term's normalized share grows more slowly.
    sub = TfidfVectorizer(min_df=1, sublinear_tf=True)
    lin = TfidfVectorizer(min_df=1, sublinear_tf=False)
    sub.fit(DOCS)
    lin.fit(DOCS)
    doc = ["refund refund refund password"]
    idx = sub.vocabulary_["refund"]
    assert sub.transform(doc)[0][idx] < lin.transform(doc)[0][idx]


def test_unseen_terms_are_ignored():
    vec = TfidfVectorizer(min_df=1)
    vec.fit(DOCS)
    assert vec.transform(["zqxwv blorpt"]) == [{}]


def test_transform_before_fit_raises():
    with pytest.raises(ValueError):
        TfidfVectorizer().transform(["hello"])


def test_bigrams_captured():
    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    vec.fit(DOCS)
    assert "refund request" not in vec.vocabulary_  # not in docs
    assert "cannot log" in vec.vocabulary_
    # "log in" never forms: "in" is a stopword and is removed first.


def test_oov_rate():
    vec = TfidfVectorizer(min_df=1)
    vec.fit(DOCS)
    assert vec.oov_rate(["refund my order"]) == 0.0
    assert vec.oov_rate(["zqxwv blorpt"]) == 1.0
    assert 0.0 < vec.oov_rate(["refund zqxwv"]) < 1.0


def test_max_features_caps_vocabulary():
    vec = TfidfVectorizer(min_df=1, max_features=3)
    vec.fit(DOCS)
    assert len(vec.vocabulary_) == 3
