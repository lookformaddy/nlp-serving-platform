"""Pytest configuration: put the src layout on the import path so plain
``pytest`` passes from the project root, with or without an install.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nlpserve.corpus import generate_corpus  # noqa: E402
from nlpserve.models import LogisticRegressionClassifier  # noqa: E402
from nlpserve.pipeline import train_version  # noqa: E402
from nlpserve.registry import ModelRegistry  # noqa: E402
from nlpserve.serving import ServingConfig, ServingLayer  # noqa: E402
from nlpserve.tfidf import TfidfVectorizer  # noqa: E402

SEED = 42


@pytest.fixture()
def corpus():
    return generate_corpus(n_per_intent=60, seed=SEED)


@pytest.fixture()
def vectorizer(corpus):
    train, _test = corpus
    vec = TfidfVectorizer()
    vec.fit([ex.text for ex in train])
    return vec


@pytest.fixture()
def classifier(vectorizer, corpus):
    train, _test = corpus
    X = vectorizer.transform([ex.text for ex in train])
    clf = LogisticRegressionClassifier(epochs=40, seed=SEED)
    clf.fit(X, [ex.label for ex in train],
            n_features=len(vectorizer.vocabulary_))
    return clf


@pytest.fixture()
def trained_version():
    version, _report = train_version("v1-test", seed=SEED,
                                     n_per_intent=60, epochs=40)
    return version


@pytest.fixture()
def registry(trained_version):
    reg = ModelRegistry()
    reg.register(trained_version, role="champion")
    return reg


@pytest.fixture()
def serving(registry):
    return ServingLayer(registry, ServingConfig(canary_pct=25))
