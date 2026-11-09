"""Tests for the from-scratch logistic regression classifier."""
import pytest

from nlpserve.corpus import INTENTS
from nlpserve.models import LogisticRegressionClassifier


def test_fit_and_predict_on_trained_fixture(classifier, vectorizer):
    for text, expected in [
        ("I was charged twice for my subscription", "billing"),
        ("I cannot log in, my password is not working", "account_access"),
        ("My order has not arrived yet", "shipping"),
        ("Please refund the faulty item", "refund"),
        ("The app crashes every time I open it", "technical_issue"),
        ("I love the new design, great work", "feedback"),
    ]:
        vec = vectorizer.transform([text])[0]
        assert classifier.predict(vec) == expected


def test_predict_proba_sums_to_one(classifier, vectorizer):
    vec = vectorizer.transform(["please refund my order"])[0]
    probs = classifier.predict_proba(vec)
    assert set(probs) == set(INTENTS)
    assert sum(probs.values()) == pytest.approx(1.0)
    assert all(0.0 <= p <= 1.0 for p in probs.values())


def test_loss_decreases_over_epochs(classifier):
    losses = classifier.loss_history_
    assert len(losses) > 1
    assert losses[-1] < losses[0]


def test_explain_returns_top_tokens(classifier, vectorizer):
    vec = vectorizer.transform(["please refund the faulty item"])[0]
    explanation = classifier.explain(vec, vectorizer.feature_names(), top_k=3)
    assert len(explanation) <= 3
    assert all(isinstance(tok, str) and isinstance(w, float)
               for tok, w in explanation)
    assert any("refund" in tok for tok, _ in explanation)


def test_predict_before_fit_raises(vectorizer):
    clf = LogisticRegressionClassifier()
    with pytest.raises(ValueError):
        clf.predict(vectorizer.transform(["hello"])[0])


def test_fit_rejects_mismatched_lengths():
    clf = LogisticRegressionClassifier()
    with pytest.raises(ValueError):
        clf.fit([{0: 1.0}], ["a", "b"], n_features=1)


def test_training_is_deterministic(vectorizer, corpus):
    train, _ = corpus
    X = vectorizer.transform([ex.text for ex in train])
    y = [ex.label for ex in train]
    clf_a = LogisticRegressionClassifier(epochs=10, seed=7)
    clf_b = LogisticRegressionClassifier(epochs=10, seed=7)
    clf_a.fit(X, y, n_features=len(vectorizer.vocabulary_))
    clf_b.fit(X, y, n_features=len(vectorizer.vocabulary_))
    vec = vectorizer.transform(["please refund my order"])[0]
    assert clf_a.predict(vec) == clf_b.predict(vec)
    assert clf_a.loss_history_ == clf_b.loss_history_
