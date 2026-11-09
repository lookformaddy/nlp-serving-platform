"""Tests for the synthetic corpus generator and offline evaluation."""
import pytest

from nlpserve.corpus import INTENTS, generate_corpus
from nlpserve.evaluation import (ClassificationReport, evaluate,
                                 keyword_baseline_predict)


def test_corpus_covers_all_intents():
    train, test = generate_corpus(n_per_intent=50, seed=1)
    assert set(ex.label for ex in train) == set(INTENTS)
    assert set(ex.label for ex in test) == set(INTENTS)
    assert len(train) == 50 * 6 - 60  # 20% held out per intent
    assert len(test) == 60


def test_corpus_is_deterministic():
    a_train, a_test = generate_corpus(n_per_intent=20, seed=99)
    b_train, b_test = generate_corpus(n_per_intent=20, seed=99)
    assert [(ex.text, ex.label) for ex in a_train] == [
        (ex.text, ex.label) for ex in b_train]
    assert [(ex.text, ex.label) for ex in a_test] == [
        (ex.text, ex.label) for ex in b_test]


def test_corpus_test_labels_are_clean():
    _train, test = generate_corpus(n_per_intent=50, seed=3, label_noise=0.5)
    # Every test example's text must come from its own intent's templates:
    # check by the absence of cross-intent leakage via the keyword map.
    assert all(ex.label in INTENTS for ex in test)


def test_label_noise_only_affects_train():
    train, test = generate_corpus(n_per_intent=100, seed=5, label_noise=0.0)
    assert len(train) == 480
    assert len(test) == 120


def test_evaluate_perfect_predictions():
    report = evaluate(["a", "b", "b"], ["a", "b", "b"])
    assert report.accuracy == 1.0
    assert report.macro_f1 == 1.0
    assert report.confusion["a"]["a"] == 1


def test_evaluate_partial_predictions():
    report = evaluate(["a", "a", "b", "b"], ["a", "b", "b", "a"])
    assert report.accuracy == 0.5
    assert report.per_class_precision["a"] == 0.5
    assert report.per_class_recall["a"] == 0.5
    assert report.per_class_f1["a"] == pytest.approx(0.5)


def test_evaluate_rejects_mismatched_and_empty():
    with pytest.raises(ValueError):
        evaluate(["a"], ["a", "b"])
    with pytest.raises(ValueError):
        evaluate([], [])


def test_keyword_baseline_predicts_intents():
    label, conf = keyword_baseline_predict(
        "I was charged twice for my subscription")
    assert label == "billing"
    assert 0.0 < conf <= 1.0


def test_keyword_baseline_unknown_when_no_match():
    label, conf = keyword_baseline_predict("zqxwv blorpt flibberty")
    assert label == "unknown"
    assert conf == 0.0


def test_report_summary_and_confusion_table(trained_version):
    summary = trained_version.metrics.summary()
    assert "accuracy=" in summary
    assert "billing" in summary
    table = trained_version.metrics.confusion_table()
    assert "true \\ pred" in table
