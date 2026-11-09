"""Training pipeline: corpus -> TF-IDF -> logistic regression ->
registry version, plus offline evaluation against the keyword baseline.
"""
from __future__ import annotations

from nlpserve.corpus import generate_corpus
from nlpserve.evaluation import evaluate, keyword_baseline_predict
from nlpserve.models import LogisticRegressionClassifier
from nlpserve.registry import ModelVersion
from nlpserve.tfidf import TfidfVectorizer

__all__ = ["train_version"]


def train_version(version: str, seed: int = 42, n_per_intent: int = 200,
                  epochs: int = 80, learning_rate: float = 0.5,
                  label_noise: float = 0.05
                  ) -> tuple[ModelVersion, dict]:
    """Train one version end to end and evaluate it on held-out data.

    Returns ``(version, report)`` where ``report`` holds the learned
    model's metrics, the keyword-baseline accuracy, and dataset sizes.
    """
    train, test = generate_corpus(n_per_intent=n_per_intent, seed=seed,
                                  label_noise=label_noise)
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform([ex.text for ex in train])
    X_test = vectorizer.transform([ex.text for ex in test])

    classifier = LogisticRegressionClassifier(
        learning_rate=learning_rate, epochs=epochs, seed=seed)
    classifier.fit(X_train, [ex.label for ex in train],
                   n_features=len(vectorizer.vocabulary_))

    y_true = [ex.label for ex in test]
    y_pred = [classifier.predict(vec) for vec in X_test]
    metrics = evaluate(y_true, y_pred)

    baseline_preds = [keyword_baseline_predict(ex.text)[0] for ex in test]
    baseline_acc = sum(int(p == t) for p, t in zip(baseline_preds, y_true)) / len(y_true)

    version_obj = ModelVersion(
        version=version, vectorizer=vectorizer, classifier=classifier,
        metrics=metrics, seed=seed,
        notes=(f"TF-IDF (1-2 grams) + multinomial logistic regression, "
               f"{len(train)} train / {len(test)} test, seed={seed}"))
    report = {
        "metrics": metrics,
        "baseline_accuracy": baseline_acc,
        "n_train": len(train),
        "n_test": len(test),
        "vocab_size": len(vectorizer.vocabulary_),
        "final_loss": classifier.loss_history_[-1] if classifier.loss_history_ else None,
    }
    return version_obj, report
