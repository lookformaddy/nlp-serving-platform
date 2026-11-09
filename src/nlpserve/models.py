"""Multinomial logistic regression, trained from scratch.

Full-batch gradient descent on sparse TF-IDF vectors (dicts of
``{feature_index: weight}``), with a softmax output, cross-entropy loss,
L2 regularization, and a simple learning-rate decay. No numpy, no sklearn:
the gradient is an explicit loop over the non-zero features of each
document, which is exactly the shape a production serving path wants —
sparse, fast, and easy to explain token by token.

``predict_proba`` returns a dict ``{label: probability}``; ``explain``
returns the top contributing tokens for a prediction as
``(token, contribution)`` pairs where contribution is
``tfidf_value * weight`` for the predicted class.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

__all__ = ["LogisticRegressionClassifier"]

Vector = dict[int, float]


def _softmax(scores: list[float]) -> list[float]:
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


@dataclass
class LogisticRegressionClassifier:
    labels: list[str] = field(default_factory=list)
    learning_rate: float = 0.5
    epochs: int = 80
    l2: float = 1e-4
    lr_decay: float = 0.98
    seed: int = 42

    weights_: list[dict[int, float]] = field(default_factory=list, init=False)
    bias_: list[float] = field(default_factory=list, init=False)
    n_features_: int = field(default=0, init=False)
    loss_history_: list[float] = field(default_factory=list, init=False)

    # -- training --------------------------------------------------------
    def fit(self, X: list[Vector], y: list[str],
            n_features: int) -> "LogisticRegressionClassifier":
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if not X:
            raise ValueError("cannot fit on empty data")
        self.labels = sorted(set(y))
        label_to_idx = {lab: i for i, lab in enumerate(self.labels)}
        n_classes = len(self.labels)
        self.n_features_ = n_features
        rng = random.Random(self.seed)
        # Sparse init: only allocate weights for features that actually
        # appear, to keep memory proportional to observed data.
        seen: set[int] = set()
        for vec in X:
            seen.update(vec.keys())
        self.weights_ = [
            {j: rng.gauss(0.0, 0.01) for j in sorted(seen)}
            for _ in range(n_classes)
        ]
        self.bias_ = [0.0] * n_classes
        lr = self.learning_rate
        n = len(X)
        for epoch in range(self.epochs):
            grad_w: list[dict[int, float]] = [dict() for _ in range(n_classes)]
            grad_b = [0.0] * n_classes
            loss = 0.0
            for vec, label in zip(X, y):
                scores = [
                    self.bias_[c] + sum(w * vec.get(j, 0.0)
                                        for j, w in self.weights_[c].items())
                    for c in range(n_classes)
                ]
                probs = _softmax(scores)
                true_idx = label_to_idx[label]
                loss -= math.log(max(probs[true_idx], 1e-12))
                for c in range(n_classes):
                    err = probs[c] - (1.0 if c == true_idx else 0.0)
                    if err == 0.0:
                        continue
                    grad_b[c] += err
                    gw = grad_w[c]
                    for j, v in vec.items():
                        gw[j] = gw.get(j, 0.0) + err * v
            for c in range(n_classes):
                w = self.weights_[c]
                gw = grad_w[c]
                for j in list(w.keys()):
                    g = gw.get(j, 0.0) / n + self.l2 * w[j]
                    w[j] -= lr * g
                self.bias_[c] -= lr * grad_b[c] / n
            self.loss_history_.append(loss / n)
            lr *= self.lr_decay
        return self

    # -- inference -------------------------------------------------------
    def _scores(self, vec: Vector) -> list[float]:
        return [
            self.bias_[c] + sum(w * vec.get(j, 0.0)
                                for j, w in self.weights_[c].items())
            for c in range(len(self.labels))
        ]

    def predict_proba(self, vec: Vector) -> dict[str, float]:
        if not self.labels:
            raise ValueError("classifier is not fitted")
        probs = _softmax(self._scores(vec))
        return {lab: p for lab, p in zip(self.labels, probs)}

    def predict(self, vec: Vector) -> str:
        probs = self.predict_proba(vec)
        return max(probs, key=lambda lab: probs[lab])

    def explain(self, vec: Vector, feature_names: list[str],
                top_k: int = 5) -> list[tuple[str, float]]:
        """Top contributing tokens for the predicted class.

        Contribution of token j is ``tfidf_j * weight_j`` for the
        predicted class — positive pushes toward the class, negative
        pushes away. Only non-zero contributions are returned.
        """
        if not self.labels:
            raise ValueError("classifier is not fitted")
        pred = self.predict(vec)
        c = self.labels.index(pred)
        contribs: list[tuple[str, float]] = []
        for j, v in vec.items():
            w = self.weights_[c].get(j, 0.0)
            if w == 0.0 or v == 0.0:
                continue
            name = feature_names[j] if j < len(feature_names) else f"f{j}"
            contribs.append((name, v * w))
        contribs.sort(key=lambda kv: abs(kv[1]), reverse=True)
        return contribs[:top_k]
