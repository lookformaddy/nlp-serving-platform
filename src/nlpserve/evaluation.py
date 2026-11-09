"""Offline evaluation: accuracy, per-class precision/recall/F1,
confusion matrix, and a keyword-baseline comparison.

The keyword baseline is a hand-built intent -> keyword map scored by
token overlap. It is deliberately weak — it exists to (a) give the
promotion gate a floor the learned model must beat, and (b) serve as
the fast fallback when the model path exceeds its latency budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from nlpserve.text import tokenize

__all__ = ["ClassificationReport", "evaluate", "KEYWORD_MAP",
           "keyword_baseline_predict"]

KEYWORD_MAP: dict[str, set[str]] = {
    "billing": {"charged", "charge", "invoice", "bill", "payment", "fee",
                "subscription", "renewed", "price", "card", "refund"},
    "account_access": {"login", "password", "account", "locked", "reset",
                       "sign", "verification", "code", "email", "suspended",
                       "recover"},
    "shipping": {"order", "delivery", "package", "tracking", "arrived",
                 "courier", "address", "shipped", "deliver", "pickup"},
    "refund": {"refund", "returned", "return", "cancelled", "faulty",
               "credit", "money", "back"},
    "technical_issue": {"crashes", "crash", "error", "loading", "load",
                        "freezes", "freeze", "buffering", "update",
                        "disconnecting", "settings", "website", "app"},
    "feedback": {"love", "great", "thank", "add", "wish", "suggestion",
                 "enjoy", "faster", "smoother", "helpful"},
}


def keyword_baseline_predict(text: str) -> tuple[str, float]:
    """Keyword-overlap classifier. Returns (label, confidence).

    Confidence is the winning intent's keyword-hit share of all hits;
    0.0 with label ``"unknown"`` when nothing matches.
    """
    tokens = set(tokenize(text))
    scores = {label: len(tokens & kws) for label, kws in KEYWORD_MAP.items()}
    best = max(scores, key=lambda lab: scores[lab])
    total = sum(scores.values())
    if total == 0:
        return "unknown", 0.0
    return best, scores[best] / total


@dataclass
class ClassificationReport:
    accuracy: float
    per_class_precision: dict[str, float] = field(default_factory=dict)
    per_class_recall: dict[str, float] = field(default_factory=dict)
    per_class_f1: dict[str, float] = field(default_factory=dict)
    macro_f1: float = 0.0
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    n: int = 0

    def summary(self) -> str:
        lines = [f"accuracy={self.accuracy:.4f}  macro_f1={self.macro_f1:.4f}"
                 f"  n={self.n}",
                 f"{'intent':<16}{'P':>7}{'R':>7}{'F1':>7}"]
        for lab in sorted(self.per_class_f1):
            lines.append(
                f"{lab:<16}{self.per_class_precision[lab]:>7.3f}"
                f"{self.per_class_recall[lab]:>7.3f}"
                f"{self.per_class_f1[lab]:>7.3f}")
        return "\n".join(lines)

    def confusion_table(self) -> str:
        labels = sorted(self.confusion)
        header = "true \\ pred".ljust(16) + "".join(l[:8].rjust(9) for l in labels)
        rows = [header]
        for true_lab in labels:
            row = true_lab.ljust(16) + "".join(
                str(self.confusion[true_lab].get(pred, 0)).rjust(9)
                for pred in labels)
            rows.append(row)
        return "\n".join(rows)


def evaluate(y_true: list[str], y_pred: list[str]) -> ClassificationReport:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("cannot evaluate empty predictions")
    labels = sorted(set(y_true) | set(y_pred))
    confusion: dict[str, dict[str, int]] = {
        t: {p: 0 for p in labels} for t in labels
    }
    correct = 0
    for t, p in zip(y_true, y_pred):
        confusion[t][p] += 1
        correct += int(t == p)
    precision, recall, f1 = {}, {}, {}
    for lab in labels:
        tp = confusion[lab][lab]
        fp = sum(confusion[t][lab] for t in labels if t != lab)
        fn = sum(confusion[lab][p] for p in labels if p != lab)
        precision[lab] = tp / (tp + fp) if tp + fp else 0.0
        recall[lab] = tp / (tp + fn) if tp + fn else 0.0
        f1[lab] = (2 * precision[lab] * recall[lab]
                   / (precision[lab] + recall[lab])
                   if precision[lab] + recall[lab] else 0.0)
    return ClassificationReport(
        accuracy=correct / len(y_true),
        per_class_precision=precision,
        per_class_recall=recall,
        per_class_f1=f1,
        macro_f1=sum(f1.values()) / len(f1) if f1 else 0.0,
        confusion=confusion,
        n=len(y_true),
    )
