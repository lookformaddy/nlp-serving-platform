"""Versioned model registry with promotion gates.

A model version is a (vectorizer, classifier) pair plus its offline
metrics and provenance. The registry tracks one champion and optionally
one challenger. Promotion is gated, never automatic: a challenger must
clear every gate before it can become champion:

- metric floor: overall accuracy >= ``min_accuracy``
- must-beat-baseline: accuracy >= keyword-baseline accuracy + ``margin``
- slice gates: every per-class F1 >= ``min_per_class_f1``

Every promotion decision is recorded with the gate results, so a
rollback or audit can replay exactly why a version was (or was not)
promoted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from nlpserve.evaluation import ClassificationReport
from nlpserve.models import LogisticRegressionClassifier
from nlpserve.tfidf import TfidfVectorizer

__all__ = ["ModelVersion", "PromotionGateResult", "ModelRegistry"]


@dataclass
class ModelVersion:
    version: str
    vectorizer: TfidfVectorizer
    classifier: LogisticRegressionClassifier
    metrics: ClassificationReport
    trained_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    seed: int = 42
    notes: str = ""

    def summary(self) -> dict:
        return {
            "version": self.version,
            "accuracy": round(self.metrics.accuracy, 4),
            "macro_f1": round(self.metrics.macro_f1, 4),
            "trained_at": self.trained_at,
            "seed": self.seed,
            "notes": self.notes,
        }


@dataclass
class PromotionGateResult:
    passed: bool
    checks: dict[str, bool]
    details: dict[str, float]

    def __str__(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        lines = [f"promotion gates: {verdict}"]
        for name, ok in self.checks.items():
            lines.append(f"  [{'ok' if ok else 'FAIL'}] {name}: "
                         f"{self.details.get(name, 0.0):.4f}")
        return "\n".join(lines)


@dataclass
class ModelRegistry:
    min_accuracy: float = 0.75
    margin_over_baseline: float = 0.05
    min_per_class_f1: float = 0.55

    versions: dict[str, ModelVersion] = field(default_factory=dict)
    champion: str | None = None
    challenger: str | None = None
    history: list[dict] = field(default_factory=list)

    def register(self, version: ModelVersion,
                 role: str = "challenger") -> None:
        if version.version in self.versions:
            raise ValueError(f"version {version.version!r} already exists")
        self.versions[version.version] = version
        if role == "champion":
            if self.champion is not None:
                raise ValueError("a champion already exists; use promote()")
            self.champion = version.version
        else:
            self.challenger = version.version
        self.history.append({"event": "registered", "version": version.version,
                             "role": role})

    def check_promotion(self, candidate: ModelVersion,
                        baseline_accuracy: float) -> PromotionGateResult:
        metrics = candidate.metrics
        worst_f1 = min(metrics.per_class_f1.values()) if metrics.per_class_f1 else 0.0
        checks = {
            "accuracy_floor": metrics.accuracy >= self.min_accuracy,
            "beats_baseline": (
                metrics.accuracy >= baseline_accuracy + self.margin_over_baseline
            ),
            "per_class_f1_floor": worst_f1 >= self.min_per_class_f1,
        }
        details = {
            "accuracy_floor": metrics.accuracy,
            "beats_baseline": metrics.accuracy - baseline_accuracy,
            "per_class_f1_floor": worst_f1,
        }
        return PromotionGateResult(passed=all(checks.values()),
                                   checks=checks, details=details)

    def promote(self, candidate_version: str,
                baseline_accuracy: float) -> PromotionGateResult:
        """Promote ``candidate_version`` to champion if all gates pass.

        Returns the gate result; on failure the champion is unchanged.
        """
        candidate = self.versions.get(candidate_version)
        if candidate is None:
            raise ValueError(f"unknown version {candidate_version!r}")
        result = self.check_promotion(candidate, baseline_accuracy)
        self.history.append({
            "event": "promotion_attempt", "version": candidate_version,
            "passed": result.passed, "checks": result.checks,
        })
        if result.passed:
            previous = self.champion
            self.champion = candidate_version
            if self.challenger == candidate_version:
                self.challenger = None
            self.history.append({"event": "promoted",
                                 "version": candidate_version,
                                 "previous_champion": previous})
        return result

    def rollback(self, version: str) -> None:
        """Restore a previously registered version as champion."""
        if version not in self.versions:
            raise ValueError(f"unknown version {version!r}")
        previous = self.champion
        self.champion = version
        self.history.append({"event": "rollback", "version": version,
                             "previous_champion": previous})

    def get_champion(self) -> ModelVersion | None:
        return self.versions.get(self.champion) if self.champion else None

    def get_challenger(self) -> ModelVersion | None:
        return self.versions.get(self.challenger) if self.challenger else None

    def list_versions(self) -> list[dict]:
        return [v.summary() for v in self.versions.values()]
