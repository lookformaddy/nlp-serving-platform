"""Serving layer: canary + shadow deployment, deterministic hash A/B
routing, latency budgets with keyword-baseline fallback, drift
monitoring, and per-prediction explanations.

Request flow for ``predict(text, request_id)``:
1. Drift monitor scores the input (OOV rate vs. training vocabulary).
2. A deterministic hash of ``request_id`` decides the serving arm:
   ``hash % 100 < canary_pct`` routes to the challenger (canary), the
   rest to the champion. No request_id -> champion.
3. The chosen model predicts under a latency budget. If it exceeds the
   budget (or raises), the fast keyword baseline answers instead and
   the outcome is recorded as a fallback.
4. If a challenger exists and the request went to the champion, the
   challenger is also scored in shadow mode; champion/challenger
   agreement is recorded but the shadow result is never returned.
5. The response carries the prediction, confidence, top contributing
   tokens, which arm served it, latency, and drift flags.

All routing is deterministic for a given request_id, so an experiment
arm is stable across retries — the property that makes A/B results
interpretable.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

from nlpserve.evaluation import keyword_baseline_predict
from nlpserve.models import LogisticRegressionClassifier
from nlpserve.registry import ModelRegistry, ModelVersion
from nlpserve.text import tokenize
from nlpserve.tfidf import TfidfVectorizer

__all__ = ["ServingConfig", "Prediction", "DriftMonitor", "ServingLayer",
           "hash_arm"]


def hash_arm(request_id: str) -> int:
    """Deterministic 0..99 bucket for a request id (stable across runs)."""
    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % 100


@dataclass
class ServingConfig:
    canary_pct: int = 10            # % of hashed traffic to the challenger
    latency_budget_ms: float = 100.0  # model path budget; over -> fallback
    oov_alert_threshold: float = 0.35  # OOV rate above this flags drift
    shadow_enabled: bool = True


@dataclass
class Prediction:
    label: str
    confidence: float
    explanation: list[tuple[str, float]] = field(default_factory=list)
    served_by: str = "champion"     # champion | challenger | fallback
    model_version: str | None = None
    latency_ms: float = 0.0
    fallback_reason: str | None = None
    drift_oov_rate: float = 0.0
    drift_flag: bool = False


@dataclass
class DriftMonitor:
    """Tracks input drift against the training vocabulary.

    Two signals: the out-of-vocabulary rate of incoming tokens, and the
    share of incoming tokens never seen in training. Both are cheap to
    compute per request and are aggregated into a running summary.
    """
    oov_alert_threshold: float = 0.35
    n_requests: int = 0
    total_oov_rate: float = 0.0
    flagged_requests: int = 0

    def score(self, text: str, vectorizer: TfidfVectorizer) -> tuple[float, bool]:
        rate = vectorizer.oov_rate([text])
        self.n_requests += 1
        self.total_oov_rate += rate
        flagged = rate > self.oov_alert_threshold
        self.flagged_requests += int(flagged)
        return rate, flagged

    def summary(self) -> dict:
        mean_oov = (self.total_oov_rate / self.n_requests
                    if self.n_requests else 0.0)
        return {
            "n_requests": self.n_requests,
            "mean_oov_rate": round(mean_oov, 4),
            "flagged_requests": self.flagged_requests,
        }


class ServingLayer:
    def __init__(self, registry: ModelRegistry,
                 config: ServingConfig | None = None):
        self.registry = registry
        self.config = config or ServingConfig()
        self.drift = DriftMonitor(
            oov_alert_threshold=self.config.oov_alert_threshold)
        self.shadow_agreements: list[bool] = []
        self.fallbacks: int = 0
        self.served: dict[str, int] = {"champion": 0, "challenger": 0,
                                       "fallback": 0}

    # -- model scoring ---------------------------------------------------
    @staticmethod
    def _model_predict(version: ModelVersion, text: str) -> Prediction:
        vec = version.vectorizer.transform([text])[0]
        probs = version.classifier.predict_proba(vec)
        label = max(probs, key=lambda lab: probs[lab])
        explanation = version.classifier.explain(
            vec, version.vectorizer.feature_names())
        return Prediction(label=label, confidence=probs[label],
                          explanation=explanation,
                          model_version=version.version)

    @staticmethod
    def _fallback_predict(text: str) -> Prediction:
        label, confidence = keyword_baseline_predict(text)
        return Prediction(label=label, confidence=confidence,
                          served_by="fallback",
                          fallback_reason="keyword_baseline")

    # -- public API ------------------------------------------------------
    def predict(self, text: str, request_id: str | None = None) -> Prediction:
        champion = self.registry.get_champion()
        if champion is None:
            raise ValueError("no champion model registered")
        start = time.perf_counter()
        oov_rate, drift_flag = self.drift.score(text, champion.vectorizer)

        challenger = self.registry.get_challenger()
        use_challenger = (
            challenger is not None
            and request_id is not None
            and hash_arm(request_id) < self.config.canary_pct
        )
        target = challenger if use_challenger else champion
        arm = "challenger" if use_challenger else "champion"

        try:
            result = self._model_predict(target, text)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if elapsed_ms > self.config.latency_budget_ms:
                raise TimeoutError(
                    f"model latency {elapsed_ms:.1f}ms exceeded budget "
                    f"{self.config.latency_budget_ms}ms")
        except Exception as exc:  # noqa: BLE001 - fallback must be total
            result = self._fallback_predict(text)
            result.fallback_reason = (
                f"{result.fallback_reason}: {type(exc).__name__}")
            self.fallbacks += 1
            self.served["fallback"] += 1
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            result.latency_ms = elapsed_ms
            result.drift_oov_rate = oov_rate
            result.drift_flag = drift_flag
            return result

        result.served_by = arm
        result.latency_ms = elapsed_ms
        result.drift_oov_rate = oov_rate
        result.drift_flag = drift_flag
        self.served[arm] += 1

        # Shadow: score the challenger on champion traffic, record
        # agreement, never return the shadow result.
        if (self.config.shadow_enabled and challenger is not None
                and not use_challenger):
            shadow = self._model_predict(challenger, text)
            self.shadow_agreements.append(shadow.label == result.label)

        return result

    def shadow_agreement_rate(self) -> float | None:
        if not self.shadow_agreements:
            return None
        return sum(self.shadow_agreements) / len(self.shadow_agreements)

    def stats(self) -> dict:
        return {
            "served": dict(self.served),
            "fallbacks": self.fallbacks,
            "shadow_agreement_rate": self.shadow_agreement_rate(),
            "shadow_n": len(self.shadow_agreements),
            "drift": self.drift.summary(),
        }
