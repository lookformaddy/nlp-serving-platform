"""FastAPI service for the NLP serving platform.

Endpoints:
    GET  /health    — liveness + champion version + drift summary
    POST /predict   — classify one text (champion/canary/fallback arms)
    GET  /models    — registry versions, champion, challenger
    GET  /stats     — serving stats: arms, fallbacks, shadow agreement, drift
    POST /promote   — promote the challenger if the registry gates pass

The module-level ``app`` is built at import with a champion trained on
the seeded corpus, so ``TestClient(app)`` works out of the box in tests.
For production wiring use :func:`build_app` with your own ServingLayer.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException

from nlpserve.pipeline import train_version
from nlpserve.registry import ModelRegistry
from nlpserve.serving import ServingConfig, ServingLayer

__all__ = ["app", "build_app"]


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000,
                      description="Support-ticket-style text to classify")
    request_id: str | None = Field(
        default=None,
        description="Stable id for deterministic canary/A-B routing")


class PredictResponse(BaseModel):
    label: str
    confidence: float
    explanation: list[tuple[str, float]]
    served_by: str
    model_version: str | None
    latency_ms: float
    fallback_reason: str | None = None
    drift_oov_rate: float
    drift_flag: bool


class PromoteRequest(BaseModel):
    baseline_accuracy: float = Field(..., ge=0.0, le=1.0)


def build_app(serving: ServingLayer) -> FastAPI:
    app = FastAPI(title="nlp-serving-platform",
                  description="Production NLP text-classification serving "
                              "(TF-IDF + from-scratch logistic regression)")

    @app.get("/health")
    def health() -> dict:
        champion = serving.registry.get_champion()
        return {
            "status": "ok",
            "champion": champion.version if champion else None,
            "challenger": serving.registry.challenger,
            "drift": serving.drift.summary(),
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        result = serving.predict(req.text, request_id=req.request_id)
        return PredictResponse(
            label=result.label,
            confidence=result.confidence,
            explanation=result.explanation,
            served_by=result.served_by,
            model_version=result.model_version,
            latency_ms=result.latency_ms,
            fallback_reason=result.fallback_reason,
            drift_oov_rate=result.drift_oov_rate,
            drift_flag=result.drift_flag,
        )

    @app.get("/models")
    def models() -> dict:
        return {
            "champion": serving.registry.champion,
            "challenger": serving.registry.challenger,
            "versions": serving.registry.list_versions(),
        }

    @app.get("/stats")
    def stats() -> dict:
        return serving.stats()

    @app.post("/promote")
    def promote(req: PromoteRequest) -> dict:
        challenger = serving.registry.get_challenger()
        if challenger is None:
            raise HTTPException(status_code=409,
                                detail="no challenger registered")
        result = serving.registry.promote(challenger.version,
                                         req.baseline_accuracy)
        return {
            "passed": result.passed,
            "checks": result.checks,
            "details": result.details,
            "champion": serving.registry.champion,
        }

    return app


def _default_serving() -> ServingLayer:
    version, _report = train_version("v1-seed42", seed=42)
    registry = ModelRegistry()
    registry.register(version, role="champion")
    return ServingLayer(registry, ServingConfig(canary_pct=10))


app = build_app(_default_serving())
