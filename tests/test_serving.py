"""Tests for the serving layer: routing, fallback, shadow, drift."""
import pytest

from nlpserve.pipeline import train_version
from nlpserve.registry import ModelRegistry
from nlpserve.serving import ServingConfig, ServingLayer, hash_arm


def test_hash_arm_is_deterministic_and_bounded():
    assert hash_arm("req-1") == hash_arm("req-1")
    assert 0 <= hash_arm("req-1") < 100
    assert 0 <= hash_arm("another-request") < 100


def test_predict_returns_label_confidence_explanation(serving):
    pred = serving.predict("I was charged twice for my subscription",
                           request_id="req-plain")
    assert pred.label == "billing"
    assert 0.0 < pred.confidence <= 1.0
    assert pred.explanation  # token explanations present
    assert pred.served_by == "champion"
    assert pred.latency_ms >= 0.0
    assert pred.drift_flag is False


def test_predict_without_champion_raises():
    reg = ModelRegistry()
    layer = ServingLayer(reg)
    with pytest.raises(ValueError):
        layer.predict("hello")


def test_canary_routing_is_deterministic(registry, trained_version):
    challenger, _ = train_version("v2-test", seed=7,
                                  n_per_intent=60, epochs=40)
    registry.register(challenger, role="challenger")
    layer = ServingLayer(registry, ServingConfig(canary_pct=25))
    # Find a request id that hashes into the canary bucket.
    canary_id = next(f"req-{i}" for i in range(200)
                     if hash_arm(f"req-{i}") < 25)
    first = layer.predict("please refund my order", request_id=canary_id)
    second = layer.predict("please refund my order", request_id=canary_id)
    assert first.served_by == "challenger"
    assert second.served_by == "challenger"  # stable across retries
    assert first.model_version == "v2-test"


def test_no_request_id_always_serves_champion(registry, trained_version):
    challenger, _ = train_version("v2-test", seed=7,
                                  n_per_intent=60, epochs=40)
    registry.register(challenger, role="challenger")
    layer = ServingLayer(registry, ServingConfig(canary_pct=100))
    pred = layer.predict("please refund my order")
    assert pred.served_by == "champion"


def test_latency_budget_triggers_keyword_fallback(serving):
    # A zero budget guarantees the model path "exceeds" it.
    serving.config.latency_budget_ms = 0.0
    pred = serving.predict("I was charged twice for my subscription",
                           request_id="req-slow")
    assert pred.served_by == "fallback"
    assert pred.label == "billing"  # keyword baseline still answers
    assert pred.fallback_reason is not None
    assert serving.fallbacks == 1


def test_shadow_scoring_records_agreement(registry, trained_version):
    challenger, _ = train_version("v2-test", seed=7,
                                  n_per_intent=60, epochs=40)
    registry.register(challenger, role="challenger")
    layer = ServingLayer(registry, ServingConfig(canary_pct=0))
    for i in range(5):
        layer.predict("please refund my order", request_id=f"shadow-{i}")
    assert layer.shadow_agreement_rate() is not None
    assert len(layer.shadow_agreements) == 5
    assert 0.0 <= layer.shadow_agreement_rate() <= 1.0


def test_no_shadow_without_challenger(serving):
    serving.predict("please refund my order", request_id="x")
    assert serving.shadow_agreement_rate() is None


def test_drift_flags_oov_text(serving):
    pred = serving.predict("zqxwv blorpt flibberty gibbet quantum",
                           request_id="req-drift")
    assert pred.drift_oov_rate == pytest.approx(1.0)
    assert pred.drift_flag is True
    assert serving.drift.summary()["flagged_requests"] == 1


def test_drift_summary_counts_requests(serving):
    serving.predict("please refund my order", request_id="d1")
    serving.predict("my password is not working", request_id="d2")
    summary = serving.drift.summary()
    assert summary["n_requests"] == 2
    assert summary["flagged_requests"] == 0


def test_stats_shape(serving):
    serving.predict("please refund my order", request_id="s1")
    stats = serving.stats()
    assert stats["served"]["champion"] == 1
    assert stats["fallbacks"] == 0
    assert "drift" in stats
