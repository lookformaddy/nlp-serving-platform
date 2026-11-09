"""API tests (FastAPI TestClient). These run only when fastapi and
httpx are installed; otherwise they are skipped, not failed.
"""
import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from nlpserve.service import app  # noqa: E402

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["champion"] == "v1-seed42"


def test_predict_returns_label_and_explanation():
    resp = client.post("/predict", json={
        "text": "I was charged twice for my subscription",
        "request_id": "api-1",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["label"] == "billing"
    assert 0.0 < data["confidence"] <= 1.0
    assert data["explanation"]
    assert data["served_by"] in ("champion", "challenger", "fallback")
    assert data["latency_ms"] >= 0.0


def test_predict_validates_empty_text():
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 422


def test_predict_validates_overlong_text():
    resp = client.post("/predict", json={"text": "x" * 2001})
    assert resp.status_code == 422


def test_models_endpoint():
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["champion"] == "v1-seed42"
    assert data["versions"]


def test_stats_endpoint():
    resp = client.get("/stats")
    assert resp.status_code == 200
    assert "served" in resp.json()
    assert "drift" in resp.json()


def test_promote_without_challenger_is_conflict():
    resp = client.post("/promote", json={"baseline_accuracy": 0.6})
    assert resp.status_code == 409
