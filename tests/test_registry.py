"""Tests for the model registry and promotion gates."""
import pytest

from nlpserve.evaluation import ClassificationReport
from nlpserve.registry import ModelRegistry, ModelVersion


def _report(accuracy=0.9, f1=0.85):
    labels = ["billing", "refund"]
    return ClassificationReport(
        accuracy=accuracy,
        per_class_precision={lab: f1 for lab in labels},
        per_class_recall={lab: f1 for lab in labels},
        per_class_f1={lab: f1 for lab in labels},
        macro_f1=f1, n=100)


def _version(name, accuracy=0.9, f1=0.85):
    return ModelVersion(version=name, vectorizer=None, classifier=None,
                        metrics=_report(accuracy, f1))


def test_register_champion_and_challenger(trained_version):
    reg = ModelRegistry()
    reg.register(trained_version, role="champion")
    assert reg.champion == trained_version.version
    assert reg.get_champion() is trained_version


def test_register_duplicate_version_raises(trained_version):
    reg = ModelRegistry()
    reg.register(trained_version, role="champion")
    with pytest.raises(ValueError):
        reg.register(trained_version, role="challenger")


def test_promotion_passes_when_gates_clear(trained_version):
    reg = ModelRegistry()
    reg.register(trained_version, role="champion")
    challenger = _version("v2", accuracy=0.92, f1=0.88)
    reg.register(challenger, role="challenger")
    result = reg.promote("v2", baseline_accuracy=0.60)
    assert result.passed
    assert reg.champion == "v2"
    assert reg.challenger is None


def test_promotion_fails_below_accuracy_floor():
    reg = ModelRegistry()
    reg.register(_version("v1", accuracy=0.9), role="champion")
    reg.register(_version("v2", accuracy=0.70, f1=0.65), role="challenger")
    result = reg.promote("v2", baseline_accuracy=0.60)
    assert not result.passed
    assert result.checks["accuracy_floor"] is False
    assert reg.champion == "v1"  # champion unchanged


def test_promotion_fails_when_not_beating_baseline():
    reg = ModelRegistry()
    reg.register(_version("v1", accuracy=0.9), role="champion")
    reg.register(_version("v2", accuracy=0.80, f1=0.80), role="challenger")
    result = reg.promote("v2", baseline_accuracy=0.78)
    assert not result.passed
    assert result.checks["beats_baseline"] is False


def test_promotion_fails_on_weak_slice():
    reg = ModelRegistry()
    reg.register(_version("v1", accuracy=0.9), role="champion")
    reg.register(_version("v2", accuracy=0.90, f1=0.40), role="challenger")
    result = reg.promote("v2", baseline_accuracy=0.60)
    assert not result.passed
    assert result.checks["per_class_f1_floor"] is False


def test_promote_unknown_version_raises():
    reg = ModelRegistry()
    with pytest.raises(ValueError):
        reg.promote("nope", baseline_accuracy=0.5)


def test_rollback_restores_previous_version(trained_version):
    reg = ModelRegistry()
    reg.register(trained_version, role="champion")
    reg.register(_version("v2", accuracy=0.92, f1=0.88), role="challenger")
    reg.promote("v2", baseline_accuracy=0.60)
    assert reg.champion == "v2"
    reg.rollback(trained_version.version)
    assert reg.champion == trained_version.version


def test_list_versions(trained_version):
    reg = ModelRegistry()
    reg.register(trained_version, role="champion")
    versions = reg.list_versions()
    assert len(versions) == 1
    assert versions[0]["version"] == trained_version.version
    assert versions[0]["accuracy"] == pytest.approx(
        trained_version.metrics.accuracy, abs=1e-4)  # summary rounds to 4dp
