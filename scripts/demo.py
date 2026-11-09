"""End-to-end demo: train, evaluate, promote, serve a few examples.

Trains a champion (v1) and a challenger (v2, different seed), runs the
offline eval, checks the promotion gates, then serves example tickets
through the full serving layer — canary routing, shadow scoring,
latency budgets, drift flags, and token explanations included.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nlpserve.evaluation import keyword_baseline_predict  # noqa: E402
from nlpserve.pipeline import train_version  # noqa: E402
from nlpserve.registry import ModelRegistry  # noqa: E402
from nlpserve.serving import ServingConfig, ServingLayer  # noqa: E402

EXAMPLES = [
    ("I was charged twice for my subscription this month", "billing"),
    ("I cannot log in, my password is not working", "account_access"),
    ("My order has not arrived yet and tracking never updated", "shipping"),
    ("Please refund the faulty item I returned last week", "refund"),
    ("The app crashes every time I try to open it", "technical_issue"),
    ("I love the new design, great work on the update", "feedback"),
    # Out-of-distribution: vocabulary the model never saw.
    ("Zqxwv blorpt flibberty gibbet quantum flux capacitor", None),
]


def main() -> int:
    print("== nlp-serving-platform demo ==\n")

    champion, report = train_version("v1-seed42", seed=42)
    metrics = report["metrics"]
    print(f"[train] champion v1-seed42: {report['n_train']} train / "
          f"{report['n_test']} test, vocab={report['vocab_size']}, "
          f"final loss={report['final_loss']:.4f}")
    print(metrics.summary())
    print(f"[eval] keyword baseline accuracy: "
          f"{report['baseline_accuracy']:.4f}")
    print(metrics.confusion_table())

    challenger, report2 = train_version("v2-seed7", seed=7)
    print(f"\n[train] challenger v2-seed7: "
          f"accuracy={report2['metrics'].accuracy:.4f} "
          f"macro_f1={report2['metrics'].macro_f1:.4f}")

    registry = ModelRegistry()
    registry.register(champion, role="champion")
    registry.register(challenger, role="challenger")
    result = registry.promote("v2-seed7", report["baseline_accuracy"])
    print(f"\n[promote] challenger v2-seed7:\n{result}")

    # A fresh challenger (v3) takes the challenger slot so the serve
    # step exercises canary routing + shadow scoring for real.
    challenger_v3, report3 = train_version("v3-seed99", seed=99)
    registry.register(challenger_v3, role="challenger")
    print(f"[train] new challenger v3-seed99: "
          f"accuracy={report3['metrics'].accuracy:.4f} "
          f"macro_f1={report3['metrics'].macro_f1:.4f}")

    serving = ServingLayer(registry, ServingConfig(canary_pct=25))
    print("\n[serve] example tickets (canary 25%):")
    for i, (text, expected) in enumerate(EXAMPLES):
        pred = serving.predict(text, request_id=f"demo-{i}")
        tokens = ", ".join(f"{t} ({w:+.2f})"
                           for t, w in pred.explanation[:3])
        mark = ""
        if expected is not None:
            mark = " ✓" if pred.label == expected else f" ✗ (expected {expected})"
        print(f"  {pred.label:<16} conf={pred.confidence:.2f} "
              f"via={pred.served_by:<10} {pred.latency_ms:5.1f}ms "
              f"oov={pred.drift_oov_rate:.0%} drift={pred.drift_flag} "
              f"[{tokens}]{mark}")

    print("\n[stats]", serving.stats())
    print("\nDemo complete: train -> evaluate -> promote -> serve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
