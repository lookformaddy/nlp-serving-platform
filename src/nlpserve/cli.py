"""Command-line interface: train, evaluate, predict, serve.

Examples:
    python -m nlpserve.cli train --version v1 --seed 42
    python -m nlpserve.cli eval --version v1
    python -m nlpserve.cli predict "I was charged twice this month"
    python -m nlpserve.cli serve --port 8000
"""
from __future__ import annotations

import argparse
import sys

from nlpserve.pipeline import train_version
from nlpserve.registry import ModelRegistry
from nlpserve.serving import ServingConfig, ServingLayer

DEFAULT_VERSION = "v1"


def _build_registry(seed: int = 42) -> tuple[ModelRegistry, ServingLayer]:
    version, report = train_version(DEFAULT_VERSION, seed=seed)
    registry = ModelRegistry()
    registry.register(version, role="champion")
    serving = ServingLayer(registry)
    return registry, serving


def cmd_train(args: argparse.Namespace) -> int:
    version, report = train_version(args.version, seed=args.seed,
                                    epochs=args.epochs)
    metrics = report["metrics"]
    print(f"trained {args.version}: {report['n_train']} train / "
          f"{report['n_test']} test, vocab={report['vocab_size']}, "
          f"final loss={report['final_loss']:.4f}")
    print(metrics.summary())
    print(f"keyword baseline accuracy: {report['baseline_accuracy']:.4f}")
    print(metrics.confusion_table())
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    version, report = train_version(args.version, seed=args.seed)
    print(report["metrics"].summary())
    print(f"baseline accuracy: {report['baseline_accuracy']:.4f}")
    print(report["metrics"].confusion_table())
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    _registry, serving = _build_registry(seed=args.seed)
    result = serving.predict(args.text, request_id=args.request_id)
    print(f"label      : {result.label}")
    print(f"confidence : {result.confidence:.3f}")
    print(f"served by  : {result.served_by}"
          f" ({result.model_version or 'keyword-baseline'})")
    print(f"latency    : {result.latency_ms:.1f} ms")
    if result.fallback_reason:
        print(f"fallback    : {result.fallback_reason}")
    print(f"drift OOV  : {result.drift_oov_rate:.2%} "
          f"(flagged={result.drift_flag})")
    if result.explanation:
        print("top tokens : " + ", ".join(
            f"{tok} ({w:+.2f})" for tok, w in result.explanation))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn  # noqa: PLC0415
    except ImportError:
        print("uvicorn is required to serve: pip install -r requirements.txt",
              file=sys.stderr)
        return 1
    from nlpserve.service import build_app  # noqa: PLC0415
    _registry, serving = _build_registry(seed=args.seed)
    app = build_app(serving)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nlpserve",
        description="Production NLP text-classification serving platform.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train a version and print eval")
    p_train.add_argument("--version", default=DEFAULT_VERSION)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--epochs", type=int, default=80)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="train and evaluate a version")
    p_eval.add_argument("--version", default=DEFAULT_VERSION)
    p_eval.add_argument("--seed", type=int, default=42)
    p_eval.set_defaults(func=cmd_eval)

    p_pred = sub.add_parser("predict", help="classify one text")
    p_pred.add_argument("text")
    p_pred.add_argument("--request-id", default=None)
    p_pred.add_argument("--seed", type=int, default=42)
    p_pred.set_defaults(func=cmd_predict)

    p_serve = sub.add_parser("serve", help="start the FastAPI service")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--seed", type=int, default=42)
    p_serve.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
