# nlp-serving-platform

Production NLP text-classification serving, built from scratch — no
sklearn, no numpy. A TF-IDF vectorizer plus a from-scratch multinomial
logistic-regression classifier for multi-class intent classification on
support-ticket-style text, wrapped in the operational machinery a real
serving platform needs: a versioned model registry with promotion
gates, canary + shadow deployment, deterministic hash A/B routing,
per-request latency budgets with a fast keyword-baseline fallback,
input-drift monitoring, and a token-level explanation for every
prediction.

## The enterprise problem

Support teams route thousands of short, noisy customer messages a day.
A classifier that only works in a notebook is not a product. It needs
versioned models with safe rollout, a fallback when it is slow or
wrong, drift detection when the language shifts, and an explanation for
every decision an agent acts on. This project builds that whole serving
surface around a classifier whose every line is inspectable.

## What it demonstrates

- **From-scratch ML:** TF-IDF (sublinear TF, smoothed IDF, 1–2 grams)
  and multinomial logistic regression trained with full-batch gradient
  descent — pure stdlib, sparse-dict math, no ML framework.
- **Model lifecycle:** versioned registry, champion/challenger, and
  promotion gates (accuracy floor, must-beat-baseline margin,
  per-class F1 floor) with a full decision history and rollback.
- **Safe rollout:** deterministic `sha256(request_id)` canary routing,
  shadow scoring with agreement rate, and a latency budget that falls
  back to a keyword baseline instead of failing the request.
- **Observability:** per-request OOV drift scoring, serving stats
  (arms, fallbacks, shadow agreement, drift), and token-level
  explanations (`tfidf × weight` per contributing token).
- **Honest evaluation:** accuracy, per-class precision/recall/F1, and
  a confusion matrix on held-out data, benchmarked against the keyword
  baseline the promotion gates must beat.

## Architecture overview

```
raw text → normalize/tokenize → TF-IDF (sparse dict)
        → logistic regression (softmax, L2)
        → ModelRegistry (champion/challenger + promotion gates)
        → ServingLayer (drift → hash A/B routing → canary/shadow
                        → latency budget → keyword fallback)
        → FastAPI service (/health /predict /models /stats /promote)
          or CLI (train / eval / predict / serve)
```

See [docs/architecture.md](docs/architecture.md) for the full diagram,
design decisions, and trade-offs.

## How to run it

```bash
# Install (core library needs only the stdlib; FastAPI/pytest are optional)
pip install -r requirements.txt

# Run the test suite (API tests auto-skip if FastAPI is absent)
make test

# End-to-end demo: train, evaluate, promote, serve examples
make demo

# Classify one message
PYTHONPATH=src python -m nlpserve.cli predict "I was charged twice this month"

# Train and evaluate a version
PYTHONPATH=src python -m nlpserve.cli train --version v1 --seed 42

# Start the API
PYTHONPATH=src python -m nlpserve.cli serve --port 8000
# curl -X POST localhost:8000/predict \
#   -H 'Content-Type: application/json' \
#   -d '{"text": "My order has not arrived yet", "request_id": "req-123"}'

# Docker
docker build -t nlp-serving-platform . && docker run -p 8000:8000 nlp-serving-platform
```

## Key design decisions and trade-offs

- **Stdlib-only core:** zero binary-dependency risk in the serving
  image and fully auditable code; the trade is slower training than a
  numpy build (fine at this corpus size, mini-batching is the known
  next step past ~100k documents).
- **Sparse dict vectors:** short ticket text touches ~10–30 features,
  so scoring is a handful of multiply-adds per class.
- **Deterministic hash routing:** the canary arm is stable across
  retries for the same request id, which is what makes A/B results
  interpretable; requests without an id always get the champion.
- **Gates, not auto-promotion:** a challenger must clear every gate
  before it becomes champion, and every attempt is recorded.
- **Fallback as a feature:** over-budget or erroring model calls fall
  back to the keyword baseline — degraded but always available — and
  every fallback is counted in `/stats`.
- **Shadow before canary:** the challenger first scores champion
  traffic in shadow mode (agreement recorded, result discarded), so
  behavioral regressions surface before any user sees the new model.

## Data

All text is synthetic, generated deterministically from hand-written
templates with slot fillers (`src/nlpserve/corpus.py`), across six
intents: `billing`, `account_access`, `shipping`, `refund`,
`technical_issue`, `feedback`. A configurable label-noise rate flips a
small fraction of training labels to mimic annotation noise; test
labels are always clean. No real customer data, no employer data, no
external datasets.

## Layout

```
src/nlpserve/
  text.py        normalization + tokenization
  tfidf.py       TF-IDF vectorizer
  models.py      logistic regression from scratch
  corpus.py      seeded synthetic corpus generator
  evaluation.py  metrics, confusion matrix, keyword baseline
  registry.py    versioned registry + promotion gates
  serving.py     canary/shadow, routing, latency, fallback, drift
  pipeline.py    train_version(): corpus -> model -> eval
  cli.py         train / eval / predict / serve
  service.py     FastAPI app
scripts/demo.py  end-to-end demo
tests/           pytest suite
docs/            architecture.md, github-actions-ci.yml
```

The CI definition ships as `docs/github-actions-ci.yml` — move it to
`.github/workflows/ci.yml` to activate it (the push integration used
for this repo cannot write workflow files directly).
