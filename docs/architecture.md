# Architecture — nlp-serving-platform

## Problem

Support teams route thousands of short, noisy customer messages a day.
A classifier that only works in a notebook is not a product: it needs
versioned models, safe rollout, a fallback when it is slow or wrong,
drift detection when the language shifts, and an explanation for every
decision an agent acts on. This project builds that whole serving
surface around a from-scratch text classifier — no sklearn, no numpy —
so every layer is inspectable.

## Diagram (description)

```
                ┌─────────────────────┐
   raw text ──▶ │  text.py            │  normalize, tokenize, stopwords
                │  (preprocessing)    │
                └─────────┬───────────┘
                          ▼
                ┌─────────────────────┐
                │  tfidf.py           │  sublinear TF, smoothed IDF,
                │  TfidfVectorizer    │  1-2 grams, L2-normalized
                └─────────┬───────────┘  sparse dicts
                          ▼
                ┌─────────────────────┐
                │  models.py          │  multinomial logistic regression
                │  LogisticRegression │  full-batch gradient descent,
                │  Classifier         │  softmax + L2, from scratch
                └─────────┬───────────┘
                          ▼
   ┌──────────────────────────────────────────────────────┐
   │  registry.py — ModelRegistry                         │
   │  versions (v1, v2, ...) · champion / challenger      │
   │  promotion gates: accuracy floor · beats-baseline    │
   │  margin · per-class F1 floor · full decision history │
   └──────────────────────┬───────────────────────────────┘
                          ▼
   ┌──────────────────────────────────────────────────────┐
   │  serving.py — ServingLayer                           │
   │  drift monitor (OOV rate) ──▶ hash A/B routing      │
   │  (sha256(request_id) % 100 < canary_pct)              │
   │        ├─ champion / challenger arm                  │
   │        ├─ shadow scoring of the challenger +         │
   │        │  agreement rate (never returned)            │
   │        └─ latency budget ──▶ keyword-baseline        │
   │           fallback (fast, always available)          │
   │  every response: label, confidence, top-token        │
   │  explanation, arm, latency, drift flags              │
   └──────────────────────┬───────────────────────────────┘
                          ▼
        ┌─────────────────┴─────────────────┐
        │ service.py (FastAPI)              │ cli.py
        │ /health /predict /models          │ train / eval /
        │ /stats /promote                   │ predict / serve
        └───────────────────────────────────┘
   evaluation.py — accuracy, per-class P/R/F1, confusion matrix,
   keyword-baseline comparison (the gate's floor and the fallback)
```

## Key design decisions

1. **Stdlib-only core.** The classifier, vectorizer, and routing are
   pure Python. That is a deliberate trade: training is slower than a
   numpy build, but the code is auditable line by line, has zero
   binary-dependency risk in the serving image, and the sparse-dict
   representation is exactly what a low-latency serving path wants.
2. **Sparse dict vectors.** TF-IDF vectors are `{feature_index: weight}`
   dicts, not dense arrays. Short ticket text touches ~10–30 features,
   so scoring is a handful of multiply-adds per class.
3. **Deterministic hash routing.** `sha256(request_id) % 100` makes the
   canary arm stable across retries for the same request — the property
   that makes A/B results interpretable. No request id means champion,
   so ad-hoc traffic never leaks into an experiment.
4. **Promotion gates, not auto-promotion.** A challenger must clear an
   accuracy floor, beat the keyword baseline by a margin, and pass a
   per-class F1 floor before it can become champion. Every attempt is
   logged with its gate results, and `rollback()` restores any prior
   version.
5. **Fallback is a feature, not an exception path.** When the model
   path exceeds its latency budget or raises, the keyword baseline
   answers instead — degraded but always available — and the fallback
   is counted and reported in `/stats`.
6. **Shadow before canary.** A challenger first scores champion
   traffic in shadow mode (agreement rate recorded, result discarded);
   only then does canary traffic reach it. That ordering catches
   behavioral regressions offline metrics miss.
7. **Drift as a first-class signal.** Every request is scored for
   out-of-vocabulary rate against the training vocabulary; requests
   above the threshold are flagged on the response and aggregated in
   the drift summary, so vocabulary shift is visible before accuracy
   degrades.

## Trade-offs

- Full-batch gradient descent converges reliably on this corpus size
  but would need mini-batching (or a different optimizer) past ~100k
  documents; the training loop is the known scaling bottleneck.
- The keyword baseline is intentionally weak — it is a floor and a
  fallback, not a competitor. Its accuracy (~55–65%) sets the bar the
  learned model must clear by a margin.
- Explanations are linear token contributions (`tfidf * weight`). They
  are faithful for this linear model by construction, but they do not
  capture interactions between tokens.

## Data

All training and test text is synthetic, generated deterministically
from hand-written templates with slot fillers (see `corpus.py`). A
configurable label-noise rate flips a small fraction of training
labels to mimic annotation noise; test labels are always clean.
No real customer data, no employer data, no external datasets.

## Running it

```bash
pip install -r requirements.txt
make test        # pytest suite (API tests skip without FastAPI)
make demo        # train -> evaluate -> promote -> serve examples
python -m nlpserve.cli predict "I was charged twice this month"
python -m nlpserve.cli serve --port 8000   # then POST /predict
docker build -t nlp-serving-platform . && docker run -p 8000:8000 nlp-serving-platform
```

## Layout

```
src/nlpserve/
  text.py        normalization + tokenization
  tfidf.py       TF-IDF vectorizer (sublinear TF, smoothed IDF)
  models.py      multinomial logistic regression from scratch
  corpus.py      seeded synthetic intent corpus generator
  evaluation.py  metrics, confusion matrix, keyword baseline
  registry.py    versioned model registry + promotion gates
  serving.py     canary/shadow, hash A/B routing, latency budget,
                 fallback, drift monitor, explanations
  pipeline.py    train_version(): corpus -> model -> eval
  cli.py         train / eval / predict / serve
  service.py     FastAPI app
scripts/demo.py  end-to-end demo
tests/           pytest suite (API tests auto-skip without FastAPI)
docs/            architecture.md, github-actions-ci.yml
```
