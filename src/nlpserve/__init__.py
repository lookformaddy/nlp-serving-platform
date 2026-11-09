"""nlpserve — production NLP text-classification serving, built from scratch.

Pure-stdlib core (no sklearn, no numpy): TF-IDF vectorization plus a
from-scratch multinomial logistic-regression classifier, wrapped in the
operational machinery a real serving platform needs — a versioned model
registry with promotion gates, canary + shadow deployment, deterministic
hash A/B routing, latency budgets with a fast keyword-baseline fallback,
input-drift monitoring, and per-prediction token explanations.
"""
from nlpserve.text import normalize, tokenize
from nlpserve.tfidf import TfidfVectorizer
from nlpserve.models import LogisticRegressionClassifier
from nlpserve.corpus import generate_corpus, INTENTS
from nlpserve.registry import ModelRegistry, PromotionGateResult
from nlpserve.serving import ServingLayer, ServingConfig, Prediction
from nlpserve.evaluation import evaluate, ClassificationReport

__all__ = [
    "normalize",
    "tokenize",
    "TfidfVectorizer",
    "LogisticRegressionClassifier",
    "generate_corpus",
    "INTENTS",
    "ModelRegistry",
    "PromotionGateResult",
    "ServingLayer",
    "ServingConfig",
    "Prediction",
    "evaluate",
    "ClassificationReport",
]

__version__ = "0.1.0"
