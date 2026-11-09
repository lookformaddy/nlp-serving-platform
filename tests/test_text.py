"""Tests for text preprocessing."""
import pytest

from nlpserve.text import STOPWORDS, normalize, tokenize


def test_normalize_lowercases_and_strips_punctuation():
    assert normalize("Hello, WORLD!  How are you?") == "hello world how are you"


def test_normalize_expands_contractions():
    assert normalize("I can't log in") == "i cannot log in"


def test_normalize_rejects_non_string():
    with pytest.raises(TypeError):
        normalize(123)


def test_tokenize_removes_stopwords_by_default():
    tokens = tokenize("I cannot log in to my account")
    assert "cannot" in tokens
    assert "log" in tokens
    assert "i" not in tokens
    assert "my" not in tokens


def test_tokenize_keeps_stopwords_when_asked():
    tokens = tokenize("I cannot log in to my account", remove_stopwords=False)
    assert "in" in tokens  # "in" is a stopword; kept when asked
    assert "to" in tokens
    assert "cannot" in tokens


def test_tokenize_empty_input_returns_empty_list():
    assert tokenize("") == []
    assert tokenize("   !!! ") == []


def test_tokenize_drops_short_tokens():
    assert "a" not in tokenize("a big refund", min_len=2)
    assert "big" in tokenize("a big refund", min_len=2)


def test_stopwords_are_lowercase_words():
    assert "the" in STOPWORDS
    assert all(w == w.lower() for w in STOPWORDS)
