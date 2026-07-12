"""Text preprocessing pipeline using spaCy and NLTK.

Handles tokenization, stopword removal, punctuation stripping and
lemmatization. The spaCy model is loaded lazily and cached so the first
call pays the model-load cost and subsequent calls reuse it.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

import nltk
import spacy

# ---------------------------------------------------------------------------
# NLTK resource bootstrap (safe to call repeatedly)
# ---------------------------------------------------------------------------

_NLTK_RESOURCES = {
    "tokenizers/punkt": "punkt",
    "tokenizers/punkt_tab": "punkt_tab",
    "corpora/stopwords": "stopwords",
    "corpora/wordnet": "wordnet",
}


def ensure_nltk_resources() -> None:
    """Download required NLTK data packages if they are missing."""
    for resource, package in _NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except Exception:  # pragma: no cover - network errors are non-fatal
                pass


# ---------------------------------------------------------------------------
# spaCy model loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _load_spacy_model(model_name: str = "en_core_web_sm") -> spacy.language.Language:
    """Load (and cache) a spaCy model, disabling unused components for speed."""
    try:
        nlp = spacy.load(model_name, disable=["ner", "parser", "lemmatizer"])
        # Re-enable lemmatizer if available — it's needed for preprocessing.
        if "lemmatizer" not in nlp.pipe_names:
            try:
                nlp.add_pipe("lemmatizer", after="tagger")
            except Exception:
                pass
        return nlp
    except OSError:
        # Fallback: blank tokenizer-only pipeline.
        return spacy.blank("en")


def get_nlp() -> spacy.language.Language:
    """Return a cached spaCy NLP pipeline."""
    ensure_nltk_resources()
    return _load_spacy_model()


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s+\-#.]")
_MULTI_SPACE_RE = re.compile(r"\s+")


def _strip_noise(text: str) -> str:
    """Remove URLs, emails and non-alphabetic noise while preserving tech tokens."""
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    # Keep +, -, # and . so tokens like C++ and .NET survive.
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.lower().strip()


def tokenize(text: str) -> list[str]:
    """Return raw word tokens using NLTK punkt tokenizer."""
    ensure_nltk_resources()
    cleaned = _strip_noise(text)
    try:
        return nltk.word_tokenize(cleaned)
    except Exception:
        return cleaned.split()


def remove_stopwords(tokens: Iterable[str]) -> list[str]:
    """Drop English stopwords and short tokens."""
    try:
        from nltk.corpus import stopwords as nltk_stopwords
        stop = set(nltk_stopwords.words("english"))
    except LookupError:
        stop = set()
    # Always keep a small set of meaningful short tokens.
    keep = {"c", "r", "go", "ai", "ml", "qa", "ux", "ui", "js", "ts", "cs", "it"}
    return [t for t in tokens if (t in keep) or (len(t) > 1 and t not in stop)]


def lemmatize(tokens: Iterable[str]) -> list[str]:
    """Lemmatize tokens using spaCy, falling back to the original token."""
    nlp = get_nlp()
    # Process in a single batch for efficiency.
    doc = nlp(" ".join(tokens))
    lemmas: list[str] = []
    for token in doc:
        lemma = token.lemma_.lower().strip()
        lemmas.append(lemma if lemma else token.text.lower().strip())
    return lemmas


def preprocess(text: str) -> list[str]:
    """Full pipeline: tokenize -> remove stopwords -> lemmatize."""
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    # Final cleanup: drop empties and pure punctuation.
    return [t for t in tokens if t and any(ch.isalpha() for ch in t)]


def preprocess_to_string(text: str) -> str:
    """Return the preprocessed tokens joined as a single string."""
    return " ".join(preprocess(text))
