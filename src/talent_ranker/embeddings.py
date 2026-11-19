"""Pretrained word vectors (GloVe) and title vectors built from them."""

import re

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

from talent_ranker.config import WORD_VECTORS
from talent_ranker.similarity import normalize_text


def load_word_vectors(name: str = WORD_VECTORS) -> KeyedVectors:
    """Download the vectors the first time (cached in ~/gensim-data), then load them."""
    import gensim.downloader  # imported here: only needed when vectors are really loaded

    return gensim.downloader.load(name)


def title_words(
    text: str, expand_abbreviations: bool = True, remove_stop_words: bool = True
) -> list[str]:
    """The lowercase words of a text, with "HR" spelled out and stop words removed."""
    text = normalize_text(text) if expand_abbreviations else text.lower()
    words = re.findall(r"[a-z]+", text)
    return [w for w in words if not (remove_stop_words and w in ENGLISH_STOP_WORDS)]


def text_vector(
    text: str,
    vectors: KeyedVectors,
    weights: dict[str, float] | None = None,
    expand_abbreviations: bool = True,
    remove_stop_words: bool = True,
) -> np.ndarray:
    """Average of the vectors of the words the model knows (all zeros if it knows none).

    With `weights`, each word counts in proportion to its weight (words without one count 1).
    """
    words = [
        w
        for w in title_words(text, expand_abbreviations, remove_stop_words)
        if w in vectors.key_to_index
    ]
    if not words:
        return np.zeros(vectors.vector_size)
    word_weights = np.array(
        [1.0 if weights is None else weights.get(w, 1.0) for w in words]
    )
    return word_weights @ vectors[words] / word_weights.sum()


def embed_texts(texts: pd.Series, vectors: KeyedVectors, **options) -> np.ndarray:
    """One vector per text, stacked into a matrix. Options are passed to `text_vector`."""
    return np.vstack([text_vector(text, vectors, **options) for text in texts])


def embedding_similarity(
    titles: pd.Series, keywords: list[str], vectors: KeyedVectors, **options
) -> pd.Series:
    """Cosine similarity between each title vector and its best-matching keyword vector."""
    title_vectors = embed_texts(titles, vectors, **options)
    keyword_vectors = embed_texts(pd.Series(keywords), vectors, **options)
    scores = cosine_similarity(title_vectors, keyword_vectors).max(axis=1)
    return pd.Series(scores, index=titles.index, name="similarity")
