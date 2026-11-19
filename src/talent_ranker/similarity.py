"""Similarity between job titles and the search keywords."""

import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize_text(text: str) -> str:
    """Lowercase and spell out "HR", so it matches "human resources" in the keywords."""
    return re.sub(r"\bhr\b", "human resources", text.lower())


def tfidf_similarity(
    titles: pd.Series,
    keywords: list[str],
    stop_words: str | None = "english",
    expand_abbreviations: bool = True,
) -> pd.Series:
    """Cosine similarity between each title and its best-matching keyword, from 0 to 1.

    The vocabulary and word weights (IDF) are learned from the titles themselves, so a word
    that appears in many titles ("human") counts less than a rarer one ("aspiring").
    """
    vectorizer = TfidfVectorizer(
        preprocessor=normalize_text if expand_abbreviations else str.lower,
        stop_words=stop_words,
    )
    title_vectors = vectorizer.fit_transform(titles)
    keyword_vectors = vectorizer.transform(keywords)
    scores = cosine_similarity(title_vectors, keyword_vectors).max(axis=1)
    return pd.Series(scores, index=titles.index, name="similarity")


def idf_weights(titles: pd.Series) -> dict[str, float]:
    """IDF weight of every word in the titles: rare words get high weights."""
    vectorizer = TfidfVectorizer(preprocessor=normalize_text, stop_words="english").fit(
        titles
    )
    return dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_, strict=True))
