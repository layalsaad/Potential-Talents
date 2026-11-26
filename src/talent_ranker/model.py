"""A model trained on the automatic labels, so that fit is a probability between 0 and 1."""

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from talent_ranker.config import MODEL_C
from talent_ranker.embeddings import embed_texts
from talent_ranker.similarity import normalize_text


def tfidf_features(titles: pd.Series) -> tuple[sparse.csr_matrix, np.ndarray]:
    """TF-IDF word weights per title (same settings as the similarity score), and the words."""
    vectorizer = TfidfVectorizer(preprocessor=normalize_text, stop_words="english")
    return vectorizer.fit_transform(titles), vectorizer.get_feature_names_out()


def glove_features(titles: pd.Series, vectors: KeyedVectors) -> sparse.csr_matrix:
    """Average GloVe vector per title, scaled to length 1 like the TF-IDF rows."""
    return sparse.csr_matrix(normalize(embed_texts(titles, vectors)))


def combine_features(*blocks: sparse.csr_matrix) -> sparse.csr_matrix:
    """Put feature blocks side by side, scaled so that every row still has length 1.

    Without the scaling, combined rows are longer, and longer rows alone make logistic
    regression more confident, which would flatter the combination unfairly.
    """
    return (sparse.hstack(blocks) / np.sqrt(len(blocks))).tocsr()


def train_fit_model(
    features, labels: pd.Series, C: float = MODEL_C
) -> LogisticRegression:
    """Train on the rows that have a label (1 or 0); unlabelled rows are ignored."""
    has_label = labels.notna().to_numpy()
    model = LogisticRegression(C=C, max_iter=1000)
    return model.fit(features[has_label], labels[has_label].astype(int).to_numpy())


def predict_fit(model: LogisticRegression, features, index: pd.Index) -> pd.Series:
    """Probability of being a fit, for every row."""
    return pd.Series(model.predict_proba(features)[:, 1], index=index, name="fit")
