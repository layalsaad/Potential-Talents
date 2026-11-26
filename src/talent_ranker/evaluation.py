"""Honest checks for a model trained on few, automatic labels."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from talent_ranker.config import MODEL_C, RANDOM_STATE


def out_of_fold_fit(
    features,
    labels: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
    C: float = MODEL_C,
    random_state: int = RANDOM_STATE,
) -> pd.Series:
    """Fit probability for each labelled row, predicted by a model that never saw its group.

    Rows in the same group (for example the same job title) always stay together, so a
    title can never be in training and testing at the same time.
    """
    has_label = labels.notna().to_numpy()
    y = labels[has_label].astype(int).to_numpy()
    cv = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    proba = cross_val_predict(
        LogisticRegression(C=C, max_iter=1000),
        features[has_label],
        y,
        groups=groups[has_label].to_numpy(),
        cv=cv,
        method="predict_proba",
    )[:, 1]
    return pd.Series(proba, index=labels.index[has_label], name="fit")


def bootstrap_stability(
    features,
    labels: pd.Series,
    n_rounds: int = 200,
    C: float = MODEL_C,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """How much the ranking moves when the model is trained on resampled labels.

    Each round draws the labelled rows with replacement, retrains, and compares the new
    ranking of all rows with the full model's ranking (Spearman: 1 = identical order).
    """
    has_label = labels.notna().to_numpy()
    full_model = LogisticRegression(C=C, max_iter=1000).fit(
        features[has_label], labels[has_label].astype(int).to_numpy()
    )
    full = pd.Series(full_model.predict_proba(features)[:, 1])
    labelled = np.flatnonzero(has_label)
    rng = np.random.default_rng(random_state)
    correlations = []
    for _ in range(n_rounds):
        sample = rng.choice(labelled, size=len(labelled), replace=True)
        y = labels.iloc[sample].astype(int).to_numpy()
        if len(set(y)) < 2:  # a resample with one class cannot train a model
            continue
        model = LogisticRegression(C=C, max_iter=1000).fit(features[sample], y)
        resampled = pd.Series(model.predict_proba(features)[:, 1])
        correlations.append(full.corr(resampled, method="spearman"))
    return np.array(correlations)
