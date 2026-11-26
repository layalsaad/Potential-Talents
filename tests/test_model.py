import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from talent_ranker.evaluation import bootstrap_stability, out_of_fold_fit
from talent_ranker.model import (
    combine_features,
    predict_fit,
    tfidf_features,
    train_fit_model,
)


@pytest.fixture
def titles():
    fits = [
        f"Aspiring Human Resources {w}"
        for w in ["Analyst", "Manager", "Student", "Lead"]
    ]
    others = [f"{w} Teacher" for w in ["English", "Math", "Music", "Art"]]
    return pd.Series(
        [*fits, *others, "Human Resources Generalist"], index=range(10, 19)
    )


@pytest.fixture
def labels(titles):
    # Four fits, four non-fits, and one unlabelled title (the last one).
    return pd.Series([1, 1, 1, 1, 0, 0, 0, 0, pd.NA], index=titles.index, dtype="Int64")


def test_tfidf_features_spell_out_hr_and_drop_stop_words():
    features, words = tfidf_features(pd.Series(["HR Manager at Acme", "Teacher"]))
    assert features.shape == (2, len(words))
    assert {"human", "resources", "manager", "acme", "teacher"} == set(words)


def test_combine_features_keeps_rows_at_length_one():
    a = sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0]])
    b = sparse.csr_matrix([[0.6, 0.8], [1.0, 0.0]])
    combined = combine_features(a, b)
    assert combined.shape == (2, 4)
    assert np.linalg.norm(combined.toarray(), axis=1) == pytest.approx([1.0, 1.0])


def test_model_ignores_unlabelled_rows_and_scores_every_row(titles, labels):
    features, _ = tfidf_features(titles)
    model = train_fit_model(features, labels)
    fit = predict_fit(model, features, titles.index)
    assert model.n_features_in_ == features.shape[1]
    assert fit.index.tolist() == titles.index.tolist()
    assert fit.between(0, 1).all()
    assert (
        fit[18] > fit[[14, 15, 16, 17]].max()
    )  # the unlabelled HR title beats the teachers


def test_out_of_fold_predicts_only_labelled_rows(titles, labels):
    features, _ = tfidf_features(titles)
    oof = out_of_fold_fit(features, labels, groups=titles, n_splits=2)
    assert oof.index.tolist() == titles.index[:8].tolist()
    assert oof.between(0, 1).all()


def test_bootstrap_stability_is_reproducible(titles, labels):
    features, _ = tfidf_features(titles)
    first = bootstrap_stability(features, labels, n_rounds=20, random_state=0)
    second = bootstrap_stability(features, labels, n_rounds=20, random_state=0)
    assert 0 < len(first) <= 20
    assert np.array_equal(first, second)
    assert ((first >= -1) & (first <= 1)).all()
