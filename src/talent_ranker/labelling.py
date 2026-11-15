"""Automatic labels from a similarity score, so labelling scales to any number of candidates."""

import pandas as pd

from talent_ranker.config import FIT_MIN_SCORE, NOT_FIT_MAX_SCORE


def auto_label(
    scores: pd.Series,
    fit_min: float = FIT_MIN_SCORE,
    not_fit_max: float = NOT_FIT_MAX_SCORE,
) -> pd.Series:
    """Label 1 (fit) for high scores, 0 (not a fit) for low ones, and <NA> in between.

    Candidates in the uncertain middle get no label, so a model trained on these labels is
    not taught guesses; the model scores them instead.
    """
    if not_fit_max >= fit_min:
        raise ValueError("not_fit_max must be lower than fit_min")
    labels = pd.Series(pd.NA, index=scores.index, dtype="Int64", name="auto_label")
    labels[scores >= fit_min] = 1
    labels[scores <= not_fit_max] = 0
    return labels
