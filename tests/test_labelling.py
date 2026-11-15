import pandas as pd
import pytest

from talent_ranker.labelling import auto_label


def test_auto_label_three_zones():
    scores = pd.Series([0.8, 0.3, 0.2, 0.0])
    labels = auto_label(scores, fit_min=0.3, not_fit_max=0.0)
    assert labels.tolist() == [1, 1, pd.NA, 0]


def test_auto_label_rejects_overlapping_thresholds():
    with pytest.raises(ValueError):
        auto_label(pd.Series([0.5]), fit_min=0.3, not_fit_max=0.4)
