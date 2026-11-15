import pandas as pd
import pytest

from talent_ranker.similarity import normalize_text, tfidf_similarity

KEYWORDS = ["aspiring human resources", "seeking human resources"]


def test_normalize_text_spells_out_hr_only_as_a_whole_word():
    assert normalize_text("HR Manager") == "human resources manager"
    assert normalize_text("HRIS Analyst") == "hris analyst"


@pytest.fixture
def titles():
    return pd.Series(
        [
            "Aspiring Human Resources Professional",
            "Seeking Human Resources Position",
            "HR Senior Specialist",
            "Native English Teacher",
        ],
        index=[10, 11, 12, 13],
    )


def test_tfidf_similarity_orders_titles_sensibly(titles):
    scores = tfidf_similarity(titles, KEYWORDS)
    assert scores.index.tolist() == [10, 11, 12, 13]
    assert scores[10] > scores[12] > scores[13]
    assert scores[11] > scores[12]  # matches the second keyword
    assert scores[13] == 0.0
    assert scores.between(0, 1).all()


def test_without_expansion_hr_is_not_recognised(titles):
    scores = tfidf_similarity(titles, KEYWORDS, expand_abbreviations=False)
    assert scores[12] == 0.0
