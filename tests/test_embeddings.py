import numpy as np
import pandas as pd
import pytest
from gensim.models import KeyedVectors

from talent_ranker.embeddings import embedding_similarity, text_vector, title_words

KEYWORDS = ["aspiring human resources"]


@pytest.fixture
def vectors():
    """Tiny made-up word vectors: HR words point one way, teaching words the other."""
    kv = KeyedVectors(vector_size=2)
    kv.add_vectors(
        ["human", "resources", "aspiring", "people", "teacher", "english"],
        np.array(
            [[1, 0], [1, 0], [0.8, 0.2], [0.9, 0.1], [0, 1], [0, 1]], dtype=np.float32
        ),
    )
    return kv


def test_title_words_spells_out_hr_and_drops_stop_words():
    assert title_words("HR Manager at Acme") == [
        "human",
        "resources",
        "manager",
        "acme",
    ]
    assert title_words(
        "HR at Acme", expand_abbreviations=False, remove_stop_words=False
    ) == [
        "hr",
        "at",
        "acme",
    ]


def test_text_vector_averages_known_words_only(vectors):
    # "zzz" is unknown and ignored; the mean of human [1, 0] and teacher [0, 1]:
    assert text_vector("human teacher zzz", vectors) == pytest.approx([0.5, 0.5])


def test_text_vector_is_zero_when_no_word_is_known(vectors):
    assert text_vector("Loparex", vectors) == pytest.approx([0.0, 0.0])


def test_text_vector_weights(vectors):
    weighted = text_vector("human teacher", vectors, weights={"human": 3.0})
    assert weighted == pytest.approx([0.75, 0.25])


def test_embedding_similarity_recognises_meaning(vectors):
    titles = pd.Series(["People", "English Teacher", "Loparex"], index=[7, 8, 9])
    scores = embedding_similarity(titles, KEYWORDS, vectors)
    assert scores.index.tolist() == [7, 8, 9]
    assert (
        scores[7] > 0.9
    )  # "people" is close to "human resources" without sharing a word
    assert scores[8] < scores[7]
    assert scores[9] == 0.0  # no known word
