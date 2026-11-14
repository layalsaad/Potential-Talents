import pandas as pd
import pytest

from talent_ranker.data import (
    clean,
    clean_title,
    deduplicate,
    load_candidates,
    parse_connections,
    validate,
)


def test_parse_connections_handles_cap():
    parsed = parse_connections(pd.Series(["500+", 85, " 12 "]))
    assert parsed["connections"].tolist() == [500, 85, 12]
    assert parsed["connections_capped"].tolist() == [True, False, False]


def test_clean_title_removes_separators_and_extra_spaces():
    assert clean_title("HR|\nManager  ") == "HR Manager"
    assert clean_title("Seeking Human  Resources") == "Seeking Human Resources"


def test_clean_keeps_ids_and_drops_fit(raw_df):
    cleaned = clean(raw_df)
    assert cleaned["id"].tolist() == [1, 2, 3, 4]
    assert "fit" not in cleaned.columns
    assert cleaned.loc[2, "location"] == "Kanada"


def test_deduplicate_merges_exact_duplicates_only(raw_df):
    profiles = deduplicate(clean(raw_df))
    # ids 1 and 2 are identical; id 4 has the same title but another location.
    assert len(profiles) == 3
    assert profiles["ids"].tolist() == [[1, 2], [3], [4]]


def test_validate_reports_missing_columns(raw_df):
    with pytest.raises(ValueError, match="Missing required columns"):
        validate(raw_df.drop(columns="connection"))


def test_validate_reports_duplicate_ids_and_nulls(raw_df):
    bad = raw_df.copy()
    bad.loc[1, "id"] = 1
    bad.loc[0, "job_title"] = None
    with pytest.raises(ValueError, match="missing values.*ids must be unique"):
        validate(bad)


def test_load_candidates_reads_excel(raw_df, tmp_path):
    path = tmp_path / "candidates.xlsx"
    raw_df.to_excel(path, index=False)
    loaded = load_candidates(path)
    assert len(loaded) == 4
    assert loaded["connections"].tolist() == [85, 85, 500, 85]


def test_load_candidates_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_candidates(tmp_path / "nope.xlsx")
