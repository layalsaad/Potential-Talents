"""Load, validate, clean and deduplicate the sourced-candidates spreadsheet."""

import re
from pathlib import Path

import pandas as pd

from talent_ranker.config import DATA_PATH, REQUIRED_COLUMNS


def load_candidates(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Read the candidates file, check it, and return one clean row per original id."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}. Place potential-talents.xlsx there (see README)."
        )
    return clean(validate(pd.read_excel(path)))


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Stop with a clear message if the file cannot be ranked safely."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    errors = []
    with_nulls = [col for col in REQUIRED_COLUMNS if df[col].isna().any()]
    if with_nulls:
        errors.append(f"missing values in {with_nulls}")
    if not df["id"].is_unique:
        errors.append("candidate ids must be unique")
    if errors:
        raise ValueError("Invalid candidates file: " + "; ".join(errors))
    return df


def clean_title(title: str) -> str:
    """Turn "|" separators and line breaks into spaces, and collapse repeated spaces."""
    title = re.sub(r"[|\n]+", " ", title)
    return re.sub(r"\s+", " ", title).strip()


def parse_connections(values: pd.Series) -> pd.DataFrame:
    """Convert "500+" and numbers to integers, flagging values that hit the 500 cap."""
    text = values.astype(str).str.strip()
    return pd.DataFrame(
        {
            "connections": pd.to_numeric(text.str.rstrip("+")).astype(int),
            "connections_capped": text.str.endswith("+"),
        },
        index=values.index,
    )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean text columns and parse connections. The empty `fit` column is dropped."""
    out = df[["id", "job_title", "location"]].copy()
    out["job_title"] = out["job_title"].map(clean_title)
    out["location"] = out["location"].str.strip()
    out[["connections", "connections_capped"]] = parse_connections(df["connection"])
    return out


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per unique profile, keeping the list of original ids.

    Rows are the same profile only if title, location AND connections all match, so two
    different people who merely share a job title are kept apart.
    """
    keys = ["job_title", "location", "connections"]
    return df.groupby(keys, sort=False, as_index=False).agg(
        ids=("id", list), connections_capped=("connections_capped", "first")
    )
