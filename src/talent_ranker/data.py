"""Load, validate, clean and deduplicate the sourced-candidates spreadsheet."""

import re
from pathlib import Path

import pandas as pd

from talent_ranker.config import DATA_PATH, REQUIRED_COLUMNS


def load_candidates(path: Path | str = DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_excel(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not df["id"].is_unique:
        raise ValueError("Candidate ids must be unique")
    return clean(df)


def clean_title(title: str) -> str:
    title = re.sub(r"[|\n]+", " ", title)  # "|" separators and line breaks → spaces
    return re.sub(r"\s+", " ", title).strip()  # collapse repeated spaces


def parse_connections(values: pd.Series) -> pd.DataFrame:
    text = values.astype(str).str.strip()
    return pd.DataFrame(
        {
            "connections": pd.to_numeric(text.str.rstrip("+")).astype(int),
            "connections_capped": text.str.endswith("+"),
        },
        index=values.index,
    )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["id", "job_title", "location"]].copy()
    out["job_title"] = out["job_title"].map(clean_title)
    out["location"] = out["location"].str.strip()
    out[["connections", "connections_capped"]] = parse_connections(df["connection"])
    return out


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """One row per unique profile, keeping every original id."""
    keys = ["job_title", "location", "connections"]
    return df.groupby(keys, sort=False, as_index=False).agg(
        ids=("id", list), connections_capped=("connections_capped", "first")
    )