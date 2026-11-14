import pandas as pd
import pytest


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """A tiny file in the same shape as the real one, including its quirks."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "job_title": [
                "Aspiring Human Resources Professional",
                "Aspiring Human Resources Professional",
                "HR|\nManager  ",
                "Aspiring Human Resources Professional",
            ],
            "location": [
                "Houston, Texas",
                "Houston, Texas",
                " Kanada",
                "Denton, Texas",
            ],
            "connection": [85, 85, "500+", 85],
            "fit": [None, None, None, None],
        }
    )
