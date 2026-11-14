"""Project-wide constants: file locations, the input schema and the default search."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "potential-talents.xlsx"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Columns the input file must contain. Note: the file says "connection" (singular).
REQUIRED_COLUMNS = ["id", "job_title", "location", "connection"]

# Default search for this role. The keywords are alternatives ("aspiring" OR "seeking").
KEYWORDS = ["aspiring human resources", "seeking human resources"]

RANDOM_STATE = 42
