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


# Automatic labelling from the TF-IDF similarity score (see notebooks/02_auto_labelling.ipynb).
# At or above FIT_MIN_SCORE -> label 1; at or below NOT_FIT_MAX_SCORE -> label 0;
# in between -> no label (too uncertain to teach a model).
FIT_MIN_SCORE = 0.3
NOT_FIT_MAX_SCORE = 0.0
