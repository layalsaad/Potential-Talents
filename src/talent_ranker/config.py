from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "potential-talents.xlsx"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REQUIRED_COLUMNS = ["id", "job_title", "location", "connection"]
KEYWORDS = ["aspiring human resources", "seeking human resources"]
RANDOM_STATE = 42