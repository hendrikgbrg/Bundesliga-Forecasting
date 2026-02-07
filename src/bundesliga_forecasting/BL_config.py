import logging
from dataclasses import dataclass
from pathlib import Path

# ==================
#  Config Variables
# ==================

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
RAW_FOLDER = "01_Raw"
CLEANED_FOLDER = "02_Cleaned"
MERGED_FOLDER = "03_Merged"
PREPARED_FOLDER = "04_Prepared"
FEATURE_FOLDER = "05_Features"
MERGED_FILE = "merged.csv"
PREPARED_FILE = "prepared.csv"
FEATURE_FILE = "features.csv"

CSV_ENCODING = "latin1"
DATE_COL = "Date"


# ================
#  Config Paths
# ================


@dataclass(frozen=True)
class Paths:
    raw: Path = DATA_ROOT / RAW_FOLDER
    cleaned: Path = DATA_ROOT / CLEANED_FOLDER
    merged: Path = DATA_ROOT / MERGED_FOLDER
    prepared: Path = DATA_ROOT / PREPARED_FOLDER
    features: Path = DATA_ROOT / FEATURE_FOLDER
    m_filename: str = MERGED_FILE
    p_filename: str = PREPARED_FILE
    f_filename: str = FEATURE_FILE


PATHS = Paths()


# ================
#  Config Logging
# ================


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="\n%(asctime)s | %(levelname)s | %(name)s | %(message)s\n",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
