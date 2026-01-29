import logging
from dataclasses import dataclass, field
from pathlib import Path

# ==================
#  Config Variables
# ==================


DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
RAW_FOLDER = "01_Raw"
CLEANED_FOLDER = "02_Cleaned"
MERGED_FOLDER = "03_Merged"
PREPARED_FOLDER = "04_Prepared"

MERGED_FILE = "merged.csv"
PREPARED_FILE = "prepared.csv"

CSV_ENCODING = "latin1"

SEASON_COL = "Season"

DATE_COL = "Date"

SEASON_START_MONTH = 7

RENAME_MAP = {
    "Dusseldorf": "Fortuna Dusseldorf",
    "Leipzig": "VfB Leipzig",
    "F Koln": "Fortuna Koln",
}


# ================
#  Config Classes
# ================


@dataclass(frozen=True)
class Columns:
    raw: list[str] = field(
        default_factory=lambda: [
            "Div",
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
        ]
    )
    home: list[str] = field(
        default_factory=lambda: [
            "Season",
            "Div",
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
        ]
    )
    away: list[str] = field(
        default_factory=lambda: [
            "Season",
            "Div",
            "Date",
            "AwayTeam",
            "HomeTeam",
            "FTAG",
            "FTHG",
        ]
    )
    team_match: list[str] = field(
        default_factory=lambda: [
            "Season",
            "Div",
            "Date",
            "Team",
            "Opponent",
            "GoalsFor",
            "GoalsAgainst",
        ]
    )
    teams: list[str] = field(default_factory=lambda: ["HomeTeam", "AwayTeam"])
    sort_by: list[str] = field(default_factory=lambda: ["Season", "Div", "Date"])


COLUMNS = Columns()


@dataclass(frozen=True)
class CreationPaths:
    raw: Path = DATA_ROOT / RAW_FOLDER
    cleaned: Path = DATA_ROOT / CLEANED_FOLDER
    merged: Path = DATA_ROOT / MERGED_FOLDER
    prepared: Path = DATA_ROOT / PREPARED_FOLDER
    m_filename: str = MERGED_FILE
    p_filename: str = PREPARED_FILE


PATHS = CreationPaths()


# ===============
# Config Logging
# ===============


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
