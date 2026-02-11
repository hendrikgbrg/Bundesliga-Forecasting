import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import cast

# ==================
#  Config Variables
# ==================

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
TEST_FOLDER = Path(__file__).resolve().parents[2] / "tests"
RAW_FOLDER = "01_Raw"
CLEANED_FOLDER = "02_Cleaned"
MERGED_FOLDER = "03_Merged"
PREPARED_FOLDER = "04_Prepared"
FEATURE_FOLDER = "05_Features"
MERGED_FILE = "merged.csv"
PREPARED_FILE = "prepared.csv"
FEATURE_FILE = "features.csv"
DIFF_FEATURE_FILE = "diff_features.csv"
T_FILE = "test.csv"

CSV_ENCODING = "latin1"


# ================
#  Config DataClasses
# ================


@dataclass(frozen=True)
class Paths:
    raw: Path = DATA_ROOT / RAW_FOLDER
    cleaned: Path = DATA_ROOT / CLEANED_FOLDER
    merged: Path = DATA_ROOT / MERGED_FOLDER
    prepared: Path = DATA_ROOT / PREPARED_FOLDER
    features: Path = DATA_ROOT / FEATURE_FOLDER
    test: Path = TEST_FOLDER
    m_filename: str = MERGED_FILE
    p_filename: str = PREPARED_FILE
    f_filename: str = FEATURE_FILE
    d_filename: str = DIFF_FEATURE_FILE
    t_filename: str = T_FILE


PATHS = Paths()


@dataclass(frozen=True)
class Columns:
    # general
    season: str = "Season"
    div: str = "Div"
    home: str = "Home"
    date: str = "Date"
    team: str = "Team"
    opp: str = "Opponent"

    # team-match level
    goalsf: str = "GoalsFor"
    goalsa: str = "GoalsAgainst"
    goaldiff: str = "GoalDiff"
    points: str = "Points"

    # post-match cumulation
    post_tgoalsf: str = "PostTotalGoalsFor"
    post_tgoalsa: str = "PostTotalGoalsAgainst"
    post_tgoaldiff: str = "PostTotalGoalDiff"
    post_tpoints: str = "PostTotalPoints"
    post_min_tpoints: str = "PostMinTotalPoints"
    post_max_tpoints: str = "PostMaxTotalPoints"
    post_rank: str = "PostRank"
    post_trank: str = "PostTotalRank"
    post_trank_performance: str = "PostTotalRankPerformance"
    post_tpoint_performance: str = "PostTotalPointPerformance"
    seasonal_win_loss_ratio: str = "SeasonalWinLossRatio"
    seasonal_win_ratio: str = "SeasonalWinRatio"
    seasonal_goal_superiority: str = "SeasonalGoalSuperiority"

    # prior-match cumulation
    prev_tgoalsf: str = "PrevTotalGoalsFor"
    prev_tgoalsa: str = "PrevTotalGoalsAgainst"
    prev_goaldiff: str = "PrevGoalDiff"
    prev_tgoaldiff: str = "PrevTotalGoalDiff"
    prev_tpoints: str = "PrevTotalPoints"
    prev_min_tpoints: str = "PrevMinTotalPoints"
    prev_max_tpoints: str = "PrevMaxTotalPoints"
    prev_rank: str = "PrevRank"
    prev_trank: str = "PrevTotalRank"
    prev_win_loss_ratio: str = "PrevWinLossRatio"
    prev_hist_win_loss_ratio: str = "PrevHistWinLossRatio"

    # potential predictor columns
    div: str = "Div"
    home: str = "Home"
    zone: str = "Zone"
    prev_goal_superiority: str = "PrevGoalSuperiority"
    prev_win_ratio: str = "PrevWinRatio"
    prev_trank_performance: str = "PrevTotalRankPerformance"
    prev_tpoint_performance: str = "PrevTotalPointPerformance"
    prev_hist_goal_superiority: str = "PrevHistoricalGoalSuperiority"
    prev_hist_win_ratio: str = "PrevHistWinRatio"
    prev_hist_trank_performance: str = "PrevHistoricalTotalRankPerformance"
    prev_hist_tpoint_performance: str = "PrevHistoricalTotalPointPerformance"


COLUMNS = Columns()


@dataclass(frozen=True)
class Predictors:
    home: str = "Home"
    zone: str = "Zone"
    prev_goal_superiority: str = "PrevGoalSuperiority"
    prev_win_ratio: str = "PrevWinRatio"
    prev_trank_performance: str = "PrevTotalRankPerformance"
    prev_tpoint_performance: str = "PrevTotalPointPerformance"
    prev_hist_goal_superiority: str = "PrevHistoricalGoalSuperiority"
    prev_hist_win_ratio: str = "PrevHistWinRatio"
    prev_hist_trank_performance: str = "PrevHistoricalTotalRankPerformance"
    prev_hist_tpoint_performance: str = "PrevHistoricalTotalPointPerformance"

    @classmethod
    def values(cls) -> list[str]:
        return [cast(str, field.default) for field in fields(cls)]


PREDICTORS = Predictors


# ================
#  Config Logging
# ================


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="\n%(asctime)s | %(levelname)s | %(name)s | %(message)s\n",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
