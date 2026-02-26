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
ELNET_FOLDER = "06_Elastic-Net_Selection"

MERGED_FILE = "merged.csv"
PREPARED_FILE = "prepared.csv"
FEATURE_FILE = "features.csv"
DAILY_TABLES_FILE = "daily_tables.csv"
COMBINED_FEATURE_FILE = "combined_features.csv"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
VALID_FILE = "valid.csv"


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
    elnet: Path = DATA_ROOT / ELNET_FOLDER
    test: Path = TEST_FOLDER
    merged_file: str = MERGED_FILE
    prepared_file: str = PREPARED_FILE
    feature_file: str = FEATURE_FILE
    daily_tables_file: str = DAILY_TABLES_FILE
    combined_file: str = COMBINED_FEATURE_FILE
    train_file: str = TRAIN_FILE
    test_file: str = TEST_FILE
    valid_file: str = VALID_FILE


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
    post_twins: str = "PostTotalWins"
    post_tlosses: str = "PostTotalLosses"
    post_tdraws: str = "PostTotalDraws"
    post_tgoalsf: str = "PostTotalGoalsFor"
    post_tgoalsa: str = "PostTotalGoalsAgainst"
    post_tgoaldiff: str = "PostTotalGoalDiff"
    post_tpoints: str = "PostTotalPoints"
    post_min_tpoints: str = "PostMinTotalPoints"
    post_max_tpoints: str = "PostMaxTotalPoints"
    post_rank: str = "PostRank"
    post_trank: str = "PostTotalRank"
    post_tpoint_performance: str = "PostTotalPointPerformance"
    post_win_ratio: str = "PostWinRatio"

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

    ## potential predictor columns ##
    # momentum
    home: str = "Home"
    prev_win_streak: str = "PrevWinStreak"
    prev_loss_streak: str = "PrevLossStreak"
    prev_rolling_point_ratio: str = "PrevRollingPointRatio"
    prev_rolling_goaldiff_ratio: str = "PrevRollingGoalDiffRatio"

    # season performance
    zone: str = "Zone"
    prev_twins: str = "PrevTotalWins"
    prev_tlosses: str = "PrevTotalLosses"
    prev_tdraws: str = "PrevTotalDraws"
    prev_rank: str = "PrevRank"
    prev_tpoint_performance: str = "PrevTotalPointPerformance"
    prev_tgoaldiff: str = "PrevTotalGoalDiff"

    # prev season performance
    prev_season_div: str = "PrevSeasonDiv"
    prev_season_trank: str = "PrevSeasonTotalRank"
    prev_season_twins: str = "PrevSeasonTotalWins"
    prev_season_tlosses: str = "PrevSeasonTotalLosses"
    prev_season_tdraws: str = "PrevSeasonTotalDraws"
    prev_season_tgoaldiff: str = "PrevSeasonTotalGoalDiff"
    prev_season_tpoint_performance: str = "PrevSeasonTotalPointPerformance"

    # historical performance
    prev_hist_div: str = "PrevHistoricalDivision"
    prev_hist_trank: str = "PrevHistoricalTotalRank"
    prev_hist_twins: str = "PrevHistoricalTotalWins"
    prev_hist_tlosses: str = "PrevHistoricalTotalLosses"
    prev_hist_tdraws: str = "PrevHistoricalTotalDraws"
    prev_hist_tgoaldiff: str = "PrevHistoricalTotalGoalDiff"
    prev_hist_tpoint_performance: str = "PrevHistoricalTotalPointPerformance"

    # relegation effect
    rel_effect_prev_season_trank: str = "RelEffectPrevSeasonTotalRank"
    rel_effect_prev_season_twins: str = "RelEffectPrevSeasonTotalWins"
    rel_effect_prev_season_tlosses: str = "RelEffectPrevSeasonTotalLosses"
    rel_effect_prev_season_tdraws: str = "RelEffectPrevSeasonTotalDraws"
    rel_effect_prev_season_tgoaldiff: str = "RelEffectPrevSeasonTotalGoalDiff"
    rel_effect_prev_season_tpoint_performance: str = (
        "RelEffectPrevSeasonTotalPointPerformance"
    )

    # promotion effect
    prom_effect_prev_season_trank: str = "PromEffectPrevSeasonTotalRank"
    prom_effect_prev_season_twins: str = "PromEffectPrevSeasonTotalWins"
    prom_effect_prev_season_tlosses: str = "PromEffectPrevSeasonTotalLosses"
    prom_effect_prev_season_tdraws: str = "PromEffectPrevSeasonTotalDraws"
    prom_effect_prev_season_tgoaldiff: str = "PromEffectPrevSeasonTotalGoalDiff"
    prom_effect_prev_season_tpoint_performance: str = (
        "PromEffectPrevSeasonTotalPointPerformance"
    )


COLUMNS = Columns()


@dataclass(frozen=True)
class Predictors:
    # momentum
    # div: str = "Div"
    home: str = "Home"
    prev_win_streak: str = "PrevWinStreak"
    prev_loss_streak: str = "PrevLossStreak"
    prev_rolling_point_ratio: str = "PrevRollingPointRatio"
    prev_rolling_goaldiff_ratio: str = "PrevRollingGoalDiffRatio"

    # season performance
    zone: str = "Zone"
    prev_twins: str = "PrevTotalWins"
    prev_tlosses: str = "PrevTotalLosses"
    prev_tdraws: str = "PrevTotalDraws"
    prev_rank: str = "PrevRank"
    prev_tpoint_performance: str = "PrevTotalPointPerformance"
    prev_tgoaldiff: str = "PrevTotalGoalDiff"

    # prev season performance
    prev_season_div: str = "PrevSeasonDiv"
    prev_season_trank: str = "PrevSeasonTotalRank"
    prev_season_twins: str = "PrevSeasonTotalWins"
    prev_season_tlosses: str = "PrevSeasonTotalLosses"
    prev_season_tdraws: str = "PrevSeasonTotalDraws"
    prev_season_tgoaldiff: str = "PrevSeasonTotalGoalDiff"
    prev_season_tpoint_performance: str = "PrevSeasonTotalPointPerformance"

    # historical performance
    prev_hist_div: str = "PrevHistoricalDivision"
    prev_hist_trank: str = "PrevHistoricalTotalRank"
    prev_hist_twins: str = "PrevHistoricalTotalWins"
    prev_hist_tlosses: str = "PrevHistoricalTotalLosses"
    prev_hist_tdraws: str = "PrevHistoricalTotalDraws"
    prev_hist_tgoaldiff: str = "PrevHistoricalTotalGoalDiff"
    prev_hist_tpoint_performance: str = "PrevHistoricalTotalPointPerformance"

    # relegation effect
    rel_effect_prev_season_trank: str = "RelEffectPrevSeasonTotalRank"
    rel_effect_prev_season_twins: str = "RelEffectPrevSeasonTotalWins"
    rel_effect_prev_season_tlosses: str = "RelEffectPrevSeasonTotalLosses"
    rel_effect_prev_season_tdraws: str = "RelEffectPrevSeasonTotalDraws"
    rel_effect_prev_season_tgoaldiff: str = "RelEffectPrevSeasonTotalGoalDiff"
    rel_effect_prev_season_tpoint_performance: str = (
        "RelEffectPrevSeasonTotalPointPerformance"
    )

    # promotion effect
    prom_effect_prev_season_trank: str = "PromEffectPrevSeasonTotalRank"
    prom_effect_prev_season_twins: str = "PromEffectPrevSeasonTotalWins"
    prom_effect_prev_season_tlosses: str = "PromEffectPrevSeasonTotalLosses"
    prom_effect_prev_season_tdraws: str = "PromEffectPrevSeasonTotalDraws"
    prom_effect_prev_season_tgoaldiff: str = "PromEffectPrevSeasonTotalGoalDiff"
    prom_effect_prev_season_tpoint_performance: str = (
        "PromEffectPrevSeasonTotalPointPerformance"
    )

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
