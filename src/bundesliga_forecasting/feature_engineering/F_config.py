from dataclasses import dataclass, field

# =========================
#  Config Parameters
# =========================


@dataclass(frozen=True)
class Columns:
    # general
    season: str = "Season"
    div: str = "Div"
    date: str = "Date"
    team: str = "Team"
    opp: str = "Opponent"

    # team-match level
    goalsf: str = "GoalsFor"
    goalsa: str = "GoalsAgainst"
    goaldiff: str = "GoalDiff"
    points: str = "Points"
    zone: str = "Zone"

    # post-match cumulation
    post_tgoalsf: str = "PostTotalGoalsFor"
    post_tgoalsa: str = "PostTotalGoalsAgainst"
    post_tgoaldiff: str = "PostTotalGoalDiff"
    post_tpoints: str = "PostTotalPoints"
    post_min_tpoints: str = "PostMinTotalPoints"
    post_max_tpoints: str = "PostMaxTotalPoints"
    post_rank: str = "PostRank"
    post_trank: str = "PostTotalRank"
    post_superiority: str = "PostSuperiority"
    post_trank_performance: str = "PostTotalRankPerformance"
    post_tpoint_performance: str = "PostTotalPointPerformance"
    # post_tperformance: str = "PostTotalPerformance"
    seasonal_win_loss_ratio: str = "SeasonalWinLossRatio"
    seasonal_win_ratio: str = "SeasonalWinRatio"
    post_hist_win_loss_ratio: str = "PostHistWinLossRatio"
    post_hist_win_ratio: str = "PostHistWinRatio"
    post_hist_superiority: str = "PostHistoricalSuperiority"
    post_hist_trank_performance: str = "PostHistoricalTotalRankPerformance"
    post_hist_tpoint_performance: str = "PostHistoricalTotalPointPerformance"
    # post_hist_tperformance: str = "PostHistoricalTotalPerformance"

    # prior-match cumulation
    prev_tgoalsf: str = "PrevTotalGoalsFor"
    prev_tgoalsa: str = "PrevTotalGoalsAgainst"
    prev_tgoaldiff: str = "PrevTotalGoalDiff"
    prev_goaldiff: str = "PrevGoalDiff"
    prev_tpoints: str = "PrevTotalPoints"
    prev_min_tpoints: str = "PrevMinTotalPoints"
    prev_max_tpoints: str = "PrevMaxTotalPoints"
    prev_rank: str = "PrevRank"
    prev_trank: str = "PrevTotalRank"
    prev_superiority: str = "PrevSuperiority"
    prev_trank_performance: str = "PrevTotalRankPerformance"
    prev_tpoint_performance: str = "PrevTotalPointPerformance"
    # prev_tperformance: str = "PrevTotalPerformance"
    prev_win_loss_ratio: str = "PrevWinLossRatio"
    prev_win_ratio: str = "PrevWinRatio"
    prev_hist_superiority: str = "PrevHistoricalSuperiority"
    prev_hist_win_loss_ratio: str = "PrevHistWinLossRatio"
    prev_hist_win_ratio: str = "PrevHistWinRatio"
    prev_hist_trank_performance: str = "PrevHistoricalTotalRankPerformance"
    prev_hist_tpoint_performance: str = "PrevHistoricalTotalPointPerformance"
    # prev_hist_tperformance: str = "PrevHistoricalTotalPerformance"


COLUMNS = Columns()


@dataclass(frozen=True)
class EWMA_Decay:
    season: float = 0.4
    history: float = 0.75
    rolling: int = 5


EWMA_DECAY = EWMA_Decay()


prev_RANK_COLS = [
    COLUMNS.prev_tpoints,
    COLUMNS.prev_tgoaldiff,
    COLUMNS.prev_tgoalsf,
]

POST_RANK_COLS = [
    COLUMNS.post_tpoints,
    COLUMNS.post_tgoaldiff,
    COLUMNS.post_tgoalsf,
]

MATCH_COLS = [
    COLUMNS.goalsf,
    COLUMNS.goalsa,
    COLUMNS.points,
]


@dataclass(frozen=True)
class Zones:
    bins: list[int] = field(default_factory=lambda: [1, 3, 6, 12, 15, 18])
    labels: list[float] = field(default_factory=lambda: [1, 0.5, 0, -0.5, -1])


ZONES = Zones()
