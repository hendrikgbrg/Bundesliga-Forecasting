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
    post_win_loss_rate: str = "PostWinLossRate"
    post_hist_superiority: str = "PostHistoricalSuperiority"
    post_hist_trank_performance: str = "PostHistoricalTotalRankPerformance"
    post_hist_tpoint_performance: str = "PostHistoricalTotalPointPerformance"
    # post_hist_tperformance: str = "PostHistoricalTotalPerformance"
    
    # prior-match cumulation
    pre_tgoalsf: str = "PreTotalGoalsFor"  
    pre_tgoalsa: str = "PreTotalGoalsAgainst"    
    pre_tgoaldiff: str = "PreTotalGoalDiff"
    pre_goaldiff: str = "PreGoalDiff"
    pre_tpoints: str = "PreTotalPoints"
    pre_min_tpoints: str = "PreMinTotalPoints"
    pre_max_tpoints: str = "PreMaxTotalPoints"
    pre_rank: str = "PreRank"
    pre_trank: str = "PreTotalRank"
    pre_superiority: str = "PreSuperiority"
    pre_trank_performance: str = "PreTotalRankPerformance"
    pre_tpoint_performance: str = "PreTotalPointPerformance"
    # pre_tperformance: str = "PreTotalPerformance"
    pre_win_loss_rate: str = "PreWinLossRate"
    pre_hist_superiority: str = "PreHistoricalSuperiority"
    pre_hist_trank_performance: str = "PreHistoricalTotalRankPerformance"
    pre_hist_tpoint_performance: str = "PreHistoricalTotalPointPerformance"
    # pre_hist_tperformance: str = "PreHistoricalTotalPerformance"

COLUMNS = Columns()

@dataclass(frozen=True)
class EWMA_Decay:
    season: float = 0.4
    history: float = 0.75
    rolling: int = 5
    
EWMA_DECAY = EWMA_Decay()


PRIOR_RANK_COLS = [
    COLUMNS.pre_tpoints,
    COLUMNS.pre_tgoaldiff,
    COLUMNS.pre_tgoalsf,
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