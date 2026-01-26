from dataclasses import dataclass, field
from typing import List

# =========================
#  Configs (experimentell)
# =========================


@dataclass(frozen=True)
class EWMAConfig:
    alpha: float = 0.75
    alpha_short: float = 0.85
    alpha_long: float = 0.65


@dataclass(frozen=True)
class ZoneConfig:
    bins: List[int] = field(default_factory=lambda: [1, 3, 6, 12, 15, 18])
    parameters: List[float] = field(default_factory=lambda: [1, 0.5, 0, -0.5, -1])


# ===================
#  Dataset constants
# ===================

SEASON_START_MONTH = 7
MATCHES_PER_SEASON = 34

RANK_SCORE_WEIGHTS = {
    "points": 1e6,
    "goals_diff": 1e3,
    "total_goals_for": 1,
}

HOME_COLS = ["Season", "Div", "Date", "HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals"]

AWAY_COLS = ["Season", "Div", "Date", "AwayTeam", "HomeTeam", "AwayGoals", "HomeGoals"]

TEAM_MATCH_COLS = [
    "Season",
    "Div",
    "Date",
    "Team",
    "Opponent",
    "GoalsFor",
    "GoalsAgainst",
]
