from dataclasses import dataclass, field
from typing import List

# =========================
#  Feature Parameters
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
#  Ranking Logic
# ===================

RANK_SCORE_WEIGHTS = {
    "points": 1e6,
    "goals_diff": 1e3,
    "total_goals_for": 1,
}
