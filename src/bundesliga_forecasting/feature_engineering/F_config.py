from dataclasses import dataclass, field

from bundesliga_forecasting.BL_config import COLUMNS

# =========================
#  Config Parameters
# =========================


@dataclass(frozen=True)
class Weights:
    season: float = 0.4
    history: float = 0.75
    rolling: int = 5


WEIGHTS = Weights()


@dataclass(frozen=True)
class Zones:
    bins: list[int] = field(default_factory=lambda: [1, 3, 6, 12, 15, 18])
    labels: list[float] = field(default_factory=lambda: [1, 0.75, 0, 5, 0.25, 0])


ZONES = Zones()


PREV_RANK_COLS = [
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

DRAW_VALUE = 0.5
