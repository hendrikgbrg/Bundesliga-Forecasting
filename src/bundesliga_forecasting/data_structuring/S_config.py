from dataclasses import dataclass, field

# ==================
#  Config Variables
# ==================

SEASON_COL = "Season"
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
class ColumnLists:
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
    team: list[str] = field(default_factory=lambda: ["HomeTeam", "AwayTeam"])
    sort_by: list[str] = field(default_factory=lambda: ["Season", "Div", "Date"])


COLUMNLISTS = ColumnLists()
