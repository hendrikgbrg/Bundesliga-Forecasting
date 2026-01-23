from pathlib import Path

import numpy as np
import pandas as pd

base_dir = Path(__file__).resolve().parent
src_dir = (base_dir / "../../data/04_Merged").resolve()
target_dir = (base_dir / "../../data/05_Features").resolve()
target_dir.mkdir(parents=True, exist_ok=True)

file_path = src_dir / "data.csv"

# read csv-file with correct date format
df = pd.read_csv(file_path, parse_dates=["Date"], dayfirst=True)

# Hyperparameters
decay = 0.75


def edit_df(df):
    def add_season(df):
        season = np.where(
            df["Date"].dt.month >= 7, df["Date"].dt.year, df["Date"].dt.year - 1
        )
        df.insert(0, "Season", season)
        return df

    def team_match_split(df):
        df1 = df[
            ["Season", "Div", "Date", "HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals"]
        ].copy()
        df2 = df[
            ["Season", "Div", "Date", "AwayTeam", "HomeTeam", "AwayGoals", "HomeGoals"]
        ].copy()
        columns = [
            "Season",
            "Div",
            "Date",
            "Team",
            "Opponent",
            "GoalsFor",
            "GoalsAgainst",
        ]
        df1.columns = columns
        df2.columns = columns
        df1.insert(5, "Home", 1)
        df2.insert(5, "Home", 0)
        combined = pd.concat([df1, df2], ignore_index=True)
        return combined

    edit_functions = [add_season, team_match_split]
    for function in edit_functions:
        df = function(df)

    return df


def create_features(df):
    def match_results(df):
        df["Points"] = np.where(
            df["GoalsFor"] > df["GoalsAgainst"],
            3,
            np.where(df["GoalsFor"] == df["GoalsAgainst"], 1, 0),
        )
        df["GoalDiff"] = df["GoalsFor"] - df["GoalsAgainst"]
        return df

    def post_match_results(df):
        # total points
        df["TotalPoints"] = df.groupby(["Season", "Team"])["Points"].cumsum()

        # total goals for and against
        for side in ["For", "Against"]:
            goals_column = "Goals" + side
            new_column = "Total" + goals_column
            df[new_column] = df.groupby(["Season", "Team"])[goals_column].cumsum()

        # total goal difference
        df["TotalGoalDiff"] = df["TotalGoalsFor"] - df["TotalGoalsAgainst"]

        return df

    def rank(df):
        # create calender to log all team results for all dates
        dates = df[["Season", "Div", "Date"]].drop_duplicates()
        teams = df[["Season", "Div", "Team"]].drop_duplicates()
        calendar = dates.merge(teams, on=["Div", "Season"], how="inner")

        # create snap view of the calendar with the actual game entries of the df
        rank_columns = ["TotalPoints", "TotalGoalDiff", "TotalGoalsFor"]
        add_columns = ["Points", "GoalsFor", "GoalsAgainst"] + rank_columns
        snap = calendar.merge(
            df[["Season", "Div", "Date", "Team"] + add_columns],
            on=["Season", "Div", "Date", "Team"],
            how="left",
        )

        # forward fill the rank-relevant columns after every game entry for the upcoming dates
        snap = snap.sort_values(["Season", "Div", "Date"])
        snap[rank_columns] = snap.groupby(["Season", "Div", "Team"])[
            rank_columns
        ].ffill()

        # fill all na-values with zero, effecting only the additional left-merged game specific df entries
        snap[add_columns] = snap[add_columns].fillna(0)
        snap = snap.reset_index(drop=True)

        # update the df to the a priori game status
        df["TotalPoints"] = df["TotalPoints"] - df["Points"]
        df["TotalGoalDiff"] = df["TotalGoalDiff"] + df["GoalsAgainst"] - df["GoalsFor"]
        df["TotalGoalsFor"] = df["TotalGoalsFor"] - df["GoalsFor"]
        df["TotalGoalsAgainst"] = df["TotalGoalsAgainst"] - df["GoalsAgainst"]

        # add prior columns to the snap frame to adjust for a priori results, effectively using only non-zero game specific entries
        snap["PriorTotalPoints"] = snap["TotalPoints"] - snap["Points"]
        snap["PriorTotalGoalDiff"] = (
            snap["TotalGoalDiff"] - snap["GoalsFor"] + snap["GoalsAgainst"]
        )
        snap["PriorTotalGoalsFor"] = snap["TotalGoalsFor"] - snap["GoalsFor"]

        # use prior and post columns of the snap view to calculate the ranks before and after every game
        snap["PriorScore"] = (
            snap["PriorTotalPoints"] * 1e6
            + snap["PriorTotalGoalDiff"] * 1e3
            + snap["PriorTotalGoalsFor"]
        )
        snap["PostScore"] = (
            snap["TotalPoints"] * 1e6
            + snap["TotalGoalDiff"] * 1e3
            + snap["TotalGoalsFor"]
        )
        snap["PriorRank"] = snap.groupby(["Season", "Div", "Date"])["PriorScore"].rank(
            method="dense", ascending=False
        )
        snap["PostRank"] = snap.groupby(["Season", "Div", "Date"])["PostScore"].rank(
            method="dense", ascending=False
        )

        snap = snap.drop(columns=["PriorScore"])
        snap = snap.drop(columns=["PostScore"])

        snap["LowestPriorRank"] = snap.groupby(["Season", "Div", "Date"])[
            "PriorRank"
        ].transform("max")
        snap["MinPriorPoints"] = snap.groupby(["Season", "Div", "Date"])[
            "PriorTotalPoints"
        ].transform("min")
        snap["MaxPriorPoints"] = snap.groupby(["Season", "Div", "Date"])[
            "PriorTotalPoints"
        ].transform("max")
        snap["LowestPostRank"] = snap.groupby(["Season", "Div", "Date"])[
            "PostRank"
        ].transform("max")
        snap["MinPostPoints"] = snap.groupby(["Season", "Div", "Date"])[
            "TotalPoints"
        ].transform("min")
        snap["MaxPostPoints"] = snap.groupby(["Season", "Div", "Date"])[
            "TotalPoints"
        ].transform("max")

        # merge the snap view's rank columns onto the original df
        df = df.merge(
            snap[
                [
                    "Season",
                    "Div",
                    "Date",
                    "Team",
                    "PriorRank",
                    "PostRank",
                    "LowestPriorRank",
                    "LowestPostRank",
                    "MinPriorPoints",
                    "MaxPriorPoints",
                    "MinPostPoints",
                    "MaxPostPoints",
                ]
            ],
            on=["Season", "Date", "Team"],
            how="left",
        )

        return df

    def zone(df):
        bins = [1, 3, 6, 12, 15, 18]
        labels = [1, 0.5, 0, -0.5, -1]

        df["Zone"] = pd.cut(
            df["PriorRank"], bins=bins, labels=labels, right=True, include_lowest=True
        )
        return df

    def team_superiority(df):
        prior_games = df.groupby(["Season", "Team"]).cumcount().replace(0, 1)
        df["PriorSuperiority"] = (
            df["TotalGoalsFor"] - df["TotalGoalsAgainst"]
        ) / prior_games
        df["PostSuperiority"] = (
            df["TotalGoalsFor"]
            + df["GoalsFor"]
            - df["TotalGoalsAgainst"]
            - df["GoalsAgainst"]
        ) / 34
        return df

    def rank_performance(df):
        df["PriorRankPerformance"] = 1 - 2 * (
            (df["PriorRank"] - 1) / np.maximum(df["LowestPriorRank"] - 1, 1)
        )
        df["PostRankPerformance"] = 1 - 2 * (
            (df["PostRank"] - 1) / np.maximum(df["LowestPostRank"] - 1, 1)
        )
        return df

    def point_performance(df):
        df["PriorPointPerformance"] = 1 - 2 * (
            (df["MaxPriorPoints"] - df["TotalPoints"])
            / np.maximum(df["MaxPriorPoints"] - df["MinPriorPoints"], 1)
        )
        df["PostPointPerformance"] = 1 - 2 * (
            (df["MaxPostPoints"] - df["TotalPoints"] - df["Points"])
            / np.maximum(df["MaxPostPoints"] - df["MinPostPoints"], 1)
        )
        return df

    def total_performance(df):
        df["TotalPriorPerformance"] = (
            df["PriorRankPerformance"] + df["PriorPointPerformance"]
        ) / 2
        df["TotalPostPerformance"] = (
            df["PostRankPerformance"] + df["PostPointPerformance"]
        ) / 2
        return df

    def win_loss_rate(df):
        wins = (
            (df["Points"] == 3).astype(int).groupby([df["Season"], df["Team"]]).cumsum()
        )
        draws = (
            (df["Points"] == 1).astype(int).groupby([df["Season"], df["Team"]]).cumsum()
        )
        losses = (
            (df["Points"] == 0)
            .astype(int)
            .groupby([df["Season"], df["Team"]])
            .cumsum()
            .replace(0, 1)
        )
        df["WinLossRate"] = (wins + draws / 3) / losses
        df["WinLossRate"] = (
            df.groupby(["Season", "Team"])["WinLossRate"].shift(1).fillna(0)
        )
        return df

    def history(df):
        columns = [
            "Season",
            "Team",
            "GoalsFor",
            "GoalsAgainst",
            "TotalGoalsFor",
            "TotalGoalsAgainst",
            "PostSuperiority",
            "PostRankPerformance",
            "PostPointPerformance",
            "TotalPostPerformance",
        ]
        # create calender to log all team results for all seasons
        seasons = df["Season"].drop_duplicates()
        teams = df["Team"].drop_duplicates()

        season_team = seasons.to_frame().merge(teams.to_frame(), how="cross")

        season_end = df[columns].groupby(["Season", "Team"], as_index=False).last()

        season_team = season_team.merge(
            season_end, on=["Season", "Team"], how="left"
        ).fillna(0)

        season_team["HistSuperiority"] = (
            season_team.groupby("Team")["PostSuperiority"]
            .shift(1)
            .fillna(0)
            .groupby(season_team["Team"])
            .ewm(alpha=decay, adjust=False)
            .mean()
            .reset_index(level=0, drop=True)
        )

        season_team["HistRankPerformance"] = (
            season_team.groupby("Team")["PostRankPerformance"]
            .shift(1)
            .fillna(0)
            .groupby(season_team["Team"])
            .ewm(alpha=decay, adjust=False)
            .mean()
            .reset_index(level=0, drop=True)
        )

        season_team["HistPointPerformance"] = (
            season_team.groupby("Team")["PostPointPerformance"]
            .shift(1)
            .fillna(0)
            .groupby(season_team["Team"])
            .ewm(alpha=decay, adjust=False)
            .mean()
            .reset_index(level=0, drop=True)
        )

        season_team["HistTotalPerformance"] = (
            season_team.groupby("Team")["TotalPostPerformance"]
            .shift(1)
            .fillna(0)
            .groupby(season_team["Team"])
            .ewm(alpha=decay, adjust=False)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df = df.merge(
            season_team[
                [
                    "Season",
                    "Team",
                    "HistSuperiority",
                    "HistRankPerformance",
                    "HistPointPerformance",
                    "HistTotalPerformance",
                ]
            ],
            on=["Season", "Team"],
            how="inner",
        )

        return df

    feature_functions = [
        match_results,
        post_match_results,
        rank,
        zone,
        team_superiority,
        rank_performance,
        point_performance,
        total_performance,
        win_loss_rate,
        history,
    ]

    df = df.sort_values(["Season", "Div", "Date"]).reset_index(drop=True)
    for function in feature_functions:
        df = function(df)

    return df


df = edit_df(df)
df = create_features(df)
df = df.sort_values(["Season", "Team"]).reset_index(drop=True)

# print(df.head(40))
print(
    df[df.Team == "Dortmund"]
    .groupby("Season", as_index=False)
    .last()[
        [
            "Season",
            "Team",
            "PostSuperiority",
            "HistSuperiority",
        ]
    ]
)
