import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bundesliga_forecasting.BL_config import PATHS
from bundesliga_forecasting.BL_utils import read_csv, save_to_csv
from bundesliga_forecasting.feature_engineering.F_config import COLUMNS

paths = PATHS
cols = COLUMNS

df = read_csv(paths.features / paths.daily_tables_file)


def _add_point_gap_col(df: pd.DataFrame, col_name: str = "PointGap") -> pd.DataFrame:
    season_end = (
        df.loc[
            df.groupby([cols.season, cols.div])[cols.date].transform("max")
            == df[cols.date]
        ]
        .copy()
        .sort_values([cols.season, cols.div, cols.post_rank], ascending=True)
    )

    season_end[col_name] = season_end.groupby([cols.season, cols.div])[
        cols.post_tpoints
    ].diff(-1)

    season_end = season_end.dropna(subset=[col_name])

    return season_end


def build_point_gap_matrix(df: pd.DataFrame, division: str) -> pd.DataFrame:
    gap_col = "PointGap"
    season_end = _add_point_gap_col(df, col_name=gap_col)
    league = season_end[season_end[cols.div] == division].copy()
    seasons = sorted(league[cols.season].unique())
    ranks = list(range(1, 19))
    idx_labels = [f"{rank}-{rank + 1}" for rank in ranks[:-1]]

    matrix = pd.DataFrame(index=idx_labels, columns=seasons, dtype=float)

    for season in seasons:
        league_table = league[league[cols.season] == season]
        matrix[season] = league_table[gap_col].values

    # matrix["average"] =
    return matrix


matrix1 = build_point_gap_matrix(df, division="D1")
matrix2 = build_point_gap_matrix(df, division="D2")

save_to_csv(matrix1, paths.features / "D1_point_gaps.csv")
save_to_csv(matrix2, paths.features / "D2_point_gaps.csv")


def plot_point_gaps(
    matrix: pd.DataFrame,
    title: str = "Point Gaps per Position Difference",
):
    plt.figure(figsize=(12, 6))

    x_labels = matrix.index
    x = np.arange(len(x_labels))

    # Div1 (blau)
    n_seasons1 = matrix.shape[1]
    for i, season in enumerate(matrix.columns):
        alpha = 0.3 + 0.7 * (
            i / (n_seasons1 - 1)
        )  # älteste Saison alpha=0.3, aktuellste alpha=1
        y_values = matrix[[season]].values
        plt.scatter(
            x,
            y_values,
            color="blue",
            alpha=alpha,
            label=season if i == n_seasons1 - 1 else None,
        )

    plt.xticks(x, x_labels, rotation=45)
    plt.xlabel("Positionsdifferenz")
    plt.ylabel("Punktedifferenz")
    plt.title(title)
    plt.tight_layout()
    plt.ylim(0, 30)
    plt.show()


# Beispielaufruf
plot_point_gaps(matrix1, title="D1 Point Gaps per Position Difference")
plot_point_gaps(matrix2, title="D2 Point Gaps per Position Difference")
