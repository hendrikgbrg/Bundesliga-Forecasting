from bundesliga_forecasting.BL_config import PATHS
from bundesliga_forecasting.BL_utils import read_csv
from bundesliga_forecasting.feature_engineering.F_config import COLUMNS

paths = PATHS
cols = COLUMNS

df = read_csv(paths.features / paths.f_filename)


show_df = df.loc[
    (df[cols.team] == "Bayern Munich"),
    [
        cols.season,
        cols.date,
        cols.team,
        cols.opp,
        cols.prev_trank_performance,
        cols.prev_tpoint_performance,
        cols.prev_goal_superiority,
        cols.prev_win_ratio,
    ],
]

print(show_df.head(10))
