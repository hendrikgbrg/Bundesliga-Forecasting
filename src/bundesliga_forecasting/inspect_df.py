from bundesliga_forecasting.BL_config import PATHS
from bundesliga_forecasting.BL_utils import read_csv
from bundesliga_forecasting.feature_engineering.F_config import COLUMNS

paths = PATHS
cols = COLUMNS

df = read_csv(paths.features / paths.feature_file)


filter_df = df.loc[
    (df[cols.team] == "Bayern Munich"),
    [
        cols.date,
        cols.team,
        cols.goaldiff,
        cols.prev_win_ratio,
        cols.prev_hist_win_ratio,
    ],
]

group_df = (
    df[df[cols.team] == "Bayern Munich"]
    .groupby([cols.season])[
        [cols.date, cols.seasonal_win_ratio, cols.prev_hist_win_ratio]
    ]
    .last()
)

print(group_df.head(10))
