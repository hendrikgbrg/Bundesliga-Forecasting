from bundesliga_forecasting.BL_config import PATHS
from bundesliga_forecasting.BL_utils import read_csv
from bundesliga_forecasting.feature_engineering.F_config import COLUMNS

paths = PATHS
cols = COLUMNS

df = read_csv(paths.features / paths.f_filename)


show_df = df[(df.Team == "Bayern Munich")][
    [
        cols.season,
        cols.date,
        cols.team,
        cols.opp,
        cols.goaldiff,
        cols.prev_win_loss_ratio,
        cols.prev_win_ratio,
    ]
].reset_index(drop=True)

print(show_df.head(10))
