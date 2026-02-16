from bundesliga_forecasting.BL_config import PATHS
from bundesliga_forecasting.BL_utils import read_csv
from bundesliga_forecasting.feature_engineering.F_config import COLUMNS

paths = PATHS
cols = COLUMNS

df = read_csv(paths.test / (paths.t_filename))


show_df = df.loc[
    (df[cols.team] == "Bayern Munich"),
    [
        cols.date,
        cols.team,
        cols.points,
        cols.prev_tpoints,
        cols.post_tpoints,
        cols.prev_rank,
        # cols.post_rank,
        # cols.post_tpoints,
        # cols.post_tgoaldiff,
        # cols.post_tgoalsf,
        # cols.prev_trank,
        # cols.post_trank,
        # cols.prev_min_tpoints,
        # cols.prev_max_tpoints,
    ],
]

print(df[cols.date].dtype)
print(show_df.head(102))
