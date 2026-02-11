from bundesliga_forecasting.BL_config import PATHS
from bundesliga_forecasting.BL_utils import read_csv
from bundesliga_forecasting.feature_engineering.F_config import COLUMNS

paths = PATHS
cols = COLUMNS

df = read_csv(paths.features / paths.f_filename)


show_df = df[
    # (df[cols.team] == "Bayern Munich") | (df[cols.opp] == "Bayern Munich"),
    [
        cols.date,
        cols.team,
        cols.opp,
        cols.prev_trank_performance,
        # cols.post_tpoints,
        # cols.post_tgoaldiff,
        # cols.post_tgoalsf,
        # cols.prev_trank,
        # cols.post_trank,
        # cols.prev_min_tpoints,
        # cols.prev_max_tpoints,
    ]
]

print(df[cols.date].dtype)
print(show_df.head(20))
