from bundesliga_forecasting.BL_config import PATHS
from bundesliga_forecasting.BL_utils import check_columns, read_csv
from bundesliga_forecasting.feature_engineering.F_config import COLUMNS

paths = PATHS
cols = COLUMNS

df = read_csv(paths.features / paths.combined_file)
all_cols = df.columns
for col in all_cols:
    print(f"\n{col}")
team = "Karlsruhe"
columns = [
    cols.season,
    cols.div,
    cols.date,
    cols.team,
    cols.prev_season_tpoint_performance,
    cols.rel_effect_prev_season_tpoint_performance,
    # cols.prev_season_tpoint_performance,
    # cols.prev_hist_tpoint_performance,
]

check_columns(df, columns)


filter_df = df.loc[
    (df[cols.team] == team),
    columns,
]

group_df = (
    df[df[cols.team] == team]
    .groupby([cols.season])[columns]
    .last()
    .reset_index(drop=True)
)

print(group_df.head(10))
