from bundesliga_forecasting.BL_config import PATHS
from bundesliga_forecasting.BL_utils import read_csv
from bundesliga_forecasting.feature_engineering.F_config import COLUMNS

paths = PATHS
cols = COLUMNS

df = read_csv(paths.merged / paths.m_filename)


show_df = df[cols.date].sort_values().drop_duplicates().tolist()

print(show_df)
