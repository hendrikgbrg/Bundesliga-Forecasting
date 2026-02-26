import logging
from pathlib import Path

from sklearn.linear_model import PoissonRegressor

from bundesliga_forecasting.BL_config import (
    COLUMNS,
    CSV_ENCODING,
    PATHS,
    PREDICTORS,
    setup_logging,
)
from bundesliga_forecasting.BL_utils import (
    ensure_dir,
    read_csv,
)
from bundesliga_forecasting.models.M_config import ELASTICNET

logger = logging.getLogger(__name__)

paths = PATHS
preds = list(PREDICTORS.values())
encoding = CSV_ENCODING
cols = COLUMNS
elnet = ELASTICNET

opp_preds = [f"{pred}_opp" for pred in preds if pred != cols.home]
preds = preds + opp_preds


def data_setup(
    src_dir: Path = paths.features,
    train_file: str = paths.train_file,
    test_file: str = paths.test_file,
) -> None:
    setup_logging()
    logger.info("Starting Elastic-Net feature selection process...")
    ensure_dir([src_dir], ["src"])

    train_path = src_dir / train_file
    test_path = src_dir / test_file

    df_train = read_csv(train_path)
    df_test = read_csv(test_path)

    X_train = df_train.drop(
        columns=[cols.goalsf, cols.season, cols.div, cols.date, cols.team]
    )
    X_test = df_test.drop(
        columns=[cols.goalsf, cols.season, cols.div, cols.date, cols.team]
    )
    y_train = df_train[cols.goalsf]
    y_test = df_test[cols.goalsf]

    model = PoissonRegressor(fit_intercept=True)
