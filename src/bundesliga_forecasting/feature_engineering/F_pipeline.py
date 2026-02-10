import logging

from bundesliga_forecasting.BL_config import setup_logging
from bundesliga_forecasting.feature_engineering.features.F01_score import (
    add_score_features,
)
from bundesliga_forecasting.feature_engineering.features.F02_ranking import (
    add_season_features,
)
from bundesliga_forecasting.feature_engineering.features.F03_performance import (
    add_performance_features,
)
from bundesliga_forecasting.feature_engineering.features.F04_history import (
    add_historical_features,
)
from bundesliga_forecasting.feature_engineering.features.F05_relativation import (
    compute_relative_features,
)

logger = logging.getLogger(__name__)


def feature_engineering() -> None:
    setup_logging()

    logger.info("Starting feature engineering pipeline...")

    add_score_features()
    add_season_features()
    add_performance_features()
    add_historical_features()
    compute_relative_features()

    logger.info("Feature engineering pipeline finished successfully.")


def main() -> None:
    feature_engineering()


if __name__ == "__main__":
    main()
