import logging

from bundesliga_forecasting.project_config import setup_logging
from bundesliga_forecasting.feature_engineering.features.score import add_score_features
from bundesliga_forecasting.feature_engineering.features.season import add_season_features
from bundesliga_forecasting.feature_engineering.features.performance import add_performance_features
from bundesliga_forecasting.feature_engineering.features.history import add_historical_features

logger = logging.getLogger(__name__)

def feature_engineering() -> None:
    setup_logging()

    logger.info("Starting feature engineering pipeline...")

    add_score_features()
    add_season_features()
    add_performance_features()
    add_historical_features()

    logger.info("Feature engineering pipeline finished successfully.")
    

def main() -> None:
    feature_engineering()


if __name__ == "__main__":
    main()