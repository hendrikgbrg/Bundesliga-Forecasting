import logging

from bundesliga_forecasting.BL_config import setup_logging
from bundesliga_forecasting.feature_engineering.features.F01_score import (
    add_score_features,
)
from bundesliga_forecasting.feature_engineering.features.F02_daily_table import (
    add_daily_comparisons,
)
from bundesliga_forecasting.feature_engineering.features.F03_momentum import (
    add_momentum,
)
from bundesliga_forecasting.feature_engineering.features.F04_current_season import (
    add_season_performance,
)
from bundesliga_forecasting.feature_engineering.features.F05_prev_season import (
    add_prev_season_performance,
)
from bundesliga_forecasting.feature_engineering.features.F06_relprom_effects import (
    add_relprom_effects,
)
from bundesliga_forecasting.feature_engineering.features.F07_history import (
    add_historical_features,
)
from bundesliga_forecasting.feature_engineering.features.F08_combine import (
    apply_feature_combination,
)

logger = logging.getLogger(__name__)


def feature_engineering() -> None:
    setup_logging()

    logger.info("Starting feature engineering pipeline...")

    add_score_features()
    add_daily_comparisons()
    add_momentum()
    add_season_performance()
    add_prev_season_performance()
    add_relprom_effects()
    add_historical_features()
    apply_feature_combination()

    logger.info("Feature engineering pipeline finished successfully.")


def main() -> None:
    feature_engineering()


if __name__ == "__main__":
    main()
