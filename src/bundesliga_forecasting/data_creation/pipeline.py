import logging

from bundesliga_forecasting.data_creation.config import setup_logging
from bundesliga_forecasting.data_creation.io.clean import clean
from bundesliga_forecasting.data_creation.io.merge import merge
from bundesliga_forecasting.data_creation.io.prepare import prepare

logger = logging.getLogger(__name__)


def data_creation() -> None:
    """
    Description:
        Step 1 ->

    Args:
        paths (CreationPaths): _description_
        csv_config (CSVCleaningConfig): _description_
        sort_cols (list[str]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    setup_logging()

    logger.info("Starting data creation pipeline...")

    clean()
    merge()
    prepare()

    logger.info("Data creation pipeline finished successfully.")
    

def main() -> None:
    data_creation()


if __name__ == "__main__":
    main()