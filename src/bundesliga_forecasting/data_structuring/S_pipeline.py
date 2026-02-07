import logging

from bundesliga_forecasting.BL_config import setup_logging
from bundesliga_forecasting.data_structuring.structure.S01_clean import clean
from bundesliga_forecasting.data_structuring.structure.S02_merge import merge
from bundesliga_forecasting.data_structuring.structure.S03_prepare import prepare

logger = logging.getLogger(__name__)


def data_structuring() -> None:
    setup_logging()

    logger.info("Starting data structuring pipeline...")

    clean()
    merge()
    prepare()

    logger.info("Data structuring pipeline finished successfully.")


def main() -> None:
    data_structuring()


if __name__ == "__main__":
    main()
