import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bundesliga_forecasting.BL_config import (
    COLUMNS,
    CSV_ENCODING,
    PATHS,
    PREDICTORS,
    setup_logging,
)
from bundesliga_forecasting.BL_utils import (
    check_columns,
    ensure_dir,
    read_csv,
    save_to_csv,
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
    src_file: str = paths.combined_file,
    target_dir: Path = paths.features,
) -> None:
    setup_logging()
    logger.info("Starting Elastic-Net feature selection process...")
    ensure_dir([src_dir, target_dir], ["src", "target"])

    input_path = src_dir / src_file

    df = read_csv(input_path)
    df = df.sort_values([cols.date]).reset_index(drop=True)
    train, test = _split(df)
    model = _train_poisson_elnet(train)

    # model output
    selected_features = _log_selected_features(model, train[preds])
    # scaler = model.named_steps["scaler"]

    df_selected = df[
        [cols.goalsf, cols.season, cols.div, cols.date, cols.team] + selected_features
    ]
    df_train = df_selected.loc[train.index]
    df_test = df_selected.loc[test.index]
    save_to_csv(df_train, paths.features / paths.train_file)
    save_to_csv(df_test, paths.features / paths.test_file)


#################################################################


def _split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Splitting the dataset into train and test datasets...")
    check_columns(df, [cols.season, cols.div, cols.date])
    seasons = sorted(df[cols.season].unique())

    train_season = seasons[:-1]
    test_season = [seasons[-1]]

    train = df[df[cols.season].isin(train_season)].drop(
        columns=[cols.season, cols.div, cols.date]
    )
    test = df[df[cols.season].isin(test_season)].drop(
        columns=[cols.season, cols.div, cols.date]
    )

    return train, test


def _train_poisson_elnet(train: pd.DataFrame) -> Pipeline:
    logger.info("Training Elastic-Net model with Gaussian loss...")

    X_train = train[preds]
    y_train = train[cols.goalsf]

    tscv = TimeSeriesSplit(n_splits=5)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                SGDRegressor(
                    loss="squared_error",
                    penalty="elasticnet",
                    fit_intercept=True,
                    max_iter=1000,
                    tol=1e-4,
                    random_state=42,
                ),
            ),
        ]
    )

    param_grid = {
        "model__alpha": np.logspace(-4, 1, 20),
        "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    }

    scorer = make_scorer(mean_squared_error, greater_is_better=False)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scorer,
        cv=tscv,
        n_jobs=-1,
        verbose=1,
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    logger.info(f"Best alpha-value: {grid.best_params_['model__alpha']}")
    logger.info(f"Best l1_ratio: {grid.best_params_['model__l1_ratio']}")

    return best_model


def _log_selected_features(model: Pipeline, X_train: pd.DataFrame) -> list[str]:
    coefs = model.named_steps["model"].coef_
    selected_mask = np.abs(coefs) > 1e-8
    selected_features = X_train.columns[selected_mask].tolist()
    removed_features = X_train.columns[~selected_mask].tolist()

    logger.info(
        f"Selected {len(selected_features)} / {len(X_train.columns)} features.\n"
        f"{selected_features}\n\n"
        f"Removed features:\n"
        f"{removed_features}"
    )

    if len(selected_features) == 0:
        raise RuntimeError(
            "All features were removed by Elastic-Net. Please check hyperparameters."
        )

    return selected_features


def main() -> None:
    data_setup()


if __name__ == "__main__":
    main()
