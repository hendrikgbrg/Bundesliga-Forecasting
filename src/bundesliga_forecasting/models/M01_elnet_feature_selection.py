import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
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
preds = PREDICTORS.values()
encoding = CSV_ENCODING
cols = COLUMNS
elnet = ELASTICNET


def data_setup(
    src_dir: Path = paths.features,
    src_file: str = paths.diff_file,
    target_dir: Path = paths.elnet,
) -> None:
    setup_logging()
    logger.info("Starting Elastic-Net feature selection process...")
    ensure_dir([src_dir, target_dir], ["src", "target"])

    input_path = src_dir / src_file

    df = read_csv(input_path)
    train, valid, test = _split(df)
    selected_features = _elnet_selection(train)
    scaler = StandardScaler()
    scaler = scaler.fit(train[selected_features])

    logger.info("Scaling selected fatures...")
    set_list = [train, valid, test]
    path_list = outputs_paths(target_dir, selected_features)
    for set, path in zip(set_list, path_list):
        set = _scaling_features(set, selected_features, scaler)
        save_to_csv(set, path)


#################################################################


def _split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("Splitting the dataset into train, validation and test datasets...")
    check_columns(df, [cols.season, cols.div, cols.date])
    seasons = sorted(df[cols.season].unique())

    train_season = seasons[:-2]
    valid_season = [seasons[-2]]
    test_season = [seasons[-1]]

    train = (
        df[df[cols.season].isin(train_season)]
        .sort_values([cols.season, cols.div, cols.date])
        .drop(columns=[cols.season, cols.div, cols.date])
        .reset_index(drop=True)
    )
    valid = (
        df[df[cols.season].isin(valid_season)]
        .sort_values([cols.season, cols.div, cols.date])
        .drop(columns=[cols.season, cols.div, cols.date])
        .reset_index(drop=True)
    )
    test = (
        df[df[cols.season].isin(test_season)]
        .sort_values([cols.season, cols.div, cols.date])
        .drop(columns=[cols.season, cols.div, cols.date])
        .reset_index(drop=True)
    )

    return train, valid, test


def _elnet_selection(df: pd.DataFrame) -> list[str]:
    logger.info("Setting up Elastic-Net Configuration...")
    check_columns(df, [cols.goalsf] + list(preds))

    X = df[list(preds)].copy()
    y = df[cols.goalsf].to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = ElasticNetCV(
        fit_intercept=elnet.fit_intercept,
        l1_ratio=elnet.l1_ratio,
        alphas=elnet.alphas,
        cv=elnet.cv,
        max_iter=elnet.max_iter,
        n_jobs=elnet.n_jobs,
    )

    model.fit(X_scaled, y)

    logger.info(f"Best alpha: {model.alpha_}")
    logger.info(f"Best l1_ratio: {model.l1_ratio_}")
    coefs = model.coef_
    selected_coefs = np.abs(coefs) > 1e-6
    selected_features = X.columns[selected_coefs].tolist()
    removed = [feature for feature in preds if feature not in selected_features]

    if len(selected_features) == 0:
        raise RuntimeError("Elastic Net removed all predictors.")

    logger.info(
        f"Selected {len(selected_features)} / {len(preds)} features with non-zero coefficients: \n{selected_features}\n\nRemoved features: \n{removed}"
    )

    return selected_features


def outputs_paths(target_dir: Path, selected_features: list[str]) -> list[Path]:
    n_preds = len(list(preds))
    n_selected = len(selected_features)
    file_list = [paths.train_file, paths.valid_file, paths.test_file]
    file_names = [
        (f"{n_selected}-{n_preds}_" if n_selected < n_preds else "full_") + file
        for file in file_list
    ]
    path_list = [target_dir / file_name for file_name in file_names]

    return path_list


def _scaling_features(
    df: pd.DataFrame,
    selected_features: list[str],
    scaler: StandardScaler,
) -> pd.DataFrame:
    check_columns(df, [cols.goalsf] + selected_features)
    out = df[[cols.goalsf] + selected_features].copy()

    out[selected_features] = scaler.transform(out[selected_features])
    out[selected_features] = out[selected_features].round(6)

    # out = _add_team_indicators(out, ref_team=ref_team)
    # out = out.drop(columns=[cols.team, cols.opp])
    return out


# def _add_team_indicators(df: pd.DataFrame, ref_team: int | str) -> pd.DataFrame:
#     team_dummies = (
#         pd.get_dummies(df[cols.team], prefix="Team_")
#         .astype(int)
#         .drop(columns=f"Team_{ref_team}", errors="ignore")
#     )
#     opp_dummies = (
#         pd.get_dummies(df[cols.opp], prefix="Opp_")
#         .astype(int)
#         .drop(columns=f"Opp_{ref_team}", errors="ignore")
#     )

#     df = pd.concat([df, team_dummies, opp_dummies], axis=1)
#     return df


def main() -> None:
    data_setup()


if __name__ == "__main__":
    main()
