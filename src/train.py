"""
Model training, time series cross-validation, and hyperparameter tuning
"""

import random

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from omegaconf import DictConfig, ListConfig, OmegaConf
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from src.ingest import ensure_stationarity, get_timestamps, preprocess_data
from src.logger import logging
from src.transform import transform_data

CONFIG: DictConfig | ListConfig = OmegaConf.load(r"./config.yaml")


def get_informative_features(feature_matrix: pd.DataFrame, target_vector: pd.Series) -> list[str]:
    """Extracts the most informative features based on the mutual information criterion

    Args:
        feature_matrix (pd.DataFrame): Matrix of lag features, window features, and
        datetime features
        target_vector (pd.Series): Stationary univariate time series

    Returns:
        list[str]: Most informative features based on the mutual information criterion
    """
    try:
        scores: list[float] = [
            mutual_info_regression(feature_matrix[[col]], target_vector)[0]
            for col in feature_matrix.columns
        ]
        scores = [score / max(scores) for score in scores]
        mi_features: list[str] = [
            col for col, score in zip(feature_matrix.columns, scores) if score > np.mean(scores)
        ]
        return mi_features
    except Exception as e:
        raise e


def train_model(
    feature_matrix: pd.DataFrame, target_vector: pd.Series, forecast_horizon: int = 24
) -> tuple[CatBoostRegressor, list[str]]:
    """Trains, cross-validates, and optimizes hyperparameters for an object
    of type, 'CatBoostRegressor'

    Args:
        feature_matrix (pd.DataFrame): Matrix of lag features, window features,
        and datetime features
        target_vector (pd.Series): Stationary univariate time series
        forecast_horizon (int, optional): Number of time steps to forecast. Defaults to 24.

    Returns:
        tuple[CatBoostRegressor, list[str]]: Trained and cross-validated model, and a list
        containing the most informative features
    """
    try:
        logging.info("Initiating time series cross-validation and hyperparameter tuning...")
        train_indices: list[int] = [
            feature_matrix.shape[0] + (i * forecast_horizon) for i in range(-10, 0)
        ]
        val_indices: list[int] = [idx + forecast_horizon for idx in train_indices]
        idx_pairs: list[tuple[int, int]] = [
            idx_pair for idx_pair in zip(train_indices, val_indices) if idx_pair[0] > 0
        ]
        n_folds: int = len(idx_pairs)
        tss: TimeSeriesSplit = TimeSeriesSplit(n_splits=n_folds, test_size=forecast_horizon, gap=0)
        grid_search: GridSearchCV = GridSearchCV(
            estimator=CatBoostRegressor(loss_function="RMSE", n_estimators=100, silent=True),
            param_grid=OmegaConf.to_container(CONFIG.train.catboost_params),
            scoring="r2",
            refit="r2",
            cv=tss,
            n_jobs=-1,
            verbose=0,
        )
        mi_features: list[str] = get_informative_features(feature_matrix, target_vector)
        grid_search.fit(feature_matrix[mi_features].values, target_vector.values)
        logging.info("Model training complete.")
        return grid_search.best_estimator_, mi_features
    except Exception as e:
        raise e


if __name__ == "__main__":
    start: str = random.choice(get_timestamps())
    df_stationary, target = preprocess_data(start).pipe(ensure_stationarity)
    x_matrix, y_vector = transform_data(df_stationary, target)
    reg, features = train_model(x_matrix, y_vector)
    print(pd.concat((x_matrix[features], y_vector), axis=1).head())
