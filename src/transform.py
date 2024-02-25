"""
Data transformation
"""

import random

from typing import List, Tuple

import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import pacf

from src.ingest import ensure_stationarity, get_timestamps, preprocess_data
from src.logger import logging


def create_lag_features(data: pd.DataFrame, target_name: str) -> pd.DataFrame:
    """Creates a matrix of lag features from a univariate time series

    Args:
        data (pd.DataFrame): DataFrame that contains a univariate time series
        target_name (str): Column name of the univariate time series

    Returns:
        pd.DataFrame: Matrix of lag features
    """
    try:
        lag_correlations: np.ndarray = pacf(data[target_name].dropna(), nlags=30, method="ywmle")
        max_lag: int = np.where(np.abs(lag_correlations) > 0.16)[0][-1]
        lags: List[pd.Series] = [
            data[target_name].dropna().shift(periods=lag) for lag in reversed(range(1, max_lag + 1))
        ]
        cols: List[str] = [f"lag_{i}" for i in reversed(range(1, max_lag + 1))]
        df: pd.DataFrame = pd.concat(lags, axis=1).dropna()
        df.columns = cols
        return df
    except Exception as e:
        raise e


def create_window_features(data: pd.DataFrame, target_name: str, max_window_size: int = 24) -> pd.DataFrame:
    """Creates a matrix of window features from a univariate time series

    Args:
        data (pd.DataFrame): DataFrame that contains a univariate time series
        target_name (str): Column name of the univariate time series
        max_window_size (int, optional): Maximum number of lags to use to create
        window features. Defaults to 24.

    Returns:
        pd.DataFrame: Matrix of window features
    """
    try:
        cols: List[str] = (
            np.array(
                [
                    [f"mean_window_{window_size}", f"std_window_{window_size}"]
                    for window_size in reversed(range(4, max_window_size + 1, 4))
                ]
            )
            .ravel()
            .tolist()
        )
        dfs: List[pd.DataFrame] = [
            data[target_name].rolling(window=window_size, min_periods=1).agg(["mean", "std"]).shift(periods=1)
            for window_size in reversed(range(4, max_window_size + 1, 4))
        ]
        df: pd.DataFrame = pd.concat(dfs, axis=1).dropna()
        df.columns = cols
        return df
    except Exception as e:
        raise e


def create_datetime_features(data: pd.DataFrame) -> pd.DataFrame:
    """Creates a matrix of datetime features from a time series index

    Args:
        data (pd.DataFrame): DataFrame that contains a Pandas Datetime Index

    Returns:
        pd.DataFrame: Matrix of datetime features
    """
    try:
        df: pd.DataFrame = pd.DataFrame(index=data.dropna().index)
        df = df.assign(
            hour=df.index.hour,
            time_of_day=[
                1 if x in range(5, 12) 
                else 2 if x in range(12, 17) 
                else 3 if x in range(17, 21) 
                else 4 for x in df.index.hour
            ],
        )
        return df
    except Exception as e:
        raise e


def transform_data(stationary_data: pd.DataFrame, target_name: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Transforms a stationary univariate time series to a matrix of lag
    features, window features, datetime features, and a target

    Args:
        stationary_data (pd.DataFrame): DataFrame that contains a stationary
        univariate time series
        target_name (str): Column name of the stationary univariate time series

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Feature matrix and target vector
    """
    try:
        logging.info("Transforming the data...")
        feature_matrix: pd.DataFrame = pd.concat(
            [
                create_datetime_features(stationary_data),
                create_window_features(stationary_data, target_name),
                create_lag_features(stationary_data, target_name),
            ],
            axis=1,
        ).dropna()
        target_vector: pd.Series = stationary_data.loc[feature_matrix.index, target_name]
        logging.info("Data transformation complete.")
        return feature_matrix, target_vector
    except Exception as e:
        raise e


if __name__ == "__main__":
    start: str = random.choice(get_timestamps())
    df_stationary, target = preprocess_data(start).pipe(ensure_stationarity)
    x_matrix, y_vector = transform_data(df_stationary, target)
    print(pd.concat((x_matrix, y_vector), axis=1).head())