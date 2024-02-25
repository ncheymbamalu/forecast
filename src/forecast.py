"""
Recursive/multi-step forecasting
"""

import random

from datetime import timedelta
from typing import List

import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from src.ingest import ensure_stationarity, get_timestamps, preprocess_data
from src.logger import logging
from src.train import train_model
from src.transform import create_datetime_features, transform_data


def get_recursive_forecast(
    stationary_data: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    target_vector: pd.Series,
    model: XGBRegressor,
    mi_features: List[str],
) -> pd.Series:
    """Generates a recursive forecast

    Args:
        stationary_data (pd.DataFrame): DataFrame that contains the pre-transformation
        stationary univariate time series
        feature_matrix (pd.DataFrame): Matrix of lag features, window features, and
        datetime features
        target_vector (pd.Series): Post-transformation stationary univariate time series
        model (XGBRegressor): Object of type, 'XGBRegressor'
        mi_features (List[str]): List containing the most informative features based 
        on the mutual information criterion

    Returns:
        pd.Series: Recursive forecast
    """
    try:
        logging.info("Forecasting initiated...")

        # extract the 'relevant' lag features and the column index of each
        lag_cols: List[str] = [col for col in feature_matrix.columns if "lag" in col]
        rel_lag_cols: List[str] = [col for col in lag_cols if col in mi_features]
        lag_col_idx: List[int] = [lag_cols.index(col) for col in rel_lag_cols]

        # extract the 'relevant' window features and the column index of each
        window_cols: List[str] = [col for col in feature_matrix.columns if "window" in col]
        rel_window_cols: List[str] = [col for col in window_cols if col in mi_features]
        window_col_idx: List[int] = [window_cols.index(col) for col in rel_window_cols]

        # extract the 'relevant' datetime features
        dt_cols: List[str] = [
            col 
            for col in feature_matrix.columns 
            if (col in mi_features) and (col not in lag_cols + window_cols)
        ]

        # create the out-of-sample timestamps
        forecast_horizon: int = 24
        start: pd.Timestamp = feature_matrix.index[-1] + timedelta(hours=1)
        end: pd.Timestamp = start + timedelta(hours=forecast_horizon - 1)
        forecast_indices: pd.DatetimeIndex = pd.date_range(start, end, freq="H")

        # create a list of lists, ...
        # where each list is a row of values for the out-of-sample datetime features
        x_dts: List[List[int]] = (
            create_datetime_features(pd.DataFrame(index=forecast_indices))[dt_cols].values.tolist()
        )

        # create a list of window sizes, ...
        # which will be used to compute values for the out-of-sample window features
        window_sizes: List[int] = sorted(set(int(col.split("_")[-1]) for col in window_cols))
        dynamic_window: List[float] = target_vector.iloc[-max(window_sizes) :].tolist()

        # fit the model
        model.fit(feature_matrix[mi_features].values, target_vector.values)

        # create the 1st input
        x_dt: List[int] = x_dts[0]
        x_window: List[List[float]] = [
            [np.mean(dynamic_window[-window_size:]), np.std(dynamic_window[-window_size:])]
            for window_size in window_sizes
        ]
        x_rel_window: List[float] = np.array(x_window).ravel()[window_col_idx].tolist()
        x_lag: List[float] = feature_matrix.iloc[-1][lag_cols].values.tolist()
        x_lag = x_lag[1:] + [target_vector.iloc[-1]]
        x_rel_lag: List[float] = np.array(x_lag)[lag_col_idx].tolist()
        x: np.ndarray = np.array(x_dt + x_rel_window + x_rel_lag)

        # get the 1st prediction and add it to a list named, 'forecast'
        yhat: float = model.predict(x.reshape(1, -1))[0]
        forecast: List[float] = [yhat]

        for x_dt in x_dts[1:]:

            # update the input
            dynamic_window = dynamic_window[1:] + [yhat]
            x_window = [
                [np.mean(dynamic_window[-window_size:]), np.std(dynamic_window[-window_size:])]
                for window_size in window_sizes
            ]
            x_rel_window = np.array(x_window).ravel()[window_col_idx].tolist()
            x_lag = x_lag[1:] + [yhat]
            x_rel_lag = np.array(x_lag)[lag_col_idx].tolist()
            x = np.array(x_dt + x_rel_window + x_rel_lag)

            # get the prediction and append it to the 'forecast' list
            yhat = model.predict(x.reshape(1, -1))[0]
            forecast.append(yhat)

        # update the forecast if the target is a once-differenced time series
        original_target: str = stationary_data.columns[0]
        forecast: np.ndarray = (
            stationary_data.loc[feature_matrix.index[-1], original_target] + np.cumsum(forecast)
            if target_vector.name == "diff"
            else np.array(forecast)
        )
        logging.info("The forecast has been generated!")
        return pd.Series(forecast, index=forecast_indices, name="recursive_forecast")
    except Exception as e:
        raise e


if __name__ == "__main__":
    initial_timestamp: str = random.choice(get_timestamps())
    df_stationary, target = preprocess_data(initial_timestamp).pipe(ensure_stationarity)
    x_matrix, y_vector = transform_data(df_stationary, target)
    reg, features = train_model(x_matrix, y_vector)
    recursive_forecast: pd.Series = get_recursive_forecast(df_stationary, x_matrix, y_vector, reg, features)
    print(recursive_forecast)