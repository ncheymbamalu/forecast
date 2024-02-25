"""
Forecast evaluation
"""

import random

from datetime import timedelta
from typing import Union

import numpy as np
import pandas as pd

from src.forecast import get_recursive_forecast
from src.ingest import ensure_stationarity, get_timestamps, preprocess_data
from src.logger import logging
from src.train import train_model
from src.transform import transform_data


def get_rsquared(y: Union[np.ndarray, pd.Series], y_hat: Union[np.ndarray, pd.Series]) -> float:
    """Computes the R² between y and y_hat

    Args:
        y (Union[np.ndarray, pd.Series]): Target vector
        yhat (Union[np.ndarray, pd.Series]): Prediction vector

    Returns:
        float: R²
    """
    try:
        y = y.ravel() if y.ndim > 1 else y
        y_hat = y_hat.ravel() if y_hat.ndim > 1 else y_hat
        base_errors: Union[np.ndarray, pd.Series] = y - y.mean()
        sst: float = base_errors.dot(base_errors)
        model_errors: Union[np.ndarray, pd.Series] = y - y_hat
        sse: float = model_errors.dot(model_errors)
        return 1 - (sse / sst)
    except Exception as e:
        raise e


def evaluate_forecast(start: str) -> None:
    """Logs the R² between the forecast and out-of-sample target

    Args:
        start (str): Starting timestamp of the in-sample target
    """
    try:
        df, target_name = preprocess_data(start).pipe(ensure_stationarity)
        x_matrix, y_vector = transform_data(df, target_name)
        model, features = train_model(x_matrix, y_vector)
        forecast: pd.Series = get_recursive_forecast(df, x_matrix, y_vector, model, features)
        forecast_start: str = str(forecast.index[0] - timedelta(hours=1))
        target: pd.Series = preprocess_data(forecast_start).loc[forecast.index].iloc[:, 0]
        metric: float = get_rsquared(target, forecast)
        logging.info("An R² of %s was produced.", round(metric, 2))
    except Exception as e:
        raise e


if __name__ == "__main__":
    evaluate_forecast(random.choice(get_timestamps()))
