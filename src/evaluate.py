"""
Forecast evaluation
"""

import numpy as np
import pandas as pd

from omegaconf import DictConfig, ListConfig, OmegaConf

from src.logger import logging

CONFIG: DictConfig | ListConfig = OmegaConf.load(r"./config.yaml")


def get_rsquared(y: pd.Series | np.ndarray, y_hat: pd.Series | np.ndarray) -> float:
    """Computes the R² between y and y_hat

    Args:
        y (pd.Series | np.ndarray): Observations
        y_hat (pd.Series | np.ndarray): Predictions

    Returns:
        float: R² between y and y_hat
    """
    try:
        y = y.ravel() if y.ndim > 1 else y
        y_hat = y_hat.ravel() if y_hat.ndim > 1 else y_hat
        base_errors: pd.Series | np.ndarray = y - y.mean()
        sst: float = base_errors.dot(base_errors)
        model_errors: pd.Series | np.ndarray = y - y_hat
        sse: float = model_errors.dot(model_errors)
        return 1 - (sse / sst)
    except Exception as e:
        raise e


def evaluate_forecast(forecast: pd.Series) -> None:
    """Logs the R² between the forecast and out-of-sample target

    Args:
        forecast (pd.Series): Forecasted time series
    """
    try:
        target: pd.Series = pd.read_csv(
            CONFIG.ingest.raw_data_path, parse_dates=["Datetime"], index_col="Datetime"
        ).loc[forecast.index, "PJME_MW"]
        metric: float = get_rsquared(target, forecast)
        logging.info("An R² of %s was produced.", round(metric, 2))
    except Exception as e:
        raise e
