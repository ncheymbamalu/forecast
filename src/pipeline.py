"""
Forecasting pipeline
"""

import random

import pandas as pd

from pydantic import BaseModel

from src.evaluate import evaluate_forecast
from src.forecast import get_recursive_forecast
from src.ingest import ensure_stationarity, get_timestamps, preprocess_data
from src.train import train_model
from src.transform import transform_data


class ForecastingPipeline(BaseModel):
    """Class to encapsulate the forecasting pipeline

    Args:
        start (str): Starting timestamp of the in-sample target time series
    """

    start: str

    def forecast(self) -> pd.Series:
        """
        Generates a recursive forecast
        """
        try:
            stationary_data, target_name = preprocess_data(self.start).pipe(ensure_stationarity)
            feature_matrix, target_vector = transform_data(stationary_data, target_name)
            model, mi_features = train_model(feature_matrix, target_vector)
            recursive_forecast: pd.Series = get_recursive_forecast(
                stationary_data, feature_matrix, target_vector, model, mi_features
            )
            evaluate_forecast(recursive_forecast)
            return recursive_forecast
        except Exception as e:
            raise e


if __name__ == "__main__":
    initial_timestamp: str = random.choice(get_timestamps())
    fp: ForecastingPipeline = ForecastingPipeline(start=initial_timestamp)
    forecast: pd.Series = fp.forecast()
    print(forecast)
