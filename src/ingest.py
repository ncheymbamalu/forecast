"""
Data ingestion and pre-processing
"""

import random

from datetime import timedelta

import pandas as pd

from omegaconf import DictConfig, ListConfig, OmegaConf
from statsmodels.tsa.stattools import adfuller

from src.logger import logging

CONFIG: DictConfig | ListConfig = OmegaConf.load(r"./config.yaml")


def get_timestamps() -> list[str]:
    """Creates a list of string-formatted timestamps

    Returns:
        list[str]: String-formatted timestamps
    """
    try:
        data: pd.DataFrame = pd.read_csv(CONFIG.ingest.raw_data_path)
        min_timestamp: str = str(pd.to_datetime(data.Datetime).min())
        max_timestamp: str = str(pd.to_datetime(data.Datetime).max() - timedelta(days=11))
        return [str(ts) for ts in pd.date_range(min_timestamp, max_timestamp, freq="H")]
    except Exception as e:
        raise e


def preprocess_data(start: str) -> pd.DataFrame:
    """Pre-processes raw time series data

    Args:
        start (str): Starting timestamp of the time series

    Returns:
        pd.DataFrame: Pre-processed time series data
    """
    try:
        logging.info("Fetching the raw data...")
        end: str = str(pd.to_datetime(start) + timedelta(days=10))
        raw_data: pd.DataFrame = pd.read_csv(
            CONFIG.ingest.raw_data_path, index_col="Datetime", parse_dates=True
        )
        processed_data: pd.DataFrame = (
            raw_data.loc[
                (raw_data.index > start)
                & (raw_data.index <= end)
                & (raw_data["PJME_MW"] > 19_200)
                & (~raw_data.index.duplicated(keep="first"))
            ]
            .rename({"PJME_MW": "energy_consumption_mw"}, axis=1)
            .sort_index()
            .asfreq(freq="H")
            .rename_axis(None)
            .copy(deep=True)
        )
        logging.info("Data ingestion and pre-processing complete.")
        return processed_data
    except Exception as e:
        raise e


def ensure_stationarity(
    data: pd.DataFrame, target_name: str = "energy_consumption_mw"
) -> tuple[pd.DataFrame, str]:
    """Returns a DataFrame that contains a stationary univariate time series
    and its column name

    Args:
        data (pd.DataFrame): DataFrame that contains a univariate time series
        target_name (str): Column name of the univariate time series

    Returns:
        tuple[pd.DataFrame, str]: DataFrame that contains a stationary univariate
        time series and its column name
    """
    try:
        p_value: float = adfuller(data[target_name].dropna())[1]
        df: pd.DataFrame = (
            data.dropna()
            .assign(
                prev=data[target_name].dropna().shift(periods=1),
                diff=data[target_name].dropna().diff(periods=1),
            )
            .copy(deep=True)
        )
        return (data, target_name) if p_value < 0.05 else (df, "diff")
    except Exception as e:
        raise e


if __name__ == "__main__":
    initial_timestamp: str = random.choice(get_timestamps())
    df_stationary, target = preprocess_data(initial_timestamp).pipe(ensure_stationarity)
    print(df_stationary.head())
