import random

import pandas as pd
import pytest

from omegaconf import DictConfig, ListConfig, OmegaConf

from src.evaluate import get_rsquared
from src.ingest import get_timestamps
from src.pipeline import ForecastingPipeline

CONFIG: DictConfig | ListConfig = OmegaConf.load(r"./config.yaml")


@pytest.mark.parametrize(
    argnames="starting_timestamp",
    argvalues=[(start) for start in set(random.sample(get_timestamps(), 100))],
)
def test_forecast(starting_timestamp: str) -> None:
    fp: ForecastingPipeline = ForecastingPipeline(start=starting_timestamp)
    forecast: pd.Series = fp.run()
    target: pd.Series = pd.read_csv(
        CONFIG.ingest.raw_data_path, parse_dates=["Datetime"], index_col="Datetime"
    ).loc[forecast.index, "PJME_MW"]
    r_squared: float = get_rsquared(target, forecast)
    assert r_squared >= 0.8**2
