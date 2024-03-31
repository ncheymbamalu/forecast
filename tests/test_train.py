import random

import pandas as pd
import pytest

from catboost import CatBoostRegressor

from src.ingest import ensure_stationarity, get_timestamps, preprocess_data
from src.train import train_model
from src.transform import transform_data


@pytest.fixture
def get_model_inputs() -> tuple[pd.DataFrame, pd.Series]:
    start: str = random.choice(get_timestamps())
    df, target_name = preprocess_data(start).pipe(ensure_stationarity)
    return transform_data(df, target_name)


def test_train_model(get_model_inputs) -> None:
    model, _ = train_model(*get_model_inputs)
    assert isinstance(model, CatBoostRegressor)
