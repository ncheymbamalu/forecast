import random

import pandas as pd
import pytest

from src.ingest import ensure_stationarity, get_timestamps, preprocess_data
from src.transform import transform_data


@pytest.fixture
def ingest_data() -> list[tuple[pd.DataFrame, str]]:
    return [
        preprocess_data(start).pipe(ensure_stationarity)
        for start in random.sample(get_timestamps(), 10)
    ]


def test_transform_data(ingest_data) -> None:
    for df, target_name in ingest_data:
        feature_matrix, target_vector = transform_data(df, target_name)
        assert isinstance(feature_matrix, pd.DataFrame)
        assert isinstance(target_vector, pd.Series)
        assert feature_matrix.shape[0] > 0
        assert feature_matrix.shape[1] > 0
        assert target_vector.shape[0] > 0
        assert feature_matrix.shape[0] == target_vector.shape[0]
        assert feature_matrix.isna().sum().sum() == 0
        assert target_vector.isna().sum() == 0
