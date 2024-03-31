import random

import pytest

from statsmodels.tsa.stattools import adfuller

from src.ingest import ensure_stationarity, get_timestamps, preprocess_data


@pytest.mark.parametrize(
    argnames="starting_timestamp",
    argvalues=[(start) for start in random.sample(get_timestamps(), 10)],
)
def test_ingest_data(starting_timestamp: str) -> None:
    df, target_name = preprocess_data(starting_timestamp).pipe(ensure_stationarity)
    assert df.shape[0] > 0
    assert df.shape[1] > 0
    assert adfuller(df[target_name].dropna())[1] < 0.05
    assert target_name in ["energy_consumption_mw", "diff"]
    if target_name == "diff":
        assert "prev" in df.columns
