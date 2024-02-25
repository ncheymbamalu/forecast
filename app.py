"""
Streamlit Web Application
"""

from typing import Tuple

import pandas as pd
import streamlit as st

from plotly import graph_objs as go

from src.forecast import get_recursive_forecast
from src.ingest import ensure_stationarity, get_timestamps, preprocess_data
from src.train import train_model
from src.transform import transform_data


@st.cache_data
def generate_forecast(start: str) -> Tuple[int, pd.DataFrame, pd.Series]:
    """Returns the forecast horizon, model inputs, and the in-sample data and forecast

    Args:
        start (str): Initial timestamp of the time series

    Returns:
        Tuple[int, pd.DataFrame, pd.Series]: Forecast horizon (int), input features
        and the target (pd.DataFrame), in-sample data and forecast (pd.Series)
    """
    try:
        df, target = preprocess_data(start).pipe(ensure_stationarity)
        x_matrix, y_vector = transform_data(df, target)
        model, features = train_model(x_matrix, y_vector)
        forecast: pd.Series = get_recursive_forecast(df, x_matrix, y_vector, model, features)
        return (
            forecast.shape[0],
            pd.concat((x_matrix[features], y_vector), axis=1),
            pd.concat((df["energy_consumption_mw"], forecast), axis=0),
        )
    except Exception as e:
        raise e


st.title("Energy Consumption Forecasting :clock9:")
initial_timestamp: str = st.selectbox("Select the starting timestamp", get_timestamps())
horizon, train_data, series = generate_forecast(initial_timestamp)
series.name = "energy_consumption_mw"

st.write("""#### Hourly Energy Consumption""")
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=series.iloc[:-horizon].index,
        y=series.iloc[:-horizon],
        name="In-sample data",
        mode="lines",
        line={"color": "aqua", "width": 2},
    )
)
fig.add_trace(
    go.Scatter(
        x=series.iloc[-horizon:].index,
        y=series.iloc[-horizon:],
        name="Forecast",
        mode="lines",
        line={"color": "aqua", "width": 4, "dash": "dot"},
    )
)
fig.update_layout(
    autosize=True,
    width=1400,
    height=600,
    xaxis={
        "showline": True,
        "showgrid": False,
        "showticklabels": True,
        "tickfont": {"family": "Arial", "size": 14},
    },
    xaxis_rangeslider_visible=True,
    yaxis={
        "showline": True,
        "showgrid": False,
        "showticklabels": True,
        "tickfont": {"family": "Arial", "size": 14},
    },
    showlegend=False,
)
fig.update_xaxes(
    title_text="Timestamp (UTC)", title_font={"size": 16, "family": "Arial"}, title_standoff=20
)
fig.update_yaxes(
    title_text="Energy Consumption (MW)",
    title_font={"size": 16, "family": "Arial"},
    title_standoff=20,
)
st.plotly_chart(fig)

if st.checkbox("Original Time Series & Forecast"):
    st.dataframe(
        series.rename_axis("timestamp_utc")
        .to_frame()
        .style.format(precision=0)
        .applymap(
            lambda _: "background-color: teal", subset=(series.tail(horizon).index, slice(None))
        )
    )

if st.checkbox("Training Data"):
    st.dataframe(train_data.rename_axis("timestamp_utc"))
