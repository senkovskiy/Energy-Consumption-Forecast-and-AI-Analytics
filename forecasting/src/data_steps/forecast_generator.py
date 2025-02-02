import pandas as pd
from catboost import CatBoostRegressor
from src.data_steps.preprocessor import (
    generate_time_delta_hours,
    generate_time_features,
    add_holiday_column,
    add_bridge_holiday_dates,
)


def generate_2_weeks_forecast_features_df(train_df: pd.DataFrame) -> pd.DataFrame:
    assert "floor" in train_df.columns, "floor column should be in train data"
    assert "timestamp" in train_df.columns, "timestamp column should be in train data"
    assert (
        len(train_df["floor"].unique()) == 1
    ), "the train dataframe should be from one floor"
    # generate time intervals the same as in train_df (median)
    interval_minutes = int(60 * train_df["time_delta_hours"].median())
    start_timestamp = train_df["timestamp"].iloc[-1] + pd.Timedelta(
        minutes=interval_minutes
    )
    intervals_in_2_weeks = int(14 * 24 * 60 // interval_minutes)
    datetime_index = pd.date_range(
        start=start_timestamp,
        periods=intervals_in_2_weeks,
        freq=f"{interval_minutes}min",
    )
    forecast_df = pd.Series(datetime_index).to_frame(name="timestamp")
    forecast_df = generate_time_features(forecast_df)
    forecast_df["floor"] = train_df["floor"].unique()[0]
    forecast_df = generate_time_delta_hours(forecast_df)
    forecast_df = add_holiday_column(forecast_df)
    forecast_df = add_bridge_holiday_dates(forecast_df)
    return forecast_df


def generate_forecast(
    model: CatBoostRegressor,
    df: pd.DataFrame,
    forecast_col_names: list[str],
    feature_columns: list | None = None,
) -> pd.DataFrame:
    assert len(df.floor.unique()) == 1, "the train dataframe should be from one floor"
    feature_columns = (
        df.columns.to_list() if feature_columns is None else feature_columns
    )
    forecasts = model.predict(df[feature_columns])
    df[forecast_col_names] = forecasts[:, [0, 1, 2]]
    return df
