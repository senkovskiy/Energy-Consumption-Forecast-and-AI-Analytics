import pandas as pd


def convert_str_to_berlin_zone_timestamp_column(
    df: pd.DataFrame, column_name: str = "recordedtimestamp"
) -> pd.DataFrame:
    df[column_name] = pd.to_datetime(df[column_name], utc=True)
    df[column_name] = df[column_name].apply(lambda x: x.tz_convert("Europe/Berlin"))
    return df
