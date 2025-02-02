import pandas as pd
import os
from src.config import S3Config
import pyarrow as pa

default_columns_to_save = S3Config.default_columns_to_save


def save_forecast_locally_as_pd(
    forecast_df: pd.DataFrame, local_path: str, columns_to_save: list[str] | None = None
):
    if columns_to_save is None:
        columns_to_save = default_columns_to_save
    forecast_df_save = forecast_df[columns_to_save]

    if os.path.exists(local_path):
        # Load existing data
        existing_df = pd.read_csv(local_path, parse_dates=["timestamp"])
        existing_df["timestamp"] = pd.to_datetime(existing_df["timestamp"], utc=True)
        existing_df["timestamp"] = existing_df["timestamp"].apply(
            lambda x: x.tz_convert("Europe/Berlin")
        )
        existing_df["date"] = existing_df["timestamp"].dt.date

        overlapping_dates = forecast_df_save["date"].unique()
        # Filter out rows in `existing_df` with overlapping dates
        existing_df = existing_df[~existing_df["date"].isin(overlapping_dates)]
        merged_df = pd.concat([existing_df, forecast_df_save], ignore_index=True)

    else:
        # If no file exists, simply use the new DataFrame
        merged_df = forecast_df_save

        # Save the updated DataFrame to the specified path
    merged_df.to_csv(local_path, mode="w+", index=False)


def save_forecast_s3(
    forecast_df,
    s3_path: str,
    columns_to_save: list[str] | None = None,
    storage_config: dict[str, str] | None = None,
    mode: str = "append",
    partition_by=None,
    partition_filters=None,
):

    from deltalake.writer import write_deltalake

    if columns_to_save is None:
        columns_to_save = default_columns_to_save
    forecast_df_save = forecast_df[columns_to_save]

    schema_forecast = pa.schema(
        [
            ("timestamp", pa.timestamp("us", tz="UTC")),
            ("weekday", pa.int32()),
            ("date", pa.date32()),
            ("hour", pa.int32()),
            ("weekofyear", pa.int32()),
            ("year_week", pa.string()),
            ("holiday", pa.int32()),
            ("bridge_holiday", pa.int32()),
            ("floor", pa.int32()),
            ("forecast_q_5", pa.float64()),
            ("forecast_q_50", pa.float64()),
            ("forecast_q_95", pa.float64()),
        ]
    )

    assert set(columns_to_save) == set(
        schema_forecast.names
    ), f"Schema {schema_forecast} does not correspond to the columns to save: {columns_to_save}"

    write_deltalake(
        s3_path,
        forecast_df_save,
        schema=schema_forecast,
        mode=mode,
        storage_options=storage_config,
        partition_by=partition_by,
        partition_filters=partition_filters,
    )


if __name__ == "__main__":
    # TODO:

    from src.utils.timestamp_type_convertor import (
        convert_str_to_berlin_zone_timestamp_column,
    )
    from src.config import EnvVars
    from src.data_steps.data_importer import get_s3_data

    parent_directory = os.path.abspath(os.path.join(os.getcwd(), ".."))
    forecast_df_path = os.path.join(parent_directory, "../data/forecast_data.csv")
    forecast_df = pd.read_csv(forecast_df_path)
    forecast_df = convert_str_to_berlin_zone_timestamp_column(
        forecast_df, column_name="timestamp"
    )
    floor = 1
    forecast_df_floor = forecast_df[forecast_df["floor"] == floor].reset_index(
        drop=True
    )

    forecast_df_floor.date = forecast_df_floor.date.astype("datetime64[ns]").dt.date

    forecast_to_save = forecast_df_floor[
        forecast_df_floor.year_week.isin(["2024-W31", "2024-W32"])
    ]

    storage_config = {
        k: v for k, v in EnvVars.__dict__.items() if not k.startswith("__")
    }

    s3_path = "s3://forecast-modbus/cologne/test/"

    save_forecast_s3(
        forecast_to_save,
        storage_config=storage_config,
        s3_path=s3_path,
        mode="overwrite",
        partition_by=["year_week"],
    )

    old_fc_load = get_s3_data(delta_table_path=s3_path, storage_config=storage_config)
    new_forecast_to_save = forecast_df_floor[
        forecast_df_floor.year_week.isin(["2024-W32", "2024-W33"])
    ]
    new_year_weeks = new_forecast_to_save.year_week.unique().tolist()
    concat_df = pd.concat(
        [
            old_fc_load[~old_fc_load.year_week.isin([new_year_weeks])],
            new_forecast_to_save,
        ]
    )

    print(concat_df.year_week.unique())
    print(old_fc_load.year_week.unique(), len(old_fc_load))
    print(old_fc_load[old_fc_load.year_week.isin(["2024-W32"])])

    save_forecast_s3(
        concat_df,
        storage_config=storage_config,
        s3_path=s3_path,
        mode="overwrite",
        partition_by=["year_week"],
    )

    df = get_s3_data(delta_table_path=s3_path, storage_config=storage_config)
    print(df.year_week.unique(), len(df), df)
    print(df[df.year_week.isin(["2024-W32"])])
    print(df[df.year_week.isin(["2024-W31"])])
