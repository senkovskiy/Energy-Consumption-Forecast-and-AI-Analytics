import pandas as pd
from deltalake import DeltaTable
import duckdb

SILVER_TABLE_BUCKET_NAME = "ts-modbus-delta-bucket"
SILVER_TABLE_NAME = "ts_modbus_silver_delta_table"
silver_delta_table_path = f"s3://{SILVER_TABLE_BUCKET_NAME}/v1/{SILVER_TABLE_NAME}/"


def get_s3_data(
    delta_table_path: str = silver_delta_table_path,
    storage_config: dict[str, str] | None = None,
) -> pd.DataFrame:

    dataset = DeltaTable(
        delta_table_path, storage_options=storage_config
    ).to_pyarrow_dataset()

    df = duckdb.execute(
        """
        SELECT * 
        FROM dataset;
        """
    ).fetch_df()

    return df
