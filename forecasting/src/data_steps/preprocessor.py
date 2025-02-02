import pandas as pd
from datetime import date, timedelta
import re
from holidays import country_holidays


def preprocess_s3_data(
    df: pd.DataFrame,
    time_stamp_column: str = "recordedtimestamp",
    floor_column: str = "systemname",
    energy_column: str = "value",
    energy_scale_factor: int = 100,
) -> pd.DataFrame:
    df = rename_data_columns(df, time_stamp_column, floor_column, energy_column)
    df = scale_energy_by_factor(df, factor=energy_scale_factor)
    df = remove_duplicated_timestamps(df)
    try:
        df["timestamp"] = df["timestamp"].dt.tz_convert("Europe/Berlin")
    except ValueError:
        df["timestamp"] = df["timestamp"].apply(lambda x: x.tz_convert("Europe/Berlin"))
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = convert_floor_to_int(df)
    df = generate_time_features(df)
    df = calculate_energy_delta(df)
    if "time_delta_hours" not in df.columns:
        df = generate_time_delta_hours(df)
    df = generate_power_column(df)
    df = add_holiday_column(df)
    df = add_bridge_holiday_dates(df)
    return df


def rename_data_columns(
    df: pd.DataFrame, time_stamp_column: str, floor_column: str, energy_column: str
) -> pd.DataFrame:
    return df.rename(
        columns={
            time_stamp_column: "timestamp",
            floor_column: "floor",
            energy_column: "energy",
        }
    )


def convert_floor_to_int(df: pd.DataFrame) -> pd.DataFrame:
    df["floor"] = df["floor"].apply(lambda x: int(re.findall(r"\d+", x)[0]))
    return df


def scale_energy_by_factor(df: pd.DataFrame, factor: int) -> pd.DataFrame:
    df["energy"] = df["energy"] / factor
    return df


def remove_duplicated_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("floor")
        .apply(lambda x: x.drop_duplicates(subset="timestamp", keep="last"))
        .reset_index(drop=True)
    )


def calculate_energy_delta(df: pd.DataFrame) -> pd.DataFrame:
    # this assumes that the dataframe is already sorted by 'timestamp'
    df["energy_delta"] = df.groupby("floor")["energy"].diff()
    # remove nans and negative energy deltas (if any)
    df = df[df["energy_delta"].notna()]
    df = df[df["energy_delta"] > 0]
    return df


def generate_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = df["timestamp"].apply(lambda x: x.date())
    df["weekday"] = df["timestamp"].apply(lambda x: x.weekday())
    df["daytime"] = df["timestamp"].apply(lambda x: x.time())
    df["hour"] = df["timestamp"].apply(lambda x: x.hour)
    df["weekofyear"] = df["timestamp"].apply(lambda x: x.week)
    df["year"] = df["timestamp"].apply(lambda x: x.year)
    df["year_week"] = df.apply(lambda x: f'{x["year"]}-W{x["weekofyear"]:02d}', axis=1)
    df["time_from_daystart"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    return df


def generate_time_delta_hours(df: pd.DataFrame) -> pd.DataFrame:
    def compute_time_deltas_hours_for_group(group):
        # time delta in fraction of an hour
        group["time_delta_hours"] = (
            group.sort_values("timestamp")["timestamp"].diff().dt.seconds / 3600
        )
        group = group.reset_index(drop=True)
        group.loc[0, "time_delta_hours"] = group.loc[1, "time_delta_hours"]
        return group

    df = (
        df.groupby("floor")
        .apply(compute_time_deltas_hours_for_group, include_groups=False)
        .reset_index()
    )
    df = df[df["time_delta_hours"] > 0]
    return df


def generate_power_column(df: pd.DataFrame) -> pd.DataFrame:
    # power in kW - the scale is independent on time_delta
    assert (
        not df["time_delta_hours"].isna().any()
    ), "Column 'time_delta_hours' contains NaN values, cannot calculate 'power'"
    df["power"] = df["energy_delta"] / df["time_delta_hours"]
    return df


def add_holiday_column(df: pd.DataFrame) -> pd.DataFrame:
    ger_holiday_days_all_years = get_holiday_dates(df)
    christmas_season_all_years = []
    for year in df.year.unique().tolist():
        christmas_season = [date(year, 12, 23) + timedelta(days=x) for x in range(10)]
        christmas_season_all_years.extend(christmas_season)
    df["holiday"] = (
        df["date"]
        .isin(christmas_season_all_years + ger_holiday_days_all_years)
        .astype(int)
    )
    return df


def add_bridge_holiday_dates(df: pd.DataFrame) -> pd.DataFrame:
    holiday_dates = get_holiday_dates(df)
    bridge_dates = get_bridge_holiday_dates(holiday_dates)
    df["bridge_holiday"] = df["date"].apply(lambda x: 1 if x in bridge_dates else 0)
    return df


def get_holiday_dates(df: pd.DataFrame) -> list[date]:
    years_list = df.year.unique().tolist()
    holiday_dates_dict = country_holidays("DE", subdiv="NW", years=years_list)
    return list(holiday_dates_dict.keys())


def get_bridge_holiday_dates(holiday_dates: list[date]) -> list[date]:
    bridge_dates = set()
    for holiday_date in holiday_dates:
        prev_day = holiday_date - timedelta(days=1)
        next_day = holiday_date + timedelta(days=1)
        # Monday bridge
        if prev_day.weekday() == 0 and prev_day not in holiday_dates:
            bridge_dates.add(prev_day)
        # Friday bridge
        if next_day.weekday() == 4 and next_day not in holiday_dates:
            bridge_dates.add(next_day)
    return sorted(list(bridge_dates))
