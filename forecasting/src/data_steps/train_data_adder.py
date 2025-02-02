import pandas as pd


def add_holiday_weeks(
    forecast_features_df: pd.DataFrame,
    df_floor_cv: pd.DataFrame,
    df_floor_test: pd.DataFrame,
    df_floor_history: pd.DataFrame,
    prev_holiday_weeks_to_take: int,
) -> pd.DataFrame:
    holidays_in_fh: bool = (forecast_features_df["holiday"] > 0).any()
    holidays_in_test: bool = (df_floor_test["holiday"] > 0).any()

    num_holidays_in_cv: int = df_floor_cv[df_floor_cv["holiday"] > 0]["date"].nunique()

    # weeks with holidays in the full dataframe:
    holiday_year_weeks_in_history = (
        df_floor_history[df_floor_history["holiday"] > 0]["year_week"].unique().tolist()
    )

    # append previous holidays if needed
    if (
        holidays_in_fh or holidays_in_test
    ) and num_holidays_in_cv < prev_holiday_weeks_to_take:

        num_holidays_to_add = prev_holiday_weeks_to_take - num_holidays_in_cv
        list_holidays_to_add = holiday_year_weeks_in_history[-num_holidays_to_add:]

        df_holiday_to_add = df_floor_history[
            df_floor_history["year_week"].isin(list_holidays_to_add)
        ]

        return pd.concat([df_holiday_to_add, df_floor_cv]).reset_index(drop=True)

    else:
        return df_floor_cv


def add_bridge_day_weeks(
    forecast_features_df: pd.DataFrame,
    df_floor_cv: pd.DataFrame,
    df_floor_test: pd.DataFrame,
    df_floor_history: pd.DataFrame,
    prev_bridge_days_to_take: int,
) -> pd.DataFrame:
    bridge_days_in_fh: bool = (forecast_features_df["bridge_holiday"] > 0).any()
    bridge_days_in_test: bool = (df_floor_test["bridge_holiday"] > 0).any()

    num_bridge_days_in_cv: int = df_floor_cv[df_floor_cv["bridge_holiday"] > 0][
        "date"
    ].nunique()

    bridge_days_year_weeks_in_history: list = (
        df_floor_history[df_floor_history["bridge_holiday"] > 0]["year_week"]
        .unique()
        .tolist()
    )

    # append previous bridge holidays if needed
    if (
        bridge_days_in_fh or bridge_days_in_test
    ) and num_bridge_days_in_cv < prev_bridge_days_to_take:

        num_bridge_days_to_add: int = prev_bridge_days_to_take - num_bridge_days_in_cv
        list_bridge_days_to_add = bridge_days_year_weeks_in_history[
            -num_bridge_days_to_add:
        ]

        df_bridge_holiday_to_add = df_floor_history[
            df_floor_history["year_week"].isin(list_bridge_days_to_add)
        ]

        return pd.concat([df_bridge_holiday_to_add, df_floor_cv]).reset_index(drop=True)

    else:
        return df_floor_cv
