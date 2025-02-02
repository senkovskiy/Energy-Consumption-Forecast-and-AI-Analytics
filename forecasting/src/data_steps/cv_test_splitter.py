import pandas as pd


def cv_test_split(
    df_floor_full: pd.DataFrame, number_of_weeks_cv: int, number_of_weeks_test: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    number_total_train_weeks = number_of_weeks_cv + number_of_weeks_test

    # split based on year_week, e.g. ""2024-W32"", to handle multiple years if needed
    list_total_train_weeks = df_floor_full.year_week.unique()[
        -number_total_train_weeks:
    ]
    list_of_weeks_cv = list_total_train_weeks[:number_of_weeks_cv]
    list_of_weeks_test = list_total_train_weeks[number_of_weeks_cv:]

    df_floor_cv = df_floor_full[df_floor_full.year_week.isin(list_of_weeks_cv)]
    df_floor_test = df_floor_full[df_floor_full.year_week.isin(list_of_weeks_test)]

    df_floor_history = df_floor_full[
        ~df_floor_full.year_week.isin(list_total_train_weeks)
    ]

    return df_floor_cv, df_floor_test, df_floor_history
