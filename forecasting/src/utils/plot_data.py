from warnings import simplefilter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_windows(y, train_windows, test_windows, title=""):
    """Visualize training and validation windows in CV
    Modified function from https://www.sktime.net/en/v0.21.0/examples/forecasting/window_splitters.html
    """

    simplefilter("ignore", category=UserWarning)

    def get_y(length, split):
        # Create a constant vector based on the split for y-axis."""
        return np.ones(length) * split

    n_splits = len(train_windows)
    n_timepoints = len(y)
    len_test = len(test_windows[0])

    train_color, test_color = sns.color_palette("colorblind")[:2]

    fig, ax = plt.subplots(figsize=plt.figaspect(0.3))

    for i in range(n_splits):
        train = train_windows[i]
        test = test_windows[i]

        ax.plot(
            np.arange(n_timepoints), get_y(n_timepoints, i), marker="o", c="lightgray"
        )
        ax.plot(
            train,
            get_y(len(train), i),
            marker="o",
            c=train_color,
            label="Window",
        )
        ax.plot(
            test,
            get_y(len_test, i),
            marker="o",
            c=test_color,
            label="Forecasting horizon",
        )
    ax.invert_yaxis()
    # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set(
        title=title,
        ylabel="Fold number",
        xlabel="Week",
        # xticklabels=y.index,
    )
    ax.set_xticks(y.values)
    # remove duplicate labels/handles
    handles, labels = [(leg[:2]) for leg in ax.get_legend_handles_labels()]
    ax.legend(handles, labels)
    return fig


def get_windows(y, cv):
    """Generate windows"""
    train_windows = []
    test_windows = []
    for train, test in cv.split(y):
        train_windows.append(train)
        test_windows.append(test)
    return [y[list(t)] for t in train_windows], [y[list(t)] for t in test_windows]


def plot_forecast_vs_actual(forecast_df, actual_df):
    import plotly.express as px

    display_df = forecast_df.copy().sort_values("timestamp").reset_index(drop=True)
    actual_df = actual_df.sort_values("timestamp").reset_index(drop=True)

    display_df["forecast_q_5"] = display_df["forecast_q_5"] * 0.25
    display_df["forecast_q_95"] = display_df["forecast_q_95"] * 0.25
    display_df["forecast_q_50"] = display_df["forecast_q_50"] * 0.25
    time_delta_hours = actual_df["time_delta_hours"]
    display_df["Energy"] = actual_df["power"] * time_delta_hours * 0.25

    fig = px.area(
        display_df,
        x="timestamp",
        y=["Energy"],
        title=f"Floor {display_df.floor.unique()}",
    )
    # fig.add_scatter(x=display_df['timestamp'], y=display_df['preds'], mode='lines', name='Predictions (MAE)')
    fig.add_scatter(
        x=display_df["timestamp"],
        y=display_df["forecast_q_50"],
        mode="lines",
        line={"color": "red"},
        name="q = 0.5",
    )
    fig.add_scatter(
        x=display_df["timestamp"],
        y=display_df["forecast_q_5"],
        mode="lines",
        line={"dash": "dot", "color": "green"},
        name="q = 0.05",
    )
    fig.add_scatter(
        x=display_df["timestamp"],
        y=display_df["forecast_q_95"],
        mode="lines",
        line={"dash": "dot", "color": "rgb(178, 102, 255)"},
        name="q = 0.95",
    )

    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Energy (kWh)")
    return fig


if __name__ == "__main__":

    plot_cv: bool = True
    plot_forecast_actual: bool = True

    # get local data from one floor
    from src.data_steps.preprocessor import preprocess_s3_data
    from src.utils.timestamp_type_convertor import (
        convert_str_to_berlin_zone_timestamp_column,
    )
    import os

    floor = 1
    parent_directory = os.path.abspath(os.path.join(os.getcwd(), ".."))
    consumption_df_path = os.path.join(parent_directory, "../data/consumption_data.csv")
    df_consumption = pd.read_csv(consumption_df_path)
    df_consumption = convert_str_to_berlin_zone_timestamp_column(df_consumption)
    df_processed_full = preprocess_s3_data(df_consumption)
    df_processed_floor = df_processed_full[df_processed_full["floor"] == floor]

    if plot_forecast_actual:
        forecast_df_path = os.path.join(parent_directory, "../data/forecast_data.csv")
        forecast_df = pd.read_csv(forecast_df_path)
        forecast_df = convert_str_to_berlin_zone_timestamp_column(
            forecast_df, column_name="timestamp"
        )
        forecast_df_floor = forecast_df[forecast_df["floor"] == floor].reset_index(
            drop=True
        )
        forecast_year_week_list = forecast_df_floor["year_week"].unique()
        actual_df = df_processed_floor[
            df_processed_floor["year_week"].isin(forecast_year_week_list)
        ]
        actual_df = actual_df.sort_values("timestamp").reset_index(drop=True)

        fig = plot_forecast_vs_actual(forecast_df_floor, actual_df)
        fig.show()

    if plot_cv:
        from sktime.forecasting.base import ForecastingHorizon
        from sktime.split import ExpandingWindowSplitter, SlidingWindowSplitter

        first_forecast_week = 31
        df_processed = df_processed_full[
            df_processed_full["weekofyear"] < first_forecast_week
        ]
        df_floor = df_processed[df_processed.floor == floor]

        # choose what to plot
        number_of_weeks_cv = 15
        number_of_weeks_test = 1
        weekofyear_array = df_floor["weekofyear"].unique()[
            -number_of_weeks_cv:-number_of_weeks_test
        ]
        print(weekofyear_array)
        year_week_array = df_floor["year_week"].unique()[
            -number_of_weeks_cv:-number_of_weeks_test
        ]

        forecast_weeks = [1, 2]
        initial_window = 9

        fh = ForecastingHorizon(forecast_weeks, is_relative=True)
        window = "expanding"  # "sliding"
        if window == "expanding":
            cv = ExpandingWindowSplitter(initial_window=initial_window, fh=fh)
        else:
            cv = SlidingWindowSplitter(window_length=initial_window, fh=fh)

        train_windows, test_windows = get_windows(year_week_array, cv)
        fig = plot_windows(pd.Series(year_week_array), train_windows, test_windows)

        plt.show()