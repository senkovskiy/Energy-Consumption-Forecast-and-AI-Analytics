from dotenv import load_dotenv
import os


load_dotenv()


class EnvVars:
    region = os.getenv("REGION")
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_S3_ALLOW_UNSAFE_RENAME = os.getenv("AWS_S3_ALLOW_UNSAFE_RENAME")


class PipelineConfig:
    first_forecast_year_week = "2024-W30"
    number_of_weeks_cv = 15
    number_of_weeks_test = 1
    floors = [1, 3, 4, 5]
    feature_columns = ["time_from_daystart", "weekday", "holiday", "bridge_holiday"]
    categorical_features = ["weekday", "holiday", "bridge_holiday"]
    target_column = "power"


class CrossValidation:
    defaults_cv = {
        "forecast_weeks": [1, 2],
        "train_weeks": 9,
        "cv_step_weeks": 1,
        "moving_window": "expanding_window",
        "min_coverage": 0.8,
        "n_startup_trials": 9,
        "seed": 2,
        "timeout": 1800,
        "n_trials": 30,
    }
    defaults_append = {"prev_holiday_weeks_to_take": 3, "prev_bridge_days_to_take": 2}
    min_metric_percent_improvement = 0.1


class QuantileConfig:
    alpha = 0.05
    quantile_levels = [alpha, 0.5, 1 - alpha]
    quantile_str = str(quantile_levels).replace("[", "").replace("]", "")
    metric_name = "MultiQuantile:alpha=" + quantile_str.replace(" ", "")
    forecast_col_names = [f"forecast_q_{int(alpha*100)}" for alpha in quantile_levels]


class ModelParams:
    # Default model parameters for CatBoostRegressor
    defaults = {
        "n_estimators": 800,
        "learning_rate": 0.05,
        "depth": 7,
        "l2_leaf_reg": 0.0001,
        "model_size_reg": 4.5e-05,
        "random_strength": 0.00075,
        "colsample_bylevel": 0.836,
        "subsample": 0.84,
        "early_stopping_rounds": 400,
    }
    model_name = "catboost"


class MLFlowConfig:
    experiment_name = "first_experiment"
    best_trial_train_run_name = "train_best_trial"


class S3Config:
    default_columns_to_save = [
        "timestamp",
        "weekday",
        "date",
        "hour",
        "weekofyear",
        "year_week",
        "holiday",
        "bridge_holiday",
        "floor",
        "forecast_q_5",
        "forecast_q_50",
        "forecast_q_95",
    ]
    save_locally = True
    local_directory = "data"
    filename = "forecast_data.csv"
