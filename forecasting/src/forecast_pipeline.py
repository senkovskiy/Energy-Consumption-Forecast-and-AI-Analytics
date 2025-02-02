from src.data_steps.cv_test_splitter import cv_test_split
from src.data_steps.forecast_generator import (
    generate_2_weeks_forecast_features_df,
    generate_forecast,
)
from src.data_steps.train_data_adder import add_holiday_weeks, add_bridge_day_weeks
from src.modeling.cv.optuna_study import get_study
from src.modeling.model_factory import ModelFactory
from src.modeling.wis_score import mean_WIS_3_and_coverage
from src.config import (
    EnvVars,
    PipelineConfig,
    CrossValidation,
    ModelParams,
    MLFlowConfig,
    S3Config,
    QuantileConfig,
)
import mlflow
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from catboost import CatBoostRegressor
from optuna.trial import FrozenTrial


def forecast_pipeline_floor(
    df_history: pd.DataFrame,
    floor: int,
    min_metric_percent_improvement: float = CrossValidation.min_metric_percent_improvement,
    cv_params: dict = CrossValidation.defaults_cv,
    first_model: bool = True,
    mlruns_folder: str | None = None,
):
    current_date: date = df_history.date.max()
    current_week: int = df_history[df_history.date == df_history.date.max()][
        "weekofyear"
    ].max()
    previous_date = current_date - timedelta(weeks=1)
    experiment_id = get_or_create_experiment(
        mlruns_folder, MLFlowConfig.experiment_name
    )

    model_name = ModelParams.model_name
    metric_name = QuantileConfig.metric_name
    final_model_version_name = f"final_model_{floor}"
    prev_params = None
    prev_model = None

    if not first_model:
        filter_string = f"""
                            tags.run_date = '{previous_date}' 
                            and tags.model_version = '{final_model_version_name}'
                        """
        previous_run = get_previous_run(experiment_id, filter_string)
        assert previous_run is not None, "Check why the previous_run is None"
        previous_run_id = previous_run.info.run_id
        prev_model_uri = f"runs:/{previous_run_id}/{model_name}_{floor}"
        prev_model = mlflow.catboost.load_model(prev_model_uri)
        assert (
            type(prev_model) == CatBoostRegressor
        ), "Currently ony catboost model is supported"
        prev_params = {
            k: float(previous_run.data.params[k])
            for k, v in ModelParams.defaults.items()
        }

    # 1. Data preparation
    df_floor_cv, df_floor_test, forecast_features_df = prepare_data(df_history, floor)

    # TODO: remove prints
    print(f"Running forecast pipeline for floor = {floor}")
    print(f"Initial model parameters: {prev_params}")
    print("CV weeks: ", df_floor_cv["weekofyear"].unique())
    print("Test weeks: ", df_floor_test["weekofyear"].unique())

    # 2. Cross-validation on CV dataframe => best_trial
    study = get_study(
        train=df_floor_cv,
        feature_columns=PipelineConfig.feature_columns,
        categorical_features=PipelineConfig.categorical_features,
        target_column=PipelineConfig.target_column,
        study_name="cv_study",
        best_params=prev_params,
        metric_name=QuantileConfig.metric_name,
        model_name=model_name,
        **cv_params,
    )

    try:
        best_trial: FrozenTrial = study.best_trial
    except ValueError:
        best_trial: FrozenTrial = min(study.trials, key=lambda trial: trial.values[0])
        print(
            "No feasible trials are completed yet. Check the 'coverage_constraints', probably it is too high."
        )

    # 3. Train the best trial model (study_model) on full CV data
    study_model = ModelFactory.get_model(
        model_name=model_name,
        metric_name=metric_name,
        params=best_trial.params,
        categorical_features=PipelineConfig.categorical_features,
    )
    X_train = df_floor_cv[PipelineConfig.feature_columns]
    y_train = df_floor_cv[PipelineConfig.target_column]
    study_model.fit(X_train, y_train)

    # 4. Test model
    X_test = df_floor_test[PipelineConfig.feature_columns]
    y_test = df_floor_test[PipelineConfig.target_column]
    study_model_predictions = study_model.predict(X_test)

    MWIS_study, _ = mean_WIS_3_and_coverage(
        y_true=y_test,
        lower=study_model_predictions[:, 0],
        middle=study_model_predictions[:, 1],
        upper=study_model_predictions[:, 2],
    )
    MAE_study = np.mean(np.abs(np.array(y_test) - study_model_predictions[:, 1]))

    # 5. Initialize default pyrameters based on the study
    final_run_name = f"study_model_{current_week}_{floor}"
    final_params = best_trial.params
    MWIS_test = MWIS_study
    MAE_test = MAE_study

    # 6. Compare with previous model
    if prev_model:
        final_params = prev_params
        prev_model_predictions = prev_model.predict(X_test)
        MWIS_prev, _ = mean_WIS_3_and_coverage(
            y_true=y_test,
            lower=prev_model_predictions[:, 0],
            middle=prev_model_predictions[:, 1],
            upper=prev_model_predictions[:, 2],
        )

        MAE_prev = np.mean(np.abs(np.array(y_test) - prev_model_predictions[:, 1]))

        print("MWIS_study, MWIS_prev", (MWIS_study, MWIS_prev))

        # rewrite parameters if the previous model is better
        metric_percent_improvement = (MWIS_prev - MWIS_study) / MWIS_prev * 100

        if metric_percent_improvement < min_metric_percent_improvement:
            final_run_name = f"prev_model_{current_week}_{floor}"
            MWIS_test = MWIS_prev
            MAE_test = MAE_prev
        else:
            # since we can perform CV with limited set of parameters, we need to update the final_params
            final_params.update(best_trial.params)

    # 5. Train the final model on all data (cv + test)
    full_train = pd.concat([df_floor_cv, df_floor_test], axis=0).reset_index(drop=True)

    with mlflow.start_run(
        experiment_id=experiment_id, run_name=final_run_name, nested=True
    ):
        assert set(final_params.keys()) == set(
            ModelParams.defaults.keys()
        ), "final_params does not have all keys"
        model = ModelFactory.get_model(
            model_name=model_name,
            metric_name=metric_name,
            params=final_params,
            categorical_features=PipelineConfig.categorical_features,
        )
        model.fit(
            full_train[PipelineConfig.feature_columns],
            full_train[PipelineConfig.target_column],
        )
        mlflow.log_params(final_params)
        metrics = dict()
        metrics["MWIS_test"] = MWIS_test  # metrics.pop(metric_name)
        metrics["MAE_test"] = MAE_test
        mlflow.log_metrics(metrics)
        mlflow.set_tag("run_date", current_date)
        mlflow.set_tag("model_version", final_model_version_name)
        assert (
            model_name == "catboost"
        ), "Currently ony catboost model is supported for mlflow log"
        mlflow.catboost.log_model(model, f"{model_name}_{floor}")

    # 7. Make predictions
    forecast_col_names = QuantileConfig.forecast_col_names
    forecast_df = generate_forecast(
        model,
        df=forecast_features_df,
        forecast_col_names=forecast_col_names,
        feature_columns=PipelineConfig.feature_columns,
    )
    # Convert Power to Energy (kWh)
    forecast_df[forecast_col_names] = forecast_df[forecast_col_names].multiply(
        forecast_df["time_delta_hours"], axis="index"
    )

    return forecast_df


def prepare_data(
    df_history: pd.DataFrame, floor: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Data preparation: filtering, splitting, and generating forecast features."""
    # we filter out the floor here
    df_history_floor = df_history[df_history["floor"] == floor]
    df_floor_cv, df_floor_test, df_floor_history = cv_test_split(
        df_history_floor,
        PipelineConfig.number_of_weeks_cv,
        PipelineConfig.number_of_weeks_test,
    )
    forecast_features_df = generate_2_weeks_forecast_features_df(df_floor_test)

    df_floor_cv = add_holiday_weeks(
        forecast_features_df,
        df_floor_cv,
        df_floor_test,
        df_floor_history,
        CrossValidation.defaults_append["prev_holiday_weeks_to_take"],
    )
    df_floor_cv = add_bridge_day_weeks(
        forecast_features_df,
        df_floor_cv,
        df_floor_test,
        df_floor_history,
        CrossValidation.defaults_append["prev_bridge_days_to_take"],
    )
    return df_floor_cv, df_floor_test, forecast_features_df


def get_previous_run(experiment_id: str, filter_string: str):
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        order_by=["start_time DESC"],
        max_results=1,
    )
    return runs[0] if runs else None


def get_or_create_experiment(mlruns_folder: str | None, experiment_name: str):
    experiment_path = str(mlruns_folder or "") + experiment_name
    experiment = mlflow.get_experiment_by_name(experiment_path)
    return (
        experiment.experiment_id
        if experiment
        else mlflow.create_experiment(experiment_path)
    )


if __name__ == "__main__":
    # TEST example:

    # 1. get first train daffy, define first forecast week
    first_forecast_year_week = PipelineConfig.first_forecast_year_week
    year = int(first_forecast_year_week[:4])
    week = int(first_forecast_year_week[-2:])
    week += 1
    first_date_of_week = datetime(year, 1, 1) + timedelta(weeks=week - 1)
    print("first forecast week:", week)
    print("first forecast day", first_date_of_week.date())

    # 2. (re)define configuration params
    first_model = True  # False - if the first forecast was generated
    cv_params = CrossValidation.defaults_cv
    cv_params["n_trials"] = 2

    # 3. load data
    load_locally: bool = True
    if load_locally:
        from src.utils.timestamp_type_convertor import (
            convert_str_to_berlin_zone_timestamp_column,
        )

        consumption_df = pd.read_csv("../data/consumption_df_full_02-02-2025.csv")
        consumption_df = convert_str_to_berlin_zone_timestamp_column(
            consumption_df, column_name="recordedtimestamp"
        )
        consumption_df = consumption_df.sort_values("recordedtimestamp")
    else:
        from src.data_steps.data_importer import get_s3_data

        storage_config = {
            k: v for k, v in EnvVars.__dict__.items() if not k.startswith("__")
        }
        consumption_df = get_s3_data(storage_config=storage_config)

    # 4. preprocess the data
    from src.data_steps.preprocessor import preprocess_s3_data

    df_processed_full = preprocess_s3_data(consumption_df)
    df_history = df_processed_full[
        df_processed_full["timestamp"].dt.date < first_date_of_week.date()
    ]

    # 5. generate forecast for each floor
    forecast_df_list = []
    for floor in PipelineConfig.floors:
        forecast_df_floor = forecast_pipeline_floor(
            df_history=df_history,
            floor=floor,
            min_metric_percent_improvement=0.1,
            first_model=first_model,
            cv_params=cv_params,
        )
        forecast_df_list.append(forecast_df_floor)

    # 6. Save forecast for all floors
    forecast_df = pd.concat(forecast_df_list)

    print(forecast_df)

    if S3Config.save_locally:
        from src.data_steps.data_saver import save_forecast_locally_as_pd
        from pathlib import Path

        save_forecast_locally_as_pd(
            forecast_df, Path(S3Config.local_directory) / S3Config.filename
        )