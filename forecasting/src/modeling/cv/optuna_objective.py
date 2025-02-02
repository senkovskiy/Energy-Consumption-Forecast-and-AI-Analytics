from sktime.split import ExpandingWindowSplitter, SlidingWindowSplitter
from sktime.forecasting.base import ForecastingHorizon
from optuna.trial import Trial
import numpy as np
import pandas as pd
from src.modeling.wis_score import mean_WIS_3_and_coverage
from src.modeling.cv.cv_params import get_cv_params
from src.modeling.model_factory import ModelFactory


def get_optuna_objective(
    trial: Trial,
    train: pd.DataFrame,
    feature_columns: list[str],
    categorical_features: list[str] | None,
    target_column: str,
    forecast_weeks: list[int],
    cv_step_weeks: int,
    train_weeks: int,
    moving_window: str,
    metric_name: str,
    best_params: dict | None,
    model_name: str,
) -> float:
    moving_windows = ["expanding_window", "sliding_window"]
    train = train.sort_values("timestamp").reset_index(drop=True)

    assert (
        moving_window in moving_windows
    ), f"'{moving_window}' is not in {moving_windows}"
    assert not train.isna().any().any(), "train contains NaN values!"

    if moving_window == "sliding_window":
        splitter = SlidingWindowSplitter(
            fh=ForecastingHorizon(forecast_weeks, is_relative=True),
            window_length=train_weeks,
            step_length=cv_step_weeks,
        )
    else:
        splitter = ExpandingWindowSplitter(
            fh=ForecastingHorizon(forecast_weeks, is_relative=True),
            initial_window=train_weeks,
            step_length=cv_step_weeks,
        )

    params = get_cv_params(trial, best_params)

    # Out-of-fold predictions and true values lists
    oof_preds = []
    y_true = []

    # lists with metrics and other information to log
    q_score_list = []
    mae_list = []
    train_valid_weeks_list = []

    # All week-years in the train
    year_week_array = train["year_week"].unique()

    # Loop over the folds created from the train
    for fold_idx, (train_weeks_idx, valid_weeks_idx) in enumerate(
        splitter.split(year_week_array)
    ):

        # Unique weeks in the fold
        train_weeks_unique = year_week_array[train_weeks_idx].tolist()
        valid_weeks_unique = year_week_array[valid_weeks_idx].tolist()

        # Create train and validation data for the fold
        train_cv = train[train["year_week"].isin(train_weeks_unique)]
        val_cv = train[train["year_week"].isin(valid_weeks_unique)]

        x_train_cv = train_cv[feature_columns]
        y_train_cv = train_cv[target_column]
        x_val_cv = val_cv[feature_columns]
        y_val_cv = val_cv[target_column]

        # Create the model, fit, predict for the fold
        model_fold = ModelFactory.get_model(
            model_name, metric_name, params, categorical_features
        )

        # Optional: additional fit parameters, depending on the model
        fit_params = {}
        if model_name == "catboost":
            # add params to use the validation dataset to identify the optimal iteration
            # (remove trees after the iteration with the best validation score)
            fit_params["use_best_model"] = True
            fit_params["eval_set"] = (x_val_cv, y_val_cv)

        model_fold.fit(x_train_cv, y_train_cv, **fit_params)

        y_pred = model_fold.predict(x_val_cv)

        assert (
            len(y_pred.shape) == 2 and y_pred.shape[1] == 3
        ), """The model predict should provide 2-dimensional output with the second dimension == 3, 
        according to the MWIS metric"""

        # Lists of Out-of-fold predictions and true values
        oof_preds.append(y_pred)
        y_true.append(y_val_cv)

        # Log metrics and other information. TODO : it should be model-independent in future
        best_scores = model_fold.get_best_score()["validation"]
        mae_list.append(best_scores["MAE"])
        q_score = best_scores[metric_name]
        q_score_list.append(q_score)
        train_valid_weeks: tuple = (
            year_week_array[train_weeks_idx].tolist(),
            year_week_array[valid_weeks_idx].tolist(),
        )
        train_valid_weeks_list.append(train_valid_weeks)

    # Concat the predicted and true values
    oof_preds = np.concatenate(oof_preds, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    # Final metric to evaluate (coverage - how many validation data fall within the confidence interval)
    MWIS, coverage = mean_WIS_3_and_coverage(
        y_true=y_true,
        lower=oof_preds[:, 0],
        middle=oof_preds[:, 1],
        upper=oof_preds[:, 2],
    )

    # Store the trial results
    trial.set_user_attr("coverage", coverage)
    trial.set_user_attr("q_score_list", q_score_list)
    trial.set_user_attr("mae_list", mae_list)
    trial.set_user_attr("MWIS", MWIS)
    trial.set_user_attr("train_valid_weeks", train_valid_weeks_list)

    return MWIS
