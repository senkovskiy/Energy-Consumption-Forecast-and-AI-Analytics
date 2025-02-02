import warnings
import pandas as pd
import optuna
from optuna.trial import Trial
from optuna.study.study import Study
from src.modeling.cv.optuna_objective import get_optuna_objective


def coverage_constraints(trial: Trial, min_coverage: float = 0.8) -> tuple[float,]:
    constraints = (min_coverage - trial.user_attrs["coverage"],)
    return constraints


def get_study(
    train: pd.DataFrame,
    feature_columns: list[str],
    categorical_features: list[str] | None,
    target_column: str,
    forecast_weeks: list[int],
    train_weeks: int,
    cv_step_weeks: int,
    moving_window: str,
    n_startup_trials: int,
    seed: int,
    min_coverage: float,
    n_trials: int,
    timeout: int,
    study_name: str,
    best_params: dict | None,
    metric_name: str,
    model_name: str,
) -> Study:

    assert (
        0 < min_coverage < 1
    ), f"min_coverage should be between 0 and 1, but given: {min_coverage}"

    with warnings.catch_warnings():
        warnings.simplefilter(
            action="ignore", category=optuna.exceptions.ExperimentalWarning
        )
        warnings.simplefilter(action="ignore", category=FutureWarning)

        study: Study = optuna.create_study(
            directions=["minimize"],
            pruner=optuna.pruners.MedianPruner(),
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=n_startup_trials,
                seed=seed,
                constraints_func=lambda trial: coverage_constraints(
                    trial, min_coverage=min_coverage
                ),
            ),
            study_name=study_name,
            load_if_exists=False,
        )

    # run optuna for a maximum of 50 trials and 1hr (see config)
    study.optimize(
        lambda trial: get_optuna_objective(
            trial=trial,
            train=train,
            feature_columns=feature_columns,
            categorical_features=categorical_features,
            target_column=target_column,
            forecast_weeks=forecast_weeks,
            cv_step_weeks=cv_step_weeks,
            train_weeks=train_weeks,
            moving_window=moving_window,
            best_params=best_params,
            metric_name=metric_name,
            model_name=model_name,
        ),
        n_trials=n_trials,
        timeout=timeout,
    )

    return study
