from optuna.trial import FrozenTrial


def get_trial_param_dict(trial: FrozenTrial) -> dict[str, str | float | int]:

    # trial_params = trial.params
    user_attrs = trial.user_attrs
    # distributions = {key: str(value) for key, value in trial.distributions.items()}

    trial_dict = {
        "trial_number": trial.number,
        "num_folds": len(trial.user_attrs["train_valid_weeks"]),
        "first_train_weeks_number": len(trial.user_attrs["train_valid_weeks"][0][0]),
        "last_train_weeks_number": len(trial.user_attrs["train_valid_weeks"][-1][0]),
        "trial_state": trial.state,
        "start_time": str(trial.datetime_start),
        "complete_time": str(trial.datetime_complete),
        # **trial_params,              # Log parameters if needed
        **user_attrs,  # Log user attributes (MWIS, coverage, etc.)
    }

    return trial_dict
