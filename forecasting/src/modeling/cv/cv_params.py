from optuna.trial import Trial
from src.config import ModelParams


def get_cv_params(trial: Trial | None = None, best_params: dict = None):

    if trial is None:
        return ModelParams.defaults

    if not best_params:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 1500, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.1, log=True),
            "depth": trial.suggest_int("depth", 2, 12),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 100, log=True),
            "model_size_reg": trial.suggest_float(
                "model_size_reg", 1e-8, 100, log=True
            ),
            "random_strength": trial.suggest_float(
                "random_strength", 1e-8, 100, log=True
            ),  # The amount of randomness to use for scoring splits
            "colsample_bylevel": trial.suggest_float(
                "colsample_bylevel", 0.1, 1
            ),  # Random subspace method. The percentage of features to use at each split selection, when features are selected over again at random.
            "subsample": trial.suggest_float(
                "subsample", 0.1, 1
            ),  # Sample rate for bagging.
            "early_stopping_rounds": trial.suggest_int(
                "early_stopping_rounds", 5, 1000, log=True
            ),  # Sets the overfitting detector type to Iter and stops the training after the specified number of iterations since the iteration with the optimal metric value.
        }

    else:
        return {
            "learning_rate": trial.suggest_float(
                "learning_rate",
                best_params["learning_rate"] * 0.5,
                best_params["learning_rate"] * 1.5,
                log=True,
            ),
            "depth": trial.suggest_int(
                "depth", max(best_params["depth"] - 1, 5), best_params["depth"] + 1
            ),
            "random_strength": trial.suggest_float(
                "random_strength",
                best_params["random_strength"] * 0.8,
                best_params["random_strength"] * 1.2,
                log=True,
            ),
            "colsample_bylevel": trial.suggest_float(
                "colsample_bylevel",
                best_params["colsample_bylevel"] * 0.6,
                min(best_params["colsample_bylevel"] * 1.4, 1),
            ),
            "early_stopping_rounds": trial.suggest_int(
                "early_stopping_rounds",
                max(best_params["early_stopping_rounds"] - 100, 10),
                best_params["early_stopping_rounds"] + 100,
                log=True,
            ),
            "n_estimators": trial.suggest_int(
                "n_estimators",
                max(best_params["n_estimators"] - 50, 50),
                best_params["n_estimators"] + 50,
                log=True,
            ),
            "l2_leaf_reg": best_params["l2_leaf_reg"],
            "model_size_reg": best_params["model_size_reg"],
            "subsample": best_params["subsample"],
        }


""" 
# This we do not tune if the model is not the first one:

"l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 
                                    best_params["l2_leaf_reg"] * 0.8, 
                                    best_params["l2_leaf_reg"] * 1.2,
                                    log=True),
"model_size_reg": trial.suggest_float("model_size_reg", 
                                       best_params["model_size_reg"] * 0.8, 
                                       best_params["model_size_reg"] * 1.2,
                                       log=True),
"subsample": trial.suggest_float("subsample", 
                                        best_params["subsample"] * 0.8, 
                                        min(best_params["subsample"] * 1.2, 1)),
"""
