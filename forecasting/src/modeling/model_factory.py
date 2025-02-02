from catboost import CatBoostRegressor


class ModelFactory:
    @staticmethod
    def get_model(
        model_name: str, metric_name: str, params: dict, categorical_features=None
    ):

        if model_name == "catboost":

            return CatBoostRegressor(
                allow_writing_files=False,
                loss_function=metric_name,  # Metric to use in training
                custom_metric=["RMSE", "MAE", "R2"],  # Additional metrics to track
                cat_features=categorical_features,
                sampling_frequency="PerTree",
                one_hot_max_size=7,
                # Use the validation dataset to identify the iteration with the optimal value of the metric specified in --eval_metric (or --loss_function).
                # This option requires a validation dataset to be provided.
                # use_best_model=True,
                verbose=False,
                **params,
            )
        # Add more model
        else:
            raise ValueError(f"Model '{model_name}' is not supported.")

    @staticmethod
    def log_model_to_mlflow(
        model_name: str, model, params, metrics, artifact_path="model"
    ):
        pass
