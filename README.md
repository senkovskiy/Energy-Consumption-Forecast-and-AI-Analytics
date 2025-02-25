# Energy Consumption Forecasting Pipeline

## Overview
This project is ... real-world forecasting pipeline designed for monitoring and analyzing building energy consumption.

The goal is to demonstrate advanced forecasting approach in action. Here we have the following features:
- **Catboost**  handle numerical and categorical features 
- ****
- **Optuna** to ---
- **Forecasting horizont** - generated on the weekly steps with `skime`
- **Time serices cross validation** --
- **MLflow Tracking**: Logs experiments, models, and results for reproducibility.

??? The energy consumption data is stored in `data/consumption_data.scv`.

forecasts energy consumption for a building with four floors, using separate ML models for each floor. The pipeline runs weekly, generating a two-week forecast for each floor. The system includes cross-validation, hyperparameter tuning with Optuna, model tracking with MLflow, and model evaluation using the Mean Weighted Interval Score (MWIS).

## Features
- **Time Series Cross-Validation**: Implements time-based cross-validation for robust model evaluation.
- **MLflow Tracking**: Logs experiments, models, and results for reproducibility.
- **Model Comparison & Selection**: Automatically selects the best-performing model based on MWIS.

## Structure
```
forecasting/
│── pyproject.toml          # Python project configuration
│── schedule_runner.ipynb   # Notebook for scheduling tasks in Databricks
│── testing_notebook.ipynb  # Notebook for testing pipeline components
│
├── data_analysis/
│   ├── first_visualization.ipynb  # Initial data analysis and visualization
│
├── notebooks/
│   ├── run_forecast_example.ipynb  # Example usage of forecasting pipeline
│
├── src/
│   ├── config.py          # Configuration parameters
│   ├── forecast_pipeline.py  # Main forecasting pipeline
│   ├── local_forecast_runner.py  # Local execution script
│   ├── utils/
│   │   ├── plot_data.py  # Visualization utilities
│   │   ├── timestamp_type_convertor.py  # Timestamp handling
│   ├── modeling/
│   │   ├── wis_score.py  # Scoring metric calculation
│   │   ├── model_factory.py  # Model selection and training
│   │   ├── cv/
│   │   │   ├── optuna_study.py  # Hyperparameter tuning with Optuna
│   │   │   ├── optuna_objective.py  # Optimization objective function
│   │   │   ├── trial_param_dict.py  # Trial parameter dictionary
│   │   │   ├── cv_params.py  # Cross-validation parameters
│   ├── data_steps/
│   │   ├── cv_test_splitter.py  # Splits data for cross-validation and testing
│   │   ├── train_data_adder.py  # Adds training data
│   │   ├── preprocessor.py  # Data preprocessing steps
│   │   ├── data_saver.py  # Saves processed data
│   │   ├── forecast_generator.py  # Generates forecasts
│   │   ├── data_importer.py  # Imports raw data
│
├── mlruns/  # MLflow tracking directory
```

## Prerequisites
- **Python Version**: Ensure you have Python 3.11 installed.
- **Poetry**: Dependency management is handled with Poetry.
  
  To install dependencies, run:
  ```bash
  poetry install
  ```
  
  To activate the virtual environment, use:
  ```bash
  poetry shell
  ```

## Pipeline Workflow

### 1. **Cross-Validation & Hyperparameter Tuning**
- Runs every Monday to tune hyperparameters using Optuna.
- Each floor has a separate model trained and evaluated.
- Uses MWIS to compare models.
- Tracked with MLflow.

### 2. **Model Selection & Forecasting**
- If it's the first model (first_model=True), it is used directly.
- Otherwise, the new model is compared with the previous one on a test week (if available).
- The better model is selected.
- The selected model is trained on combined CV + test data.
- Forecasts for the next two weeks are generated.

### 3. **Generating a Forecast**
To manually trigger the forecast generation, run:
```bash
poetry run python src/forecast_pipeline.py
```
This will generate and store forecasts for all floors.

## Model Tracking with MLflow
- ML models are logged in `mlruns/`.
- To view the MLflow UI, run:
  ```bash
  poetry run mlflow ui
  ```
  Then open `http://localhost:5000` in a browser.

## Metrics
- **Mean Weighted Interval Score (MWIS)** is used to compare models.
- MWIS is calculated in `wis_score.py`.

## Next Steps
- Improve feature engineering.
- Experiment with additional ML algorithms.
- Automate deployment of forecast results.


