"""Module for training and evaluating regression models with cross-validation.

This module provides functions to:
- Retrieve the best hyperparameters from previous CSV logs (`get_best_params`).
- Fit and evaluate multiple regression models using cross-validation (`make_predictions`).
- Plot the distribution of cross-validation metrics via boxplots (`plot_data`).

It stores and retrieves model parameters from CSV files created by a Bayesian
optimisation workflow, then uses them to instantiate models. After training, it
saves the best models to disk as pickle files, logs cross-validation metrics,
and optionally plots the results.
"""

import ast
import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, LeaveOneOut, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from tqdm import tqdm
from xgboost import XGBRegressor

plt.style.use(["science", "no-latex"])


def get_best_params(title: str, model_name: str) -> dict[str, object]:
    """Retrieve the best hyperparameters for a given model from a CSV file.

    Looks for a CSV file named 'results/{title}_{model_name}_cv_results.csv'
    generated by a hyperparameter search. It extracts the row with
    'rank_test_score' == 1 and parses the 'params' column to produce a dict.

    Args:
        title: Identifier or prefix for the results file.
        model_name: Name of the model (e.g. 'XGB', 'RF').

    Returns:
        Dictionary of parameters for the model. If the file or parameters cannot
        be found/parsed, returns an empty dict.
    """
    csv_path = f"results/{title}_{model_name}_cv_results.csv"
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file for {model_name} not found. Using default parameters.")
        return {}

    df = pd.read_csv(csv_path)
    if "rank_test_score" not in df.columns or "params" not in df.columns:
        print(
            f"Warning: CSV file for {model_name} does not have expected columns. Using default parameters."  # noqa: E501
        )
        return {}

    best_rows = df[df["rank_test_score"] == 1]
    if best_rows.empty:
        print(f"Warning: No best row found for {model_name}. Using default parameters.")
        return {}

    best_row = best_rows.iloc[0]
    params_str = best_row["params"]

    params_str = re.sub(r"OrderedDict\((.*)\)", r"\1", params_str)

    try:
        parsed_params = ast.literal_eval(params_str)
        # Convert e.g. [("max_depth", 5), ...] to {"max_depth": 5, ...}
        if isinstance(parsed_params, list):
            parsed_params = dict(parsed_params)
        parsed_params = {str(k): v for k, v in parsed_params.items()}
    except (ValueError, SyntaxError):
        print(f"Warning: Could not parse parameters for {model_name}. Using default parameters.")
        return {}

    return parsed_params


def make_predictions(
    x_data: np.ndarray,
    y_data: np.ndarray,
    cv: int | None = 5,
    n_jobs: int = 1,
    scoring: list[str] | None = None,
    title: str = "Initial Scores",
    models: list[tuple[str, object]] | None = None,
    csv: bool = True,
    plot_results: bool = True,
) -> None:
    """Fit and evaluate multiple regression models using cross-validation.

    Models will use previously saved-best parameters if CSV files exist
    (retrieved via `get_best_params`). If not found, default parameters are used.

    Args:
        x_data: Feature matrix of shape (num_samples, num_features).
        y_data: Target array of shape (num_samples,).
        cv: Number of CV folds or None for Leave-One-Out. Defaults to 5.
        n_jobs: Number of parallel jobs for cross_validate. Defaults to 1.
        scoring: List of scoring metrics. If None, defaults to a standard set.
        title: String prefix for output file naming and figure titles.
        models: List of (model_name, model_instance). If None, uses XGB, RF, KNN, SVM.
        csv: If True, save cross-validation results to CSV.
        plot_results: If True, also generate boxplots for each metric.

    Returns:
        None. But writes scores to CSV, saves fitted models to disk, and
        optionally plots the results.
    """
    if scoring is None:
        scoring = [
            "r2",
            "neg_root_mean_squared_error",
            "neg_mean_absolute_error",
            "neg_mean_squared_error",
        ]

    cv_split = LeaveOneOut() if cv is None else KFold(n_splits=cv, shuffle=True, random_state=42)

    if models is None:
        models = [
            ("XGB", XGBRegressor(random_state=42, **get_best_params(title, "XGB"))),
            ("RF", RandomForestRegressor(random_state=42, **get_best_params(title, "RF"))),
            ("KNN", KNeighborsRegressor(**get_best_params(title, "KNN"))),
            ("SVM", SVR(**get_best_params(title, "SVM"))),
        ]

    r2_scores, rmse_scores, mae_scores, mse_scores = [], [], [], []

    for name, model_instance in tqdm(models, desc="Running ML models"):
        model_instance.fit(x_data, y_data)
        os.makedirs("new_models", exist_ok=True)
        with open(f"new_models/best_{title}_{name}.pkl", "wb") as file_out:
            pickle.dump(model_instance, file_out)

        cv_results = cross_validate(
            estimator=model_instance,
            X=x_data,
            y=y_data,
            scoring=scoring,
            cv=cv_split,
            n_jobs=n_jobs,
        )
        df_cv_results = pd.DataFrame(cv_results)

        r2_scores.append(df_cv_results["test_r2"].to_numpy())
        rmse_scores.append(-df_cv_results["test_neg_root_mean_squared_error"].to_numpy())
        mae_scores.append(-df_cv_results["test_neg_mean_absolute_error"].to_numpy())
        mse_scores.append(-df_cv_results["test_neg_mean_squared_error"].to_numpy())

        if csv:
            os.makedirs("results", exist_ok=True)
            df_cv_results.to_csv(f"results/{title}_tuned_cv_scores_{name}.csv", index=False)

    if plot_results:
        r2_df = pd.DataFrame(np.array(r2_scores).T, columns=[m[0] for m in models])
        rmse_df = pd.DataFrame(np.array(rmse_scores).T, columns=[m[0] for m in models])
        mae_df = pd.DataFrame(np.array(mae_scores).T, columns=[m[0] for m in models])
        mse_df = pd.DataFrame(np.array(mse_scores).T, columns=[m[0] for m in models])

        score_frames = [("R2", r2_df), ("RMSE", rmse_df), ("MAE", mae_df), ("MSE", mse_df)]
        os.makedirs("results/cross_val", exist_ok=True)
        os.makedirs("plots", exist_ok=True)
        for score_name, df_scores in score_frames:
            df_scores.to_excel(f"results/cross_val/{title}_{score_name}.xlsx", index=False)
            plot_data(df_scores, title=title, score=score_name)


def plot_data(df_scores: pd.DataFrame, title: str, score: str) -> None:
    """Plot a boxplot of cross-validation results for a given metric.

    Args:
        df_scores: DataFrame with shape (num_folds, num_models).
        title: Prefix used for the output plot file name.
        score: Metric name ('R2', 'RMSE', 'MAE', 'MSE', etc.).
    """
    font_size = 20
    plt.rcParams.update({"font.size": font_size})
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    stacked_df = df_scores.stack().reset_index()  # noqa: PD013
    stacked_df.columns = ["Fold", "Model", score]

    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Model", y=score, data=stacked_df, palette="Set2")
    plot_filename = f"plots/{title}-{score}.png"
    plt.savefig(plot_filename)
    plt.show()