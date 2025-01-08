"""Bayesian hyperparameter optimisation with cross-validation for various regressors.

This script defines two functions:
1. `bayesian_optimization_cv`: Runs Bayesian optimisation using `skopt` for a
   set of regressors and logs their best parameters and scores.
2. `plot_optimization_results`: Plots the best cross-validation scores from
   the optimiser results as a bar plot.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
from tqdm import tqdm
from xgboost import XGBRegressor


def bayesian_optimisation_cv(
    x: np.ndarray,
    y: np.ndarray,
    n_iter: int = 2,
    cv: int = 5,
    n_jobs: int = 5,
    scoring: str = "r2",
    title: str = "Bayesian Optimization Results",
    models: list[tuple[str, object, dict[str, object]]] | None = None,
    csv: bool = True,
    plot_results: bool = False,
) -> dict[str, dict[str, object]]:
    """Perform Bayesian hyperparameter optimisation using cross-validation.

    Args:
        x (np.ndarray): Array of shape (n_samples, n_features) with training features.
        y (np.ndarray): Array of shape (n_samples,) with training targets.
        n_iter (int): Number of iterations for the Bayesian optimisation.
        cv (int): Number of cross-validation folds.
        n_jobs (int): Number of CPU cores to use during model fitting.
        scoring (str): Scoring metric for model evaluation.
        title (str): Title for the output CSVs and plots.
        models (list[tuple[str, object, dict[str, object]]] | None):
            Optional list of (name, estimator, param_space) for each model.
        csv (bool): If True, save cross-validation results to CSV.
        plot_results (bool): If True, generate a bar plot of the best scores.

    Returns:
        dict[str, dict[str, object]]:
            A dictionary keyed by model name with values containing best score,
            best parameters, and the best-fitted model.
    """
    if models is None:
        models = [
            (
                "XGB",
                XGBRegressor(random_state=42),
                {
                    "max_depth": Integer(1, 10),
                    "learning_rate": Real(0.01, 1.0, "log-uniform"),
                    "n_estimators": Integer(10, 1000),
                    "min_child_weight": Integer(1, 10),
                    "subsample": Real(0.2, 1.0),
                    "colsample_bytree": Real(0.2, 1.0),
                },
            ),
            (
                "RF",
                RandomForestRegressor(random_state=42),
                {
                    "n_estimators": Integer(10, 1000),
                    "max_depth": Integer(1, 20),
                    "min_samples_split": Integer(2, 20),
                    "min_samples_leaf": Integer(1, 20),
                },
            ),
            (
                "KNN",
                KNeighborsRegressor(),
                {
                    "n_neighbors": Integer(1, 20),
                    "weights": Categorical(["uniform", "distance"]),
                    "p": Integer(1, 2),
                },
            ),
            (
                "SVM",
                SVR(),
                {
                    "C": Real(0.1, 100, "log-uniform"),
                    "kernel": Categorical(["linear", "rbf", "poly"]),
                    "gamma": Real(0.0001, 1.0, "log-uniform"),
                    "epsilon": Real(0.01, 1.0, "log-uniform"),
                },
            ),
        ]

    cv_split = KFold(n_splits=cv, shuffle=True, random_state=42)
    results: dict[str, dict[str, object]] = {}

    for name, model, param_space in tqdm(models, desc="Optimizing models"):
        bayes_search = BayesSearchCV(
            estimator=model,
            search_spaces=param_space,
            n_iter=n_iter,
            cv=cv_split,
            n_jobs=n_jobs,
            scoring=scoring,
            random_state=42,
            verbose=0,
        )
        bayes_search.fit(x, y)

        best_score = bayes_search.best_score_
        best_params = bayes_search.best_params_
        best_model = bayes_search.best_estimator_

        results[name] = {
            "best_score": best_score,
            "best_params": best_params,
            "best_model": best_model,
        }

        if csv:
            os.makedirs("results", exist_ok=True)
            cv_results_df = pd.DataFrame(bayes_search.cv_results_)
            cv_results_df.to_csv(f"results/{title}_{name}_cv_results.csv", index=False)

    if plot_results:
        plot_optimisation_results(results, title)

    return results


def plot_optimisation_results(results: dict[str, dict[str, object]], title: str) -> None:
    """Plot a bar chart of best scores from the Bayesian optimisation.

    Args:
        results (dict[str, dict[str, object]]): Results dictionary from
            bayesian_optimization_cv.
        title (str): Title prefix for the plot file.
    """
    os.makedirs("plots", exist_ok=True)
    model_names = list(results.keys())
    scores = [results[m]["best_score"] for m in model_names]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=model_names, y=scores)
    plt.title(f"{title} - Best Scores")
    plt.ylabel("Score (higher is better)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"plots/{title}_best_scores.png")
    plt.show()
