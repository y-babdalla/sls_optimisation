import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical


def bayesian_optimization_cv(
    X,
    y,
    n_iter=2,
    cv=5,
    n_jobs=5,
    scoring="r2",
    title="Bayesian Optimization Results",
    models=None,
    csv=True,
    plot_results=False,
):
    """
    Function to perform Bayesian optimization for hyperparameter tuning using 5-fold cross-validation
    :param X: numpy array of features
    :param y: numpy array of target values
    :param n_iter: number of iterations for Bayesian optimization, default is 100
    :param cv: number of folds in cross-validation, default is 5
    :param n_jobs: number of CPUs used, default is 5
    :param scoring: scoring metric to be used for optimization, default is 'r2'
    :param title: title of the plots and saved files
    :param models: a list of tuples (name, model, param_space), default includes XGB, RF, KNN, SVM
    :param csv: boolean, True if csv to be saved, False if not, default is True
    :param plot_results: boolean, True if plot results, false if not, default is True
    :return: Dictionary of best models and their scores
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
    results = {}

    for name, model, param_space in tqdm(models, desc="Optimizing models"):
        bayes_search = BayesSearchCV(
            model,
            param_space,
            n_iter=n_iter,
            cv=cv_split,
            n_jobs=n_jobs,
            scoring=scoring,
            random_state=42,
            verbose=0,
        )

        bayes_search.fit(X, y)

        best_score = bayes_search.best_score_
        best_params = bayes_search.best_params_
        best_model = bayes_search.best_estimator_

        results[name] = {
            "best_score": best_score,
            "best_params": best_params,
            "best_model": best_model,
        }

        if csv:
            pd.DataFrame(bayes_search.cv_results_).to_csv(
                f"results/{title}_{name}_cv_results.csv"
            )

    if plot_results:
        plot_optimization_results(results, title)

    return results


def plot_optimization_results(results, title):
    """
    Function to plot optimization results
    :param results: Dictionary of results from Bayesian optimization
    :param title: Title of the graph
    """
    scores = [results[model]["best_score"] for model in results]
    models = list(results.keys())

    plt.figure(figsize=(10, 6))
    sns.barplot(x=models, y=scores)
    plt.title(f"{title} - Best Scores")
    plt.ylabel("Mean Squared Error")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"plots/{title}_best_scores.png")
    plt.show()
