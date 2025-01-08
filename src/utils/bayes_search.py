"""Perform a Bayesian hyperparameter search for an ensemble-based MLP.

This module leverages Optuna to optimise hyperparameters via k-fold cross-validation
for either regression or classification tasks using an ensemble MLP model.
"""

import os

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold

from src.recommenders.ensemble_mlp import Ensemble

torch.set_float32_matmul_precision("medium")


def objective(
    trial: optuna.trial.Trial,
    x_train: np.ndarray,
    y_train: np.ndarray,
    data_type: str,
    task: str = "regression",
) -> float:
    """Objective function for Bayesian optimisation using Optuna.

    This function draws hyperparameters from Optuna's search space
    and trains a model on multiple folds of the provided dataset.
    For regression, the R² score is used; for classification, the
    accuracy is measured.

    Args:
        trial: Optuna Trial object which provides hyperparameter suggestions.
        x_train: Training feature set of shape (N, D).
        y_train: Corresponding training labels of shape (N,).
        data_type: Indicates dataset transformation type ('embedding', etc.).
        task: Specifies the learning task: 'regression' or 'classification'.

    Returns:
        The mean validation metric score (R² or accuracy) over 5 folds.
    """
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-1)
    hidden_features = trial.suggest_categorical(
        "hidden_features",
        [
            [16, 16],
            [32, 32],
            [64, 64],
            [128, 128],
            [16, 16, 16],
            [32, 32, 32],
            [64, 64, 64],
            [128, 128, 128],
        ],
    )
    dropout = trial.suggest_uniform("dropout", 0.0, 0.5)
    num_members = trial.suggest_int("num_members", 5, 20)

    embedding = data_type in ["embedding", "embedding_no_attention"]
    attention = data_type == "embedding"
    attention_dropout = 0.0
    if attention:
        attention_dropout = trial.suggest_uniform("attention_dropout", 0.0, 0.2)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    val_scores: list[float] = []

    for train_index, val_index in kf.split(x_train):
        x_tr, x_val = x_train[train_index], x_train[val_index]
        y_tr, y_val = y_train[train_index], y_train[val_index]

        model = Ensemble(
            in_features=x_train.shape[1] if not embedding else 32,
            out_features=1,
            bias=True,
            norm="batch",
            embedding=embedding,
            embedding_num_items=77,
            attention=attention,
            attention_inner_dim=64,
            attention_num_heads=4,
            task=task,
            epochs=100,
            batch_size=32,
            seed=42,
            gpu=0,
            logger=True,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            hidden_features=hidden_features,
            dropout=dropout,
            num_members=num_members,
            attention_dropout=attention_dropout if attention else None,
        )
        model.fit(x_tr, y_tr)

        if task == "regression":
            predictions, _ = model.predict(x_val)
            val_score = r2_score(y_val, predictions)
        else:
            predictions = model.predict(x_val)
            val_score = accuracy_score(y_val, predictions)

        val_scores.append(val_score)

    return float(np.mean(val_scores))


def bayesian_search(
    x_train: np.ndarray,
    y_train: np.ndarray,
    data_type: str,
    y_value: str,
    task: str = "regression",
    n_trials: int = 100,
) -> dict[str, float]:
    """Perform a Bayesian hyperparameter search for the ensemble-based MLP model.

    This function creates an Optuna study, defines an objective for hyperparameter
    tuning, and executes the search with a specified number of trials. The best
    hyperparameters and their performance are saved as a CSV file.

    Args:
        x_train: Training feature set of shape (N, D).
        y_train: Corresponding training labels of shape (N,).
        data_type: Indicates dataset transformation type ('embedding', etc.).
        y_value: Additional identifier for the output CSV file name.
        task: Specifies the learning task: 'regression' or 'classification'.
        n_trials: Number of trials to run for the Bayesian optimisation.

    Returns:
        Dictionary containing the best hyperparameters and the associated performance.
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, x_train, y_train, data_type, task), n_trials=n_trials
    )

    best_params = study.best_params
    best_val_score = study.best_value
    best_params["performance"] = best_val_score

    results = pd.DataFrame([best_params])
    csv_path = (
        f"{os.path.dirname(os.path.realpath(__file__))}"
        f"/../regression_scores/bayes/bayesiansearch_{data_type}_{y_value}.csv"
    )
    results.to_csv(csv_path, index=False)

    return best_params
