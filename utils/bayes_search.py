import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold
import os
import pandas as pd
import optuna
from recommenders.ensemble_mlp_regression import Ensemble

torch.set_float32_matmul_precision("medium")

def objective(trial, x_train, y_train, data_type, task="regression"):
    """
    Objective function for Bayesian optimisation using Optuna.

    Args:
        trial: A trial object which suggests hyperparameter values.
        x_train: The training x dataset.
        y_train: The training y dataset.
        data_type: the dataset going through the model.
        task: regression or classification

    Returns:
        The mean validation score over 5-fold cross-validation.
    """
    # Define the hyperparameter search space
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-1)
    hidden_features = trial.suggest_categorical('hidden_features', [
        [16, 16],
        [32, 32],
        [64, 64],
        [128, 128],
        [16, 16, 16],
        [32, 32, 32],
        [64, 64, 64],
        [128, 128, 128],
    ])
    dropout = trial.suggest_uniform('dropout', 0.0, 0.5)
    num_members = trial.suggest_int('num_members', 5, 20)

    embedding = data_type in ["embedding", "embedding_no_attention"]
    attention = data_type == "embedding"

    if attention:
        attention_dropout = trial.suggest_uniform('attention_dropout', 0.0, 0.2)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    val_scores = []

    for train_index, val_index in kf.split(x_train):
        x_tr, x_val = x_train[train_index], x_train[val_index]
        y_tr, y_val = y_train[train_index], y_train[val_index]

        # Instantiate the model with the current set of hyperparameters
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
            attention_dropout=attention_dropout if attention else None
        )

        # Evaluate the model
        model.fit(x_tr, y_tr)

        if task == "regression":
            y_pred, _ = model.predict(x_val)
            val_score = r2_score(y_val, y_pred)
        else:
            y_pred = model.predict(x_val)
            val_score = accuracy_score(y_val, y_pred)

        val_scores.append(val_score)

    mean_val_score = np.mean(val_scores)
    return mean_val_score

def bayesian_search(x_train, y_train, data_type, y_value, task="regression", n_trials=100):
    """
    Perform a Bayesian search for hyperparameter tuning the deep ensemble with 5-fold cross-validation.

    Args:
        x_train: The training x dataset.
        y_train: The training y dataset.
        data_type: the dataset going through the model.
        y_value: additional identifier for the output file.
        task: regression or classification
        n_trials: The number of trials for Bayesian optimisation.

    Returns:
        The best set of hyperparameters found during the search.
    """
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, x_train, y_train, data_type, task), n_trials=n_trials)

    best_params = study.best_params
    best_val_score = study.best_value

    best_params["performance"] = best_val_score

    results = pd.DataFrame([best_params])
    results.to_csv(
        f"{os.path.dirname(os.path.realpath(__file__))}/../regression_scores/bayes/bayesiansearch_{data_type}_{y_value}.csv",
        index=False
    )

    return best_params
