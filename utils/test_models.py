"""
Run models using cross-validation
"""
import pickle
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
from tqdm import tqdm
import os
import ast

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, cross_validate, LeaveOneOut, KFold
from xgboost import XGBRegressor

plt.style.use(["science", "no-latex"])


def get_best_params(title, model_name):
    """
    Function to get best parameters for a model from its CSV file
    """
    csv_path = f"results/{title}_{model_name}_cv_results.csv"
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file for {model_name} not found. Using default parameters.")
        return {}

    df = pd.read_csv(csv_path)
    best_row = df[df['rank_test_score'] == 1].iloc[0]
    params_str = best_row['params']

    # Remove OrderedDict and convert to regular dict string
    params_str = re.sub(r'OrderedDict\((.*)\)', r'\1', params_str)

    # Use ast.literal_eval to safely evaluate the string
    try:
        params = ast.literal_eval(params_str)

        # If params is a list of tuples, convert it to a dictionary
        if isinstance(params, list):
            params = dict(params)

        # Ensure all keys are strings
        params = {str(k): v for k, v in params.items()}

    except ValueError:
        print(f"Warning: Could not parse parameters for {model_name}. Using default parameters.")
        return {}

    return params


def make_predictions(
        X,
        y,
        cv=5,
        n_jobs=1,
        scoring=None,
        title="Initial Scores",
        models=None,
        csv=True,
        plot_results=True,
):
    if scoring is None:
        scoring = ["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error", "neg_mean_squared_error"]

    if cv is None:
        cv = LeaveOneOut()
    # else:
    #     cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Instantiate models
    if models is None:
        models = [
            ("XGB", XGBRegressor(random_state=42, **get_best_params(title, "XGB"))),
            ("RF", RandomForestRegressor(random_state=42, **get_best_params(title, "RF"))),
            ("KNN", KNeighborsRegressor(**get_best_params(title, "KNN"))),
            ("SVM", SVR(**get_best_params(title, "SVM"))),
        ]


    cv = KFold(n_splits=cv, shuffle=True, random_state=42)
    # Run models for regression
    r2_scores = []
    rmse = []
    mae = []
    mse = []
    for model in tqdm(models, desc="Running ML models"):
        ml_model = model[1]
        ml_model.fit(X,y)
        with open(f'new_models/best_{title}_{model[0]}.pkl', 'wb') as file:
            pickle.dump(model, file)
        ml_scores = cross_validate(
            model[1], X, y, scoring=scoring, cv=cv, n_jobs=n_jobs
        )

        df_results = pd.DataFrame(ml_scores)

        r2_scores.append(ml_scores["test_r2"])
        rmse.append(-ml_scores["test_neg_root_mean_squared_error"])
        mae.append(-ml_scores["test_neg_mean_absolute_error"])
        mse.append(-ml_scores["test_neg_mean_squared_error"])

        if csv:
            df_results.to_csv(f"results/{title}_tuned_cv_scores.csv")

    # Call plotting function
    if plot_results is True:
        r2_scores = np.stack(r2_scores)
        r2_scores = pd.DataFrame(
            r2_scores.T,
            columns=[model[0] for model in models],
        )
        rmse = np.stack(rmse)
        rmse = pd.DataFrame(
            rmse.T,
            columns=[model[0] for model in models],
        )
        mae = np.stack(mae)
        mae = pd.DataFrame(
            mae.T,
            columns=[model[0] for model in models],
        )
        mse = np.stack(mse)
        mse = pd.DataFrame(
            mse.T,
            columns=[model[0] for model in models],
        )
        scores = [
            ("R2", r2_scores),
            ("RMSE", rmse),
            ("MAE", mae),
            ("MSE", mse),
        ]


        for score in scores:
            plot_data(score[1], title=title, score=score[0])
            pd.DataFrame(score[1]).to_excel(f"results/cross_val/{title}_{score[0]}.xlsx")


# The plot_data function remains unchanged
def plot_data(df, title, score):
    """
    Function to plot box plots
    :param df: DataFrame to be plotted
    :param title: Title of the graph
    :param score: Machine Learning scoring used
    :return: None
    """
    font = 20
    plt.rcParams.update({"font.size": font})
    plt.xticks(fontsize=font)
    plt.yticks(fontsize=font)
    sns.set(style="whitegrid")
    df = pd.DataFrame(df.stack(), index=None)
    df = df.reset_index().drop(["level_0"], axis=1)
    df.rename(columns={0: f"{score}", "level_1": "Model"}, inplace=True)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Model", y=f"{score}", data=df, palette="Set2")
    plt.savefig(f"plots/{title}-{score}.png")
    plt.show()
