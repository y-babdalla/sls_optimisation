from typing import Callable, Dict, List, Tuple, Union, Optional, Any

import numpy as np
import torch

import sys
import csv
import argparse
import random

sys.path.append("./..")
sys.path.append("./../..")

from recommenders.ensemble_mlp import Ensemble, EnsembleMLP
from tqdm import tqdm
from recommenders.proportion.utils import binary_entropy

good_materials = [58, 61, 63, 85, 88, 94, 95, 107]


class ProportionRecommenderRS:
    """This class implements the Random Search algorithm in order to optimize the proportions.

    It adds random noise to the proportions to the best point found so far at each iteration.
    If the new point is better than the best point found so far, it becomes the new best point.

    Args:
        objective_function (Callable): Function to optimize.
        initial_x (np.ndarray): Initial point for the optimization.
        num_iterations (int): Number of iterations to evaluate.
        eta (float): Multiplier for the noise.
        l (float): Weighting factor for the entropy term.
    """

    def __init__(
        self,
        objective_function: Callable,
        initial_x: np.ndarray,
        num_iterations: int = 1000,
        eta: float = 0.1,
        num_materials: int = 6,
        l: float = 0.0,
    ):
        self._objective_function = objective_function
        self._initial_x = initial_x
        self._num_iterations = num_iterations
        self._eta = eta
        self._num_materials = num_materials
        self._l = l

    def optimise(
        self,
    ) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Union[float, np.ndarray]]]]:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--params", nargs="+", help="Parameters to optimize", default=[]
        )
        args = parser.parse_args()

        if "all" in args.params:
            parameters_to_optimize = [
                f"param{str(i).zfill(3)}" for i in range(len(self._initial_x))
            ]
        else:
            parameters_from_args = [
                f"param{str(int(param)).zfill(3)}" for param in args.params
            ]
            parameters_from_initial = [
                f"param{str(i).zfill(3)}"
                for i in range(len(self._initial_x))
                if self._initial_x[i] != 0
            ]
            parameters_to_optimize = list(
                set(parameters_from_args + parameters_from_initial)
            )

        best_x = torch.tensor(self._initial_x, dtype=torch.float32)
        best_fitness = self._objective_function(best_x)

        xs = [best_x[0].clone().detach().numpy()]
        fitness = [best_fitness]

        for _ in range(self._num_iterations):
            noise = torch.randn_like(best_x)
            x = best_x + self._eta * noise

            all_params = [
                f"param{str(i).zfill(3)}" for i in range(len(self._initial_x))
            ]
            if len(parameters_to_optimize) > self._num_materials:
                parameters_to_optimize = random.sample(
                    parameters_to_optimize, self._num_materials
                )
            elif len(parameters_to_optimize) < self._num_materials:
                available_params = list(set(all_params) - set(parameters_to_optimize))
                additional_params = random.sample(
                    available_params, self._num_materials - len(parameters_to_optimize)
                )
                parameters_to_optimize.extend(additional_params)

            parameters_mask = {
                f"param{str(i).zfill(3)}": (
                    f"param{str(i).zfill(3)}" in parameters_to_optimize
                )
                for i in range(len(self._initial_x))
            }
            pbounds = {
                param: (0, 1) if change else (0, 0)
                for param, change in parameters_mask.items()
            }
            pbounds["param001"] = (0.03, 0.03)

            # Apply the provided normalization
            for i, param_name in enumerate(pbounds.keys()):
                if pbounds[param_name] == (0, 1):
                    x[i] = torch.clamp(x[i], 0, 1)
                else:
                    x[i] = best_x[i]
            x[1] = 0.03
            sum_remaining = torch.sum(x) - 0.03
            for i in range(len(x)):
                if i != 1:
                    x[i] = x[i] * 0.97 / sum_remaining

            loss = self._objective_function(x, l=self._l)
            if loss.item() < best_fitness:
                best_fitness = loss.item()
                best_x = x.clone()

            xs.append(x[0].clone().detach().numpy())
            fitness.append(loss.item())

        return {"best_x": best_x, "best_fitness": best_fitness}, {
            "x": xs,
            "fitness": fitness,
        }


if __name__ == "__main__":
    import os
    import pandas as pd

    data_sls = pd.read_excel(
        f"{os.path.dirname(os.path.realpath(__file__))}/../../data_sets/new_formulations.xlsx",
    )
    data_sls = data_sls.replace({"Yes": 1, "No": 0})
    data_sls = data_sls.loc[:, "Formulation":"Print"]
    x, y = (
        data_sls.loc[:, "Paracetamol":"Xylitol"].fillna(0),
        data_sls["Print"],
    )
    x = x.mul(0.01)
    x = np.array(x)
    y = np.array(y)

    base_model = EnsembleMLP(
        in_features=x.shape[1],
        out_features=1,
        hidden_features=[32, 32, 32],
        bias=True,
        norm="batch",
        dropout=0.3,
        embedding=False,
        attention=False,
        task="classification",
        num_members=5,
    )

    model = Ensemble(
        model=base_model,
        learning_rate=0.1,
        weight_decay=0.0001,
        epochs=100,
        batch_size=32,
        seed=4,
        gpu=None,
        task="classification",
    )

    model.load("model_full.pt")

    fitted = model.predict(x)

    # Freeze the base model
    for param in base_model.parameters():
        param.requires_grad = False

    # Define objective function
    def objective_function(
        x: Optional[torch.Tensor] = None, l: float = 0.0, **kwargs: Any
    ) -> torch.Tensor:
        # Make predictions with x
        if x is None:  # The parameters were specified as kwargs
            x = torch.tensor(
                [[kwargs[key] for key in kwargs.keys()]],
                dtype=torch.float32,
                requires_grad=False,
            ).numpy()
        if len(x.shape) == 1:
            if isinstance(x, torch.Tensor):
                x = x.unsqueeze(0)
            elif isinstance(x, np.ndarray):
                x = np.expand_dims(x, axis=0)
        preds = model.predict_proba(x, return_tensor=True)
        entropy = binary_entropy(preds)
        # Return a large loss if any of the drugs are modified
        if torch.any(x[0, :17] != torch.tensor(initial_x[:17], dtype=torch.float32)):
            return torch.tensor(1e6)

        # Calculate the mean-squared error between `1` which is the desired output and the predictions
        loss = torch.mean((torch.ones_like(preds) - preds) ** 2)
        return loss + l * entropy

    fails = []
    for i in range(len(fitted)):
        if fitted[i] == 0 and y[i] == 0:
            fails.append(x[i])

    l_values = [1, 0.1, 0.01]
    for l_value in l_values:
        with open(
            f"{os.path.dirname(os.path.realpath(__file__))}/../../recommenders/proportion/results_rs_{l_value}.csv",
            "a",
            newline="",
        ) as csvfile:
            writer = csv.writer(csvfile)

            # Write header only if the file is empty (this checks for an empty file or a file that doesn't exist yet)
            if csvfile.tell() == 0:
                writer.writerow(
                    ["initial point", "initial value", "new print", "new value"]
                )

            for fail in tqdm(fails):
                initial_x = fail
                initial_value = model.predict_proba(np.expand_dims(fail, axis=0))
                optimizer = ProportionRecommenderRS(
                    objective_function,
                    initial_x,
                    eta=0.03,
                    num_iterations=3000,
                    num_materials=6,
                    l=l_value,
                )
                best_params, history = optimizer.optimise()
                new_print = best_params["best_x"]
                new_value = model.predict_proba(best_params["best_x"].unsqueeze(0))

                # Append the results to the CSV
                writer.writerow([initial_x, initial_value, new_print, new_value])
