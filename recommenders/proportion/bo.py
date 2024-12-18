import argparse
import csv
import random
from typing import Dict, List, Tuple, Union, Callable, Optional, Any

from tqdm import tqdm

import numpy as np
import torch

import sys

sys.path.append("./..")
sys.path.append("./../..")

from recommenders.ensemble_mlp import Ensemble, EnsembleMLP

from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from recommenders.proportion.utils import binary_entropy


class ProportionRecommenderBO:
    """This class implements the Bayesian Optimization algorithm in order to optimize the proportion
    of the materials in the recommendation to improve the printability of the medicine.

    Args:
        objective_function (Callable): Function to optimize.
        initial_x (np.ndarray): Initial point for the optimization.
        acquisition_function (Optional[UtilityFunction]): Acquisition function to use.
        num_iterations (int): Number of iterations to evaluate.
        num_materials (int): Number of materials to optimize.
        l (float): Weighting factor for the entropy term.
    """

    def __init__(
        self,
        objective_function: Callable,
        initial_x: np.ndarray,
        acquisition_function: Optional[UtilityFunction] = None,
        num_iterations: int = 1000,
        num_materials: int = 6,
        l: float = 0.0,
    ):
        # Wrap the objective function to maximize it by minimizing the negative of it
        # return .item() because the objective function returns a tensor
        def _objective_function(*args: Any, **kwargs: Any) -> float:
            return -objective_function(l=l, *args, **kwargs).item()

        self._objective_function = _objective_function
        self._initial_x = initial_x
        self._acquisition_function = acquisition_function
        self._num_iterations = num_iterations
        self._num_materials = num_materials
        self._l = l

        all_params = [f"param{str(i).zfill(3)}" for i in range(len(self._initial_x))]

        # Extract parameters that have non-zero values in the initial formulation.
        parameters_from_initial = [
            f"param{str(i).zfill(3)}"
            for i in range(len(self._initial_x))
            if self._initial_x[i] != 0
        ]

        # If there are fewer parameters in initial formulation than self._num_materials, add random ones.
        if len(parameters_from_initial) < self._num_materials:
            available_params = list(set(all_params) - set(parameters_from_initial))
            additional_params = random.sample(
                available_params, self._num_materials - len(parameters_from_initial)
            )
            parameters_to_optimize = parameters_from_initial + additional_params

        # If there are more parameters in the initial formulation than self._num_materials, use them as is.
        else:
            parameters_to_optimize = parameters_from_initial

        parameters_mask = {
            param: (param in parameters_to_optimize) for param in all_params
        }
        pbounds = {
            param: (0, 1)
            if change
            else (self._initial_x[int(param[5:])], self._initial_x[int(param[5:])])
            for param, change in parameters_mask.items()
        }

        # Keep the bounds of the first 17 indices same as in the initial material
        for i in range(17):
            param_key = f"param{str(i).zfill(3)}"
            pbounds[param_key] = (self._initial_x[i], self._initial_x[i])

        self._optimizer = BayesianOptimization(
            f=self._objective_function,
            allow_duplicate_points=True,
            pbounds=pbounds,
            verbose=0,
        )

    def optimise(
        self,
    ) -> Tuple[Dict[str, float], List[Dict[str, Union[float, List[float]]]]]:
        """This method optimizes the proportion of the materials in the recommendation
        to improve the printability of the medicine.

        Returns:
            Tuple[Dict[str, float], List[Dict[str, Union[float, List[float]]]]]: Tuple with the best parameters and the history of the optimization.
        """
        self._optimizer.probe(
            params=self._initial_x[0]
            if len(self._initial_x.shape) > 1
            else self._initial_x,
            lazy=True,
        )

        self._optimizer.maximize(
            init_points=1,
            n_iter=self._num_iterations,
            acquisition_function=self._acquisition_function,
        )
        best = {
            "best_x": normalize_input(
                torch.tensor(
                    [
                        [
                            self._optimizer.max["params"][f"param{str(i).zfill(3)}"]
                            for i in range(self._initial_x.shape[0])
                        ]
                    ],
                    dtype=torch.float32,
                ).numpy()
            ),
            "best_fitness": -self._optimizer.max["target"],
        }
        history = {"x": [], "fitness": []}
        for i in range(len(self._optimizer.res)):
            history["x"].append(
                torch.tensor(
                    [
                        [
                            self._optimizer.res[i]["params"][f"param{str(j).zfill(3)}"]
                            for j in range(self._initial_x.shape[0])
                        ]
                    ],
                    dtype=torch.float32,
                )[0].numpy()
            )
            history["fitness"].append(-self._optimizer.res[i]["target"])
        history["x"] = np.array(history["x"])
        return best, history


def normalize_input(x: np.ndarray) -> np.ndarray:
    x = x.squeeze()  # Removing any singleton dimensions
    first_part = x[:17]
    second_part = x[17:]
    remaining_sum = 1 - np.sum(first_part)
    if np.sum(second_part) == 0:
        normalized_second_part = second_part
    else:
        normalized_second_part = second_part * (remaining_sum / np.sum(second_part))
    return np.concatenate([first_part, normalized_second_part])


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
        x = normalize_input(np.array(x))

        if len(x.shape) == 1:
            if isinstance(x, torch.Tensor):
                x = x.unsqueeze(0)
            elif isinstance(x, np.ndarray):
                x = np.expand_dims(x, axis=0)
        preds = model.predict_proba(x, return_tensor=True)
        entropy = binary_entropy(preds)
        # Calculate the mean-squared error between `1` which is the desired output and the predictions
        loss = torch.mean((torch.ones_like(preds) - preds) ** 2)
        return loss + l * entropy


iterations = [200, 300, 400, 1000]
fails = []

for i in range(len(fitted)):
    if fitted[i] == 0 and y[i] == 0:
        fails.append(x[i])
for iteration in tqdm(iterations):
    with open(
        f"{os.path.dirname(os.path.realpath(__file__))}/../../recommenders/proportion/results_bo_{iteration}.csv",
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
            try:
                initial_x = fail
                initial_value = model.predict_proba(np.expand_dims(initial_x, 0))
                optimizer = ProportionRecommenderBO(
                    objective_function=objective_function,
                    initial_x=initial_x,
                    num_iterations=iteration,
                    num_materials=6,
                )
                best_params, history = optimizer.optimise()
                new_print = best_params["best_x"]
                new_value = model.predict_proba(
                    np.expand_dims(best_params["best_x"], 0)
                )

                # Append the results to the CSV
                writer.writerow([initial_x, initial_value, new_print, new_value])
            except ValueError:
                print("fail")
