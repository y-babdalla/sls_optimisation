"""Main module for proportion recommender with Bayesian Optimisation.

This module implements a Bayesian Optimisation algorithm to determine optimal
proportions of materials for improved medicine printability.
"""

import csv
import os
import random
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import torch
from bayes_opt import BayesianOptimization
from tqdm import tqdm

from src.recommenders.ensemble_mlp import Ensemble, EnsembleMLP
from src.recommenders.proportion.utils import binary_entropy


class ProportionRecommenderBO:
    """Run Bayesian Optimisation to find material proportions that improve printability.

    This class uses a Bayesian Optimiser (from `bayes_opt`) to maximise a given
    objective function that scores the printability of certain material
    proportions in 3D-printed medicine formulations.

    Args:
        objective_function (Callable): The objective function to optimise. Must return
            a torch.Tensor.
        initial_x (np.ndarray): The initial point for optimisation, representing
            proportions of materials.
        acquisition_function (UtilityFunction | None): The acquisition function used
            by the Bayesian optimiser.
        num_iterations (int): Number of optimiser iterations.
        num_materials (int): Number of materials to optimise.
        lambda_value (float): Weighting factor for the entropy term in the objective
            function (commonly referred to as 'l' in the original code).
    """

    def __init__(
        self,
        objective_function: Callable,
        initial_x: np.ndarray,
        acquisition_function: Callable | None = None,
        num_iterations: int = 1000,
        num_materials: int = 6,
        lambda_value: float = 0.0,
    ) -> None:
        def negative_objective_function(*args: Any, **kwargs: Any) -> float:
            """Return the negative of the user-defined objective function."""
            score = objective_function(lambda_value=lambda_value, *args, **kwargs).item()
            return -score

        self.objective_function = negative_objective_function
        self.initial_x = initial_x
        self.acquisition_function = acquisition_function
        self.num_iterations = num_iterations
        self.num_materials = num_materials
        self.lambda_value = lambda_value

        self.all_params = [f"param{str(i).zfill(3)}" for i in range(len(self.initial_x))]

        self.parameters_to_optimize = self._select_parameters_for_optimisation()
        pbounds = self._create_parameter_bounds()

        # Lock first 17 parameters to their initial values
        self._lock_initial_parameters(pbounds, 17)

        self._optimizer = BayesianOptimization(
            f=self.objective_function, allow_duplicate_points=True, pbounds=pbounds, verbose=0
        )

    def _select_parameters_for_optimisation(self) -> list[str]:
        """Select which parameters will actually be optimised.

        Returns:
            list[str]: The names of the parameters chosen for optimisation.
        """
        # Take only those parameters that have non-zero values in the initial vector.
        parameters_from_initial = [
            f"param{str(i).zfill(3)}" for i, value in enumerate(self.initial_x) if value != 0
        ]

        # If there are fewer non-zero parameters than `num_materials`, add random ones.
        if len(parameters_from_initial) < self.num_materials:
            available_params = list(set(self.all_params) - set(parameters_from_initial))
            additional_params = random.sample(
                available_params, self.num_materials - len(parameters_from_initial)
            )
            return parameters_from_initial + additional_params

        return parameters_from_initial

    def _create_parameter_bounds(self) -> dict[str, tuple[float, float]]:
        """Create the parameter bounds for the Bayesian Optimiser.

        Returns:
            dict[str, tuple[float, float]]: Dictionary mapping parameter names
            to their (lower_bound, upper_bound).
        """
        parameters_mask = {
            param: (param in self.parameters_to_optimize) for param in self.all_params
        }
        return {
            param: (0, 1)
            if parameters_mask[param]
            else (self.initial_x[int(param[5:])], self.initial_x[int(param[5:])])
            for param in self.all_params
        }

    def _lock_initial_parameters(
        self, pbounds: dict[str, tuple[float, float]], lock_count: int
    ) -> None:
        """Lock the first `lock_count` parameters to their initial values.

        This modifies `pbounds` in place, ensuring that parameters
        in the specified range cannot vary during optimisation.
        """
        for i in range(lock_count):
            param_key = f"param{str(i).zfill(3)}"
            pbounds[param_key] = (self.initial_x[i], self.initial_x[i])

    def optimise(self) -> tuple[dict[str, np.ndarray | float], dict[str, np.ndarray]]:
        """Run the Bayesian Optimisation procedure.

        Returns:
            (dict[str, np.ndarray | float], dict[str, np.ndarray]):
                A dictionary with the best parameters and the best fitness (loss),
                and a dictionary containing the optimisation history (parameter
                points tested and their corresponding fitness values).
        """
        # Probe the initial point
        self._optimizer.probe(
            params=self.initial_x if len(self.initial_x.shape) == 1 else self.initial_x[0],
            lazy=True,
        )

        self._optimizer.maximize(
            init_points=1,
            n_iter=self.num_iterations,
            acquisition_function=self.acquisition_function,
        )

        best_result = {
            "best_x": normalize_input(
                torch.tensor(
                    [
                        [
                            self._optimizer.max["params"][f"param{str(i).zfill(3)}"]
                            for i in range(self.initial_x.shape[0])
                        ]
                    ],
                    dtype=torch.float32,
                ).numpy()
            ),
            "best_fitness": -self._optimizer.max["target"],
        }

        optimisation_history = {"x": [], "fitness": []}
        for res in self._optimizer.res:
            param_values = [
                res["params"][f"param{str(j).zfill(3)}"] for j in range(self.initial_x.shape[0])
            ]
            optimisation_history["x"].append(
                torch.tensor([param_values], dtype=torch.float32)[0].numpy()
            )
            optimisation_history["fitness"].append(-res["target"])

        optimisation_history["x"] = np.array(optimisation_history["x"])
        return best_result, optimisation_history


def normalize_input(x_array: np.ndarray) -> np.ndarray:
    """Normalise an array so that the first 17 entries remain fixed sums.

    Args:
        x_array (np.ndarray): Input array representing material proportions.

    Returns:
        np.ndarray: Normalised array that respects total sum constraints.
    """
    x_array = x_array.squeeze()
    first_part = x_array[:17]
    second_part = x_array[17:]
    remaining_sum = 1 - np.sum(first_part)

    if np.sum(second_part) == 0:
        # If the second part sums to zero, we cannot scale it
        normalized_second_part = second_part
    else:
        normalized_second_part = second_part * (remaining_sum / np.sum(second_part))

    return np.concatenate([first_part, normalized_second_part])


def objective_function(
    x_array: torch.Tensor | np.ndarray | None = None, lambda_value: float = 0.0, **kwargs: Any
) -> torch.Tensor:
    """Compute the objective function for the current material proportions.

    If x_array is not provided, it is inferred from kwargs. The objective is
    based on how far the model's prediction is from the target (1 for 'printable'),
    plus an entropy term weighted by lambda_value.

    Args:
        x_array (torch.Tensor | np.ndarray | None): Current proportions to evaluate.
        lambda_value (float): Weighting factor for an entropy term.
        **kwargs (Any): Named parameter proportions if x_array is not given.

    Returns:
        torch.Tensor: The computed objective value.
    """
    # Obtain array from keyword arguments if x_array is not explicitly given
    if x_array is None:
        x_array = torch.tensor(
            [[kwargs[key] for key in kwargs]], dtype=torch.float32, requires_grad=False
        ).numpy()

    x_array = normalize_input(np.array(x_array))
    if len(x_array.shape) == 1:
        x_array = np.expand_dims(x_array, axis=0)

    preds = model.predict_proba(x_array, return_tensor=True)
    entropy_val = binary_entropy(preds)

    loss = torch.mean((torch.ones_like(preds) - preds) ** 2)
    return loss + lambda_value * entropy_val


if __name__ == "__main__":
    data_sls = pd.read_excel(
        f"{os.path.dirname(os.path.realpath(__file__))}/../../data_sets/new_formulations.xlsx"
    )
    data_sls = data_sls.replace({"Yes": 1, "No": 0})
    data_sls = data_sls.loc[:, "Formulation":"Print"]

    x_values = data_sls.loc[:, "Paracetamol":"Xylitol"].fillna(0) * 0.01
    y_values = data_sls["Print"]
    x_values = np.array(x_values)
    y_values = np.array(y_values)

    base_model = EnsembleMLP(
        in_features=x_values.shape[1],
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

    for param in base_model.parameters():
        param.requires_grad = False

    fitted_predictions = model.predict(x_values)

    # Collect samples that the model fails on
    failed_samples = []
    for idx, pred_val in enumerate(fitted_predictions):
        if pred_val == 0 and y_values[idx] == 0:
            failed_samples.append(x_values[idx])

    iteration_values = [200, 300, 400, 1000]

    for iteration_count in tqdm(iteration_values):
        csv_filename = (
            f"{os.path.dirname(os.path.realpath(__file__))}"
            f"/../../recommenders/proportion/results_bo_{iteration_count}.csv"
        )

        with open(csv_filename, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if csvfile.tell() == 0:
                writer.writerow(["initial point", "initial value", "new print", "new value"])

            for failed_sample in tqdm(failed_samples):
                try:
                    initial_val = model.predict_proba(np.expand_dims(failed_sample, 0))

                    optimiser = ProportionRecommenderBO(
                        objective_function=objective_function,
                        initial_x=failed_sample,
                        num_iterations=iteration_count,
                        num_materials=6,
                        lambda_value=0.0,
                    )
                    best_params, optim_history = optimiser.optimise()

                    new_print = best_params["best_x"]
                    new_val = model.predict_proba(np.expand_dims(new_print, 0))

                    writer.writerow([failed_sample, initial_val, new_print, new_val])
                except ValueError:
                    print("fail")
