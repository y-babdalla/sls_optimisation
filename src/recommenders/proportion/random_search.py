"""Main script for a random search-based proportion recommender.

This module implements a random search algorithm to optimise material
proportions for improved 3D-printed medicine printability. The class
`ProportionRecommenderRS` defines the random search procedure, and the
example usage at the end demonstrates how to use it with a pre-trained
ensemble model and record results in a CSV file.
"""

import argparse
import csv
import random
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from src.recommenders.ensemble_mlp import Ensemble, EnsembleMLP
from src.recommenders.proportion.utils import binary_entropy


class ProportionRecommenderRS:
    """Random Search optimiser for discovering improved material proportions.

    This class explores the search space by repeatedly generating random
    perturbations from the current best solution. If a new, perturbed solution
    yields a lower objective value, it becomes the new best solution.

    Args:
        objective_function (Callable): Objective/loss function to minimise.
        initial_x (np.ndarray): Initial proportion vector.
        num_iterations (int): Number of random search iterations.
        eta (float): Noise scale factor.
        num_materials (int): Number of materials to optimise.
        lambda_value (float): Weighting factor for an entropy term in the objective.
    """

    def __init__(
        self,
        objective_function: Callable,
        initial_x: np.ndarray,
        num_iterations: int = 1000,
        eta: float = 0.1,
        num_materials: int = 6,
        lambda_value: float = 0.0,
    ) -> None:
        self.objective_function = objective_function
        self.initial_x = initial_x
        self.num_iterations = num_iterations
        self.eta = eta
        self.num_materials = num_materials
        self.lambda_value = lambda_value

    def optimise(
        self,
    ) -> tuple[dict[str, torch.Tensor | float], dict[str, list[torch.Tensor | float]]]:
        """Run random search to find improved material proportions.

        This method also parses command-line arguments to determine which
        parameters are optimised. If "all" is specified, all parameters
        can be changed; otherwise, only the intersection of user-specified
        parameters and non-zero initial parameters is used.

        Returns:
            (dict[str, torch.Tensor | float], dict[str, list[torch.Tensor | float]]):
                A dictionary with the best parameter vector ('best_x') and its
                fitness ('best_fitness'), and a history dictionary with 'x' and
                'fitness' lists.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--params",
            nargs="+",
            help="Parameters to optimise (use 'all' to optimise everything).",
            default=[],
        )
        args = parser.parse_args()

        if "all" in args.params:
            parameters_to_optimise = [
                f"param{str(i).zfill(3)}" for i in range(len(self.initial_x))
            ]
        else:
            # Transform numeric arguments into 'paramXYZ' form
            parameters_from_args = [f"param{str(int(param)).zfill(3)}" for param in args.params]
            parameters_from_initial = [
                f"param{str(i).zfill(3)}" for i, val in enumerate(self.initial_x) if val != 0
            ]
            parameters_to_optimise = list(set(parameters_from_args + parameters_from_initial))

        best_x = torch.tensor(self.initial_x, dtype=torch.float32)
        best_fitness = self.objective_function(best_x)

        x_history = [best_x[0].clone().detach().numpy()]
        fitness_history = [best_fitness.item() if hasattr(best_fitness, "item") else best_fitness]

        for _ in range(self.num_iterations):
            noise = torch.randn_like(best_x)
            candidate = best_x + self.eta * noise

            all_params = [f"param{str(i).zfill(3)}" for i in range(len(self.initial_x))]
            if len(parameters_to_optimise) > self.num_materials:
                parameters_to_optimise = random.sample(parameters_to_optimise, self.num_materials)
            elif len(parameters_to_optimise) < self.num_materials:
                available_params = list(set(all_params) - set(parameters_to_optimise))
                additional_params = random.sample(
                    available_params, self.num_materials - len(parameters_to_optimise)
                )
                parameters_to_optimise.extend(additional_params)

            # Mark which parameters can vary (0-1) vs. fixed (0 or original)
            param_mask = {
                f"param{str(i).zfill(3)}": (f"param{str(i).zfill(3)}" in parameters_to_optimise)
                for i in range(len(self.initial_x))
            }
            pbounds = {p: (0, 1) if can_change else (0, 0) for p, can_change in param_mask.items()}
            pbounds["param001"] = (0.03, 0.03)

            # Apply bounds to the candidate
            for i, param_name in enumerate(pbounds.keys()):
                if pbounds[param_name] == (0, 1):
                    candidate[i] = torch.clamp(candidate[i], 0, 1)
                else:
                    candidate[i] = best_x[i]

            candidate[1] = 0.03

            sum_remaining = torch.sum(candidate) - 0.03
            for i in range(len(candidate)):
                if i != 1:
                    candidate[i] = (
                        candidate[i] * 0.97 / sum_remaining if sum_remaining != 0 else 0.0
                    )

            candidate_fitness = self.objective_function(candidate, l=self.lambda_value)

            if candidate_fitness.item() < best_fitness.item():
                best_fitness = candidate_fitness
                best_x = candidate.clone()

            x_history.append(candidate[0].clone().detach().numpy())
            fitness_history.append(candidate_fitness.item())

        return (
            {"best_x": best_x, "best_fitness": best_fitness.item()},
            {"x": x_history, "fitness": fitness_history},
        )


if __name__ == "__main__":
    import os

    import pandas as pd

    data_sls = pd.read_excel(
        f"{os.path.dirname(os.path.realpath(__file__))}/../../data_sets/new_formulations.xlsx"
    )
    data_sls = data_sls.replace({"Yes": 1, "No": 0})
    data_sls = data_sls.loc[:, "Formulation":"Print"]
    x_values = data_sls.loc[:, "Paracetamol":"Xylitol"].fillna(0)
    y_values = data_sls["Print"]

    x_values = x_values.mul(0.01)
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

    fitted_values = model.predict(x_values)

    for param in base_model.parameters():
        param.requires_grad = False

    def objective_function(
        x: torch.Tensor | None = None, l: float = 0.0, **kwargs: Any
    ) -> torch.Tensor:
        """Compute the loss as mean-squared error plus entropy, ensuring first 17 are unchanged."""
        if x is None:
            x = torch.tensor(
                [[kwargs[k] for k in kwargs]], dtype=torch.float32, requires_grad=False
            ).numpy()
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=0)

        preds = model.predict_proba(x, return_tensor=True)
        ent = binary_entropy(preds)

        # If the first 17 are changed from the original, return large penalty
        initial_x_subset = torch.tensor(x_values[:17], dtype=torch.float32)
        if torch.any(x[0, :17] != initial_x_subset):
            return torch.tensor(1e6)

        # MSE from desired output of 1
        loss_mse = torch.mean((torch.ones_like(preds) - preds) ** 2)
        return loss_mse + l * ent

    # Identify failed samples
    fails = [
        x_val
        for x_val, pred_val, true_val in zip(x_values, fitted_values, y_values, strict=False)
        if (pred_val == 0 and true_val == 0)
    ]

    # Try different lambda values and record results
    l_values = [1, 0.1, 0.01]
    for l_val in l_values:
        csv_filename = (
            f"{os.path.dirname(os.path.realpath(__file__))}"
            f"/../../recommenders/proportion/results_rs_{l_val}.csv"
        )
        with open(csv_filename, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if csvfile.tell() == 0:
                writer.writerow(["initial point", "initial value", "new print", "new value"])

            for fail in tqdm(fails):
                init_val = model.predict_proba(np.expand_dims(fail, axis=0))
                optimiser_rs = ProportionRecommenderRS(
                    objective_function=objective_function,
                    initial_x=fail,
                    eta=0.03,
                    num_iterations=3000,
                    num_materials=6,
                    lambda_value=l_val,
                )

                best_params, history = optimiser_rs.optimise()
                new_print = best_params["best_x"]
                new_val = model.predict_proba(new_print.unsqueeze(0))

                writer.writerow([fail, init_val, new_print, new_val])
