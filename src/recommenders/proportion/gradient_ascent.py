"""Main script for proportion recommender with gradient ascent.

This module defines a `ProportionRecommenderGD` class that employs gradient ascent
to optimise material proportions for improved 3D-printed medicine outcomes.
It includes an example usage in the `if __name__ == "__main__":` block.
"""

import csv
import random
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm

from src.recommenders.ensemble_mlp import Ensemble, EnsembleMLP
from src.recommenders.proportion.utils import binary_entropy


class ProportionRecommenderGA:
    """Use gradient ascent to optimise material proportions in 3D-printed medicine.

    Args:
        objective_function (Callable): Function returning a torch.Tensor, used as the loss.
        initial_x (np.ndarray): Initial material proportions.
        num_iterations (int): Number of gradient ascent steps.
        learning_rate (float): Learning rate for the optimiser.
        noise_sigma (float): Standard deviation of random noise added to gradients.
        num_materials (int): Number of materials to optimise.
        lambda_value (float): Weighting factor for an entropy term in the objective.
    """

    def __init__(
        self,
        objective_function: Callable,
        initial_x: np.ndarray,
        num_iterations: int = 1000,
        learning_rate: float = 0.001,
        noise_sigma: float = 0.0,
        num_materials: int = 6,
        lambda_value: float = 0.0,
    ) -> None:
        self.objective_function = objective_function
        self.initial_x = initial_x
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.noise_sigma = noise_sigma
        self.num_materials = num_materials
        self.lambda_value = lambda_value

    def optimise(
        self,
    ) -> tuple[dict[str, np.ndarray | float], dict[str, list[float | np.ndarray]]]:
        """Run the gradient ascent optimisation and return best results and history.

        Returns:
            (dict[str, np.ndarray | float], dict[str, list[float | np.ndarray]]):
                A dictionary with 'best_x' and 'best_fitness' keys, plus a history
                dictionary containing parameter vectors and fitness values over time.
        """
        xs: list[np.ndarray] = []
        fitnesses: list[float] = []
        best_fitness: float | None = None
        best_x: np.ndarray | None = None

        x_var = Variable(torch.tensor(self.initial_x, dtype=torch.float32), requires_grad=True)
        optimiser = optim.SGD([x_var], lr=self.learning_rate, weight_decay=0.0, momentum=0.9)

        gradient_mask = torch.zeros_like(x_var)
        gradient_mask.fill_(0.0)
        gradient_mask[:17] = 0.0  # keep first 17 locked
        selected_indices = random.sample(list(range(17, len(x_var))), self.num_materials)
        for idx in selected_indices:
            gradient_mask[idx] = 1.0

        for _ in tqdm(range(self.num_iterations)):
            optimiser.zero_grad()
            loss = self.objective_function(x_var, l=self.lambda_value)
            loss.backward()

            # Apply mask to update only selected entries
            x_var.grad *= gradient_mask

            # Add Gaussian noise to the gradient for selected entries
            x_var.grad += torch.randn_like(x_var) * self.noise_sigma * gradient_mask

            optimiser.step()

            # Post-update normalisation and clipping
            x_clone = x_var.clone()
            sum_first_17 = torch.sum(x_clone[:17])
            x_clone[17:] = (
                x_clone[17:] * (1.0 - sum_first_17) / torch.sum(x_clone[17:])
                if torch.sum(x_clone[17:]) != 0
                else x_clone[17:]
            )
            x_clone = torch.clamp(x_clone, 0, 1)
            x_var.data = x_clone.data

            current_fitness = loss.item()
            if best_fitness is None or current_fitness < best_fitness:
                best_fitness = current_fitness
                best_x = x_var.detach().numpy()

            xs.append(x_var[0].detach().numpy())
            fitnesses.append(current_fitness)

        return {"best_x": best_x, "best_fitness": best_fitness}, {"x": xs, "fitness": fitnesses}


if __name__ == "__main__":
    import os

    import matplotlib.pyplot as plt  # noqa: F401
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Load data
    data_sls = pd.read_excel(
        f"{os.path.dirname(os.path.realpath(__file__))}/../../data_sets/new_formulations.xlsx"
    )
    data_sls = data_sls.replace({"Yes": 1, "No": 0})
    data_sls = data_sls.loc[:, "Formulation":"Print"]
    x_values, y_values = data_sls.loc[:, "Paracetamol":"Xylitol"].fillna(0), data_sls["Print"]
    x_values = x_values.mul(0.01)
    x_values = np.array(x_values)
    y_values = np.array(y_values)

    x_train, x_test, y_train, y_test = train_test_split(
        x_values, y_values, test_size=0.2, random_state=4
    )
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=4)

    # Initialise and load ensemble model
    base_model = EnsembleMLP(
        in_features=x_train.shape[1],
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

    # Freeze the base model
    for parameter in base_model.parameters():
        parameter.requires_grad = False

    def objective_function(
        x: torch.Tensor | None = None, l: float = 0.0, **kwargs: Any
    ) -> torch.Tensor:
        """Compute the mean-squared error from 1, plus an entropy term weighted by l."""
        if x is None:
            x = torch.tensor(
                [[kwargs[key] for key in kwargs]], dtype=torch.float32, requires_grad=False
            ).numpy()
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=0)

        preds = model.predict_proba(x, return_tensor=True)
        entropy = binary_entropy(preds)
        loss = torch.mean((torch.ones_like(preds) - preds) ** 2)
        return loss + l * entropy

    fails = [
        x_values[i]
        for i, pred_val in enumerate(fitted_values)
        if pred_val == 0 and y_values[i] == 0
    ]

    csv_filename = (
        f"{os.path.dirname(os.path.realpath(__file__))}"
        f"/../../recommenders/proportion/gd_results_{l_val}.csv"
    )
    with open(csv_filename, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if csvfile.tell() == 0:
            writer.writerow(["initial point", "initial value", "new print", "new value"])

        for fail_sample in tqdm(fails):
            init_x = fail_sample
            init_val = model.predict_proba(np.expand_dims(fail_sample, axis=0))

            optimiser = ProportionRecommenderGA(
                objective_function,
                init_x,
                learning_rate=0.03,
                num_iterations=4000,
                noise_sigma=0.001,
                num_materials=6,
            )
            best_params, history = optimiser.optimise()

            new_print = best_params["best_x"]
            new_val = model.predict_proba(np.expand_dims(new_print, axis=0))

            writer.writerow([init_x, init_val, new_print, new_val])
