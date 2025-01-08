"""Main script for proportion recommender with Differential Evolution.

This script implements a Differential Evolution algorithm for optimising
proportions in 3D-printed medicine formulations, using an existing ensemble
model as an evaluator. It loads data, builds or loads a model, and then
optimises material proportions to improve printability.
"""

import csv
import os
import random
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.recommenders.ensemble_mlp import Ensemble, EnsembleMLP
from src.recommenders.proportion.utils import binary_entropy


class ProportionRecommenderDE:
    """Class implementing the Differential Evolution algorithm to optimise material proportions.

    This class uses Differential Evolution to search for optimal proportions of materials in a
    3D-printed medicine formulation, aiming to improve printability.

    Args:
        objective_function (Callable): The objective function to optimise.
            Should return a torch.Tensor.
        initial_x (np.ndarray): Initial proportions for optimisation.
        num_iterations (int): Number of iterations of the DE algorithm.
        pop_size (int): Population size for the DE algorithm.
        mutation_factor (float): Factor controlling the 'distance' for mutation.
        crossover_factor (float): Probability of crossing over genes.
        num_materials (int): Number of materials to optimise.
        lambda_value (float): Weighting factor for an entropy term in the objective function.
    """

    def __init__(
        self,
        objective_function: Callable,
        initial_x: np.ndarray,
        num_iterations: int = 1000,
        pop_size: int = 100,
        mutation_factor: float = 0.4,
        crossover_factor: float = 0.8,
        num_materials: int = 6,
        lambda_value: float = 0.0,
    ) -> None:
        def _objective_function(*args: Any, **kwargs: Any) -> float:
            """Evaluate the user-defined objective function and return its float value."""
            return objective_function(l=lambda_value, *args, **kwargs).item()

        self._objective_function = _objective_function
        self._initial_x = initial_x
        self._num_iterations = num_iterations
        self._pop_size = pop_size
        self._mutation_factor = mutation_factor
        self._crossover_factor = crossover_factor
        self._num_materials = num_materials
        self._lambda_value = lambda_value

        self._history: dict[str, list[Any]] = {"x": [], "fitness": []}
        self._best_fitness: float = np.inf
        self._best_x: np.ndarray | None = None

        all_params = [f"param{str(i).zfill(3)}" for i in range(len(self._initial_x))]

        # Extract parameters that have non-zero values in the initial formulation
        parameters_from_initial = [
            f"param{str(i).zfill(3)}" for i, val in enumerate(self._initial_x) if val != 0
        ]

        # If there are fewer parameters in initial_x than num_materials, add random ones
        if len(parameters_from_initial) < self._num_materials:
            available_params = list(set(all_params) - set(parameters_from_initial))
            additional_params = random.sample(
                available_params, self._num_materials - len(parameters_from_initial)
            )
            parameters_to_optimise = parameters_from_initial + additional_params
        else:
            parameters_to_optimise = parameters_from_initial

        parameters_mask = {param: (param in parameters_to_optimise) for param in all_params}
        pbounds = {
            param: (0, 1)
            if parameters_mask[param]
            else (self._initial_x[int(param[5:])], self._initial_x[int(param[5:])])
            for param in all_params
        }

        # Keep the first 17 indices locked to initial values
        for i in range(17):
            param_key = f"param{str(i).zfill(3)}"
            pbounds[param_key] = (self._initial_x[i], self._initial_x[i])

        self._pbounds = [pbounds[f"param{str(i).zfill(3)}"] for i in range(len(self._initial_x))]
        self.selected_indices = [int(param[5:]) for param in parameters_to_optimise]

    def _init_population(self) -> np.ndarray:
        """Initialise a random population within valid parameter bounds."""
        population = []
        for _ in range(self._pop_size):
            vector = np.random.rand(len(self._initial_x))
            vector = self._check_bounds(vector)
            population.append(vector)
        return np.array(population)

    def _check_bounds(self, vector: np.ndarray) -> np.ndarray:
        """Clip each value of `vector` to the pre-defined parameter bounds."""
        for i, val in enumerate(vector):
            lower_bound, upper_bound = self._pbounds[i]
            vector[i] = max(min(val, upper_bound), lower_bound)

        # Ensure the first 17 indices are locked to initial values
        vector[:17] = self._initial_x[:17]

        # Lock any non-selected parameters
        not_selected_indices = np.setdiff1d(
            np.arange(17, len(self._initial_x)), self.selected_indices
        )
        vector[not_selected_indices] = self._initial_x[not_selected_indices]

        # Rescale the rest so total sum remains 1
        sum_first_17 = np.sum(vector[:17])
        required_sum_for_rest = 1 - sum_first_17
        sum_rest = np.sum(vector[17:])
        if sum_rest != 0:
            normalisation_factor = required_sum_for_rest / sum_rest
            vector[17:] *= normalisation_factor

        return vector

    def _mutate(self, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> np.ndarray:
        """Apply DE mutation strategy to generate a mutant vector."""
        mutant = v1 + self._mutation_factor * (v2 - v3)
        return self._check_bounds(mutant)

    def _crossover(self, current: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """Perform crossover between the current vector and the mutant vector."""
        crossover_vector = np.random.rand(len(current)) < self._crossover_factor
        trial = np.where(crossover_vector, mutant, current)
        return self._check_bounds(trial)

    def optimise(self) -> tuple[dict[str, torch.Tensor | float], dict[str, np.ndarray]]:
        """Optimise the material proportions using the DE algorithm.

        Returns:
            (dict[str, torch.Tensor | float], dict[str, np.ndarray]): A tuple
            containing:
            - A dictionary with the best parameter vector and its fitness.
            - A history dictionary logging the best parameter vectors and fitnesses.
        """
        self.population = self._init_population()

        for _ in range(self._num_iterations):
            for j in range(self._pop_size):
                current_vector = self.population[j]
                # Sample 3 distinct vectors different from j
                i1, i2, i3 = random.sample([idx for idx in range(self._pop_size) if idx != j], 3)
                v1, v2, v3 = self.population[i1], self.population[i2], self.population[i3]

                mutant_vector = self._mutate(v1, v2, v3)
                trial_vector = self._crossover(current_vector, mutant_vector)

                current_fitness = self._objective_function(current_vector)
                trial_fitness = self._objective_function(trial_vector)

                if trial_fitness < current_fitness:
                    self.population[j] = trial_vector

                if trial_fitness < self._best_fitness:
                    self._best_fitness = trial_fitness
                    self._best_x = trial_vector
                    self._history["x"].append(self._best_x)
                    self._history["fitness"].append(self._best_fitness)

        best = {"best_x": torch.tensor([self._best_x]).float(), "best_fitness": self._best_fitness}
        self._history["x"] = np.array(self._history["x"])
        return best, self._history


if __name__ == "__main__":
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
        """Compute a fitness score for given proportions x, including an entropy term."""
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

    initial_x = x_values[-1]
    optimiser_de = ProportionRecommenderDE(
        objective_function=objective_function, initial_x=initial_x, num_iterations=100
    )
    best_params_de, history_de = optimiser_de.optimise()

    print("Best x:", best_params_de["best_x"].shape)
    print("Best score:", model.predict_proba(best_params_de["best_x"]))
    print("Best fitness:", best_params_de["best_fitness"])

    l_values = [1, 0.1, 0.01]
    failed_samples = [
        x_values[i]
        for i, pred_val in enumerate(fitted_values)
        if pred_val == 0 and y_values[i] == 0
    ]

    for l_value in tqdm(l_values):
        csv_filename = (
            f"{os.path.dirname(os.path.realpath(__file__))}"
            f"/../../recommenders/proportion/results_de_{l_value}.csv"
        )
        with open(csv_filename, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if csvfile.tell() == 0:
                writer.writerow(["initial point", "initial value", "new print", "new value"])

            for failed_sample in tqdm(failed_samples):
                initial_value = model.predict_proba(np.expand_dims(failed_sample, 0))
                optimiser_de = ProportionRecommenderDE(
                    objective_function=objective_function,
                    initial_x=failed_sample,
                    num_iterations=50,
                    num_materials=6,
                    lambda_value=l_value,
                )
                best_params_de, history_de = optimiser_de.optimise()
                new_print = best_params_de["best_x"]
                new_value = model.predict_proba(new_print)
                writer.writerow([failed_sample, initial_value, new_print, new_value])
