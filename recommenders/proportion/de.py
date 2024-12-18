import csv
import random
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

sys.path.append("./..")
sys.path.append("./../..")

from recommenders.ensemble_mlp import Ensemble, EnsembleMLP
from recommenders.proportion.utils import binary_entropy


class ProportionRecommenderDE:
    """This class implements the Differential Evolution algorithm in order to optimize the proportion

    Args:
        objective_function (Callable): Function to optimize.
        initial_x (np.ndarray): Initial point for the optimization.
        num_iterations (int): Number of iterations to evaluate.
        pop_size (int): Population size.
        mutation_factor (float): Mutation factor.
        crossover_factor (float): Crossover factor.
        num_materials (int): Number of materials to optimize.
        l (float): Weighting factor for the entropy term.
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
        l: float = 0.0,
    ):
        # return .item() because the objective function returns a tensor
        def _objective_function(*args: Any, **kwargs: Any) -> float:
            return objective_function(l=l, *args, **kwargs).item()

        self._objective_function = _objective_function

        self._initial_x = initial_x
        self._num_iterations = num_iterations
        self._pop_size = pop_size
        self._mutation_factor = mutation_factor
        self._crossover_factor = crossover_factor
        self._l = l

        # Define a callback function to save the history of the optimization
        self._history = {"x": [], "fitness": []}
        self._best_fitness = np.inf
        self._best_x = None

        self._num_materials = num_materials

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

        self._pbounds = [
            pbounds[f"param{str(i).zfill(3)}"] for i in range(len(self._initial_x))
        ]
        self.selected_indices = [int(param[5:]) for param in parameters_to_optimize]

    def _init_population(self) -> np.ndarray:
        """Initialize a random population within the bounds by checking with the the `_check_bounds` method."""
        population = []
        for _ in range(self._pop_size):
            vector = np.random.rand(len(self._initial_x))
            vector = self._check_bounds(vector)
            population.append(vector)

        return np.array(population)

    def _check_bounds(self, vector: np.ndarray) -> np.ndarray:
        """Check if the vector is within the bounds. If not, clip the values to the bounds."""
        for i in range(len(vector)):
            lower_bound = self._pbounds[i][0]
            upper_bound = self._pbounds[i][1]
            vector[i] = max(min(vector[i], upper_bound), lower_bound)

        # Make sure to also check for the bounds of the first 17 indices
        vector[:17] = self._initial_x[:17]
        not_selected_indices = np.setdiff1d(
            np.arange(17, len(self._initial_x)), self.selected_indices
        )
        vector[not_selected_indices] = self._initial_x[not_selected_indices]

        # Normalize the remaining indices
        sum_first_17 = np.sum(vector[:17])
        required_sum_for_rest = 1 - sum_first_17
        sum_rest = np.sum(vector[17:])
        if sum_rest != 0:  # Avoid division by zero
            normalization_factor = required_sum_for_rest / sum_rest
            vector[17:] *= normalization_factor

        return vector

    def _mutate(self, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> np.ndarray:
        """Mutate the population using the DE mutation strategy"""
        mutant = v1 + self._mutation_factor * (v2 - v3)

        # Clip the values to the bounds, depenting on `self._pbounds`
        mutant = self._check_bounds(mutant)
        return mutant

    def _crossover(self, current: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """Perform the crossover between the current vector and the mutant vector"""
        # Create a random vector of booleans
        crossover_vector = np.random.rand(len(current)) < self._crossover_factor
        # Perform the crossover
        trial = np.where(crossover_vector, mutant, current)
        # Clip the values to the bounds, depenting on `self._pbounds`
        trial = self._check_bounds(trial)
        return trial

    def optimise(
        self,
    ) -> Tuple[Dict[str, float], List[Dict[str, Union[float, List[float]]]]]:
        """This method optimizes the proportion of the materials in the recommendation
        to improve the printability of the medicine.

        Returns:
            Tuple[Dict[str, float], List[Dict[str, Union[float, List[float]]]]]: Tuple with the best parameters and the history of the optimization.
        """

        self.population = self._init_population()

        for _ in range(self._num_iterations):
            for j in range(self._pop_size):
                current_vector = self.population[j]
                # Sample 3 random vectors where the indeix of the current vector is not included
                i1, i2, i3 = random.sample(
                    [x for x in range(self._pop_size) if x != j], 3
                )

                v1, v2, v3 = (
                    self.population[i1],
                    self.population[i2],
                    self.population[i3],
                )

                # Create a mutant vector
                mutant = self._mutate(v1, v2, v3)

                # Create a trial vector
                trial = self._crossover(current_vector, mutant)

                # Evaluate the current vector
                current_fitness = self._objective_function(current_vector)

                # Evaluate the trial vector
                trial_fitness = self._objective_function(trial)

                # If the trial vector is better than the current vector, replace the current vector
                if trial_fitness < current_fitness:
                    self.population[j] = trial

                if trial_fitness < self._best_fitness:
                    self._best_fitness = trial_fitness
                    self._best_x = trial
                    self._history["x"].append(self._best_x)
                    self._history["fitness"].append(self._best_fitness)

        best = {
            "best_x": torch.tensor([self._best_x]).float(),
            "best_fitness": self._best_fitness,
        }
        self._history["x"] = np.array(self._history["x"])
        return best, self._history


if __name__ == "__main__":
    import os
    import pandas as pd

    from tqdm import tqdm

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
        # Ensure x has two dimensions, as the subsequent code seems to expect this
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

    # Create optimizer
    initial_x = x[-1]
    optimizer = ProportionRecommenderDE(
        objective_function=objective_function,
        initial_x=initial_x,
        num_iterations=100,
    )

    # Optimize proportions
    best_params, history = optimizer.optimise()

    # Print results
    print("Best x:", best_params["best_x"].shape)
    print("Best score:", model.predict_proba(best_params["best_x"]))
    print("Best fitness:", best_params["best_fitness"])
    ...
    l_values = [1, 0.1, 0.01]
    fails = []

    for i in range(len(fitted)):
        if fitted[i] == 0 and y[i] == 0:
            fails.append(x[i])
    for l_value in tqdm(l_values):
        with open(
            f"{os.path.dirname(os.path.realpath(__file__))}/../../recommenders/proportion/results_de_{l_value}.csv",
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
                initial_value = model.predict_proba(np.expand_dims(initial_x, 0))
                optimizer = ProportionRecommenderDE(
                    objective_function=objective_function,
                    initial_x=initial_x,
                    num_iterations=50,
                    num_materials=6,
                    l=l_value,
                )
                best_params, history = optimizer.optimise()
                new_print = best_params["best_x"]
                new_value = model.predict_proba(best_params["best_x"])

                # Append the results to the CSV
                writer.writerow([initial_x, initial_value, new_print, new_value])
