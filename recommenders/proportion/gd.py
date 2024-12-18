import csv
import random
from typing import Callable, Dict, List, Tuple, Union, Optional, Any

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable

import sys

sys.path.append("./..")
sys.path.append("./../..")

from recommenders.ensemble_mlp import Ensemble, EnsembleMLP
from tqdm import tqdm

from recommenders.proportion.utils import binary_entropy


class ProportionRecommenderGD:
    """This class implements the Gradient Descent algorithm in order to optimize the proportion
    of the materials in the recommendation to improve the printability of the medicine.

    Args:
        objective_function (Callable): Function to optimize.
        initial_x (np.ndarray): Initial point for the optimization.
        num_iterations (int): Number of iterations to evaluate.
        learning_rate (float): Learning rate for optimization.
        noise_sigma (float): Standard deviation of the noise to add to the logits.
        num_materials (int): Number of materials to optimize.
        l (float): Weighting factor for the entropy term.
    """

    def __init__(
        self,
        objective_function: Callable,
        initial_x: np.ndarray,
        num_iterations: int = 1000,
        learning_rate: float = 0.001,
        noise_sigma: float = 0.0,
        num_materials: int = 6,
        l: float = 0.0,
    ):
        self._objective_function = objective_function
        self._initial_x = initial_x
        self._num_iterations = num_iterations
        self._learning_rate = learning_rate
        self._noise_sigma = noise_sigma
        self._num_materials = num_materials

        self._l = l

    def optimise(
        self,
    ) -> Tuple[Dict[str, float], List[Dict[str, Union[float, List[float]]]]]:
        """This method optimizes the proportion of the materials in the recommendation
        to improve the printability of the medicine.

        Returns:
            Tuple[Dict[str, float], List[Dict[str, Union[float, List[float]]]]]: Tuple with the best parameters and the history of the optimization.
        """
        xs = []
        fitness = []
        best_fitness = None
        best_x = None

        # Initialize the first point
        x = Variable(
            torch.tensor(self._initial_x, dtype=torch.float32), requires_grad=True
        )

        optimiser = optim.SGD(
            [x], lr=self._learning_rate, weight_decay=0.0, momentum=0.9
        )
        # Initialize a mask for gradient update
        grad_mask = torch.zeros_like(x)

        # Reset the mask
        grad_mask.fill_(0.0)

        # Set the first 17 to zero as before
        grad_mask[:17] = 0.0

        # Randomly select n parameters from the remaining ones for optimization
        selected_indices = random.sample(list(range(17, len(x))), self._num_materials)
        for idx in selected_indices:
            grad_mask[idx] = 1.0

        for _ in tqdm(range(self._num_iterations)):
            optimiser.zero_grad()
            loss = self._objective_function(x, l=self._l)

            loss.backward()

            # Multiply the gradient with the mask
            x.grad *= grad_mask

            # Add noise to the gradient
            x.grad += torch.randn_like(x) * self._noise_sigma * grad_mask
            optimiser.step()

            # Create a clone of x to avoid in-place operation on the original tensor
            x_clone = x.clone()

            # Normalize the remaining values of the clone so the entire tensor adds up to 1
            sum_of_first_17 = torch.sum(x_clone[:17])
            x_clone[17:] = (
                x_clone[17:] * (1.0 - sum_of_first_17) / torch.sum(x_clone[17:])
            )

            # Clip the values to ensure they're between 0 and 1
            x_clone = torch.clamp(x_clone, 0, 1)

            # Replace the data of x with the modified clone
            x.data = x_clone.data

            # Save the best fitness and the corresponding proportions
            if best_fitness is None or loss.item() < best_fitness:
                best_fitness = loss.item()
                best_x = x.clone().detach().numpy()

            # Save the fitness and the corresponding proportions
            xs.append(x[0].clone().detach().numpy())
            fitness.append(loss.item())
        return {"best_x": best_x, "best_fitness": best_fitness}, {
            "x": xs,
            "fitness": fitness,
        }


# Example usage
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
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

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=4
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_test, y_test, test_size=0.5, random_state=4
    )

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

        # Calculate the mean-squared error between `1` which is the desired output and the predictions
        loss = torch.mean((torch.ones_like(preds) - preds) ** 2)
        return loss + l * entropy

    fails = []
    for i in range(len(fitted)):
        if fitted[i] == 0 and y[i] == 0:
            fails.append(x[i])

    l_values = [1, 0.1, 0.01]
    for l_value in tqdm(l_values):
        with open(
            f"{os.path.dirname(os.path.realpath(__file__))}/../../recommenders/proportion/gd_results_{l_value}.csv",
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
                optimizer = ProportionRecommenderGD(
                    objective_function,
                    initial_x,
                    learning_rate=0.03,
                    num_iterations=4000,
                    noise_sigma=0.001,
                    num_materials=6,
                    l=l_value,
                )
                best_params, history = optimizer.optimise()
                new_print = best_params["best_x"]
                new_value = model.predict_proba(
                    np.expand_dims(best_params["best_x"], 0)
                )

                # Append the results to the CSV
                writer.writerow([initial_x, initial_value, new_print, new_value])
