"""Creates augmented formulation data with optional embeddings, SMILES, and powder features.

This script defines a function `make_formulations` that can transform a DataFrame of
formulation data into various augmented feature representations. It can generate
embedding-like inputs, attach SMILES fingerprints, attach powder properties, and
optionally preserve the original proportions. The script also provides an example
`__main__` section illustrating its usage.

Example:
    >>> data_sls = pd.read_excel("my_formulations.xlsx")
    >>> y = pd.read_excel("my_labels.xlsx")
    >>> features = make_formulations(
    ...     data_sls,
    ...     smiles=True,
    ...     proportions=True,
    ...     powder=None,
    ...     powder_range="median",
    ...     embedding=False,
    ...     seed=42,
    ...     original=False,
    ...     mfp="new",
    ...     y=y,
    ... )
    >>> print(features.shape)

Note:
    - `y` must be provided if `powder` is not None, since a train-test split
      is performed for powder scaling.
    - The `embedding` part expects to return a dict with "x" and "y" keys
      if `embedding=True`.
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.simplefilter(action="ignore", category=FutureWarning)


def make_formulations(  # noqa: C901, PLR0912
    data: pd.DataFrame,
    smiles: bool = True,
    proportions: bool = True,
    powder: bool | None = None,
    powder_range: str = "median",
    embedding: bool = False,
    seed: int = 42,
    original: bool = False,
    mfp: str = "old",
    y: np.ndarray | None = None,
) -> np.ndarray | dict[str, np.ndarray]:
    """Create augmented formulation data from an input DataFrame.

    Depending on the arguments, this function can:
      - Return embeddings (with shape `(num_samples, list_of_lists, 2)`) if
        `embedding` is True.
      - Combine SMILES fingerprints for each material in a formulation, scaling
        by the proportion if `proportions` is True.
      - Add powder properties (optionally scaled), requiring `y` for splitting
        if `powder` is not None.
      - Append original formulation proportions if `original` is True.

    Args:
        data: DataFrame where each row is a formulation, columns are material names.
            Cell values are percentages (e.g. 50 means 50%).
        smiles: If True, appends SMILES fingerprints.
        proportions: If True, scale SMILES by the proportion of each material.
        powder: If not None, merges powder properties from an external Excel file.
            If True, indicates usage of the "particles" sheet from the same file
            used for SMILES data.
        powder_range: One of {"all", "median", "range"}. Selects which
            powder columns are retained before scaling.
        embedding: If True, return "x" and "y" (where each row is a list of
            [material_index, proportion]) suitable for embedding-based usage.
        seed: Random seed used for train_test_split (powder scaling).
        original: If True, also append the raw proportion data (data_sls) to
            the final DataFrame.
        mfp: 'old' (2048 bits) or 'new' (1024 bits) for SMILES columns.
        y: Target labels required if `powder` is not None, to allow scaling
            with train_test_split.

    Returns:
        - If embedding=True, returns a dict with:
            {
                "x": (np.ndarray) shape [num_samples, variable_length, 2],
                "y": (np.ndarray) the same 'y' if given, else raises an error.
            }
        - Otherwise, returns a (num_samples, num_features) numpy array.
    """
    data_sls: pd.DataFrame = data.copy()
    all_formulations = pd.DataFrame()

    if embedding:
        if y is None:
            raise ValueError("Must provide 'y' when using embeddings to return both x and y.")
        # Map each column name (material) to an integer index
        materials = {name: index for index, name in enumerate(data_sls.columns)}
        # Build a list of lists: for each row, a list of [material_index, proportion]
        embeddings_list: list[list[list[float]]] = []
        for _, row in data_sls.iterrows():
            non_nan_cols = row.fillna(0)
            formulation_row = []
            for j in range(len(non_nan_cols)):
                material_name = non_nan_cols.index[j]
                proportion_value = non_nan_cols[j] * 0.01
                formulation_row.append([materials[material_name], proportion_value])
            embeddings_list.append(formulation_row)

        return {"x": np.array(embeddings_list, dtype=object), "y": np.array(y)}

    if smiles:
        if mfp == "old":
            sheet = "smiles"
            end = 2047
        elif mfp == "new":
            sheet = "1024_smiles"
            end = 1023
        else:
            raise ValueError("mfp must be 'old' or 'new'")

        base_path = os.path.dirname(os.path.realpath(__file__))
        full_path = os.path.join(base_path, "..", "data_sets", "sls_smiles_data.xlsx")
        smile_data = pd.read_excel(full_path, sheet_name=sheet)

        material_names = [material.strip() for material in smile_data["Material"]]
        smile_data.index = material_names
        # Only numeric columns from 0 -> end
        smiles_subset = smile_data.loc[:, 0:end]

        formulation_smiles = pd.DataFrame()
        for _, row in data_sls.iterrows():
            non_zero_part = row[row != 0]
            combined_series = pd.Series(dtype=float)

            for j in range(len(non_zero_part)):
                mat_name = non_zero_part.index[j].strip()
                proportion_val = non_zero_part[j] * 0.01
                mat_smiles = smiles_subset.loc[mat_name, :]

                if proportions:
                    mat_smiles = mat_smiles.multiply(proportion_val)

                combined_series = pd.concat([combined_series, mat_smiles])

            combined_series = combined_series.reset_index(drop=True)
            formulation_smiles = pd.concat(
                [formulation_smiles, combined_series.to_frame().T], ignore_index=True
            )

        formulation_smiles = formulation_smiles.fillna(0)
        all_formulations = pd.concat([all_formulations, formulation_smiles], axis=1)
        # Pad to ensure at least 12288 columns
        if all_formulations.shape[1] < 12288:
            needed = 12288 - all_formulations.shape[1]
            shape = (all_formulations.shape[0], needed)
            all_formulations = pd.concat([all_formulations, pd.DataFrame(np.zeros(shape))], axis=1)

    if powder is not None:
        if y is None:
            raise ValueError("Must provide 'y' when using powder features.")
        base_path = os.path.dirname(os.path.realpath(__file__))
        full_path = os.path.join(base_path, "..", "data_sets", "sls_smiles_data.xlsx")
        powders_data = pd.read_excel(full_path, sheet_name="particles")

        powders_data.index = powders_data["Material"]
        powders_subset = powders_data.loc[:, "D10":"filtered"].copy()

        if powder_range == "all":
            powders_subset.drop(columns=["range"], inplace=True)
        elif powder_range == "median":
            powders_subset.drop(columns=["D10", "D90", "range"], inplace=True)
        elif powder_range == "range":
            powders_subset.drop(columns=["D10", "D90"], inplace=True)
        else:
            msg = "powder_range must be one of {'all', 'median', 'range'}."
            raise ValueError(msg)

        formulation_powder = pd.DataFrame()
        for _, row in data_sls.iterrows():
            non_na = row.dropna()
            combined = pd.Series(dtype=float)
            for j in range(len(non_na)):
                mat_key = non_na.index[j]
                mat_powder = powders_subset.loc[mat_key, :]
                combined = pd.concat([combined, mat_powder])
            combined = combined.reset_index(drop=True)
            formulation_powder = pd.concat(
                [formulation_powder, combined.to_frame().T], ignore_index=True
            )

        formulation_powder = formulation_powder.fillna(0)
        x_tr, x_ts, y_tr, _ = train_test_split(
            formulation_powder, y, test_size=0.2, random_state=seed
        )
        scaler = MinMaxScaler()
        scaler.fit(x_tr)
        powder_scaled = pd.DataFrame(scaler.transform(formulation_powder))

        all_formulations = pd.concat([all_formulations, powder_scaled], axis=1)

        if original:
            data_sls_nonan = data_sls.fillna(0)
            all_formulations = pd.concat([all_formulations, data_sls_nonan], axis=1)

    return all_formulations.to_numpy()


if __name__ == "__main__":
    # For demonstration, creating a dummy DataFrame and labels
    dummy_data = pd.DataFrame({"MaterialA": [50, 0, 25], "MaterialB": [50, 100, 75]})
    y_dummy = np.array([1, 0, 1])

    table = make_formulations(
        data=dummy_data, proportions=True, original=False, mfp="new", y=y_dummy
    )

    print(pd.DataFrame(table))
    print(table.shape)
