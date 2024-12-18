import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

warnings.simplefilter(action="ignore", category=FutureWarning)


def make_formulations(
    data,
    smiles=True,
    proportions=True,
    powder=None,
    powder_range="median",
    embedding=False,
    seed=42,
    original=False,
    mfp="old",
):
    data_sls = data
    all_formulations = pd.DataFrame([])

    if embedding:
        all_formulations = []
        materials = {name: index for index, name in enumerate(list(data_sls.columns))}
        for i, row in data_sls.iterrows():
            non_nan_cols = row.fillna(0)
            formulation = []
            for j in range(len(non_nan_cols)):
                key = non_nan_cols.index[j]
                value = non_nan_cols[j] * 0.01
                formulation.append([materials[key], value])
            all_formulations.append(formulation)
        return {"x": np.array(all_formulations), "y": np.array(y)}
    else:
        pass

    if smiles:
        if mfp == "old":
            sheet = "smiles"
            end = 2047
        elif mfp == "new":
            sheet = "1024_smiles"
            end = 1023
        smile = pd.read_excel(
            f"{os.path.dirname(os.path.realpath(__file__))}/../data_sets/sls_smiles_data.xlsx",
            sheet_name=sheet,
        )

        materials = [material.strip() for material in list(smile["Material"])]
        smile.index = materials
        smiles = smile.loc[:, 0:end]

        formulation_smiles = pd.DataFrame([])

        for i, row in data_sls.iterrows():
            non_nan_cols = row[row != 0]
            formulation = pd.Series([])
            for j in range(len(non_nan_cols)):
                key = non_nan_cols.index[j].strip()
                value = non_nan_cols[j] * 0.01
                drug_smiles = smiles.loc[key, :]
                if proportions:
                    drug_smiles = drug_smiles.multiply(other=value)
                formulation = pd.concat([formulation, drug_smiles])
            formulation = formulation.reset_index(drop=True)
            formulation_smiles = formulation_smiles.append(
                formulation, ignore_index=True
            )
        formulation_smiles = formulation_smiles.fillna(0)
        all_formulations = pd.concat([all_formulations, formulation_smiles], axis=1)

        if all_formulations.shape[1] < 12288:
            diff = 12288 - all_formulations.shape[1]
            shape = (all_formulations.shape[0], diff)
            zeros = pd.DataFrame(np.zeros(shape=shape))
            all_formulations = pd.concat([all_formulations, zeros], axis=1)

    if powder is not None:
        powders = pd.read_excel(
            f"{os.path.dirname(os.path.realpath(__file__))}/../data_sets/sls_smiles_data.xlsx",
            sheet_name="particles",
        )
        powders.index = powders["Material"]
        powders = powders.loc[:, "D10":"filtered"]
        if powder_range == "all":
            powders = powders.drop(["range"], axis=1)
        elif powder_range == "median":
            powders = powders.drop(["D10", "D90", "range"], axis=1)
        elif powder_range == "range":
            powders = powders.drop(["D10", "D90"], axis=1)

        formulation_powder = pd.DataFrame([])

        for i, row in data_sls.iterrows():
            non_nan_cols = row.dropna()
            formulation = pd.Series([])
            for j in range(len(non_nan_cols)):
                key = non_nan_cols.index[j]
                powder = powders.loc[key, :]
                formulation = pd.concat([formulation, powder])
            formulation = formulation.reset_index(drop=True)
            formulation_powder = formulation_powder.append(
                formulation, ignore_index=True
            )
        formulation_powder = formulation_powder.fillna(0)
        scaler = MinMaxScaler()
        x_train, x_test, y_train, y_test = train_test_split(
            formulation_powder, y, test_size=0.2, random_state=seed
        )
        scaler.fit(x_train)
        formulation_powder = pd.DataFrame(scaler.transform(formulation_powder))
        all_formulations = pd.concat([all_formulations, formulation_powder], axis=1)

        if original:
            data_sls = data_sls.fillna(0)
            all_formulations = pd.concat([all_formulations, data_sls], axis=1)

    return np.array(all_formulations)


if __name__ == "__main__":
    table = make_formulations(proportions=True, original=False, mfp="new")
    table = pd.DataFrame(table["x"])
    print(table)
    print(table.shape)
