# data_loader.py

import pandas as pd

def load_data(input_data_1, input_data_2):
    data_1 = pd.read_csv(input_data_1)
    data_2 = pd.read_csv(input_data_2)

    # Apply any necessary preprocessing here
    # For example: normalization, missing value imputation, etc.
    return data_1, data_2
