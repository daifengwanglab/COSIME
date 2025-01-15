import pandas as pd

def load_data(*datasets, **kwargs):
    processed_datasets = []
    for data_path in datasets:
        data = pd.read_csv(data_path, index_col=0)
        X, y = data.drop(columns='y'), data['y']

        # Apply any necessary preprocessing here
        # TODO

        # Append
        processed_datasets.append((X, y))

    return processed_datasets
