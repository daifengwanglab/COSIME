import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def load_and_prepare_data(batch_size, data1_path, data2_path):
    """Load, preprocess and return DataLoader objects for the datasets"""

    # 1. Load data from CSV files
    data1 = pd.read_csv(data1_path, header=0, sep=',', encoding='utf-8')
    data2 = pd.read_csv(data2_path, header=0, sep=',', encoding='utf-8')

    # 2. Shuffle the data
    data1 = data1.sample(frac=1).reset_index(drop=True)
    data2 = data2.sample(frac=1).reset_index(drop=True)

    # Align the second dataset (data2) based on the first dataset's indices
    data2 = data2.loc[data1.index]

    # 3. Extract features and labels
    data_A = data1.drop(columns=['y']).values # your labels
    data_B = data2.drop(columns=['y']).values # your labels

    labels_A = (data1['y'].values == 1).astype(np.int64)
    labels_B = (data2['y'].values == 1).astype(np.int64)

    # 4. Preprocessing - Min-max scaling and standardization
    scaler_A = MinMaxScaler()
    data_A_scaled = scaler_A.fit_transform(data_A)

    scaler_B = MinMaxScaler()
    data_B_scaled = scaler_B.fit_transform(data_B)

    # 5. Split the data into training, validation, and holdout sets
    train_indices_A_initial, holdout_indices_A = train_test_split(np.arange(len(data_A_scaled)), test_size=0.25, random_state=100, shuffle=True)
    train_indices_B_initial, holdout_indices_B = train_test_split(np.arange(len(data_B_scaled)), test_size=0.25, random_state=100, shuffle=True)

    train_indices_A, val_indices_A = train_test_split(train_indices_A_initial, test_size=100, shuffle=True)
    train_indices_B, val_indices_B = train_test_split(train_indices_B_initial, test_size=100, shuffle=True)

    # 6. Create training, validation, and holdout datasets
    train_data_A = data_A_scaled[train_indices_A]
    val_data_A = data_A_scaled[val_indices_A]
    holdout_data_A = data_A_scaled[holdout_indices_A]

    train_labels_A = labels_A[train_indices_A]
    val_labels_A = labels_A[val_indices_A]
    holdout_labels_A = labels_A[holdout_indices_A]

    train_data_B = data_B_scaled[train_indices_B]
    val_data_B = data_B_scaled[val_indices_B]
    holdout_data_B = data_B_scaled[holdout_indices_B]

    train_labels_B = labels_B[train_indices_B]
    val_labels_B = labels_B[val_indices_B]
    holdout_labels_B = labels_B[holdout_indices_B]

    # 7. Convert NumPy arrays to PyTorch tensors
    train_data_A_tensor = torch.tensor(train_data_A, dtype=torch.float32)
    train_labels_A_tensor = torch.tensor(train_labels_A, dtype=torch.int64)
    val_data_A_tensor = torch.tensor(val_data_A, dtype=torch.float32)
    val_labels_A_tensor = torch.tensor(val_labels_A, dtype=torch.int64)
    holdout_data_A_tensor = torch.tensor(holdout_data_A, dtype=torch.float32)
    holdout_labels_A_tensor = torch.tensor(holdout_labels_A, dtype=torch.int64)

    train_data_B_tensor = torch.tensor(train_data_B, dtype=torch.float32)
    train_labels_B_tensor = torch.tensor(train_labels_B, dtype=torch.int64)
    val_data_B_tensor = torch.tensor(val_data_B, dtype=torch.float32)
    val_labels_B_tensor = torch.tensor(val_labels_B, dtype=torch.int64)
    holdout_data_B_tensor = torch.tensor(holdout_data_B, dtype=torch.float32)
    holdout_labels_B_tensor = torch.tensor(holdout_labels_B, dtype=torch.int64)

    # 8. Create DataLoader objects for training, validation, and holdout datasets
    train_loader_A = DataLoader(TensorDataset(train_data_A_tensor, train_labels_A_tensor), batch_size=batch_size, shuffle=True)
    val_loader_A = DataLoader(TensorDataset(val_data_A_tensor, val_labels_A_tensor), batch_size=batch_size)
    holdout_loader_A = DataLoader(TensorDataset(holdout_data_A_tensor, holdout_labels_A_tensor), batch_size=batch_size)

    train_loader_B = DataLoader(TensorDataset(train_data_B_tensor, train_labels_B_tensor), batch_size=batch_size, shuffle=True)
    val_loader_B = DataLoader(TensorDataset(val_data_B_tensor, val_labels_B_tensor), batch_size=batch_size)
    holdout_loader_B = DataLoader(TensorDataset(holdout_data_B_tensor, holdout_labels_B_tensor), batch_size=batch_size)

    return (train_loader_A, val_loader_A, holdout_loader_A), (train_loader_B, val_loader_B, holdout_loader_B)
