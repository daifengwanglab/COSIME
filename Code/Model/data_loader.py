import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def load_and_prepare_data(batch_size, log_path, data1_path, data2_path, task_type='binary', **kwargs):
    """
    Load and prepare data for training and evaluation.
    
    :param batch_size: Batch size for the DataLoader.
    :param output: Unused in current function, may be for logging.
    :param name: Unused in current function, may be for logging.
    :param data1_path: Path to the first CSV file (Dataset A).
    :param data2_path: Path to the second CSV file (Dataset B).
    :param task_type: The type of task ('binary' or 'continuous'). Determines how labels are processed.
    :return: DataLoader for training, validation, and holdout sets for both datasets (A and B).
    """
    
    if task_type == 'continuous':
        # Load data from CSV files for continuous task
        data1 = pd.read_csv(data1_path, header=0, index_col=0)
        data2 = pd.read_csv(data2_path, header=0, index_col=0)
        
        # Extract features and labels
        data_A = data1.drop(columns=['y']).values
        data_B = data2.drop(columns=['y']).values
        labels_A = data1.loc[:, data1.columns == 'y'].values
        labels_B = data2.loc[:, data2.columns == 'y'].values
        
        # Scaling for continuous data
        scaler_A = StandardScaler()
        data_A_scaled = scaler_A.fit_transform(data_A)
        
        scaler_B = StandardScaler()
        data_B_scaled = scaler_B.fit_transform(data_B)
    
    else:
        # Load data from CSV files for binary task
        data1 = pd.read_csv(data1_path, header=0, index_col=0)
        data2 = pd.read_csv(data2_path, header=0, index_col=0)
        
        # Extract features and labels
        data_A = data1.drop(columns=['y']).values
        data_B = data2.drop(columns=['y']).values
        labels_A = data1.loc[:, data1.columns == 'y'].values
        labels_B = data2.loc[:, data2.columns == 'y'].values
        
        # Scaling for binary data (MinMaxScaler or StandardScaler)
        scaler_A = MinMaxScaler()
        data_A_scaled = scaler_A.fit_transform(data_A)
        
        scaler_B = MinMaxScaler()
        data_B_scaled = scaler_B.fit_transform(data_B)
    
    # Split the data into training, validation, and holdout sets
    train_indices_A_initial, holdout_indices_A = train_test_split(np.arange(len(data_A_scaled)), test_size=0.25, random_state=200, shuffle=True)
    train_indices_B_initial, holdout_indices_B = train_test_split(np.arange(len(data_B_scaled)), test_size=0.25, random_state=200, shuffle=True)

    train_indices_A, val_indices_A = train_test_split(train_indices_A_initial, test_size=0.2, shuffle=True)
    train_indices_B, val_indices_B = train_test_split(train_indices_B_initial, test_size=0.2, shuffle=True)

    # Create datasets for each split
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

    # Convert NumPy arrays to PyTorch tensors
    train_data_A_tensor = torch.tensor(train_data_A, dtype=torch.float32)
    train_labels_A_tensor = torch.tensor(train_labels_A, dtype=torch.float32 if task_type == 'continuous' else torch.int64)
    
    val_data_A_tensor = torch.tensor(val_data_A, dtype=torch.float32)
    val_labels_A_tensor = torch.tensor(val_labels_A, dtype=torch.float32 if task_type == 'continuous' else torch.int64)
    
    holdout_data_A_tensor = torch.tensor(holdout_data_A, dtype=torch.float32)
    holdout_labels_A_tensor = torch.tensor(holdout_labels_A, dtype=torch.float32 if task_type == 'continuous' else torch.int64)

    train_data_B_tensor = torch.tensor(train_data_B, dtype=torch.float32)
    train_labels_B_tensor = torch.tensor(train_labels_B, dtype=torch.float32 if task_type == 'continuous' else torch.int64)
    
    val_data_B_tensor = torch.tensor(val_data_B, dtype=torch.float32)
    val_labels_B_tensor = torch.tensor(val_labels_B, dtype=torch.float32 if task_type == 'continuous' else torch.int64)
    
    holdout_data_B_tensor = torch.tensor(holdout_data_B, dtype=torch.float32)
    holdout_labels_B_tensor = torch.tensor(holdout_labels_B, dtype=torch.float32 if task_type == 'continuous' else torch.int64)

    # Create DataLoader objects
    train_loader_A = DataLoader(TensorDataset(train_data_A_tensor, train_labels_A_tensor), batch_size=batch_size, shuffle=True)
    val_loader_A = DataLoader(TensorDataset(val_data_A_tensor, val_labels_A_tensor), batch_size=batch_size)
    holdout_loader_A = DataLoader(TensorDataset(holdout_data_A_tensor, holdout_labels_A_tensor), batch_size=batch_size)

    train_loader_B = DataLoader(TensorDataset(train_data_B_tensor, train_labels_B_tensor), batch_size=batch_size, shuffle=True)
    val_loader_B = DataLoader(TensorDataset(val_data_B_tensor, val_labels_B_tensor), batch_size=batch_size)
    holdout_loader_B = DataLoader(TensorDataset(holdout_data_B_tensor, holdout_labels_B_tensor), batch_size=batch_size)

    return train_loader_A, val_loader_A, holdout_loader_A, train_loader_B, val_loader_B, holdout_loader_B
