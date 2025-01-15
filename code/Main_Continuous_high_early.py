
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

import ray
from ray import tune

import os

# # Initialize Ray (if not already initialized)
# ray.init()

# Define paths
name = 'Continuous_high_early'

dir = '/Users/jeromechoi/Documents/jerome/Documents/WISC/BMI/Daifeng Wang/Cooperative learning/Coop_DOT/Results/Simluated/new_111624/Continuous_high_early_test/'
output = '/Users/jeromechoi/Documents/jerome/Documents/WISC/BMI/Daifeng Wang/Cooperative learning/Coop_DOT/Results/Simluated/new_111624/Continuous_high_early_test/'

# dir = '/project8/Dissertation_JC/Coop_DOT/Simulation/111624/data/'
# output = '/project8/Dissertation_JC/Coop_DOT/Simulation/111624/Continuous_high_early/'

data1_path = dir + 'continuous_high_Early_x1.csv'
data2_path =  dir + 'continuous_high_Early_x2.csv'
model_path = output + name + '.pt'
history_path = output  + name + '_history.csv'
plot1_path = output  + name + '_train_losses.png'
plot2_path = output +  name + '_validation_losses.png'
plot3_path = output + name + '_eval_losses.png'
plot4_path = output + name + '_MSE.png'
MSE_path = output + name + '_MSE.csv'
testset_path = output + name + '_testset.csv'
interaction_path = output + name + '_interaction.csv'
feature_importance_path = output + name + '_FI.csv'
train_data_path = output
holdout_data_path = output

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import sys
sys.argv=['']
del sys

# %%
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=300, type=int)
args = vars(parser.parse_args())

# %%
epochs = args['epochs']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
data1 = pd.read_csv(data1_path, header=0,index_col=0)
data2 = pd.read_csv(data2_path,header=0,index_col=0)

# %%
data_A = data1.drop(columns=['y']).values
data_B = data2.drop(columns=['y']).values

# Min-max scaling
scaler_A = MinMaxScaler()
data_A_scaled = scaler_A.fit_transform(data_A)

scaler_B = MinMaxScaler()
data_B_scaled = scaler_B.fit_transform(data_B)

# Standardization
scaler_A = StandardScaler()
data_A = scaler_A.fit_transform(data_A)

scaler_B = StandardScaler()
data_B = scaler_B.fit_transform(data_B)

labels_A = data1['y'].values
labels_B = data2['y'].values

print(f"labels_A shape: {labels_A.shape}")  # Should be (num_samples,)
print(f"labels_B shape: {labels_B.shape}")  # Should be (num_samples,)

print(f"data_A shape: {data_A.shape}")
print(f"data_B shape: {data_B.shape}")

# Define KL divergence loss function
def KL_divergence(mu, logsigma):
    loss = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
    return loss

# Optimal trasnport
def LOT(mu_src, std_src, mu_tgt, std_tgt, reg=0.1, reg_m=1.0, num_iterations=10, device='cpu', 
        source_weights=None, target_weights=None, idx_q=None, idx_r=None, 
        transport_plan=None, LOT_batch_size=None, domain_regularization=False):
    """
    Learnable Optimal Transport (LOT) function to compute the OT loss between two Gaussian distributions, with improvements.
    
    Args:
    - mu_src, std_src: Means and standard deviations for the source distribution (distribution 1).
    - mu_tgt, std_tgt: Means and standard deviations for the target distribution (distribution 2).
    - reg: Regularization parameter for Sinkhorn's algorithm.
    - reg_m: Scaling factor for the regularization term in the dual variables update.
    - num_iterations: Number of iterations for Sinkhorn's algorithm.
    - device: Device to run the calculations on ('cpu' or 'cuda').
    - source_weights, target_weights: Optional weights for the source and target distributions.
    - idx_q, idx_r: Optional indices for weighting.
    - transport_plan: Optional initial transport plan.
    - LOT_batch_size: If provided, will compute the transport plan in mini-batches for scalability.
    - domain_regularization: Whether to apply domain-specific regularization (e.g., entropy regularization).
    
    Returns:
    - ot_loss: The computed optimal transport loss.
    - transport_plan: The transport plan matrix.
    """

    # Number of elements in the source and target distributions
    n_src = mu_src.size(0)  # Number of samples in source distribution
    n_tgt = mu_tgt.size(0)  # Number of samples in target distribution

    # Step 1: Check for NaNs or Infinities in inputs and replace them with defaults
    mu_src = torch.nan_to_num(mu_src, nan=0.0, posinf=0.0, neginf=0.0)
    std_src = torch.nan_to_num(std_src, nan=1e-6, posinf=1e-6, neginf=1e-6)  # std instead of var
    mu_tgt = torch.nan_to_num(mu_tgt, nan=0.0, posinf=0.0, neginf=0.0)
    std_tgt = torch.nan_to_num(std_tgt, nan=1e-6, posinf=1e-6, neginf=1e-6)  # std instead of var

    # Ensure standard deviations are not zero (to avoid division by zero issues)
    std_src = torch.clamp(std_src, min=1e-6)  
    std_tgt = torch.clamp(std_tgt, min=1e-6)

    # Step 2: Handle the weights for the distributions (source_weights, target_weights)
    if source_weights is None:
        weights_src = torch.ones(n_src, 1) / n_src  # Uniform distribution over source
    else:
        query_batch_weight = source_weights[idx_q] if idx_q is not None else source_weights
        weights_src = query_batch_weight / torch.sum(query_batch_weight)

    if target_weights is None:
        weights_tgt = torch.ones(n_tgt, 1) / n_tgt  # Uniform distribution over target
    else:
        ref_batch_weight = target_weights[idx_r] if idx_r is not None else target_weights
        weights_tgt = ref_batch_weight / torch.sum(ref_batch_weight)

    weights_src = weights_src.to(device)
    weights_tgt = weights_tgt.to(device)

    # Step 3: Initialize transport plan (learnable transport plan)
    if transport_plan is None:
        transport_plan = torch.ones(n_src, n_tgt) / (n_src * n_tgt)
        transport_plan = transport_plan.to(device)
    
    transport_plan = torch.nn.Parameter(transport_plan, requires_grad=True)  # Make transport plan learnable

    # Step 4: Initialize dual variables
    dual_vars = (torch.ones(n_src, 1) / n_src).to(device)
    dual_update_factor = reg_m / (reg_m + reg)  # Scaling factor for dual variables update

    # Step 5: Perform mini-batch processing
    if LOT_batch_size is None:
        # If no mini-batch size is provided, process the entire dataset in one batch
        batches = [(mu_src, std_src, mu_tgt, std_tgt)]
    else:
        # Split the data into mini-batches
        batches = [(mu_src[i:i+LOT_batch_size], std_src[i:i+LOT_batch_size], mu_tgt[i:i+LOT_batch_size], std_tgt[i:i+LOT_batch_size]) 
                   for i in range(0, n_src, LOT_batch_size)]

    # Step 6: Iterative optimization of the transport plan using Sinkhorn's algorithm
    for m in range(num_iterations):
        for mu_batch, std_batch, mu_tgt_batch, std_tgt_batch in batches:
            # Compute pairwise distances for the mini-batch
            dist_mu = torch.cdist(mu_batch.unsqueeze(0), mu_tgt_batch.unsqueeze(0), p=2).squeeze(0)
            dist_std = torch.cdist(std_batch.unsqueeze(0), std_tgt_batch.unsqueeze(0), p=2).squeeze(0)

            # Compute the total cost matrix (mean + standard deviation)
            cost_matrix = dist_mu + dist_std + 1e-6  # Adding small constant for numerical stability

            # Compute the transport kernel and scaling factors
            transport_kernel = torch.exp(-cost_matrix / (reg * torch.max(torch.abs(cost_matrix)))) * transport_plan
            scaling_factors = weights_tgt / (torch.t(transport_kernel) @ dual_vars)

            # Dual variables update
            for i in range(10):
                dual_vars = (weights_src / (transport_kernel @ scaling_factors)) ** dual_update_factor
                scaling_factors = (weights_tgt / (torch.t(transport_kernel) @ dual_vars)) ** dual_update_factor

            # Update the transport plan based on the dual variables
            transport_plan = (dual_vars @ torch.t(scaling_factors)) * transport_kernel

            # Domain-specific regularization (entropy regularization)
            if domain_regularization:
                entropy_reg = -torch.sum(transport_plan * torch.log(transport_plan + 1e-6))
                transport_plan = transport_plan - reg * entropy_reg

    # Step 7: Handle NaN in transport plan (reset to uniform if NaN)
    if torch.isnan(transport_plan).sum() > 0:
        transport_plan = torch.ones(n_src, n_tgt) / (n_src * n_tgt)
        transport_plan = transport_plan.to(device)

    # Step 8: Compute the final optimal transport loss
    ot_loss = (cost_matrix * transport_plan.detach()).sum()

    return ot_loss


# Custom dataset class
class CustomData(Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.data = transform(data) if transform else data
        self.data_labels = target_transform(labels) if target_transform else labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.data_labels[idx]

# Load data function
def load_data(batch_size, data, labels, train_idx=None, val_idx=None):
    if train_idx is not None:
        train_data = data[train_idx]
        train_labels = labels[train_idx]
        train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        train_loader = None

    if val_idx is not None:
        val_data = data[val_idx]
        val_labels = labels[val_idx]
        val_dataset = TensorDataset(torch.tensor(val_data, dtype=torch.float32), torch.tensor(val_labels, dtype=torch.float32))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None

    return train_loader, val_loader

# Define remote function for data loading and preparation
@ray.remote
def load_and_prepare_data(batch_size, output, name, data1_path, data2_path):
    # Load data from CSV files
    data1 = pd.read_csv(data1_path, header=0,index_col=0)
    data2 = pd.read_csv(data2_path,header=0,index_col=0)

    # data2 = data2.loc[data1.index]

    # Extract features and labels
    data_A = data1.drop(columns=['y']).values
    data_B = data2.drop(columns=['y']).values

    labels_A = data1.loc[:, data1.columns=='y'].values
    labels_B = data2.loc[:, data2.columns=='y'].values

    # Preprocessing - Min-max scaling and standardization
    # scaler_A = MinMaxScaler()
    # data_A_scaled = scaler_A.fit_transform(data_A)

    # scaler_B = MinMaxScaler()
    # data_B_scaled = scaler_B.fit_transform(data_B)

    scaler_A_std = StandardScaler()
    data_A_std = scaler_A_std.fit_transform(data_A)

    scaler_B_std = StandardScaler()
    data_B_std = scaler_B_std.fit_transform(data_B)

    # Split the data into training, validation, and holdout sets
    train_indices_A_initial, holdout_indices_A = train_test_split(np.arange(len(data_A_std)), test_size=0.25, random_state=200, shuffle = True)
    train_indices_B_initial, holdout_indices_B = train_test_split(np.arange(len(data_B_std)), test_size=0.25, random_state=200, shuffle = True)

    train_indices_A, val_indices_A = train_test_split(train_indices_A_initial, test_size=0.2, shuffle = True)
    train_indices_B, val_indices_B = train_test_split(train_indices_B_initial, test_size=0.2, shuffle = True)

    train_data_A = data_A_std[train_indices_A]
    val_data_A = data_A_std[val_indices_A]
    holdout_data_A = data_A_std[holdout_indices_A]

    train_labels_A = labels_A[train_indices_A]
    val_labels_A = labels_A[val_indices_A]
    holdout_labels_A = labels_A[holdout_indices_A]

    train_data_B = data_B_std[train_indices_B]
    val_data_B = data_B_std[val_indices_B]
    holdout_data_B = data_B_std[holdout_indices_B]

    train_labels_B = labels_B[train_indices_B]
    val_labels_B = labels_B[val_indices_B]
    holdout_labels_B = labels_B[holdout_indices_B]

    # Convert NumPy arrays to PyTorch tensors
    train_data_A_tensor = torch.tensor(train_data_A, dtype=torch.float32)
    train_labels_A_tensor = torch.tensor(train_labels_A, dtype=torch.float64)
    val_data_A_tensor = torch.tensor(val_data_A, dtype=torch.float32)
    # val_labels_A_tensor = torch.tensor(val_labels_A, dtype=torch.int32)
    val_labels_A_tensor = torch.tensor(val_labels_A)
    holdout_data_A_tensor = torch.tensor(holdout_data_A, dtype=torch.float32)
    holdout_labels_A_tensor = torch.tensor(holdout_labels_A, dtype=torch.float64)

    train_data_B_tensor = torch.tensor(train_data_B, dtype=torch.float32)
    train_labels_B_tensor = torch.tensor(train_labels_B, dtype=torch.int64)
    val_data_B_tensor = torch.tensor(val_data_B, dtype=torch.float32)
    val_labels_B_tensor = torch.tensor(val_labels_B, dtype=torch.int64)
    holdout_data_B_tensor = torch.tensor(holdout_data_B, dtype=torch.float32)
    holdout_labels_B_tensor = torch.tensor(holdout_labels_B, dtype=torch.float64)

    # print("val_labels_A_tensor:")
    # print(val_labels_A_tensor)
    # print("Shape:", val_labels_A_tensor.shape)

    # Create DataLoader objects
    train_loader_A = DataLoader(TensorDataset(train_data_A_tensor, train_labels_A_tensor), batch_size=batch_size, shuffle=True)
    val_loader_A = DataLoader(TensorDataset(val_data_A_tensor, val_labels_A_tensor), batch_size=batch_size)
    holdout_loader_A = DataLoader(TensorDataset(holdout_data_A_tensor, holdout_labels_A_tensor), batch_size=batch_size)

    train_loader_B = DataLoader(TensorDataset(train_data_B_tensor, train_labels_B_tensor), batch_size=batch_size, shuffle=True)
    val_loader_B = DataLoader(TensorDataset(val_data_B_tensor, val_labels_B_tensor), batch_size=batch_size)
    holdout_loader_B = DataLoader(TensorDataset(holdout_data_B_tensor, holdout_labels_B_tensor), batch_size=batch_size)

    return {
        'train_loader_A': DataLoader(TensorDataset(train_data_A_tensor, train_labels_A_tensor), batch_size=batch_size, shuffle=True),
        'val_loader_A': DataLoader(TensorDataset(val_data_A_tensor, val_labels_A_tensor), batch_size=batch_size),
        'holdout_loader_A': DataLoader(TensorDataset(holdout_data_A_tensor, holdout_labels_A_tensor), batch_size=batch_size),
        'train_loader_B': DataLoader(TensorDataset(train_data_B_tensor, train_labels_B_tensor), batch_size=batch_size, shuffle=True),
        'val_loader_B': DataLoader(TensorDataset(val_data_B_tensor, val_labels_B_tensor), batch_size=batch_size),
        'holdout_loader_B': DataLoader(TensorDataset(holdout_data_B_tensor, holdout_labels_B_tensor), batch_size=batch_size),
        'train_data_A_tensor': train_data_A_tensor,
        'train_labels_A_tensor': train_labels_A_tensor,
        'val_data_A_tensor': val_data_A_tensor,
        'val_labels_A_tensor': val_labels_A_tensor,
        'train_data_B_tensor': train_data_B_tensor,
        'train_labels_B_tensor': train_labels_B_tensor,
        'val_data_B_tensor': val_data_B_tensor,
        'val_labels_B_tensor': val_labels_B_tensor,
        'train_indices_A': train_indices_A,
        'train_indices_A_initial': train_indices_A_initial,
        'val_indices_A': val_indices_A,
        'train_indices_B': train_indices_B,
        'train_indices_B_initial': train_indices_B_initial,
        'val_indices_B': val_indices_B,
        'holdout_indices_A': holdout_indices_A,
        'holdout_indices_B': holdout_indices_B,
        'holdout_data_A_tensor': holdout_data_A_tensor,
        'holdout_data_B_tensor': holdout_data_B_tensor,
        'holdout_labels_A_tensor': holdout_labels_A_tensor,
        'holdout_labels_B_tensor': holdout_labels_B_tensor
    }

# Define model class
class Model(nn.Module):
    def __init__(self, *input_dims, dim, dropout):
        super(Model, self).__init__()

        # Check for input dims
        assert len(input_dims) > 0, 'Must provide at least one input dim.'

        self.num_modalities = len(input_dims)
        self.dropout = dropout

        self.layers = nn.ModuleList([
            nn.Sequential(
                # nn.Sequential(
                # nn.Linear(input_dim, int(input_dim * 2)),
                # nn.BatchNorm1d(int(input_dim * 2)),
                # nn.LeakyReLU(),

                # nn.Linear(int(input_dim * 2), int(input_dim * 1)),
                # nn.BatchNorm1d(int(input_dim * 1)),
                # nn.LeakyReLU(),

                nn.Linear(int(input_dim * 1), int(input_dim * 0.5)),
                nn.BatchNorm1d(int(input_dim * 0.5)),
                nn.LeakyReLU(),
                nn.Dropout(dropout),

                # nn.Linear(int(input_dim * 0.75), int(input_dim * 0.5)),
                # nn.BatchNorm1d(int(input_dim * 0.5)),
                # nn.LeakyReLU(),

                # nn.Linear(input_dim * 3, input_dim * 2),
                # nn.BatchNorm1d(input_dim * 2),
                # nn.LeakyReLU(),

                # nn.Linear(input_dim * 2, input_dim),
                # nn.BatchNorm1d(input_dim),
                # nn.LeakyReLU(),

                # nn.Linear(input_dim, int(input_dim//2)),
                # nn.BatchNorm1d(int(input_dim//2)),
                # nn.LeakyReLU(),

                nn.Linear(int(input_dim*0.5), dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout)
                )
            
            for input_dim in input_dims
        ])

        # Adding residual connections
        self.residual_layers = nn.ModuleList([
            nn.Linear(input_dim, dim)  # To match the output dimensions if needed
            for input_dim in input_dims
        ])
        
        self.mu = nn.ModuleList([
            nn.Linear(dim, dim)
            for _ in input_dims
        ])

        self.logsigma = nn.ModuleList([
            nn.Linear(dim, dim)
            for _ in input_dims
        ])

        # self.Net = nn.Sequential(
        #      torch.nn.Linear(int(dim), 1),
        #     )

        self.Net = nn.Sequential(

             # torch.nn.Linear(dim, int(dim*0.5)),
             # torch.nn.BatchNorm1d(int(dim*0.5)),
             # nn.LeakyReLU(),

             # torch.nn.Linear(int(dim*0.75), int(dim*0.5)),
             # torch.nn.BatchNorm1d(int(dim*0.5)),
             # nn.LeakyReLU(),

             # torch.nn.Linear(int(dim*0.6), int(dim*0.4)),
             # torch.nn.BatchNorm1d(int(dim*0.4)),
             # nn.LeakyReLU(),
                          
             # torch.nn.Linear(int(dim), int(dim*0.5)),
             # torch.nn.BatchNorm1d(int(dim*0.5)),
             # nn.LeakyReLU(),

             nn.Linear(int(dim), 1),
            )
    
    def net_forward(self, *X):
        def process(x, layer):
            # x = self.layers[layer](x)
            # x = x.view(x.size(0), -1)
            # mu = self.mu[layer](x)
            # logsigma = self.logsigma[layer](x)
            # z = self.gaussian_sampler(mu, logsigma)

            x_out = self.layers[layer](x)

            # Add residual connection
            residual = self.residual_layers[layer](x)  # Match input dimensions
            x_out = x_out + residual  # Residual addition

            x_out = x_out.view(x_out.size(0), -1)  # Flatten
            mu = self.mu[layer](x_out)
            logsigma = self.logsigma[layer](x_out)
            z = self.gaussian_sampler(mu, logsigma)

            return z, mu, logsigma

        return [process(x, layer) for layer, x in enumerate(X)]

    def gaussian_sampler(self, mu, logsigma):
        if self.training:
            logsigma = torch.clamp(logsigma, -10, 10)  # Clip to a reasonable range
            std = torch.exp(logsigma / 2)
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def NN_forward(self, X):
        return self.Net(X)
    
    def forward(self, *X):
        # [[0,1], [1,3], ]
        # Convert input tensors to Float if they are not already
        X = [x.float() for x in X]

        # Obtain outputs from net_forward
        net_logits = self.net_forward(*X)

        # Ensure net_logits is a list of tuples
        if isinstance(net_logits, list) and all(isinstance(item, tuple) for item in net_logits):
            # Extract tensors from each tuple in the list
            tensors = [item[0] for item in net_logits]

            # Stack the tensors along a new dimension
            stacked_logits = torch.stack(tensors, dim=0)  # Shape [num_models, batch_size, num_features]

            # Average the logits along the new dimension
            averaged_logits = stacked_logits.mean(dim=0)  # Shape [batch_size, num_features]
        else:
            raise TypeError("net_logits must be a list of tuples containing tensors")

        # Pass through logistic forward
        NN_logits = self.NN_forward(averaged_logits)

        # Return converted tensors
        return net_logits, NN_logits

# Function to load holdout data
def load_holdout_data(holdout_data_path, name):
    holdout_data_A = np.loadtxt(holdout_data_path + name + '_holdout_data_A.csv', delimiter=',')
    holdout_data_B = np.loadtxt(holdout_data_path + name + '_holdout_data_B.csv', delimiter=',')
    holdout_labels_A = np.loadtxt(holdout_data_path + name + '_holdout_labels_A.csv', delimiter=',')
    holdout_labels_B = np.loadtxt(holdout_data_path + name + '_holdout_labels_B.csv', delimiter=',')
    holdout_tensor_data_A = torch.tensor(holdout_data_A, dtype=torch.float32)
    holdout_tensor_data_B = torch.tensor(holdout_data_B, dtype=torch.float32)
    holdout_tensor_labels_A = torch.tensor(holdout_labels_A, dtype=torch.float32)
    holdout_tensor_labels_B = torch.tensor(holdout_labels_B, dtype=torch.float32)
    holdout_dataset_A = TensorDataset(holdout_tensor_data_A, holdout_tensor_labels_A)
    holdout_dataset_B = TensorDataset(holdout_tensor_data_B, holdout_tensor_labels_B)
    holdout_loader_A = DataLoader(holdout_dataset_A, batch_size=batch_size, shuffle=False)
    holdout_loader_B = DataLoader(holdout_dataset_B, batch_size=batch_size, shuffle=False)

    return holdout_loader_A, holdout_loader_B

# Function to evaluate model on holdout set
def evaluate_holdout(model, data_loader_A, data_loader_B, criterion, device):
    model.eval()

    total_KLD_loss_A = 0
    total_KLD_loss_B = 0
    total_OT_loss = 0
    total_classification_loss = 0
    num_batches = 0

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for i, (batch_A, batch_B) in enumerate(zip(data_loader_A, data_loader_B)):

            data_A, labels_A = batch_A
            data_B, labels_B = batch_B

            data_A = data_A.to(device)
            data_B = data_B.to(device)
            labels_A = labels_A.to(device)
            labels_B = labels_B.to(device)

            data_A = torch.tensor(data_A, dtype=torch.float32)
            data_B = torch.tensor(data_B, dtype=torch.float32)

            data_A_tensors = data_A.clone().detach().requires_grad_(True)
            data_B_tensors = data_B.clone().detach().requires_grad_(True)

            nets, logits = model(data_A_tensors, data_B_tensors)
            (z_A, mu_A, logsigma_A), (z_B, mu_B, logsigma_B) = nets

            KLD_loss_A = KL_divergence(mu_A, logsigma_A)
            KLD_loss_B = KL_divergence(mu_B, logsigma_B)

            labels = labels_A.float()

            OT_loss = LOT(mu_A, logsigma_A, mu_B, logsigma_B)

            # Check if logits is a list
            if isinstance(logits, list):
                # Convert list of tensors to a single tensor
                logits = torch.cat(logits, dim=1)

            # Unpack logits
            classification_logits = logits.squeeze()

            # Reshape labels to match the shape of classification_logits
            labels = labels.view(-1).float()

            classification_loss = criterion(classification_logits, labels)
            
            # Accumulate losses
            total_KLD_loss_A += KLD_loss_A.item()
            total_KLD_loss_B += KLD_loss_B.item()
            total_OT_loss += OT_loss.item()
            total_classification_loss += classification_loss.item()
            num_batches += 1

            # Store logits and labels for accuracy
            all_logits.append(classification_logits)
            all_labels.append(labels)

    avg_KLD_loss_A = total_KLD_loss_A / num_batches
    avg_KLD_loss_B = total_KLD_loss_B / num_batches
    avg_OT_loss = total_OT_loss / num_batches
    avg_classification_loss = total_classification_loss / num_batches

    all_logits = torch.cat(all_logits, dim=0).squeeze()
    all_labels = torch.cat(all_labels, dim=0).squeeze()

    return avg_KLD_loss_A, avg_KLD_loss_B, avg_OT_loss, avg_classification_loss, all_logits, all_labels

# Function for normalizing the losses
def compute_weighted_loss(KLD_loss_A, KLD_loss_B, OT_loss, classification_loss, KLD_A_weight, KLD_B_weight, OT_weight, CL_weight):
    
    # Magnitudes of each loss
    magnitude_KLD_A = KLD_loss_A.item()
    magnitude_KLD_B = KLD_loss_B.item()
    magnitude_OT = OT_loss.item()
    magnitude_CL = classification_loss.item()

    total_magnitude = magnitude_KLD_A + magnitude_KLD_B + magnitude_OT + magnitude_CL
    
    # Inverse losses
    inverse_magnitude_KLD_A = magnitude_KLD_A / (total_magnitude + 1e-8)  + 1e-8# Adding a small epsilon to avoid division by zero
    inverse_magnitude_KLD_B = magnitude_KLD_B / (total_magnitude + 1e-8) + 1e-8
    inverse_magnitude_OT = magnitude_OT / (total_magnitude + 1e-8) + 1e-8
    inverse_magnitude_CL = magnitude_CL / (total_magnitude + 1e-8) + 1e-8

    # Compute the weighted total loss
    total_loss = (KLD_A_weight / inverse_magnitude_KLD_A * KLD_loss_A +
                  KLD_B_weight / inverse_magnitude_KLD_B * KLD_loss_B +
                  OT_weight / inverse_magnitude_OT * OT_loss +
                  CL_weight / inverse_magnitude_CL * classification_loss)
    
    return total_loss


# Early stopping
class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

# Perform k-fold cross-validation
k = 5
splits = KFold(n_splits=k, shuffle=True, random_state=42)


def train_model(config, fold_number, max_epochs, early_stopper, device, output, name, train_loader_A, train_loader_B,
    val_loader_A,val_loader_B,data1_path, data2_path):

    # Extract parameters from config
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    learning_gamma = config["learning_gamma"]
    KLD_A_weight = config["KLD_A_weight"]
    KLD_B_weight = config["KLD_B_weight"]
    OT_weight = config["OT_weight"]
    CL_weight = config["CL_weight"]
    dropout = config["dropout"]
    dim = config["dim"]
    earlystop_patience = config["earlystop_patience"]
    delta = config["delta"]
    decay = config["decay"]

    # Initialize history dictionary to store losses
    history = {
        'KLD_train_loss_A': [],
        'KLD_train_loss_B': [],
        'OT_train_loss': [],
        'classification_train_loss': [],
        'KLD_val_loss_A': [],
        'KLD_val_loss_B': [],
        'OT_val_loss': [],
        'classification_val_loss': [],
        'fold_num': [],
        'epoch_num': []
    }

    # Initialize separate history for holdout evaluations
    holdout_history = {
        'KLD_eval_loss_A': [],
        'KLD_eval_loss_B': [],
        'OT_eval_loss': [],
        'classification_eval_loss': [],
        'fold_num': []
    }

    # Call load_and_prepare_data to get loaders and tensors
    data_loaders = load_and_prepare_data.remote(batch_size, output=output, name=name, data1_path=data1_path, data2_path=data2_path)

    # Retrieve results using ray.get
    data = ray.get(data_loaders)

    # Extract loaders and tensors from data_loaders
    train_loader_A = data['train_loader_A']
    val_loader_A = data['val_loader_A']
    holdout_loader_A = data['holdout_loader_A']
    train_loader_B = data['train_loader_B']
    val_loader_B = data['val_loader_B']
    holdout_loader_B = data['holdout_loader_B']
    train_data_A_tensor = data['train_data_A_tensor']
    train_labels_A_tensor = data['train_labels_A_tensor']
    val_data_A_tensor = data['val_data_A_tensor']
    val_labels_A_tensor = data['val_labels_A_tensor']
    train_data_B_tensor = data['train_data_B_tensor']
    train_labels_B_tensor = data['train_labels_B_tensor']
    val_data_B_tensor = data['val_data_B_tensor']
    train_indices_A = data['train_indices_A']
    train_indices_A_initial = data['train_indices_A_initial']
    val_indices_A = data['val_indices_A']
    val_labels_B_tensor = data['val_labels_B_tensor']
    train_indices_B = data['train_indices_B']
    train_indices_B_initial = data['train_indices_B_initial']
    val_indices_B = data['val_indices_B']
    holdout_indices_A = data['holdout_indices_A']
    holdout_indices_B = data['holdout_indices_B']
    holdout_data_A_tensor = data['holdout_data_A_tensor']
    holdout_data_B_tensor = data['holdout_data_B_tensor']
    holdout_labels_A_tensor = data['holdout_labels_A_tensor']
    holdout_labels_B_tensor = data['holdout_labels_B_tensor']

    # Convert tensors to numpy arrays if necessary
    train_indices_A_tensor_np = train_indices_A_initial.numpy() if isinstance(train_indices_A_initial, torch.Tensor) else train_indices_A_initial
    train_indices_B_tensor_np = train_indices_B_initial.numpy() if isinstance(train_indices_B_initial, torch.Tensor) else train_indices_B_initial

    holdout_indices_A_np = holdout_indices_A.numpy() if isinstance(holdout_indices_A, torch.Tensor) else holdout_indices_A
    holdout_indices_B_np = holdout_indices_B.numpy() if isinstance(holdout_indices_B, torch.Tensor) else holdout_indices_B

    holdout_data_A_np = holdout_data_A_tensor.numpy() if isinstance(holdout_data_A_tensor, torch.Tensor) else holdout_data_A_tensor
    holdout_data_B_np = holdout_data_B_tensor.numpy() if isinstance(holdout_data_B_tensor, torch.Tensor) else holdout_data_B_tensor

    holdout_data_A_np = holdout_data_A_tensor.numpy() if isinstance(holdout_data_A_tensor, torch.Tensor) else holdout_data_A_tensor
    holdout_data_B_np = holdout_data_B_tensor.numpy() if isinstance(holdout_data_B_tensor, torch.Tensor) else holdout_data_B_tensor

    holdout_labels_A_np = holdout_labels_A_tensor.numpy() if isinstance(holdout_labels_A_tensor, torch.Tensor) else holdout_labels_A_tensor
    holdout_labels_B_np = holdout_labels_B_tensor.numpy() if isinstance(holdout_labels_B_tensor, torch.Tensor) else holdout_labels_B_tensor

    # Save train indices
    np.savetxt(output + name + '_train_indices_A.csv', train_indices_A_tensor_np, delimiter=',')
    np.savetxt(output + name + '_train_indices_B.csv', train_indices_B_tensor_np, delimiter=',')

    # Save holdout indices
    np.savetxt(output + name + '_holdout_indices_A.csv', holdout_indices_A_np, delimiter=',')
    np.savetxt(output + name + '_holdout_indices_B.csv', holdout_indices_B_np, delimiter=',')

    # Save holdout data to CSV
    np.savetxt(output + name + '_holdout_data_A.csv', holdout_data_A_np, delimiter=',')
    np.savetxt(output + name + '_holdout_data_B.csv', holdout_data_B_np, delimiter=',')

    # Save holdout labels to CSV
    np.savetxt(output + name + '_holdout_labels_A.csv', holdout_labels_A_np, delimiter=',', fmt='%d')
    np.savetxt(output + name + '_holdout_labels_B.csv', holdout_labels_B_np, delimiter=',', fmt='%d')

    # Extract hyperparameters from config
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    learning_gamma = config["learning_gamma"]
    KLD_A_weight = config["KLD_A_weight"]
    KLD_B_weight = config["KLD_B_weight"]
    OT_weight = config["OT_weight"]
    CL_weight = config["CL_weight"]
    dropout = config["dropout"]
    dim = config["dim"]
    earlystop_patience = config["earlystop_patience"]
    delta = config["delta"]
    decay = config['decay']

    best_model_state = None
    best_predicted_probs = None
    best_targets = None
    best_model_state_overall = None

    all_models = []


    for fold, (pair_A, pair_B) in enumerate(zip(splits.split(data_A), splits.split(data_B))):

        (train_idx_A, val_idx_A) = pair_A
        (train_idx_B, val_idx_B) = pair_B

        print(f'Fold {fold + 1}')

        # Initialize model, optimizer, and best weights
        input_dims = [data_A.shape[1],data_B.shape[1]]

        model = Model(*input_dims, dim=dim, dropout=dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=learning_gamma, patience=earlystop_patience, verbose=True)

        best_val_loss = float('inf')
        best_model_state = None

        # Initialize variables for early stopping
        early_stopper = EarlyStopper(patience=earlystop_patience, min_delta=delta)

        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}')

            # Training phase
            model.train()

            KLD_loss_A_epoch = 0.0
            KLD_loss_B_epoch = 0.0
            OT_loss_epoch = 0.0
            classification_loss_epoch = 0.0
            
            total_iterations = (len(train_loader_A) + len(train_loader_B)) // 2  # Calculate total iterations
            for idx, ((data_A_train, labels_A_train), (data_B_train, labels_B_train)) in tqdm(enumerate(zip(train_loader_A, train_loader_B)), total=total_iterations, desc='Training'):

                data_A_train = data_A_train.to(device)
                data_B_train = data_B_train.to(device)
                labels_A_train = labels_A_train.to(device)
                labels_B_train = labels_B_train.to(device)

                optimizer.zero_grad()

                nets, logits = model(data_A_train, data_B_train)
                (z_A, mu_A, logsigma_A), (z_B, mu_B, logsigma_B) = nets

                # logits_A, logits_B = logits

                KLD_loss_A = KL_divergence(mu_A, logsigma_A)
                KLD_loss_B = KL_divergence(mu_B, logsigma_B)

                # logits = torch.cat((logits_A, logits_B), dim=1)

                labels = labels_A_train.float()

                classification_loss = criterion(logits, labels)

                OT_loss = LOT(mu_A, logsigma_A, mu_B, logsigma_B)

                # Compute the weighted loss using compute_weighted_loss function
                loss = compute_weighted_loss(KLD_loss_A, KLD_loss_B, OT_loss, classification_loss, KLD_A_weight, KLD_B_weight, OT_weight, CL_weight)

                loss.backward()
                optimizer.step()

                # Accumulate losses for the epoch
                KLD_loss_A_epoch += KLD_loss_A.item()
                KLD_loss_B_epoch += KLD_loss_B.item()
                OT_loss_epoch += OT_loss.item()
                classification_loss_epoch += classification_loss.item()
                
            # Calculate average losses for the epoch
            avg_KLD_loss_A = KLD_loss_A_epoch / len(train_loader_A)
            avg_KLD_loss_B = KLD_loss_B_epoch / len(train_loader_B)
            avg_OT_loss = OT_loss_epoch / total_iterations
            avg_classification_loss = classification_loss_epoch / total_iterations

            # Append training loss values to history
            history['KLD_train_loss_A'].append(avg_KLD_loss_A)
            history['KLD_train_loss_B'].append(avg_KLD_loss_B)
            history['OT_train_loss'].append(avg_OT_loss)
            history['classification_train_loss'].append(avg_classification_loss)

            # Print losses after each epoch during training
            print(f"Avg KLD_A (train) Loss: {avg_KLD_loss_A:.4f}")
            print(f"Avg KLD_B (train) Loss: {avg_KLD_loss_B:.4f}")
            print(f"Avg OT (train) Loss: {avg_OT_loss:.4f}")
            print(f"Avg Classification (train) Loss: {avg_classification_loss:.4f}")
      
            # Validation phase
            model.eval()
            val_KLD_loss_A, val_KLD_loss_B, val_OT_loss, val_classification_loss, predicted_values, actual_values = evaluate_holdout(model, val_loader_A, val_loader_B, nn.MSELoss(), device)

            # Print or use val_classification_loss in your logging or comparison
            print(f"KLD_A (val) Loss: {val_KLD_loss_A:.4f}")
            print(f"KLD_B (val) Loss: {val_KLD_loss_B:.4f}")
            print(f"OT (val) Loss: {val_OT_loss:.4f}")
            print(f"Classification (val) Loss: {val_classification_loss:.4f}")

            # Append validation loss values to history
            history['KLD_val_loss_A'].append(val_KLD_loss_A)
            history['KLD_val_loss_B'].append(val_KLD_loss_B)
            history['OT_val_loss'].append(val_OT_loss)
            history['classification_val_loss'].append(val_classification_loss)

            # Append fold and epoch numbers to history
            history['fold_num'].append(fold + 1)
            history['epoch_num'].append(epoch + 1)

            # Update best model state if the current model is better
            if val_classification_loss < best_val_loss:
                best_val_loss = val_classification_loss
                best_model_state_fold = model.state_dict()
                # torch.save(best_model_state_fold, f'{output}/{name}_best_model.pt')
                print("Updated best validation loss")

            else:
                print(f"No improvement: val_classification_loss={val_classification_loss:.4f}, best_val_loss={best_val_loss:.4f}")

            # Check early stopping condition
            if early_stopper(val_classification_loss):
                print(f'Early stopping at epoch {epoch} with validation loss {val_classification_loss:.4f}')
                break

            # Early stopping and scheduler step
            print(f"Scheduler step with val_classification_loss: {val_classification_loss:.4f}")
            scheduler.step(val_classification_loss)

            # Print current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.6f}")

            # End of epoch loop

        #Load the best model state for holdout evaluation
        model.load_state_dict(best_model_state_fold)

        # Evaluate performance on holdout set
        with torch.no_grad():
        
            holdout_KLD_loss_A, holdout_KLD_loss_B, holdout_OT_loss, holdout_classification_loss, predicted_values, actual_values = evaluate_holdout(model, holdout_loader_A, holdout_loader_B, nn.MSELoss(), device)

            predicted_values = torch.tensor(predicted_values)
            actual_values = torch.tensor(actual_values)

            # Save holdout evaluation losses to history
            holdout_history['KLD_eval_loss_A'].append(holdout_KLD_loss_A)
            holdout_history['KLD_eval_loss_B'].append(holdout_KLD_loss_B)
            holdout_history['OT_eval_loss'].append(holdout_OT_loss)
            holdout_history['classification_eval_loss'].append(holdout_classification_loss)
            holdout_history['fold_num'].append(fold + 1)

            # Print holdout evaluation losses after each epoch
            print(f"KLD_A (eval) Loss: {holdout_KLD_loss_A:.4f}")
            print(f"KLD_B (eval) Loss: {holdout_KLD_loss_B:.4f}")
            print(f"OT (eval) Loss: {holdout_OT_loss:.4f}")
            print(f"Classification (eval) Loss: {holdout_classification_loss:.4f}")

            all_models.append(model.state_dict())

    # End of fold loop

    return (model, history, holdout_history, predicted_values, actual_values, {"all_models": all_models})


# Function to save history to CSV
def save_history_to_csv(history, config_name, output_folder):
    # Convert history dictionary to DataFrame
    history_df = pd.DataFrame(history)
    
    # Save history DataFrame to CSV file
    csv_filename = f"training_history_{config_name}.csv"
    csv_path = os.path.join(output_folder, csv_filename)
    history_df.to_csv(csv_path, index=False)

    print(f"Saved training history for {config_name} to: {csv_path}")
