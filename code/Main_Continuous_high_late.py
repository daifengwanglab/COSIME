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

from cosime.data import *
from cosime.loss import *
from cosime.models import *

# # Initialize Ray (if not already initialized)
# ray.init()

# Define paths
name = 'Continuous_high_late'

dir = '/Users/jeromechoi/Documents/jerome/Documents/WISC/BMI/Daifeng Wang/Cooperative learning/Coop_DOT/Results/Simluated/new_111624/Continuous_high_late_test/'
output = '/Users/jeromechoi/Documents/jerome/Documents/WISC/BMI/Daifeng Wang/Cooperative learning/Coop_DOT/Results/Simluated/new_111624/Continuous_high_late_test/'

# dir = '/project8/Dissertation_JC/Coop_DOT/Simulation/111624/data/'
# output = '/project8/Dissertation_JC/Coop_DOT/Simulation/111624/Continuous_high_early/'

data1_path = dir + 'continuous_high_late_x1.csv'
data2_path =  dir + 'continuous_high_late_x2.csv'
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

# Perform k-fold cross-validation
k = 5
splits = KFold(n_splits=k, shuffle=True, random_state=42)
