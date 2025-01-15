# data_loader.py

import pandas as pd
import torch

def load_data(input_data_path):
    
    print(f"Loading data from {input_data_path}...")
    data = pd.read_csv(input_data_path)
    
    return torch.tensor(data.values, dtype=torch.float32)
