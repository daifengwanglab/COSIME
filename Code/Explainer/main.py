import argparse
import os
import sys
import logging
import torch
import pandas as pd
from data_loader import load_data
from shapley_computation import monte_carlo_shapley_early_fusion, monte_carlo_shapley_late_fusion


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LoggerWriter:
    def __init__(self, logger):
        self.logger = logger

    def write(self, message):
        if message.strip():
            self.logger.info(message.strip())

    def flush(self):
        pass

def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    sys.stdout = LoggerWriter(logger)

    return logger


def load_and_wrap_model(model_path, model_script_path, input_dims, dim=150, dropout=0.5):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        sys.path.insert(0, os.path.dirname(model_script_path))
        model_script = __import__(os.path.basename(model_script_path).replace('.py', ''))
    except ModuleNotFoundError:
        raise ValueError(f"Could not find the model script at {model_script_path}. Please ensure it is accessible.")

    try:
        model_class = getattr(model_script, 'Model') 
    except AttributeError:
        raise ValueError(f"The script {model_script_path} does not define a 'Model' class. Please check the script.")

    model = model_class(*input_dims, dim=dim, dropout=dropout).to(device)
    
    state_dict = torch.load(model_path, weights_only=True)
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    class WrapperModel(torch.nn.Module):
        def __init__(self, model, input_dims):
            super(WrapperModel, self).__init__()
            self.model = model
            self.input_dims = input_dims

        def forward(self, X):
            X1 = X[:, :self.input_dims[0]]
            X2 = X[:, self.input_dims[0]:]

            _, predicted = self.model(X1, X2)

            return predicted

    wrapper = WrapperModel(model, input_dims)
    return wrapper

def main():
    parser = argparse.ArgumentParser(description="Compute Shapley values and interaction effects")
    parser.add_argument('--input_data', required=True, help="Path to input data file (CSV)")
    parser.add_argument('--input_model', required=True, help="Path to trained model file (.pt)")
    parser.add_argument('--fusion', choices=['early', 'late'], required=True, help="Fusion method: 'early' or 'late'")
    parser.add_argument('--save', required=True, help="Directory to save the Shapley values output")
    parser.add_argument('--log', required=True, help="Log file for the process")
    
    parser.add_argument('--dim', type=int, default=150, help="The dimension for the model (default: 150)")
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate for the model (default: 0.5)")
    parser.add_argument('--mc_iterations', type=int, default=10, help="Number of Monte Carlo iterations")
    parser.add_argument('--batch_size', type=int, default=10, help="Batch size (default: 32)")
    parser.add_argument('--interaction', type=bool, default=True, help="Whether to compute interaction effects (default: True)")

    parser.add_argument('--input_dims', required=True, type=str, help="Input dimensions in the form 'dim1,dim2' (e.g., '100,100')")

    parser.add_argument('--model_script_path', type=str, required=True, help="Path to the user's model script.")
    
    args = parser.parse_args()
    
    input_dims = tuple(map(int, args.input_dims.split(',')))

    logger = setup_logger(args.log)

    X = load_data(args.input_data)

    model = load_and_wrap_model(args.input_model, args.model_script_path, input_dims, dim=args.dim, dropout=args.dropout)

    if args.fusion == 'early':
        shapley_matrix, interaction_matrix = monte_carlo_shapley_early_fusion(
            model, X, mc_iterations=args.mc_iterations, batch_size=args.batch_size, interaction=args.interaction, logger=logger
        )
    elif args.fusion == 'late':
        shapley_matrix, interaction_matrix = monte_carlo_shapley_late_fusion(
            model, X, mc_iterations=args.mc_iterations, batch_size=args.batch_size, interaction=args.interaction, logger=logger
        )

    os.makedirs(args.save, exist_ok=True)
    
    shapley_file = os.path.join(args.save, 'shapley_values.csv')
    print(f"Saving Shapley values to {shapley_file}")
    shapley_df = pd.DataFrame(shapley_matrix, columns=[f"Feature_{i+1}" for i in range(shapley_matrix.shape[1])])
    shapley_df.to_csv(shapley_file, index=False)

    if args.interaction:
        interaction_matrix_file = os.path.join(args.save, 'interaction_matrix.csv')
        print(f"Saving Interaction matrix to {interaction_matrix_file}")
        interaction_matrix_df = pd.DataFrame(interaction_matrix, columns=[f"Feature_{i+1}" for i in range(interaction_matrix.shape[1])])
        interaction_matrix_df.to_csv(interaction_matrix_file, index=False)

if __name__ == "__main__":
    main()
