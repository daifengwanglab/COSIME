import argparse
import torch
import logging
import time
import os
import pandas as pd
from data_loader_revised import load_and_prepare_data
from models import Model
from train_revised import train_model_binary, train_model_continuous

def parse_args():
    parser = argparse.ArgumentParser(description="Train a multi-modal model.")
    parser.add_argument('--input_data_1', type=str, required=True, help="Path to the first data modality.")
    parser.add_argument('--input_data_2', type=str, required=True, help="Path to the second data modality.")
    parser.add_argument('--type', dest='task_type', type=str, choices=['binary', 'continuous'], required=True, help="Task type: binary or continuous.")
    parser.add_argument('--predictor', dest='predictor_type', type=str, choices=['regression', 'NN'], required=True, help="Predictor type: regression or neural network.")
    parser.add_argument('--fusion', type=str, choices=['early', 'late'], required=True, help="Fusion type: early or late.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size.")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate.")
    parser.add_argument('--learning_gamma', type=float, default=0.0001, help="Learning gamma.")
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate.")
    parser.add_argument('--kld_1_weight', type=float, default=0.02, help="KLD weight.")
    parser.add_argument('--kld_2_weight', type=float, default=0.02, help="KLD weight.")
    parser.add_argument('--ot_weight', type=float, default=0.02, help="OT weight.")
    parser.add_argument('--cl_weight', type=float, default=0.9, help="CL weight.")
    parser.add_argument('--dim', type=int, default=100, help="Dimensionality of the embeddings.")
    parser.add_argument('--earlystop_patience', type=int, default=40, help="Early stop patience in training.")
    parser.add_argument('--delta', type=float, default=0.001, help="Minimum improvement required to reset early stopping counter.")
    parser.add_argument('--decay', type=float, default=0.001, help="Decrease in learning rate during training.")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs.")
    parser.add_argument('--save', dest='save_path', type=str, required=True, help="Path to save the model.")
    parser.add_argument('--log', dest='log_path', type=str, required=True, help="Path to the log file.")
    parser.add_argument('--splits', type=int, default=5, help="Number of splits for cross-validation.")  # New splits argument
    return parser.parse_args()

def setup_logging(log_file):
    # Set up logging configuration
    logging.basicConfig(
        filename=log_file, 
        level=logging.INFO, 
        format='%(asctime)s - %(message)s',
        filemode='w'  # 'w' for overwriting log file each time, use 'a' for appending
    )
    # Also print logs to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)

def create_directory(path):
    """Create the directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def save_results(model, history, holdout_history, best_predicted_values, best_actual_values, save_path):
    # 1. Save the model as a .pt file
    torch.save(model.state_dict(), f'{save_path}/best_model.pt')
    logging.info(f"Model saved to {save_path}/best_model.pt")

    # 2. Save history and holdout_history as CSV files
    history_df = pd.DataFrame(history)
    holdout_history_df = pd.DataFrame(holdout_history)

    history_df.to_csv(f'{save_path}/history.csv', index=False)
    holdout_history_df.to_csv(f'{save_path}/holdout_history.csv', index=False)

    logging.info(f"History saved to {save_path}/history.csv")
    logging.info(f"Holdout history saved to {save_path}/holdout_history.csv")

    # 3. Save best predicted and actual values as CSV files
    best_predicted_df = pd.DataFrame(best_predicted_values)
    best_actual_df = pd.DataFrame(best_actual_values)

    best_predicted_df.to_csv(f'{save_path}/best_predicted_values.csv', index=False)
    best_actual_df.to_csv(f'{save_path}/best_actual_values.csv', index=False)

    logging.info(f"Best predicted values saved to {save_path}/best_predicted_values.csv")
    logging.info(f"Best actual values saved to {save_path}/best_actual_values.csv")

def main():
    kwargs = vars(parse_args())
    
    # Set up logging to both file and console
    setup_logging(kwargs['log_path'])

    # Log the start of the run
    logging.info("Starting training process...")

    # Record start time
    start_time = time.time()

    # Load data using load_and_prepare_data
    train_loader_A, val_loader_A, holdout_loader_A, train_loader_B, val_loader_B, holdout_loader_B = load_and_prepare_data(
        batch_size=kwargs['batch_size'],
        output=None,  # Output is not used, could be added for logging if necessary
        name=None,  # Name is not used, could be added for logging if necessary
        data1_path=kwargs['input_data_1'],
        data2_path=kwargs['input_data_2'],
        task_type=kwargs['task_type']
    )
    logging.info(f"Data loaded successfully from {kwargs['input_data_1']} and {kwargs['input_data_2']}.")

    # Get input dimensions from the first batch in the DataLoader
    sample_A = next(iter(train_loader_A))  # First batch from DataLoader A
    sample_B = next(iter(train_loader_B))  # First batch from DataLoader B
    input_dim_A = sample_A[0].shape[1]  # Shape of features (rows: batch_size, cols: num_features)
    input_dim_B = sample_B[0].shape[1]  # Shape of features (rows: batch_size, cols: num_features)
    logging.info(f"Input dimensions - A: {input_dim_A}, B: {input_dim_B}")

    # Load model based on type and fusion method
    model = Model(
        *[input_dim_A, input_dim_B],
        m_type=kwargs['task_type'],  # Use the 'type' argument as 'm_type'
        **kwargs  # Pass the rest of the arguments
    )
    logging.info("Model initialized successfully.")

    # Train the model based on the task type (binary or continuous)
    if kwargs['task_type'] == 'binary':
        model, history, holdout_history, best_predicted_values, best_actual_values = train_model_binary(
            model, 
            kwargs['input_data_1'], 
            kwargs['input_data_2'], 
            kwargs['learning_rate'], 
            kwargs['task_type'],  # Using kwargs['task_type'] here
            kwargs['epochs'], 
            kwargs['save_path'], 
            kwargs['splits'],  # Pass the splits argument here
            kwargs['device'], 
            **kwargs  # Passing other arguments as **kwargs
        )

    elif kwargs['task_type'] == 'continuous':
        model, history, holdout_history, best_predicted_values, best_actual_values = train_model_continuous(
            model, 
            kwargs['input_data_1'], 
            kwargs['input_data_2'], 
            kwargs['learning_rate'], 
            kwargs['task_type'],  # Using kwargs['task_type'] here
            kwargs['epochs'], 
            kwargs['save_path'], 
            kwargs['splits'],  # Pass the splits argument here
            kwargs['device'], 
            **kwargs  # Passing other arguments as **kwargs
        )

    # Calculate and log training time
    training_time = time.time() - start_time
    logging.info(f"Training completed in {training_time:.2f} seconds.")

    # Create directory to save results
    create_directory(kwargs['save_path'])

    # Save the results: model, history, holdout history, best predictions, and actual values
    save_results(model, history, holdout_history, best_predicted_values, best_actual_values, kwargs['save_path'])

if __name__ == '__main__':
    main()
