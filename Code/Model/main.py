import argparse
import torch
import logging
import time
from train import train_model
from data_loader import load_data
from models import Model

def parse_args():
    parser = argparse.ArgumentParser(description="Train a multi-modal model.")
    parser.add_argument('--input_data_1', type=str, required=True, help="Path to the first data modality.")
    parser.add_argument('--input_data_2', type=str, required=True, help="Path to the second data modality.")
    parser.add_argument('--type', dest='m_type', type=str, choices=['binary', 'continuous'], required=True, help="Task type: binary or continuous.")
    parser.add_argument('--fusion', type=str, choices=['early', 'late'], required=True, help="Fusion type: early or late.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size.")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate.")
    parser.add_argument('--kld_1_weight', type=float, default=0.02, help="KLD weight.")
    parser.add_argument('--kld_2_weight', type=float, default=0.02, help="KLD weight.")
    parser.add_argument('--ot_weight', type=float, default=0.02, help="OT weight.")
    parser.add_argument('--cl_weight', type=float, default=0.9, help="CL weight.")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs.")
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate.")
    parser.add_argument('--dim', type=int, default=100, help="Dimensionality of the embeddings.")
    parser.add_argument('--save', dest='save_path', type=str, required=True, help="Path to save the model.")
    parser.add_argument('--log', dest='log_path', type=str, required=True, help="Path to the log file.")
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

def main():
    kwargs = vars(parse_args())
    
    # Set up logging to both file and console
    setup_logging(kwargs['log_path'])

    # Log the start of the run
    logging.info("Starting training process...")

    # Record start time
    start_time = time.time()

    # Load data
    datasets = load_data(kwargs['input_data_1'], kwargs['input_data_2'], **kwargs)
    logging.info(f"Data loaded successfully. Datasets: {kwargs['input_data_1']}, {kwargs['input_data_2']}")

    # Load model based on type and fusion method
    model = Model(*[data[0].shape[1] for data in datasets], **kwargs)
    logging.info("Model initialized successfully.")

    # Train the model and capture all outputs
    model, history, holdout_history, best_predicted_values, best_actual_values = train_model(model, *datasets, **kwargs)

    # Calculate and log training time
    training_time = time.time() - start_time
    logging.info(f"Training completed in {training_time:.2f} seconds.")

    # Optionally save the model
    torch.save(model.state_dict(), kwargs['save_path'])
    logging.info(f"Model saved to {kwargs['save_path']}.")

if __name__ == '__main__':
    main()
