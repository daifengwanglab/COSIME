import argparse
import torch
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

def main():
    kwargs = vars(parse_args())

    # Load data
    datasets = load_data(kwargs['input_data_1'], kwargs['input_data_2'], **kwargs)

    # Load model based on type and fusion method
    model = Model(*[data[0].shape[1] for data in datasets], **kwargs)

    # Train the model
    train_model(model, *datasets, **kwargs)

if __name__ == '__main__':
    main()
