import argparse
import torch

import cosime

def parse_args():
    parser = argparse.ArgumentParser(description="Train a multi-modal model.")
    parser.add_argument('--input_data_1', type=str, required=True, help="Path to the first data modality.")
    parser.add_argument('--input_data_2', type=str, required=True, help="Path to the second data modality.")
    parser.add_argument('--type', type=str, choices=['binary', 'continuous'], required=True, help="Task type: binary or continuous.")
    parser.add_argument('--fusion', type=str, choices=['early', 'late'], required=True, help="Fusion type: early or late.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size.")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate.")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs.")
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate.")
    parser.add_argument('--dim', type=int, default=100, help="Dimensionality of the embeddings.")
    parser.add_argument('--save', type=str, required=True, help="Path to save the model.")
    parser.add_argument('--log', type=str, required=True, help="Path to the log file.")
    return parser.parse_args()

def main():
    args = parse_args()
    # config = get_config(args)
    print(args)

    # Load data
    data_1, data_2 = cosime.data.load_data(args.input_data_1, args.input_data_2)

    # Load model based on type and fusion method
    model = cosime.models.load_model(config)

    # Train the model
    cosime.models.train_model(model, data_1, data_2, config)

if __name__ == '__main__':
    main()
