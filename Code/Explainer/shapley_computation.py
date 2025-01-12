# shapley_computation.py

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import time
import sys
import logging
import pandas as pd
import numpy as np

def monte_carlo_shapley_early_fusion(model, X, mc_iterations, max_memory_usage_gb=2, batch_size=32, interaction=True, logger=None):

    start_time = time.time()

    num_samples = X.shape[0]
    num_features = X.shape[1]

    # Initialize matrices
    shapley_matrix = np.zeros((num_samples, num_features))
    interaction_matrix = np.zeros((num_features, num_features))

    # If batch_size is not provided, calculate it based on memory usage
    if batch_size is None:
        single_input_size_gb = X.element_size() * X.nelement() / 1e9
        max_batch_size = int(max_memory_usage_gb / (4 * single_input_size_gb))
        batch_size = max(1, max_batch_size)
        print(f"Using calculated batch size: {batch_size} based on available memory.")
    else:
        print(f"Using user-defined batch size: {batch_size}.")

    # Compute memory required for the batch
    batch_memory_gb = batch_size * X.element_size() * X.size(1) / 1e9  # GB per batch
    if batch_memory_gb > max_memory_usage_gb:
        batch_size = int(max_memory_usage_gb / (single_input_size_gb * X.size(1)))
        print(f"Batch size exceeded memory limit. Adjusted batch size: {batch_size}")

    # Determine number of batches
    num_batches = int(np.ceil(num_samples / batch_size))

    # Start computing Shapley values for each sample and feature
    feature_start_time_total = time.time()

    for feature_idx in tqdm(range(num_features), desc="Computing Shapley values for features"):
        feature_start_time = time.time()

        for batch_idx in tqdm(range(num_batches), desc=f"Computing for Feature {feature_idx + 1}", leave=False):
            # Get the batch
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            X_batch = X[start_idx:end_idx, :]

            # Monte Carlo sampling for Shapley values
            for sample_idx in range(X_batch.shape[0]):
                marginal_contribs = []
                for _ in range(mc_iterations):
                    # Mask the feature
                    X_masked = X_batch.clone()
                    X_masked[sample_idx, feature_idx] = 0

                    pred_masked = model(X_masked)

                    # Calculate the marginal contribution of the feature
                    pred_full = model(X_batch)

                    marginal_contrib = torch.mean(pred_full - pred_masked)
                    marginal_contribs.append(marginal_contrib)

                # Store the Shapley value for this sample and feature
                marginal_contribs = torch.tensor(marginal_contribs, dtype=torch.float32)
                shapley_matrix[start_idx + sample_idx, feature_idx] = torch.mean(marginal_contribs)

        feature_end_time = time.time()
        print(f"Shapley values for Feature {feature_idx + 1} computed in {feature_end_time - feature_start_time:.2f} seconds.")

    feature_end_time_total = time.time()
        print(f"Shapley values computed in {feature_end_time_total - feature_start_time_total:.2f} seconds.")

    # Save shapley_matrix to CSV
    print("Saving Shapley values to CSV...")
    shapley_df = pd.DataFrame(shapley_matrix, columns=[f"Feature_{i+1}" for i in range(num_features)])
    shapley_df.to_csv("/project8/Dissertation_JC/Coop_DOT/Interaction/Simulated_late/binary_high_late/shapley_values.csv", index=False)
    print("Shapley values saved to 'shapley_values.csv'.")

    # Start computing interaction effects if interaction=True
    if interaction:
        print("Starting computation of interaction effects...")
        interaction_start_time = time.time()

        for i in tqdm(range(num_features), desc="Computing interaction effects", leave=False):
            for j in tqdm(range(i + 1, num_features), desc=f"Computing pairwise interaction {i+1}-{i+1}", leave=False):

                interaction_contribs = []
                for batch_idx in range(num_batches):
                    # Get the batch
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, num_samples)
                    X_batch = X[start_idx:end_idx, :]

                    # Monte Carlo sampling for interaction effects
                    for _ in range(mc_iterations):
                        if i == j:
                            # For self-interaction, mask the feature and compute contribution
                            X_masked_i = X_batch.clone()
                            X_masked_i[:, i] = 0
                            pred_masked_i = model(X_masked_i)

                            # Shapley-Taylor self-contribution should compute the marginal effect of the feature by:
                            # calculating the difference between the prediction without the feature and the prediction with the feature
                            pred_full = model(X_batch)

                            interaction_contrib = torch.mean(pred_full - pred_masked_i)

                        else:
                            # When i != j (pairwise interaction)
                            # Mask both features i and j
                            X_masked_ij = X_batch.clone()
                            X_masked_ij[:, [i, j]] = 0
                            pred_masked_ij = model(X_masked_ij)

                            # Mask only feature i
                            X_masked_i = X_batch.clone()
                            X_masked_i[:, i] = 0
                            pred_masked_i = model(X_masked_i)

                            # Mask only feature j
                            X_masked_j = X_batch.clone()
                            X_masked_j[:, j] = 0
                            pred_masked_j = model(X_masked_j)

                            # Compute the pairwise interaction effect
                            interaction_contrib = torch.mean(pred_masked_ij - pred_masked_i - pred_masked_j + model(X_batch))

                        # Append the computed contribution
                        interaction_contribs.append(interaction_contrib)

                # Convert list to Tensor and compute mean
                interaction_contribs_tensor = torch.stack(interaction_contribs)
                interaction_matrix[i, j] = interaction_matrix[j, i] = torch.mean(interaction_contribs_tensor)

        interaction_end_time = time.time()
        print(f"Interaction effects computed in {interaction_end_time - interaction_start_time:.2f} seconds.")

    end_time = time.time()
    print(f"Total computation time: {end_time - start_time:.2f} seconds.")

    return shapley_matrix, interaction_matrix, num_features


def monte_carlo_shapley_late_fusion(model, X, mc_iterations, max_memory_usage_gb=2, batch_size=32, interaction=True, logger=None):

    start_time = time.time()

    num_samples = X.shape[0]
    num_features = X.shape[1]

    # Initialize matrices
    shapley_matrix = np.zeros((num_samples, num_features))
    interaction_matrix = np.zeros((num_features, num_features))

    # If batch_size is not provided, calculate it based on memory usage
    if batch_size is None:
        single_input_size_gb = X.element_size() * X.nelement() / 1e9
        max_batch_size = int(max_memory_usage_gb / (4 * single_input_size_gb))
        batch_size = max(1, max_batch_size)
        print(f"Using calculated batch size: {batch_size} based on available memory.")
    else:
        print(f"Using user-defined batch size: {batch_size}.")

    # Compute memory required for the batch
    batch_memory_gb = batch_size * X.element_size() * X.size(1) / 1e9
    if batch_memory_gb > max_memory_usage_gb:
        batch_size = int(max_memory_usage_gb / (single_input_size_gb * X.size(1)))
        print(f"Batch size exceeded memory limit. Adjusted batch size: {batch_size}")

    num_batches = int(np.ceil(num_samples / batch_size))

    # Start computing Shapley values for each sample and feature
    feature_start_time_total = time.time()

    for feature_idx in tqdm(range(num_features), desc="Computing Shapley values for features"):
        feature_start_time = time.time() 

        for batch_idx in tqdm(range(num_batches), desc=f"Computing for Feature {feature_idx + 1}", leave=False):
            # Get the batch
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            X_batch = X[start_idx:end_idx, :]

            # Monte Carlo sampling for Shapley values
            for sample_idx in range(X_batch.shape[0]):
                marginal_contribs = []
                for _ in range(mc_iterations):
                    # Mask the feature and calculate the prediction
                    X_masked = X_batch.clone()
                    X_masked[sample_idx, feature_idx] = 0

                    output_masked = model(X_masked)

                    output_A_mask, output_B_mask = output_masked
                    pred_masked = torch.cat((output_A_mask, output_B_mask), dim=0)

                    # Calculate the marginal contribution of the feature
                    output_full = model(X_batch)

                    output_A_full, output_B_full = output_full
                    pred_full = torch.cat((output_A_full, output_B_full), dim=0)

                    marginal_contrib = torch.mean(pred_full - pred_masked)
                    marginal_contribs.append(marginal_contrib)

                # Store the Shapley value for this sample and feature
                marginal_contribs = torch.tensor(marginal_contribs, dtype=torch.float32)
                shapley_matrix[start_idx + sample_idx, feature_idx] = torch.mean(marginal_contribs)

        feature_end_time = time.time() 
        print(f"Shapley values for Feature {feature_idx + 1} computed in {feature_end_time - feature_start_time:.2f} seconds.")

    feature_end_time_total = time.time()
        print(f"Shapley values computed in {feature_end_time_total - feature_start_time_total:.2f} seconds.")

    # Save shapley_matrix to CSV
    print("Saving Shapley values to CSV...")
    shapley_df = pd.DataFrame(shapley_matrix, columns=[f"Feature_{i+1}" for i in range(num_features)])
    shapley_df.to_csv("/project8/Dissertation_JC/Coop_DOT/Interaction/Simulated_late/binary_high_late/shapley_values.csv", index=False)
    print("Shapley values saved to 'shapley_values.csv'.")

    # Start computing interaction effects if interaction=True
    if interaction:
        print("Starting computation of interaction effects...")
        interaction_start_time = time.time()

        for i in tqdm(range(num_features), desc="Computing interaction effects", leave=False):
            for j in tqdm(range(i + 1, num_features), desc=f"Computing pairwise interaction {i+1}-{i+1}", leave=False):

                interaction_contribs = []
                for batch_idx in range(num_batches):
                    # Get the batch
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, num_samples)
                    X_batch = X[start_idx:end_idx, :]

                    # Monte Carlo sampling for interaction effects
                    for _ in range(mc_iterations):
                        if i == j:
                            # For self-interaction, mask the feature and compute contribution
                            X_masked_i = X_batch.clone()
                            X_masked_i[:, i] = 0
                            output_masked_i = model(X_masked_i)

                            output_A_masked_i , output_B_masked_i = output_masked_i
                            pred_masked_i = torch.cat((output_A_masked_i, output_B_masked_i), dim=0)

                            # Shapley-Taylor self-contribution should compute the marginal effect of the feature by:
                            # calculating the difference between the prediction without the feature and the prediction with the feature
                            output_full = model(X_batch)

                            output_A_full , output_B_full = output_full
                            pred_full = torch.cat((output_A_full, output_B_full), dim=0)

                            interaction_contrib = torch.mean(pred_full - pred_masked_i)

                        else:
                            # When i != j (pairwise interaction)
                            # Mask both features i and j
                            X_masked_ij = X_batch.clone()
                            X_masked_ij[:, [i, j]] = 0
                            output_masked_ij = model(X_masked_ij)

                            output_A_masked_ij , output_B_masked_ij = output_masked_ij
                            pred_masked_ij = torch.cat((output_A_masked_ij, output_B_masked_ij), dim=0)

                            # Mask only feature i
                            X_masked_i = X_batch.clone()
                            X_masked_i[:, i] = 0
                            output_masked_i = model(X_masked_i)

                            output_A_masked_i , output_B_masked_i = output_masked_i
                            pred_masked_i = torch.cat((output_A_masked_i, output_B_masked_i), dim=0)

                            # Mask only feature j
                            X_masked_j = X_batch.clone()
                            X_masked_j[:, j] = 0
                            output_masked_j = model(X_masked_j)

                            output_A_masked_j , output_B_masked_j = output_masked_j
                            pred_masked_j = torch.cat((output_A_masked_j, output_B_masked_j), dim=0)

                            output_full = model(X_batch)
                            output_A_full, output_B_full = output_full
                            output_full = torch.cat((output_A_full, output_B_full), dim=0)


                            # Compute the pairwise interaction effect
                            interaction_contrib = torch.mean(pred_masked_ij - pred_masked_i - pred_masked_j + output_full)

                        # Append the computed contribution
                        interaction_contribs.append(interaction_contrib)

                # Convert list to Tensor and compute mean
                interaction_contribs_tensor = torch.stack(interaction_contribs)
                interaction_matrix[i, j] = interaction_matrix[j, i] = torch.mean(interaction_contribs_tensor)

        interaction_end_time = time.time()
        print(f"Interaction effects computed in {interaction_end_time - interaction_start_time:.2f} seconds.")

    end_time = time.time()
    print(f"Total computation time: {end_time - start_time:.2f} seconds.")

    return shapley_matrix, interaction_matrix, num_features


