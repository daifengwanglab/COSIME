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

    shapley_matrix = np.zeros((num_samples, num_features))
    interaction_matrix = np.zeros((num_features, num_features))

    if batch_size is None:
        single_input_size_gb = X.element_size() * X.nelement() / 1e9
        max_batch_size = int(max_memory_usage_gb / (4 * single_input_size_gb))
        batch_size = max(1, max_batch_size)
        print(f"Using calculated batch size: {batch_size} based on available memory.")
    else:
        print(f"Using user-defined batch size: {batch_size}.")

    batch_memory_gb = batch_size * X.element_size() * X.size(1) / 1e9
    if batch_memory_gb > max_memory_usage_gb:
        batch_size = int(max_memory_usage_gb / (single_input_size_gb * X.size(1)))
        print(f"Batch size exceeded memory limit. Adjusted batch size: {batch_size}")

    num_batches = int(np.ceil(num_samples / batch_size))

    for feature_idx in tqdm(range(num_features), desc="Computing Shapley values for features"):
        feature_start_time = time.time()  # Time for each feature

        for batch_idx in tqdm(range(num_batches), desc=f"Computing for Feature {feature_idx + 1}", leave=False):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            X_batch = X[start_idx:end_idx, :]

            for sample_idx in range(X_batch.shape[0]):
                marginal_contribs = []
                for _ in range(mc_iterations):
                    X_masked = X_batch.clone()
                    X_masked[sample_idx, feature_idx] = 0

                    pred_masked = model(X_masked)

                    pred_full = model(X_batch)
                    marginal_contrib = torch.mean(pred_full - pred_masked)
                    marginal_contribs.append(marginal_contrib)

                marginal_contribs = torch.tensor(marginal_contribs, dtype=torch.float32)
                shapley_matrix[start_idx + sample_idx, feature_idx] = torch.mean(marginal_contribs)

        feature_end_time = time.time()
        print(f"Shapley values for Feature {feature_idx + 1} computed in {feature_end_time - feature_start_time:.2f} seconds.")

    print("Saving Shapley values to CSV...")
    shapley_df = pd.DataFrame(shapley_matrix, columns=[f"Feature_{i+1}" for i in range(num_features)])
    shapley_df.to_csv("/project8/Dissertation_JC/Coop_DOT/Interaction/Simulated_late/binary_low_early/shapley_values.csv", index=False)
    print("Shapley values saved to 'shapley_values.csv'.")

    if interaction:
        print("Starting computation of interaction effects...")
        interaction_start_time = time.time()

        total_interactions = num_features + (num_features * (num_features - 1)) // 2 

        progress_bar = tqdm(total=total_interactions, desc="Computing interaction effects", leave=False)

        for i in range(num_features):
            for j in range(i, num_features):
                interaction_contribs = []
                with tqdm(total=mc_iterations, desc=f"Computing interaction {i+1}-{j+1}", leave=False) as pbar:
                    for batch_idx in range(num_batches):
                        start_idx = batch_idx * batch_size
                        end_idx = min((batch_idx + 1) * batch_size, num_samples)
                        X_batch = X[start_idx:end_idx, :]

                        for _ in range(mc_iterations):
                            if i == j:

                                X_masked_i = X_batch.clone()
                                X_masked_i[:, i] = 0
                                pred_masked_i = model(X_masked_i)

                                pred_full = model(X_batch)
                                interaction_contrib = torch.mean(pred_full - pred_masked_i)

                            else:
                                X_masked_ij = X_batch.clone()
                                X_masked_ij[:, [i, j]] = 0
                                pred_masked_ij = model(X_masked_ij)

                                X_masked_i = X_batch.clone()
                                X_masked_i[:, i] = 0
                                pred_masked_i = model(X_masked_i)

                                X_masked_j = X_batch.clone()
                                X_masked_j[:, j] = 0
                                pred_masked_j = model(X_masked_j)

                                interaction_contrib = torch.mean(pred_masked_ij - pred_masked_i - pred_masked_j + model(X_batch))

                            interaction_contribs.append(interaction_contrib)

                        pbar.update(1)

                    interaction_contribs_tensor = torch.stack(interaction_contribs)
                    interaction_matrix[i, j] = interaction_matrix[j, i] = torch.mean(interaction_contribs_tensor)

                progress_bar.update(1)

                if logger:
                    logger.info(f"Iteration {i + 1}/{mc_iterations} complete.") 

                tqdm.write(f"Iteration {i + 1}/{mc_iterations} complete.")


        progress_bar.close()

        interaction_end_time = time.time()
        print(f"Interaction effects computed in {interaction_end_time - interaction_start_time:.2f} seconds.")

        
    end_time = time.time()
    print(f"Total computation time: {end_time - start_time:.2f} seconds.")

    return shapley_matrix, interaction_matrix, num_features

def monte_carlo_shapley_late_fusion(model, X, mc_iterations, max_memory_usage_gb=2, batch_size=32, interaction=True, logger=None):
    
    start_time = time.time()

    num_samples = X.shape[0]
    num_features = X.shape[1]

    shapley_matrix = np.zeros((num_samples, num_features))
    interaction_matrix = np.zeros((num_features, num_features))

    if batch_size is None:
        single_input_size_gb = X.element_size() * X.nelement() / 1e9
        max_batch_size = int(max_memory_usage_gb / (4 * single_input_size_gb))
        batch_size = max(1, max_batch_size)
        print(f"Using calculated batch size: {batch_size} based on available memory.")
    else:
        print(f"Using user-defined batch size: {batch_size}.")

    batch_memory_gb = batch_size * X.element_size() * X.size(1) / 1e9
    if batch_memory_gb > max_memory_usage_gb:
        batch_size = int(max_memory_usage_gb / (single_input_size_gb * X.size(1)))
        print(f"Batch size exceeded memory limit. Adjusted batch size: {batch_size}")

    num_batches = int(np.ceil(num_samples / batch_size))

    for feature_idx in tqdm(range(num_features), desc="Computing Shapley values for features"):
        feature_start_time = time.time()

        for batch_idx in tqdm(range(num_batches), desc=f"Computing for Feature {feature_idx + 1}", leave=False):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            X_batch = X[start_idx:end_idx, :]

            for sample_idx in range(X_batch.shape[0]):
                marginal_contribs = []
                for _ in range(mc_iterations):
                    X_masked = X_batch.clone()
                    X_masked[sample_idx, feature_idx] = 0

                    logistic_logits = model(X_masked)
                    logits_A_mask, logits_B_mask = logistic_logits

                    pred_masked = torch.cat((logits_A_mask, logits_B_mask), dim=0)

                    logistic_logits = model(X_batch)
                    logits_A_full, logits_B_full = logistic_logits
                    pred_full = torch.cat((logits_A_full,logits_B_full), dim=0) 

                    marginal_contrib = torch.mean(pred_full - pred_masked)
                    marginal_contribs.append(marginal_contrib)

                marginal_contribs = torch.tensor(marginal_contribs, dtype=torch.float32)
                shapley_matrix[start_idx + sample_idx, feature_idx] = torch.mean(marginal_contribs)

        feature_end_time = time.time()
        print(f"Shapley values for Feature {feature_idx + 1} computed in {feature_end_time - feature_start_time:.2f} seconds.")


    print("Saving Shapley values to CSV...")
    shapley_df = pd.DataFrame(shapley_matrix, columns=[f"Feature_{i+1}" for i in range(num_features)])
    shapley_df.to_csv("/project8/Dissertation_JC/Coop_DOT/Interaction/Simulated_late/binary_high_late/shapley_values.csv", index=False)
    print("Shapley values saved to 'shapley_values.csv'.")

    if interaction:
        print("Starting computation of interaction effects...")
        interaction_start_time = time.time()

        total_interactions = num_features + (num_features * (num_features - 1)) // 2

        progress_bar = tqdm(total=total_interactions, desc="Computing interaction effects", leave=False)

        for i in range(num_features):
            for j in range(i, num_features):
                pair_start_time = time.time()

                with tqdm(total=mc_iterations, desc=f"Computing interaction {i+1}-{j+1}", leave=False) as pbar:
                    interaction_contribs = []
                    for batch_idx in range(num_batches):
                        start_idx = batch_idx * batch_size
                        end_idx = min((batch_idx + 1) * batch_size, num_samples)
                        X_batch = X[start_idx:end_idx, :]

                        for _ in range(mc_iterations):
                            if i == j:
                                X_masked_i = X_batch.clone()
                                X_masked_i[:, i] = 0
                                logistic_logits = model(X_masked_i)
                                logits_A_mask_i, logits_B_mask_i = logistic_logits
                                pred_masked_i = torch.cat((logits_A_mask_i, logits_B_mask_i), dim=0)

                                logistic_logits = model(X_batch)
                                logits_A_mask_full, logits_B_mask_full = logistic_logits
                                pred_full = torch.cat((logits_A_mask_full, logits_B_mask_full), dim=0)

                                interaction_contrib = torch.mean(pred_full - pred_masked_i)

                            else:
                                X_masked_ij = X_batch.clone()
                                X_masked_ij[:, [i, j]] = 0
                                logistic_logits = model(X_masked_ij)
                                logits_A_mask_ij, logits_B_mask_ij = logistic_logits
                                pred_masked_ij = torch.cat((logits_A_mask_ij, logits_B_mask_ij), dim=0)

                                X_masked_i = X_batch.clone()
                                X_masked_i[:, i] = 0
                                logistic_logits = model(X_masked_i)
                                logits_A_mask_i, logits_B_mask_i = logistic_logits
                                pred_masked_i = torch.cat((logits_A_mask_i, logits_B_mask_i), dim=0)

                                X_masked_j = X_batch.clone()
                                X_masked_j[:, j] = 0
                                logistic_logits = model(X_masked_j)
                                logits_A_mask_j, logits_B_mask_j = logistic_logits
                                pred_masked_j = torch.cat((logits_A_mask_j, logits_B_mask_j), dim=0)

                                logistic_logits = model(X_batch)
                                logits_A_mask_full, logits_B_mask_full = logistic_logits
                                pred_full = torch.cat((logits_A_mask_full, logits_B_mask_full), dim=0)

                                interaction_contrib = torch.mean(pred_full - pred_masked_ij - pred_masked_i - pred_masked_j)

                            interaction_contribs.append(interaction_contrib)

                        interaction_matrix[i, j] = torch.mean(torch.tensor(interaction_contribs, dtype=torch.float32))

                    pbar.update(1)

                pair_end_time = time.time()
                pair_time = pair_end_time - pair_start_time
                print(f"Interaction {i+1}-{j+1} computed in {pair_time:.2f} seconds.")

                progress_bar.update(1)

        progress_bar.close()

        interaction_end_time = time.time()
        print(f"Total interaction effects computed in {interaction_end_time - interaction_start_time:.2f} seconds.")

    end_time = time.time()
    print(f"Total computation time: {end_time - start_time:.2f} seconds.")
        
    return shapley_matrix, interaction_matrix, num_features


