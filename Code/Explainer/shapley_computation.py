# shapley_computation.py

import logging
import os
import sys
import time

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pandas as pd
import numpy as np

def monte_carlo_shapley_early_fusion(model, X, mc_iterations, max_memory_usage_gb=2, batch_size=32, interaction=True, logger=None, export_dir=None):

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

    def _pred_1d(out, batch_n):
        if isinstance(out, (tuple, list)):
            out = out[-1]
        out = out.view(batch_n, -1).mean(dim=1)
        return out

    for feature_idx in tqdm(range(num_features), desc="Computing Shapley values for features"):
        feature_start_time = time.time()  # Time for each feature

        for batch_idx in tqdm(range(num_batches), desc=f"Computing for Feature {feature_idx + 1}", leave=False):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            X_batch = X[start_idx:end_idx, :]
            bsz = X_batch.shape[0]
            device = X_batch.device

            contrib_sum = torch.zeros(bsz, device=device, dtype=torch.float32)

            for _ in range(mc_iterations):
                perm = torch.randperm(num_features, device=device)
                pos = (perm == feature_idx).nonzero(as_tuple=False).item()
                S_feats = perm[:pos]

                mask = torch.zeros(num_features, device=device, dtype=X_batch.dtype)
                if S_feats.numel() > 0:
                    mask[S_feats] = 1.0

                X_S = X_batch * mask
                X_Si = X_S.clone()
                X_Si[:, feature_idx] = X_batch[:, feature_idx]

                pred_S = _pred_1d(model(X_S), bsz)
                pred_Si = _pred_1d(model(X_Si), bsz)

                contrib_sum += (pred_Si - pred_S)

            contrib_avg = (contrib_sum / float(mc_iterations)).detach().cpu().numpy()
            shapley_matrix[start_idx:end_idx, feature_idx] = contrib_avg

        feature_end_time = time.time()
        print(f"Shapley values for Feature {feature_idx + 1} computed in {feature_end_time - feature_start_time:.2f} seconds.")

    print("Saving Shapley values to CSV...")
    shapley_df = pd.DataFrame(shapley_matrix, columns=[f"Feature_{i+1}" for i in range(num_features)])
    if export_dir is not None:
        shap_path = os.path.join(export_dir, 'shapley_values.csv')
        shapley_df.to_csv(shap_path, index=False)
        print(f"Shapley values saved to '{shap_path}'.")

    if interaction:
        print("Starting computation of interaction effects...")
        interaction_start_time = time.time()

        total_interactions = num_features + (num_features * (num_features - 1)) // 2

        progress_bar = tqdm(total=total_interactions, desc="Computing interaction effects", leave=False)

        for i in range(num_features):
            for j in range(i, num_features):
                tau_sum = 0.0
                with tqdm(total=mc_iterations, desc=f"Computing interaction {i+1}-{j+1}", leave=False) as pbar:
                    for _ in range(mc_iterations):
                        perm = torch.randperm(num_features, device=X.device)
                        pos_i = (perm == i).nonzero(as_tuple=False).item()
                        pos_j = (perm == j).nonzero(as_tuple=False).item()
                        cut = min(pos_i, pos_j)
                        S_feats = perm[:cut]

                        mask = torch.zeros(num_features, device=X.device, dtype=X.dtype)
                        if S_feats.numel() > 0:
                            mask[S_feats] = 1.0

                        tau_batch_sum = 0.0
                        for batch_idx in range(num_batches):
                            start_idx = batch_idx * batch_size
                            end_idx = min((batch_idx + 1) * batch_size, num_samples)
                            X_batch = X[start_idx:end_idx, :]
                            bsz = X_batch.shape[0]

                            X_S = X_batch * mask
                            X_Si = X_S.clone()
                            X_Si[:, i] = X_batch[:, i]
                            X_Sj = X_S.clone()
                            X_Sj[:, j] = X_batch[:, j]
                            X_Sij = X_S.clone()
                            X_Sij[:, i] = X_batch[:, i]
                            X_Sij[:, j] = X_batch[:, j]

                            pred_S = _pred_1d(model(X_S), bsz)
                            pred_Si = _pred_1d(model(X_Si), bsz)
                            pred_Sj = _pred_1d(model(X_Sj), bsz)
                            pred_Sij = _pred_1d(model(X_Sij), bsz)

                            tau_vec = pred_Sij - pred_Si - pred_Sj + pred_S
                            tau_batch_sum += tau_vec.mean().item()

                        tau_sum += (tau_batch_sum / float(num_batches))
                        pbar.update(1)

                interaction_matrix[i, j] = interaction_matrix[j, i] = tau_sum / float(mc_iterations)

                progress_bar.update(1)

                if logger:
                    logger.info(f"Iteration {i + 1}/{mc_iterations} complete.")

                tqdm.write(f"Iteration {i + 1}/{mc_iterations} complete.")

        progress_bar.close()

        interaction_end_time = time.time()
        print(f"Interaction effects computed in {interaction_end_time - interaction_start_time:.2f} seconds.")

    end_time = time.time()
    print(f"Total computation time: {end_time - start_time:.2f} seconds.")

    return shapley_matrix, interaction_matrix

def monte_carlo_shapley_late_fusion(model, X, mc_iterations, max_memory_usage_gb=2, batch_size=32, interaction=True, logger=None, export_dir=None):

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

    def _pred_1d_late(out, batch_n):
        if isinstance(out, (tuple, list)) and len(out) == 2:
            a, b = out
            a = a.view(batch_n, -1).mean(dim=1)
            b = b.view(batch_n, -1).mean(dim=1)
            return 0.5 * (a + b)
        out = out.view(batch_n, -1).mean(dim=1)
        return out

    for feature_idx in tqdm(range(num_features), desc="Computing Shapley values for features"):
        feature_start_time = time.time()

        for batch_idx in tqdm(range(num_batches), desc=f"Computing for Feature {feature_idx + 1}", leave=False):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            X_batch = X[start_idx:end_idx, :]
            bsz = X_batch.shape[0]
            device = X_batch.device

            contrib_sum = torch.zeros(bsz, device=device, dtype=torch.float32)

            for _ in range(mc_iterations):
                perm = torch.randperm(num_features, device=device)
                pos = (perm == feature_idx).nonzero(as_tuple=False).item()
                S_feats = perm[:pos]

                mask = torch.zeros(num_features, device=device, dtype=X_batch.dtype)
                if S_feats.numel() > 0:
                    mask[S_feats] = 1.0

                X_S = X_batch * mask
                X_Si = X_S.clone()
                X_Si[:, feature_idx] = X_batch[:, feature_idx]

                pred_S = _pred_1d_late(model(X_S), bsz)
                pred_Si = _pred_1d_late(model(X_Si), bsz)

                contrib_sum += (pred_Si - pred_S)

            contrib_avg = (contrib_sum / float(mc_iterations)).detach().cpu().numpy()
            shapley_matrix[start_idx:end_idx, feature_idx] = contrib_avg

        feature_end_time = time.time()
        print(f"Shapley values for Feature {feature_idx + 1} computed in {feature_end_time - feature_start_time:.2f} seconds.")

    print("Saving Shapley values to CSV...")
    shapley_df = pd.DataFrame(shapley_matrix, columns=[f"Feature_{i+1}" for i in range(num_features)])
    if export_dir is not None:
        shap_path = os.path.join(export_dir, 'shapley_values.csv')
        shapley_df.to_csv(shap_path, index=False)
        print(f"Shapley values saved to '{shap_path}'.")

    if interaction:
        print("Starting computation of interaction effects...")
        interaction_start_time = time.time()

        total_interactions = num_features + (num_features * (num_features - 1)) // 2

        progress_bar = tqdm(total=total_interactions, desc="Computing interaction effects", leave=False)

        for i in range(num_features):
            for j in range(i, num_features):
                tau_sum = 0.0
                with tqdm(total=mc_iterations, desc=f"Computing interaction {i+1}-{j+1}", leave=False) as pbar:
                    for _ in range(mc_iterations):
                        perm = torch.randperm(num_features, device=X.device)
                        pos_i = (perm == i).nonzero(as_tuple=False).item()
                        pos_j = (perm == j).nonzero(as_tuple=False).item()
                        cut = min(pos_i, pos_j)
                        S_feats = perm[:cut]

                        mask = torch.zeros(num_features, device=X.device, dtype=X.dtype)
                        if S_feats.numel() > 0:
                            mask[S_feats] = 1.0

                        tau_batch_sum = 0.0
                        for batch_idx in range(num_batches):
                            start_idx = batch_idx * batch_size
                            end_idx = min((batch_idx + 1) * batch_size, num_samples)
                            X_batch = X[start_idx:end_idx, :]
                            bsz = X_batch.shape[0]

                            X_S = X_batch * mask
                            X_Si = X_S.clone()
                            X_Si[:, i] = X_batch[:, i]
                            X_Sj = X_S.clone()
                            X_Sj[:, j] = X_batch[:, j]
                            X_Sij = X_S.clone()
                            X_Sij[:, i] = X_batch[:, i]
                            X_Sij[:, j] = X_batch[:, j]

                            pred_S = _pred_1d_late(model(X_S), bsz)
                            pred_Si = _pred_1d_late(model(X_Si), bsz)
                            pred_Sj = _pred_1d_late(model(X_Sj), bsz)
                            pred_Sij = _pred_1d_late(model(X_Sij), bsz)

                            tau_vec = pred_Sij - pred_Si - pred_Sj + pred_S
                            tau_batch_sum += tau_vec.mean().item()

                        tau_sum += (tau_batch_sum / float(num_batches))
                        pbar.update(1)

                interaction_matrix[i, j] = interaction_matrix[j, i] = tau_sum / float(mc_iterations)

                progress_bar.update(1)

        progress_bar.close()

        interaction_end_time = time.time()
        print(f"Total interaction effects computed in {interaction_end_time - interaction_start_time:.2f} seconds.")

    end_time = time.time()
    print(f"Total computation time: {end_time - start_time:.2f} seconds.")

    return shapley_matrix, interaction_matrix
