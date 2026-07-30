# shapley_computation.py

import os
import time
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def _validate_inputs(X: torch.Tensor, mc_iterations: int, batch_size: Optional[int]) -> None:
    if not isinstance(X, torch.Tensor):
        raise TypeError("X must be a torch.Tensor.")
    if X.ndim != 2:
        raise ValueError(f"X must be two-dimensional; received shape {tuple(X.shape)}.")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"X must contain at least one sample and one feature; received {tuple(X.shape)}.")
    if mc_iterations < 1:
        raise ValueError("mc_iterations must be at least 1.")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be at least 1 or None.")


def _resolve_batch_size(
    X: torch.Tensor,
    batch_size: Optional[int],
    max_memory_usage_gb: float,
) -> int:

    bytes_per_row = X.element_size() * X.shape[1]
    max_bytes = max_memory_usage_gb * 1e9
    calculated = max(1, int(max_bytes // max(4 * bytes_per_row, 1)))

    if batch_size is None:
        resolved = min(X.shape[0], calculated)
        print(f"Using calculated batch size: {resolved} based on the memory limit.")
        return resolved

    resolved = min(batch_size, X.shape[0])
    estimated_bytes = 4 * resolved * bytes_per_row
    if estimated_bytes > max_bytes:
        resolved = min(X.shape[0], calculated)
        print(f"Requested batch size exceeded the memory limit. Adjusted batch size: {resolved}.")
    else:
        print(f"Using user-defined batch size: {resolved}.")
    return resolved


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _compute_mc_shapley(
    model,
    X: torch.Tensor,
    pred_1d: Callable[[object, int], torch.Tensor],
    mc_iterations: int,
    batch_size: int,
    generator: torch.Generator,
) -> np.ndarray:
    
    num_samples, num_features = X.shape
    num_batches = int(np.ceil(num_samples / batch_size))
    shapley_matrix = np.zeros((num_samples, num_features), dtype=np.float64)

    for feature_idx in tqdm(range(num_features), desc="Computing Shapley values for features"):
        feature_start_time = time.time()

        for batch_idx in tqdm(
            range(num_batches),
            desc=f"Feature {feature_idx + 1}/{num_features}",
            leave=False,
        ):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            X_batch = X[start_idx:end_idx]
            bsz = X_batch.shape[0]
            contrib_sum = torch.zeros(bsz, device=X.device, dtype=torch.float64)

            for _ in range(mc_iterations):
                
                permutation = torch.randperm(
                    num_features, device=X.device, generator=generator
                )
                position = int((permutation == feature_idx).nonzero(as_tuple=False).item())
                S_features = permutation[:position]

                mask = torch.zeros(num_features, device=X.device, dtype=X.dtype)
                if S_features.numel() > 0:
                    mask[S_features] = 1

                X_S = X_batch * mask
                X_Si = X_S.clone()
                X_Si[:, feature_idx] = X_batch[:, feature_idx]

                with torch.no_grad():
                    pred_S = pred_1d(model(X_S), bsz)
                    pred_Si = pred_1d(model(X_Si), bsz)

                contrib_sum += (pred_Si - pred_S).to(torch.float64)

            shapley_matrix[start_idx:end_idx, feature_idx] = (
                contrib_sum / float(mc_iterations)
            ).cpu().numpy()

        elapsed = time.time() - feature_start_time
        print(
            f"Shapley values for feature {feature_idx + 1} "
            f"computed in {elapsed:.2f} seconds."
        )

    return shapley_matrix


def _compute_mc_interactions(
    model,
    X: torch.Tensor,
    pred_1d: Callable[[object, int], torch.Tensor],
    shapley_matrix: np.ndarray,
    mc_iterations: int,
    batch_size: int,
    generator: torch.Generator,
    logger=None,
) -> np.ndarray:

    num_samples, num_features = X.shape
    num_batches = int(np.ceil(num_samples / batch_size))
    interaction_matrix = np.zeros((num_features, num_features), dtype=np.float64)

    feature_means = shapley_matrix.mean(axis=0)
    np.fill_diagonal(interaction_matrix, feature_means)

    total_pairs = num_features * (num_features - 1) // 2
    progress_bar = tqdm(total=total_pairs, desc="Computing off-diagonal interactions")

    for i in range(num_features):
        for j in range(i + 1, num_features):
            mc_patient_mean_sum = 0.0

            for _ in range(mc_iterations):
       
                permutation = torch.randperm(
                    num_features, device=X.device, generator=generator
                )
                pos_i = int((permutation == i).nonzero(as_tuple=False).item())
                pos_j = int((permutation == j).nonzero(as_tuple=False).item())
                S_features = permutation[: min(pos_i, pos_j)]

                mask = torch.zeros(num_features, device=X.device, dtype=X.dtype)
                if S_features.numel() > 0:
                    mask[S_features] = 1

                sample_sum = 0.0
                sample_count = 0

                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, num_samples)
                    X_batch = X[start_idx:end_idx]
                    bsz = X_batch.shape[0]

                    X_S = X_batch * mask
                    X_Si = X_S.clone()
                    X_Si[:, i] = X_batch[:, i]
                    X_Sj = X_S.clone()
                    X_Sj[:, j] = X_batch[:, j]
                    X_Sij = X_S.clone()
                    X_Sij[:, i] = X_batch[:, i]
                    X_Sij[:, j] = X_batch[:, j]

                    with torch.no_grad():
                        pred_S = pred_1d(model(X_S), bsz)
                        pred_Si = pred_1d(model(X_Si), bsz)
                        pred_Sj = pred_1d(model(X_Sj), bsz)
                        pred_Sij = pred_1d(model(X_Sij), bsz)

                    tau = pred_Sij - pred_Si - pred_Sj + pred_S
                    sample_sum += tau.to(torch.float64).sum().item()
                    sample_count += bsz

                mc_patient_mean_sum += sample_sum / float(sample_count)

            pair_value = mc_patient_mean_sum / float(mc_iterations)
            interaction_matrix[i, j] = pair_value
            interaction_matrix[j, i] = pair_value
            progress_bar.update(1)

            message = (
                f"Completed interaction pair {i + 1}-{j + 1} "
                f"of {num_features} features."
            )
            if logger is not None:
                logger.info(message)

    progress_bar.close()

    max_diagonal_difference = float(
        np.max(np.abs(np.diag(interaction_matrix) - feature_means))
    )
    print(
        "Maximum absolute difference between the interaction diagonal "
        f"and FI-column means: {max_diagonal_difference:.12g}"
    )

    return interaction_matrix


def _save_outputs(
    shapley_matrix: np.ndarray,
    interaction_matrix: np.ndarray,
    export_dir: Optional[str],
    feature_names=None,
) -> None:
    if export_dir is None:
        return

    os.makedirs(export_dir, exist_ok=True)
    num_features = shapley_matrix.shape[1]
    if feature_names is None:
        feature_names = [f"Feature_{i + 1}" for i in range(num_features)]
    if len(feature_names) != num_features:
        raise ValueError(
            f"feature_names has {len(feature_names)} entries but X has {num_features} features."
        )

    shap_path = os.path.join(export_dir, "shapley_values.csv")
    interaction_path = os.path.join(export_dir, "interaction_matrix.csv")

    pd.DataFrame(shapley_matrix, columns=feature_names).to_csv(shap_path, index=False)
    pd.DataFrame(
        interaction_matrix,
        index=feature_names,
        columns=feature_names,
    ).to_csv(interaction_path, index=True)

    print(f"Shapley values saved to '{shap_path}'.")
    print(f"Interaction matrix saved to '{interaction_path}'.")


def _run_shapley(
    model,
    X: torch.Tensor,
    pred_1d: Callable[[object, int], torch.Tensor],
    mc_iterations: int,
    max_memory_usage_gb: float,
    batch_size: Optional[int],
    interaction: bool,
    logger,
    export_dir: Optional[str],
    seed: int,
    feature_names=None,
):
    _validate_inputs(X, mc_iterations, batch_size)
    start_time = time.time()

    num_samples, num_features = X.shape
    print(f"Samples in input X: {num_samples}")
    print(f"Features in input X: {num_features}")
    print(f"MC iterations: {mc_iterations}")
    print(f"Monte Carlo seed: {seed}")

    resolved_batch_size = _resolve_batch_size(X, batch_size, max_memory_usage_gb)
    print(f"Number of patient batches: {int(np.ceil(num_samples / resolved_batch_size))}")

    model.eval()
    generator = _make_generator(X.device, seed)

    shapley_matrix = _compute_mc_shapley(
        model=model,
        X=X,
        pred_1d=pred_1d,
        mc_iterations=mc_iterations,
        batch_size=resolved_batch_size,
        generator=generator,
    )

    if interaction:
        interaction_start = time.time()
        interaction_matrix = _compute_mc_interactions(
            model=model,
            X=X,
            pred_1d=pred_1d,
            shapley_matrix=shapley_matrix,
            mc_iterations=mc_iterations,
            batch_size=resolved_batch_size,
            generator=generator,
            logger=logger,
        )
        print(
            f"Interaction effects computed in "
            f"{time.time() - interaction_start:.2f} seconds."
        )
    else:
        interaction_matrix = np.zeros((num_features, num_features), dtype=np.float64)

    _save_outputs(
        shapley_matrix,
        interaction_matrix,
        export_dir,
        feature_names=feature_names,
    )

    print(f"Total computation time: {time.time() - start_time:.2f} seconds.")
    return shapley_matrix, interaction_matrix


def monte_carlo_shapley_early_fusion(
    model,
    X,
    mc_iterations,
    max_memory_usage_gb=2,
    batch_size=32,
    interaction=True,
    logger=None,
    export_dir=None,
    seed=2026,
    feature_names=None,
):
    """Early-fusion model"""

    def _pred_1d(out, batch_n):
        if isinstance(out, (tuple, list)):
            out = out[-1]
        return out.reshape(batch_n, -1).mean(dim=1)

    return _run_shapley(
        model=model,
        X=X,
        pred_1d=_pred_1d,
        mc_iterations=mc_iterations,
        max_memory_usage_gb=max_memory_usage_gb,
        batch_size=batch_size,
        interaction=interaction,
        logger=logger,
        export_dir=export_dir,
        seed=seed,
        feature_names=feature_names,
    )


def monte_carlo_shapley_late_fusion(
    model,
    X,
    mc_iterations,
    max_memory_usage_gb=2,
    batch_size=32,
    interaction=True,
    logger=None,
    export_dir=None,
    seed=2026,
    feature_names=None,
):
    """Late-fusion model."""

    def _pred_1d_late(out, batch_n):
        if isinstance(out, (tuple, list)) and len(out) == 2:
            first, second = out
            first = first.reshape(batch_n, -1).mean(dim=1)
            second = second.reshape(batch_n, -1).mean(dim=1)
            return 0.5 * (first + second)
        if isinstance(out, (tuple, list)):
            out = out[-1]
        return out.reshape(batch_n, -1).mean(dim=1)

    return _run_shapley(
        model=model,
        X=X,
        pred_1d=_pred_1d_late,
        mc_iterations=mc_iterations,
        max_memory_usage_gb=max_memory_usage_gb,
        batch_size=batch_size,
        interaction=interaction,
        logger=logger,
        export_dir=export_dir,
        seed=seed,
        feature_names=feature_names,
    )
