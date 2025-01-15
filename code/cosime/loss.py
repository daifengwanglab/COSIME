import torch


def KL_divergence(mu, logsigma):
    loss = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
    return loss


def LOT(mu_src, std_src, mu_tgt, std_tgt, reg=0.1, reg_m=1.0, num_iterations=10, device='cpu', 
        source_weights=None, target_weights=None, idx_q=None, idx_r=None, 
        transport_plan=None, LOT_batch_size=None, domain_regularization=False):
    """
    Learnable Optimal Transport (LOT) function to compute the OT loss between two Gaussian distributions, with improvements.
    
    Args:
    - mu_src, std_src: Means and standard deviations for the source distribution (distribution 1).
    - mu_tgt, std_tgt: Means and standard deviations for the target distribution (distribution 2).
    - reg: Regularization parameter for Sinkhorn's algorithm.
    - reg_m: Scaling factor for the regularization term in the dual variables update.
    - num_iterations: Number of iterations for Sinkhorn's algorithm.
    - device: Device to run the calculations on ('cpu' or 'cuda').
    - source_weights, target_weights: Optional weights for the source and target distributions.
    - idx_q, idx_r: Optional indices for weighting.
    - transport_plan: Optional initial transport plan.
    - LOT_batch_size: If provided, will compute the transport plan in mini-batches for scalability.
    - domain_regularization: Whether to apply domain-specific regularization (e.g., entropy regularization).
    
    Returns:
    - ot_loss: The computed optimal transport loss.
    - transport_plan: The transport plan matrix.
    """

    # Number of elements in the source and target distributions
    n_src = mu_src.size(0)  # Number of samples in source distribution
    n_tgt = mu_tgt.size(0)  # Number of samples in target distribution

    # Step 1: Check for NaNs or Infinities in inputs and replace them with defaults
    mu_src = torch.nan_to_num(mu_src, nan=0.0, posinf=0.0, neginf=0.0)
    std_src = torch.nan_to_num(std_src, nan=1e-6, posinf=1e-6, neginf=1e-6)  # std instead of var
    mu_tgt = torch.nan_to_num(mu_tgt, nan=0.0, posinf=0.0, neginf=0.0)
    std_tgt = torch.nan_to_num(std_tgt, nan=1e-6, posinf=1e-6, neginf=1e-6)  # std instead of var

    # Ensure standard deviations are not zero (to avoid division by zero issues)
    std_src = torch.clamp(std_src, min=1e-6)  
    std_tgt = torch.clamp(std_tgt, min=1e-6)

    # Step 2: Handle the weights for the distributions (source_weights, target_weights)
    if source_weights is None:
        weights_src = torch.ones(n_src, 1) / n_src  # Uniform distribution over source
    else:
        query_batch_weight = source_weights[idx_q] if idx_q is not None else source_weights
        weights_src = query_batch_weight / torch.sum(query_batch_weight)

    if target_weights is None:
        weights_tgt = torch.ones(n_tgt, 1) / n_tgt  # Uniform distribution over target
    else:
        ref_batch_weight = target_weights[idx_r] if idx_r is not None else target_weights
        weights_tgt = ref_batch_weight / torch.sum(ref_batch_weight)

    weights_src = weights_src.to(device)
    weights_tgt = weights_tgt.to(device)

    # Step 3: Initialize transport plan (learnable transport plan)
    if transport_plan is None:
        transport_plan = torch.ones(n_src, n_tgt) / (n_src * n_tgt)
        transport_plan = transport_plan.to(device)
    
    transport_plan = torch.nn.Parameter(transport_plan, requires_grad=True)  # Make transport plan learnable

    # Step 4: Initialize dual variables
    dual_vars = (torch.ones(n_src, 1) / n_src).to(device)
    dual_update_factor = reg_m / (reg_m + reg)  # Scaling factor for dual variables update

    # Step 5: Perform mini-batch processing
    if LOT_batch_size is None:
        # If no mini-batch size is provided, process the entire dataset in one batch
        batches = [(mu_src, std_src, mu_tgt, std_tgt)]
    else:
        # Split the data into mini-batches
        batches = [(mu_src[i:i+LOT_batch_size], std_src[i:i+LOT_batch_size], mu_tgt[i:i+LOT_batch_size], std_tgt[i:i+LOT_batch_size]) 
                   for i in range(0, n_src, LOT_batch_size)]

    # Step 6: Iterative optimization of the transport plan using Sinkhorn's algorithm
    for m in range(num_iterations):
        for mu_batch, std_batch, mu_tgt_batch, std_tgt_batch in batches:
            # Compute pairwise distances for the mini-batch
            dist_mu = torch.cdist(mu_batch.unsqueeze(0), mu_tgt_batch.unsqueeze(0), p=2).squeeze(0)
            dist_std = torch.cdist(std_batch.unsqueeze(0), std_tgt_batch.unsqueeze(0), p=2).squeeze(0)

            # Compute the total cost matrix (mean + standard deviation)
            cost_matrix = dist_mu + dist_std + 1e-6  # Adding small constant for numerical stability

            # Compute the transport kernel and scaling factors
            transport_kernel = torch.exp(-cost_matrix / (reg * torch.max(torch.abs(cost_matrix)))) * transport_plan
            scaling_factors = weights_tgt / (torch.t(transport_kernel) @ dual_vars)

            # Dual variables update
            for i in range(10):
                dual_vars = (weights_src / (transport_kernel @ scaling_factors)) ** dual_update_factor
                scaling_factors = (weights_tgt / (torch.t(transport_kernel) @ dual_vars)) ** dual_update_factor

            # Update the transport plan based on the dual variables
            transport_plan = (dual_vars @ torch.t(scaling_factors)) * transport_kernel

            # Domain-specific regularization (entropy regularization)
            if domain_regularization:
                entropy_reg = -torch.sum(transport_plan * torch.log(transport_plan + 1e-6))
                transport_plan = transport_plan - reg * entropy_reg

    # Step 7: Handle NaN in transport plan (reset to uniform if NaN)
    if torch.isnan(transport_plan).sum() > 0:
        transport_plan = torch.ones(n_src, n_tgt) / (n_src * n_tgt)
        transport_plan = transport_plan.to(device)

    # Step 8: Compute the final optimal transport loss
    ot_loss = (cost_matrix * transport_plan.detach()).sum()

    return ot_loss


def compute_weighted_loss(KLD_loss_A, KLD_loss_B, OT_loss, classification_loss, KLD_A_weight, KLD_B_weight, OT_weight, CL_weight):
    
    # Magnitudes of each loss
    magnitude_KLD_A = KLD_loss_A.item()
    magnitude_KLD_B = KLD_loss_B.item()
    magnitude_OT = OT_loss.item()
    magnitude_CL = classification_loss.item()

    total_magnitude = magnitude_KLD_A + magnitude_KLD_B + magnitude_OT + magnitude_CL
    
    # Inverse losses
    inverse_magnitude_KLD_A = magnitude_KLD_A / (total_magnitude + 1e-8)  + 1e-8# Adding a small epsilon to avoid division by zero
    inverse_magnitude_KLD_B = magnitude_KLD_B / (total_magnitude + 1e-8) + 1e-8
    inverse_magnitude_OT = magnitude_OT / (total_magnitude + 1e-8) + 1e-8
    inverse_magnitude_CL = magnitude_CL / (total_magnitude + 1e-8) + 1e-8

    # Compute the weighted total loss
    total_loss = (KLD_A_weight / inverse_magnitude_KLD_A * KLD_loss_A +
                  KLD_B_weight / inverse_magnitude_KLD_B * KLD_loss_B +
                  OT_weight / inverse_magnitude_OT * OT_loss +
                  CL_weight / inverse_magnitude_CL * classification_loss)
    
    return total_loss