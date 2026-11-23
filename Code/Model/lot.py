"""Learnable optimal transport (LOT) between the latent distributions of two views."""
import torch


def LOT(mu_src, std_src, mu_tgt, std_tgt, reg=0.1, reg_m=1.0,
        num_iterations=10, device=None, source_weights=None, target_weights=None,
        idx_q=None, idx_r=None, transport_plan=None, LOT_batch_size=None,
        domain_regularization=False):
    if mu_src.ndim != 2 or mu_tgt.ndim != 2:
        raise ValueError('LOT expects matrices of latent means and standard deviations')
    if mu_src.shape != std_src.shape or mu_tgt.shape != std_tgt.shape:
        raise ValueError('LOT means and standard deviations must have matching shapes')
    if mu_src.shape[1] != mu_tgt.shape[1]:
        raise ValueError('LOT latent dimensions must match')
    if reg <= 0 or reg_m <= 0 or num_iterations < 1:
        raise ValueError('LOT requires reg > 0, reg_m > 0, num_iterations >= 1')
    if LOT_batch_size is not None and LOT_batch_size < 1:
        raise ValueError('LOT_batch_size must be positive')
    device = mu_src.device if device is None else torch.device(device)
    if mu_tgt.device != mu_src.device or device != mu_src.device:
        raise ValueError('LOT inputs and device must match')
    mu_src = torch.nan_to_num(mu_src, nan=0., posinf=0., neginf=0.)
    mu_tgt = torch.nan_to_num(mu_tgt, nan=0., posinf=0., neginf=0.)
    std_src = torch.nan_to_num(std_src, nan=1e-6, posinf=1e-6, neginf=1e-6).clamp_min(1e-6)
    std_tgt = torch.nan_to_num(std_tgt, nan=1e-6, posinf=1e-6, neginf=1e-6).clamp_min(1e-6)
    ns, nt = mu_src.shape[0], mu_tgt.shape[0]
    def weights(raw, idx, n):
        if raw is None:
            return torch.full((n, 1), 1. / n, device=device, dtype=mu_src.dtype)
        v = torch.as_tensor(raw, dtype=mu_src.dtype, device=device)
        if idx is not None:
            v = v[idx]
        v = v.reshape(-1, 1)
        if len(v) != n or (v < 0).any() or v.sum() <= 0:
            raise ValueError('Invalid LOT distribution weights')
        return v / v.sum()
    a = weights(source_weights, idx_q, ns)
    b = weights(target_weights, idx_r, nt)
    if transport_plan is None:
        plan = torch.full((ns, nt), 1. / (ns * nt), device=device, dtype=mu_src.dtype)
    else:
        plan = torch.as_tensor(transport_plan, device=device, dtype=mu_src.dtype)
        if plan.shape != (ns, nt):
            raise ValueError('LOT transport plan has incompatible dimensions')
    # Mini-batch LOT: every source block is paired with every target block, and each pair is
    # solved as its own OT problem. Block masses weight the loss, so its scale matches the
    # full-batch objective; nonuniform weights are renormalized within each block.
    if LOT_batch_size is not None and (ns > LOT_batch_size or nt > LOT_batch_size):
        if transport_plan is not None:
            raise ValueError('An initial full transport_plan cannot be reused for independent mini-batch OT blocks')
        src_blocks = list(range(0, ns, LOT_batch_size))
        tgt_blocks = list(range(0, nt, LOT_batch_size))
        result = mu_src.new_zeros(())
        for si in src_blocks:
            sj = min(si + LOT_batch_size, ns)
            ai = a[si:sj]
            for ti in tgt_blocks:
                tj = min(ti + LOT_batch_size, nt)
                bi = b[ti:tj]
                block_mass = ai.sum() * bi.sum()
                block_loss = LOT(
                    mu_src[si:sj], std_src[si:sj],
                    mu_tgt[ti:tj], std_tgt[ti:tj],
                    reg=reg, reg_m=reg_m, num_iterations=num_iterations,
                    device=device, source_weights=ai.flatten(),
                    target_weights=bi.flatten(), LOT_batch_size=None,
                    domain_regularization=domain_regularization,
                )
                result = result + block_mass * block_loss
        return result
    # The transport plan is a new parameter in each call; it is not part of the model optimizer.
    plan = torch.nn.Parameter(plan, requires_grad=True)
    dual = torch.full((ns, 1), 1. / ns, device=device, dtype=mu_src.dtype)
    factor = reg_m / (reg_m + reg)
    cost = torch.cdist(mu_src, mu_tgt, p=2) + torch.cdist(std_src, std_tgt, p=2) + 1e-6
    for _ in range(num_iterations):
        kernel = torch.exp(-cost / (reg * cost.abs().max())) * plan
        scale = b / (kernel.T @ dual).clamp_min(1e-12)
        for _ in range(10):
            dual = (a / (kernel @ scale).clamp_min(1e-12)).pow(factor)
            scale = (b / (kernel.T @ dual).clamp_min(1e-12)).pow(factor)
        plan = (dual @ scale.T) * kernel
        if domain_regularization:
            entropy = -(plan * (plan + 1e-6).log()).sum()
            plan = plan - reg * entropy
    if torch.isnan(plan).any():
        plan = torch.full((ns, nt), 1. / (ns * nt), device=device, dtype=mu_src.dtype)
    return (cost * plan.detach()).sum()
