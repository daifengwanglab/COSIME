"""COSIME loss: KL divergence per view, LOT between views, and the prediction loss,
combined with magnitude-adaptive weights."""
import torch
from torch import nn

from lot import LOT

def kld(mu, logvar):
    # Summed over observations and latent dimensions.
    return -.5 * torch.sum(1 + logvar - mu.square() - logvar.exp())

def lot_loss(enc_a, enc_b, cfg):
    _, mu_a, lv_a = enc_a
    _, mu_b, lv_b = enc_b
    opts = cfg.get('lot', {})
    # logsigma is passed as LOT's std argument.
    return LOT(mu_a, lv_a, mu_b, lv_b,
               reg=float(opts.get('reg', 0.1)),
               reg_m=float(opts.get('reg_m', 1.0)),
               num_iterations=int(opts.get('num_iterations', 10)),
               device=mu_a.device,
               LOT_batch_size=opts.get('batch_size'),
               domain_regularization=bool(opts.get('domain_regularization', False)))

def weighted_loss(enc, pred, y, cfg):
    loss_cfg = cfg.get('loss', {})
    model_cfg = cfg['model']
    prediction = nn.functional.binary_cross_entropy_with_logits(pred, y) if model_cfg.get('type', 'continuous') == 'binary' else nn.functional.mse_loss(pred, y)
    parts = [kld(mu, lv) for _, mu, lv in enc]
    weights = [float(loss_cfg.get('kld_1_weight', .02)), float(loss_cfg.get('kld_2_weight', .02))][:len(enc)]
    if len(enc) == 2:
        parts.append(lot_loss(enc[0], enc[1], cfg))
        weights.append(float(loss_cfg.get('ot_weight', .02)))
    parts.append(prediction)
    weights.append(float(loss_cfg.get('cl_weight', .9)))
    # Magnitude-adaptive weights: each part is scaled by its share of the total.
    magnitudes = [p.detach().clamp_min(0) for p in parts]
    total = torch.stack(magnitudes).sum()
    value = sum(w * p / (m / (total + 1e-8) + 1e-8) for p, m, w in zip(parts, magnitudes, weights))
    return value, float(prediction.detach())
