"""Evaluation of a trained COSIME model on validation or holdout samples."""
import numpy as np
import torch
from torch import nn
from sklearn.metrics import precision_recall_curve

from loss import kld, lot_loss
from utils import metrics


def best_f1(y, prob):
    """Threshold with the highest F1 (predicted 1 when prob >= threshold), its F1 and the accuracy."""
    precision, recall, thresholds = precision_recall_curve(y, prob)
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    k = int(np.argmax(f1))
    return {'f1_threshold': float(thresholds[k]), 'f1': float(f1[k]),
            'accuracy_f1_threshold': float(np.mean((prob >= thresholds[k]) == y))}


def evaluate(model, xs, y, indices, batch_size, device, cfg):
    """Evaluate the model on the samples in indices.

    Returns the predictions (logits for a binary outcome) and a summary: the mean over batches of
    each loss part (KLD of each view; LOT for two views), the prediction loss over all samples, and
    the metrics (binary: accuracy at 0.5 and at the best-F1 threshold, AUROC, AUPRC;
    continuous: MSE, MAE, R2).
    """
    model.eval()
    outcome_type = cfg['model'].get('type', 'continuous')
    preds, parts = [], []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            enc, prediction = model([torch.from_numpy(x[idx]).to(device) for x in xs])
            preds.append(prediction.cpu().numpy())
            row = {f'kld_{j}': float(kld(mu, lv)) for j, (_, mu, lv) in enumerate(enc, start=1)}
            if len(enc) == 2:
                row['lot'] = float(lot_loss(enc[0], enc[1], cfg))
            parts.append(row)
    result = np.concatenate(preds)
    target = y[indices]
    if outcome_type == 'continuous':
        loss = float(np.mean((result - target) ** 2))
    else:
        loss = float(nn.functional.binary_cross_entropy_with_logits(torch.from_numpy(result), torch.from_numpy(target)).item())
    summary = {k: float(np.mean([p[k] for p in parts])) for k in parts[0]}
    summary['prediction_loss'] = loss
    summary.update(metrics(target, result, outcome_type))
    if outcome_type == 'binary' and len(np.unique(target)) == 2:
        summary.update(best_f1(target, 1 / (1 + np.exp(-np.clip(result, -80, 80)))))
    return result, summary


def evaluate_late(model, xs, ys, indices, batch_size, device, cfg):
    """Late fusion: evaluate each view on its own samples (indices: one array per view).

    Returns the predictions of each view and a summary: the KLD of each view (mean over its
    batches), the LOT between the two views (mean over pairs of batches), and the prediction loss
    and metrics over the samples of all views together.
    """
    model.eval()
    outcome_type = cfg['model'].get('type', 'continuous')
    preds, klds, encs = [], [], []
    with torch.no_grad():
        for j, (x, rows) in enumerate(zip(xs, indices)):
            p, k, e = [], [], []
            for start in range(0, len(rows), batch_size):
                idx = rows[start:start + batch_size]
                enc, prediction = model.view_forward(j, torch.from_numpy(x[idx]).to(device))
                p.append(prediction.cpu().numpy()); k.append(float(kld(enc[1], enc[2]))); e.append(enc)
            preds.append(np.concatenate(p)); klds.append(float(np.mean(k))); encs.append(e)
        summary = {f'kld_{j}': v for j, v in enumerate(klds, start=1)}
        if len(xs) == 2:
            summary['lot'] = float(np.mean([float(lot_loss(a, b, cfg)) for a, b in zip(*encs)]))
    result = np.concatenate(preds)
    target = np.concatenate([y[rows] for y, rows in zip(ys, indices)])
    if outcome_type == 'continuous':
        summary['prediction_loss'] = float(np.mean((result - target) ** 2))
    else:
        summary['prediction_loss'] = float(nn.functional.binary_cross_entropy_with_logits(torch.from_numpy(result), torch.from_numpy(target)).item())
    summary.update(metrics(target, result, outcome_type))
    if outcome_type == 'binary' and len(np.unique(target)) == 2:
        summary.update(best_f1(target, 1 / (1 + np.exp(-np.clip(result, -80, 80)))))
    return preds, summary
