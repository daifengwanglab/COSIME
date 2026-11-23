"""Evaluation metrics."""
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,roc_auc_score,average_precision_score,accuracy_score

def metrics(y, prediction, outcome_type):
    if outcome_type == 'continuous':
        return {'mse': float(mean_squared_error(y, prediction)), 'mae': float(mean_absolute_error(y, prediction)), 'r2': float(r2_score(y, prediction)) if len(y) > 1 else None}
    prob = 1 / (1 + np.exp(-np.clip(prediction, -80, 80)))
    return {'accuracy': float(accuracy_score(y, prob >= .5)), 'auroc': float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else None, 'auprc': float(average_precision_score(y, prob)) if len(np.unique(y)) == 2 else None}
