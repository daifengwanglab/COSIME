# python Code/Explainer/COSIME_FI.py --config Code/Explainer/COSIME_FI.yml > COSIME_FI.log 2>&1
"""COSIME explainer: Shapley feature importance and feature interactions on the holdout data.

Feature importance: Monte Carlo permutation Shapley values.
Interactions: order-2 Shapley-Taylor (Monte Carlo permutations), or Integrated Hessians
(double integral, Gauss-Legendre quadrature). Off-diagonal interactions only; the diagonal is zero.
Masked features take a reference value: the mean of each feature over the input rows (the most
common value for features with fewer than 4 distinct values), or zero.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Model'))
from config import resolve  # noqa: E402
from model_loader import load_model  # noqa: E402


def load_inputs(paths, expected_names, device, fusion='early'):
    """Read the processed FI CSV written by training (one file for one or more views).

    Each view has its own sample-ID column; padded (blank) rows are dropped per view.
    """
    if isinstance(paths, str):
        paths = [paths]
    if len(paths) != 1:
        raise ValueError('COSIME_FI expects one processed FI input CSV')
    path = resolve(paths[0])
    df = pd.read_csv(path)
    multi = len(expected_names) > 1
    xs, names, ids = [], [], []
    if multi:
        for j, expected in enumerate(expected_names, 1):
            id_col = f'view{j}__sample_id'
            cols = [f'view{j}__{name}' for name in expected]
            missing = [c for c in [id_col, *cols] if c not in df]
            if missing:
                raise ValueError(f'{path}: missing view {j} columns: {missing}')
            valid = df[id_col].notna() & df[id_col].astype(str).str.strip().ne('')
            part = df.loc[valid, cols]
            x = part.apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float32)
            if not np.isfinite(x).all():
                raise ValueError(f'{path}: nonfinite features in actual view {j} samples')
            xs.append(torch.tensor(x, device=device))
            ids.append(df.loc[valid, id_col].astype(str).tolist())
            names.extend([f'view{j}::{s}' for s in expected])
    else:
        expected = expected_names[0]
        if df.columns.drop('y', errors='ignore').tolist() != expected:
            raise ValueError(f'{path}: feature names/order do not match saved best model')
        x = df[expected].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError(f'{path}: nonfinite features')
        xs = [torch.tensor(x, device=device)]
        names = list(expected)
        id_path = Path(path).with_name('FI_holdout_sample_ids.csv')
        ids = [pd.read_csv(id_path)['sample_id'].astype(str).tolist() if id_path.exists()
               else [str(i) for i in range(len(x))]]
    if fusion != 'late' and len({tuple(v) for v in ids}) != 1:
        raise ValueError('Early-fusion sample IDs/order differ between views')
    return xs, names, ids

def predict(model, flat, dims):
    xs = list(torch.split(flat, dims, dim=1))
    output = model(xs)
    return (output[-1] if isinstance(output, tuple) else output).reshape(-1)


def make_reference(x, mask):
    """Value a masked feature takes.

    mean: the mean over the input rows, or the most common value for features with fewer
          than 4 distinct values (binary and small counts), so a masked feature takes a
          realistic value.
    zero: 0 for every feature.
    """
    if mask == 'zero':
        return torch.zeros(x.shape[1], dtype=x.dtype, device=x.device)
    if mask != 'mean':
        raise ValueError("feature_importance.mask must be 'mean' or 'zero'")
    reference = x.mean(dim=0)
    for j in range(x.shape[1]):
        values, counts = torch.unique(x[:, j], return_counts=True)
        if len(values) < 4:
            reference[j] = values[counts.argmax()]
    return reference


def shapley_values(model, x, dims, permutations, batch_size, seed, reference):
    """Monte Carlo permutation Shapley values.

    For each sampled permutation, features are revealed one at a time, starting from the
    reference, and f(S union {j}) - f(S) is assigned to feature j. The mean over permutations
    approximates the Shapley value of each sample. For each sample, the values add up to
    f(x) - f(reference).
    """
    if permutations < 1:
        raise ValueError('feature_importance.permutations must be positive')
    n, d = x.shape
    rng = np.random.default_rng(seed)
    scores = np.zeros((n, d), dtype=np.float64)
    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc='Shapley feature importance batches'):
            batch = x[start:start + batch_size]
            contribution = torch.zeros_like(batch, dtype=batch.dtype)
            baseline = reference.expand_as(batch).clone()
            for _ in tqdm(range(permutations), desc='Shapley permutations', leave=False):
                order = rng.permutation(d)
                coalition = baseline.clone()
                previous = predict(model, coalition, dims)
                for feature in order:
                    coalition[:, feature] = batch[:, feature]
                    current = predict(model, coalition, dims)
                    contribution[:, feature] += (current - previous).to(contribution.dtype)
                    previous = current
            scores[start:start + len(batch)] = (contribution / permutations).cpu().numpy()
    return scores.astype(np.float32)


def shapley_taylor(model, x, dims, permutations, batch_size, seed, reference):
    """Order-2 Shapley-Taylor: E_perm[Delta_ij f(S_before_first(i,j))].

    Masked features take the reference value. The diagonal is set to zero.
    """
    n, d = x.shape
    rng = np.random.default_rng(seed)
    result = np.zeros((d, d), dtype=np.float64)
    for start in tqdm(range(0, n, batch_size), desc='Shapley-Taylor batches'):
        batch = x[start:start + batch_size]
        for _ in tqdm(range(permutations), desc='Shapley-Taylor permutations', leave=False):
            order = rng.permutation(d)
            pos = np.empty(d, dtype=np.int64)
            pos[order] = np.arange(d)
            # Cache coalition predictions for each prefix and its 1- and 2-feature additions.
            # For pair i,j use the prefix before whichever appears first.
            cache = {}
            def coalition(indices):
                key = tuple(sorted(indices))
                if key not in cache:
                    masked = reference.expand_as(batch).clone()
                    if key:
                        masked[:, list(key)] = batch[:, list(key)]
                    with torch.no_grad():
                        cache[key] = predict(model, masked, dims)
                return cache[key]
            for i in range(d):
                for j in range(i + 1, d):
                    prefix = order[:min(pos[i], pos[j])].tolist()
                    base = coalition(prefix)
                    fi = coalition(prefix + [i])
                    fj = coalition(prefix + [j])
                    fij = coalition(prefix + [i, j])
                    result[i, j] += (fij - fi - fj + base).sum().item()
    result /= (n * permutations)
    return result + result.T


def integrated_hessians(model, x, dims, steps, batch_size, reference):
    """Off-diagonal Integrated Hessians (Janizek et al.).

    I_ij = delta_i delta_j int_0^1 int_0^1 alpha beta
           d2 f(b + alpha beta delta)/(dx_i dx_j) d alpha d beta,
    with baseline b = reference and delta = x - b; Gauss-Legendre quadrature.
    """
    from numpy.polynomial.legendre import leggauss
    nodes, weights = leggauss(steps)
    nodes = (nodes + 1) / 2
    weights = weights / 2
    n, d = x.shape
    total = torch.zeros((d, d), dtype=x.dtype, device=x.device)
    for start in tqdm(range(0, n, batch_size), desc='Integrated Hessians batches'):
        batch = x[start:start + batch_size]
        delta = batch - reference
        for a, wa in zip(nodes, weights):
            for b, wb in zip(nodes, weights):
                point = (reference + delta * float(a * b)).detach().requires_grad_(True)
                output = predict(model, point, dims)
                first = torch.autograd.grad(output.sum(), point, create_graph=True)[0]
                # Hessian rows: gradient of each first partial w.r.t. all features.
                for i in range(d):
                    row = torch.autograd.grad(first[:, i].sum(), point,
                                              retain_graph=i < d - 1, allow_unused=True)[0]
                    if row is not None:
                        total[i] += (row * delta[:, i:i+1] * delta).sum(dim=0).detach().to(total.dtype) * float(wa * wb * a * b)
    result = (total / n).cpu().numpy()
    result = (result + result.T) / 2
    np.fill_diagonal(result, 0.0)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = resolve(cfg['output']['directory'])
    out.mkdir(parents=True, exist_ok=True)
    choice = cfg.get('compute', {}).get('device', 'auto')
    if choice == 'auto':
        choice = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = torch.device(choice)
    saved_best_model = torch.load(resolve(cfg['model']['best_model_path']), map_location='cpu', weights_only=False)
    if not isinstance(saved_best_model, dict) or 'model_state_dict' not in saved_best_model:
        raise ValueError('Expected saved COSIME best model with model_state_dict')
    fusion = saved_best_model['config'].get('model', {}).get('fusion', 'early')
    xs, names, ids = load_inputs(cfg['data']['paths'], saved_best_model['feature_names'], device, fusion)
    dims = [len(a) for a in saved_best_model['feature_names']]
    model = load_model(dims, saved_best_model['config']).to(device)
    model.load_state_dict(saved_best_model['model_state_dict'])
    model.eval()
    if not xs or any(v.shape[0] == 0 for v in xs):
        raise ValueError('No holdout samples in at least one modality')
    if fusion == 'late' and len(xs) > 1:
        # Shared subjects: matched by subject ID, not by row position.
        # No shared subjects: independent pairs of samples (one per view) are drawn. They are not
        # observed subjects, so cross-view interactions refer to this product-of-marginals reference.
        shared = set(ids[0])
        for view_ids in ids[1:]:
            shared.intersection_update(view_ids)
        ordered = list(dict.fromkeys(sid for sid in ids[0] if sid in shared))
        if ordered:
            aligned = []
            for view_x, view_ids in zip(xs, ids):
                index = {sid: i for i, sid in enumerate(view_ids)}
                if len(index) != len(view_ids):
                    raise ValueError('Duplicate sample IDs in a late-fusion view')
                aligned.append(view_x[[index[sid] for sid in ordered]])
            xs = aligned
            pd.DataFrame({'sample_id': ordered}).to_csv(out / 'FI_evaluation_pairs.csv', index=False)
            print(f'Late fusion: {len(ordered)} matched subjects; FI uses these paired samples.')
        else:
            pairing = cfg.get('late_fusion', {}).get('unmatched_strategy', 'independent_pairs')
            if pairing != 'independent_pairs':
                raise ValueError('late_fusion.unmatched_strategy must be independent_pairs')
            seed = int(cfg.get('late_fusion', {}).get('pairing_seed', 42))
            n_pairs = int(cfg.get('late_fusion', {}).get('num_reference_pairs', max(len(v) for v in xs)))
            if n_pairs < 1:
                raise ValueError('late_fusion.num_reference_pairs must be positive')
            rng = np.random.default_rng(seed)
            # Sampling with replacement, independently per view.
            pair_indices = [rng.integers(0, len(v), size=n_pairs) for v in xs]
            pairing_table = pd.DataFrame({
                f'view{j}_sample_id': [ids[j - 1][i] for i in idx]
                for j, idx in enumerate(pair_indices, start=1)
            })
            pairing_table.to_csv(out / 'FI_evaluation_pairs.csv', index=False)
            xs = [v[idx] for v, idx in zip(xs, pair_indices)]
            print(f'Late fusion: no matched subjects. Evaluating {n_pairs} independently '
                  'sampled reference pairs (not actual paired subjects). '
                  'Cross-view interactions describe a product-of-marginals reference.')
    elif any(v.shape[0] != xs[0].shape[0] for v in xs):
        raise ValueError('Early-fusion views must have matching sample counts')
    x = torch.cat(xs, dim=1)
    compute = cfg.get('compute', {})
    batch_size = int(compute.get('batch_size', 16))
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    fi_cfg = cfg.get('feature_importance', {})
    interaction_cfg = cfg.get('interaction', {})
    run_importance = fi_cfg.get('enabled', True)
    run_interaction = interaction_cfg.get('enabled', True)
    if not isinstance(run_importance, bool) or not isinstance(run_interaction, bool):
        raise ValueError('feature_importance.enabled and interaction.enabled must be true or false')
    if not run_importance and not run_interaction:
        raise ValueError('Enable at least one of feature_importance or interaction in the YAML file')
    mask = str(fi_cfg.get('mask', 'mean')).lower()
    reference = make_reference(x, mask)
    pd.DataFrame([reference.cpu().numpy()], columns=names).to_csv(out / 'reference.csv', index=False)
    print(f'Masked features take the {mask} reference (saved to reference.csv).', flush=True)

    if run_importance:
        fi_permutations = int(fi_cfg.get('permutations', 50))
        fi_seed = int(fi_cfg.get('seed', 42))
        _fi_start = time.perf_counter()
        try:
            scores = shapley_values(model, x, dims, fi_permutations, batch_size, fi_seed, reference)
            pd.DataFrame(scores, columns=names).to_csv(out / 'FI_matrix.csv', index=False)
        finally:
            _fi_elapsed = time.perf_counter() - _fi_start
            print(f'COSIME_FI feature importance run time: {_fi_elapsed:.2f} seconds '
                  f'({_fi_elapsed / 60:.2f} minutes; {_fi_elapsed / 3600:.2f} hours)', flush=True)
        print(f'FI_matrix.csv: Monte Carlo permutation Shapley values ({mask} reference).')
    else:
        print('Feature importance disabled; skipping Shapley values.', flush=True)

    if run_interaction:
        method = interaction_cfg.get('method', 'shapley_taylor').strip().lower().replace('-', '_').replace(' ', '_')
        if method not in ('shapley_taylor', 'integrated_hessian', 'integrated_hessians'):
            raise ValueError('interaction.method must be shapley_taylor or integrated_hessian')
        _interaction_start = time.perf_counter()
        try:
            if method == 'shapley_taylor':
                iterations = int(interaction_cfg.get('permutations', 50))
                if iterations < 1:
                    raise ValueError('interaction.permutations must be positive')
                mat = shapley_taylor(model, x, dims, iterations, batch_size,
                                     int(interaction_cfg.get('seed', 42)), reference)
            else:
                steps = int(interaction_cfg.get('integration_steps', 16))
                if steps < 1:
                    raise ValueError('interaction.integration_steps must be positive')
                mat = integrated_hessians(model, x, dims, steps, batch_size, reference)
            pd.DataFrame(mat, index=names, columns=names).to_csv(out / 'interaction_matrix.csv')
        finally:
            _interaction_elapsed = time.perf_counter() - _interaction_start
            print(f'COSIME_FI feature interaction run time: {_interaction_elapsed:.2f} seconds '
                  f'({_interaction_elapsed / 60:.2f} minutes; {_interaction_elapsed / 3600:.2f} hours)', flush=True)
        print(f'Completed {method}. Outputs: {out}.', flush=True)
        print('interaction_matrix.csv: mean pairwise interactions over samples; the diagonal is zero.')
    else:
        print('Feature interactions disabled; skipping interaction analysis.', flush=True)


if __name__ == '__main__':
    _start = time.perf_counter()
    try:
        main()
    finally:
        _elapsed = time.perf_counter() - _start
        print(f'COSIME_FI total run time: {_elapsed:.2f} seconds ({_elapsed / 60:.2f} minutes; {_elapsed / 3600:.2f} hours)', flush=True)
