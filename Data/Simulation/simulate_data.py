"""Simulated multi-view data with a known structure.

Each view has the same design:
  features 1-20   linear (main) effects on the outcome: positive for the first half, negative for the second
  features 21-25  main effects and pairwise interactions among themselves
  features 26-100 noise
Late fusion: each view is drawn for its own samples.
"""
import argparse
import numpy as np
import pandas as pd


def simulate_view(rng, name, n, n_features, n_main, block, beta, block_beta, w, outcome):
    x = rng.normal(size=(n, n_features))
    signs = np.where(np.arange(n_main) < n_main // 2, 1.0, -1.0)
    main = x[:, :n_main] @ (beta * signs)
    b = x[:, block[0] - 1:block[1]]
    main = main + b @ np.full(b.shape[1], block_beta)
    inter = w * sum(b[:, i] * b[:, j] for i in range(b.shape[1]) for j in range(i + 1, b.shape[1]))
    eta = main + inter
    if outcome == 'binary':
        y = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    else:
        y = eta + rng.normal(size=n)
    d = pd.DataFrame(x, columns=[f'feature_{name}_{k + 1}' for k in range(n_features)],
                     index=[f'{name}_{i + 1}' for i in range(n)])
    d['y'] = y
    return d, float(np.var(main)), float(np.var(inter))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=3000)
    p.add_argument('--features', type=int, default=100)
    p.add_argument('--main', type=int, default=20)
    p.add_argument('--block', type=int, nargs=2, default=[21, 25])
    p.add_argument('--beta', type=float, default=0.3)
    p.add_argument('--block-beta', type=float, default=0.5)
    p.add_argument('--w', type=float, default=1.0)
    p.add_argument('--outcome', choices=['binary', 'continuous'], default='binary')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--prefix', default='binary_high_late')
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    for k, name in enumerate(['A', 'B']):
        d, vm, vi = simulate_view(rng, name, a.n, a.features, a.main, a.block, a.beta, a.block_beta, a.w, a.outcome)
        d.to_csv(f'{a.prefix}_x{k + 1}.csv')
        print(f'view {name}: variance main {vm:.2f}, interaction {vi:.2f}, outcome mean {d.y.mean():.2f}')


if __name__ == '__main__':
    main()
