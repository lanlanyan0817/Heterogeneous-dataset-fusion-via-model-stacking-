"""Reproduce Section 4.3 Table 4 and Figures 7–8 from verified Step 06/07 outputs.

Usage: python analysis/section_4_3_stacking_gains.py --seed 42
The seed is illustrative; confirm the representative seed used in the paper.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]
DENSITIES = ROOT / 'results' / 'predictive_densities'
FUSION = ROOT / 'results' / 'model_fusion' / 'domain2_test_splits_stan.csv'
OUT = ROOT / 'results' / 'analysis' / 'section_4_3'
SEEDS = [12,22,32,42,52,62,72,82,92,102]
N_STREET, N_LAB = 60, 40


def log_mix(a, b, w):
    x = a + np.log(w) if w > 0 else np.full_like(a, -np.inf)
    y = b + np.log1p(-w) if w < 1 else np.full_like(b, -np.inf)
    return np.logaddexp(x, y)


def load_seed(seed, weights):
    path = DENSITIES / f'domain2_seed_{seed}.npz'
    with np.load(path, allow_pickle=False) as z:
        a, b = z['lp_web_test'].copy(), z['lp_lab_test'].copy()
    assert a.shape == b.shape == (N_STREET + N_LAB,), f'Unexpected test shapes: {seed}'
    assert np.isfinite(a).all() and np.isfinite(b).all(), f'Non-finite values: {seed}'
    s = log_mix(a, b, weights)
    return {'Street-MCPT (Bayesian)': a, 'Lab-MCPT (Bayesian)': b, 'Stacking (PSIS-LOO)': s}


def plot_kde(ax, values, name, color):
    vals = np.asarray(values)
    if np.ptp(vals) < 1e-10:
        ax.axvline(float(vals.mean()), label=name, color=color)
        return
    grid = np.linspace(float(vals.min()) - 0.5, float(vals.max()) + 0.5, 400)
    kde = gaussian_kde(vals)
    density = kde(grid)
    ax.plot(grid, density, color=color, linewidth=1.4, label=name)
    ax.fill_between(grid, density, alpha=0.22, color=color)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, choices=SEEDS,
                        help='Representative split for Figures 7 and 8 (default: 42)')
    args = parser.parse_args()
    if not FUSION.exists():
        raise FileNotFoundError(f'Missing Step 07 results: {FUSION}')
    df = pd.read_csv(FUSION)
    selected = df[df['method'] == 'stacking'].copy()
    if len(selected) != len(SEEDS) or set(selected.seed.astype(int)) != set(SEEDS):
        raise ValueError('Expected exactly one Domain 2 stacking result for each of the 10 seeds')
    selected = selected.set_index('seed')
    OUT.mkdir(parents=True, exist_ok=True)
    rows, per_seed = [], {}
    for seed in SEEDS:
        r = selected.loc[seed]
        w = float(r['w_web'])
        if not (0 <= w <= 1) or not np.isclose(w + float(r['w_lab']), 1, atol=1e-9):
            raise ValueError(f'Invalid stacking weights at seed {seed}')
        values = load_seed(seed, w)
        if not np.isclose(values['Stacking (PSIS-LOO)'].mean(), float(r['test_avg']), atol=1e-9):
            raise AssertionError(f'Stacking pooled score mismatch for seed {seed}')
        per_seed[seed] = values
        for name, vec in values.items():
            rows.append({'seed': seed, 'Model': name,
                         'Street-test': float(vec[:N_STREET].mean()),
                         'Lab-test': float(vec[N_STREET:].mean()),
                         'Pooled': float(vec.mean())})
    detailed = pd.DataFrame(rows)
    detailed.to_csv(OUT / 'table_4_per_seed.csv', index=False)
    summary = detailed.groupby('Model', as_index=False)[['Street-test', 'Lab-test', 'Pooled']].mean()
    order = ['Street-MCPT (Bayesian)', 'Lab-MCPT (Bayesian)', 'Stacking (PSIS-LOO)']
    summary = summary.set_index('Model').loc[order].reset_index()
    summary.to_csv(OUT / 'table_4_domain2_slices.csv', index=False)
    (OUT / 'table_4_domain2_slices.tex').write_text(
        summary.to_latex(index=False, float_format=lambda x: f'{x:.4f}', escape=False), encoding='utf-8')
    print('TABLE 4 (10-seed average):\n' + summary.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    print('\nSlice-weighted pooled consistency:')
    for _, row in summary.iterrows():
        expected = (N_STREET * row['Street-test'] + N_LAB * row['Lab-test']) / (N_STREET + N_LAB)
        assert np.isclose(expected, row['Pooled'], atol=1e-10)
        print(f"  {row['Model']}: PASS")

    values = per_seed[args.seed]
    palette = {'Street-MCPT (Bayesian)': 'tab:blue', 'Lab-MCPT (Bayesian)': 'tab:orange',
               'Stacking (PSIS-LOO)': 'tab:green'}
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
    for ax, sl, title in zip(axes, [slice(0, N_STREET), slice(N_STREET, None)],
                              ['Street-test slice', 'Lab-test slice']):
        for name in order:
            plot_kde(ax, values[name][sl], name, palette[name])
        ax.set_title(title)
        ax.set_xlabel('Log predictive density')
        ax.set_ylabel('Density')
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=3, fontsize=8)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'figure_7_domain2_distributions_seed_{args.seed}.{ext}', dpi=250)
    plt.close(fig)

    delta = values['Stacking (PSIS-LOO)'][N_STREET:] - values['Street-MCPT (Bayesian)'][N_STREET:]
    fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    # Fixed, narrow bins make the count scale comparable with the historical Figure 8.
    # The KDE is rescaled to expected counts per bin (not probability density).
    bin_edges = np.linspace(-0.2, 1.1, 27)
    counts, edges, _ = ax.hist(delta, bins=bin_edges, color='tab:blue', alpha=0.45,
                               edgecolor='white', label='Pointwise improvement')
    if np.ptp(delta) > 1e-10:
        xx = np.linspace(edges[0], edges[-1], 400)
        yy = gaussian_kde(delta)(xx) * len(delta) * np.diff(edges).mean()
        ax.plot(xx, yy, color='tab:blue', linewidth=1.4, label='KDE (count scale)')
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_title('Stacking vs. Street-MCPT (Lab-test slice)')
    ax.set_xlabel('Improvement (log predictive density)')
    ax.set_ylabel('Count')
    ax.grid(alpha=0.2)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'figure_8_lab_test_improvement_seed_{args.seed}.{ext}', dpi=250)
    plt.close(fig)
    pd.DataFrame({'observation': np.arange(1, N_LAB + 1), 'improvement': delta}).to_csv(
        OUT / f'figure_8_pointwise_improvement_seed_{args.seed}.csv', index=False)
    print(f'\nRepresentative seed: {args.seed}; Lab-test positive improvements: {(delta > 0).sum()}/{len(delta)}')
    print(f'Lab-test improvement: mean={delta.mean():.6f}, min={delta.min():.6f}, max={delta.max():.6f}')
    print('Saved Table 4, Figure 7, Figure 8 to:', OUT)
    print('NOTE: Figure styles and representative seed may differ from historical paper figures.')

if __name__ == '__main__':
    main()
