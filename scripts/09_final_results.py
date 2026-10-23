"""Reproduce paper Tables 2-3 and Figures 5-6 from saved per-seed results.

Run: python scripts/09_final_results.py
Requires Step 07 model fusion and Step 08 MLE baseline outputs.
No estimation is performed; existing input files are never modified.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results' / 'baselines'
FUSION = ROOT / 'results' / 'model_fusion'
OUT = ROOT / 'results' / 'final'
SEEDS = [12, 22, 32, 42, 52, 62, 72, 82, 92, 102]
METHODS = ['pseudo-BMA', 'BB-pseudo-BMA', 'stacking']
BASELINES = {'domain1': ['Street', 'Pooled'], 'domain2': ['Street', 'Lab', 'Pooled']}
LABELS = {'Street': 'Street-MCPT (MLE)', 'Lab': 'Lab-MCPT (MLE)',
          'Pooled': 'Pooled-MCPT (MLE)'}
COLORS = {'stacking': '#238b45', 'pseudo-BMA': '#e68632',
          'BB-pseudo-BMA': '#1f77b4', 'Street': '#c83b43',
          'Lab': '#7851a9', 'Pooled': '#846d5b'}


def check_seeds(df, key, path):
    if 'seed' not in df or set(df.seed.astype(int)) != set(SEEDS):
        raise ValueError(f'{path}: wrong or missing seeds')
    if df.duplicated(key).any():
        raise ValueError(f'{path}: duplicate {key}')


def load(domain):
    bp = BASE / f'{domain}_per_seed.csv'
    fp = FUSION / f'{domain}_test_splits_stan.csv'
    b, f = pd.read_csv(bp), pd.read_csv(fp)
    check_seeds(b, ['seed'], bp)
    check_seeds(f, ['seed', 'method'], fp)
    if len(b) != 10 or len(f) != 30 or set(f.method) != set(METHODS):
        raise ValueError(f'{domain}: expected 10 baseline rows and 30 fusion rows')
    required_b = [f'{name}_MCPT_{suffix}' for name in BASELINES[domain] for suffix in ('avg', 'var')]
    required_f = ['test_avg', 'test_var', 'w_web', 'w_lab']
    for col in required_b:
        if col not in b: raise ValueError(f'{bp}: missing {col}')
    for col in required_f:
        if col not in f: raise ValueError(f'{fp}: missing {col}')
    if not np.isfinite(b[required_b].to_numpy(float)).all() or not np.isfinite(f[required_f].to_numpy(float)).all():
        raise ValueError(f'{domain}: non-finite scores')
    if not np.allclose(f.w_web + f.w_lab, 1, atol=1e-8):
        raise ValueError(f'{domain}: invalid weights')
    b = b.set_index('seed').loc[SEEDS].reset_index()
    f = f.sort_values(['seed', 'method']).reset_index(drop=True)
    return b, f


def table_data(domain, b, f):
    rows = []
    for name in BASELINES[domain]:
        rows.append(dict(Model=LABELS[name], Avg_Log_Score=b[f'{name}_MCPT_avg'].mean(),
                         Variance=b[f'{name}_MCPT_var'].mean(),
                         Weights_Street=np.nan, Weights_Lab=np.nan))
    for method in METHODS:
        sub = f.loc[f.method == method]
        rows.append(dict(Model=method, Avg_Log_Score=sub.test_avg.mean(),
                         Variance=sub.test_var.mean(), Weights_Street=sub.w_web.mean(),
                         Weights_Lab=sub.w_lab.mean()))
    return pd.DataFrame(rows)


def write_latex(df, domain):
    title = 'Target Domain 1: Street data only' if domain == 'domain1' else 'Target Domain 2: Pooled Street and Lab data'
    label = 'tab:domain1_performance' if domain == 'domain1' else 'tab:domain2_performance'
    lines = [r'\begin{table}[htbp]', r'\centering',
             rf'\caption{{Predictive performance and model fusion weights ({title}).}}',
             rf'\label{{{label}}}', r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Model & Avg. Log Score & Variance & Weights\_Street & Weights\_Lab \\', r'\midrule']
    for _, row in df.iterrows():
        vals = [row['Model'].replace('_', r'\_')]
        for col in ('Avg_Log_Score', 'Variance', 'Weights_Street', 'Weights_Lab'):
            val = row[col]
            vals.append('--' if pd.isna(val) else f'{val:.4f}')
        lines.append(' & '.join(vals) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    (OUT / f'table_{2 if domain == "domain1" else 3}_{domain}.tex').write_text('\n'.join(lines), encoding='utf-8')


def make_figure(domain, b, f):
    """Save independent panels for LaTeX subfigure references."""
    by_method = {m: f.loc[f.method == m].set_index('seed').loc[SEEDS] for m in METHODS}
    stack = by_method['stacking'].test_avg.to_numpy(float)
    x = np.arange(1, 11)
    fusion_curves = [by_method[m].test_avg.to_numpy(float) for m in METHODS]
    overlap = all(np.allclose(stack, arr, rtol=0, atol=1e-7) for arr in fusion_curves)

    # Panel (a): predictive performance across splits
    fig_a, ax1 = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    if overlap:
        ax1.plot(x, stack, '-o', color=COLORS['stacking'], markersize=4,
                 label='PSIS-LOO stacking / pseudo-BMA / BB-pseudo-BMA')
    else:
        for method in METHODS:
            ax1.plot(x, by_method[method].test_avg.to_numpy(float), '-o', markersize=3,
                     label=method, color=COLORS[method])
    for name in BASELINES[domain]:
        ax1.plot(x, b[f'{name}_MCPT_avg'].to_numpy(float), '-o', markersize=3,
                 color=COLORS[name], label=LABELS[name])
    ax1.set_title('Predictive performance across data splits', fontsize=10)
    ax1.set_xlabel('Data split')
    ax1.set_ylabel('Average log predictive score')
    ax1.legend(fontsize=7, loc='best')
    ax1.set_xticks(x)
    ax1.grid(alpha=0.3)
    ax1.set_axisbelow(True)
    for extension in ('png', 'pdf'):
        fig_a.savefig(OUT / f'performance_across_splits_{domain}.{extension}',
                      dpi=300, bbox_inches='tight')
    plt.close(fig_a)

    # Panel (b): stacking improvement over MLE baselines
    fig_b, ax2 = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    for name in BASELINES[domain]:
        delta = stack - b[f'{name}_MCPT_avg'].to_numpy(float)
        ax2.plot(x, delta, '-o', markersize=4, color=COLORS[name],
                 label=f'PSIS-LOO stacking - {LABELS[name]}')
    ax2.axhline(0, color='gray', ls='--', lw=0.8)
    ax2.set_title('Per-split improvement of PSIS-LOO stacking over baselines', fontsize=10)
    ax2.set_xlabel('Data split')
    ax2.set_ylabel('Improvement in average log score')
    ax2.legend(fontsize=7, loc='best')
    ax2.set_xticks(x)
    ax2.grid(alpha=0.3)
    ax2.set_axisbelow(True)
    for extension in ('png', 'pdf'):
        fig_b.savefig(OUT / f'paired_improvement_vs_baselines_{domain}.{extension}',
                      dpi=300, bbox_inches='tight')
    plt.close(fig_b)
    return overlap




def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for domain in ('domain1', 'domain2'):
        b, f = load(domain)
        table = table_data(domain, b, f)
        n = 2 if domain == 'domain1' else 3
        table.to_csv(OUT / f'table_{n}_{domain}.csv', index=False, float_format='%.12g')
        write_latex(table, domain)
        overlap = make_figure(domain, b, f)
        print(f'\nTABLE {n} — {domain}')
        print(table.to_string(index=False, float_format=lambda v: f'{v:.6f}'))
        print(f'FIGURE {5 if domain == "domain1" else 6}: saved two separate panels (PNG + PDF); fusion curves coincide={overlap}')
    print(f'\nPASS: Tables 2–3 (CSV + LaTeX), Figures 5–6 (four separate panels, PNG + PDF) saved to {OUT}')


if __name__ == '__main__':
    main()
