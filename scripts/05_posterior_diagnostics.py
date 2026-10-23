"""Read-only diagnostics for 20 re-estimated Stan InferenceData NetCDF files.

Run from anywhere: python scripts/05_posterior_diagnostics.py
Outputs results/posterior_diagnostics.csv and results/posterior_parameter_summary.csv.
Does not sample or modify posterior files.
"""
from pathlib import Path
import sys
import warnings

import arviz as az
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
POSTERIOR = ROOT / 'posterior'
RESULTS = ROOT / 'results'
SEEDS = [12, 22, 32, 42, 52, 62, 72, 82, 92, 102]
MODELS = [('Web', 'idata_web_stan_', 244), ('Lab', 'idata_lab_et_stan_', 160)]


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    records, parameters, errors = [], [], []
    for model, prefix, expected_tasks in MODELS:
        for seed in SEEDS:
            path = POSTERIOR / f'{prefix}{seed}.nc'
            if not path.is_file():
                errors.append(f'MISSING: {path}')
                continue
            try:
                idata = az.from_netcdf(path)
                if not hasattr(idata, 'posterior'):
                    raise ValueError('No posterior group')
                post = idata.posterior
                chains = int(post.sizes.get('chain', 0))
                draws = int(post.sizes.get('draw', 0))
                # Stan scalar parameters, excluding generated quantities and
                # task-level output arrays, to keep diagnostics interpretable.
                excluded = {'log_lik', 'choice_prob', 'lp__'}
                names = [name for name, arr in post.data_vars.items()
                         if name not in excluded and arr.ndim == 2
                         and set(arr.dims) == {'chain', 'draw'}]
                if not names:
                    raise ValueError('No scalar posterior parameters found')
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    summary = az.summary(idata, var_names=names, kind='all', round_to=None)
                for param, s in summary.iterrows():
                    parameters.append({
                        'model': model, 'seed': seed, 'parameter': param,
                        'mean': s.get('mean', np.nan), 'sd': s.get('sd', np.nan),
                        'r_hat': s.get('r_hat', np.nan),
                        'ess_bulk': s.get('ess_bulk', np.nan),
                        'ess_tail': s.get('ess_tail', np.nan),
                    })
                if hasattr(idata, 'sample_stats') and 'diverging' in idata.sample_stats:
                    divergences = int(idata.sample_stats['diverging'].sum().item())
                else:
                    divergences = np.nan
                task_count = np.nan
                if hasattr(idata, 'log_likelihood') and 'log_lik' in idata.log_likelihood:
                    ll = idata.log_likelihood['log_lik']
                    task_dims = [d for d in ll.dims if d not in ('chain', 'draw')]
                    task_count = int(np.prod([ll.sizes[d] for d in task_dims]))
                max_rhat = float(summary['r_hat'].max())
                min_bulk = float(summary['ess_bulk'].min())
                min_tail = float(summary['ess_tail'].min())
                status = ('CHECK' if chains != 4 or draws != 1000
                          or task_count != expected_tasks
                          or not np.isfinite(max_rhat)
                          or max_rhat > 1.01
                          or min_bulk < 400 or min_tail < 400
                          or not np.isfinite(divergences) or divergences > 0
                          else 'PASS')
                records.append({
                    'model': model, 'seed': seed, 'file': str(path.relative_to(ROOT)),
                    'chains': chains, 'draws_per_chain': draws,
                    'log_lik_tasks': task_count, 'expected_tasks': expected_tasks,
                    'num_scalar_parameters': len(names),
                    'max_r_hat': max_rhat, 'min_ess_bulk': min_bulk,
                    'min_ess_tail': min_tail, 'divergences': divergences,
                    'status': status,
                })
                print(f'{status:5s} {model:3s} seed={seed:3d}: '
                      f'Rhat={max_rhat:.4f}, bulkESS={min_bulk:.0f}, '
                      f'tailESS={min_tail:.0f}, divergences={divergences}, '
                      f'tasks={task_count}')
            except Exception as exc:
                errors.append(f'ERROR {path}: {type(exc).__name__}: {exc}')
    pd.DataFrame(records).to_csv(RESULTS / 'posterior_diagnostics.csv', index=False)
    pd.DataFrame(parameters).to_csv(RESULTS / 'posterior_parameter_summary.csv', index=False)
    print(f'\nChecked {len(records)}/20 posterior files; '
          f'{sum(r["status"] == "PASS" for r in records)} PASS, '
          f'{sum(r["status"] == "CHECK" for r in records)} CHECK, '
          f'{len(errors)} errors.')
    print('Saved results/posterior_diagnostics.csv and results/posterior_parameter_summary.csv')
    for err in errors:
        print(err, file=sys.stderr)
    if errors:
        sys.exit(1)


if __name__ == '__main__':
    main()
