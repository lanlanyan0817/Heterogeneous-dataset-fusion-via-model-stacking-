"""Step 07: three constant-weight Bayesian predictive fusion methods.

Reads the pointwise log predictive densities saved by Step 06; does not
re-estimate Stan models or recompute likelihoods. Weights are learned from
training observations and evaluated unchanged on held-out test observations.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'results' / 'predictive_densities'
OUTPUT = ROOT / 'results' / 'model_fusion'
SEEDS = [12, 22, 32, 42, 52, 62, 72, 82, 92, 102]
DOMAINS = {'domain1': (244, 60), 'domain2': (404, 100)}
BB_SEED = 42
N_BOOT = 1000


def log_mixture(lp_web, lp_lab, weights):
    """Pointwise log(w_web*p_web + w_lab*p_lab), stable at zero weights."""
    a = np.log(weights[0]) + lp_web if weights[0] > 0 else np.full_like(lp_web, -np.inf)
    b = np.log(weights[1]) + lp_lab if weights[1] > 0 else np.full_like(lp_lab, -np.inf)
    return np.logaddexp(a, b)


def stacking_weights(lp_web, lp_lab):
    # Optimize on the closed simplex, including the boundary solutions.
    opt = minimize_scalar(lambda w: -float(np.sum(log_mixture(lp_web, lp_lab, [w, 1-w]))),
                          bounds=(0.0, 1.0), method='bounded',
                          options={'xatol': 1e-12})
    if not opt.success:
        raise RuntimeError(f'Stacking optimization failed: {opt.message}')
    candidates = [0.0, float(opt.x), 1.0]
    w = min(candidates, key=lambda x: -float(np.sum(log_mixture(lp_web, lp_lab, [x, 1-x]))))
    return np.array([w, 1-w], dtype=float)


def pseudo_bma_weights(lp_web, lp_lab):
    elpd = np.array([lp_web.sum(), lp_lab.sum()])
    weights = np.exp(elpd - np.max(elpd))
    return weights / weights.sum()


def bb_pseudo_bma_weights(lp_web, lp_lab, n_boot=N_BOOT, seed=BB_SEED):
    rng = np.random.default_rng(seed)
    n = len(lp_web)
    pointwise = np.vstack([lp_web, lp_lab])
    mean_weights = np.zeros(2, dtype=float)
    for _ in range(n_boot):
        bb = rng.dirichlet(np.ones(n))
        elpd = n * (pointwise @ bb)
        weights = np.exp(elpd - np.max(elpd))
        mean_weights += weights / weights.sum()
    return mean_weights / n_boot


def load_density(domain, seed, n_train, n_test):
    path = INPUT / f'{domain}_seed_{seed}.npz'
    if not path.exists():
        raise FileNotFoundError(f'Missing Step 06 output: {path}')
    with np.load(path, allow_pickle=False) as f:
        out = {key: np.asarray(f[key], dtype=float) for key in
               ('lp_web_train', 'lp_lab_train', 'lp_web_test', 'lp_lab_test')}
    for key, value in out.items():
        expected = n_train if key.endswith('train') else n_test
        if value.shape != (expected,) or not np.isfinite(value).all():
            raise ValueError(f'{path}: invalid {key}, shape={value.shape}')
    return out


def scores(arr):
    return float(np.mean(arr)), float(np.var(arr))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for domain, (n_train, n_test) in DOMAINS.items():
        train_rows, test_rows = [], []
        for seed in SEEDS:
            data = load_density(domain, seed, n_train, n_test)
            wt = data['lp_web_train']; lt = data['lp_lab_train']
            we = data['lp_web_test']; le = data['lp_lab_test']
            methods = {
                'stacking': stacking_weights(wt, lt),
                'pseudo-BMA': pseudo_bma_weights(wt, lt),
                'BB-pseudo-BMA': bb_pseudo_bma_weights(wt, lt),
            }
            for method, weights in methods.items():
                if not np.isfinite(weights).all() or not np.isclose(weights.sum(), 1, atol=1e-10):
                    raise ValueError(f'Invalid weights: {domain} seed {seed} {method}: {weights}')
                train_lp = log_mixture(wt, lt, weights)
                test_lp = log_mixture(we, le, weights)
                train_avg, train_var = scores(train_lp)
                test_avg, test_var = scores(test_lp)
                web_avg, web_var = scores(wt)
                lab_avg, lab_var = scores(lt)
                train_rows.append(dict(seed=seed, method=method, w_web=weights[0], w_lab=weights[1],
                                       stack_avg=train_avg, stack_var=train_var, web_avg=web_avg,
                                       web_var=web_var, lab_avg=lab_avg, lab_var=lab_var))
                test_rows.append(dict(seed=seed, method=method, w_web=weights[0], w_lab=weights[1],
                                      test_avg=test_avg, test_var=test_var,
                                      web_test_avg=float(we.mean()), lab_test_avg=float(le.mean())))
            print(f'PASS {domain} seed={seed}: 3 fusion methods; train={n_train}, test={n_test}', flush=True)
        train_df = pd.DataFrame(train_rows)
        test_df = pd.DataFrame(test_rows)
        train_df.to_csv(OUTPUT / f'{domain}_train_splits_stan.csv', index=False)
        test_df.to_csv(OUTPUT / f'{domain}_test_splits_stan.csv', index=False)
        train_summary = train_df.groupby('method', as_index=False).mean(numeric_only=True).drop(columns=['seed'])
        test_summary = test_df.groupby('method', as_index=False).mean(numeric_only=True).drop(columns=['seed'])
        train_summary.to_csv(OUTPUT / f'{domain}_train_summary_stan.csv', index=False)
        test_summary.to_csv(OUTPUT / f'{domain}_test_summary_stan.csv', index=False)
        print(f'\n{domain.upper()} TEST SUMMARY (mean over 10 seeds):\n{test_summary.to_string(index=False)}\n', flush=True)
    print('PASS: 2 domains x 10 seeds x 3 methods; 8 CSV files saved to', OUTPUT)


if __name__ == '__main__':
    main()
