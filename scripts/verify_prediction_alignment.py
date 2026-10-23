"""Independent checks of cross-model task alignment and saved Stan row order.

Run after Steps 03, 04 and 06; reads existing posterior and prediction files.
No sampling or predictive-density recomputation is performed.
Checks cannot independently establish historical task-ID provenance.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import arviz as az
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mcpt_core import (split_by_individual, prepare_mcpt_dataframe,
                       fit_mcpt_scaler, apply_mcpt_scaler, build_mcpt_arrays,
                       compute_attention_from_counts, scenario_level_attention_transfer)
SEEDS = [12,22,32,42,52,62,72,82,92,102]
DATA = ROOT / 'data'
OUT = ROOT / 'results' / 'diagnostics'

def keys(arr):
    return list(arr.task_index[['ID','Scenario']].itertuples(index=False, name=None))

def check(label, left, right, expected):
    a, b = keys(left), keys(right)
    if len(a) != expected or len(b) != expected:
        raise AssertionError(f'{label}: expected {expected}, got {len(a)} and {len(b)}')
    if len(set(a)) != len(a) or len(set(b)) != len(b):
        raise AssertionError(f'{label}: duplicate task keys')
    if a != b:
        idx = next(i for i,(x,y) in enumerate(zip(a,b)) if x != y)
        raise AssertionError(f'{label}: first mismatch index={idx}, {a[idx]} vs {b[idx]}')
    return {'check':label, 'tasks':expected, 'status':'PASS'}

def build(prep, scaler, **kwargs):
    return build_mcpt_arrays(apply_mcpt_scaler(prep, scaler), id_col='ID', **kwargs)

def verify_cross_model_task_alignment():
    lab_raw = pd.read_csv(DATA / 'lab data.csv')
    web_raw = pd.read_csv(DATA / 'online data.csv')
    records = []
    for seed in SEEDS:
        lab_tr, lab_te, _ = split_by_individual(lab_raw, id_col='ID', test_frac=0.2, seed=seed)
        web_tr, web_te, _ = split_by_individual(web_raw, id_col='ID', test_frac=0.2, seed=seed)
        lp, ltp = prepare_mcpt_dataframe(lab_tr, id_col='ID'), prepare_mcpt_dataframe(lab_te, id_col='ID')
        wp, wtp = prepare_mcpt_dataframe(web_tr, id_col='ID'), prepare_mcpt_dataframe(web_te, id_col='ID')
        ls, ws = fit_mcpt_scaler(lp), fit_mcpt_scaler(wp)
        att_tr = compute_attention_from_counts(lab_tr, id_col='ID')
        att_te = compute_attention_from_counts(lab_te, id_col='ID')
        scen_att = scenario_level_attention_transfer(lab_tr, id_col='ID')
        pairs = [
            ('Web1 train: Web LOO vs Lab OOS', build(wp,ws), build(wp,ls,scenario_attention=scen_att),244),
            ('Lab1 train: Web OOS vs Lab LOO', build(lp,ws), build(lp,ls,task_attention=att_tr),160),
            ('Web2 test: Web vs Lab', build(wtp,ws), build(wtp,ls,scenario_attention=scen_att),60),
            ('Lab2 test: Web vs Lab', build(ltp,ws), build(ltp,ls,task_attention=att_te),40),
        ]
        for label, a, b, expected in pairs:
            records.append({'seed':seed, **check(label,a,b,expected)})
        path = ROOT / 'results' / 'predictive_densities' / f'domain2_seed_{seed}.npz'
        with np.load(path) as z:
            for name,n in [('lp_web_train',404),('lp_lab_train',404),('lp_web_test',100),('lp_lab_test',100)]:
                if z[name].shape != (n,) or not np.isfinite(z[name]).all():
                    raise AssertionError(f'{path.name}: {name} invalid')
        print(f'Seed {seed:3d}: 4/4 task alignment checks PASS; saved arrays PASS')
    target = OUT / 'domain2_task_alignment_verification.csv'
    pd.DataFrame(records).to_csv(target,index=False)
    print(f'\nPASS: {len(records)}/{len(records)} task alignment checks; 10/10 saved result files')
    print(f'Saved: {target}')
    print('NOTE: This validates current preprocessing paths and task keys. It does not independently prove Stan log_lik row ordering.')

def verify(model, seed, raw):
    train, _, _ = split_by_individual(raw, id_col='ID', test_frac=0.2, seed=seed)
    prep = prepare_mcpt_dataframe(train, id_col='ID')
    norm = apply_mcpt_scaler(prep, fit_mcpt_scaler(prep))
    att = compute_attention_from_counts(train, id_col='ID') if model == 'Lab' else None
    arr = build_mcpt_arrays(norm, id_col='ID', task_attention=att)
    name = f'idata_lab_et_stan_{seed}.nc' if model == 'Lab' else f'idata_web_stan_{seed}.nc'
    idata = az.from_netcdf(ROOT / 'posterior' / name)
    T = len(arr.task_index)

    # Independent check of saved observed choices vs task-level preprocessing order.
    assert 'observed_data' in idata.groups(), f'{model} {seed}: no observed_data group'
    saved_y = np.asarray(idata.observed_data['choice'].values)
    expected_y = np.argmax(arr.y, axis=1).astype(int) + 1
    assert saved_y.shape == expected_y.shape, (model,seed,'observed_data shape',saved_y.shape,expected_y.shape)
    np.testing.assert_array_equal(saved_y, expected_y, err_msg=f'{model} {seed}: observed choices misaligned')

    # Verify task coordinate is an unpermuted positional index.
    ll = idata.log_likelihood['log_lik']
    assert ll.dims[-1] == 'task', (model,seed,'log_lik dimensions',ll.dims)
    assert ll.shape[-1] == T, (model,seed,'log_lik task count')
    np.testing.assert_array_equal(ll.coords['task'].values, np.arange(T))

    # Generated quantities include choice_prob[t] calculated from the same U[t]
    # as log_lik[t]. Check this identity for sampled draws and every task.
    assert 'choice_prob' in idata.posterior, (model,seed,'choice_prob missing')
    prob = idata.posterior['choice_prob']
    assert prob.dims[-2:] == ('task','alternative'), (model,seed,'choice_prob dimensions',prob.dims)
    assert prob.shape[-2:] == (T,2), (model,seed,'choice_prob shape')
    np.testing.assert_array_equal(prob.coords['task'].values, np.arange(T))
    # Select up to 3 draws from first chain; enough to detect systematic indexing issues.
    ll_small = np.asarray(ll.isel(chain=0, draw=slice(0,3)).values)
    prob_small = np.asarray(prob.isel(chain=0, draw=slice(0,3)).values)
    chosen = saved_y.astype(int) - 1
    selected_prob = np.take_along_axis(prob_small,chosen[None,:,None],axis=2).squeeze(-1)
    # Allow small floating-point differences from probability/log transformations.
    np.testing.assert_allclose(ll_small,np.log(selected_prob),rtol=1e-6,atol=1e-6,
        err_msg=f'{model} {seed}: log_lik / choice_prob row mismatch')
    keys = list(map(tuple,arr.task_index[['ID','Scenario']].itertuples(index=False,name=None)))
    assert len(keys)==len(set(keys))==T, (model,seed,'duplicate task keys')
    print(f'PASS {model:3} seed={seed:3}: {T} tasks; observed choice rows, task coords, log_lik / choice_prob')
    return dict(model=model,seed=seed,tasks=T,observed_choices='PASS',task_coordinates='PASS',
                loglik_choiceprob='PASS')

def verify_saved_stan_loglik_order():
    rows=[]
    for model, filename in [('Web','online data.csv'),('Lab','lab data.csv')]:
        raw = pd.read_csv(ROOT/'data'/filename)
        for seed in SEEDS:
            rows.append(verify(model,seed,raw))
    out=ROOT/'results'/'diagnostics'/'stan_loglik_row_order_verification.csv'
    out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out,index=False)
    print('\nPASS: 20/20 saved posterior files; 60/60 checks')
    print('Saved:',out)
    print('Scope: checks saved observed choices and generated-quantity row consistency.')
    print('Limitation: if distinct tasks have identical choice outcomes, these checks alone')
    print('cannot establish the historical mapping of every posterior row to a unique task ID.')

def main():
    verify_cross_model_task_alignment()
    verify_saved_stan_loglik_order()
    print('PASS: prediction alignment checks completed.')

if __name__ == '__main__':
    main()
