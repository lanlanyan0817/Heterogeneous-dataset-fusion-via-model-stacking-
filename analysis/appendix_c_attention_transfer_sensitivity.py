"""Appendix C: attention-transfer sensitivity in pooled target domain (Domain 2).

Run from repository root: python analysis/appendix_c_attention_transfer_sensitivity.py
Requires existing Step 06 .npz and posterior .nc files. Never overwrites Step 06.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import arviz as az
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mcpt_core import (split_by_individual, compute_attention_from_counts,
    prepare_mcpt_dataframe, fit_mcpt_scaler, apply_mcpt_scaler, build_mcpt_arrays,
    mcpt_task_logp_et_draws, extract_lab_et_theta_draws_from_idata)

SEEDS = [12,22,32,42,52,62,72,82,92,102]
METHODS = ['scenario_mean','scenario_median','global_mean','uniform']
ATT = ['a_RC','a_OC1','a_OC2','a_DR']
OUT = ROOT/'results'/'analysis'/'appendix_c'
OUT.mkdir(parents=True,exist_ok=True)

def transfer(lab_train, method):
    task = compute_attention_from_counts(lab_train, id_col='ID')
    scenarios = pd.DataFrame({'Scenario':sorted(task['Scenario'].dropna().unique())})
    if method == 'scenario_mean':
        return task.groupby('Scenario',sort=True)[ATT].mean().reset_index()
    if method == 'scenario_median':
        # Historical definition: component-wise median, without renormalization.
        return task.groupby('Scenario',sort=True)[ATT].median().reset_index()
    if method == 'global_mean':
        a = task[ATT].mean()
        for col in ATT: scenarios[col] = float(a[col])
        return scenarios
    if method == 'uniform':
        for col in ATT: scenarios[col] = .25
        return scenarios
    raise ValueError(method)

def keys(arr):
    return list(arr.task_index[['ID','Scenario']].itertuples(index=False,name=None))

def logmeanexp(a):
    return logsumexp(a,axis=0)-np.log(a.shape[0])

def weights(lp_web, lp_lab):
    def objective(z):
        w = expit(z[0])
        return -float(np.sum(np.logaddexp(np.log(w)+lp_web,np.log1p(-w)+lp_lab)))
    fit=minimize(objective,np.array([0.0]),method='BFGS',tol=1e-9)
    w=float(expit(fit.x[0]))
    return w,1-w,bool(fit.success),str(fit.message)

def mixture(a,b,w):
    return np.logaddexp(np.log(w)+a,np.log1p(-w)+b)

def checked_npz(path):
    with np.load(path) as z:
        expected={'lp_web_train':404,'lp_lab_train':404,'lp_web_test':100,'lp_lab_test':100}
        data={k:np.asarray(z[k],dtype=float) for k in expected}
    for k,n in expected.items():
        if data[k].shape!=(n,) or not np.isfinite(data[k]).all():
            raise ValueError(f'{path}: invalid {k} shape/values')
    return data

def main():
    lab_raw=pd.read_csv(ROOT/'data'/'lab data.csv')
    web_raw=pd.read_csv(ROOT/'data'/'online data.csv')
    rows=[]; diag=[]; attention_diag=[]
    for seed in SEEDS:
        print(f'\nseed {seed}',flush=True)
        lab_train,_,_=split_by_individual(lab_raw,id_col='ID',test_frac=.2,seed=seed)
        web_train,web_test,_=split_by_individual(web_raw,id_col='ID',test_frac=.2,seed=seed)
        base=checked_npz(ROOT/'results'/'predictive_densities'/f'domain2_seed_{seed}.npz')
        scaler=fit_mcpt_scaler(prepare_mcpt_dataframe(lab_train,id_col='ID'))
        web_train_norm=apply_mcpt_scaler(prepare_mcpt_dataframe(web_train,id_col='ID'),scaler)
        web_test_norm=apply_mcpt_scaler(prepare_mcpt_dataframe(web_test,id_col='ID'),scaler)
        posterior=az.from_netcdf(ROOT/'posterior'/f'idata_lab_et_stan_{seed}.nc')
        theta=extract_lab_et_theta_draws_from_idata(posterior,model_prefix='')
        baseline_lp=None; baseline_att=None
        for method in METHODS:
            proxy=transfer(lab_train,method)
            att_sum=proxy[ATT].sum(axis=1)
            for _,r in proxy.iterrows():
                attention_diag.append(dict(seed=seed,method=method,scenario=r['Scenario'],
                    **{c:float(r[c]) for c in ATT},attention_sum=float(sum(r[c] for c in ATT))))
            train=build_mcpt_arrays(web_train_norm,id_col='ID',scenario_attention=proxy)
            test=build_mcpt_arrays(web_test_norm,id_col='ID',scenario_attention=proxy)
            assert len(train.y)==244 and len(test.y)==60
            if baseline_att is None: baseline_att=(train.A.copy(),test.A.copy())
            else:
                pass
            missing_train=int((~train.task_index['Scenario'].isin(proxy['Scenario'])).sum())
            missing_test=int((~test.task_index['Scenario'].isin(proxy['Scenario'])).sum())
            if method=='scenario_mean':
                lp_lab_train=base['lp_lab_train'][:244].copy()
                lp_lab_test=base['lp_lab_test'][:60].copy()
                # Independently recompute once, to verify canonical ordering AND numerical identity.
            draws_train=mcpt_task_logp_et_draws(theta,train.X,train.p_weight,train.refer,train.y,train.A,use_ev=False)
            draws_test=mcpt_task_logp_et_draws(theta,test.X,test.p_weight,test.refer,test.y,test.A,use_ev=False)
            good_train=np.isfinite(draws_train).all(axis=1)
            good_test=np.isfinite(draws_test).all(axis=1)
            if not good_train.any() or not good_test.any(): raise ValueError('No finite draws')
            calc_train=logmeanexp(draws_train[good_train]); calc_test=logmeanexp(draws_test[good_test])
            if method=='scenario_mean':
                err_train=float(np.max(np.abs(calc_train-base['lp_lab_train'][:244])))
                err_test=float(np.max(np.abs(calc_test-base['lp_lab_test'][:60])))
                if err_train>1e-8 or err_test>1e-8:
                    raise AssertionError(f'seed {seed}: scenario mean mismatch with Step06; train={err_train}, test={err_test}')
                baseline_lp=(calc_train.copy(),calc_test.copy())
            else:
                lp_lab_train=calc_train; lp_lab_test=calc_test
            lab_train_full=np.concatenate([lp_lab_train,base['lp_lab_train'][244:]])
            lab_test_full=np.concatenate([lp_lab_test,base['lp_lab_test'][60:]])
            wweb,wlab,ok,message=weights(base['lp_web_train'],lab_train_full)
            lp_mix=mixture(base['lp_web_test'],lab_test_full,wweb)
            rows.append(dict(seed=seed,method=method,w_street=wweb,w_lab=wlab,
                avg_log_score=float(np.mean(lp_mix)),variance=float(np.var(lp_mix)),
                optimizer_success=ok,optimizer_message=message))
            diag.append(dict(seed=seed,method=method,
                max_abs_lab_on_web_train_change=float(np.max(np.abs(calc_train-baseline_lp[0]))),
                mean_abs_lab_on_web_train_change=float(np.mean(np.abs(calc_train-baseline_lp[0]))),
                max_abs_lab_on_web_test_change=float(np.max(np.abs(calc_test-baseline_lp[1]))),
                mean_abs_lab_on_web_test_change=float(np.mean(np.abs(calc_test-baseline_lp[1]))),
                mean_abs_attention_train_change=float(np.mean(np.abs(train.A-baseline_att[0]))),
                mean_abs_attention_test_change=float(np.mean(np.abs(test.A-baseline_att[1]))),
                n_missing_scenario_train=missing_train,n_missing_scenario_test=missing_test,
                n_finite_draws_train=int(good_train.sum()),n_finite_draws_test=int(good_test.sum()),
                min_median_attention_sum=float(att_sum.min()) if method=='scenario_median' else np.nan,
                max_median_attention_sum=float(att_sum.max()) if method=='scenario_median' else np.nan))
            print(f'  {method:16} w_street={wweb:.6f} avg={np.mean(lp_mix):.6f} var={np.var(lp_mix):.6f}',flush=True)
    detail=pd.DataFrame(rows)
    summary=detail.groupby('method',sort=False)[['w_street','w_lab','avg_log_score','variance']].mean().reset_index()
    summary['n_seeds']=detail.groupby('method',sort=False).size().to_numpy()
    detail.to_csv(OUT/'table10_by_seed.csv',index=False)
    summary.to_csv(OUT/'table10_summary.csv',index=False)
    pd.DataFrame(diag).to_csv(OUT/'attention_transfer_diagnostics.csv',index=False)
    pd.DataFrame(attention_diag).to_csv(OUT/'attention_vectors_by_seed.csv',index=False)
    print('\nAPPENDIX C — TABLE 10 (10-seed means)')
    print(summary.to_string(index=False,float_format=lambda v:f'{v:.6f}'))
    print('\nSaved:',OUT)

if __name__=='__main__': main()
