"""Section 4.4: composition sensitivity of PSIS-LOO stacking.
Uses Step 06 saved Domain-2 pointwise densities; no Stan rerun.
Run from model_stacking: python analysis/section_4_4_composition_sensitivity.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mcpt_core import (split_by_individual, prepare_mcpt_dataframe, fit_mcpt_scaler,
                       apply_mcpt_scaler, build_mcpt_arrays)

SEEDS = [12,22,32,42,52,62,72,82,92,102]
SHARES = [0.2,0.3,0.4,0.5]
REPEATS = 100
BASE_SEED = 20260913
TOTAL = 300
OUT = ROOT / 'results' / 'analysis' / 'section_4_4'
OUT.mkdir(parents=True,exist_ok=True)

def task_index(raw, seed):
    train, _, _ = split_by_individual(raw, id_col='ID', test_frac=0.2, seed=seed)
    prep = prepare_mcpt_dataframe(train, id_col='ID')
    scaler = fit_mcpt_scaler(prep)
    arr = build_mcpt_arrays(apply_mcpt_scaler(prep,scaler),id_col='ID')
    return arr.task_index.reset_index(drop=True)

def stacking_weight(web, lab):
    def neg_score(w):
        return -np.sum(np.logaddexp(np.log(w)+web, np.log1p(-w)+lab)) if 0 < w < 1 else -np.sum(web if w==1 else lab)
    fit = minimize_scalar(neg_score, bounds=(1e-12,1-1e-12),method='bounded',options={'xatol':1e-12})
    candidates = [(0.,neg_score(0.)),(float(fit.x),float(fit.fun)),(1.,neg_score(1.))]
    return min(candidates,key=lambda x:x[1])[0]

def pooled_score(web,lab,w):
    if w == 0: return lab.copy()
    if w == 1: return web.copy()
    return np.logaddexp(np.log(w)+web,np.log1p(-w)+lab)

def main():
    web_raw=pd.read_csv(ROOT/'data'/'online data.csv')
    lab_raw=pd.read_csv(ROOT/'data'/'lab data.csv')
    rows=[]
    for seed in SEEDS:
        wi=task_index(web_raw,seed)
        li=task_index(lab_raw,seed)
        with np.load(ROOT/'results'/'predictive_densities'/f'domain2_seed_{seed}.npz') as d:
            wt=d['lp_web_train'].copy(); lt=d['lp_lab_train'].copy()
            we=d['lp_web_test'].copy(); le=d['lp_lab_test'].copy()
        assert len(wi)==244 and len(li)==160 and len(wt)==len(lt)==404 and len(we)==len(le)==100
        assert wi['ID'].astype(str).nunique()==122 and li['ID'].astype(str).nunique()==32
        web_ids=wi['ID'].astype(str).drop_duplicates().to_numpy()
        lab_ids=li['ID'].astype(str).drop_duplicates().to_numpy()
        # The saved Domain-2 arrays are ordered Web1 (244), Lab1 (160).
        for share_idx,share in enumerate(SHARES):
            n_lab=round(TOTAL*share); n_web=TOTAL-n_lab
            assert n_lab%5==0 and n_web%2==0
            for repeat in range(REPEATS):
                rng=np.random.default_rng(np.random.SeedSequence([BASE_SEED,seed,share_idx,repeat]))
                selected_web=rng.choice(web_ids,size=n_web//2,replace=False)
                selected_lab=rng.choice(lab_ids,size=n_lab//5,replace=False)
                wp=np.flatnonzero(wi['ID'].astype(str).isin(selected_web).to_numpy())
                lp=np.flatnonzero(li['ID'].astype(str).isin(selected_lab).to_numpy())
                assert len(wp)==n_web and len(lp)==n_lab
                pos=np.r_[wp,244+lp]
                w=stacking_weight(wt[pos],lt[pos])
                test=pooled_score(we,le,w)
                rows.append(dict(seed=seed,lab_share=share,web_share=1-share,repeat=repeat,
                                 n_lab_tasks=n_lab,n_web_tasks=n_web,w_web=w,w_lab=1-w,
                                 test_avg=float(test.mean()),test_var=float(test.var())))
        print(f'PASS seed={seed}: {len(SHARES)*REPEATS} composition subsamples',flush=True)
    reps=pd.DataFrame(rows)
    by_seed=reps.groupby(['seed','lab_share'],as_index=False).agg(
        w_web=('w_web','mean'),w_lab=('w_lab','mean'),test_avg=('test_avg','mean'),test_var=('test_var','mean'))
    summary=by_seed.groupby('lab_share',as_index=False).agg(
        w_web=('w_web','mean'),w_web_sd=('w_web','std'),
        w_lab=('w_lab','mean'),w_lab_sd=('w_lab','std'),
        test_avg=('test_avg','mean'),test_avg_sd=('test_avg','std'),test_var=('test_var','mean'))
    summary['web_share']=1-summary['lab_share']
    summary['lab_tasks']=(TOTAL*summary['lab_share']).astype(int)
    summary['web_tasks']=TOTAL-summary['lab_tasks']
    reps.to_csv(OUT/'composition_replicates.csv',index=False)
    by_seed.to_csv(OUT/'composition_by_seed.csv',index=False)
    summary.to_csv(OUT/'composition_summary.csv',index=False)
    table=summary[['lab_share','web_share','lab_tasks','web_tasks','w_web','w_lab','test_avg','test_var']]
    table.to_csv(OUT/'table5.csv',index=False)
    with open(OUT/'table5.tex','w') as f:
        f.write(table.to_latex(index=False,float_format=lambda x:f'{x:.3f}'))
    # Separate panels allow Overleaf subfigure references Figure 9(a), 9(b).
    x=summary['lab_share'].to_numpy()*100
    fig,ax=plt.subplots(figsize=(5.5,3.7))
    ax.errorbar(x,summary['w_lab'],yerr=summary['w_lab_sd'],marker='o',capsize=3,label='Lab model')
    ax.errorbar(x,summary['w_web'],yerr=summary['w_web_sd'],marker='s',capsize=3,label='Street model')
    ax.plot(x,x/100,'--',label='Lab share')
    ax.set(xlabel='Lab share in pooled target training data (%)',ylabel='Stacking weight',ylim=(0,1))
    ax.legend();fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(OUT/f'figure9a_weights.{ext}',dpi=300)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.5,3.7))
    ax.errorbar(x,summary['test_avg'],yerr=summary['test_avg_sd'],marker='o',capsize=3)
    ax.set(xlabel='Lab share in pooled target training data (%)',ylabel='Average test log predictive density')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(OUT/f'figure9b_performance.{ext}',dpi=300)
    plt.close(fig)
    print('\nTABLE 5 (new run):\n'+table.to_string(index=False,float_format=lambda v:f'{v:.4f}'))
    print(f'\nPASS: {len(reps)} replicates; files saved to {OUT}')
    print('Error bars = SD across 10 seed-level means (100 subsamples averaged within each seed).')

if __name__=='__main__': main()
