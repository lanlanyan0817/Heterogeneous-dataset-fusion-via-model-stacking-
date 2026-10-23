"""Section 4.5 / Table 6: logit-based individual-specific stacking.
Run from repo root: python analysis/section_4_5_logit_stacking.py
Uses Step 06 predictive densities and Step 4.5A task-ID manifests.
No Stan runs; no changes to historical files.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data'
DENS = ROOT/'results'/'predictive_densities'
IDS = ROOT/'results'/'logit_stacking'
OUT = ROOT/'results'/'analysis'/'section_4_5'
SEEDS = [12,22,32,42,52,62,72,82,92,102]
COLS = ['Freq_p_week','DIS_mid','DIS_high','AT2_2','AT2_3','AT2_4','EXP_mid','EXP_high']
EPS = 1e-12


def features(df, prefix):
    base = df[['ID','Freq_p_week','Dis_p_day','AT2','experience_months']].drop_duplicates('ID').copy()
    base['participant_id'] = prefix + base['ID'].astype(str)
    freq = base['Freq_p_week'].astype(float)
    dis = base['Dis_p_day'].astype(float)
    exp = base['experience_months'].astype(float)
    at = base['AT2']
    result = pd.DataFrame({'participant_id':base['participant_id'], 'Freq_p_week':freq,
        'DIS_mid':((dis>250)&(dis<=450)).astype(int), 'DIS_high':(dis>450).astype(int),
        'AT2_2':(at==2).astype(int),'AT2_3':(at==3).astype(int),'AT2_4':(at==4).astype(int),
        'EXP_mid':((exp>30)&(exp<=96)).astype(int),'EXP_high':(exp>96).astype(int)})
    if result[COLS].isna().any().any() or result.participant_id.duplicated().any():
        raise ValueError('Invalid individual features')
    return result


def load_manifest(domain,seed,subset):
    p=IDS/f'{domain}_seed_{seed}_{subset}_task_ids.csv'
    m=pd.read_csv(p,dtype={'participant_id':str,'original_id':str,'scenario':str})
    if not np.array_equal(m.position.to_numpy(),np.arange(len(m))):
        raise ValueError(f'Manifest positions invalid: {p}')
    if m.duplicated(['participant_id','scenario']).any():
        raise ValueError(f'Duplicate task keys: {p}')
    return m


def unique_design(manifest, feature_table, scaler=None):
    ids=pd.Index(pd.unique(manifest.participant_id.astype(str)))
    feats=feature_table.set_index('participant_id').loc[ids,COLS].copy()
    if scaler is None:
        scaler=StandardScaler().fit(feats[['Freq_p_week']])
    feats.loc[:,'Freq_p_week']=scaler.transform(feats[['Freq_p_week']]).ravel()
    z=feats.to_numpy(dtype=float)
    lookup={v:i for i,v in enumerate(ids)}
    obs_idx=np.array([lookup[v] for v in manifest.participant_id.astype(str)],dtype=int)
    return z,obs_idx,scaler


def mix_lp(lpw,lpl,wlab):
    wlab=np.clip(wlab,EPS,1.0)
    wweb=np.clip(1.0-wlab,EPS,1.0)
    return np.logaddexp(np.log(wweb)+lpw,np.log(wlab)+lpl)


def fit_logit(lpw,lpl,z,obs_idx):
    def objective(par):
        w=expit(par[0]+z@par[1:])[obs_idx]
        return -float(np.sum(mix_lp(lpw,lpl,w)))+0.1*float(par[1:]@par[1:])
    res=minimize(objective,np.zeros(z.shape[1]+1),method='BFGS')
    if not np.isfinite(res.fun) or not np.isfinite(res.x).all():
        raise RuntimeError('Nonfinite logit fit')
    return res


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    feat=pd.concat([features(pd.read_csv(DATA/'online data.csv'),''),
                    features(pd.read_csv(DATA/'lab data.csv'),'lab_')],ignore_index=True)
    if feat.participant_id.duplicated().any():
        raise ValueError('Cross-source participant IDs collide')
    rows=[]; coeff=[]
    for domain in ['domain1','domain2']:
        for seed in SEEDS:
            mt=load_manifest(domain,seed,'train'); mv=load_manifest(domain,seed,'test')
            if set(mt.participant_id)&set(mv.participant_id):
                raise ValueError(f'Participant train/test overlap: {domain} {seed}')
            zt,it,scaler=unique_design(mt,feat)
            zv,iv,_=unique_design(mv,feat,scaler)
            with np.load(DENS/f'{domain}_seed_{seed}.npz',allow_pickle=False) as d:
                a,b,c,e=(np.asarray(d[k],dtype=float) for k in
                    ['lp_web_train','lp_lab_train','lp_web_test','lp_lab_test'])
            if len(a)!=len(mt) or len(b)!=len(mt) or len(c)!=len(mv) or len(e)!=len(mv):
                raise ValueError(f'Density/task length mismatch: {domain} {seed}')
            res=fit_logit(a,b,zt,it)
            wt=expit(res.x[0]+zt@res.x[1:]); wv=expit(res.x[0]+zv@res.x[1:])
            lp=mix_lp(c,e,wv[iv])
            rows.append({'domain':domain,'seed':seed,'method':'stacking-logit',
                'test_avg':float(lp.mean()),'test_var':float(lp.var()),
                'w_web':float((1-wv).mean()),'w_lab':float(wv.mean()),
                'optimizer_success':bool(res.success),'optimizer_message':str(res.message),
                'objective':float(res.fun),'n_train_persons':len(zt),'n_test_persons':len(zv)})
            coeff.append({'domain':domain,'seed':seed,'intercept':res.x[0],
                          **dict(zip(COLS,res.x[1:]))})
            print(f'{domain} seed={seed:3d}: test_avg={lp.mean():.6f}, '
                  f'test_var={lp.var():.6f}, avg_w_lab={wv.mean():.4f}, '
                  f'BFGS success={res.success}')
    detail=pd.DataFrame(rows)
    detail.to_csv(OUT/'table6_logit_per_seed.csv',index=False)
    pd.DataFrame(coeff).to_csv(OUT/'table6_logit_coefficients.csv',index=False)
    summary=detail.groupby('domain',as_index=False).agg(
        test_avg=('test_avg','mean'),test_var=('test_var','mean'),
        w_web=('w_web','mean'),w_lab=('w_lab','mean'),
        n_seeds=('seed','nunique'),n_optimizer_success=('optimizer_success','sum'))
    summary.to_csv(OUT/'table6_logit_summary.csv',index=False)
    print('\nTABLE 6 — logit-based stacking (10-seed averages):')
    print(summary.to_string(index=False,float_format=lambda v:f'{v:.6f}'))
    print(f'\nSaved to {OUT}')
    print('NOTE: task identities reconstructed from Step 06 ordering; verify against historical results.')

if __name__=='__main__':
    main()
