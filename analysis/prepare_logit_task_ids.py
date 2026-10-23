"""Step 4.5A: reconstruct task-to-person mapping for existing Step 06 NPZ files.
No posterior re-estimation and no predictive-density recalculation.
Run from repository root: python analysis/prepare_logit_task_ids.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mcpt_core import (split_by_individual, prepare_mcpt_dataframe,
                       fit_mcpt_scaler, apply_mcpt_scaler, build_mcpt_arrays)

SEEDS = [12,22,32,42,52,62,72,82,92,102]
DATA = ROOT / 'data'
DENSITIES = ROOT / 'results' / 'predictive_densities'
OUT = ROOT / 'results' / 'logit_stacking'
OUT.mkdir(parents=True, exist_ok=True)

def keys(df, scaler):
    prep = prepare_mcpt_dataframe(df, id_col='ID')
    arr = build_mcpt_arrays(apply_mcpt_scaler(prep, scaler), id_col='ID')
    return arr.task_index[['ID','Scenario']].reset_index(drop=True)

def make_part(df, scaler, prefix, subset):
    idx = keys(df, scaler)
    return pd.DataFrame({'participant_id': prefix + idx['ID'].astype(str),
                         'original_id': idx['ID'].astype(str),
                         'scenario': idx['Scenario'].astype(str),
                         'source': 'lab' if prefix else 'web',
                         'subset': subset})

def check(density_path, parts, domain, seed):
    if not density_path.exists():
        raise FileNotFoundError(f'Missing Step 06 output: {density_path}')
    with np.load(density_path) as z:
        for subset in ['train','test']:
            df = pd.concat(parts[subset], ignore_index=True)
            for model in ['web','lab']:
                arr = z[f'lp_{model}_{subset}']
                if len(arr) != len(df) or not np.isfinite(arr).all():
                    raise ValueError(f'{domain} seed {seed}: {model}/{subset} length or finite mismatch')
            if df[['participant_id','scenario','source']].isna().any().any():
                raise ValueError('Missing task identifiers')
            if df.duplicated(['participant_id','scenario']).any():
                raise ValueError(f'{domain} seed {seed}: duplicate task keys in {subset}')
            df.insert(0,'position',np.arange(len(df)))
            df.insert(0,'seed',seed)
            df.insert(0,'domain',domain)
            df.to_csv(OUT / f'{domain}_seed_{seed}_{subset}_task_ids.csv',index=False)
            yield subset, len(df), df['participant_id'].nunique()

def main():
    web = pd.read_csv(DATA / 'online data.csv')
    lab = pd.read_csv(DATA / 'lab data.csv')
    for seed in SEEDS:
        wt, wv, _ = split_by_individual(web,id_col='ID',test_frac=.2,seed=seed)
        lt, lv, _ = split_by_individual(lab,id_col='ID',test_frac=.2,seed=seed)
        ws = fit_mcpt_scaler(prepare_mcpt_dataframe(wt,id_col='ID'))
        ls = fit_mcpt_scaler(prepare_mcpt_dataframe(lt,id_col='ID'))
        # Reconstruct task order using the same original IDs and source-specific
        # scalers as Step 06. Lab IDs are prefixed only AFTER task ordering.
        parts = {
          'train': [make_part(wt,ws,'','train'),make_part(lt,ls,'lab_','train')],
          'test': [make_part(wv,ws,'','test'),make_part(lv,ls,'lab_','test')]
        }
        for domain in ['domain1','domain2']:
            use = {s: parts[s][:1] if domain=='domain1' else parts[s] for s in ['train','test']}
            result = list(check(DENSITIES / f'{domain}_seed_{seed}.npz',use,domain,seed))
            print(f'PASS {domain} seed={seed}: '+', '.join(f'{s}={n} tasks/{p} persons' for s,n,p in result))
    print(f'\nSaved 40 task-ID manifests to: {OUT}')
    print('Note: alignment with Step 06 is reconstructed from its preprocessing logic;')
    print('the NPZ files do not themselves contain independent task-key metadata.')

if __name__=='__main__':
    main()
