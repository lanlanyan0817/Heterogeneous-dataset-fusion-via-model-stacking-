"""Independent checks for participant splits and MCPT training inputs.

Run after Step 02; does not compile or sample Stan.
"""
from pathlib import Path
import importlib.util
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mcpt_core import (split_by_individual, prepare_mcpt_dataframe,
                       fit_mcpt_scaler, apply_mcpt_scaler, build_mcpt_arrays,
                       compute_attention_from_counts)
MANIFEST = ROOT / 'splits' / 'split_manifest.csv'
SEEDS = [12,22,32,42,52,62,72,82,92,102]
ATT = ['a_RC','a_OC1','a_OC2','a_DR']
ATTR = ['RC','OC','OC_1','DR']

def load_script(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / 'scripts' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def verify_splits():
    if not MANIFEST.exists():
        raise FileNotFoundError(f'Missing {MANIFEST}; run 02_data_split.py first')
    manifest = pd.read_csv(MANIFEST, dtype={'participant_id': str})
    expected_columns = {'dataset', 'seed', 'participant_id', 'assignment'}
    if not expected_columns.issubset(manifest.columns):
        raise ValueError(f'Manifest missing columns: {expected_columns - set(manifest.columns)}')
    if manifest.duplicated(['dataset', 'seed', 'participant_id']).any():
        raise AssertionError('Duplicate dataset/seed/participant_id rows in manifest')
    if not manifest['assignment'].isin(['train', 'test']).all():
        raise AssertionError('Unexpected assignment in manifest')

    configs = [
        ('Lab', '04_lab_ET_mcpt_stan.py', 'lab_estimation', 'LAB_CSV'),
        ('Online', '03_web_mcpt_stan.py', 'web_estimation', 'WEB_CSV'),
    ]
    checks = 0
    for dataset, filename, module_name, csv_attr in configs:
        estimation = load_script(filename, module_name)
        raw = pd.read_csv(getattr(estimation, csv_attr))
        ids = set(raw[estimation.ID_COL].astype(str))
        seeds = list(estimation.SEEDS)
        present_seeds = set(manifest.loc[manifest.dataset == dataset, 'seed'])
        assert present_seeds == set(seeds), f'{dataset}: seed mismatch'
        for seed in seeds:
            train, test, _ = estimation.split_by_individual(
                raw, estimation.ID_COL, test_frac=estimation.TEST_FRAC, seed=seed
            )
            subset = manifest.loc[(manifest.dataset == dataset) & (manifest.seed == seed)]
            manifest_train = set(subset.loc[subset.assignment == 'train', 'participant_id'])
            manifest_test = set(subset.loc[subset.assignment == 'test', 'participant_id'])
            expected_train = set(train[estimation.ID_COL].astype(str))
            expected_test = set(test[estimation.ID_COL].astype(str))
            assert manifest_train == expected_train, f'{dataset}, seed {seed}: TRAIN IDs differ'
            assert manifest_test == expected_test, f'{dataset}, seed {seed}: TEST IDs differ'
            assert manifest_train.isdisjoint(manifest_test), f'{dataset}, seed {seed}: overlapping IDs'
            assert manifest_train | manifest_test == ids, f'{dataset}, seed {seed}: missing/extra IDs'
            checks += 1
            print(f'PASS {dataset:6s} seed={seed:3d}: train={len(expected_train):3d}, test={len(expected_test):2d}')
    assert set(manifest.dataset) == {c[0] for c in configs}, 'Unexpected dataset in manifest'
    print(f'\nPASS: {checks}/{checks} dataset-seed assignments match estimation-script splitting.')
    print('Note: This does not verify downstream evaluation scripts or historical posterior provenance.')

def verify(label, filename, seed, is_lab):
    raw = pd.read_csv(ROOT / 'data' / filename)
    train, test, test_ids = split_by_individual(raw, 'ID', test_frac=.2, seed=seed)
    assert set(train.ID).isdisjoint(set(test.ID))
    assert set(test.ID) == set(test_ids)
    expected_ids = (32,8,160) if is_lab else (122,30,244)
    assert (train.ID.nunique(),test.ID.nunique()) == expected_ids[:2]
    assert len(train) == expected_ids[2]*2

    prep = prepare_mcpt_dataframe(train, id_col='ID')
    scaler = fit_mcpt_scaler(prep)
    norm = apply_mcpt_scaler(prep, scaler)
    attention = compute_attention_from_counts(train, id_col='ID') if is_lab else None
    arr = build_mcpt_arrays(norm, id_col='ID', task_attention=attention)
    T = expected_ids[2]
    assert arr.X.shape == (T,2,4)
    assert arr.refer.shape == (T,1,4)
    assert arr.p_weight.shape == (T,2,4)
    assert arr.y.shape == (T,2)
    for key in ('X','refer','p_weight','y'):
        assert np.isfinite(getattr(arr,key)).all(), (label,seed,key)
    assert np.allclose(arr.y.sum(axis=1),1), (label,seed,'chosen')
    assert (arr.task_index[['ID','Scenario']].drop_duplicates().shape[0] == T)

    # Independently match every array row to its sorted source task.
    grouped = list(norm.sort_values(['ID','Scenario','alt']).groupby(['ID','Scenario'],sort=True))
    assert len(grouped)==T
    for i, ((pid,scen), rows) in enumerate(grouped):
        assert (arr.task_index.iloc[i]['ID'],arr.task_index.iloc[i]['Scenario']) == (pid,scen)
        assert rows['alt'].tolist() == [1,3], (label,seed,pid,scen,'alts')
        assert np.allclose(arr.X[i],rows[ATTR].to_numpy(dtype=float))
        assert np.allclose(arr.y[i],rows['chosen'].to_numpy(dtype=float))
        ev = rows.iloc[1]
        assert np.allclose(arr.p_weight[i,1,1:3], [ev.P1,ev.P2])
        ref = np.array([arr.X[i,0,0],
                        -(arr.X[i,1,1]*ev.P1+arr.X[i,1,2]*ev.P2),
                        -(arr.X[i,1,1]*ev.P1+arr.X[i,1,2]*ev.P2),
                        arr.X[i,0,3]])
        assert np.allclose(arr.refer[i,0],ref)

    # Independently recompute training-only scaling for each attribute.
    for col,low,high,sign in [('RC',scaler.rc_min,scaler.rc_max,-1),
                               ('OC',scaler.oc_min,scaler.oc_max,-1),
                               ('OC_1',scaler.oc1_min,scaler.oc1_max,-1),
                               ('DR',scaler.dr_min,scaler.dr_max,1)]:
        assert np.isclose(low,prep[col].min()) and np.isclose(high,prep[col].max())
        expected = sign*(prep[col].to_numpy()-low)/(high-low+1e-12)
        assert np.allclose(norm[col].to_numpy(),expected), (label,seed,col)

    if is_lab:
        assert arr.A.shape == (T,4) and np.isfinite(arr.A).all()
        assert np.allclose(arr.A.sum(axis=1),1)
        expected_A = arr.task_index.merge(attention,on=['ID','Scenario'],how='left',validate='one_to_one')[ATT].to_numpy()
        assert np.isfinite(expected_A).all(), (label,seed,'missing attention')
        assert np.allclose(arr.A,expected_A)
    else:
        assert arr.A is None
    print(f'PASS {label:6} seed={seed:3}: {T} tasks; scaler, arrays, ordering'+(', attention' if is_lab else ''))

def verify_training_inputs():
    for label, filename, is_lab in [('Web','online data.csv',False),('Lab','lab data.csv',True)]:
        for seed in SEEDS:
            verify(label,filename,seed,is_lab)
    print('\nPASS: 20/20 training input checks; no Stan sampling performed.')
    print('Scope: internal input consistency only; does not verify historical posterior provenance or downstream ordering.')

def main():
    verify_splits()
    verify_training_inputs()
    print('PASS: data splits and training inputs verified.')

if __name__ == '__main__':
    main()
