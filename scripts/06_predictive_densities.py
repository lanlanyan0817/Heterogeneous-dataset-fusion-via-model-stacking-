"""Step 06: pointwise PSIS-LOO and out-of-sample predictive densities.
Adapted from the user's stacking_multisplits_stan.py. No fusion weights estimated here.
Output .npz files store ordered pointwise log predictive densities for Step 07.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import arviz as az

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from mcpt_core import (
    split_by_individual, scenario_level_attention_transfer, compute_attention_from_counts,
    prepare_mcpt_dataframe, fit_mcpt_scaler, apply_mcpt_scaler, build_mcpt_arrays,
    mcpt_task_logp_draws, mcpt_task_logp_et_draws,
    extract_web_theta_draws_from_idata, extract_lab_et_theta_draws_from_idata,
)
DATA_DIR = PROJECT_ROOT / "data"
POSTERIOR_DIR = PROJECT_ROOT / "posterior"
RESULTS_DIR = PROJECT_ROOT / "results" / "predictive_densities"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LAB_CSV = DATA_DIR / "lab data.csv"
WEB_CSV = DATA_DIR / "online data.csv"
SEEDS = [12,22,32,42,52,62,72,82,92,102]
TEST_FRAC = 0.2
ID_COL = "ID"

def task_keys(arr):
    return list(map(tuple, arr.task_index[[ID_COL,"Scenario"]].itertuples(index=False, name=None)))

def assert_keys(a,b,label):
    if a != b:
        mismatch = next(((i,x,y) for i,(x,y) in enumerate(zip(a,b)) if x != y), None)
        raise AssertionError(f"{label}: task ordering mismatch; first difference={mismatch}; lengths={len(a)},{len(b)}")

def check_loo(idata, expected, model, seed):
    loo = az.loo(idata, var_name="log_lik", pointwise=True)
    k = np.asarray(loo.pareto_k.values)
    if len(k) != expected:
        raise ValueError(f"{model} seed {seed}: LOO length {len(k)} != {expected}")
    threshold = min(1 - 1 / np.log10(idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"]), 0.7)
    return {"model":model,"seed":seed,"n_tasks":len(k),"max_pareto_k":float(np.max(k)),"n_k_over_threshold":int(np.sum(k > threshold)),"threshold":float(threshold),"elpd_loo":float(loo.elpd_loo)}

def logmeanexp(a, axis=0):
    a_max = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(
        a_max + np.log(np.mean(np.exp(a - a_max), axis=axis, keepdims=True)),
        axis=axis
    )

def filter_finite_draws(loglik_draws, label):
    """
    Remove posterior draws containing NaN or Inf values
    across any observation.

    Parameters
    ----------
    loglik_draws : np.ndarray
        Shape (n_draws, n_tasks).

    label : str
        Description used in warning messages.
    """
    good_mask = np.isfinite(loglik_draws).all(axis=1)

    n_total = loglik_draws.shape[0]
    n_good = int(np.sum(good_mask))
    n_bad = n_total - n_good

    if n_bad > 0:
        print(
            f"WARNING [{label}]: "
            f"removed {n_bad}/{n_total} "
            f"non-finite posterior draws."
        )

    if n_good == 0:
        raise ValueError(
            f"No finite posterior draws remain for {label}."
        )

    return loglik_draws[good_mask]

def get_domain1_pointwise(seed, lab_raw, web_raw):
    # split
    lab_train, lab_test, _= split_by_individual(lab_raw, id_col=ID_COL, test_frac=TEST_FRAC, seed=seed)
    web_train, web_test, _  = split_by_individual(web_raw, id_col=ID_COL, test_frac=TEST_FRAC, seed=seed)

    # preprocessing
    lab_train_prep = prepare_mcpt_dataframe(lab_train, id_col=ID_COL)
    web_train_prep = prepare_mcpt_dataframe(web_train, id_col=ID_COL)

    lab_scaler = fit_mcpt_scaler(lab_train_prep)
    web_scaler = fit_mcpt_scaler(web_train_prep)
    scen_att = scenario_level_attention_transfer(lab_train, id_col=ID_COL)

    # load Stan posterior
    idata_web = az.from_netcdf(POSTERIOR_DIR / f"idata_web_stan_{seed}.nc")
    idata_lab_et = az.from_netcdf(POSTERIOR_DIR / f"idata_lab_et_stan_{seed}.nc")

    # web model on web-train: LOO
    loo_web = az.loo(idata_web, var_name="log_lik", pointwise=True)
    lp_web = np.asarray(loo_web.loo_i.values, dtype="float64")

    # lab_et model on web-train: OOS
    theta_draws_lab_et = extract_lab_et_theta_draws_from_idata(
        idata_lab_et,
        model_prefix=""
    )

    web_train_for_lab_norm = apply_mcpt_scaler(web_train_prep, lab_scaler)
    web_train_for_lab_et = build_mcpt_arrays(
        web_train_for_lab_norm,
        id_col=ID_COL,
        scenario_attention=scen_att
    )

    loglik_draws_lab_et = mcpt_task_logp_et_draws(  #calcuate the log probability of true observations for each posteior draw, each task 
        theta_draws_lab_et,
        web_train_for_lab_et.X,
        web_train_for_lab_et.p_weight,
        web_train_for_lab_et.refer,
        web_train_for_lab_et.y,
        web_train_for_lab_et.A,
        use_ev=False
    )

    loglik_draws_lab_et = filter_finite_draws(
    loglik_draws_lab_et,
    label=f"seed {seed}: lab model on web train",)
    lp_lab = logmeanexp(loglik_draws_lab_et, axis=0)

    # ============================================================
# (NEW) web model on web_test
# ============================================================

    # 1. get posterior draws
    theta_draws_web = extract_web_theta_draws_from_idata(
        idata_web,
        model_prefix=""
    )

    # 2. apply web scaler


    web_test_prep = prepare_mcpt_dataframe(web_test, id_col=ID_COL)
    web_test_for_web_norm = apply_mcpt_scaler(web_test_prep, web_scaler)

    web_test_for_web = build_mcpt_arrays(
        web_test_for_web_norm,
        id_col=ID_COL
    )

    # 3. calculate log likelihood draws
    loglik_draws_web_test = mcpt_task_logp_draws(  #calcuate the log probability of true observations for each posteior draw, each test task 
        theta_draws_web,
        web_test_for_web.X,
        web_test_for_web.p_weight,
        web_test_for_web.refer,
        web_test_for_web.y,
        use_ev=False
    )

    loglik_draws_web_test = filter_finite_draws(
    loglik_draws_web_test,
    label=f"seed {seed}: web model on web test",)


    lp_web_test = logmeanexp(loglik_draws_web_test, axis=0)


    # apply lab scaler
    web_test_for_lab_norm = apply_mcpt_scaler(web_test_prep, lab_scaler)

    web_test_for_lab_et = build_mcpt_arrays(
        web_test_for_lab_norm,
        id_col=ID_COL,
        scenario_attention=scen_att
    )

    loglik_draws_lab_et_test = mcpt_task_logp_et_draws(  #calcuate the log probability of true observations for each posteior draw, each test task 
        theta_draws_lab_et,
        web_test_for_lab_et.X,
        web_test_for_lab_et.p_weight,
        web_test_for_lab_et.refer,
        web_test_for_lab_et.y,
        web_test_for_lab_et.A,
        use_ev=False
    )

    loglik_draws_lab_et_test = filter_finite_draws(
    loglik_draws_lab_et_test,
    label=f"seed {seed}: lab model on web test",)

    lp_lab_test = logmeanexp(loglik_draws_lab_et_test, axis=0) 

    # Alignment: LOO task ordering must equal the estimation arrays; both model predictions on Web must match.
    expected_web_train = task_keys(build_mcpt_arrays(apply_mcpt_scaler(web_train_prep, web_scaler), id_col=ID_COL))
    assert_keys(expected_web_train, task_keys(web_train_for_lab_et), "Domain1 train Web LOO vs Lab OOS")
    assert_keys(task_keys(web_test_for_web), task_keys(web_test_for_lab_et), "Domain1 test Web vs Lab")
    assert len(lp_web) == len(lp_lab) == len(expected_web_train) == 244
    assert len(lp_web_test) == len(lp_lab_test) == 60

    return lp_web, lp_lab, lp_web_test, lp_lab_test

def get_domain2_pointwise(seed, lab_raw, web_raw):
    # split
    lab_train, lab_test, _ = split_by_individual(
        lab_raw, id_col=ID_COL, test_frac=TEST_FRAC, seed=seed
    )
    web_train, web_test, _ = split_by_individual(
        web_raw, id_col=ID_COL, test_frac=TEST_FRAC, seed=seed
    )

    # Preserve original Lab IDs: prefixing changes task sorting and breaks LOO alignment.

    # attentions for lab data
    lab_train_att = compute_attention_from_counts(lab_train, id_col=ID_COL)
    lab_test_att = compute_attention_from_counts(lab_test, id_col=ID_COL)

    # preprocessing
    lab_train_prep = prepare_mcpt_dataframe(lab_train, id_col=ID_COL)
    web_train_prep = prepare_mcpt_dataframe(web_train, id_col=ID_COL)
    lab_test_prep = prepare_mcpt_dataframe(lab_test, id_col=ID_COL)
    web_test_prep = prepare_mcpt_dataframe(web_test, id_col=ID_COL)

    lab_scaler = fit_mcpt_scaler(lab_train_prep)
    web_scaler = fit_mcpt_scaler(web_train_prep)
    scen_att = scenario_level_attention_transfer(lab_train, id_col=ID_COL)

    # load Stan posterior
    idata_web = az.from_netcdf(POSTERIOR_DIR / f"idata_web_stan_{seed}.nc")
    idata_lab_et = az.from_netcdf(POSTERIOR_DIR / f"idata_lab_et_stan_{seed}.nc")

    theta_draws_web = extract_web_theta_draws_from_idata(
        idata_web,
        model_prefix=""
    )
    theta_draws_lab_et = extract_lab_et_theta_draws_from_idata(
        idata_lab_et,
        model_prefix=""
    )

    # ============================================================
    # Part 1. TRAIN / weight-learning layer
    # ============================================================

    # (A) web on web-train: LOO
    loo_web = az.loo(idata_web, var_name="log_lik", pointwise=True)
    lp_web_on_webtrain = np.asarray(loo_web.loo_i.values, dtype="float64")

    # (B) web on lab-train: OOS
    lab_train_for_web_norm = apply_mcpt_scaler(lab_train_prep, web_scaler)
    lab_train_for_web = build_mcpt_arrays(
        lab_train_for_web_norm,
        id_col=ID_COL
    )

    loglik_draws_web_on_labtrain = mcpt_task_logp_draws(
        theta_draws_web,
        lab_train_for_web.X,
        lab_train_for_web.p_weight,
        lab_train_for_web.refer,
        lab_train_for_web.y,
        use_ev=False
    )

    loglik_draws_web_on_labtrain = filter_finite_draws(
    loglik_draws_web_on_labtrain,
    label=f"seed {seed}: web model on lab train",)
    lp_web_on_labtrain = logmeanexp(loglik_draws_web_on_labtrain, axis=0)

    # (C) lab_et on web-train: OOS
    web_train_for_lab_norm = apply_mcpt_scaler(web_train_prep, lab_scaler)
    web_train_for_lab_et = build_mcpt_arrays(
        web_train_for_lab_norm,
        id_col=ID_COL,
        scenario_attention=scen_att
    )

    loglik_draws_lab_et_on_webtrain = mcpt_task_logp_et_draws(
        theta_draws_lab_et,
        web_train_for_lab_et.X,
        web_train_for_lab_et.p_weight,
        web_train_for_lab_et.refer,
        web_train_for_lab_et.y,
        web_train_for_lab_et.A,
        use_ev=False
    )

    loglik_draws_lab_et_on_webtrain = filter_finite_draws(
    loglik_draws_lab_et_on_webtrain,
    label=f"seed {seed}: lab model on web train",)
    lp_lab_on_webtrain = logmeanexp(loglik_draws_lab_et_on_webtrain, axis=0)

    # (D) lab_et on lab-train: LOO
    loo_lab_et = az.loo(idata_lab_et, var_name="log_lik", pointwise=True)
    lp_lab_on_labtrain = np.asarray(loo_lab_et.loo_i.values, dtype="float64")

    lp_web = np.concatenate([lp_web_on_webtrain, lp_web_on_labtrain], axis=0)
    lp_lab = np.concatenate([lp_lab_on_webtrain, lp_lab_on_labtrain], axis=0)

    # ============================================================
    # Part 2. TEST layer: pooled test = web_test + lab_test
    # ============================================================

    # ---------- web model on web_test ----------
    web_test_for_web_norm = apply_mcpt_scaler(web_test_prep, web_scaler)
    web_test_for_web = build_mcpt_arrays(
        web_test_for_web_norm,
        id_col=ID_COL
    )

    loglik_draws_web_on_webtest = mcpt_task_logp_draws(
        theta_draws_web,
        web_test_for_web.X,
        web_test_for_web.p_weight,
        web_test_for_web.refer,
        web_test_for_web.y,
        use_ev=False
    )

    loglik_draws_web_on_webtest = filter_finite_draws(
    loglik_draws_web_on_webtest,
    label=f"seed {seed}: web model on web test",)
    lp_web_on_webtest = logmeanexp(loglik_draws_web_on_webtest, axis=0)

    # ---------- web model on lab_test ----------
    lab_test_for_web_norm = apply_mcpt_scaler(lab_test_prep, web_scaler)
    lab_test_for_web = build_mcpt_arrays(
        lab_test_for_web_norm,
        id_col=ID_COL
    )

    loglik_draws_web_on_labtest = mcpt_task_logp_draws(
        theta_draws_web,
        lab_test_for_web.X,
        lab_test_for_web.p_weight,
        lab_test_for_web.refer,
        lab_test_for_web.y,
        use_ev=False
    )

    loglik_draws_web_on_labtest = filter_finite_draws(
    loglik_draws_web_on_labtest,
    label=f"seed {seed}: web model on lab test",)
    lp_web_on_labtest = logmeanexp(loglik_draws_web_on_labtest, axis=0)

    # ---------- lab_et model on web_test ----------
    web_test_for_lab_norm = apply_mcpt_scaler(web_test_prep, lab_scaler)
    web_test_for_lab_et = build_mcpt_arrays(
        web_test_for_lab_norm,
        id_col=ID_COL,
        scenario_attention=scen_att
    )

    loglik_draws_lab_on_webtest = mcpt_task_logp_et_draws(
        theta_draws_lab_et,
        web_test_for_lab_et.X,
        web_test_for_lab_et.p_weight,
        web_test_for_lab_et.refer,
        web_test_for_lab_et.y,
        web_test_for_lab_et.A,
        use_ev=False
    )

    loglik_draws_lab_on_webtest = filter_finite_draws(
    loglik_draws_lab_on_webtest,
    label=f"seed {seed}: lab model on web test",)
    lp_lab_on_webtest = logmeanexp(loglik_draws_lab_on_webtest, axis=0)

    # ---------- lab_et model on lab_test ----------
    lab_test_for_lab_norm = apply_mcpt_scaler(lab_test_prep, lab_scaler)
    lab_test_for_lab_et = build_mcpt_arrays(
        lab_test_for_lab_norm,
        id_col=ID_COL,
        task_attention=lab_test_att
    )

    loglik_draws_lab_on_labtest = mcpt_task_logp_et_draws(
        theta_draws_lab_et,
        lab_test_for_lab_et.X,
        lab_test_for_lab_et.p_weight,
        lab_test_for_lab_et.refer,
        lab_test_for_lab_et.y,
        lab_test_for_lab_et.A,
        use_ev=False
    )

    loglik_draws_lab_on_labtest = filter_finite_draws(
    loglik_draws_lab_on_labtest,
    label=f"seed {seed}: lab model on lab test",)
    lp_lab_on_labtest = logmeanexp(loglik_draws_lab_on_labtest, axis=0)

    # pooled test
    lp_web_test = np.concatenate([lp_web_on_webtest, lp_web_on_labtest], axis=0)
    lp_lab_test = np.concatenate([lp_lab_on_webtest, lp_lab_on_labtest], axis=0)

    # Explicitly verify pointwise correspondence, not only array lengths.
    assert_keys(task_keys(web_train_for_lab_et), task_keys(build_mcpt_arrays(apply_mcpt_scaler(web_train_prep, web_scaler), id_col=ID_COL)), "Domain2 Web train LOO/OOS")
    assert_keys(task_keys(lab_train_for_web), task_keys(build_mcpt_arrays(apply_mcpt_scaler(lab_train_prep, lab_scaler), id_col=ID_COL, task_attention=lab_train_att)), "Domain2 Lab train OOS/LOO")
    assert_keys(task_keys(web_test_for_web), task_keys(web_test_for_lab_et), "Domain2 Web test")
    assert_keys(task_keys(lab_test_for_web), task_keys(lab_test_for_lab_et), "Domain2 Lab test")
    assert len(lp_web) == len(lp_lab) == 404
    assert len(lp_web_test) == len(lp_lab_test) == 100

    return lp_web, lp_lab, lp_web_test, lp_lab_test

def main():
    lab_raw = pd.read_csv(LAB_CSV)
    web_raw = pd.read_csv(WEB_CSV)
    diagnostics = []
    for seed in SEEDS:
        print(f"Seed {seed}: computing Domain 1 and Domain 2 predictive densities...", flush=True)
        d1 = get_domain1_pointwise(seed, lab_raw, web_raw)
        d2 = get_domain2_pointwise(seed, lab_raw, web_raw)
        for domain, vals, ntrain, ntest in [("domain1",d1,244,60),("domain2",d2,404,100)]:
            lp_web,lp_lab,lp_web_test,lp_lab_test = vals
            for name,arr,expected in [("lp_web_train",lp_web,ntrain),("lp_lab_train",lp_lab,ntrain),("lp_web_test",lp_web_test,ntest),("lp_lab_test",lp_lab_test,ntest)]:
                if arr.shape != (expected,) or not np.isfinite(arr).all():
                    raise ValueError(f"{domain} seed={seed} {name}: bad shape or nonfinite values: {arr.shape}")
            np.savez_compressed(RESULTS_DIR / f"{domain}_seed_{seed}.npz",lp_web_train=lp_web,lp_lab_train=lp_lab,lp_web_test=lp_web_test,lp_lab_test=lp_lab_test)
            print(f"  PASS {domain}: train={ntrain}, test={ntest}", flush=True)
        # PSIS diagnostics once per model/seed (not twice for two domains)
        for model,filename,n in [("Web",f"idata_web_stan_{seed}.nc",244),("Lab",f"idata_lab_et_stan_{seed}.nc",160)]:
            idata = az.from_netcdf(POSTERIOR_DIR / filename)
            diagnostics.append(check_loo(idata,n,model,seed))
    out = pd.DataFrame(diagnostics)
    out.to_csv(RESULTS_DIR / "psis_pareto_diagnostics.csv",index=False)
    print("PASS: 20 domain-seed predictive-density files saved; 20 PSIS diagnostics rows.")
    print("PSIS observations above threshold:",int(out.n_k_over_threshold.sum()))
    print("Output:",RESULTS_DIR)

if __name__ == "__main__":
    main()
