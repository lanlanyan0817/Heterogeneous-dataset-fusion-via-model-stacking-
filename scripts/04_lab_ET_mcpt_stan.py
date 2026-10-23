from pathlib import Path

import numpy as np
import pandas as pd
import arviz as az

from cmdstanpy import CmdStanModel

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcpt_core import (
    split_by_individual,
    compute_attention_from_counts,
    prepare_mcpt_dataframe,
    fit_mcpt_scaler,
    apply_mcpt_scaler,
    build_mcpt_arrays,
)

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
POSTERIOR_DIR = PROJECT_ROOT / "posterior"

LAB_CSV = DATA_DIR / "lab data.csv"

POSTERIOR_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Settings
# ============================================================
SEEDS = [12, 22, 32, 42, 52, 62, 72, 82, 92, 102]
TEST_FRAC = 0.2
ID_COL = "ID"

STAN_SEED = 42
CHAINS = 4
PARALLEL_CHAINS = 4
ITER_WARMUP = 2000
ITER_SAMPLING = 1000

STAN_CODE = r"""
functions {
    real safe_clip(real x, real lo, real hi) {
        if (x < lo) return lo;
        if (x > hi) return hi;
        return x;
    }

    real safe_value(real z, real alpha, real lambda_) {
        real zc;
        real v;

        zc = safe_clip(z, -1e6, 1e6);

        if (zc < 0) {
            v = -lambda_ * pow(-zc, alpha);
        } else {
            v = pow(zc, alpha);
        }

        return safe_clip(v, -1e6, 1e6);
    }
}

data {
    int<lower=1> T;
    array[T] int<lower=1, upper=2> y;      // 1=CV, 2=EV

    array[T, 2, 4] real X;
    array[T, 2, 4] real p_weight;
    array[T, 4] real refer;
    array[T, 4] real A;                    // [a_RC, a_OC1, a_OC2, a_DR]
}

parameters {
    real eta_asc;
    real eta_alpha;
    real eta_lambda;
    real eta_r;

    real eta_beta_rc0;
    real eta_beta_oc0;
    real eta_beta_dr0;

    real eta_gamma_rc;
    real eta_gamma_oc;
    real eta_gamma_dr;
}

transformed parameters {
    real asc;
    real<lower=0, upper=1> alpha;
    real<lower=1> lambda_;
    real<lower=0> r;

    real<lower=0> beta_rc0;
    real<lower=0> beta_oc0;
    real<lower=0> beta_dr0;

    real gamma_rc;
    real gamma_oc;
    real gamma_dr;

    asc = eta_asc;
    alpha = inv_logit(eta_alpha);

    // stabilized positive transforms
    lambda_ = 1 + fmin(exp(eta_lambda), 20);
    r = fmin(exp(eta_r), 3);

    beta_rc0 = fmin(exp(eta_beta_rc0), 20);
    beta_oc0 = fmin(exp(eta_beta_oc0), 20);
    beta_dr0 = fmin(exp(eta_beta_dr0), 20);

    gamma_rc = eta_gamma_rc;
    gamma_oc = eta_gamma_oc;
    gamma_dr = eta_gamma_dr;
}

model {
    // priors
    eta_asc      ~ normal(0, 1.0);
    eta_alpha    ~ normal(0, 0.75);
    eta_lambda   ~ normal(0, 0.3);
    eta_r        ~ normal(0, 0.2);

    eta_beta_rc0 ~ normal(0, 0.3);
    eta_beta_oc0 ~ normal(0, 0.3);
    eta_beta_dr0 ~ normal(0, 0.3);

    eta_gamma_rc ~ normal(0, 0.5);
    eta_gamma_oc ~ normal(0, 0.5);
    eta_gamma_dr ~ normal(0, 0.5);

    // likelihood
    for (t in 1:T) {
        vector[2] U;

        real u_cv;
        real u_ev;

        real gA_rc;
        real gA_oc1;
        real gA_oc2;
        real gA_dr;

        real beta_rc;
        real beta_oc;
        real beta_oc1;
        real beta_dr;

        real z;
        real v;

        real p1;
        real p1_t;
        real p2_t;
 
        // --------------------------------------------------
        // attention-adjusted coefficients
        // g(A) = A - 0.25
        // softplus(x) = log1p_exp(x)
        // --------------------------------------------------
        gA_rc  = A[t, 1] - 0.25;
        gA_oc1 = A[t, 2] - 0.25;
        gA_oc2 = A[t, 3] - 0.25;
        gA_dr  = A[t, 4] - 0.25;

        beta_rc  = log1p_exp(beta_rc0 + gamma_rc * gA_rc);
        beta_oc  = log1p_exp(beta_oc0 + gamma_oc * gA_oc1);
        beta_oc1 = log1p_exp(beta_oc0 + gamma_oc * gA_oc2);
        beta_dr  = log1p_exp(beta_dr0 + gamma_dr * gA_dr);

        // extra safety
        beta_rc  = safe_clip(beta_rc,  1e-12, 1e6);
        beta_oc  = safe_clip(beta_oc,  1e-12, 1e6);
        beta_oc1 = safe_clip(beta_oc1, 1e-12, 1e6);
        beta_dr  = safe_clip(beta_dr,  1e-12, 1e6);

        // --------------------------------------------------
        // probability weighting
        // --------------------------------------------------
        p1 = fmin(fmax(p_weight[t, 2, 2], 1e-12), 1 - 1e-12);
        p1_t = exp(-pow(-log(p1), r));
        p1_t = safe_clip(p1_t, 1e-12, 1 - 1e-12);
        p2_t = 1 - p1_t;
        p2_t = safe_clip(p2_t, 1e-12, 1 - 1e-12);

        // --------------------------------------------------
        // CV utility: RC, OC, DR => [1,2,4]
        // --------------------------------------------------
        u_cv = 0;

        z = (X[t, 1, 1] - refer[t, 1]) * beta_rc;
        v = safe_value(z, alpha, lambda_);
        u_cv += v * p_weight[t, 1, 1];

        z = (X[t, 1, 2] - refer[t, 2]) * beta_oc;
        v = safe_value(z, alpha, lambda_);
        u_cv += v * p_weight[t, 1, 2];

        z = (X[t, 1, 4] - refer[t, 4]) * beta_dr;
        v = safe_value(z, alpha, lambda_);
        u_cv += v * p_weight[t, 1, 4];

        // --------------------------------------------------
        // EV utility: RC, OC, OC_1, DR
        // --------------------------------------------------
        u_ev = 0;

        z = (X[t, 2, 1] - refer[t, 1]) * beta_rc;
        v = safe_value(z, alpha, lambda_);
        u_ev += v * p_weight[t, 2, 1];

        z = (X[t, 2, 2] - refer[t, 2]) * beta_oc;
        v = safe_value(z, alpha, lambda_);
        u_ev += v * p1_t;

        z = (X[t, 2, 3] - refer[t, 3]) * beta_oc1;
        v = safe_value(z, alpha, lambda_);
        u_ev += v * p2_t;

        z = (X[t, 2, 4] - refer[t, 4]) * beta_dr;
        v = safe_value(z, alpha, lambda_);
        u_ev += v * p_weight[t, 2, 4];

        u_ev += asc;

        U[1] = u_cv;
        U[2] = u_ev;

        // stabilized logits
        for (i in 1:2) {
            if (U[i] > 20) U[i] = 20;
            if (U[i] < -20) U[i] = -20;
        }

        y[t] ~ categorical_logit(U);
    }
}

generated quantities {
    vector[T] log_lik;
    array[T] vector[2] choice_prob;

    for (t in 1:T) {
        vector[2] U;
        vector[2] P;

        real u_cv;
        real u_ev;

        real gA_rc;
        real gA_oc1;
        real gA_oc2;
        real gA_dr;

        real beta_rc;
        real beta_oc;
        real beta_oc1;
        real beta_dr;

        real z;
        real v;

        real p1;
        real p1_t;
        real p2_t;

        gA_rc  = A[t, 1] - 0.25;
        gA_oc1 = A[t, 2] - 0.25;
        gA_oc2 = A[t, 3] - 0.25;
        gA_dr  = A[t, 4] - 0.25;

        beta_rc  = log1p_exp(beta_rc0 + gamma_rc * gA_rc);
        beta_oc  = log1p_exp(beta_oc0 + gamma_oc * gA_oc1);
        beta_oc1 = log1p_exp(beta_oc0 + gamma_oc * gA_oc2);
        beta_dr  = log1p_exp(beta_dr0 + gamma_dr * gA_dr);

        beta_rc  = safe_clip(beta_rc,  1e-12, 1e6);
        beta_oc  = safe_clip(beta_oc,  1e-12, 1e6);
        beta_oc1 = safe_clip(beta_oc1, 1e-12, 1e6);
        beta_dr  = safe_clip(beta_dr,  1e-12, 1e6);

        p1 = fmin(fmax(p_weight[t, 2, 2], 1e-12), 1 - 1e-12);
        p1_t = exp(-pow(-log(p1), r));
        p1_t = safe_clip(p1_t, 1e-12, 1 - 1e-12);
        p2_t = 1 - p1_t;
        p2_t = safe_clip(p2_t, 1e-12, 1 - 1e-12);

        u_cv = 0;

        z = (X[t, 1, 1] - refer[t, 1]) * beta_rc;
        v = safe_value(z, alpha, lambda_);
        u_cv += v * p_weight[t, 1, 1];

        z = (X[t, 1, 2] - refer[t, 2]) * beta_oc;
        v = safe_value(z, alpha, lambda_);
        u_cv += v * p_weight[t, 1, 2];

        z = (X[t, 1, 4] - refer[t, 4]) * beta_dr;
        v = safe_value(z, alpha, lambda_);
        u_cv += v * p_weight[t, 1, 4];

        u_ev = 0;

        z = (X[t, 2, 1] - refer[t, 1]) * beta_rc;
        v = safe_value(z, alpha, lambda_);
        u_ev += v * p_weight[t, 2, 1];

        z = (X[t, 2, 2] - refer[t, 2]) * beta_oc;
        v = safe_value(z, alpha, lambda_);
        u_ev += v * p1_t;

        z = (X[t, 2, 3] - refer[t, 3]) * beta_oc1;
        v = safe_value(z, alpha, lambda_);
        u_ev += v * p2_t;

        z = (X[t, 2, 4] - refer[t, 4]) * beta_dr;
        v = safe_value(z, alpha, lambda_);
        u_ev += v * p_weight[t, 2, 4];

        u_ev += asc;

        U[1] = u_cv;
        U[2] = u_ev;

        for (i in 1:2) {
            if (U[i] > 20) U[i] = 20;
            if (U[i] < -20) U[i] = -20;
        }

        log_lik[t] = categorical_logit_lpmf(y[t] | U);

        U = U - max(U);
        P = softmax(U);

        for (i in 1:2) {
            if (P[i] < 1e-12) P[i] = 1e-12;
        }
        P = P / sum(P);

        choice_prob[t] = P;
    }
}
"""


def prepare_lab_et_mcpt_stan_data(X, p_weight, refer, A, y):
    """
    X        : (T, 2, 4)
    p_weight : (T, 2, 4)
    refer    : (T, 4)
    A        : (T, 4)
    y        : (T, 2) one-hot
    """
    T = X.shape[0]
    y_idx = np.argmax(y, axis=1).astype(int) + 1  # Stan categorical is 1-based

    stan_data = {
        "T": int(T),
        "y": y_idx.tolist(),
        "X": X.tolist(),
        "p_weight": p_weight.tolist(),
        "refer": refer.tolist(),
        "A": A.tolist(),
    }
    return stan_data


def build_lab_et_mcpt_stan_model(stan_file):
    stan_path = Path(stan_file)
    stan_path.write_text(STAN_CODE, encoding="utf-8")
    return CmdStanModel(stan_file=str(stan_path))


def fit_lab_et_mcpt_for_seed(seed, lab_raw, model):
    # ------------------------------------------------------------
    # Split
    # ------------------------------------------------------------
    lab_train_df, lab_test_df, test_ids = split_by_individual(
        lab_raw,
        ID_COL,
        test_frac=TEST_FRAC,
        seed=seed,
    )

    # ------------------------------------------------------------
    # Attention from lab training data
    # ------------------------------------------------------------
    lab_train_att = compute_attention_from_counts(
        lab_train_df,
        id_col=ID_COL,
    )

    # ------------------------------------------------------------
    # Prepare + scale
    # ------------------------------------------------------------
    lab_train_prep = prepare_mcpt_dataframe(
        lab_train_df,
        id_col=ID_COL,
    )

    lab_scaler = fit_mcpt_scaler(lab_train_prep)

    lab_train_norm = apply_mcpt_scaler(
        lab_train_prep,
        lab_scaler,
    )

    # ------------------------------------------------------------
    # Build arrays
    # ------------------------------------------------------------
    lab_train_et = build_mcpt_arrays(
        lab_train_norm,
        id_col=ID_COL,
        task_attention=lab_train_att,
    )

    # ------------------------------------------------------------
    # Basic checks
    # ------------------------------------------------------------
    print("X has NaN:", np.isnan(lab_train_et.X).any())
    print("p_weight has NaN:", np.isnan(lab_train_et.p_weight).any())
    print("refer has NaN:", np.isnan(lab_train_et.refer[:, 0, :]).any())
    print("A has NaN:", np.isnan(lab_train_et.A).any())
    print("y has NaN:", np.isnan(lab_train_et.y).any())

    # ------------------------------------------------------------
    # Stan data
    # ------------------------------------------------------------
    stan_data = prepare_lab_et_mcpt_stan_data(
        X=lab_train_et.X,
        p_weight=lab_train_et.p_weight,
        refer=lab_train_et.refer[:, 0, :],
        A=lab_train_et.A,
        y=lab_train_et.y,
    )

    # ------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------
    fit = model.sample(
        data=stan_data,
        chains=CHAINS,
        parallel_chains=PARALLEL_CHAINS,
        iter_warmup=ITER_WARMUP,
        iter_sampling=ITER_SAMPLING,
        seed=STAN_SEED,
        refresh=100,
    )

    # ------------------------------------------------------------
    # Convert to ArviZ
    # ------------------------------------------------------------
    idata_lab_et = az.from_cmdstanpy(
        posterior=fit,
        log_likelihood="log_lik",
        observed_data={"choice": np.array(stan_data["y"])},
        coords={
            "task": np.arange(stan_data["T"]),
            "alternative": ["CV", "EV"],
        },
        dims={
            "log_lik": ["task"],
            "choice_prob": ["task", "alternative"],
        },
    )

    # ------------------------------------------------------------
    # Save posterior
    # ------------------------------------------------------------
    out_file = POSTERIOR_DIR / f"idata_lab_et_stan_{seed}.nc"
    idata_lab_et.to_netcdf(out_file)

    print(
        f"Seed {seed}: "
        f"train individuals={lab_train_df[ID_COL].nunique()}, "
        f"test individuals={lab_test_df[ID_COL].nunique()}, "
        f"train tasks={stan_data['T']}"
    )

    print(f"Saved: {out_file}")


def main():
    lab_raw = pd.read_csv(LAB_CSV)

    model = build_lab_et_mcpt_stan_model(
        PROJECT_ROOT / "estimation" / "lab_et_mcpt_model.stan"
    )

    for seed in SEEDS:
        fit_lab_et_mcpt_for_seed(
            seed=seed,
            lab_raw=lab_raw,
            model=model,
        )


if __name__ == "__main__":
    main()