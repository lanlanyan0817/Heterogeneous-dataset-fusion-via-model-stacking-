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

WEB_CSV = DATA_DIR / "online data.csv"

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
    int<lower=1> T;                     // number of tasks
    array[T] int<lower=1, upper=2> y;   // chosen alternative: 1=CV, 2=EV

    array[T, 2, 4] real X;              // attributes
    array[T, 2, 4] real p_weight;       // objective probabilities / weights
    array[T, 4] real refer;             // reference point
}

parameters {
    // unconstrained parameters
    real eta_asc;
    real eta_alpha;
    real eta_lambda;
    real eta_r;

    real eta_beta_rc;
    real eta_beta_oc;
    real eta_beta_dr;
}

transformed parameters {
    // constrained parameters
    real asc;
    real<lower=0, upper=1> alpha;
    real<lower=1> lambda_;
    real<lower=0> r;

    real<lower=0> beta_rc;
    real<lower=0> beta_oc;
    real<lower=0> beta_dr;

    vector[4] beta;

    asc = eta_asc;
    alpha = inv_logit(eta_alpha);
    lambda_ = 1 + fmin(exp(eta_lambda), 20);
    r = fmin(exp(eta_r), 3);

    beta_rc = fmin(exp(eta_beta_rc), 20);
    beta_oc = fmin(exp(eta_beta_oc), 20);
    beta_dr = fmin(exp(eta_beta_dr), 20);

    // [RC, OC, OC_1, DR], with beta_OC_1 = beta_OC
    beta[1] = beta_rc;
    beta[2] = beta_oc;
    beta[3] = beta_oc;
    beta[4] = beta_dr;
}

model {
    // priors
    eta_asc     ~ normal(0, 1.0);
    eta_alpha   ~ normal(0, 0.75);
    eta_lambda  ~ normal(0, 0.3);
    eta_r       ~ normal(0, 0.2);

    eta_beta_rc ~ normal(0, 0.3);
    eta_beta_oc ~ normal(0, 0.3);
    eta_beta_dr ~ normal(0, 0.3);

    // likelihood
    for (t in 1:T) {
        vector[2] U;
        vector[2] P;

        // utilities
        real u_cv;
        real u_ev;

        // transformed/valued attributes
        real z;
        real v;

        real p1;
        real p1_t;
        real p2_t;

        // ----------------------------
        // CV utility
        // uses attributes [RC, OC, DR] => indices [1,2,4] in Stan
        // ----------------------------
        u_cv = 0;

        // attr 1: RC
        z = (X[t, 1, 1] - refer[t, 1]) * beta[1];
        if (z < 0) {
            v = -lambda_ * pow(-z, alpha);
        } else {
            v = pow(z, alpha);
        }
        u_cv += v * p_weight[t, 1, 1];

        // attr 2: OC
        z = (X[t, 1, 2] - refer[t, 2]) * beta[2];
        if (z < 0) {
            v = -lambda_ * pow(-z, alpha);
        } else {
            v = pow(z, alpha);
        }
        u_cv += v * p_weight[t, 1, 2];

        // attr 4: DR
        z = (X[t, 1, 4] - refer[t, 4]) * beta[4];
        if (z < 0) {
            v = -lambda_ * pow(-z, alpha);
        } else {
            v = pow(z, alpha);
        }
        u_cv += v * p_weight[t, 1, 4];

        // ----------------------------
        // EV utility
        // uses all 4 attributes
        // OC / OC_1 use weighted probabilities
        // ----------------------------
        p1 = fmin(fmax(p_weight[t, 2, 2], 1e-12), 1 - 1e-12);
        p1_t = exp(-pow(-log(p1), r));
        p2_t = 1 - p1_t;

        u_ev = 0;

        // attr 1: RC
        z = (X[t, 2, 1] - refer[t, 1]) * beta[1];
        if (z < 0) {
            v = -lambda_ * pow(-z, alpha);
        } else {
            v = pow(z, alpha);
        }
        u_ev += v * p_weight[t, 2, 1];

        // attr 2: OC
        z = (X[t, 2, 2] - refer[t, 2]) * beta[2];
        if (z < 0) {
            v = -lambda_ * pow(-z, alpha);
        } else {
            v = pow(z, alpha);
        }
        u_ev += v * p1_t;

        // attr 3: OC_1
        z = (X[t, 2, 3] - refer[t, 3]) * beta[3];
        if (z < 0) {
            v = -lambda_ * pow(-z, alpha);
        } else {
            v = pow(z, alpha);
        }
        u_ev += v * p2_t;

        // attr 4: DR
        z = (X[t, 2, 4] - refer[t, 4]) * beta[4];
        if (z < 0) {
            v = -lambda_ * pow(-z, alpha);
        } else {
            v = pow(z, alpha);
        }
        u_ev += v * p_weight[t, 2, 4];

        u_ev += asc;

        U[1] = u_cv;
        U[2] = u_ev;

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

        real z;
        real v;

        real p1;
        real p1_t;
        real p2_t;

        // ----------------------------
        // CV utility
        // ----------------------------
        u_cv = 0;

        z = (X[t, 1, 1] - refer[t, 1]) * beta[1];
        if (z < 0) v = -lambda_ * pow(-z, alpha);
        else       v = pow(z, alpha);
        u_cv += v * p_weight[t, 1, 1];

        z = (X[t, 1, 2] - refer[t, 2]) * beta[2];
        if (z < 0) v = -lambda_ * pow(-z, alpha);
        else       v = pow(z, alpha);
        u_cv += v * p_weight[t, 1, 2];

        z = (X[t, 1, 4] - refer[t, 4]) * beta[4];
        if (z < 0) v = -lambda_ * pow(-z, alpha);
        else       v = pow(z, alpha);
        u_cv += v * p_weight[t, 1, 4];

        // ----------------------------
        // EV utility
        // ----------------------------
        p1 = fmin(fmax(p_weight[t, 2, 2], 1e-12), 1 - 1e-12);
        p1_t = exp(-pow(-log(p1), r));
        p2_t = 1 - p1_t;

        u_ev = 0;

        z = (X[t, 2, 1] - refer[t, 1]) * beta[1];
        if (z < 0) v = -lambda_ * pow(-z, alpha);
        else       v = pow(z, alpha);
        u_ev += v * p_weight[t, 2, 1];

        z = (X[t, 2, 2] - refer[t, 2]) * beta[2];
        if (z < 0) v = -lambda_ * pow(-z, alpha);
        else       v = pow(z, alpha);
        u_ev += v * p1_t;

        z = (X[t, 2, 3] - refer[t, 3]) * beta[3];
        if (z < 0) v = -lambda_ * pow(-z, alpha);
        else       v = pow(z, alpha);
        u_ev += v * p2_t;

        z = (X[t, 2, 4] - refer[t, 4]) * beta[4];
        if (z < 0) v = -lambda_ * pow(-z, alpha);
        else       v = pow(z, alpha);
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


def prepare_web_mcpt_stan_data(X, p_weight, refer, y):
    """
    Convert arrays into Stan data dict.

    Inputs
    ------
    X        : shape (T, 2, 4)
    p_weight : shape (T, 2, 4)
    refer    : shape (T, 4)
    y        : shape (T, 2) one-hot

    Returns
    -------
    data dict for Stan
    """
    T = X.shape[0]

    # Stan categorical is 1-based: 1 for CV, 2 for EV
    y_idx = np.argmax(y, axis=1).astype(int) + 1

    stan_data = {
        "T": int(T),
        "y": y_idx.tolist(),
        "X": X.tolist(),
        "p_weight": p_weight.tolist(),
        "refer": refer.tolist(),
    }
    return stan_data


def build_web_mcpt_stan_model(stan_file):
    """
    Write Stan code to disk and compile the model.
    """
    stan_path = Path(stan_file)
    stan_path.write_text(STAN_CODE, encoding="utf-8")
    return CmdStanModel(stan_file=str(stan_path))



def fit_web_mcpt_for_seed(seed, web_raw, model):
    web_train_df, web_test_df, test_ids = split_by_individual(
        web_raw,
        ID_COL,
        test_frac=TEST_FRAC,
        seed=seed,
    )

    web_train_prep = prepare_mcpt_dataframe(
        web_train_df,
        id_col=ID_COL,
    )

    web_scaler = fit_mcpt_scaler(web_train_prep)

    web_train_norm = apply_mcpt_scaler(
        web_train_prep,
        web_scaler,
    )

    web_train = build_mcpt_arrays(
        web_train_norm,
        id_col=ID_COL,
    )

    stan_data = prepare_web_mcpt_stan_data(
        X=web_train.X,
        p_weight=web_train.p_weight,
        refer=web_train.refer[:, 0, :],
        y=web_train.y,
    )

    fit = model.sample(
        data=stan_data,
        chains=CHAINS,
        parallel_chains=PARALLEL_CHAINS,
        iter_warmup=ITER_WARMUP,
        iter_sampling=ITER_SAMPLING,
        seed=STAN_SEED,
        refresh=100,
    )

    idata_web = az.from_cmdstanpy(
        posterior=fit,
        log_likelihood="log_lik",
        posterior_predictive=None,
        observed_data={"choice": stan_data["y"]},
        coords={
            "task": np.arange(stan_data["T"]),
            "alternative": ["CV", "EV"],
        },
        dims={
            "log_lik": ["task"],
            "choice_prob": ["task", "alternative"],
        },
    )

    out_file = POSTERIOR_DIR / f"idata_web_stan_{seed}.nc"
    idata_web.to_netcdf(out_file)

    print(
        f"Seed {seed}: "
        f"train individuals={web_train_df[ID_COL].nunique()}, "
        f"test individuals={web_test_df[ID_COL].nunique()}, "
        f"train tasks={stan_data['T']}"
    )
    print(f"Saved: {out_file}")

def main():
    web_raw = pd.read_csv(WEB_CSV)

    model = build_web_mcpt_stan_model(
        PROJECT_ROOT / "estimation" / "web_mcpt_model.stan"
    )

    for seed in SEEDS:
        fit_web_mcpt_for_seed(
            seed=seed,
            web_raw=web_raw,
            model=model,
        )


if __name__ == "__main__":
    main()