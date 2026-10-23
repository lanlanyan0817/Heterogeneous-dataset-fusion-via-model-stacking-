
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
