
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
