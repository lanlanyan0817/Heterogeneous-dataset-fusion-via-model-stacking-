import numpy as np
import pandas as pd
import scipy as sp
import arviz as az
from dataclasses import dataclass
from typing import Tuple, List, Dict

# A. Data/preprocessing
# ----------------------------
# 0) Attenton computation from eye-tracking counts (for later use in lab-ET model) 
# ----------------------------
def softplus(x):
    # stable softplus
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def compute_attention_from_counts(df: pd.DataFrame, id_col: str = "ID") -> pd.DataFrame:
    """
    Returns a task-level dataframe with columns:
    ID, Scenario, a_RC, a_OC1, a_OC2, a_DR
    """
    needed = ["N_RC", "N_OC1", "N_OC2", "N_DR"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} for eye-tracking attention.")

    task = df[[id_col, "Scenario"] + needed].drop_duplicates().copy() #original df has two rows per task due to alt, so drop_duplicates to get task-level 
    # ensure numeric
    for c in needed:
        task[c] = pd.to_numeric(task[c], errors="coerce").fillna(0.0) #convert to numeric, coerce errors to NaN, then fill NaN with 0.0

    denom = task[needed].sum(axis=1).to_numpy()
    denom = np.where(denom <= 0, 1.0, denom)

    task["a_RC"]  = task["N_RC"].to_numpy()  / denom
    task["a_OC1"] = task["N_OC1"].to_numpy() / denom
    task["a_OC2"] = task["N_OC2"].to_numpy() / denom
    task["a_DR"]  = task["N_DR"].to_numpy()  / denom

    # if all counts were 0 => denom=1 but numerators 0 => all zeros; set uniform
    zero_mask = (task[["a_RC","a_OC1","a_OC2","a_DR"]].sum(axis=1) == 0)
    task.loc[zero_mask, ["a_RC","a_OC1","a_OC2","a_DR"]] = 0.25 #uniform attention if no counts (ID 112225 with scenario R1.4S in lab dataset has this issue)

    return task[[id_col, "Scenario", "a_RC", "a_OC1", "a_OC2", "a_DR"]]

def scenario_level_attention_transfer(lab_train_df: pd.DataFrame, id_col: str = "ID") -> pd.DataFrame:
    """
    Compute scenario-level mean attention from lab-train:
    Scenario -> (a_RC, a_OC1, a_OC2, a_DR)
    """
    task_att = compute_attention_from_counts(lab_train_df, id_col=id_col)
    scen_att = task_att.groupby("Scenario")[["a_RC","a_OC1","a_OC2","a_DR"]].mean().reset_index()
    return scen_att

# ----------------------------
# 1) Fixed individual-level split
# ----------------------------
def split_by_individual(df: pd.DataFrame, id_col: str, test_frac: float = 0.2, seed: int =42): 
    rng = np.random.default_rng(seed)
    ids = np.array(df[id_col].unique())
    rng.shuffle(ids)
    n_test = int(np.round(len(ids) * test_frac))
    test_ids = set(ids[:n_test].tolist())
    train_df = df[~df[id_col].isin(test_ids)].copy()
    test_df = df[df[id_col].isin(test_ids)].copy()
    return train_df, test_df, sorted(list(test_ids))



# ----------------------------
# 2) Data -> arrays
# ----------------------------
@dataclass
class MCPTScaler:
    rc_min: float
    rc_max: float
    oc_min: float
    oc_max: float
    oc1_min: float
    oc1_max: float
    dr_min: float
    dr_max: float

def prepare_mcpt_dataframe(df: pd.DataFrame, id_col: str = "ID") -> pd.DataFrame:
    d = df.copy()
    d = d.sort_values([id_col, "Scenario", "alt"]).reset_index(drop=True)

    # OC_1 and EV replacement (alt==3)
    d["OC_1"] = d["OC"]
    mask_ev = (d["alt"] == 3)
    d.loc[mask_ev, ["OC", "OC_1"]] = d.loc[mask_ev, ["P1_V", "P2_V"]].to_numpy()
    d[["OC", "OC_1"]] = d[["OC", "OC_1"]].astype("float64")

    # log transform DR
    d["DR"] = np.log(d["DR"].astype(float))

    return d

def fit_mcpt_scaler(prepared_df: pd.DataFrame) -> MCPTScaler:
    d = prepared_df.copy()
    return MCPTScaler(
        rc_min=float(d["RC"].min()),
        rc_max=float(d["RC"].max()),
        oc_min=float(d["OC"].min()),
        oc_max=float(d["OC"].max()),
        oc1_min=float(d["OC_1"].min()),
        oc1_max=float(d["OC_1"].max()),
        dr_min=float(d["DR"].min()),
        dr_max=float(d["DR"].max()),
    )

def apply_mcpt_scaler(d: pd.DataFrame, scaler: MCPTScaler) -> pd.DataFrame:
    d = d.copy()

    d["RC"] = -((d["RC"] - scaler.rc_min) / (scaler.rc_max - scaler.rc_min + 1e-12))
    d["OC"] = -((d["OC"] - scaler.oc_min) / (scaler.oc_max - scaler.oc_min + 1e-12))
    d["OC_1"] = -((d["OC_1"] - scaler.oc1_min) / (scaler.oc1_max - scaler.oc1_min + 1e-12))
    d["DR"] = ((d["DR"] - scaler.dr_min) / (scaler.dr_max - scaler.dr_min + 1e-12))

    return d


@dataclass
class MCPTArrays:
    X: np.ndarray
    p_weight: np.ndarray
    refer: np.ndarray
    y: np.ndarray
    task_index: pd.DataFrame  # columns: [ID, Scenario]
    A: np.ndarray | None = None  # (N_tasks, 4) attention weights if available


def build_mcpt_arrays(df: pd.DataFrame, id_col: str = "ID",
                      task_attention: pd.DataFrame | None = None,
                      scenario_attention: pd.DataFrame | None = None) -> MCPTArrays:
    """
    Build MCPT arrays from a dataframe that has ALREADY been:
    1) prepared via prepare_mcpt_dataframe()
    2) scaled via apply_mcpt_scaler()

    So this function no longer does replacement / log transform / normalization.
    """
    d = df.copy()
    d = d.sort_values([id_col, "Scenario", "alt"]).reset_index(drop=True)

    attr_cols = ["RC", "OC", "OC_1", "DR"]
    N_attr = len(attr_cols)

    tasks = d[[id_col, "Scenario"]].drop_duplicates().reset_index(drop=True)
    N_tasks = len(tasks)

    # alt is always {1,3} => 2 alts per task
    alts_per_task = d.groupby([id_col, "Scenario"])["alt"].nunique().values
    if not np.all(alts_per_task == 2):
        raise ValueError("Expected exactly 2 alternatives per task (alt in {1,3}).")

    X = np.zeros((N_tasks, 2, N_attr), dtype="float64")
    y = np.zeros((N_tasks, 2), dtype="float64")
    p_weight = np.ones((N_tasks, 2, N_attr), dtype="float64")

    task_rows = []
    for t, row in tasks.iterrows():
        pid = row[id_col]
        scen = row["Scenario"]
        sub = d[(d[id_col] == pid) & (d["Scenario"] == scen)].sort_values("alt")
        alts = sub["alt"].to_numpy()  # should be [1,3]

        X[t, :, :] = sub[attr_cols].to_numpy()
        y[t, :] = sub["chosen"].to_numpy()

        # EV alt (alt==3) has P1/P2 for OC/OC_1 dims (index 1,2)
        ev_pos = int(np.where(alts == 3)[0][0])
        p_weight[t, ev_pos, 1] = float(sub.iloc[ev_pos]["P1"])
        p_weight[t, ev_pos, 2] = float(sub.iloc[ev_pos]["P2"])

        task_rows.append((pid, scen))

    # Reference levels (ICEV reference as in your original code)
    refer = np.zeros((N_tasks, 1, N_attr), dtype="float64")
    for t in range(N_tasks):
        pid, scen = task_rows[t]
        sub = d[(d[id_col] == pid) & (d["Scenario"] == scen)].sort_values("alt")
        alts = sub["alt"].to_numpy()

        cv_pos = int(np.where(alts == 1)[0][0])
        ev_pos = int(np.where(alts == 3)[0][0])

        refer[t, 0, 0] = X[t, cv_pos, 0]
        refer[t, 0, 3] = X[t, cv_pos, 3]

        P1 = float(sub.iloc[ev_pos]["P1"])
        P2 = float(sub.iloc[ev_pos]["P2"])
        refer[t, 0, 1] = -np.sum(X[t, ev_pos, 1:3] * np.array([P1, P2]))
        refer[t, 0, 2] = -np.sum(X[t, ev_pos, 1:3] * np.array([P1, P2]))

    task_index = pd.DataFrame(task_rows, columns=[id_col, "Scenario"])

    A = None
    if task_attention is not None:
        key = tasks.copy()
        key = key.merge(task_attention, on=[id_col, "Scenario"], how="left")
        A = key[["a_RC", "a_OC1", "a_OC2", "a_DR"]].to_numpy(dtype="float64")
    elif scenario_attention is not None:
        key = tasks.copy()
        key = key.merge(scenario_attention, on=["Scenario"], how="left")
        A = key[["a_RC", "a_OC1", "a_OC2", "a_DR"]].to_numpy(dtype="float64")

    if A is not None:
        mask = np.isnan(A).any(axis=1)
        A[mask, :] = 0.25

    return MCPTArrays(X=X, p_weight=p_weight, refer=refer, y=y, task_index=task_index, A=A)

# MCPT model core
# ----------------------------
# 3) MCPT likelihood + per-task logp
# ----------------------------
def mcpt_task_logp(param, X, p_weight, refer, y_choice, use_ev: bool = False) -> np.ndarray:
    """
    Returns vector of per-task log predictive density: log p(y_t | x_t, param)
    """
    asc = param[0]
    afa = param[1]
    lamuda = param[2]
    ra = param[3]
    beta = np.array([param[4], param[5], param[5], param[6]], dtype="float64")

    datasetn = (X - refer) * beta
    neg_mask = datasetn < 0
    datasetn[neg_mask] = -lamuda * np.power((-datasetn[neg_mask]), afa)
    datasetn[~neg_mask] = np.power(datasetn[~neg_mask], afa)

    if use_ev:
        p_weightn = np.exp(-np.power(-np.log(p_weight + 1e-12), ra))
    else:
        p_weightn = p_weight.copy()

        # EV is the second alternative in our array ordering (alt sorted [1,3] => EV index=1)
        ev_idx = 1

        p1 = p_weight[:, ev_idx, 1].astype("float64")
        # safety (should already be in (0,1))
        p1 = np.nan_to_num(p1, nan=0.5)
        p1 = np.clip(p1, 1e-12, 1.0 - 1e-12)

        p1t = np.exp(-np.power(-np.log(p1), ra))

        p_weightn[:, ev_idx, 1] = p1t
        p_weightn[:, ev_idx, 2] = 1.0 - p1t
        # everything else stays as-is (CV row stays at 1s)

    u = np.zeros((X.shape[0], 2), dtype="float64")
    # alt order is [1,3] after sort, but our arrays preserve that sort order
    u[:, 0] = np.sum(datasetn[:, 0, [0, 1, 3]] * p_weightn[:, 0, [0, 1, 3]], axis=1)
    u[:, 1] = np.sum(datasetn[:, 1, :] * p_weightn[:, 1, :], axis=1) + asc

    # stable softmax
    u = u - np.max(u, axis=1, keepdims=True)
    V = np.exp(u)
    P = V / np.sum(V, axis=1, keepdims=True)
    P = np.clip(P, 1e-12, 1.0)

    # per-task logp = sum_j y_{tj} log P_{tj} ; y is one-hot across alts
    return np.sum(y_choice * np.log(P), axis=1)





def mcpt_task_logp_et(param, X, p_weight, refer, y_choice, A, use_ev: bool = False) -> np.ndarray:
    """
    param includes: [asc, alpha, lambda, r, beta_RC0, beta_OC0, beta_DR0, gamma_RC, gamma_OC, gamma_DR]
    A: (N_tasks, 4) attention weights [a_RC, a_OC1, a_OC2, a_DR]
    """
    asc = param[0]
    afa = param[1]
    lamuda = param[2]
    ra = param[3]

    beta_rc0 = param[4]
    beta_oc0 = param[5]
    beta_dr0 = param[6]

    gamma_rc = param[7]
    gamma_oc = param[8]
    gamma_dr = param[9]

    # g(a) = a - 1/4
    gA = A - 0.25

    beta_rc = softplus(beta_rc0 + gamma_rc * gA[:, 0])
    beta_oc  = softplus(beta_oc0 + gamma_oc * gA[:, 1])
    beta_oc1 = softplus(beta_oc0 + gamma_oc * gA[:, 2])
    beta_dr  = softplus(beta_dr0 + gamma_dr * gA[:, 3])

    # build task-level beta tensor (N_tasks, N_attr)
    beta_task = np.stack([beta_rc, beta_oc, beta_oc1, beta_dr], axis=1)  # (N,4)

    datasetn = (X - refer) * beta_task[:, None, :]  # broadcast to (N,2,4)

    neg_mask = datasetn < 0
    datasetn[neg_mask] = -lamuda * np.power((-datasetn[neg_mask]), afa)
    datasetn[~neg_mask] = np.power(datasetn[~neg_mask], afa)

    # probability weighting: only EV alt (index 1)
    if use_ev:
        p_weightn = np.exp(-np.power(-np.log(p_weight + 1e-12), ra))
    else:
        p_weightn = p_weight.copy()
        ev_idx = 1
        p1 = p_weight[:, ev_idx, 1].astype("float64")
        p1 = np.nan_to_num(p1, nan=0.5)
        p1 = np.clip(p1, 1e-12, 1.0 - 1e-12)
        p1t = np.exp(-np.power(-np.log(p1), ra))
        p_weightn[:, ev_idx, 1] = p1t
        p_weightn[:, ev_idx, 2] = 1.0 - p1t

    u = np.zeros((X.shape[0], 2), dtype="float64")
    u[:, 0] = np.sum(datasetn[:, 0, [0, 1, 3]] * p_weightn[:, 0, [0, 1, 3]], axis=1)
    u[:, 1] = np.sum(datasetn[:, 1, :] * p_weightn[:, 1, :], axis=1) + asc

    u = u - np.max(u, axis=1, keepdims=True)
    V = np.exp(u)
    P = V / np.sum(V, axis=1, keepdims=True)
    P = np.clip(P, 1e-12, 1.0)

    return np.sum(y_choice * np.log(P), axis=1)



def mcpt_task_logp_draws(theta_draws: np.ndarray,
                         X: np.ndarray,
                         p_weight: np.ndarray,
                         refer: np.ndarray,
                         y_choice: np.ndarray,
                         use_ev: bool = False) -> np.ndarray:
    """
    Compute per-task log-likelihood for many parameter draws.

    Parameters
    ----------
    theta_draws : array, shape (n_draws, n_params)
    X, p_weight, refer, y_choice : same as existing mcpt_task_logp inputs

    Returns
    -------
    loglik_draws : array, shape (n_draws, n_tasks)
        each row is per-task log predictive density under one parameter draw
    """
    n_draws = theta_draws.shape[0]
    n_tasks = X.shape[0]

    out = np.zeros((n_draws, n_tasks), dtype="float64")
    for s in range(n_draws):
        out[s, :] = mcpt_task_logp(
            theta_draws[s],
            X,
            p_weight,
            refer,
            y_choice,
            use_ev=use_ev
        )
    return out

def make_arviz_idata_from_loglik(loglik_draws: np.ndarray, var_name: str = "y") -> az.InferenceData:
    """
    Convert log-likelihood matrix (draw, obs) into InferenceData for ArviZ.
    """
    loglik = loglik_draws[np.newaxis, :, :]  # -> (chain=1, draw, obs)

    # dummy posterior just to satisfy ArviZ structure requirements
    n_draws = loglik_draws.shape[0]
    posterior_dummy = {"theta_dummy": np.zeros((1, n_draws), dtype="float64")}

    idata = az.from_dict(
        posterior=posterior_dummy,
        log_likelihood={var_name: loglik}
    )
    return idata









def mcpt_task_logp_et_draws(theta_draws: np.ndarray,
                            X: np.ndarray,
                            p_weight: np.ndarray,
                            refer: np.ndarray,
                            y_choice: np.ndarray,
                            A: np.ndarray,
                            use_ev: bool = False) -> np.ndarray:
    """
    Compute per-task log-likelihood for many ET parameter draws.

    Returns
    -------
    loglik_draws : array, shape (n_draws, n_tasks)
    """
    n_draws = theta_draws.shape[0]
    n_tasks = X.shape[0]

    out = np.zeros((n_draws, n_tasks), dtype="float64")
    for s in range(n_draws):
        out[s, :] = mcpt_task_logp_et(
            theta_draws[s],
            X,
            p_weight,
            refer,
            y_choice,
            A,
            use_ev=use_ev
        )
    return out

def extract_lab_et_theta_draws_from_idata(idata, model_prefix: str = "lab_et_mcpt::") -> np.ndarray:
    """
    Extract constrained posterior parameter draws for Lab_ET_MCPT from PyMC/ArviZ InferenceData.

    Expected constrained variables:
        asc, alpha, lambda_, r, beta_rc0, beta_oc0, beta_dr0, gamma_rc, gamma_oc, gamma_dr

    Returns
    -------
    theta_draws : np.ndarray, shape (n_draws_total, 10)
        columns are:
        [asc, alpha, lambda_, r, beta_rc0, beta_oc0, beta_dr0, gamma_rc, gamma_oc, gamma_dr]
    """
    posterior = idata.posterior

    # stack chain and draw into one sample dimension
    posterior_s = posterior.stack(sample=("chain", "draw"))

    names = [
        "asc",
        "alpha",
        "lambda_",
        "r",
        "beta_rc0",
        "beta_oc0",
        "beta_dr0",
        "gamma_rc",
        "gamma_oc",
        "gamma_dr",
    ]

    arrays = []
    for n in names:
        full_name = f"{model_prefix}{n}"
        if full_name not in posterior_s.data_vars:
            raise KeyError(f"Posterior variable '{full_name}' not found in idata.posterior")
        arrays.append(np.asarray(posterior_s[full_name].values, dtype="float64"))

    theta_draws = np.column_stack(arrays)
    return theta_draws


def extract_web_theta_draws_from_idata(idata, model_prefix: str = "web_mcpt::") -> np.ndarray:
    """
    Extract constrained posterior parameter draws for Web_MCPT from PyMC/ArviZ InferenceData.

    Returns
    -------
    theta_draws : np.ndarray, shape (n_draws_total, 7)
        columns are:
        [asc, alpha, lambda_, r, beta_rc, beta_oc, beta_dr]
    """
    posterior = idata.posterior.stack(sample=("chain", "draw"))

    names = [
        "asc",
        "alpha",
        "lambda_",
        "r",
        "beta_rc",
        "beta_oc",
        "beta_dr",
    ]

    arrays = []
    for n in names:
        full_name = f"{model_prefix}{n}"
        if full_name not in posterior.data_vars:
            raise KeyError(f"Posterior variable '{full_name}' not found in idata.posterior")
        arrays.append(np.asarray(posterior[full_name].values, dtype="float64"))

    theta_draws = np.column_stack(arrays)
    return theta_draws