"""Reproduce the paper's MLE baselines for Target Domains 1 and 2.

Run from the repository root (or any directory):
    python scripts/08_baseline_models.py

The MCPT specifications, preprocessing, optimization settings, and per-seed
calculations are preserved from the two original domain implementations.
Results are written to results/baselines/.
"""
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
import scipy as sp

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT = ROOT / "results" / "baselines"
SEEDS = [12, 22, 32, 42, 52, 62, 72, 82, 92, 102]


def softplus(x):
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

    task = df[[id_col, "Scenario"] + needed].drop_duplicates().copy()
    for c in needed:
        task[c] = pd.to_numeric(task[c], errors="coerce").fillna(0.0)

    denom = task[needed].sum(axis=1).to_numpy()
    denom = np.where(denom <= 0, 1.0, denom)

    task["a_RC"]  = task["N_RC"].to_numpy()  / denom
    task["a_OC1"] = task["N_OC1"].to_numpy() / denom
    task["a_OC2"] = task["N_OC2"].to_numpy() / denom
    task["a_DR"]  = task["N_DR"].to_numpy()  / denom

    zero_mask = (task[["a_RC","a_OC1","a_OC2","a_DR"]].sum(axis=1) == 0)
    task.loc[zero_mask, ["a_RC","a_OC1","a_OC2","a_DR"]] = 0.25

    return task[[id_col, "Scenario", "a_RC", "a_OC1", "a_OC2", "a_DR"]]

def scenario_level_attention_transfer(lab_train_df: pd.DataFrame, id_col: str = "ID") -> pd.DataFrame:
    task_att = compute_attention_from_counts(lab_train_df, id_col=id_col)
    scen_att = task_att.groupby("Scenario")[["a_RC","a_OC1","a_OC2","a_DR"]].mean().reset_index()
    return scen_att

# ----------------------------
# 1) Fixed individual-level split
# ----------------------------
def split_by_individual(df: pd.DataFrame, id_col: str, test_frac: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    ids = np.array(df[id_col].unique())
    rng.shuffle(ids)
    n_test = int(np.round(len(ids) * test_frac))
    test_ids = set(ids[:n_test].tolist())
    train_df = df[~df[id_col].isin(test_ids)].copy()
    test_df = df[df[id_col].isin(test_ids)].copy()
    return train_df, test_df, sorted(list(test_ids))

def make_kfolds(ids: np.ndarray, n_folds: int = 5, seed: int = 123) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    ids = ids.copy()
    rng.shuffle(ids)
    folds = np.array_split(ids, n_folds)
    return folds

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

@dataclass
class MCPTArrays:
    X: np.ndarray
    p_weight: np.ndarray
    refer: np.ndarray
    y: np.ndarray
    task_index: pd.DataFrame
    A: Optional[np.ndarray] = None

def prepare_mcpt_dataframe(df: pd.DataFrame, id_col: str = "ID") -> pd.DataFrame:
    d = df.copy()
    d = d.sort_values([id_col, "Scenario", "alt"]).reset_index(drop=True)

    d["OC_1"] = d["OC"]
    mask_ev = (d["alt"] == 3)
    d.loc[mask_ev, ["OC", "OC_1"]] = d.loc[mask_ev, ["P1_V", "P2_V"]].to_numpy()
    d[["OC", "OC_1"]] = d[["OC", "OC_1"]].astype("float64")

    dr_raw = d["DR"].astype(float)
    if (dr_raw <= 0).any():
        bad_n = int((dr_raw <= 0).sum())
        raise ValueError(f"DR contains {bad_n} non-positive values; log transform invalid.")
    d["DR"] = np.log(dr_raw)

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

def prepare_and_scale(train_df: pd.DataFrame,
                      eval_df: Optional[pd.DataFrame] = None,
                      id_col: str = "ID"):
    """
    Fit scaler on train_df only, then apply to train_df and optionally eval_df.
    Returns:
        train_scaled, eval_scaled(or None), scaler
    """
    train_prepared = prepare_mcpt_dataframe(train_df, id_col=id_col)
    scaler = fit_mcpt_scaler(train_prepared)
    train_scaled = apply_mcpt_scaler(train_prepared, scaler)

    eval_scaled = None
    if eval_df is not None:
        eval_prepared = prepare_mcpt_dataframe(eval_df, id_col=id_col)
        eval_scaled = apply_mcpt_scaler(eval_prepared, scaler)

    return train_scaled, eval_scaled, scaler

def build_mcpt_arrays(df: pd.DataFrame, id_col: str = "ID",
                      task_attention: Optional[pd.DataFrame] = None,
                      scenario_attention: Optional[pd.DataFrame] = None) -> MCPTArrays:
    """
    df must already be prepared and scaled outside this function.
    """
    d = df.copy()
    d = d.sort_values([id_col, "Scenario", "alt"]).reset_index(drop=True)

    for (pid, scen), g in d.groupby([id_col, "Scenario"]):
        alt_set = set(g["alt"].tolist())
        if alt_set != {1, 3}:
            raise ValueError(f"Task ({pid}, {scen}) has alt set {alt_set}, expected {{1, 3}}.")

    attr_cols = ["RC", "OC", "OC_1", "DR"]
    N_attr = len(attr_cols)

    tasks = d[[id_col, "Scenario"]].drop_duplicates().reset_index(drop=True)
    N_tasks = len(tasks)

    X = np.zeros((N_tasks, 2, N_attr), dtype="float64")
    y = np.zeros((N_tasks, 2), dtype="float64")
    p_weight = np.ones((N_tasks, 2, N_attr), dtype="float64")

    task_rows = []
    for t, row in tasks.iterrows():
        pid = row[id_col]
        scen = row["Scenario"]
        sub = d[(d[id_col] == pid) & (d["Scenario"] == scen)].sort_values("alt")
        alts = sub["alt"].to_numpy()

        X[t, :, :] = sub[attr_cols].to_numpy()
        y[t, :] = sub["chosen"].to_numpy()

        ev_pos = int(np.where(alts == 3)[0][0])
        p_weight[t, ev_pos, 1] = float(sub.iloc[ev_pos]["P1"])
        p_weight[t, ev_pos, 2] = float(sub.iloc[ev_pos]["P2"])

        task_rows.append((pid, scen))

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
        A = key[["a_RC","a_OC1","a_OC2","a_DR"]].to_numpy(dtype="float64")
    elif scenario_attention is not None:
        key = tasks.copy()
        key = key.merge(scenario_attention, on=["Scenario"], how="left")
        A = key[["a_RC","a_OC1","a_OC2","a_DR"]].to_numpy(dtype="float64")

    if A is not None:
        mask = np.isnan(A).any(axis=1)
        A[mask, :] = 0.25

    return MCPTArrays(X=X, p_weight=p_weight, refer=refer, y=y, task_index=task_index, A=A)

# ----------------------------
# 3) MCPT likelihood + per-task logp
# ----------------------------
def mcpt_task_logp(param, X, p_weight, refer, y_choice, use_ev: bool = False) -> np.ndarray:
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

def mcpt_task_prob(param, X, p_weight, refer, use_ev: bool = False) -> np.ndarray:
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

    return P

def mcpt_neg_ll(param, X, p_weight, refer, y_choice, use_ev: bool = False) -> float:
    return -float(np.sum(mcpt_task_logp(param, X, p_weight, refer, y_choice, use_ev=use_ev)))

def mcpt_task_logp_et(param, X, p_weight, refer, y_choice, A, use_ev: bool = False) -> np.ndarray:
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

    gA = A - 0.25

    beta_rc = softplus(beta_rc0 + gamma_rc * gA[:, 0])
    beta_oc  = softplus(beta_oc0 + gamma_oc * gA[:, 1])
    beta_oc1 = softplus(beta_oc0 + gamma_oc * gA[:, 2])
    beta_dr  = softplus(beta_dr0 + gamma_dr * gA[:, 3])

    beta_task = np.stack([beta_rc, beta_oc, beta_oc1, beta_dr], axis=1)

    datasetn = (X - refer) * beta_task[:, None, :]

    neg_mask = datasetn < 0
    datasetn[neg_mask] = -lamuda * np.power((-datasetn[neg_mask]), afa)
    datasetn[~neg_mask] = np.power(datasetn[~neg_mask], afa)

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

def mcpt_task_prob_et(param, X, p_weight, refer, A, use_ev: bool = False) -> np.ndarray:
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

    gA = A - 0.25

    beta_rc = softplus(beta_rc0 + gamma_rc * gA[:, 0])
    beta_oc  = softplus(beta_oc0 + gamma_oc * gA[:, 1])
    beta_oc1 = softplus(beta_oc0 + gamma_oc * gA[:, 2])
    beta_dr  = softplus(beta_dr0 + gamma_dr * gA[:, 3])

    beta_task = np.stack([beta_rc, beta_oc, beta_oc1, beta_dr], axis=1)

    datasetn = (X - refer) * beta_task[:, None, :]

    neg_mask = datasetn < 0
    datasetn[neg_mask] = -lamuda * np.power((-datasetn[neg_mask]), afa)
    datasetn[~neg_mask] = np.power(datasetn[~neg_mask], afa)

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

    return P

def mcpt_neg_ll_et(param, X, p_weight, refer, y_choice, A, use_ev: bool = False):
    return -float(np.sum(mcpt_task_logp_et(param, X, p_weight, refer, y_choice, A, use_ev=use_ev)))

def fit_mcpt_mle_et(arr: MCPTArrays, initial: np.ndarray, bounds, use_ev: bool = False):
    if arr.A is None:
        raise ValueError("ET model requires arr.A (attention weights).")

    res = sp.optimize.minimize(
        fun=mcpt_neg_ll_et,
        x0=initial,
        args=(arr.X, arr.p_weight, arr.refer, arr.y, arr.A, use_ev),
        method="L-BFGS-B",
        bounds=bounds,
        tol=1e-2,
        options={"disp": True}
    )
    return res

def fit_mcpt_mle(arr: MCPTArrays, initial: np.ndarray, bounds: Tuple[Tuple[float,float], ...], use_ev: bool = False):
    res = sp.optimize.minimize(
        fun=mcpt_neg_ll,
        x0=initial,
        args=(arr.X, arr.p_weight, arr.refer, arr.y, use_ev),
        method="L-BFGS-B",
        bounds=bounds,
        tol=1e-2,
        options={"disp": True}
    )
    return res

def evaluate_from_logp_and_prob(task_logp: np.ndarray, P: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    total = float(np.sum(task_logp))
    n_tasks = int(task_logp.shape[0])
    avg = float(total / max(n_tasks, 1))
    var = float(np.var(task_logp))

    pred_choice = np.argmax(P, axis=1)
    true_choice = np.argmax(y, axis=1)
    acc = float(np.mean(pred_choice == true_choice))

    return {
        "total_log_score": total,
        "avg_log_score": avg,
        "var_log_score": var,
        "accuracy": acc,
        "n_tasks": n_tasks
    }

def run_domain1_split(datasplit_seed: int, path: str) -> Dict[str, float]:
    """
    Target Domain 1: Street test data only.

    Paper baselines:
      - Street-MCPT (baseline)
      - Pooled-MCPT (street + lab, baseline)

    Numerical settings are kept consistent with
    evaluate_paper_baselines_domain2.py:
      L-BFGS-B, same initial values, bounds, tolerance,
      preprocessing/scaling, and ET-aware pooled formulation.
    """
    lab = pd.read_csv(path + "lab data.csv")
    web = pd.read_csv(path + "online data.csv")

    lab_train_df, lab_test_df, _ = split_by_individual(
        lab, "ID", test_frac=0.2, seed=datasplit_seed
    )
    web_train_df, web_test_df, _ = split_by_individual(
        web, "ID", test_frac=0.2, seed=datasplit_seed
    )

    # Preserve the historical dataset namespaces used in the reproduced baseline code.
    lab_train_df = lab_train_df.copy()
    lab_test_df = lab_test_df.copy()
    web_train_df = web_train_df.copy()
    web_test_df = web_test_df.copy()

    lab_train_df["ID"] = "lab_" + lab_train_df["ID"].astype(str)
    lab_test_df["ID"] = "lab_" + lab_test_df["ID"].astype(str)
    web_train_df["ID"] = "web_" + web_train_df["ID"].astype(str)
    web_test_df["ID"] = "web_" + web_test_df["ID"].astype(str)

    print("\n==============================")
    print(f"Data split seed = {datasplit_seed}")
    print(
        f"Lab: train individuals={lab_train_df['ID'].nunique()}, "
        f"test individuals={lab_test_df['ID'].nunique()}"
    )
    print(
        f"Web: train individuals={web_train_df['ID'].nunique()}, "
        f"test individuals={web_test_df['ID'].nunique()}"
    )

    bounds = (
        (-np.inf, np.inf),  # ASC
        (0, 1),             # alpha
        (1, np.inf),        # lambda
        (0, np.inf),        # r
        (0, np.inf),        # beta_RC
        (0, np.inf),        # beta_OC
        (0, np.inf),        # beta_DR
    )
    initial = np.array(
        [2.4, 0.88, 3.9, 4.3, 1.15, 4.9, 4.18],
        dtype="float64"
    )

    initial_et = np.array(
        [2.4, 0.88, 3.9, 4.3, 1.15, 4.9, 4.18, 0.0, 0.0, 0.0],
        dtype="float64"
    )
    bounds_et = (
        (-np.inf, np.inf),  # ASC
        (0, 1),             # alpha
        (1, np.inf),        # lambda
        (0, np.inf),        # r
        (0, np.inf),        # beta_RC0
        (0, np.inf),        # beta_OC0
        (0, np.inf),        # beta_DR0
        (-np.inf, np.inf),  # gamma_RC
        (-np.inf, np.inf),  # gamma_OC
        (-np.inf, np.inf),  # gamma_DR
    )

    # ========================================================
    # Street-MCPT (baseline)
    # Train: Street train
    # Test:  Street test only
    # ========================================================

    web_train_scaled, web_test_scaled, _ = prepare_and_scale(
        web_train_df, web_test_df, id_col="ID"
    )

    web_train = build_mcpt_arrays(
        web_train_scaled, id_col="ID"
    )
    web_test = build_mcpt_arrays(
        web_test_scaled, id_col="ID"
    )

    res_street = fit_mcpt_mle(
        web_train,
        initial,
        bounds,
        use_ev=False
    )
    theta_street = res_street.x

    logp_street = mcpt_task_logp(
        theta_street,
        web_test.X,
        web_test.p_weight,
        web_test.refer,
        web_test.y,
        use_ev=False
    )
    prob_street = mcpt_task_prob(
        theta_street,
        web_test.X,
        web_test.p_weight,
        web_test.refer,
        use_ev=False
    )

    eval_street = evaluate_from_logp_and_prob(
        logp_street,
        prob_street,
        web_test.y
    )

    # ========================================================
    # Pooled-MCPT (street + lab, baseline)
    #
    # Same ET-aware pooled baseline as Domain 2 historical
    # Model 7b:
    #   - fit pooled scaler on Street-train + Lab-train
    #   - observed task attention for Lab
    #   - scenario-level transferred attention for Street
    #   - fit one pooled ET-aware MCPT model
    #
    # Target Domain 1 evaluation is Street test only.
    # ========================================================

    lab_train_att = compute_attention_from_counts(
        lab_train_df, id_col="ID"
    )
    scen_att = scenario_level_attention_transfer(
        lab_train_df, id_col="ID"
    )

    pooled_train_df = pd.concat(
        [web_train_df, lab_train_df],
        axis=0,
        ignore_index=True
    )

    # The second dataframe is not used to fit the scaler; it is supplied
    # only to preserve the same prepare_and_scale helper behavior.
    _, _, pooled_scaler = prepare_and_scale(
        pooled_train_df,
        web_test_df,
        id_col="ID"
    )

    pooled_train_lab_df = apply_mcpt_scaler(
        prepare_mcpt_dataframe(lab_train_df, id_col="ID"),
        pooled_scaler
    )
    pooled_train_web_df = apply_mcpt_scaler(
        prepare_mcpt_dataframe(web_train_df, id_col="ID"),
        pooled_scaler
    )
    web_test_pooled_df = apply_mcpt_scaler(
        prepare_mcpt_dataframe(web_test_df, id_col="ID"),
        pooled_scaler
    )

    pooled_train_lab = build_mcpt_arrays(
        pooled_train_lab_df,
        id_col="ID",
        task_attention=lab_train_att
    )
    pooled_train_web = build_mcpt_arrays(
        pooled_train_web_df,
        id_col="ID",
        scenario_attention=scen_att
    )
    web_test_pooled = build_mcpt_arrays(
        web_test_pooled_df,
        id_col="ID",
        scenario_attention=scen_att
    )

    pooled_train_ET = MCPTArrays(
        X=np.concatenate(
            [pooled_train_lab.X, pooled_train_web.X],
            axis=0
        ),
        p_weight=np.concatenate(
            [pooled_train_lab.p_weight, pooled_train_web.p_weight],
            axis=0
        ),
        refer=np.concatenate(
            [pooled_train_lab.refer, pooled_train_web.refer],
            axis=0
        ),
        y=np.concatenate(
            [pooled_train_lab.y, pooled_train_web.y],
            axis=0
        ),
        task_index=pd.concat(
            [pooled_train_lab.task_index, pooled_train_web.task_index],
            ignore_index=True
        ),
        A=np.concatenate(
            [pooled_train_lab.A, pooled_train_web.A],
            axis=0
        )
    )

    res_pooled = fit_mcpt_mle_et(
        pooled_train_ET,
        initial_et,
        bounds_et,
        use_ev=False
    )
    theta_pooled = res_pooled.x

    logp_pooled = mcpt_task_logp_et(
        theta_pooled,
        web_test_pooled.X,
        web_test_pooled.p_weight,
        web_test_pooled.refer,
        web_test_pooled.y,
        web_test_pooled.A,
        use_ev=False
    )
    prob_pooled = mcpt_task_prob_et(
        theta_pooled,
        web_test_pooled.X,
        web_test_pooled.p_weight,
        web_test_pooled.refer,
        web_test_pooled.A,
        use_ev=False
    )

    eval_pooled = evaluate_from_logp_and_prob(
        logp_pooled,
        prob_pooled,
        web_test_pooled.y
    )

    print(
        f"[Street-MCPT (baseline)] street-test: "
        f"total={eval_street['total_log_score']:.2f}, "
        f"avg={eval_street['avg_log_score']:.4f}, "
        f"var={eval_street['var_log_score']:.4f}, "
        f"acc={eval_street['accuracy']:.4f}, "
        f"T={eval_street['n_tasks']}"
    )
    print(
        f"[Pooled-MCPT (street + lab, baseline)] street-test: "
        f"total={eval_pooled['total_log_score']:.2f}, "
        f"avg={eval_pooled['avg_log_score']:.4f}, "
        f"var={eval_pooled['var_log_score']:.4f}, "
        f"acc={eval_pooled['accuracy']:.4f}, "
        f"T={eval_pooled['n_tasks']}"
    )

    return {
        "seed": datasplit_seed,

        "Street_MCPT_total": eval_street["total_log_score"],
        "Street_MCPT_avg": eval_street["avg_log_score"],
        "Street_MCPT_var": eval_street["var_log_score"],
        "Street_MCPT_acc": eval_street["accuracy"],

        "Pooled_MCPT_total": eval_pooled["total_log_score"],
        "Pooled_MCPT_avg": eval_pooled["avg_log_score"],
        "Pooled_MCPT_var": eval_pooled["var_log_score"],
        "Pooled_MCPT_acc": eval_pooled["accuracy"],
    }

def run_domain2_split(datasplit_seed: int, path: str) -> Dict[str, float]:
    lab = pd.read_csv(path + "lab data.csv")
    web = pd.read_csv(path + "online data.csv")

    lab_train_df, lab_test_df, _ = split_by_individual(lab, "ID", test_frac=0.2, seed=datasplit_seed)
    web_train_df, web_test_df, _ = split_by_individual(web, "ID", test_frac=0.2, seed=datasplit_seed)

    # avoid pooled ID collisions
    lab_train_df = lab_train_df.copy()
    lab_test_df = lab_test_df.copy()
    web_train_df = web_train_df.copy()
    web_test_df = web_test_df.copy()

    lab_train_df["ID"] = "lab_" + lab_train_df["ID"].astype(str)
    lab_test_df["ID"] = "lab_" + lab_test_df["ID"].astype(str)
    web_train_df["ID"] = "web_" + web_train_df["ID"].astype(str)
    web_test_df["ID"] = "web_" + web_test_df["ID"].astype(str)

    print("\n==============================")
    print(f"Data split seed = {datasplit_seed}")
    print(f"Lab: train individuals={lab_train_df['ID'].nunique()}, test individuals={lab_test_df['ID'].nunique()}")
    print(f"Web: train individuals={web_train_df['ID'].nunique()}, test individuals={web_test_df['ID'].nunique()}")

    bounds = (
        (-np.inf, np.inf),
        (0, 1),
        (1, np.inf),
        (0, np.inf),
        (0, np.inf),
        (0, np.inf),
        (0, np.inf),
    )
    initial = np.array([2.4, 0.88, 3.9, 4.3, 1.15, 4.9, 4.18], dtype="float64")

    initial_et = np.array([2.4, 0.88, 3.9, 4.3, 1.15, 4.9, 4.18, 0.0, 0.0, 0.0], dtype="float64")
    bounds_et = (
        (-np.inf, np.inf),
        (0, 1),
        (1, np.inf),
        (0, np.inf),
        (0, np.inf),
        (0, np.inf),
        (0, np.inf),
        (-np.inf, np.inf),
        (-np.inf, np.inf),
        (-np.inf, np.inf),
    )

    # attention objects
    lab_train_att = compute_attention_from_counts(lab_train_df, id_col="ID")
    lab_test_att  = compute_attention_from_counts(lab_test_df,  id_col="ID")
    scen_att = scenario_level_attention_transfer(lab_train_df, id_col="ID")

    # ------------------------------------------------------------
    # scaling objects
    # ------------------------------------------------------------
    web_train_scaled, web_test_scaled, web_scaler = prepare_and_scale(
        web_train_df, web_test_df, id_col="ID"
    )

    lab_train_scaled, lab_test_scaled, lab_scaler = prepare_and_scale(
        lab_train_df, lab_test_df, id_col="ID"
    )

    web_train_scaled_for_lab = apply_mcpt_scaler(
        prepare_mcpt_dataframe(web_train_df, id_col="ID"), lab_scaler
    )
    web_test_scaled_for_lab = apply_mcpt_scaler(
        prepare_mcpt_dataframe(web_test_df, id_col="ID"), lab_scaler
    )

    lab_train_scaled_for_web = apply_mcpt_scaler(
        prepare_mcpt_dataframe(lab_train_df, id_col="ID"), web_scaler
    )
    lab_test_scaled_for_web = apply_mcpt_scaler(
        prepare_mcpt_dataframe(lab_test_df, id_col="ID"), web_scaler
    )

    # ------------------------------------------------------------
    # Build arrays
    # ------------------------------------------------------------
    # Model 4 web model
    web_train = build_mcpt_arrays(web_train_scaled, id_col="ID")
    web_test  = build_mcpt_arrays(web_test_scaled,  id_col="ID")
    lab_train_for_web = build_mcpt_arrays(lab_train_scaled_for_web, id_col="ID")
    lab_test_for_web  = build_mcpt_arrays(lab_test_scaled_for_web,  id_col="ID")

    # Model 5 lab ET model
    lab_train_ET = build_mcpt_arrays(lab_train_scaled, id_col="ID", task_attention=lab_train_att)
    lab_test_ET  = build_mcpt_arrays(lab_test_scaled,  id_col="ID", task_attention=lab_test_att)

    web_train_for_labET = build_mcpt_arrays(web_train_scaled_for_lab, id_col="ID", scenario_attention=scen_att)
    web_test_for_labET  = build_mcpt_arrays(web_test_scaled_for_lab,  id_col="ID", scenario_attention=scen_att)

    # ------------------------------------------------------------
    # Step 1. Fit base models
    # ------------------------------------------------------------
    res_web = fit_mcpt_mle(web_train, initial, bounds, use_ev=False)
    theta_web = res_web.x

    res_lab_et = fit_mcpt_mle_et(lab_train_ET, initial_et, bounds_et, use_ev=False)
    theta_lab_et = res_lab_et.x



    # ------------------------------------------------------------
    # Step 3. Score pooled test
    # ------------------------------------------------------------
    score_logp_m5_on_labtest = mcpt_task_logp_et(
        theta_lab_et,
        lab_test_ET.X, lab_test_ET.p_weight, lab_test_ET.refer, lab_test_ET.y, lab_test_ET.A,
        use_ev=False
    )
    score_prob_m5_on_labtest = mcpt_task_prob_et(
        theta_lab_et,
        lab_test_ET.X, lab_test_ET.p_weight, lab_test_ET.refer, lab_test_ET.A,
        use_ev=False
    )

    score_logp_m4_on_labtest = mcpt_task_logp(
        theta_web,
        lab_test_for_web.X, lab_test_for_web.p_weight, lab_test_for_web.refer, lab_test_for_web.y,
        use_ev=False
    )
    score_prob_m4_on_labtest = mcpt_task_prob(
        theta_web,
        lab_test_for_web.X, lab_test_for_web.p_weight, lab_test_for_web.refer,
        use_ev=False
    )

    score_logp_m5_on_webtest = mcpt_task_logp_et(
        theta_lab_et,
        web_test_for_labET.X, web_test_for_labET.p_weight, web_test_for_labET.refer, web_test_for_labET.y,
        web_test_for_labET.A,
        use_ev=False
    )
    score_prob_m5_on_webtest = mcpt_task_prob_et(
        theta_lab_et,
        web_test_for_labET.X, web_test_for_labET.p_weight, web_test_for_labET.refer, web_test_for_labET.A,
        use_ev=False
    )

    score_logp_m4_on_webtest = mcpt_task_logp(
        theta_web,
        web_test.X, web_test.p_weight, web_test.refer, web_test.y,
        use_ev=False
    )
    score_prob_m4_on_webtest = mcpt_task_prob(
        theta_web,
        web_test.X, web_test.p_weight, web_test.refer,
        use_ev=False
    )

    pooled_test_logp_m5 = np.concatenate([score_logp_m5_on_labtest, score_logp_m5_on_webtest], axis=0)
    pooled_test_logp_m4 = np.concatenate([score_logp_m4_on_labtest, score_logp_m4_on_webtest], axis=0)

    pooled_test_prob_m5 = np.concatenate([score_prob_m5_on_labtest, score_prob_m5_on_webtest], axis=0)
    pooled_test_prob_m4 = np.concatenate([score_prob_m4_on_labtest, score_prob_m4_on_webtest], axis=0)

    pooled_test_y = np.concatenate([lab_test_ET.y, web_test.y], axis=0)

    # Model 6
    #pooled_test_logp_m6 = stacking_logp([pooled_test_logp_m4, pooled_test_logp_m5], w_m6)
    #pooled_test_prob_m6 = stacking_prob([pooled_test_prob_m4, pooled_test_prob_m5], w_m6)
    #eval_M6 = evaluate_from_logp_and_prob(pooled_test_logp_m6, pooled_test_prob_m6, pooled_test_y)

    # Model 4
    eval_M4 = evaluate_from_logp_and_prob(pooled_test_logp_m4, pooled_test_prob_m4, pooled_test_y)

    # Model 5
    eval_M5 = evaluate_from_logp_and_prob(pooled_test_logp_m5, pooled_test_prob_m5, pooled_test_y)

    # ------------------------------------------------------------
    # Pooled-MCPT (street + lab, baseline) [historical Model 7b]
    # pooled scaler fit on pooled training data
    # ------------------------------------------------------------
    pooled_train_df = pd.concat([web_train_df, lab_train_df], axis=0, ignore_index=True)
    pooled_test_df = pd.concat([web_test_df, lab_test_df], axis=0, ignore_index=True)
    _, _, pooled_scaler = prepare_and_scale(pooled_train_df, pooled_test_df, id_col="ID")

    # Model 7b pooled ET-aware
    # scaler fit on pooled_train only
    # ------------------------------------------------------------
    pooled_train_lab_df = apply_mcpt_scaler(
        prepare_mcpt_dataframe(lab_train_df, id_col="ID"), pooled_scaler
    )
    pooled_test_lab_df = apply_mcpt_scaler(
        prepare_mcpt_dataframe(lab_test_df, id_col="ID"), pooled_scaler
    )
    pooled_train_web_df = apply_mcpt_scaler(
        prepare_mcpt_dataframe(web_train_df, id_col="ID"), pooled_scaler
    )
    pooled_test_web_df = apply_mcpt_scaler(
        prepare_mcpt_dataframe(web_test_df, id_col="ID"), pooled_scaler
    )

    pooled_train_lab = build_mcpt_arrays(
        pooled_train_lab_df, id_col="ID", task_attention=lab_train_att
    )
    pooled_test_lab = build_mcpt_arrays(
        pooled_test_lab_df, id_col="ID", task_attention=lab_test_att
    )

    pooled_train_web = build_mcpt_arrays(
        pooled_train_web_df, id_col="ID", scenario_attention=scen_att
    )
    pooled_test_web = build_mcpt_arrays(
        pooled_test_web_df, id_col="ID", scenario_attention=scen_att
    )

    pooled_train_ET = MCPTArrays(
        X=np.concatenate([pooled_train_lab.X, pooled_train_web.X], axis=0),
        p_weight=np.concatenate([pooled_train_lab.p_weight, pooled_train_web.p_weight], axis=0),
        refer=np.concatenate([pooled_train_lab.refer, pooled_train_web.refer], axis=0),
        y=np.concatenate([pooled_train_lab.y, pooled_train_web.y], axis=0),
        task_index=pd.concat([pooled_train_lab.task_index, pooled_train_web.task_index], ignore_index=True),
        A=np.concatenate([pooled_train_lab.A, pooled_train_web.A], axis=0)
    )

    pooled_test_ET = MCPTArrays(
        X=np.concatenate([pooled_test_lab.X, pooled_test_web.X], axis=0),
        p_weight=np.concatenate([pooled_test_lab.p_weight, pooled_test_web.p_weight], axis=0),
        refer=np.concatenate([pooled_test_lab.refer, pooled_test_web.refer], axis=0),
        y=np.concatenate([pooled_test_lab.y, pooled_test_web.y], axis=0),
        task_index=pd.concat([pooled_test_lab.task_index, pooled_test_web.task_index], ignore_index=True),
        A=np.concatenate([pooled_test_lab.A, pooled_test_web.A], axis=0)
    )

    res_pooled_ET = fit_mcpt_mle_et(
        pooled_train_ET,
        initial_et,
        bounds_et,
        use_ev=False
    )
    theta_pooled_ET = res_pooled_ET.x

    pooled_test_logp_m7b = mcpt_task_logp_et(
        theta_pooled_ET,
        pooled_test_ET.X, pooled_test_ET.p_weight, pooled_test_ET.refer, pooled_test_ET.y, pooled_test_ET.A,
        use_ev=False
    )
    pooled_test_prob_m7b = mcpt_task_prob_et(
        theta_pooled_ET,
        pooled_test_ET.X, pooled_test_ET.p_weight, pooled_test_ET.refer, pooled_test_ET.A,
        use_ev=False
    )
    eval_M7b = evaluate_from_logp_and_prob(pooled_test_logp_m7b, pooled_test_prob_m7b, pooled_test_ET.y)

    print(
        f"[Street-MCPT (baseline)] pooled-test: total={eval_M4['total_log_score']:.2f}, "
        f"avg={eval_M4['avg_log_score']:.4f}, var={eval_M4['var_log_score']:.4f}, "
        f"acc={eval_M4['accuracy']:.4f}, T={eval_M4['n_tasks']}"
    )
    print(
        f"[Lab-MCPT (baseline)] pooled-test: total={eval_M5['total_log_score']:.2f}, "
        f"avg={eval_M5['avg_log_score']:.4f}, var={eval_M5['var_log_score']:.4f}, "
        f"acc={eval_M5['accuracy']:.4f}, T={eval_M5['n_tasks']}"
    )
    #print(
        #f"[Model 6: Experiment-specific stacked model] pooled-test: total={eval_M6['total_log_score']:.2f}, "
        #f"avg={eval_M6['avg_log_score']:.4f}, var={eval_M6['var_log_score']:.4f}, "
        #f"acc={eval_M6['accuracy']:.4f}, T={eval_M6['n_tasks']}"
    #)
    print(
        f"[Pooled-MCPT (street + lab, baseline)] pooled-test: total={eval_M7b['total_log_score']:.2f}, "
        f"avg={eval_M7b['avg_log_score']:.4f}, var={eval_M7b['var_log_score']:.4f}, "
        f"acc={eval_M7b['accuracy']:.4f}, T={eval_M7b['n_tasks']}"
    )

    return {
        "seed": datasplit_seed,

        "Street_MCPT_total": eval_M4["total_log_score"],
        "Street_MCPT_avg": eval_M4["avg_log_score"],
        "Street_MCPT_var": eval_M4["var_log_score"],
        "Street_MCPT_acc": eval_M4["accuracy"],

        "Lab_MCPT_total": eval_M5["total_log_score"],
        "Lab_MCPT_avg": eval_M5["avg_log_score"],
        "Lab_MCPT_var": eval_M5["var_log_score"],
        "Lab_MCPT_acc": eval_M5["accuracy"],

        #"M6_total": eval_M6["total_log_score"],
        #"M6_avg": eval_M6["avg_log_score"],
        #"M6_var": eval_M6["var_log_score"],
        #"M6_acc": eval_M6["accuracy"],
        #"M6_w_model4_web": float(w_m6[0]),
        #"M6_w_model5_labET": float(w_m6[1]),

        #"M6_lab_total": eval_M6_lab["total_log_score"],
        #"M6_lab_avg": eval_M6_lab["avg_log_score"],
        #"M6_lab_var": eval_M6_lab["var_log_score"],
        #"M6_lab_acc": eval_M6_lab["accuracy"],

        #"M6_web_total": eval_M6_web["total_log_score"],
        #"M6_web_avg": eval_M6_web["avg_log_score"],
        #"M6_web_var": eval_M6_web["var_log_score"],
        #"M6_web_acc": eval_M6_web["accuracy"],

        "Pooled_MCPT_total": eval_M7b["total_log_score"],
        "Pooled_MCPT_avg": eval_M7b["avg_log_score"],
        "Pooled_MCPT_var": eval_M7b["var_log_score"],
        "Pooled_MCPT_acc": eval_M7b["accuracy"],
    }


def main():
    for filename in ("lab data.csv", "online data.csv"):
        if not (DATA_DIR / filename).is_file():
            raise FileNotFoundError(f"Missing input data: {DATA_DIR / filename}")
    OUT.mkdir(parents=True, exist_ok=True)
    data_path = str(DATA_DIR) + "/"
    summary = []
    for domain, run_split, models in (
        ("domain1", run_domain1_split, ("Street", "Pooled")),
        ("domain2", run_domain2_split, ("Street", "Lab", "Pooled")),
    ):
        rows = []
        for seed in SEEDS:
            print(f"\n>>> {domain} seed={seed}", flush=True)
            result = run_split(datasplit_seed=seed, path=data_path)
            if int(result["seed"]) != seed:
                raise ValueError(f"Unexpected seed in {domain}: {result['seed']}")
            rows.append(result)
        df = pd.DataFrame(rows)
        df.to_csv(OUT / f"{domain}_per_seed.csv", index=False)
        for model in models:
            avg = df[f"{model}_MCPT_avg"].to_numpy(float)
            var = df[f"{model}_MCPT_var"].to_numpy(float)
            if not (np.isfinite(avg).all() and np.isfinite(var).all()):
                raise ValueError(f"Non-finite scores: {domain} {model}")
            summary.append({
                "domain": domain,
                "model": f"{model}-MCPT (MLE)",
                "test_avg": float(avg.mean()),
                "test_var": float(var.mean()),
                "sd_across_seeds_avg": float(avg.std(ddof=1)),
                "n_seeds": len(avg),
            })
    result = pd.DataFrame(summary)
    result.to_csv(OUT / "baseline_summary.csv", index=False)
    print("\n=== MLE BASELINE SUMMARY ===")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nSaved under: {OUT}")
    print("NOTE: Optimizer convergence status is not captured by the original fitting functions; inspect logs.")

if __name__ == "__main__":
    main()
