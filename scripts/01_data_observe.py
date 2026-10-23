"""
data_observe.py

Descriptive audit and paper-ready data summaries for the Web and Lab datasets.

Current location:
    model_stacking/scripts/01_data_observe.py

Expected input files in model_stacking/data/:
    lab data.csv
    online data.csv

Main outputs:
    data_observe_outputs/
        lab_basic_summary.csv
        online_basic_summary.csv
        dataset_description.csv
        lab_fixation_summary.csv
        lab_attention_summary.csv
        lab_attention_by_scenario.csv
        lab_attention_by_individual.csv
        lab_attention_correlation.csv
        lab_zero_fixation_summary.csv
        table_dataset_description.tex
        table_lab_attention_summary.tex
        fig_fixation_counts.png
        fig_total_fixation_counts.png
        fig_attention_shares.png
        fig_centered_attention.png
        fig_attention_by_scenario.png

Notes
-----
1. Attention is defined task-wise as:
       A_k = N_k / sum_j N_j
   for k in {RC, OC1, OC2, DR}.

2. For tasks with zero total fixation count, attention is set to
   (0.25, 0.25, 0.25, 0.25), consistent with the modelling code.

3. The value 0.25 is treated as the equal-attention benchmark:
       1 / 4 = 0.25
   It is not assumed to be the empirical mean of each attention variable.

4. The script is descriptive only. It does not fit any MCPT model or
   create participant-level train/test splits.

5. Data-quality checks raise errors rather than silently selecting
   inconsistent alternative rows or converting missing counts to zero.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ID_COL = "ID"
SCENARIO_COL = "Scenario"
ALT_COL = "alt"
CHOSEN_COL = "chosen"

FIX_COLS = ["N_RC", "N_OC1", "N_OC2", "N_DR"]
ATT_COLS = ["a_RC", "a_OC1", "a_OC2", "a_DR"]
CENTERED_ATT_COLS = ["gA_RC", "gA_OC1", "gA_OC2", "gA_DR"]

EQUAL_ATTENTION = 0.25


def ensure_numeric(df: pd.DataFrame, cols):
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def validate_choice_tasks(df: pd.DataFrame, dataset_name: str) -> None:
    """Validate long-format choice tasks without modifying the input."""
    required = [ID_COL, SCENARIO_COL, ALT_COL, CHOSEN_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name}: missing columns: {missing}")
    if df[required].isna().any().any():
        raise ValueError(f"{dataset_name}: missing task identifiers or choice fields")
    keys = [ID_COL, SCENARIO_COL]
    if df.duplicated(keys + [ALT_COL]).any():
        raise ValueError(f"{dataset_name}: duplicate participant-task-alternative rows")
    counts = df.groupby(keys, dropna=False)[ALT_COL].nunique()
    if counts.nunique() != 1:
        raise ValueError(f"{dataset_name}: inconsistent number of alternatives per task")
    chosen = pd.to_numeric(df[CHOSEN_COL], errors="coerce")
    if chosen.isna().any() or not chosen.isin([0, 1]).all():
        raise ValueError(f"{dataset_name}: chosen must be binary and nonmissing")
    chosen_per_task = chosen.groupby([df[ID_COL], df[SCENARIO_COL]]).sum()
    if not (chosen_per_task == 1).all():
        raise ValueError(f"{dataset_name}: expected exactly one chosen alternative per task")


def task_level_rows(df: pd.DataFrame, id_col: str = ID_COL) -> pd.DataFrame:
    """Return one row per participant-task after checking repeated fixation counts."""
    needed = [id_col, SCENARIO_COL] + FIX_COLS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    keys = [id_col, SCENARIO_COL]
    # Missing fixation values are not automatically interpreted as zero.
    if df[FIX_COLS].isna().any().any():
        raise ValueError("Missing fixation counts: check raw data before proceeding")
    numeric_counts = df[FIX_COLS].apply(pd.to_numeric, errors="coerce")
    if numeric_counts.isna().any().any():
        raise ValueError("Non-numeric fixation counts detected")
    if (numeric_counts < 0).any().any():
        raise ValueError("Negative fixation counts detected")
    check = pd.concat([df[keys], numeric_counts], axis=1)
    if (check.groupby(keys, dropna=False)[FIX_COLS].nunique(dropna=False) > 1).any().any():
        raise ValueError("Fixation counts differ between alternatives of the same task")
    sort_cols = keys + ([ALT_COL] if ALT_COL in df.columns else [])
    return df.sort_values(sort_cols).drop_duplicates(subset=keys).copy()


def compute_attention_from_counts(df_task: pd.DataFrame) -> pd.DataFrame:
    """
    Compute task-level attention shares from raw fixation counts.
    """
    missing = [c for c in FIX_COLS if c not in df_task.columns]
    if missing:
        raise ValueError(f"Missing fixation-count columns: {missing}")

    out = df_task.copy()
    out = ensure_numeric(out, FIX_COLS)
    if out[FIX_COLS].isna().any().any():
        raise ValueError("Missing/non-numeric fixation counts cannot be treated as zero")

    out["N_total"] = out[FIX_COLS].sum(axis=1)

    denom = out["N_total"].to_numpy(dtype=float)
    safe_denom = np.where(denom > 0, denom, 1.0)

    out["a_RC"] = out["N_RC"].to_numpy(dtype=float) / safe_denom
    out["a_OC1"] = out["N_OC1"].to_numpy(dtype=float) / safe_denom
    out["a_OC2"] = out["N_OC2"].to_numpy(dtype=float) / safe_denom
    out["a_DR"] = out["N_DR"].to_numpy(dtype=float) / safe_denom

    zero_mask = out["N_total"] <= 0
    out.loc[zero_mask, ATT_COLS] = EQUAL_ATTENTION

    out["attention_sum"] = out[ATT_COLS].sum(axis=1)

    out["gA_RC"] = out["a_RC"] - EQUAL_ATTENTION
    out["gA_OC1"] = out["a_OC1"] - EQUAL_ATTENTION
    out["gA_OC2"] = out["a_OC2"] - EQUAL_ATTENTION
    out["gA_DR"] = out["a_DR"] - EQUAL_ATTENTION

    return out


def describe_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    rows = []
    for col in cols:
        if col not in df.columns:
            continue

        s = pd.to_numeric(df[col], errors="coerce")
        rows.append(
            {
                "variable": col,
                "n": int(s.notna().sum()),
                "missing": int(s.isna().sum()),
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "q25": float(s.quantile(0.25)),
                "median": float(s.median()),
                "q75": float(s.quantile(0.75)),
                "max": float(s.max()),
            }
        )
    return pd.DataFrame(rows)


def basic_dataset_summary(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    One-row dataset summary suitable for a paper's data-description table.
    """
    n_rows = len(df)

    n_individuals = df[ID_COL].nunique() if ID_COL in df.columns else np.nan

    if ID_COL in df.columns and SCENARIO_COL in df.columns:
        n_tasks = df[[ID_COL, SCENARIO_COL]].drop_duplicates().shape[0]
    else:
        n_tasks = np.nan

    n_scenarios = df[SCENARIO_COL].nunique() if SCENARIO_COL in df.columns else np.nan
    n_alternatives = df[ALT_COL].nunique() if ALT_COL in df.columns else np.nan

    tasks_per_person = (
        n_tasks / n_individuals
        if pd.notna(n_tasks) and pd.notna(n_individuals) and n_individuals > 0
        else np.nan
    )

    if CHOSEN_COL in df.columns:
        chosen_numeric = pd.to_numeric(df[CHOSEN_COL], errors="coerce")
        chosen_rate = float(chosen_numeric.mean())
    else:
        chosen_rate = np.nan

    return pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "rows": n_rows,
                "individuals": n_individuals,
                "choice_tasks": n_tasks,
                "tasks_per_individual": tasks_per_person,
                "unique_scenarios": n_scenarios,
                "alternatives_per_task": n_alternatives,
                "mean_chosen_indicator": chosen_rate,
            }
        ]
    )


def choice_share_summary(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Optional descriptive table of observed choice shares by alternative.
    """
    if ALT_COL not in df.columns or CHOSEN_COL not in df.columns:
        return pd.DataFrame()

    d = df[[ALT_COL, CHOSEN_COL]].copy()
    d[CHOSEN_COL] = pd.to_numeric(d[CHOSEN_COL], errors="coerce")

    out = (
        d.groupby(ALT_COL)[CHOSEN_COL]
         .agg(["sum", "count", "mean"])
         .reset_index()
         .rename(
             columns={
                 "sum": "chosen_count",
                 "count": "rows",
                 "mean": "choice_share",
             }
         )
    )
    out.insert(0, "dataset", dataset_name)
    return out


def save_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str):
    """
    Save a compact LaTeX table for manuscript use.
    """
    latex = df.to_latex(
        index=False,
        float_format=lambda x: f"{x:.3f}",
        caption=caption,
        label=label,
        escape=True,
    )
    path.write_text(latex, encoding="utf-8")


def plot_fixation_counts(task_att: pd.DataFrame, outdir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()

    for ax, col in zip(axes, FIX_COLS):
        x = pd.to_numeric(task_att[col], errors="coerce").dropna()
        ax.hist(x, bins=25)
        ax.set_title(col)
        ax.set_xlabel("Fixation count")
        ax.set_ylabel("Number of tasks")

    fig.suptitle("Lab task-level fixation-count distributions")
    fig.tight_layout()
    fig.savefig(outdir / "fig_fixation_counts.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_total_fixation_counts(task_att: pd.DataFrame, outdir: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(task_att["N_total"].dropna(), bins=25)
    ax.set_title("Total fixation count per lab choice task")
    ax.set_xlabel("Total fixation count")
    ax.set_ylabel("Number of tasks")
    fig.tight_layout()
    fig.savefig(outdir / "fig_total_fixation_counts.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_attention_shares(task_att: pd.DataFrame, outdir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()

    for ax, col in zip(axes, ATT_COLS):
        x = pd.to_numeric(task_att[col], errors="coerce").dropna()
        ax.hist(x, bins=20)
        ax.axvline(EQUAL_ATTENTION, linestyle="--")
        ax.set_title(col)
        ax.set_xlabel("Attention share")
        ax.set_ylabel("Number of tasks")
        ax.set_xlim(0, 1)

    fig.suptitle("Lab task-level attention shares")
    fig.tight_layout()
    fig.savefig(outdir / "fig_attention_shares.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_centered_attention(task_att: pd.DataFrame, outdir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()

    for ax, col in zip(axes, CENTERED_ATT_COLS):
        x = pd.to_numeric(task_att[col], errors="coerce").dropna()
        ax.hist(x, bins=20)
        ax.axvline(0.0, linestyle="--")
        ax.set_title(col)
        ax.set_xlabel("A - 0.25")
        ax.set_ylabel("Number of tasks")

    fig.suptitle("Lab attention centered at the equal-attention benchmark")
    fig.tight_layout()
    fig.savefig(outdir / "fig_centered_attention.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_attention_by_scenario(task_att: pd.DataFrame, outdir: Path):
    if SCENARIO_COL not in task_att.columns:
        return

    scen = (
        task_att.groupby(SCENARIO_COL)[ATT_COLS]
                .mean()
                .sort_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    for col in ATT_COLS:
        ax.plot(scen.index.astype(str), scen[col], marker="o", label=col)

    ax.axhline(EQUAL_ATTENTION, linestyle="--")
    ax.set_title("Mean lab attention share by scenario")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Mean attention share")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "fig_attention_by_scenario.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    script_dir = Path(__file__).resolve().parents[1] / "data"
    outdir = script_dir / "data_observe_outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    lab_path = script_dir / "lab data.csv"
    online_path = script_dir / "online data.csv"

    if not lab_path.exists():
        raise FileNotFoundError(
            f"Cannot find {lab_path}. Put 'lab data.csv' in the data/ folder."
        )
    if not online_path.exists():
        raise FileNotFoundError(
            f"Cannot find {online_path}. Put 'online data.csv' in the data/ folder."
        )

    lab = pd.read_csv(lab_path)
    online = pd.read_csv(online_path)

    validate_choice_tasks(lab, "Lab")
    validate_choice_tasks(online, "Online")

    # ----------------------------
    # 1. Basic dataset descriptions
    # ----------------------------
    lab_basic = basic_dataset_summary(lab, "Lab")
    online_basic = basic_dataset_summary(online, "Online")
    dataset_description = pd.concat([lab_basic, online_basic], ignore_index=True)

    lab_basic.to_csv(outdir / "lab_basic_summary.csv", index=False)
    online_basic.to_csv(outdir / "online_basic_summary.csv", index=False)
    dataset_description.to_csv(outdir / "dataset_description.csv", index=False)

    save_latex_table(
        dataset_description,
        outdir / "table_dataset_description.tex",
        caption="Descriptive summary of the lab and online datasets.",
        label="tab:dataset_description",
    )

    # Observed choice shares (alternative-level summary)
    choice_shares = pd.concat(
        [
            choice_share_summary(lab, "Lab"),
            choice_share_summary(online, "Online"),
        ],
        ignore_index=True,
    )
    if not choice_shares.empty:
        choice_shares.to_csv(outdir / "choice_share_summary.csv", index=False)

    # ----------------------------
    # 2. Lab fixation and attention audit
    # ----------------------------
    lab_task = task_level_rows(lab)
    lab_task_att = compute_attention_from_counts(lab_task)

    fixation_summary = describe_numeric(
        lab_task_att,
        FIX_COLS + ["N_total"],
    )
    fixation_summary.to_csv(outdir / "lab_fixation_summary.csv", index=False)

    attention_summary = describe_numeric(
        lab_task_att,
        ATT_COLS + CENTERED_ATT_COLS,
    )
    attention_summary.to_csv(outdir / "lab_attention_summary.csv", index=False)

    save_latex_table(
        attention_summary[
            attention_summary["variable"].isin(ATT_COLS)
        ],
        outdir / "table_lab_attention_summary.tex",
        caption=(
            "Task-level attention-share distributions in the lab data. "
            "The equal-attention benchmark is 0.25 for each of the four attributes."
        ),
        label="tab:lab_attention_summary",
    )

    # Verify attention normalization
    max_attention_sum_error = float(
        np.max(np.abs(lab_task_att["attention_sum"] - 1.0))
    )

    if not np.isclose(max_attention_sum_error, 0.0, atol=1e-10):
        raise ValueError("Attention shares do not sum to one")

    zero_total_n = int((lab_task_att["N_total"] <= 0).sum())
    zero_total_share = zero_total_n / max(len(lab_task_att), 1)

    zero_summary = pd.DataFrame(
        [
            {
                "lab_choice_tasks": len(lab_task_att),
                "zero_total_fixation_tasks": zero_total_n,
                "zero_total_fixation_share": zero_total_share,
                "max_abs_attention_sum_minus_1": max_attention_sum_error,
                "equal_attention_benchmark": EQUAL_ATTENTION,
            }
        ]
    )
    zero_summary.to_csv(outdir / "lab_zero_fixation_summary.csv", index=False)

    # Scenario-level attention
    scenario_summary = (
        lab_task_att.groupby(SCENARIO_COL)
                    .agg(
                        n_lab_tasks=(ID_COL, "size"),
                        **{
                            f"{col}_mean": (col, "mean")
                            for col in ATT_COLS
                        },
                        **{
                            f"{col}_std": (col, "std")
                            for col in ATT_COLS
                        },
                        **{
                            f"{col}_median": (col, "median")
                            for col in ATT_COLS
                        },
                    )
                    .reset_index()
    )

    scenario_summary.to_csv(
        outdir / "lab_attention_by_scenario.csv",
        index=False,
    )

    # Individual-level mean attention
    individual_summary = (
        lab_task_att.groupby(ID_COL)[ATT_COLS]
                    .mean()
                    .reset_index()
    )
    individual_summary.to_csv(
        outdir / "lab_attention_by_individual.csv",
        index=False,
    )

    # Correlation matrix
    corr = lab_task_att[ATT_COLS].corr()
    corr.to_csv(outdir / "lab_attention_correlation.csv")

    # ----------------------------
    # 3. Plots
    # ----------------------------
    plot_fixation_counts(lab_task_att, outdir)
    plot_total_fixation_counts(lab_task_att, outdir)
    plot_attention_shares(lab_task_att, outdir)
    plot_centered_attention(lab_task_att, outdir)
    plot_attention_by_scenario(lab_task_att, outdir)

    # ----------------------------
    # 4. Console report
    # ----------------------------
    print("\n==================================================")
    print("DATASET DESCRIPTION")
    print("==================================================")
    print(dataset_description.to_string(index=False))

    print("\n==================================================")
    print("LAB FIXATION SUMMARY")
    print("==================================================")
    print(fixation_summary.to_string(index=False))

    print("\n==================================================")
    print("LAB ATTENTION SUMMARY")
    print("==================================================")
    print(attention_summary.to_string(index=False))

    print("\n==================================================")
    print("LAB ATTENTION VALIDATION")
    print("==================================================")
    print(f"Choice tasks: {len(lab_task_att)}")
    print(
        f"Zero-total-fixation tasks: {zero_total_n} "
        f"({100 * zero_total_share:.2f}%)"
    )
    print(
        "Maximum absolute deviation of attention shares from sum=1: "
        f"{max_attention_sum_error:.3e}"
    )
    print(
        "Equal-attention benchmark used for centering: "
        f"{EQUAL_ATTENTION:.2f}"
    )

    print("\nMean task-level lab attention shares:")
    for col in ATT_COLS:
        print(f"  {col}: {lab_task_att[col].mean():.4f}")

    print("\nMean centered attention (A - 0.25):")
    for col in CENTERED_ATT_COLS:
        print(f"  {col}: {lab_task_att[col].mean():+.4f}")

    print("\nOutputs saved to:")
    print(outdir)


if __name__ == "__main__":
    main()
