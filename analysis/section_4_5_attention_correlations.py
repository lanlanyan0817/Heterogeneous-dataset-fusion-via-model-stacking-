"""
Spearman correlation analysis between individual-level characteristics
and average eye-tracking attention allocation.

Analysis design
---------------
1. Compute task-level attention shares from fixation counts.
2. Average attention across all tasks for each individual.
3. Retain the four original individual-level variables used to construct
   the logit-based stacking covariates:
      - Freq_p_week
      - Dis_p_day
      - AT2
      - experience_months
4. Compute pairwise Spearman rank correlations with:
      - a_RC
      - a_OC1
      - a_OC2
      - a_DR

Place in model_stacking/analysis/.
Run from repository root:
    python analysis/section_4_5_attention_correlations.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcpt_core import compute_attention_from_counts


# ============================================================
# Settings
# ============================================================
BASE_DIR = ROOT
LAB_CSV = ROOT / "data" / "lab data.csv"
OUTPUT_DIR = ROOT / "results" / "analysis" / "section_4_5"

ID_COL = "ID"

COVARIATES = {
    "Freq_p_week": "Driving frequency",
    "Dis_p_day": "Daily driving distance",
    "AT2": "Attitudinal preference",
    "experience_months": "Driving experience",
}

ATTENTION_VARS = {
    "a_RC": r"$A_{RC}$",
    "a_OC1": r"$A_{OC1}$",
    "a_OC2": r"$A_{OC2}$",
    "a_DR": r"$A_{DR}$",
}


# ============================================================
# Helpers
# ============================================================
def significance_stars(p_value: float) -> str:
    """Return conventional significance stars."""
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "The following required columns are missing from the lab dataset: "
            + ", ".join(missing)
        )


def build_individual_level_dataset(lab_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct one row per individual.

    Attention is first computed at the task level and then averaged across
    all tasks completed by each individual. Individual characteristics are
    extracted once per individual and merged with average attention.
    """
    required = [
        ID_COL,
        "Scenario",
        "N_RC",
        "N_OC1",
        "N_OC2",
        "N_DR",
        *COVARIATES.keys(),
    ]
    validate_columns(lab_df, required)

    # Task-level attention shares:
    # ID, Scenario, a_RC, a_OC1, a_OC2, a_DR
    task_attention = compute_attention_from_counts(lab_df, id_col=ID_COL)

    # Average attention across all tasks for each individual
    individual_attention = (
        task_attention
        .groupby(ID_COL, as_index=False)[list(ATTENTION_VARS.keys())]
        .mean()
    )

    # Verify each individual has consistent characteristics across tasks.
    for col in COVARIATES:
        inconsistent = lab_df.groupby(ID_COL)[col].nunique(dropna=False).gt(1)
        if inconsistent.any():
            raise ValueError(f"Inconsistent {col} within participants: {inconsistent[inconsistent].index.tolist()}")

    # Extract one set of individual characteristics per person
    individual_covariates = (
        lab_df[[ID_COL, *COVARIATES.keys()]]
        .drop_duplicates(subset=[ID_COL])
        .copy()
    )

    # Ensure numerical representation
    for col in COVARIATES:
        individual_covariates[col] = pd.to_numeric(
            individual_covariates[col], errors="coerce"
        )

    if individual_covariates[list(COVARIATES)].isna().any().any():
        raise ValueError("Missing or nonnumeric individual covariates")

    analysis_df = individual_covariates.merge(
        individual_attention,
        on=ID_COL,
        how="inner",
        validate="one_to_one",
    )

    return analysis_df


def compute_spearman_tables(
    analysis_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute pairwise Spearman rho, p-values, pairwise sample sizes,
    and a formatted rho-with-stars table.
    """
    rho = pd.DataFrame(
        index=list(COVARIATES.values()),
        columns=list(ATTENTION_VARS.values()),
        dtype=float,
    )
    pvals = rho.copy()
    n_pairs = rho.copy()

    for cov_col, cov_label in COVARIATES.items():
        for att_col, att_label in ATTENTION_VARS.items():
            pair = analysis_df[[cov_col, att_col]].dropna()
            n = len(pair)
            n_pairs.loc[cov_label, att_label] = n

            if n < 3:
                warnings.warn(
                    f"Too few complete observations for {cov_col} and {att_col}."
                )
                rho.loc[cov_label, att_label] = np.nan
                pvals.loc[cov_label, att_label] = np.nan
                continue

            if pair[cov_col].nunique() < 2 or pair[att_col].nunique() < 2:
                warnings.warn(
                    f"Constant variable encountered for {cov_col} and {att_col}."
                )
                rho.loc[cov_label, att_label] = np.nan
                pvals.loc[cov_label, att_label] = np.nan
                continue

            result = spearmanr(
                pair[cov_col].to_numpy(),
                pair[att_col].to_numpy(),
                nan_policy="omit",
            )
            rho.loc[cov_label, att_label] = float(result.statistic)
            pvals.loc[cov_label, att_label] = float(result.pvalue)

    formatted = pd.DataFrame(
        index=rho.index,
        columns=rho.columns,
        dtype=object,
    )

    for row in rho.index:
        for col in rho.columns:
            r = rho.loc[row, col]
            p = pvals.loc[row, col]
            if pd.isna(r):
                formatted.loc[row, col] = ""
            else:
                formatted.loc[row, col] = (
                    f"{r:.3f}{significance_stars(p)}"
                )

    return rho, pvals, n_pairs.astype("Int64"), formatted


def export_latex_table(formatted: pd.DataFrame, output_path: Path) -> None:
    """Export an Overleaf-ready table."""
    latex = formatted.to_latex(
        escape=False,
        caption=(
            "Spearman correlations between individual characteristics "
            "and average attention allocation."
        ),
        label="tab:attention_covariate_correlations",
        column_format="lcccc",
        position="ht",
    )

    note = (
        "\n% Notes: Attention is averaged across all choice tasks for each individual.\n"
        "% * p < 0.05, ** p < 0.01, *** p < 0.001.\n"
    )
    output_path.write_text(latex + note, encoding="utf-8")


# ============================================================
# Main
# ============================================================
def main() -> None:
    if not LAB_CSV.exists():
        raise FileNotFoundError(
            f"Could not find '{LAB_CSV.name}' in {BASE_DIR}."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lab_df = pd.read_csv(LAB_CSV)
    analysis_df = build_individual_level_dataset(lab_df)

    print(f"Lab participants: {analysis_df[ID_COL].nunique()}; expected 40")
    if analysis_df[ID_COL].nunique() != 40:
        raise ValueError("Unexpected number of Lab participants")
    rho, pvals, n_pairs, formatted = compute_spearman_tables(analysis_df)

    # Save analysis-ready individual-level data
    analysis_df.to_csv(
        OUTPUT_DIR / "individual_level_attention_covariates.csv",
        index=False,
    )

    # Save numerical results
    rho.to_csv(OUTPUT_DIR / "spearman_rho.csv")
    pvals.to_csv(OUTPUT_DIR / "spearman_pvalues.csv")
    n_pairs.to_csv(OUTPUT_DIR / "spearman_pairwise_n.csv")
    formatted.to_csv(OUTPUT_DIR / "spearman_formatted.csv")

    # Save Overleaf-ready table
    export_latex_table(
        formatted,
        OUTPUT_DIR / "spearman_correlation_table.tex",
    )

    print("\nIndividual-level sample size:", analysis_df[ID_COL].nunique())
    print("\nSpearman rho:")
    print(rho.round(3))
    print("\nP-values:")
    print(pvals.round(4))
    print("\nFormatted table:")
    print(formatted)
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
