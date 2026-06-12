from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AgeMatchingResult:
    matched: pd.DataFrame
    audit: pd.DataFrame
    summary: dict[str, float | int]


def _standardized_mean_difference(case_age: pd.Series, control_age: pd.Series) -> float:
    pooled_variance = (case_age.var(ddof=1) + control_age.var(ddof=1)) / 2
    if pooled_variance <= 0 or np.isnan(pooled_variance):
        return 0.0
    return float((case_age.mean() - control_age.mean()) / np.sqrt(pooled_variance))


def frequency_match_controls_by_age(
    data: pd.DataFrame,
    *,
    age_column: str = "age_at_index_years",
    cohort_column: str = "COHORT",
    case_value: int = 1,
    control_value: int = 0,
    controls_per_case: float = 1.0,
    age_resolution: float = 1.0,
    random_state: int = 42,
) -> AgeMatchingResult:
    """Frequency-match controls to the case age distribution.

    Matching uses the rounded age visible to the model rather than broad,
    manually selected age bins. All cases are retained. For each age stratum,
    up to ``controls_per_case * case_count`` controls are sampled without
    replacement. The function is intended for training cohorts; evaluation
    cohorts should retain their natural age distribution.
    """
    if controls_per_case <= 0:
        raise ValueError("controls_per_case must be greater than zero")
    if age_resolution <= 0:
        raise ValueError("age_resolution must be greater than zero")

    required = {age_column, cohort_column}
    missing_columns = required - set(data.columns)
    if missing_columns:
        raise KeyError(f"Missing required columns: {sorted(missing_columns)}")
    if data[age_column].isna().any():
        raise ValueError(f"{age_column} contains missing values")

    cases = data[data[cohort_column] == case_value].copy()
    controls = data[data[cohort_column] == control_value].copy()
    other_labels = data[~data[cohort_column].isin([case_value, control_value])]
    if not other_labels.empty:
        raise ValueError(f"Unexpected cohort labels: {sorted(other_labels[cohort_column].unique())}")
    if cases.empty or controls.empty:
        raise ValueError("Both case and control rows are required")

    stratum_column = "_age_match_stratum"
    cases[stratum_column] = np.round(cases[age_column] / age_resolution).astype(int)
    controls[stratum_column] = np.round(controls[age_column] / age_resolution).astype(int)

    rng = np.random.default_rng(random_state)
    selected_control_indices: list[int] = []
    audit_rows = []
    case_counts = cases[stratum_column].value_counts().sort_index()
    control_groups = controls.groupby(stratum_column).groups

    for stratum, case_count in case_counts.items():
        available_indices = np.asarray(list(control_groups.get(stratum, [])), dtype=object)
        requested = int(np.ceil(case_count * controls_per_case))
        selected_count = min(requested, len(available_indices))
        if selected_count:
            selected = rng.choice(available_indices, size=selected_count, replace=False)
            selected_control_indices.extend(selected.tolist())
        audit_rows.append(
            {
                "age": stratum * age_resolution,
                "case_count": int(case_count),
                "available_control_count": int(len(available_indices)),
                "requested_control_count": requested,
                "selected_control_count": selected_count,
                "control_shortfall": requested - selected_count,
            }
        )

    selected_controls = controls.loc[selected_control_indices]
    matched = pd.concat([cases, selected_controls], axis=0)
    matched = matched.sample(frac=1, random_state=random_state).drop(columns=stratum_column)
    audit = pd.DataFrame(audit_rows)

    summary = {
        "original_case_count": len(cases),
        "original_control_count": len(controls),
        "matched_case_count": len(cases),
        "matched_control_count": len(selected_controls),
        "case_mean_age": float(cases[age_column].mean()),
        "original_control_mean_age": float(controls[age_column].mean()),
        "matched_control_mean_age": float(selected_controls[age_column].mean()),
        "smd_before": _standardized_mean_difference(cases[age_column], controls[age_column]),
        "smd_after": _standardized_mean_difference(cases[age_column], selected_controls[age_column]),
        "total_control_shortfall": int(audit["control_shortfall"].sum()),
    }
    return AgeMatchingResult(matched=matched, audit=audit, summary=summary)
