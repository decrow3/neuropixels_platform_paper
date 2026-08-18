"""Named, validated unit-population masks for Figure 3 comparisons."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


POPULATION_PROFILES = (
    "pipeline_baseline",
    "common_qc",
    "published_like",
    "intersection",
    "pilot_rf_qc_diagnostic",
)

COMMON_QC_RULE = (
    "amplitude_cutoff < 0.1; presence_ratio > 0.8; "
    "ISI violations ratio < 0.5"
)
PUBLISHED_LIKE_RULE = (
    "p_value_rf < 0.01; area_rf < 2500 deg^2; snr > 1; "
    "firing_rate_dg > 0.1 Hz"
)


class PopulationUnavailableError(ValueError):
    """Raised when a named population cannot be computed from available data."""


def _require_columns(
    table: pd.DataFrame,
    columns: Iterable[str],
    *,
    profile: str,
    dataset: str,
) -> None:
    missing = sorted(set(columns).difference(table.columns))
    if missing:
        raise PopulationUnavailableError(
            f"Population profile '{profile}' is unavailable for {dataset}: "
            f"missing columns {missing}"
        )


def common_qc_mask(table: pd.DataFrame, *, dataset: str) -> pd.Series:
    """Apply homologous Allen/Mouse waveform-quality thresholds."""
    isi_column = (
        "isi_violations_ratio"
        if "isi_violations_ratio" in table.columns
        else "isi_violations"
    )
    _require_columns(
        table,
        ("amplitude_cutoff", "presence_ratio", isi_column),
        profile="common_qc",
        dataset=dataset,
    )
    mask = (
        (pd.to_numeric(table["amplitude_cutoff"], errors="coerce") < 0.1)
        & (pd.to_numeric(table["presence_ratio"], errors="coerce") > 0.8)
        & (pd.to_numeric(table[isi_column], errors="coerce") < 0.5)
    )
    if dataset == "mousev2" and "default_qc" in table.columns:
        declared = table["default_qc"].fillna(False).astype(bool)
        if not mask.equals(declared):
            mismatches = int((mask != declared).sum())
            raise ValueError(
                f"MouseV2 default_qc drift: {mismatches} units disagree with "
                f"the declared common rule ({COMMON_QC_RULE})"
            )
    return mask.fillna(False)


def published_like_mask(table: pd.DataFrame, *, dataset: str) -> pd.Series:
    """Reproduce the released Figure 3 base population filters."""
    required = ("p_value_rf", "area_rf", "snr", "firing_rate_dg")
    _require_columns(
        table,
        required,
        profile="published_like",
        dataset=dataset,
    )
    return (
        (pd.to_numeric(table["p_value_rf"], errors="coerce") < 0.01)
        & (pd.to_numeric(table["area_rf"], errors="coerce") < 2500)
        & (pd.to_numeric(table["snr"], errors="coerce") > 1)
        & (pd.to_numeric(table["firing_rate_dg"], errors="coerce") > 0.1)
    ).fillna(False)


def population_mask(
    table: pd.DataFrame,
    *,
    profile: str,
    dataset: str,
) -> pd.Series:
    """Return a named population mask without silently weakening its rule."""
    if profile not in POPULATION_PROFILES:
        raise ValueError(
            f"Unknown population profile '{profile}'; choose from {POPULATION_PROFILES}"
        )
    if dataset not in {"allen", "mousev2"}:
        raise ValueError("dataset must be 'allen' or 'mousev2'")
    if profile == "pipeline_baseline":
        return pd.Series(True, index=table.index)
    if profile == "common_qc":
        return common_qc_mask(table, dataset=dataset)
    if profile == "published_like":
        return published_like_mask(table, dataset=dataset)
    if profile == "intersection":
        return common_qc_mask(table, dataset=dataset) & published_like_mask(
            table, dataset=dataset
        )

    if dataset != "mousev2":
        raise PopulationUnavailableError(
            "Population profile 'pilot_rf_qc_diagnostic' is MouseV2-only and "
            "cannot define a cross-dataset comparison"
        )
    _require_columns(
        table,
        ("pilot_qc",),
        profile="pilot_rf_qc_diagnostic",
        dataset=dataset,
    )
    return table["pilot_qc"].fillna(False).astype(bool)


def apply_population_profile(
    table: pd.DataFrame,
    *,
    profile: str,
    dataset: str,
) -> pd.DataFrame:
    """Filter a table and retain the selected profile as provenance."""
    selected = table.loc[population_mask(table, profile=profile, dataset=dataset)].copy()
    selected["population_profile"] = profile
    return selected
