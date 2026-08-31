"""Measurement-error-corrected HVA/V1 session-variance ratio.

Reconstruct the exact current Figure 3 session x group means from their unit-level
sources, attach finite-neuron standard errors, and de-noise each session's sample
variance across groups:

    latent_variance = observed_sample_variance - mean(group_mean_se ** 2).

For independent group-mean errors this is unbiased even when group precisions
differ. Negative per-session estimates are retained to avoid upward clipping bias.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.figure3_mousev2 import load_allen_units, load_mousev2_units  # noqa: E402
from scripts.figure3_robust_spread_comparison import (  # noqa: E402
    ALLEN_HARMONIZED_F1_F0,
    AREA_ORDER,
    MOUSE_CANONICAL_GRATING_METRICS,
    MOUSE_HARMONIZED_F1_F0,
    MOUSE_PARAMETRIC_RF_FITS,
    MOUSE_TIMESCALE_TRIAL_BRIDGE,
    historical_proxy_full20_population,
    valid_values,
)


CURRENT_MEANS = ROOT / "Figure3/Figure3_robust_session_group_means.csv"
TTFS_BASE = ROOT / "artifacts/figure3/06r_probe_flash_photoartifact_pilot/ttfs_source_provenance"
MOUSE_TTFS_UNITS = TTFS_BASE / "figure3_response_filtered_preferred_ttfs_units.csv"
ALLEN_TTFS_AUDIT = TTFS_BASE / "figure3_response_filtered_ttfs_all_areas_unit_audit.csv"
OUTPUT = ROOT / "artifacts/figure3/07_big_picture_concrete_first/measurement_corrected_variance_ratio"
METRICS = ["TTFS (ms)", "log10 F1/F0", "Response timescale (ms)"]
MIN_UNITS = 5
N_CELL_BOOTSTRAP = 5_000
N_SESSION_BOOTSTRAP = 50_000
SEED = 20260826


def analytic_cell(
    values: np.ndarray,
    *,
    dataset: str,
    metric: str,
    session_id: object,
    group: object,
    source: str,
) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise ValueError("A cell needs at least two finite unit values")
    return {
        "dataset": dataset,
        "metric": metric,
        "session_id": str(session_id),
        "group": str(group),
        "mean": float(values.mean()),
        "n_units": int(len(values)),
        "unique_units_for_se": int(len(values)),
        "group_mean_se": float(values.std(ddof=1) / np.sqrt(len(values))),
        "se_method": "unit analytic SE",
        "source": source,
    }


def build_ttfs_cells() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    mouse = pd.read_csv(MOUSE_TTFS_UNITS)
    mouse = mouse.loc[
        mouse["cohort"].eq("MouseV2")
        & mouse["selected"].astype(bool)
        & pd.to_numeric(mouse["preferred_0_250_ttfs_ms"], errors="coerce").lt(100)
    ].copy()
    for (session_id, group), part in mouse.groupby(["session_id", "location"], sort=True):
        if len(part) < 10:
            continue
        rows.append(
            analytic_cell(
                part["preferred_0_250_ttfs_ms"].to_numpy(float),
                dataset="Within-V1", metric="TTFS (ms)", session_id=session_id,
                group=group, source=str(MOUSE_TTFS_UNITS.relative_to(ROOT)),
            )
        )

    allen = pd.read_csv(ALLEN_TTFS_AUDIT)
    allen = allen.loc[
        allen["area_coarse"].isin(AREA_ORDER)
        & allen["selected_positive_responder_area"].astype(bool)
        & pd.to_numeric(allen["preferred_0_250_ttfs_ms"], errors="coerce").lt(100)
    ].copy()
    for (session_id, group), part in allen.groupby(["session_id", "area_coarse"], sort=True):
        if len(part) < 10:
            continue
        rows.append(
            analytic_cell(
                part["preferred_0_250_ttfs_ms"].to_numpy(float),
                dataset="Post-V1", metric="TTFS (ms)", session_id=session_id,
                group=group, source=str(ALLEN_TTFS_AUDIT.relative_to(ROOT)),
            )
        )
    return pd.DataFrame(rows)


def build_f1_f0_cells() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    mouse = pd.read_csv(MOUSE_HARMONIZED_F1_F0)
    mouse = mouse.loc[
        mouse["default_qc"].astype(bool)
        & pd.to_numeric(mouse["f1_f0_dg_common_support"], errors="coerce").gt(0)
    ].copy()
    mouse["value"] = np.log10(mouse["f1_f0_dg_common_support"].astype(float))
    for (session_id, group), part in mouse.groupby(["site_number", "probe_letter"], sort=True):
        if len(part) < MIN_UNITS:
            continue
        rows.append(
            analytic_cell(
                part["value"].to_numpy(float), dataset="Within-V1",
                metric="log10 F1/F0", session_id=session_id, group=group,
                source=str(MOUSE_HARMONIZED_F1_F0.relative_to(ROOT)),
            )
        )

    allen = pd.read_csv(ALLEN_HARMONIZED_F1_F0)
    allen = allen.loc[
        allen["area_coarse"].isin(AREA_ORDER)
        & pd.to_numeric(allen["f1_f0_dg_harmonized"], errors="coerce").gt(0)
    ].copy()
    allen["value"] = np.log10(allen["f1_f0_dg_harmonized"].astype(float))
    for (session_id, group), part in allen.groupby(
        ["ecephys_session_id", "area_coarse"], sort=True
    ):
        if len(part) < MIN_UNITS:
            continue
        rows.append(
            analytic_cell(
                part["value"].to_numpy(float), dataset="Post-V1",
                metric="log10 F1/F0", session_id=session_id, group=group,
                source=str(ALLEN_HARMONIZED_F1_F0.relative_to(ROOT)),
            )
        )
    return pd.DataFrame(rows)


def matched_timescale_cluster_bootstrap(
    part: pd.DataFrame,
    *,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, float, int, int, float]:
    """Bootstrap units as clusters while preserving all ten trial-subsample fits."""
    matrix = part.pivot(index="unit_id", columns="subsample", values="timescale_ms")
    if matrix.shape[1] != 10:
        raise ValueError("Matched-timescale cell does not contain all ten subsamples")
    values = matrix.to_numpy(float)
    valid = np.isfinite(values)
    observed_draw_means = np.nansum(values, axis=0) / valid.sum(axis=0)
    observed = float(observed_draw_means.mean())
    n_units = len(matrix)
    draws = np.empty(repetitions, dtype=float)
    batch_size = 250
    cursor = 0
    attempted = 0
    while cursor < repetitions:
        count = min(batch_size, max(repetitions - cursor, 1))
        indices = rng.integers(0, n_units, size=(count, n_units))
        attempted += count
        sampled = values[indices]
        sampled_valid = np.isfinite(sampled)
        column_counts = sampled_valid.sum(axis=1)
        keep = np.all(column_counts > 0, axis=1)
        if not np.any(keep):
            if attempted > repetitions * 100:
                raise ValueError("Timescale unit bootstrap acceptance fell below 1%")
            continue
        column_means = np.nansum(sampled[keep], axis=1) / column_counts[keep]
        accepted = column_means.mean(axis=1)
        take = min(len(accepted), repetitions - cursor)
        draws[cursor:cursor + take] = accepted[:take]
        cursor += take
    mean_valid_units = int(np.rint(valid.sum(axis=0).mean()))
    rejection_fraction = 1.0 - repetitions / attempted
    return observed, float(draws.std(ddof=1)), mean_valid_units, n_units, rejection_fraction


def build_timescale_cells(rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    allen = load_allen_units(population_profile="common_qc")
    allen = allen.loc[allen["area_coarse"].isin(AREA_ORDER)]
    for (session_id, group), part in allen.groupby(
        ["ecephys_session_id", "area_coarse"], sort=True
    ):
        values = valid_values(part, "timescale_ac", 2)
        if len(values) < MIN_UNITS:
            continue
        rows.append(
            analytic_cell(
                values, dataset="Post-V1", metric="Response timescale (ms)",
                session_id=session_id, group=group, source="data/unit_table.csv",
            )
        )

    mouse = load_mousev2_units(
        apply_qc=False,
        grating_metrics_dir=MOUSE_CANONICAL_GRATING_METRICS,
        population_profile="common_qc",
    )
    mouse, _, _ = historical_proxy_full20_population(
        mouse, rf_path=MOUSE_PARAMETRIC_RF_FITS
    )
    metadata = mouse[["unit_id", "session_num", "probe_letter"]].drop_duplicates("unit_id")
    bridge = pd.read_csv(MOUSE_TIMESCALE_TRIAL_BRIDGE)
    bridge = bridge.loc[
        bridge["view"].eq("mouse_matched_150")
        & bridge["valid_timescale"].astype(bool)
    ].merge(metadata, on="unit_id", validate="many_to_one")
    draw_counts = (
        bridge.groupby(["session_id", "probe_letter", "subsample"])
        .size().rename("n").reset_index()
    )
    eligible_draws = draw_counts.loc[draw_counts["n"].ge(MIN_UNITS)]
    complete = (
        eligible_draws.groupby(["session_id", "probe_letter"])["subsample"]
        .nunique().loc[lambda x: x.eq(10)].reset_index()[["session_id", "probe_letter"]]
    )
    bridge = bridge.merge(complete, on=["session_id", "probe_letter"], how="inner")
    for (session_id, group), part in bridge.groupby(["session_id", "probe_letter"], sort=True):
        mean, se, mean_n, unique_n, rejection_fraction = matched_timescale_cluster_bootstrap(
            part, repetitions=N_CELL_BOOTSTRAP, rng=rng
        )
        rows.append(
            {
                "dataset": "Within-V1",
                "metric": "Response timescale (ms)",
                "session_id": str(session_id),
                "group": str(group),
                "mean": mean,
                "n_units": mean_n,
                "unique_units_for_se": unique_n,
                "group_mean_se": se,
                "se_method": "unit-cluster bootstrap across ten matched trial draws",
                "bootstrap_rejection_fraction": rejection_fraction,
                "source": str(MOUSE_TIMESCALE_TRIAL_BRIDGE.relative_to(ROOT)),
            }
        )
    return pd.DataFrame(rows)


def validate_against_current(cells: pd.DataFrame) -> pd.DataFrame:
    target = pd.read_csv(CURRENT_MEANS, dtype={"session_id": str})
    target = target[["dataset", "metric", "session_id", "group", "mean", "n_units"]]
    merged = target.merge(
        cells,
        on=["dataset", "metric", "session_id", "group"],
        how="outer",
        suffixes=("_target", "_reconstructed"),
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError(merged.loc[~merged["_merge"].eq("both")].to_string(index=False))
    merged["absolute_mean_error"] = (
        merged["mean_target"] - merged["mean_reconstructed"]
    ).abs()
    merged["n_units_error"] = merged["n_units_target"] - merged["n_units_reconstructed"]
    if merged["absolute_mean_error"].max() > 1e-10:
        raise ValueError(
            f"Reconstructed means differ by up to {merged['absolute_mean_error'].max()}"
        )
    if not merged["n_units_error"].eq(0).all():
        raise ValueError("Reconstructed unit counts differ from current Figure 3")
    return merged


def corrected_session_variances(cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, metric, session_id), part in cells.groupby(
        ["dataset", "metric", "session_id"], sort=False
    ):
        if len(part) < 2:
            continue
        observed = float(part["mean"].var(ddof=1))
        noise = float(np.mean(np.square(part["group_mean_se"])))
        rows.append(
            {
                "dataset": dataset,
                "metric": metric,
                "session_id": str(session_id),
                "n_groups": int(len(part)),
                "observed_variance": observed,
                "estimated_measurement_variance": noise,
                "corrected_variance_unclipped": observed - noise,
                "noise_fraction_of_observed": noise / observed if observed > 0 else np.nan,
                "groups_present": ";".join(sorted(part["group"].astype(str))),
            }
        )
    return pd.DataFrame(rows)


def summarize_ratio(
    sessions: pd.DataFrame,
    *,
    comparison: str,
    exact_hva_groups: int | None,
    min_hva_groups: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        local = sessions.loc[sessions["metric"].eq(metric)]
        v1 = local.loc[
            local["dataset"].eq("Within-V1") & local["n_groups"].eq(4)
        ].copy()
        hva_keep = local["dataset"].eq("Post-V1") & local["n_groups"].ge(min_hva_groups)
        if exact_hva_groups is not None:
            hva_keep &= local["n_groups"].eq(exact_hva_groups)
        hva = local.loc[hva_keep].copy()
        v1_values = v1["corrected_variance_unclipped"].to_numpy(float)
        hva_values = hva["corrected_variance_unclipped"].to_numpy(float)
        v1_draws = rng.choice(
            v1_values, size=(N_SESSION_BOOTSTRAP, len(v1_values)), replace=True
        ).mean(axis=1)
        hva_draws = rng.choice(
            hva_values, size=(N_SESSION_BOOTSTRAP, len(hva_values)), replace=True
        ).mean(axis=1)
        valid = v1_draws > 0
        ratio_draws = hva_draws[valid] / v1_draws[valid]
        difference_draws = hva_draws - v1_draws
        v1_mean = float(v1_values.mean())
        hva_mean = float(hva_values.mean())
        rows.append(
            {
                "comparison": comparison,
                "metric": metric,
                "v1_sessions": len(v1),
                "hva_sessions": len(hva),
                "mean_v1_observed_variance": float(v1["observed_variance"].mean()),
                "mean_hva_observed_variance": float(hva["observed_variance"].mean()),
                "mean_v1_measurement_variance": float(v1["estimated_measurement_variance"].mean()),
                "mean_hva_measurement_variance": float(hva["estimated_measurement_variance"].mean()),
                "mean_v1_corrected_variance": v1_mean,
                "mean_hva_corrected_variance": hva_mean,
                "corrected_hva_to_v1_variance_ratio": hva_mean / v1_mean,
                "ratio_ci_low": float(np.quantile(ratio_draws, 0.025)),
                "ratio_ci_high": float(np.quantile(ratio_draws, 0.975)),
                "bootstrap_probability_ratio_le_1": float(np.mean(ratio_draws <= 1)),
                "bootstrap_fraction_nonpositive_v1_denominator": float(np.mean(~valid)),
                "corrected_hva_minus_v1_variance": hva_mean - v1_mean,
                "difference_ci_low": float(np.quantile(difference_draws, 0.025)),
                "difference_ci_high": float(np.quantile(difference_draws, 0.975)),
                "bootstrap_probability_difference_le_0": float(np.mean(difference_draws <= 0)),
                "mean_v1_noise_fraction": float(v1["estimated_measurement_variance"].mean() / v1["observed_variance"].mean()),
                "mean_hva_noise_fraction": float(hva["estimated_measurement_variance"].mean() / hva["observed_variance"].mean()),
            }
        )
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, validation: pd.DataFrame, path: Path) -> None:
    primary = summary.loc[summary["comparison"].eq("exactly_four_groups")]
    sensitivity = summary.loc[summary["comparison"].eq("hva_at_least_three_groups")]
    lines = [
        "# Measurement-error-corrected Figure 3 variance ratio",
        "",
        "## Exact-current-input gate",
        "",
        f"All {len(validation)} reconstructed session × group cells match the current",
        f"Figure 3 means to maximum absolute error {validation['absolute_mean_error'].max():.3g}",
        "and match every retained unit count.",
        "",
        "## Correction",
        "",
        "Within each session, the observed sample variance across group means is",
        "corrected by subtracting the mean squared group-mean SE. Negative session",
        "estimates are retained. TTFS, F1/F0, and Allen timescale use unit-level",
        "analytic SEs. MouseV2 matched-timescale SEs use a unit-cluster bootstrap",
        "that preserves each unit's ten trial-subsample fits.",
        "",
        "## Primary result: exactly four groups in each dataset",
        "",
        "| Metric | Sessions V1/HVA | Observed variance V1/HVA | Estimated noise V1/HVA | Corrected variance V1/HVA | Corrected HVA/V1 ratio | Conditional ratio interval | Corrected difference (95% CI) | P(difference <= 0) | Nonpositive V1 bootstrap means |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary.itertuples(index=False):
        lines.append(
            f"| {row.metric} | {row.v1_sessions}/{row.hva_sessions} | "
            f"{row.mean_v1_observed_variance:.5g}/{row.mean_hva_observed_variance:.5g} | "
            f"{row.mean_v1_measurement_variance:.5g}/{row.mean_hva_measurement_variance:.5g} | "
            f"{row.mean_v1_corrected_variance:.5g}/{row.mean_hva_corrected_variance:.5g} | "
            f"{row.corrected_hva_to_v1_variance_ratio:.2f}× | "
            f"[{row.ratio_ci_low:.2f}, {row.ratio_ci_high:.2f}] | "
            f"{row.corrected_hva_minus_v1_variance:.5g} "
            f"([{row.difference_ci_low:.5g}, {row.difference_ci_high:.5g}]) | "
            f"{row.bootstrap_probability_difference_le_0:.4f} | "
            f"{row.bootstrap_fraction_nonpositive_v1_denominator:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Sensitivity: four V1 probes versus at least three HVA areas",
            "",
            "| Metric | HVA sessions | Corrected HVA/V1 ratio | Conditional ratio interval | Corrected difference (95% CI) | P(difference <= 0) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sensitivity.itertuples(index=False):
        lines.append(
            f"| {row.metric} | {row.hva_sessions} | "
            f"{row.corrected_hva_to_v1_variance_ratio:.2f}× | "
            f"[{row.ratio_ci_low:.2f}, {row.ratio_ci_high:.2f}] | "
            f"{row.corrected_hva_minus_v1_variance:.5g} "
            f"([{row.difference_ci_low:.5g}, {row.difference_ci_high:.5g}]) | "
            f"{row.bootstrap_probability_difference_le_0:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Limits",
            "",
            "The correction treats neuron-sampling errors for different groups as",
            "independent and conditions on all unit-selection and metric-validity gates.",
            "It does not propagate shared trial noise, spike-sorting dependence, or",
            "uncertainty from selecting TTFS responders and valid timescale fits.",
            "For TTFS and timescale, 18–23% of primary V1 session-bootstrap means",
            "are nonpositive after unbiased subtraction. Their ratio intervals are",
            "therefore conditional on a positive denominator and intrinsically unstable;",
            "the unbounded variance difference is the more reliable uncertainty summary.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    cells = pd.concat(
        [build_ttfs_cells(), build_f1_f0_cells(), build_timescale_cells(rng)],
        ignore_index=True,
    )
    validation = validate_against_current(cells)
    sessions = corrected_session_variances(cells)
    primary = summarize_ratio(
        sessions, comparison="exactly_four_groups", exact_hva_groups=4,
        min_hva_groups=4, rng=rng,
    )
    sensitivity = summarize_ratio(
        sessions, comparison="hva_at_least_three_groups", exact_hva_groups=None,
        min_hva_groups=3, rng=rng,
    )
    summary = pd.concat([primary, sensitivity], ignore_index=True)
    cells.to_csv(OUTPUT / "current_session_group_measurement_error.csv", index=False)
    validation.to_csv(OUTPUT / "exact_current_input_validation.csv", index=False)
    sessions.to_csv(OUTPUT / "corrected_session_variances.csv", index=False)
    summary.to_csv(OUTPUT / "measurement_corrected_variance_ratio_summary.csv", index=False)
    write_report(summary, validation, OUTPUT / "MEASUREMENT_CORRECTED_VARIANCE_RATIO.md")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
