#!/usr/bin/env python3
"""Audit timescale inclusion criteria and descriptive variability sensitivities."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.figure3_mousev2 import load_allen_units, load_mousev2_units  # noqa: E402
from scripts.figure3_robust_spread_comparison import (  # noqa: E402
    MOUSE_CANONICAL_GRATING_METRICS,
    MOUSE_PARAMETRIC_RF_FITS,
    MOUSE_TIMESCALE_TRIAL_BRIDGE,
    historical_proxy_full20_population,
)


OUTPUT = ROOT / (
    "artifacts/figure3/07_big_picture_concrete_first/"
    "timescale_inclusion_variability_audit"
)
HVA_GROUPS = ["LM", "RL", "AL", "PM", "AM"]
MIN_UNITS = 5


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) / weights.sum()
    return float(np.interp(q, cumulative, values))


def weighted_sd(values: np.ndarray, weights: np.ndarray) -> float:
    mean = np.average(values, weights=weights)
    return float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))


def retain_complete_cells(
    table: pd.DataFrame, *, draw_column: str | None,
) -> pd.DataFrame:
    if draw_column is None:
        counts = table.groupby(["session_id", "group"]).size()
        keep = counts.loc[lambda x: x.ge(MIN_UNITS)].index
    else:
        counts = table.groupby(["session_id", "group", draw_column]).size()
        eligible = counts.loc[lambda x: x.ge(MIN_UNITS)].reset_index()
        keep = (
            eligible.groupby(["session_id", "group"])[draw_column]
            .nunique().loc[lambda x: x.eq(10)].index
        )
    index = pd.MultiIndex.from_frame(table[["session_id", "group"]])
    return table.loc[index.isin(keep)].copy()


def add_weights(table: pd.DataFrame, *, draw_column: str | None) -> pd.DataFrame:
    result = table.copy()
    if draw_column is None:
        draws = 1
    else:
        draws = result.groupby(["group", "session_id", "unit_id"])[draw_column].transform("nunique")
    neurons = result.groupby(["group", "session_id"])["unit_id"].transform("nunique")
    sessions = result.groupby("group")["session_id"].transform("nunique")
    result["weight"] = 1.0 / (draws * neurons * sessions)
    return result


def summarize(
    table: pd.DataFrame, *, source: str, scenario: str, draw_column: str | None,
) -> dict[str, object]:
    weighted = add_weights(table, draw_column=draw_column)
    group_rows = []
    session_rows = []
    for group, local in weighted.groupby("group"):
        values = local["value"].to_numpy(float)
        weights = local["weight"].to_numpy(float)
        group_rows.append({
            "group": group,
            "median": weighted_quantile(values, weights, 0.5),
            "sd": weighted_sd(values, weights),
            "tail100": np.average(values > 100, weights=weights),
        })
        if draw_column is None:
            local_means = local.groupby("session_id")["value"].mean()
        else:
            draw_means = local.groupby(["session_id", draw_column])["value"].mean()
            local_means = draw_means.groupby("session_id").mean()
        session_rows.append((group, float(local_means.mean())))
    group_summary = pd.DataFrame(group_rows)
    session_centers = pd.Series(dict(session_rows), dtype=float)
    return {
        "source": source,
        "scenario": scenario,
        "rows": len(table),
        "unique_neurons": table["unit_id"].nunique(),
        "sessions": table["session_id"].nunique(),
        "session_group_cells": table[["session_id", "group"]].drop_duplicates().shape[0],
        "groups": table["group"].nunique(),
        "mean_group_weighted_sd_ms": group_summary["sd"].mean(),
        "group_median_span_ms": group_summary["median"].max() - group_summary["median"].min(),
        "group_session_center_span_ms": session_centers.max() - session_centers.min(),
        "mean_group_tail_over_100": group_summary["tail100"].mean(),
    }


def build_mouse_base() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common = load_mousev2_units(
        apply_qc=False,
        grating_metrics_dir=MOUSE_CANONICAL_GRATING_METRICS,
        population_profile="common_qc",
    )
    primary, rf_flow, _ = historical_proxy_full20_population(
        common, rf_path=MOUSE_PARAMETRIC_RF_FITS
    )
    bridge = pd.read_csv(MOUSE_TIMESCALE_TRIAL_BRIDGE, dtype={"unit_id": str})
    bridge = bridge.loc[bridge["view"].eq("mouse_matched_150")].copy()
    bridge["session_id"] = bridge["session_id"].astype(str)
    bridge["unit_id"] = bridge["unit_id"].astype(str)
    metadata = common[["unit_id", "probe_letter"]].copy()
    metadata["unit_id"] = metadata["unit_id"].astype(str)
    bridge = bridge.merge(metadata, on="unit_id", how="inner", validate="many_to_one")
    bridge = bridge.rename(columns={"probe_letter": "group", "timescale_ms": "value"})
    primary_ids = set(primary["unit_id"].astype(str))
    bridge["primary_population"] = bridge["unit_id"].isin(primary_ids)
    return common, bridge, rf_flow


def select_mouse(
    bridge: pd.DataFrame, *, primary_population: bool, error_lt: float = 20,
    spikes_gt: float = 50, tau_max: float = 300, complete_neuron_draws: bool = False,
) -> pd.DataFrame:
    keep = (
        bridge["fit_ok"].astype(bool)
        & pd.to_numeric(bridge["value"], errors="coerce").between(1, tau_max)
        & pd.to_numeric(bridge["fit_error_ms"], errors="coerce").lt(error_lt)
        & pd.to_numeric(bridge["spike_count"], errors="coerce").gt(spikes_gt)
    )
    if primary_population:
        keep &= bridge["primary_population"]
    selected = bridge.loc[keep].copy()
    if complete_neuron_draws:
        counts = selected.groupby("unit_id")["subsample"].nunique()
        selected = selected.loc[selected["unit_id"].isin(counts.loc[lambda x: x.eq(10)].index)]
    return retain_complete_cells(selected, draw_column="subsample")


def build_allen_base(profile: str) -> pd.DataFrame:
    allen = load_allen_units(population_profile=profile)
    allen = allen.loc[allen["area_coarse"].isin(["V1", *HVA_GROUPS])].copy()
    return allen.rename(columns={
        "ecephys_session_id": "session_id",
        "ecephys_unit_id": "unit_id",
        "area_coarse": "group",
        "timescale_ac": "value",
        "err_ac": "fit_error",
        "spike_count_ac": "spike_count",
    })


def select_allen(
    allen: pd.DataFrame, *, groups: list[str], error_lt: float = 20,
    spikes_gt: float = 50, tau_max: float = 300,
) -> pd.DataFrame:
    local = allen.loc[allen["group"].isin(groups)].copy()
    keep = (
        pd.to_numeric(local["value"], errors="coerce").between(1, tau_max)
        & pd.to_numeric(local["fit_error"], errors="coerce").lt(error_lt)
        & pd.to_numeric(local["spike_count"], errors="coerce").gt(spikes_gt)
    )
    local = local.loc[keep].copy()
    local["session_id"] = local["session_id"].astype(str)
    local["unit_id"] = local["unit_id"].astype(str)
    return retain_complete_cells(local, draw_column=None)


def inclusion_flow(
    common_mouse: pd.DataFrame, bridge: pd.DataFrame, rf_flow: pd.DataFrame,
    allen_common: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for record in rf_flow.itertuples(index=False):
        rows.append({
            "source": "MouseV2 V1", "step": record.step, "rule": record.rule,
            "input": record.input_units, "retained": record.retained_units,
            "excluded": record.excluded_at_step,
        })
    primary = bridge.loc[bridge["primary_population"]]
    gates = [
        ("matched_fit_available", primary["fit_attempted"].astype(bool), "Matched 150-flash fit attempted"),
        ("tau_1_300", pd.to_numeric(primary["value"], errors="coerce").between(1, 300), "1 <= tau <= 300 ms"),
        ("spikes_gt50", pd.to_numeric(primary["spike_count"], errors="coerce").gt(50), ">50 fitting-window spikes"),
        ("error_lt20", pd.to_numeric(primary["fit_error_ms"], errors="coerce").lt(20), "Fitted tau error <20 ms"),
    ]
    keep = pd.Series(True, index=primary.index)
    for step, gate, rule in gates:
        before = int(keep.sum()); keep &= gate; after = int(keep.sum())
        rows.append({"source": "MouseV2 V1 matched-fit rows", "step": step, "rule": rule,
                     "input": before, "retained": after, "excluded": before - after})
    before = int(keep.sum())
    final_mouse = select_mouse(bridge, primary_population=True)
    rows.append({
        "source": "MouseV2 V1 matched-fit rows", "step": "complete_session_probe",
        "rule": "Each session x probe has >=5 valid neurons in all 10 draws",
        "input": before, "retained": len(final_mouse), "excluded": before - len(final_mouse),
    })

    for label, groups in [("Allen VISp", ["V1"]), ("Allen cortical HVAs", HVA_GROUPS)]:
        local = allen_common.loc[allen_common["group"].isin(groups)]
        gates = [
            ("tau_1_300", pd.to_numeric(local["value"], errors="coerce").between(1, 300), "1 <= tau <= 300 ms"),
            ("spikes_gt50", pd.to_numeric(local["spike_count"], errors="coerce").gt(50), ">50 fitting-window spikes"),
            ("error_lt20", pd.to_numeric(local["fit_error"], errors="coerce").lt(20), "Fitted tau error <20 ms"),
        ]
        keep = pd.Series(True, index=local.index)
        rows.append({"source": label, "step": "common_qc", "rule": "Common waveform QC",
                     "input": len(local), "retained": len(local), "excluded": 0})
        for step, gate, rule in gates:
            before = int(keep.sum()); keep &= gate; after = int(keep.sum())
            rows.append({"source": label, "step": step, "rule": rule,
                         "input": before, "retained": after, "excluded": before - after})
        valid = local.loc[keep].copy()
        valid["session_id"] = valid["session_id"].astype(str)
        valid["unit_id"] = valid["unit_id"].astype(str)
        final = retain_complete_cells(valid, draw_column=None)
        rows.append({
            "source": label, "step": "min_session_area_population",
            "rule": ">=5 valid neurons per session x area", "input": len(valid),
            "retained": len(final), "excluded": len(valid) - len(final),
        })
    return pd.DataFrame(rows)


def select_tail_cases(
    mouse: pd.DataFrame, visp: pd.DataFrame, hva: pd.DataFrame,
) -> pd.DataFrame:
    mouse_cases = (
        mouse.groupby(["session_id", "group", "unit_id"], as_index=False)
        .agg(
            max_tau_ms=("value", "max"), median_tau_ms=("value", "median"),
            valid_draws=("subsample", "nunique"),
            median_fit_error_ms=("fit_error_ms", "median"),
            median_spike_count=("spike_count", "median"),
        )
        .nlargest(12, "max_tau_ms")
    )
    mouse_cases.insert(0, "source", "MouseV2 V1")
    mouse_cases.insert(1, "selection_role", "largest neuron-level maximum")
    allen_rows = []
    for source, table in [("Allen VISp", visp), ("Allen cortical HVA", hva)]:
        local = table.nlargest(12, "value").copy()
        for record in local.itertuples(index=False):
            allen_rows.append({
                "source": source, "selection_role": "largest valid neuron value",
                "session_id": record.session_id, "group": record.group,
                "unit_id": record.unit_id, "max_tau_ms": record.value,
                "median_tau_ms": record.value, "valid_draws": 1,
                "median_fit_error_ms": record.fit_error,
                "median_spike_count": record.spike_count,
            })
    return pd.concat([mouse_cases, pd.DataFrame(allen_rows)], ignore_index=True)


def associations(
    mouse: pd.DataFrame, visp: pd.DataFrame, hva: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for source, table, error_column, spike_column in [
        ("MouseV2 V1 fit rows", mouse, "fit_error_ms", "spike_count"),
        ("Allen VISp neurons", visp, "fit_error", "spike_count"),
        ("Allen HVA neurons", hva, "fit_error", "spike_count"),
    ]:
        for factor in [error_column, spike_column]:
            rho, p = spearmanr(table["value"], table[factor], nan_policy="omit")
            rows.append({"source": source, "factor": factor, "spearman_rho": rho,
                         "p_value": p, "rows": len(table)})
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    common_mouse, bridge, rf_flow = build_mouse_base()
    allen_common = build_allen_base("common_qc")
    allen_intersection = build_allen_base("intersection")

    scenarios = []
    mouse_specs = [
        ("primary", dict(primary_population=True)),
        ("common_qc_without_rf_proxy", dict(primary_population=False)),
        ("primary_error_lt10", dict(primary_population=True, error_lt=10)),
        ("primary_spikes_gt100", dict(primary_population=True, spikes_gt=100)),
        ("primary_complete_10_draw_neurons", dict(primary_population=True, complete_neuron_draws=True)),
        ("primary_tau_le100", dict(primary_population=True, tau_max=100)),
    ]
    mouse_tables = {}
    for name, kwargs in mouse_specs:
        table = select_mouse(bridge, **kwargs)
        mouse_tables[name] = table
        scenarios.append(summarize(table, source="MouseV2 V1", scenario=name, draw_column="subsample"))

    allen_tables = {}
    for label, groups in [("Allen VISp", ["V1"]), ("Allen cortical HVAs", HVA_GROUPS)]:
        specs = [
            ("common_qc_primary", allen_common, {}),
            ("intersection_qc", allen_intersection, {}),
            ("common_qc_error_lt10", allen_common, {"error_lt": 10}),
            ("common_qc_spikes_gt100", allen_common, {"spikes_gt": 100}),
            ("common_qc_tau_le100", allen_common, {"tau_max": 100}),
        ]
        for name, source, kwargs in specs:
            table = select_allen(source, groups=groups, **kwargs)
            allen_tables[(label, name)] = table
            scenarios.append(summarize(table, source=label, scenario=name, draw_column=None))

    flow = inclusion_flow(common_mouse, bridge, rf_flow, allen_common)
    sensitivity = pd.DataFrame(scenarios)
    association = associations(
        mouse_tables["primary"],
        allen_tables[("Allen VISp", "common_qc_primary")],
        allen_tables[("Allen cortical HVAs", "common_qc_primary")],
    )
    draw_coverage = (
        mouse_tables["primary"].groupby(["session_id", "group", "unit_id"])["subsample"]
        .nunique().value_counts().sort_index().rename_axis("valid_draws")
        .reset_index(name="neurons")
    )
    tail_cases = select_tail_cases(
        mouse_tables["primary"],
        allen_tables[("Allen VISp", "common_qc_primary")],
        allen_tables[("Allen cortical HVAs", "common_qc_primary")],
    )

    flow.to_csv(OUTPUT / "inclusion_flow.csv", index=False)
    sensitivity.to_csv(OUTPUT / "variability_sensitivity.csv", index=False)
    association.to_csv(OUTPUT / "timescale_factor_associations.csv", index=False)
    draw_coverage.to_csv(OUTPUT / "mousev2_valid_draw_coverage.csv", index=False)
    tail_cases.to_csv(OUTPUT / "selected_tail_cases.csv", index=False)
    print("INCLUSION FLOW\n", flow.to_string(index=False))
    print("\nSENSITIVITY\n", sensitivity.to_string(index=False))
    print("\nASSOCIATIONS\n", association.to_string(index=False))
    print("\nDRAW COVERAGE\n", draw_coverage.to_string(index=False))


if __name__ == "__main__":
    main()
