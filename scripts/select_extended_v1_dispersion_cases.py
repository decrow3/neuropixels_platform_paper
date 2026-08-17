#!/usr/bin/env python3
"""Fit covariance-trace translations across sessions and select four added cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.checkpoint_v1_absolute_size_dispersion_translation import (
    FEATURES,
    build_full_session_surfaces,
    deterministic_split,
    leave_one_out_template,
    robust_scale,
)
from scripts.drilldown_v1_dispersion_translation import (
    candidate_predictions,
    component_losses,
    make_interpolators,
    optimum_row,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts" / "v1_absolute_size_dispersion_translation_checkpoint"
DEFAULT_INPUT = CHECKPOINT / "uncensored_size_sensitivity" / "v1_unit_descriptors.csv.gz"
DEFAULT_EXISTING = CHECKPOINT / "uncensored_size_sensitivity" / "selected_case_audit.csv"
DEFAULT_ALL_DISPERSION = CHECKPOINT / "uncensored_size_sensitivity" / "translation_optima_all_sessions.csv"
DEFAULT_SUPPORT = CHECKPOINT / "uncensored_size_sensitivity" / "session_support_summary.csv"
DEFAULT_UNITS = ROOT / "data" / "unit_table.csv"
DEFAULT_OUTPUT = CHECKPOINT / "extended_case_selection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--existing-selection", type=Path, default=DEFAULT_EXISTING)
    parser.add_argument("--all-dispersion", type=Path, default=DEFAULT_ALL_DISPERSION)
    parser.add_argument("--support", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--unit-table", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output}; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)

    population = pd.read_csv(args.input.resolve(), low_memory=False)
    axis = np.arange(-90.0, 92.0, 2.0)
    x_mesh, y_mesh = np.meshgrid(axis, axis)
    grid_points = np.column_stack([x_mesh.ravel(), y_mesh.ravel()])
    surfaces, _ = build_full_session_surfaces(population, grid_points, bandwidth=12.0)
    scales = np.array(
        [
            robust_scale(population[feature].to_numpy(float), 0.10 if index == 0 else 0.05)
            for index, feature in enumerate(FEATURES)
        ]
    )
    shift_axis = np.arange(-30.0, 32.0, 2.0)
    shift_az, shift_el = np.meshgrid(shift_axis, shift_axis)
    shifts = np.column_stack([shift_az.ravel(), shift_el.ravel()])
    rows = []
    for session_id, local in population.groupby("ecephys_session_id", observed=True):
        session_id = int(session_id)
        template = leave_one_out_template(surfaces, session_id)[0]
        interpolators = make_interpolators(template, axis)
        half_zero, half_one = deterministic_split(local, session_id)
        local_results = {}
        for subset, table in {"full": local, "half_0": half_zero, "half_1": half_one}.items():
            observed = table[list(FEATURES)].to_numpy(float)
            predicted = candidate_predictions(table, interpolators, shifts)
            losses = component_losses(observed, predicted, scales, (1,))
            result = optimum_row(losses, shifts)
            local_results[subset] = result
            rows.append(
                {
                    "ecephys_session_id": session_id,
                    "target_subset": subset,
                    **result,
                    "at_bound": bool(
                        abs(result["shift_azimuth_deg"]) >= 30
                        or abs(result["shift_elevation_deg"]) >= 30
                    ),
                }
            )
        distance = float(
            np.hypot(
                local_results["half_0"]["shift_azimuth_deg"]
                - local_results["half_1"]["shift_azimuth_deg"],
                local_results["half_0"]["shift_elevation_deg"]
                - local_results["half_1"]["shift_elevation_deg"],
            )
        )
        for row in rows[-3:]:
            row["trace_split_half_vector_difference_deg"] = distance
    optima = pd.DataFrame(rows)

    full = optima.loc[optima["target_subset"].eq("full")].copy()
    support = pd.read_csv(args.support.resolve())
    full = full.merge(support, on="ecephys_session_id", how="left")
    units = pd.read_csv(
        args.unit_table.resolve(),
        usecols=["ecephys_unit_id", "ecephys_session_id", "anterior_posterior_ccf_coordinate"],
        low_memory=False,
    )
    ids = population[["ecephys_unit_id", "ecephys_session_id"]]
    ccf = ids.merge(units, on=["ecephys_unit_id", "ecephys_session_id"], how="left").groupby(
        "ecephys_session_id", observed=True
    )["anterior_posterior_ccf_coordinate"].count().rename("ccf_units").reset_index()
    full = full.merge(ccf, on="ecephys_session_id", how="left")
    full["ccf_available"] = full["ccf_units"].ge(10)
    all_dispersion = pd.read_csv(args.all_dispersion.resolve())
    all_dispersion = all_dispersion.loc[
        all_dispersion["target_subset"].eq("full") & all_dispersion["mode"].eq("dispersion"),
        ["ecephys_session_id", "split_half_vector_difference_deg"],
    ].rename(columns={"split_half_vector_difference_deg": "all_dispersion_split_difference_deg"})
    full = full.merge(all_dispersion, on="ecephys_session_id", how="left")
    full["trace_improvement_over_all_dispersion_deg"] = (
        full["all_dispersion_split_difference_deg"]
        - full["trace_split_half_vector_difference_deg"]
    )

    existing = set(pd.read_csv(args.existing_selection.resolve())["ecephys_session_id"].astype(int))
    eligible = full.loc[
        ~full["ecephys_session_id"].isin(existing)
        & full["v1_units"].ge(60)
        & ~full["at_bound"]
        & np.isfinite(full["trace_split_half_vector_difference_deg"])
    ].copy()
    selected_rows = []
    used = set(existing)

    success_pool = eligible.loc[eligible["ccf_available"] & ~eligible["ecephys_session_id"].isin(used)]
    success = success_pool.sort_values("trace_split_half_vector_difference_deg").iloc[0]
    selected_rows.append(
        {
            "selection_role": "CCF-available covariance-trace success",
            "criterion": "minimum trace split-half difference among non-bound sessions with >=60 V1 units and unit-level CCF",
            **success.to_dict(),
        }
    )
    used.add(int(success.ecephys_session_id))

    remaining = eligible.loc[~eligible["ecephys_session_id"].isin(used)].copy()
    median_value = float(eligible["trace_split_half_vector_difference_deg"].median())
    remaining["distance_to_eligible_median"] = (
        remaining["trace_split_half_vector_difference_deg"] - median_value
    ).abs()
    typical = remaining.sort_values("distance_to_eligible_median").iloc[0]
    selected_rows.append(
        {
            "selection_role": "typical covariance-trace case",
            "criterion": "closest trace split-half difference to the eligible-session median",
            **typical.to_dict(),
        }
    )
    used.add(int(typical.ecephys_session_id))

    rescue_pool = eligible.loc[~eligible["ecephys_session_id"].isin(used)].copy()
    rescue = rescue_pool.sort_values("trace_improvement_over_all_dispersion_deg", ascending=False).iloc[0]
    selected_rows.append(
        {
            "selection_role": "trace rescues full-dispersion instability",
            "criterion": "maximum reduction in split-half difference from all dispersion to covariance trace",
            **rescue.to_dict(),
        }
    )
    used.add(int(rescue.ecephys_session_id))

    failure_pool = full.loc[
        ~full["ecephys_session_id"].isin(used)
        & full["v1_units"].ge(60)
        & np.isfinite(full["trace_split_half_vector_difference_deg"])
    ].copy()
    failure = failure_pool.sort_values(
        ["at_bound", "trace_split_half_vector_difference_deg"], ascending=[False, False]
    ).iloc[0]
    selected_rows.append(
        {
            "selection_role": "covariance-trace failure or boundary case",
            "criterion": "boundary optimum prioritized, then maximum trace split-half difference",
            **failure.to_dict(),
        }
    )
    selected = pd.DataFrame(selected_rows)

    optima.to_csv(output / "covariance_trace_optima_all_sessions.csv", index=False)
    full.to_csv(output / "covariance_trace_session_audit.csv", index=False)
    selected.to_csv(output / "extended_case_selection.csv", index=False)
    manifest = {
        "status": "exploratory extended-case selection",
        "excluded_existing_sessions": sorted(existing),
        "selection_roles": selected[
            ["ecephys_session_id", "selection_role", "criterion"]
        ].to_dict(orient="records"),
        "selection_feature": "RF-space local covariance trace only",
        "outputs": [
            "covariance_trace_optima_all_sessions.csv",
            "covariance_trace_session_audit.csv",
            "extended_case_selection.csv",
        ],
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
