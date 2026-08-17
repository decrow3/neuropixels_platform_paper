#!/usr/bin/env python3
"""Fit rotated point and aperture RFs across one cached Allen population."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from render_allen_point_aperture_rf_examples import parameter_vector
from render_allen_rotated_point_aperture_rf_examples import fit_rotated


ROOT = Path(__file__).resolve().parents[1]
SESSION_ID = 746083955
GAZE_DIR = ROOT / "artifacts" / "allen_population_gaze_rf" / f"session_{SESSION_ID}"
AXIS_DIR = ROOT / "artifacts" / "allen_aperture_rf_comparison" / f"session_{SESSION_ID}"
DEFAULT_OUTPUT = ROOT / "artifacts" / "allen_population_rotated_rf" / f"session_{SESSION_ID}"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit-units", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def choose_units(population, limit):
    selected = population.loc[population["published_like_qc"]].copy()
    if limit is None or limit >= len(selected):
        return selected
    # Deterministic interleaving keeps both anatomical groups and analysis splits.
    pieces = []
    pools = {
        (group, split): list(local.index)
        for (group, split), local in selected.groupby(["group", "unit_split"], observed=True)
    }
    keys = [("V1", "calibration"), ("V1", "evaluation"),
            ("HVA", "calibration"), ("HVA", "evaluation")]
    while len(pieces) < limit and any(pools.values()):
        for key in keys:
            if pools.get(key) and len(pieces) < limit:
                pieces.append(pools[key].pop(0))
    return selected.loc[pieces].sort_index()


def summarize(table):
    augmented = pd.concat([table, table.assign(group="all")], ignore_index=True)
    rows = []
    for (model, split, group), local in augmented.groupby(
        ["spatial_model", "unit_split", "group"], observed=True
    ):
        informative = local["axis_ratio"].ge(1.2) & ~local["sigma_lower_bound"]
        rows.append({
            "spatial_model": model,
            "unit_split": split,
            "group": group,
            "units": len(local),
            "median_rotation_test_gain": local["rotation_test_gain"].median(),
            "fraction_rotation_test_gain_positive": local["rotation_test_gain"].gt(0).mean(),
            "median_rotation_train_gain": local["rotation_train_gain"].median(),
            "median_log2_area_change": local["rotation_log2_area_ratio"].median(),
            "sigma_lower_bound_fraction": local["sigma_lower_bound"].mean(),
            "sigma_upper_bound_fraction": local["sigma_upper_bound"].mean(),
            "informative_angle_units": int(informative.sum()),
            "median_abs_angle_informative_deg": local.loc[
                informative, "major_axis_angle_deg"
            ].abs().median(),
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "rotated_population_fits.csv"
    if checkpoint.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(f"{checkpoint} exists; use --resume or --overwrite")

    population = pd.read_csv(GAZE_DIR / "visual_unit_population.csv", low_memory=False)
    trials = pd.read_csv(GAZE_DIR / "gabor_trial_gaze_table.csv", low_memory=False)
    axis_fits = pd.read_csv(AXIS_DIR / "unit_model_comparison.csv", low_memory=False)
    spike_file = np.load(GAZE_DIR / "gabor_spike_counts.npz")
    counts = spike_file["counts"]
    selected = choose_units(population, args.limit_units)
    selected_ids = set(selected["ecephys_unit_id"].astype(int))

    existing = pd.DataFrame()
    if args.resume and checkpoint.exists():
        existing = pd.read_csv(checkpoint, low_memory=False)
        existing = existing.loc[existing["ecephys_unit_id"].isin(selected_ids)].copy()
    done = set(zip(existing.get("ecephys_unit_id", []), existing.get("spatial_model", [])))
    rows = existing.to_dict("records")

    valid = trials["valid_gaze"].to_numpy(bool)
    train = valid & trials["trial_split"].eq("train").to_numpy(bool)
    test = valid & trials["trial_split"].eq("test").to_numpy(bool)
    x = trials["x_position"].to_numpy(float)
    y = trials["y_position"].to_numpy(float)
    orientation = trials["orientation_index"].to_numpy(int)

    new_fits = 0
    for progress, unit in enumerate(selected.itertuples(), start=1):
        unit_id = int(unit.ecephys_unit_id)
        unit_counts = counts[unit.Index].astype(float)
        for model in ("point", "aperture"):
            if (unit_id, model) in done:
                continue
            axis_row = axis_fits.loc[
                axis_fits["ecephys_unit_id"].eq(unit_id)
                & axis_fits["spatial_model"].eq(model)
                & axis_fits["gaze_condition"].eq("nominal")
            ].iloc[0]
            parameters, metrics = fit_rotated(
                unit_counts, x, y, orientation, train, test, unit.group,
                model, parameter_vector(axis_row),
            )
            rows.append({
                "ecephys_unit_id": unit_id,
                "group": unit.group,
                "ecephys_structure_acronym": unit.ecephys_structure_acronym,
                "unit_split": unit.unit_split,
                "spatial_model": model,
                "rotation_test_gain": axis_row["test_poisson_deviance"]
                - metrics["test_poisson_deviance"],
                "rotation_train_gain": axis_row["train_poisson_deviance"]
                - metrics["train_poisson_deviance"],
                "axis_test_poisson_deviance": axis_row["test_poisson_deviance"],
                "axis_train_poisson_deviance": axis_row["train_poisson_deviance"],
                "axis_area_deg2": axis_row["latent_halfmax_area_deg2"],
                "rotation_log2_area_ratio": np.log2(
                    metrics["latent_halfmax_area_deg2"] / axis_row["latent_halfmax_area_deg2"]
                ),
                "parameter_baseline": parameters[0],
                "parameter_amplitude_0": parameters[1],
                "parameter_amplitude_45": parameters[2],
                "parameter_amplitude_90": parameters[3],
                **metrics,
            })
            new_fits += 1
        if progress % 10 == 0 or progress == len(selected):
            pd.DataFrame(rows).to_csv(checkpoint, index=False, float_format="%.9g")
            print(f"Rotated population: {progress}/{len(selected)} units; {new_fits} new fits", flush=True)

    table = pd.DataFrame(rows).sort_values(["ecephys_unit_id", "spatial_model"])
    table.to_csv(checkpoint, index=False, float_format="%.9g")
    summary = summarize(table)
    summary.to_csv(output / "rotated_population_summary.csv", index=False, float_format="%.9g")
    run = {
        "session_id": SESSION_ID,
        "units": int(table["ecephys_unit_id"].nunique()),
        "fits": int(len(table)),
        "limited_checkpoint": args.limit_units is not None,
        "models": ["point", "aperture"],
        "gaze_condition": "nominal",
        "angle_multistarts_deg": [-89, -45, 0, 45, 89],
        "selection": "published_like_qc",
        "primary_split": "evaluation",
    }
    (output / "run_summary.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
