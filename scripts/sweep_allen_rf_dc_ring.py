#!/usr/bin/env python3
"""Sensitivity of the exploratory DC-return ring under artificial RF cropping."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import validate_allen_rf_multicase_cropping as crop


RADIUS_PAIRS = ((3.0, 4.0), (4.0, 5.0), (5.0, 6.0))
WEIGHTS = (1.0, 4.0, 16.0)
OUTPUT = crop.DEFAULT_OUTPUT


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def crop_state(full_mask, full_peak, x_coordinates, y_coordinates):
    retained = full_mask[np.ix_(y_coordinates.astype(int), x_coordinates.astype(int))]
    component_censored = bool(retained.sum() < full_mask.sum())
    peak_removed = bool(
        full_peak[0] not in y_coordinates or full_peak[1] not in x_coordinates
    )
    if peak_removed:
        return "original peak removed"
    if component_censored:
        return "component censored; peak retained"
    return "component intact"


def fit_record(matrix, x_coordinates, y_coordinates, center, radius, weight):
    parameters, audit = crop.fit_baseline_variant(
        matrix,
        x_coordinates,
        y_coordinates,
        center_extension_px=2.0,
        sigma_upper_px=8.0,
        ring_center=center if radius is not None else None,
        ring_radius_px=radius,
        ring_total_weight=weight,
    )
    return {
        **audit,
        "center_x_px": parameters[3],
        "center_y_px": parameters[2],
        "major_sigma_deg": 10.0 * max(parameters[4], parameters[5]),
    }


def run_sweep(cases):
    rows = []
    configurations = [("no ring", None, 0.0, np.nan, np.nan)]
    for v1_radius, hva_radius in RADIUS_PAIRS:
        for weight in WEIGHTS:
            label = f"ring V1/HVA {int(v1_radius * 10)}/{int(hva_radius * 10)} deg; weight {weight:g}"
            configurations.append((label, (v1_radius, hva_radius), weight, v1_radius, hva_radius))

    for case in cases.itertuples(index=False):
        unit_id = int(case.ecephys_unit_id)
        matrix = np.loadtxt(
            crop.DEFAULT_MAPS / f"unit_{unit_id}_observed_map.csv", delimiter=","
        )
        full_x = np.arange(matrix.shape[1], dtype=float)
        full_y = np.arange(matrix.shape[0], dtype=float)
        full_mask, _, _, _, _ = crop.threshold_metrics(matrix, full_x, full_y)
        full_peak = np.unravel_index(np.argmax(matrix), matrix.shape)
        for direction in crop.DIRECTIONS:
            for removed in range(5):
                cropped, x_coordinates, y_coordinates = crop.crop_map(
                    matrix, direction, removed
                )
                _, threshold_x, threshold_y, _, _ = crop.threshold_metrics(
                    cropped, x_coordinates, y_coordinates
                )
                state = crop_state(full_mask, full_peak, x_coordinates, y_coordinates)
                for label, radii, weight, v1_radius, hva_radius in configurations:
                    radius = None
                    if radii is not None:
                        radius = radii[0] if case.group == "V1" else radii[1]
                    fit = fit_record(
                        cropped,
                        x_coordinates,
                        y_coordinates,
                        (threshold_x, threshold_y),
                        radius,
                        weight,
                    )
                    rows.append(
                        {
                            "unit_id": unit_id,
                            "group": case.group,
                            "direction": direction,
                            "rows_or_columns_removed": removed,
                            "crop_stratum": state,
                            "configuration": label,
                            "v1_radius_deg": v1_radius * 10 if np.isfinite(v1_radius) else np.nan,
                            "hva_radius_deg": hva_radius * 10 if np.isfinite(hva_radius) else np.nan,
                            "ring_weight_pixel_equivalents": weight,
                            **fit,
                        }
                    )
    result = pd.DataFrame(rows)
    for (_, configuration, direction), indices in result.groupby(
        ["unit_id", "configuration", "direction"], observed=True
    ).groups.items():
        local = result.loc[indices]
        reference = local.loc[local["rows_or_columns_removed"].eq(0)].iloc[0]
        result.loc[indices, "absolute_log2_major_sigma_ratio"] = np.abs(
            np.log2(local["major_sigma_deg"] / reference["major_sigma_deg"])
        )
        result.loc[indices, "center_error_deg"] = 10.0 * np.hypot(
            local["center_x_px"] - reference["center_x_px"],
            local["center_y_px"] - reference["center_y_px"],
        )
    return result


def summarize(trajectories):
    cropped = trajectories.loc[trajectories["rows_or_columns_removed"].gt(0)].copy()
    full = trajectories.loc[trajectories["rows_or_columns_removed"].eq(0)].drop_duplicates(
        ["unit_id", "configuration"]
    )
    no_ring = full.loc[full["configuration"].eq("no ring")].set_index("unit_id")
    rows = []
    for configuration, selected in cropped.groupby("configuration", observed=True):
        full_selected = full.loc[full["configuration"].eq(configuration)].set_index("unit_id")
        common = full_selected.index.intersection(no_ring.index)
        base = {
            "configuration": configuration,
            "v1_radius_deg": selected["v1_radius_deg"].iloc[0],
            "hva_radius_deg": selected["hva_radius_deg"].iloc[0],
            "ring_weight_pixel_equivalents": selected["ring_weight_pixel_equivalents"].iloc[0],
            "full_map_median_sigma_ratio_to_no_ring": np.median(
                full_selected.loc[common, "major_sigma_deg"] / no_ring.loc[common, "major_sigma_deg"]
            ),
            "full_map_median_rmse_increase": np.median(
                full_selected.loc[common, "data_rmse"] - no_ring.loc[common, "data_rmse"]
            ),
        }
        for stratum in (
            "component intact",
            "component censored; peak retained",
            "original peak removed",
        ):
            local = selected.loc[selected["crop_stratum"].eq(stratum)]
            prefix = {
                "component intact": "intact",
                "component censored; peak retained": "censored_peak_retained",
                "original peak removed": "peak_removed",
            }[stratum]
            base[f"{prefix}_fits"] = len(local)
            base[f"{prefix}_median_abs_log2_sigma_error"] = local[
                "absolute_log2_major_sigma_ratio"
            ].median()
            base[f"{prefix}_q90_center_error_deg"] = local["center_error_deg"].quantile(0.9)
            base[f"{prefix}_failure_or_bound_fraction"] = (
                ~local["success"].astype(bool) | local["at_bound"].astype(bool)
            ).mean()
        rows.append(base)
    return pd.DataFrame(rows).sort_values(
        ["censored_peak_retained_q90_center_error_deg", "censored_peak_retained_median_abs_log2_sigma_error"]
    )


def render(summary, path):
    rings = summary.loc[~summary["configuration"].eq("no ring")].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    for weight, selected in rings.groupby("ring_weight_pixel_equivalents"):
        selected = selected.sort_values("v1_radius_deg")
        label = f"weight {weight:g} cells"
        axes[0].plot(selected["v1_radius_deg"], selected["censored_peak_retained_median_abs_log2_sigma_error"], marker="o", label=label)
        axes[1].plot(selected["v1_radius_deg"], selected["censored_peak_retained_q90_center_error_deg"], marker="o", label=label)
        axes[2].plot(selected["v1_radius_deg"], selected["full_map_median_sigma_ratio_to_no_ring"], marker="o", label=label)
    no_ring = summary.loc[summary["configuration"].eq("no ring")].iloc[0]
    axes[0].axhline(no_ring["censored_peak_retained_median_abs_log2_sigma_error"], color="#777777", linestyle="--", label="no ring")
    axes[1].axhline(no_ring["censored_peak_retained_q90_center_error_deg"], color="#777777", linestyle="--")
    axes[2].axhline(1, color="#777777", linestyle="--")
    axes[0].set(title="Censored component: width error", ylabel="Median |log₂(crop/full major σ)|")
    axes[1].set(title="Censored component: center tail", ylabel="90th-percentile center error (deg)")
    axes[2].set(title="Full-map shrinkage", ylabel="Median σ ratio to no-ring fit")
    for axis in axes:
        axis.set_xlabel("V1 ring radius (deg); HVA is +10°")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("DC-return ring sensitivity across four native RF maps")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    output = OUTPUT.resolve()
    cases = crop.select_cases(crop.DEFAULT_UNITS)
    trajectories = run_sweep(cases)
    summary = summarize(trajectories)
    trajectory_path = output / "dc_ring_sensitivity_trajectories.csv"
    summary_path = output / "dc_ring_sensitivity_summary.csv"
    figure_path = output / "Figure_dc_ring_sensitivity.png"
    trajectories.to_csv(trajectory_path, index=False, float_format="%.8g")
    summary.to_csv(summary_path, index=False, float_format="%.8g")
    render(summary, figure_path)
    manifest_path = output / "dc_ring_sensitivity_manifest.json"
    manifest = {
        "radius_pairs_px_v1_hva": RADIUS_PAIRS,
        "ring_weights_in_observed_pixel_equivalents": WEIGHTS,
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)},
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (trajectory_path, summary_path, figure_path)
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote DC-ring sensitivity sweep to {output}")


if __name__ == "__main__":
    main()
