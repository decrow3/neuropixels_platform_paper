#!/usr/bin/env python3
"""Trace RF metric bias as rows or columns encroach on a stable interior RF."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from allensdk.brain_observatory.ecephys.stimulus_analysis.receptive_field_mapping import (
    fit_2d_gaussian,
    threshold_rf,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP = (
    ROOT / "artifacts" / "allen_rf_improved_fit_diagnostic" / "checkpoint1"
    / "unit_951867908_observed_map.csv"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "allen_rf_artificial_cropping" / "checkpoint1"
UNIT_ID = 951867908
DIRECTIONS = ("top", "bottom", "left", "right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum-removed", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def crop_map(matrix: np.ndarray, direction: str, removed: int):
    rows, columns = matrix.shape
    if direction == "top":
        row_slice, column_slice = slice(removed, rows), slice(0, columns)
    elif direction == "bottom":
        row_slice, column_slice = slice(0, rows - removed), slice(0, columns)
    elif direction == "left":
        row_slice, column_slice = slice(0, rows), slice(removed, columns)
    elif direction == "right":
        row_slice, column_slice = slice(0, rows), slice(0, columns - removed)
    else:
        raise ValueError(direction)
    cropped = matrix[row_slice, column_slice]
    y_coordinates = np.arange(rows, dtype=float)[row_slice]
    x_coordinates = np.arange(columns, dtype=float)[column_slice]
    return cropped, x_coordinates, y_coordinates


def gaussian_prediction(parameters, x_mesh, y_mesh):
    baseline, amplitude, center_y, center_x, sigma_y, sigma_x = parameters
    return baseline + amplitude * np.exp(
        -0.5
        * (
            ((y_mesh - center_y) / sigma_y) ** 2
            + ((x_mesh - center_x) / sigma_x) ** 2
        )
    )


def fit_baseline_gaussian(matrix, x_coordinates, y_coordinates):
    x_mesh, y_mesh = np.meshgrid(x_coordinates, y_coordinates)
    baseline = max(float(np.quantile(matrix, 0.2)), 0.0)
    peak_row, peak_column = np.unravel_index(np.argmax(matrix), matrix.shape)
    initial = np.array(
        [
            baseline,
            max(float(matrix.max() - baseline), 1e-3),
            y_coordinates[peak_row],
            x_coordinates[peak_column],
            1.5,
            1.5,
        ]
    )
    lower = np.array(
        [0.0, 0.0, y_coordinates.min(), x_coordinates.min(), 0.35, 0.35]
    )
    upper = np.array(
        [np.inf, np.inf, y_coordinates.max(), x_coordinates.max(), 4.0, 4.0]
    )
    fit = least_squares(
        lambda p: (gaussian_prediction(p, x_mesh, y_mesh) - matrix).ravel(),
        np.clip(initial, lower + 1e-8, upper - 1e-8),
        bounds=(lower, upper),
        max_nfev=20000,
        method="trf",
    )
    prediction = gaussian_prediction(fit.x, x_mesh, y_mesh)
    at_bound = bool(
        np.any(np.isclose(fit.x, lower, atol=1e-5, rtol=0))
        or np.any(np.isclose(fit.x, upper, atol=1e-5, rtol=0))
    )
    return fit.x, prediction, bool(fit.success), at_bound


def mask_touches_edge(mask: np.ndarray) -> bool:
    return bool(
        mask[0, :].any()
        or mask[-1, :].any()
        or mask[:, 0].any()
        or mask[:, -1].any()
    )


def analyze_crop(matrix, direction, removed):
    cropped, x_coordinates, y_coordinates = crop_map(matrix, direction, removed)
    x_offset = x_coordinates.min()
    y_offset = y_coordinates.min()

    allen_parameters, allen_success = fit_2d_gaussian(cropped)
    allen_parameters = np.asarray(allen_parameters, dtype=float)
    baseline_parameters, baseline_prediction, baseline_success, baseline_at_bound = (
        fit_baseline_gaussian(cropped, x_coordinates, y_coordinates)
    )
    mask, mask_x_local, mask_y_local, area_pixels = threshold_rf(cropped, 1.0)
    mask_x = mask_x_local + x_offset
    mask_y = mask_y_local + y_offset
    return {
        "unit_id": UNIT_ID,
        "direction": direction,
        "rows_or_columns_removed": removed,
        "remaining_rows": cropped.shape[0],
        "remaining_columns": cropped.shape[1],
        "crop_x_min_px": x_coordinates.min(),
        "crop_x_max_px": x_coordinates.max(),
        "crop_y_min_px": y_coordinates.min(),
        "crop_y_max_px": y_coordinates.max(),
        "allen_success": bool(allen_success),
        # Allen's helper passes np.indices as (x, y) to a function whose
        # arguments are effectively (row, column), so its returned center_y
        # tracks columns and center_x tracks rows. This is hidden for 9x9 maps
        # but must be corrected for rectangular artificial crops.
        "allen_center_x_px": allen_parameters[1] + x_offset,
        "allen_center_y_px": allen_parameters[2] + y_offset,
        "allen_width_sigma_deg": abs(allen_parameters[4]) * 10.0,
        "allen_height_sigma_deg": abs(allen_parameters[3]) * 10.0,
        "allen_major_sigma_deg": max(abs(allen_parameters[3]), abs(allen_parameters[4])) * 10.0,
        "baseline_success": baseline_success,
        "baseline_at_bound": baseline_at_bound,
        "baseline_spike_count": baseline_parameters[0],
        "baseline_center_x_px": baseline_parameters[3],
        "baseline_center_y_px": baseline_parameters[2],
        "baseline_width_sigma_deg": baseline_parameters[5] * 10.0,
        "baseline_height_sigma_deg": baseline_parameters[4] * 10.0,
        "baseline_major_sigma_deg": max(baseline_parameters[4], baseline_parameters[5]) * 10.0,
        "baseline_rmse_per_pixel": float(
            np.sqrt(np.square(baseline_prediction - cropped).mean())
        ),
        "threshold_center_x_px": mask_x,
        "threshold_center_y_px": mask_y,
        "threshold_area_pixels": area_pixels,
        "threshold_area_deg2": area_pixels * 100.0,
        "threshold_component_touches_crop_edge": mask_touches_edge(mask),
    }, cropped


def add_reference_changes(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    reference = result.loc[result["rows_or_columns_removed"].eq(0)].iloc[0]
    for model in ("allen", "baseline"):
        result[f"{model}_center_error_deg"] = 10.0 * np.hypot(
            result[f"{model}_center_x_px"] - reference[f"{model}_center_x_px"],
            result[f"{model}_center_y_px"] - reference[f"{model}_center_y_px"],
        )
        result[f"{model}_major_sigma_ratio"] = (
            result[f"{model}_major_sigma_deg"] / reference[f"{model}_major_sigma_deg"]
        )
    result["threshold_center_error_deg"] = 10.0 * np.hypot(
        result["threshold_center_x_px"] - reference["threshold_center_x_px"],
        result["threshold_center_y_px"] - reference["threshold_center_y_px"],
    )
    result["threshold_area_ratio"] = (
        result["threshold_area_deg2"] / reference["threshold_area_deg2"]
    )
    return result


def render_map_trajectory(matrix: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(14.5, 3.2))
    limit = matrix.max()
    for removed, ax in enumerate(axes):
        cropped, x_coordinates, y_coordinates = crop_map(matrix, "top", removed)
        artist = ax.imshow(
            cropped,
            cmap="viridis",
            vmin=0,
            vmax=limit,
            origin="upper",
            extent=(
                x_coordinates.min() - 0.5,
                x_coordinates.max() + 0.5,
                y_coordinates.max() + 0.5,
                y_coordinates.min() - 0.5,
            ),
        )
        ax.set_title(f"Top rows removed: {removed}")
        ax.set(xticks=range(9), yticks=range(9), xlabel="Original x pixel")
        if removed == 0:
            ax.set_ylabel("Original y pixel")
        fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(f"Unit {UNIT_ID}: artificial top-edge encroachment on the native 9×9 RF map")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_metric_trajectory(table: pd.DataFrame, path: Path) -> None:
    colors = {"top": "#b23a48", "bottom": "#d97736", "left": "#39738c", "right": "#7a6f9b"}
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2))
    for direction in DIRECTIONS:
        selected = table.loc[table["direction"].eq(direction)].sort_values(
            "rows_or_columns_removed"
        )
        axes[0, 0].plot(
            selected["rows_or_columns_removed"], selected["allen_major_sigma_ratio"],
            marker="o", color=colors[direction], label=direction,
        )
        axes[0, 1].plot(
            selected["rows_or_columns_removed"], selected["baseline_major_sigma_ratio"],
            marker="o", color=colors[direction], label=direction,
        )
        axes[1, 0].plot(
            selected["rows_or_columns_removed"], selected["threshold_area_ratio"],
            marker="o", color=colors[direction], label=direction,
        )
        axes[1, 1].plot(
            selected["rows_or_columns_removed"], selected["baseline_center_error_deg"],
            marker="o", color=colors[direction], label=direction,
        )
    axes[0, 0].axhline(1, color="#333333", linestyle="--", linewidth=1)
    axes[0, 1].axhline(1, color="#333333", linestyle="--", linewidth=1)
    axes[1, 0].axhline(1, color="#333333", linestyle="--", linewidth=1)
    axes[0, 0].set(title="Allen no-baseline Gaussian", ylabel="Major σ / full-map estimate")
    axes[0, 1].set(title="Baseline + bounded Gaussian", ylabel="Major σ / full-map estimate")
    axes[1, 0].set(title="Allen threshold component", ylabel="Area / full-map area")
    axes[1, 1].set(title="Baseline-Gaussian center shift", ylabel="Center error from full map (deg)")
    for ax in axes.ravel():
        ax.set_xlabel("Rows or columns removed")
        ax.set_xticks(range(5))
        ax.grid(alpha=0.18)
    axes[0, 0].legend(frameon=False, ncol=2)
    fig.suptitle(f"Unit {UNIT_ID}: RF metric response to artificial edge encroachment")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = np.loadtxt(args.map.resolve(), delimiter=",")
    rows = []
    for direction in DIRECTIONS:
        for removed in range(args.maximum_removed + 1):
            row, _ = analyze_crop(matrix, direction, removed)
            rows.append(row)
    results = add_reference_changes(pd.DataFrame(rows))
    results.to_csv(output_dir / "artificial_crop_trajectory.csv", index=False, float_format="%.8g")
    map_figure = output_dir / "Figure_top_edge_crop_maps.png"
    metric_figure = output_dir / "Figure_crop_metric_trajectory.png"
    render_map_trajectory(matrix, map_figure)
    render_metric_trajectory(results, metric_figure)

    top = results.loc[results["direction"].eq("top")].sort_values("rows_or_columns_removed")
    report = [
        "# Artificial RF-edge cropping: initial concrete checkpoint",
        "",
        f"The native 9×9 spike-count map for stable interior V1 unit {UNIT_ID} was cropped from each edge one row or column at a time. Original visual-grid coordinates were preserved. The uncropped map is the within-unit reference.",
        "",
        "The top-edge trajectory is the most direct encroachment contrast because this RF response occupies rows 3–5. Removing three top rows places the sampled boundary directly against the response; removing four deletes the original peak row.",
        "",
        f"At three top rows removed, Allen's no-baseline major sigma changes from **{top.iloc[0].allen_major_sigma_deg:.2f}°** to **{top.iloc[3].allen_major_sigma_deg:.2f}°**, the baseline-aware major sigma changes from **{top.iloc[0].baseline_major_sigma_deg:.2f}°** to **{top.iloc[3].baseline_major_sigma_deg:.2f}°**, and threshold area changes from **{top.iloc[0].threshold_area_deg2:.0f}** to **{top.iloc[3].threshold_area_deg2:.0f} deg²**.",
        "",
        f"Although Allen's sigma changes only modestly at that three-row crop, its fitted center moves **{top.iloc[3].allen_center_error_deg:.1f}°** from the full-map estimate. The baseline-aware center moves only **{top.iloc[3].baseline_center_error_deg:.2f}°**. Once four top rows are removed and the original peak row is deleted, Allen's major sigma expands **{top.iloc[4].allen_major_sigma_ratio:.2f}×** and its center moves **{top.iloc[4].allen_center_error_deg:.1f}°**; the baseline-aware sigma changes **{top.iloc[4].baseline_major_sigma_ratio:.2f}×**, its center moves **{top.iloc[4].baseline_center_error_deg:.1f}°**, and its bound flag turns on.",
        "",
        "The thresholded area is non-monotonic: it grows by 33–67% as smoothing and thresholding are recomputed on the cropped array, then falls after the peak is removed. A component-touching-edge flag is therefore necessary, but area change cannot be inferred from lost pixels alone.",
        "",
        "Allen's Gaussian helper also swaps its returned row/column center labels. This is hidden on square 9×9 maps but was exposed by rectangular crops; the saved trajectory corrects the labels back to the original visual axes.",
        "",
        "This checkpoint measures estimator response to controlled loss of spatial support. It does not yet compare the proposed DC-ring penalty or establish population-level bias.",
    ]
    (output_dir / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    outputs = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "run_manifest.json"
    }
    manifest = {
        "checkpoint": "single stable interior RF artificial-crop trajectory",
        "unit_id": UNIT_ID,
        "source_map": {"path": str(args.map.resolve()), "sha256": sha256(args.map.resolve())},
        "parameters": {
            "directions": DIRECTIONS,
            "maximum_rows_or_columns_removed": args.maximum_removed,
            "pixel_spacing_deg": 10.0,
            "threshold_mask_sd": 1.0,
        },
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "outputs": outputs,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote artificial-cropping checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
