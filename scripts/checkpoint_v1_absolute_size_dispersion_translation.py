#!/usr/bin/env python3
"""Concrete checkpoint for V1 translation from absolute RF size and RF-center dispersion.

This is deliberately an exploratory, inspectable checkpoint rather than a
production registration.  It keeps RF centers near and beyond the stimulus-grid
edge, preserves absolute RF area between animals, and uses a fixed
leave-one-animal-out template when comparing independent target-cell halves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.interpolate import RegularGridInterpolator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "artifacts"
    / "allen_full_rf_production_v1"
    / "03_aggregate"
    / "all_session_unit_geometry_fits.csv"
)
DEFAULT_UNITS = ROOT / "data" / "unit_table.csv"
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "v1_absolute_size_dispersion_translation_checkpoint"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--unit-table", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--surface-bandwidth-deg", type=float, default=12.0)
    parser.add_argument("--dispersion-bandwidth-deg", type=float, default=15.0)
    parser.add_argument("--translation-bound-deg", type=float, default=30.0)
    parser.add_argument("--translation-step-deg", type=float, default=2.0)
    parser.add_argument(
        "--exclude-parameter-bound-size",
        action="store_true",
        help="Set RF area to missing only for parameter-bound fits; retain their RF centers for dispersion.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def robust_scale(values: np.ndarray, floor: float) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return floor
    center = np.median(values)
    return max(float(1.4826 * np.median(np.abs(values - center))), floor)


def huber_mean(residual: np.ndarray) -> float:
    residual = np.asarray(residual, float)
    absolute = np.abs(residual)
    loss = np.where(absolute <= 1.0, 0.5 * residual**2, absolute - 0.5)
    return float(np.mean(loss))


def prepare_population(fit_path: Path, unit_path: Path) -> pd.DataFrame:
    fits = pd.read_csv(fit_path, low_memory=False)
    fits = fits.loc[
        fits["spatial_model"].eq("aperture")
        & fits["ecephys_structure_acronym"].eq("VISp")
    ].drop_duplicates("ecephys_unit_id")
    units = pd.read_csv(
        unit_path,
        usecols=[
            "ecephys_unit_id",
            "ecephys_session_id",
            "specimen_id",
            "session_type",
            "ecephys_probe_id",
        ],
        low_memory=False,
    )
    population = fits.merge(units, on="ecephys_unit_id", how="left", suffixes=("", "_released"))
    population["ecephys_session_id"] = population["session_id"].astype(int)
    population["rf_azimuth_deg"] = population["axis_center_x_deg"]
    population["rf_elevation_deg"] = population["axis_center_y_deg"]
    population["log2_rf_area"] = np.log2(population["axis_area_deg2"])
    required = ["rf_azimuth_deg", "rf_elevation_deg", "log2_rf_area"]
    population = population.dropna(subset=required).copy()
    population = population.loc[
        np.isfinite(population[required]).all(axis=1)
        & population["axis_area_deg2"].gt(0)
    ].copy()
    return population.reset_index(drop=True)


def local_dispersion(points: np.ndarray, bandwidth: float) -> tuple[np.ndarray, np.ndarray]:
    """Translation-invariant local covariance descriptors at each RF center."""
    points = np.asarray(points, float)
    delta = points[None, :, :] - points[:, None, :]
    distance2 = np.sum(delta**2, axis=2)
    weights = np.exp(-0.5 * distance2 / bandwidth**2)
    np.fill_diagonal(weights, 0.0)
    weight_sum = weights.sum(axis=1)
    effective = weight_sum**2 / np.maximum((weights**2).sum(axis=1), 1e-12)
    mean = weights @ points / np.maximum(weight_sum[:, None], 1e-12)
    centered = points[None, :, :] - mean[:, None, :]
    cxx = np.sum(weights * centered[:, :, 0] ** 2, axis=1) / np.maximum(weight_sum, 1e-12)
    cyy = np.sum(weights * centered[:, :, 1] ** 2, axis=1) / np.maximum(weight_sum, 1e-12)
    cxy = np.sum(weights * centered[:, :, 0] * centered[:, :, 1], axis=1) / np.maximum(weight_sum, 1e-12)
    trace = np.maximum(cxx + cyy, 1e-6)
    features = np.column_stack(
        [
            np.log2(trace),
            (cxx - cyy) / trace,
            2.0 * cxy / trace,
        ]
    )
    features[effective < 3.0] = np.nan
    return features, effective


def assign_descriptors(table: pd.DataFrame, bandwidth: float) -> pd.DataFrame:
    frames = []
    for session_id, local in table.groupby("ecephys_session_id", observed=True):
        local = local.copy()
        features, effective = local_dispersion(
            local[["rf_azimuth_deg", "rf_elevation_deg"]].to_numpy(float), bandwidth
        )
        local["dispersion_log2_trace"] = features[:, 0]
        local["dispersion_anisotropy_x"] = features[:, 1]
        local["dispersion_anisotropy_xy"] = features[:, 2]
        local["dispersion_effective_neighbors"] = effective
        frames.append(local)
    return pd.concat(frames, ignore_index=True)


FEATURES = (
    "log2_rf_area",
    "dispersion_log2_trace",
    "dispersion_anisotropy_x",
    "dispersion_anisotropy_xy",
)


def smooth_surface(
    table: pd.DataFrame,
    grid_points: np.ndarray,
    bandwidth: float,
) -> tuple[np.ndarray, np.ndarray]:
    points = table[["rf_azimuth_deg", "rf_elevation_deg"]].to_numpy(float)
    delta = grid_points[:, None, :] - points[None, :, :]
    weights = np.exp(-0.5 * np.sum(delta**2, axis=2) / bandwidth**2)
    values = np.full((len(grid_points), len(FEATURES)), np.nan)
    effective_min = np.full(len(grid_points), np.nan)
    effective_by_feature = []
    for feature_index, feature in enumerate(FEATURES):
        observed = table[feature].to_numpy(float)
        finite = np.isfinite(observed)
        local_weights = weights[:, finite]
        numerator = local_weights @ observed[finite]
        denominator = local_weights.sum(axis=1)
        effective = denominator**2 / np.maximum((local_weights**2).sum(axis=1), 1e-12)
        supported = effective >= 3.0
        values[supported, feature_index] = numerator[supported] / denominator[supported]
        effective_by_feature.append(effective)
    effective_min = np.min(np.column_stack(effective_by_feature), axis=1)
    return values, effective_min


def build_full_session_surfaces(
    population: pd.DataFrame,
    grid_points: np.ndarray,
    bandwidth: float,
) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    surfaces = {}
    rows = []
    for session_id, local in population.groupby("ecephys_session_id", observed=True):
        values, effective = smooth_surface(local, grid_points, bandwidth)
        surfaces[int(session_id)] = values
        rows.append(
            {
                "ecephys_session_id": int(session_id),
                "v1_units": len(local),
                "censored_fraction": float(local["axis_censored"].mean()),
                "median_absolute_rf_area_deg2": float(local["axis_area_deg2"].median()),
                "median_rf_azimuth_deg": float(local["rf_azimuth_deg"].median()),
                "median_rf_elevation_deg": float(local["rf_elevation_deg"].median()),
                "median_dispersion_effective_neighbors": float(
                    local["dispersion_effective_neighbors"].median()
                ),
                "supported_grid_points": int(np.sum(effective >= 3.0)),
            }
        )
    return surfaces, pd.DataFrame(rows)


def leave_one_out_template(
    surfaces: dict[int, np.ndarray], held_session: int, minimum_sessions: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    stack = np.stack([value for key, value in surfaces.items() if key != held_session])
    support = np.sum(np.isfinite(stack), axis=0)
    template = np.divide(
        np.nansum(stack, axis=0),
        support,
        out=np.full(support.shape, np.nan, dtype=float),
        where=support > 0,
    )
    template[support < minimum_sessions] = np.nan
    return template, support


def make_interpolators(
    template: np.ndarray, axis: np.ndarray
) -> list[RegularGridInterpolator]:
    shaped = template.reshape(len(axis), len(axis), len(FEATURES))
    return [
        RegularGridInterpolator(
            (axis, axis), shaped[:, :, index], bounds_error=False, fill_value=np.nan
        )
        for index in range(len(FEATURES))
    ]


def evaluate_shift_grid(
    table: pd.DataFrame,
    interpolators: list[RegularGridInterpolator],
    shift_values: np.ndarray,
    scales: np.ndarray,
) -> pd.DataFrame:
    points = table[["rf_azimuth_deg", "rf_elevation_deg"]].to_numpy(float)
    observed = table[list(FEATURES)].to_numpy(float)
    rows = []
    for shift_el in shift_values:
        for shift_az in shift_values:
            shifted = points + np.array([shift_az, shift_el])
            query = shifted[:, [1, 0]]
            predicted = np.column_stack([interpolator(query) for interpolator in interpolators])
            valid_size = np.isfinite(observed[:, 0]) & np.isfinite(predicted[:, 0])
            valid_dispersion = np.isfinite(observed[:, 1:]).all(axis=1) & np.isfinite(
                predicted[:, 1:]
            ).all(axis=1)
            size_coverage = float(valid_size.mean())
            dispersion_coverage = float(valid_dispersion.mean())
            size_loss = np.nan
            dispersion_loss = np.nan
            if valid_size.sum() >= 10:
                size_loss = huber_mean(
                    (observed[valid_size, 0] - predicted[valid_size, 0]) / scales[0]
                ) + 0.75 * (1.0 - size_coverage)
            if valid_dispersion.sum() >= 10:
                normalized = (
                    observed[valid_dispersion, 1:] - predicted[valid_dispersion, 1:]
                ) / scales[1:]
                dispersion_loss = float(
                    np.mean([huber_mean(normalized[:, index]) for index in range(3)])
                ) + 0.75 * (1.0 - dispersion_coverage)
            combined_loss = (
                0.5 * size_loss + 0.5 * dispersion_loss
                if np.isfinite(size_loss) and np.isfinite(dispersion_loss)
                else np.nan
            )
            rows.append(
                {
                    "shift_azimuth_deg": shift_az,
                    "shift_elevation_deg": shift_el,
                    "size_loss": size_loss,
                    "dispersion_loss": dispersion_loss,
                    "combined_loss": combined_loss,
                    "size_coverage": size_coverage,
                    "dispersion_coverage": dispersion_coverage,
                }
            )
    return pd.DataFrame(rows)


def optimum(surface: pd.DataFrame, loss_column: str) -> dict[str, float]:
    finite = surface.dropna(subset=[loss_column])
    if finite.empty:
        return {
            "shift_azimuth_deg": np.nan,
            "shift_elevation_deg": np.nan,
            "minimum_loss": np.nan,
            "basin_grid_points_delta_005": np.nan,
            "at_bound": True,
        }
    row = finite.loc[finite[loss_column].idxmin()]
    minimum = float(row[loss_column])
    bound = max(
        float(surface["shift_azimuth_deg"].abs().max()),
        float(surface["shift_elevation_deg"].abs().max()),
    )
    return {
        "shift_azimuth_deg": float(row["shift_azimuth_deg"]),
        "shift_elevation_deg": float(row["shift_elevation_deg"]),
        "minimum_loss": minimum,
        "basin_grid_points_delta_005": int((finite[loss_column] <= minimum + 0.05).sum()),
        "at_bound": bool(
            abs(row["shift_azimuth_deg"]) >= bound
            or abs(row["shift_elevation_deg"]) >= bound
        ),
    }


def deterministic_split(table: pd.DataFrame, session_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260816 + int(session_id))
    order = rng.permutation(len(table))
    halves = []
    for half in (0, 1):
        selected = table.iloc[order[half::2]].copy()
        halves.append(assign_descriptors(selected, bandwidth=15.0))
    return halves[0], halves[1]


def fit_all_sessions(
    population: pd.DataFrame,
    surfaces: dict[int, np.ndarray],
    axis: np.ndarray,
    shift_values: np.ndarray,
    scales: np.ndarray,
) -> tuple[pd.DataFrame, dict[tuple[int, str], pd.DataFrame]]:
    rows = []
    landscapes = {}
    for session_id, local in population.groupby("ecephys_session_id", observed=True):
        session_id = int(session_id)
        template, _ = leave_one_out_template(surfaces, session_id)
        interpolators = make_interpolators(template, axis)
        half_zero, half_one = deterministic_split(local, session_id)
        variants = {"full": local, "half_0": half_zero, "half_1": half_one}
        local_optima = {}
        for variant, selected in variants.items():
            landscape = evaluate_shift_grid(selected, interpolators, shift_values, scales)
            landscape["ecephys_session_id"] = session_id
            landscape["target_subset"] = variant
            landscapes[(session_id, variant)] = landscape
            for mode in ("size", "dispersion", "combined"):
                found = optimum(landscape, f"{mode}_loss")
                local_optima[(variant, mode)] = found
                rows.append(
                    {
                        "ecephys_session_id": session_id,
                        "target_subset": variant,
                        "mode": mode,
                        **found,
                    }
                )
        for mode in ("size", "dispersion", "combined"):
            first = local_optima[("half_0", mode)]
            second = local_optima[("half_1", mode)]
            distance = np.hypot(
                first["shift_azimuth_deg"] - second["shift_azimuth_deg"],
                first["shift_elevation_deg"] - second["shift_elevation_deg"],
            )
            for row in rows[-9:]:
                if row["mode"] == mode:
                    row["split_half_vector_difference_deg"] = float(distance)
    return pd.DataFrame(rows), landscapes


def select_cases(optima: pd.DataFrame, support: pd.DataFrame) -> pd.DataFrame:
    combined = optima.loc[
        optima["target_subset"].eq("full") & optima["mode"].eq("combined")
    ].merge(support, on="ecephys_session_id", how="left")
    eligible = combined.loc[
        combined["v1_units"].ge(60)
        & ~combined["at_bound"]
        & np.isfinite(combined["split_half_vector_difference_deg"])
    ].sort_values("split_half_vector_difference_deg")
    rows = []
    if len(eligible):
        success = eligible.iloc[0]
        median = eligible.iloc[(len(eligible) - 1) // 2]
        for role, row, criterion in (
            (
                "most reproducible non-bound case",
                success,
                "minimum combined split-half vector difference among sessions with >=60 V1 units",
            ),
            (
                "median reproducibility non-bound case",
                median,
                "middle combined split-half vector difference among sessions with >=60 V1 units",
            ),
        ):
            rows.append({"selection_role": role, "criterion": criterion, **row.to_dict()})
    failures = combined.loc[np.isfinite(combined["split_half_vector_difference_deg"])].sort_values(
        ["at_bound", "split_half_vector_difference_deg"], ascending=[False, False]
    )
    if len(failures):
        row = failures.iloc[0]
        rows.append(
            {
                "selection_role": "failure-prone case",
                "criterion": "boundary optimum prioritized, then maximum combined split-half vector difference",
                **row.to_dict(),
            }
        )
    return pd.DataFrame(rows).drop_duplicates("ecephys_session_id")


def plot_loss(
    axis_object: plt.Axes,
    landscape: pd.DataFrame,
    loss_column: str,
    title: str,
    optima: pd.DataFrame,
) -> None:
    x = np.sort(landscape["shift_azimuth_deg"].unique())
    y = np.sort(landscape["shift_elevation_deg"].unique())
    z = landscape.pivot(
        index="shift_elevation_deg", columns="shift_azimuth_deg", values=loss_column
    ).reindex(index=y, columns=x).to_numpy(float)
    finite = z[np.isfinite(z)]
    relative = z - np.nanmin(z)
    upper = max(float(np.nanquantile(relative[np.isfinite(relative)], 0.8)), 0.05)
    artist = axis_object.imshow(
        relative,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap="magma_r",
        norm=Normalize(0, upper),
        aspect="equal",
    )
    colors = {"full": "white", "half_0": "#27c2ff", "half_1": "#55e06f"}
    markers = {"full": "*", "half_0": "o", "half_1": "s"}
    for row in optima.itertuples(index=False):
        axis_object.scatter(
            row.shift_azimuth_deg,
            row.shift_elevation_deg,
            marker=markers[row.target_subset],
            s=85 if row.target_subset == "full" else 42,
            facecolor=colors[row.target_subset],
            edgecolor="black",
            linewidth=0.7,
            zorder=5,
            label=row.target_subset.replace("_", " "),
        )
    axis_object.axhline(0, color="#999999", linewidth=0.6)
    axis_object.axvline(0, color="#999999", linewidth=0.6)
    axis_object.set(
        xlabel="Azimuth correction (deg)",
        ylabel="Elevation correction (deg)",
        title=title,
    )
    axis_object.grid(alpha=0.12)
    if len(finite):
        plt.colorbar(artist, ax=axis_object, fraction=0.046, pad=0.03, label="loss above optimum")


def render_cases(
    population: pd.DataFrame,
    optima: pd.DataFrame,
    landscapes: dict[tuple[int, str], pd.DataFrame],
    selected: pd.DataFrame,
    output: Path,
) -> None:
    rows = len(selected)
    figure, axes = plt.subplots(rows, 5, figsize=(22, 5.1 * rows), squeeze=False)
    area_limits = np.nanquantile(population["log2_rf_area"], [0.02, 0.98])
    dispersion_limits = np.nanquantile(population["dispersion_log2_trace"], [0.02, 0.98])
    for row_index, selected_row in enumerate(selected.itertuples(index=False)):
        session_id = int(selected_row.ecephys_session_id)
        local = population.loc[population["ecephys_session_id"].eq(session_id)]
        area_axis = axes[row_index, 0]
        scatter = area_axis.scatter(
            local["rf_azimuth_deg"],
            local["rf_elevation_deg"],
            c=local["log2_rf_area"],
            cmap="viridis",
            norm=Normalize(*area_limits),
            s=25,
            alpha=0.82,
        )
        censored = local.loc[local["axis_censored"]]
        area_axis.scatter(
            censored["rf_azimuth_deg"],
            censored["rf_elevation_deg"],
            facecolors="none",
            edgecolors="#ef476f",
            s=52,
            linewidth=0.9,
            label="parameter-bound fit",
        )
        area_axis.set(
            xlabel="Observed RF azimuth (deg)",
            ylabel="Observed RF elevation (deg)",
            title=(
                f"{selected_row.selection_role}\nsession {session_id}: absolute RF size"
                f"\nN={len(local)}, bound={local.axis_censored.mean():.0%}"
            ),
            aspect="equal",
        )
        figure.colorbar(scatter, ax=area_axis, fraction=0.046, pad=0.03, label="log₂ RF area (deg²)")
        if len(censored):
            area_axis.legend(loc="lower left", fontsize=7)

        dispersion_axis = axes[row_index, 1]
        scatter = dispersion_axis.scatter(
            local["rf_azimuth_deg"],
            local["rf_elevation_deg"],
            c=local["dispersion_log2_trace"],
            cmap="cividis",
            norm=Normalize(*dispersion_limits),
            s=25,
            alpha=0.85,
        )
        dispersion_axis.set(
            xlabel="Observed RF azimuth (deg)",
            ylabel="Observed RF elevation (deg)",
            title="Local RF-center covariance trace",
            aspect="equal",
        )
        figure.colorbar(
            scatter, ax=dispersion_axis, fraction=0.046, pad=0.03, label="log₂ covariance trace"
        )

        full_landscape = landscapes[(session_id, "full")]
        for column, mode in enumerate(("size", "dispersion", "combined"), start=2):
            local_optima = optima.loc[
                optima["ecephys_session_id"].eq(session_id) & optima["mode"].eq(mode)
            ]
            plot_loss(
                axes[row_index, column],
                full_landscape,
                f"{mode}_loss",
                f"{mode.capitalize()} translation objective",
                local_optima,
            )
        for axis_object in axes[row_index, :2]:
            axis_object.grid(alpha=0.15)
    handles, labels = axes[0, 4].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles[:3], labels[:3], loc="upper center", bbox_to_anchor=(0.5, 0.958),
            ncol=3, frameon=False,
        )
    figure.suptitle(
        "V1 translation checkpoint: absolute improved RF size and local RF-center dispersion\n"
        "No SF/TF; no screen-edge exclusion; fixed leave-one-animal-out template for both target halves",
        fontsize=16,
        y=0.997,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.925])
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output}; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)

    population = prepare_population(args.input.resolve(), args.unit_table.resolve())
    population = assign_descriptors(population, args.dispersion_bandwidth_deg)
    population["log2_rf_area_all_fits"] = population["log2_rf_area"]
    if args.exclude_parameter_bound_size:
        population.loc[population["axis_censored"], "log2_rf_area"] = np.nan
    axis = np.arange(-90.0, 90.0 + args.translation_step_deg, args.translation_step_deg)
    x_mesh, y_mesh = np.meshgrid(axis, axis)
    grid_points = np.column_stack([x_mesh.ravel(), y_mesh.ravel()])
    surfaces, support = build_full_session_surfaces(
        population, grid_points, args.surface_bandwidth_deg
    )
    scales = np.array(
        [
            robust_scale(population[feature].to_numpy(float), 0.10 if index == 0 else 0.05)
            for index, feature in enumerate(FEATURES)
        ]
    )
    shift_values = np.arange(
        -args.translation_bound_deg,
        args.translation_bound_deg + args.translation_step_deg,
        args.translation_step_deg,
    )
    optima, landscapes = fit_all_sessions(
        population, surfaces, axis, shift_values, scales
    )
    selected = select_cases(optima, support)

    population.to_csv(output / "v1_unit_descriptors.csv.gz", index=False, compression="gzip")
    support.to_csv(output / "session_support_summary.csv", index=False)
    optima.to_csv(output / "translation_optima_all_sessions.csv", index=False)
    selected.to_csv(output / "selected_case_audit.csv", index=False)
    selected_landscapes = pd.concat(
        [
            landscape
            for (session_id, _), landscape in landscapes.items()
            if session_id in set(selected["ecephys_session_id"].astype(int))
        ],
        ignore_index=True,
    )
    selected_landscapes.to_csv(output / "selected_case_objective_landscapes.csv.gz", index=False, compression="gzip")
    figure_path = output / "Figure_v1_absolute_size_dispersion_translation_cases.png"
    render_cases(population, optima, landscapes, selected, figure_path)

    manifest = {
        "status": "exploratory concrete-case checkpoint",
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input.resolve()),
        "unit_table": str(args.unit_table.resolve()),
        "sessions": int(population["ecephys_session_id"].nunique()),
        "v1_units": int(len(population)),
        "methods": {
            "rf_size": "absolute log2 axis-aligned analytic-aperture RF area; no animal normalization",
            "edge_rule": "no RF-center edge exclusion",
            "parameter_bound_rule": (
                "all RF centers retained; parameter-bound area values omitted only from RF-size surfaces"
                if args.exclude_parameter_bound_size
                else "all fits retained and visibly flagged; no primary censor exclusion"
            ),
            "dispersion": "15-degree Gaussian local covariance trace and normalized anisotropy components",
            "template": "leave-one-animal-out equal-session mean surfaces",
            "split_half": "target cells split independently; same full external template used for both halves",
            "translation": f"grid search +/-{args.translation_bound_deg:g} deg in {args.translation_step_deg:g}-deg steps",
            "sf_tf": "not used",
        },
        "feature_scales": dict(zip(FEATURES, scales.tolist())),
        "selected_sessions": selected[
            ["ecephys_session_id", "selection_role", "criterion"]
        ].to_dict(orient="records"),
        "outputs": [
            "Figure_v1_absolute_size_dispersion_translation_cases.png",
            "selected_case_audit.csv",
            "selected_case_objective_landscapes.csv.gz",
            "translation_optima_all_sessions.csv",
            "session_support_summary.csv",
            "v1_unit_descriptors.csv.gz",
        ],
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
