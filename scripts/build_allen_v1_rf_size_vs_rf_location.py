#!/usr/bin/env python3
"""Build a probe-balanced Allen V1 RF-size surface over measured RF location."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, theilslopes

from scripts.build_allen_v1_rf_size_cortical_surface import (
    cluster_bootstrap,
    grouped_cross_validation,
    sha256,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FITS = (
    ROOT / "artifacts/allen_full_rf_production_v1/03_aggregate/all_session_unit_geometry_fits.csv"
)
DEFAULT_UNITS = ROOT / "data/unit_table.csv"
DEFAULT_RELIABILITY = (
    ROOT / "artifacts/allen_full_rf_production_v1/03_aggregate/allen_rf_split_half_reliability.csv.gz"
)
DEFAULT_OUTPUT = ROOT / "artifacts/v1_rf_size_vs_rf_location_probe_balanced"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", type=Path, default=DEFAULT_FITS)
    parser.add_argument("--units", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--reliability", type=Path, default=DEFAULT_RELIABILITY)
    parser.add_argument("--minimum-split-half-reliability", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bandwidth-deg", type=float, default=12.0)
    parser.add_argument("--support-radius-deg", type=float, default=24.0)
    parser.add_argument("--min-probes", type=int, default=5)
    parser.add_argument("--min-effective-probes", type=float, default=3.0)
    parser.add_argument("--edge-exclusion-deg", type=float, default=10.0)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser.parse_args()


def robust_mad(values: pd.Series) -> float:
    values = values.to_numpy(float)
    center = np.median(values)
    return float(1.4826 * np.median(np.abs(values - center)))


def load_population(
    fits_path: Path,
    units_path: Path,
    edge_exclusion_deg: float,
    reliability_path: Path | None = None,
    minimum_reliability: float | None = None,
):
    fits = pd.read_csv(fits_path, low_memory=False)
    source = fits.loc[
        fits.spatial_model.eq("aperture") & fits.ecephys_structure_acronym.eq("VISp")
    ].drop_duplicates("ecephys_unit_id")
    units = pd.read_csv(
        units_path,
        usecols=["ecephys_unit_id", "ecephys_session_id", "specimen_id", "ecephys_probe_id"],
        low_memory=False,
    )
    data = source.merge(units, on="ecephys_unit_id", how="left", validate="one_to_one")
    if minimum_reliability is not None:
        if reliability_path is None:
            raise ValueError("A reliability table is required when minimum_reliability is set")
        reliability = pd.read_csv(
            reliability_path,
            usecols=["ecephys_unit_id", "rf_split_half_spearman_brown"],
            low_memory=False,
        )
        data = data.merge(reliability, on="ecephys_unit_id", how="left", validate="one_to_one")
    else:
        data["rf_split_half_spearman_brown"] = np.nan
    data["log2_rf_area"] = np.log2(data.axis_area_deg2)
    finite = np.isfinite(
        data[["axis_center_x_deg", "axis_center_y_deg", "log2_rf_area"]]
    ).all(axis=1)
    pre_reliability = (
        finite
        & data.axis_area_deg2.gt(0)
        & ~data.axis_censored.astype(bool)
        & data.axis_edge_distance_deg.gt(edge_exclusion_deg)
    )
    eligible = pre_reliability.copy()
    if minimum_reliability is not None:
        eligible &= data.rf_split_half_spearman_brown.gt(minimum_reliability)
    audit = {
        "source_v1_aperture_units": int(len(source)),
        "finite_rf_center_and_area": int(finite.sum()),
        "parameter_bound_excluded": int((finite & data.axis_censored.astype(bool)).sum()),
        "edge_excluded": int(
            (finite & ~data.axis_censored.astype(bool) & ~data.axis_edge_distance_deg.gt(edge_exclusion_deg)).sum()
        ),
        "eligible_units": int(eligible.sum()),
    }
    if minimum_reliability is not None:
        audit.update(
            {
                "pre_reliability_eligible_units": int(pre_reliability.sum()),
                "reliability_missing": int(
                    (pre_reliability & data.rf_split_half_spearman_brown.isna()).sum()
                ),
                "reliability_at_or_below_threshold": int(
                    (
                        pre_reliability
                        & data.rf_split_half_spearman_brown.notna()
                        & ~data.rf_split_half_spearman_brown.gt(minimum_reliability)
                    ).sum()
                ),
                "minimum_split_half_spearman_brown_exclusive": minimum_reliability,
            }
        )
    return data.loc[eligible].copy(), audit


def collapse_probes(data: pd.DataFrame) -> pd.DataFrame:
    probes = (
        data.groupby(["ecephys_session_id", "specimen_id", "ecephys_probe_id"], observed=True)
        .agg(
            n_units=("ecephys_unit_id", "size"),
            raw_rf_azimuth_deg=("axis_center_x_deg", "median"),
            raw_rf_elevation_deg=("axis_center_y_deg", "median"),
            rf_azimuth_iqr_deg=("axis_center_x_deg", lambda x: x.quantile(0.75) - x.quantile(0.25)),
            rf_elevation_iqr_deg=("axis_center_y_deg", lambda x: x.quantile(0.75) - x.quantile(0.25)),
            median_log2_rf_area=("log2_rf_area", "median"),
            mad_log2_rf_area=("log2_rf_area", robust_mad),
            q25_log2_rf_area=("log2_rf_area", lambda x: x.quantile(0.25)),
            q75_log2_rf_area=("log2_rf_area", lambda x: x.quantile(0.75)),
            minimum_rf_split_half_reliability=("rf_split_half_spearman_brown", "min"),
            median_rf_split_half_reliability=("rf_split_half_spearman_brown", "median"),
        )
        .reset_index()
    )
    probes["rf_azimuth_deg"] = probes.raw_rf_azimuth_deg + 50.0
    probes["rf_elevation_deg"] = probes.raw_rf_elevation_deg + 10.0
    probes["observed_eccentricity_deg"] = np.hypot(
        probes.raw_rf_azimuth_deg, probes.raw_rf_elevation_deg
    )
    # Aliases allow reuse of the rigorously tested grouped-CV implementation.
    probes["ccf_ml_mm"] = probes.rf_azimuth_deg
    probes["ccf_ap_mm"] = probes.rf_elevation_deg
    return probes.sort_values(["ecephys_session_id", "ecephys_probe_id"]).reset_index(drop=True)


def make_grid(probes: pd.DataFrame, step_deg: float = 1.0, margin_deg: float = 8.0):
    az = np.arange(
        np.floor(probes.rf_azimuth_deg.min() - margin_deg),
        np.ceil(probes.rf_azimuth_deg.max() + margin_deg) + step_deg / 2,
        step_deg,
    )
    el = np.arange(
        np.floor(probes.rf_elevation_deg.min() - margin_deg),
        np.ceil(probes.rf_elevation_deg.max() + margin_deg) + step_deg / 2,
        step_deg,
    )
    grid_az, grid_el = np.meshgrid(az, el)
    return az, el, grid_az, grid_el


def kernel_components(probes, grid_az, grid_el, bandwidth_deg, radius_deg):
    points = probes[["rf_azimuth_deg", "rf_elevation_deg"]].to_numpy(float)
    grid = np.column_stack([grid_az.ravel(), grid_el.ravel()])
    distance = np.sqrt(np.sum((grid[:, None, :] - points[None, :, :]) ** 2, axis=2))
    weights = np.exp(-0.5 * (distance / bandwidth_deg) ** 2)
    weights[distance > radius_deg] = 0.0
    total = weights.sum(axis=1)
    effective = total**2 / np.maximum(np.square(weights).sum(axis=1), 1e-12)
    count = (distance <= radius_deg).sum(axis=1)
    return weights, effective, count


def select_cases(cv: pd.DataFrame) -> pd.DataFrame:
    valid = cv.dropna(subset=["cv_prediction_log2_rf_area"])
    candidates = [
        ("smallest RF size", cv.median_log2_rf_area.idxmin(), "minimum probe median"),
        ("largest RF size", cv.median_log2_rf_area.idxmax(), "maximum probe median"),
        ("most central RF", cv.observed_eccentricity_deg.idxmin(), "minimum observed eccentricity"),
        ("most eccentric RF", cv.observed_eccentricity_deg.idxmax(), "maximum observed eccentricity"),
    ]
    if len(valid):
        candidates.extend(
            [
                ("larger than spatial prediction", valid.cv_error.idxmax(), "largest positive grouped-CV residual"),
                ("smaller than spatial prediction", valid.cv_error.idxmin(), "largest negative grouped-CV residual"),
            ]
        )
    rows, used = [], set()
    for role, index, criterion in candidates:
        if index in used:
            continue
        used.add(index)
        row = cv.loc[index].to_dict()
        row.update({"selection_role": role, "selection_criterion": criterion})
        rows.append(row)
    return pd.DataFrame(rows)


def render_main(probes, cv, cases, az, el, surface, effective, count, low, high, mask, metrics, path):
    shape = (len(el), len(az))
    surface_2d = np.where(mask, surface, np.nan).reshape(shape)
    half_width = np.where(mask, (high - low) / 2.0, np.nan).reshape(shape)
    effective_2d = effective.reshape(shape)
    count_2d = count.reshape(shape)
    values = probes.median_log2_rf_area.to_numpy(float)
    vmin, vmax = np.quantile(values, [0.02, 0.98])
    fig, axes = plt.subplots(2, 2, figsize=(14, 11.5), constrained_layout=True)

    ax = axes[0, 0]
    raw = ax.scatter(
        probes.rf_azimuth_deg,
        probes.rf_elevation_deg,
        c=values,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        s=32 + 0.45 * probes.n_units,
        edgecolors="white",
        linewidths=0.8,
    )
    for _, case in cases.iterrows():
        ax.text(case.rf_azimuth_deg, case.rf_elevation_deg, str(int(case.ecephys_session_id))[-4:], fontsize=7)
    fig.colorbar(raw, ax=ax, label="Probe median log₂ RF area (deg²)")
    ax.set_title(f"A  Primary observations: {len(probes)} probe RF centers / {probes.specimen_id.nunique()} animals")

    ax = axes[0, 1]
    image = ax.pcolormesh(az, el, surface_2d, cmap="viridis", vmin=vmin, vmax=vmax, shading="auto")
    levels = [level for level in (5, 10, 20, 30, 40) if level <= np.nanmax(count_2d)]
    if levels:
        contours = ax.contour(az, el, count_2d, levels=levels, colors="white", linewidths=0.7)
        ax.clabel(contours, fontsize=7, fmt="%d probes")
    ax.scatter(probes.rf_azimuth_deg, probes.rf_elevation_deg, s=14, color="black", alpha=0.65)
    fig.colorbar(image, ax=ax, label="Probe-balanced smoothed log₂ RF area (deg²)")
    ax.set_title("B  RF-size surface over measured RF location; contours show support")

    ax = axes[1, 0]
    uncertainty = ax.pcolormesh(az, el, half_width, cmap="magma", shading="auto")
    levels = [level for level in (3, 5, 10, 20, 30) if level <= np.nanmax(effective_2d)]
    if levels:
        contours = ax.contour(az, el, effective_2d, levels=levels, colors="white", linewidths=0.7)
        ax.clabel(contours, fontsize=7, fmt="n_eff %.0f")
    ax.scatter(probes.rf_azimuth_deg, probes.rf_elevation_deg, s=13, color="#22d3ee", edgecolors="black", linewidths=0.25)
    fig.colorbar(uncertainty, ax=ax, label="Animal-bootstrap 95% CI half-width (log₂ units)")
    ax.set_title("C  Animal-clustered bootstrap uncertainty")

    ax = axes[1, 1]
    valid = cv.dropna(subset=["cv_prediction_log2_rf_area"])
    ax.scatter(
        valid.cv_prediction_log2_rf_area,
        valid.median_log2_rf_area,
        c=valid.observed_eccentricity_deg,
        cmap="plasma",
        s=55,
        edgecolors="white",
        linewidths=0.7,
    )
    limits = [
        min(valid.cv_prediction_log2_rf_area.min(), valid.median_log2_rf_area.min()) - 0.08,
        max(valid.cv_prediction_log2_rf_area.max(), valid.median_log2_rf_area.max()) + 0.08,
    ]
    ax.plot(limits, limits, "--", color="#333333", linewidth=1)
    ax.set(xlim=limits, ylim=limits, aspect="equal")
    ax.set_title(
        "D  Repeated animal-grouped 5-fold prediction\n"
        f"MAE {metrics['median_absolute_error']:.3f} vs baseline {metrics['baseline_median_absolute_error']:.3f}; "
        f"{100 * metrics['fraction_better_than_baseline']:.0f}% probes improve"
    )
    ax.set_xlabel("Predicted probe median log₂ RF area")
    ax.set_ylabel("Observed probe median log₂ RF area")

    for ax in axes.flat[:3]:
        ax.set_xlabel("Observed RF azimuth (deg)")
        ax.set_ylabel("Observed RF elevation (deg)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#dddddd", linewidth=0.35, alpha=0.45)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Allen V1 RF size versus measured RF location — probe/animal-balanced exploratory surface\n"
        "RF display convention adds +50° azimuth and +10° elevation; no fitted session translation",
        fontsize=14,
    )
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def render_eccentricity(probes, cv, cases, path):
    x = probes.observed_eccentricity_deg.to_numpy(float)
    y = probes.median_log2_rf_area.to_numpy(float)
    rho, p_value = spearmanr(x, y)
    slope, intercept, low_slope, high_slope = theilslopes(y, x, alpha=0.95)
    line_x = np.linspace(x.min(), x.max(), 100)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)
    ax = axes[0]
    ax.scatter(x, y, c=probes.rf_azimuth_deg, cmap="viridis", s=45 + 0.35 * probes.n_units, edgecolors="white")
    ax.plot(line_x, intercept + slope * line_x, color="#c0392b", linewidth=1.5)
    ax.set(
        title=f"RF size versus observed eccentricity\nSpearman ρ={rho:.2f}, p={p_value:.3g}; Theil–Sen slope={slope:+.4f} [{low_slope:+.4f}, {high_slope:+.4f}]",
        xlabel="Observed RF eccentricity from screen center (deg)",
        ylabel="Probe median log₂ RF area (deg²)",
    )
    ax.grid(color="#dddddd", linewidth=0.4)

    ax = axes[1]
    valid = cv.dropna(subset=["cv_error"])
    limit = float(np.nanmax(np.abs(valid.cv_error)))
    scatter = ax.scatter(
        valid.rf_azimuth_deg,
        valid.rf_elevation_deg,
        c=valid.cv_error,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        s=45 + 0.35 * valid.n_units,
        edgecolors="white",
        linewidths=0.7,
    )
    for _, case in cases.iterrows():
        if np.isfinite(case.get("cv_error", np.nan)):
            ax.scatter(case.rf_azimuth_deg, case.rf_elevation_deg, facecolors="none", edgecolors="black", s=170, linewidths=1.2)
            ax.text(case.rf_azimuth_deg, case.rf_elevation_deg, f" {case.selection_role}\n {int(case.ecephys_session_id)}", fontsize=7)
    fig.colorbar(scatter, ax=ax, label="Observed − grouped-CV prediction (log₂ RF area)")
    ax.set(
        title="Selected cases and all spatial prediction residuals",
        xlabel="Observed RF azimuth (deg)",
        ylabel="Observed RF elevation (deg)",
        aspect="equal",
    )
    ax.grid(color="#dddddd", linewidth=0.4)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return {
        "spearman_rho": float(rho),
        "spearman_p": float(p_value),
        "theil_sen_slope_log2_per_deg": float(slope),
        "theil_sen_slope_ci_low": float(low_slope),
        "theil_sen_slope_ci_high": float(high_slope),
    }


def main() -> None:
    args = parse_args()
    fits_path, units_path = args.fits.resolve(), args.units.resolve()
    reliability_path = args.reliability.resolve() if args.minimum_split_half_reliability is not None else None
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    population, audit = load_population(
        fits_path,
        units_path,
        args.edge_exclusion_deg,
        reliability_path,
        args.minimum_split_half_reliability,
    )
    probes = collapse_probes(population)
    probes.to_csv(output / "probe_rf_location_and_size.csv", index=False, float_format="%.9g")

    az, el, grid_az, grid_el = make_grid(probes)
    weights, effective, count = kernel_components(
        probes, grid_az, grid_el, args.bandwidth_deg, args.support_radius_deg
    )
    values = probes.median_log2_rf_area.to_numpy(float)
    surface = weights @ values / np.maximum(weights.sum(axis=1), 1e-12)
    mask = (count >= args.min_probes) & (effective >= args.min_effective_probes)
    low, bootstrap_median, high = cluster_bootstrap(
        probes, weights, args.bootstrap_repeats, args.seed
    )

    cv_draws, cv, cv_metrics = grouped_cross_validation(
        probes,
        args.bandwidth_deg,
        args.support_radius_deg,
        args.min_probes,
        args.min_effective_probes,
        repeats=100,
        folds=5,
        seed=args.seed + 1,
    )
    cv_draws.to_csv(output / "grouped_cross_validation_draws.csv.gz", index=False, compression="gzip", float_format="%.9g")
    cv.to_csv(output / "grouped_cross_validation_probe_summary.csv", index=False, float_format="%.9g")
    cases = select_cases(cv)
    cases.to_csv(output / "case_selection.csv", index=False, float_format="%.9g")

    sensitivity_rows = []
    for index, bandwidth in enumerate((8.0, 12.0, 16.0, 20.0)):
        radius = max(20.0, 2.0 * bandwidth)
        _, _, metrics = grouped_cross_validation(
            probes,
            bandwidth,
            radius,
            args.min_probes,
            args.min_effective_probes,
            repeats=25,
            folds=5,
            seed=args.seed + 100 + index,
        )
        sensitivity_rows.append({"bandwidth_deg": bandwidth, "support_radius_deg": radius, **metrics})
    pd.DataFrame(sensitivity_rows).to_csv(output / "bandwidth_sensitivity.csv", index=False, float_format="%.9g")

    grid = pd.DataFrame(
        {
            "rf_azimuth_deg": grid_az.ravel(),
            "rf_elevation_deg": grid_el.ravel(),
            "smoothed_log2_rf_area": np.where(mask, surface, np.nan),
            "bootstrap_median_log2_rf_area": np.where(mask, bootstrap_median, np.nan),
            "bootstrap_ci_low": np.where(mask, low, np.nan),
            "bootstrap_ci_high": np.where(mask, high, np.nan),
            "support_probe_count": count,
            "effective_probe_count": effective,
            "supported": mask,
        }
    )
    grid.to_csv(output / "surface_grid.csv.gz", index=False, compression="gzip", float_format="%.9g")

    main_figure = output / "Figure_v1_rf_size_vs_rf_location_probe_balanced.png"
    detail_figure = output / "Figure_v1_rf_size_vs_eccentricity_and_cases.png"
    render_main(
        probes, cv, cases, az, el, surface, effective, count, low, high, mask, cv_metrics, main_figure
    )
    eccentricity = render_eccentricity(probes, cv, cases, detail_figure)

    manifest = {
        "status": "exploratory probe-balanced RF-size surface over measured RF location",
        "inputs": {
            "fits": {"path": str(fits_path), "sha256": sha256(fits_path)},
            "units": {"path": str(units_path), "sha256": sha256(units_path)},
        },
        "audit": audit,
        "n_probes": int(len(probes)),
        "n_animals": int(probes.specimen_id.nunique()),
        "metric": "probe median log2 axis-aligned analytic-aperture RF half-maximum ellipse area (deg2)",
        "rf_location": "probe median measured aperture-fit RF center; +50 deg azimuth and +10 deg elevation display offsets only; no fitted per-session registration",
        "surface": {
            "bandwidth_deg": args.bandwidth_deg,
            "support_radius_deg": args.support_radius_deg,
            "min_probes": args.min_probes,
            "min_effective_probes": args.min_effective_probes,
            "supported_grid_fraction": float(mask.mean()),
        },
        "bootstrap": {"cluster": "specimen_id", "repeats": args.bootstrap_repeats, "seed": args.seed},
        "grouped_cross_validation": cv_metrics,
        "eccentricity": eccentricity,
        "limitations": [
            "RF center and RF size are estimated from the same fitted response surface, so their measurement errors may be coupled.",
            "Most animals contribute one V1 probe; animal-wide RF-size differences remain confounded with sampled RF location.",
            "The common +50/+10 display offsets do not correct unvalidated per-session gaze or visual-field translations.",
            "The smooth surface is descriptive and masked outside local multi-probe support.",
        ],
    }
    manifest["minimum_split_half_spearman_brown_exclusive"] = args.minimum_split_half_reliability
    if reliability_path is not None:
        manifest["inputs"]["reliability"] = {
            "path": str(reliability_path),
            "sha256": sha256(reliability_path),
        }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    lines = [
        "# Probe-balanced Allen V1 RF size versus measured RF location",
        "",
        "## Cohort and construction",
        "",
        f"- {audit['eligible_units']} eligible units collapsed to {len(probes)} V1 probe estimates from {probes.specimen_id.nunique()} animals.",
        f"- Parameter-bound fits and RF centers within {args.edge_exclusion_deg:g} degrees of a sampled stimulus edge were excluded.",
        "- RF center is the per-probe median measured center. The +50 degree azimuth and +10 degree elevation constants are display offsets only; no per-session registration is applied.",
        "- RF size is the per-probe median log2 analytic-aperture half-maximum ellipse area.",
        "",
        "## Held-out validation",
        "",
        f"Repeated animal-grouped five-fold predictions covered {cv_metrics['n_probe_predictions']}/{len(probes)} probes. Median absolute error was {cv_metrics['median_absolute_error']:.3f} versus {cv_metrics['baseline_median_absolute_error']:.3f} for the training-cohort median. Median paired improvement was {cv_metrics['median_absolute_error_improvement']:+.3f}; {100 * cv_metrics['fraction_better_than_baseline']:.1f}% of probes and {100 * cv_metrics['fraction_folds_with_positive_improvement']:.1f}% of folds improved (one-sided paired Wilcoxon p={cv_metrics['paired_wilcoxon_baseline_error_greater_p']:.3g}).",
        "",
        "## Eccentricity",
        "",
        f"Probe RF size versus observed eccentricity: Spearman rho={eccentricity['spearman_rho']:.3f}, p={eccentricity['spearman_p']:.3g}; Theil-Sen slope={eccentricity['theil_sen_slope_log2_per_deg']:+.4f} log2 units/degree (95% CI {eccentricity['theil_sen_slope_ci_low']:+.4f} to {eccentricity['theil_sen_slope_ci_high']:+.4f}).",
        "",
        "## Interpretation limits",
        "",
        "- RF center and size come from the same unit-level fit and therefore are not independent measurements.",
        "- Animal effects and sampled RF location are difficult to separate when most animals contribute one probe.",
        "- Cross-session visual translations are not applied because the prior registration attempts were not validated.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python -m scripts.build_allen_v1_rf_size_vs_rf_location",
        "```",
    ]
    if args.minimum_split_half_reliability is not None:
        lines.insert(
            lines.index("- RF center is the per-probe median measured center. The +50 degree azimuth and +10 degree elevation constants are display offsets only; no per-session registration is applied."),
            f"- Required unit RF split-half Spearman-Brown reliability > {args.minimum_split_half_reliability:g} (strict inequality).",
        )
        lines[-2] += (
            f" --minimum-split-half-reliability {args.minimum_split_half_reliability:g}"
            f" --output-dir {output}"
        )
    (output / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"audit": audit, "n_probes": len(probes), "n_animals": probes.specimen_id.nunique(), "grouped_cv": cv_metrics, "eccentricity": eccentricity}, indent=2))
    print(main_figure)


if __name__ == "__main__":
    main()
