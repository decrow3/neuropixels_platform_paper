#!/usr/bin/env python3
"""Build a probe-balanced Allen V1 RF-size map in cortical CCF coordinates.

The spatial replicate is a probe track, not a unit.  Eligible V1 units are
collapsed to one robust RF-size estimate and one AP/ML cortical-column position
per probe.  The displayed surface is therefore a smooth summary of independent
probe/animal observations rather than a unit-weighted image.
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
from scipy.stats import spearmanr, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FITS = (
    ROOT / "artifacts/allen_full_rf_production_v1/03_aggregate/all_session_unit_geometry_fits.csv"
)
DEFAULT_UNITS = ROOT / "data/unit_table.csv"
DEFAULT_RELIABILITY = (
    ROOT / "artifacts/allen_full_rf_production_v1/03_aggregate/allen_rf_split_half_reliability.csv.gz"
)
DEFAULT_OUTPUT = ROOT / "artifacts/v1_rf_size_cortical_surface_probe_balanced"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", type=Path, default=DEFAULT_FITS)
    parser.add_argument("--units", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--reliability", type=Path, default=DEFAULT_RELIABILITY)
    parser.add_argument("--minimum-split-half-reliability", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bandwidth-mm", type=float, default=0.35)
    parser.add_argument("--support-radius-mm", type=float, default=0.60)
    parser.add_argument("--min-probes", type=int, default=5)
    parser.add_argument("--min-effective-probes", type=float, default=3.0)
    parser.add_argument("--edge-exclusion-deg", type=float, default=10.0)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
) -> tuple[pd.DataFrame, dict]:
    fits = pd.read_csv(fits_path, low_memory=False)
    source_v1 = fits.loc[
        fits.spatial_model.eq("aperture") & fits.ecephys_structure_acronym.eq("VISp")
    ].drop_duplicates("ecephys_unit_id")
    units = pd.read_csv(
        units_path,
        usecols=[
            "ecephys_unit_id",
            "ecephys_session_id",
            "specimen_id",
            "ecephys_probe_id",
            "anterior_posterior_ccf_coordinate",
            "left_right_ccf_coordinate",
            "dorsal_ventral_ccf_coordinate",
        ],
        low_memory=False,
    )
    data = source_v1.merge(units, on="ecephys_unit_id", how="left", validate="one_to_one")
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
        data[
            [
                "log2_rf_area",
                "anterior_posterior_ccf_coordinate",
                "left_right_ccf_coordinate",
            ]
        ]
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
        "source_v1_aperture_units": int(len(source_v1)),
        "finite_rf_and_ap_ml_ccf": int(finite.sum()),
        "parameter_bound_excluded": int((finite & data.axis_censored.astype(bool)).sum()),
        "edge_excluded": int((finite & ~data.axis_censored.astype(bool) & ~data.axis_edge_distance_deg.gt(edge_exclusion_deg)).sum()),
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
    grouped = data.groupby(
        ["ecephys_session_id", "specimen_id", "ecephys_probe_id"], observed=True
    )
    probes = grouped.agg(
        n_units=("ecephys_unit_id", "size"),
        ccf_ap_um=("anterior_posterior_ccf_coordinate", "median"),
        ccf_ml_um=("left_right_ccf_coordinate", "median"),
        ccf_dv_um=("dorsal_ventral_ccf_coordinate", "median"),
        ccf_ap_span_um=("anterior_posterior_ccf_coordinate", lambda x: x.max() - x.min()),
        ccf_ml_span_um=("left_right_ccf_coordinate", lambda x: x.max() - x.min()),
        median_log2_rf_area=("log2_rf_area", "median"),
        mad_log2_rf_area=("log2_rf_area", robust_mad),
        q25_log2_rf_area=("log2_rf_area", lambda x: x.quantile(0.25)),
        q75_log2_rf_area=("log2_rf_area", lambda x: x.quantile(0.75)),
        minimum_rf_split_half_reliability=("rf_split_half_spearman_brown", "min"),
        median_rf_split_half_reliability=("rf_split_half_spearman_brown", "median"),
    ).reset_index()
    probes["ccf_ap_mm"] = probes.ccf_ap_um / 1000.0
    probes["ccf_ml_mm"] = probes.ccf_ml_um / 1000.0
    probes["ccf_dv_mm"] = probes.ccf_dv_um / 1000.0
    return probes.sort_values(["ecephys_session_id", "ecephys_probe_id"]).reset_index(drop=True)


def grid_coordinates(probes: pd.DataFrame, step_mm: float = 0.025, margin_mm: float = 0.25):
    ml = np.arange(
        np.floor((probes.ccf_ml_mm.min() - margin_mm) / step_mm) * step_mm,
        np.ceil((probes.ccf_ml_mm.max() + margin_mm) / step_mm) * step_mm + step_mm / 2,
        step_mm,
    )
    ap = np.arange(
        np.floor((probes.ccf_ap_mm.min() - margin_mm) / step_mm) * step_mm,
        np.ceil((probes.ccf_ap_mm.max() + margin_mm) / step_mm) * step_mm + step_mm / 2,
        step_mm,
    )
    grid_ml, grid_ap = np.meshgrid(ml, ap)
    return ml, ap, grid_ml, grid_ap


def kernel_components(
    probes: pd.DataFrame,
    grid_ml: np.ndarray,
    grid_ap: np.ndarray,
    bandwidth_mm: float,
    support_radius_mm: float,
):
    points = probes[["ccf_ml_mm", "ccf_ap_mm"]].to_numpy(float)
    grid_points = np.column_stack([grid_ml.ravel(), grid_ap.ravel()])
    distance = np.sqrt(np.sum((grid_points[:, None, :] - points[None, :, :]) ** 2, axis=2))
    weights = np.exp(-0.5 * (distance / bandwidth_mm) ** 2)
    weights[distance > support_radius_mm] = 0.0
    weight_sum = weights.sum(axis=1)
    effective = weight_sum**2 / np.maximum(np.square(weights).sum(axis=1), 1e-12)
    count = (distance <= support_radius_mm).sum(axis=1)
    return weights, effective, count


def smooth_surface(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
    return weights @ values / np.maximum(weights.sum(axis=1), 1e-12)


def cluster_bootstrap(
    probes: pd.DataFrame,
    weights: np.ndarray,
    repeats: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    animals = probes.specimen_id.astype(str).to_numpy()
    unique_animals = np.unique(animals)
    animal_index = {animal: index for index, animal in enumerate(unique_animals)}
    probe_animal_index = np.array([animal_index[value] for value in animals], dtype=int)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique_animals), size=(repeats, len(unique_animals)))
    counts = np.zeros((repeats, len(unique_animals)), dtype=float)
    for row in range(repeats):
        counts[row] = np.bincount(draws[row], minlength=len(unique_animals))
    probe_counts = counts[:, probe_animal_index]
    values = probes.median_log2_rf_area.to_numpy(float)
    numerator = (probe_counts * values[None, :]) @ weights.T
    denominator = probe_counts @ weights.T
    samples = numerator / np.where(denominator > 0, denominator, np.nan)
    valid_columns = weights.sum(axis=1) > 0
    low = np.full(weights.shape[0], np.nan)
    median = np.full(weights.shape[0], np.nan)
    high = np.full(weights.shape[0], np.nan)
    low[valid_columns] = np.nanpercentile(samples[:, valid_columns], 2.5, axis=0)
    median[valid_columns] = np.nanpercentile(samples[:, valid_columns], 50.0, axis=0)
    high[valid_columns] = np.nanpercentile(samples[:, valid_columns], 97.5, axis=0)
    return low, median, high


def leave_one_animal_out(
    probes: pd.DataFrame,
    bandwidth_mm: float,
    support_radius_mm: float,
    min_probes: int,
    min_effective: float,
) -> tuple[pd.DataFrame, dict]:
    points = probes[["ccf_ml_mm", "ccf_ap_mm"]].to_numpy(float)
    values = probes.median_log2_rf_area.to_numpy(float)
    animals = probes.specimen_id.astype(str).to_numpy()
    rows = []
    for index, point in enumerate(points):
        distance = np.sqrt(np.sum((points - point) ** 2, axis=1))
        allowed = animals != animals[index]
        weights = np.exp(-0.5 * (distance / bandwidth_mm) ** 2)
        weights[(distance > support_radius_mm) | ~allowed] = 0.0
        count = int(np.sum((distance <= support_radius_mm) & allowed))
        effective = float(weights.sum() ** 2 / max(np.square(weights).sum(), 1e-12))
        valid = count >= min_probes and effective >= min_effective
        prediction = float(np.average(values, weights=weights)) if valid else np.nan
        baseline = float(np.median(values[allowed]))
        rows.append(
            {
                **probes.iloc[index].to_dict(),
                "loo_prediction_log2_rf_area": prediction,
                "loo_baseline_log2_rf_area": baseline,
                "loo_neighbor_probes": count,
                "loo_effective_probes": effective,
                "loo_error": values[index] - prediction if valid else np.nan,
                "baseline_error": values[index] - baseline,
            }
        )
    table = pd.DataFrame(rows)
    valid = table.loo_prediction_log2_rf_area.notna()
    observed = table.loc[valid, "median_log2_rf_area"].to_numpy(float)
    predicted = table.loc[valid, "loo_prediction_log2_rf_area"].to_numpy(float)
    baseline_error = table.loc[valid, "baseline_error"].abs().to_numpy(float)
    model_error = table.loc[valid, "loo_error"].abs().to_numpy(float)
    rho, rho_p = spearmanr(observed, predicted) if len(observed) >= 3 else (np.nan, np.nan)
    metrics = {
        "n_probe_predictions": int(valid.sum()),
        "median_absolute_error": float(np.median(model_error)) if len(model_error) else None,
        "baseline_median_absolute_error": float(np.median(baseline_error)) if len(baseline_error) else None,
        "median_absolute_error_improvement": float(np.median(baseline_error - model_error)) if len(model_error) else None,
        "fraction_better_than_baseline": float(np.mean(model_error < baseline_error)) if len(model_error) else None,
        "spearman_rho": float(rho) if np.isfinite(rho) else None,
        "spearman_p": float(rho_p) if np.isfinite(rho_p) else None,
    }
    return table, metrics


def grouped_cross_validation(
    probes: pd.DataFrame,
    bandwidth_mm: float,
    support_radius_mm: float,
    min_probes: int,
    min_effective: float,
    repeats: int,
    folds: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    points = probes[["ccf_ml_mm", "ccf_ap_mm"]].to_numpy(float)
    values = probes.median_log2_rf_area.to_numpy(float)
    animals = probes.specimen_id.astype(str).to_numpy()
    unique_animals = np.unique(animals)
    rng = np.random.default_rng(seed)
    rows = []
    for repeat in range(repeats):
        shuffled = rng.permutation(unique_animals)
        for fold, held_animals in enumerate(np.array_split(shuffled, folds)):
            train = ~np.isin(animals, held_animals)
            for index in np.flatnonzero(~train):
                distance = np.sqrt(np.sum((points[train] - points[index]) ** 2, axis=1))
                weights = np.exp(-0.5 * (distance / bandwidth_mm) ** 2)
                weights[distance > support_radius_mm] = 0.0
                count = int(np.sum(distance <= support_radius_mm))
                effective = float(weights.sum() ** 2 / max(np.square(weights).sum(), 1e-12))
                valid = count >= min_probes and effective >= min_effective
                prediction = float(np.average(values[train], weights=weights)) if valid else np.nan
                baseline = float(np.median(values[train]))
                rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "probe_index": index,
                        "ecephys_session_id": probes.iloc[index].ecephys_session_id,
                        "specimen_id": probes.iloc[index].specimen_id,
                        "ecephys_probe_id": probes.iloc[index].ecephys_probe_id,
                        "observed_log2_rf_area": values[index],
                        "prediction_log2_rf_area": prediction,
                        "baseline_log2_rf_area": baseline,
                        "neighbor_probes": count,
                        "effective_probes": effective,
                    }
                )
    predictions = pd.DataFrame(rows)
    summary = (
        predictions.groupby("probe_index", observed=True)
        .agg(
            cv_prediction_log2_rf_area=("prediction_log2_rf_area", "median"),
            cv_baseline_log2_rf_area=("baseline_log2_rf_area", "median"),
            cv_prediction_repeats=("prediction_log2_rf_area", "count"),
            cv_prediction_iqr=("prediction_log2_rf_area", lambda x: x.quantile(0.75) - x.quantile(0.25)),
        )
        .reset_index()
        .merge(probes.reset_index(names="probe_index"), on="probe_index", how="left", validate="one_to_one")
    )
    summary["cv_error"] = summary.median_log2_rf_area - summary.cv_prediction_log2_rf_area
    summary["cv_baseline_error"] = summary.median_log2_rf_area - summary.cv_baseline_log2_rf_area
    valid = summary.cv_prediction_log2_rf_area.notna()
    observed = summary.loc[valid, "median_log2_rf_area"].to_numpy(float)
    predicted = summary.loc[valid, "cv_prediction_log2_rf_area"].to_numpy(float)
    model_error = summary.loc[valid, "cv_error"].abs().to_numpy(float)
    baseline_error = summary.loc[valid, "cv_baseline_error"].abs().to_numpy(float)
    valid_draws = predictions.dropna(subset=["prediction_log2_rf_area"]).copy()
    valid_draws["model_absolute_error"] = (
        valid_draws.observed_log2_rf_area - valid_draws.prediction_log2_rf_area
    ).abs()
    valid_draws["baseline_absolute_error"] = (
        valid_draws.observed_log2_rf_area - valid_draws.baseline_log2_rf_area
    ).abs()
    fold_scores = (
        valid_draws.groupby(["repeat", "fold"], observed=True)
        .apply(
            lambda x: pd.Series(
                {
                    "median_improvement": np.median(
                        x.baseline_absolute_error - x.model_absolute_error
                    ),
                    "fraction_improved": np.mean(
                        x.model_absolute_error < x.baseline_absolute_error
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    try:
        paired_p = float(wilcoxon(baseline_error, model_error, alternative="greater").pvalue)
    except ValueError:
        paired_p = np.nan
    metrics = {
        "repeats": repeats,
        "folds": folds,
        "n_probe_predictions": int(valid.sum()),
        "median_absolute_error": float(np.median(model_error)) if len(model_error) else None,
        "baseline_median_absolute_error": float(np.median(baseline_error)) if len(baseline_error) else None,
        "median_absolute_error_improvement": float(np.median(baseline_error - model_error)) if len(model_error) else None,
        "fraction_better_than_baseline": float(np.mean(model_error < baseline_error)) if len(model_error) else None,
        "median_fold_improvement": float(fold_scores.median_improvement.median()),
        "fraction_folds_with_positive_improvement": float(np.mean(fold_scores.median_improvement > 0)),
        "paired_wilcoxon_baseline_error_greater_p": paired_p if np.isfinite(paired_p) else None,
        "prediction_sd": float(np.std(predicted, ddof=1)) if len(predicted) > 1 else None,
        "observed_sd": float(np.std(observed, ddof=1)) if len(observed) > 1 else None,
    }
    return predictions, summary, metrics


def choose_cases(predictions: pd.DataFrame) -> pd.DataFrame:
    valid = predictions.dropna(subset=["cv_prediction_log2_rf_area"]).copy()
    center = float(predictions.median_log2_rf_area.median())
    candidates = [
        ("smallest RF size", predictions.median_log2_rf_area.idxmin(), "minimum probe median"),
        ("typical RF size", (predictions.median_log2_rf_area - center).abs().idxmin(), "closest to cohort median"),
        ("largest RF size", predictions.median_log2_rf_area.idxmax(), "maximum probe median"),
    ]
    if len(valid):
        candidates.extend(
            [
                ("larger than spatial prediction", valid.cv_error.idxmax(), "largest positive grouped-CV residual"),
                ("smaller than spatial prediction", valid.cv_error.idxmin(), "largest negative grouped-CV residual"),
            ]
        )
    rows = []
    used = set()
    for role, index, criterion in candidates:
        if index in used:
            continue
        used.add(index)
        row = predictions.loc[index].to_dict()
        row.update({"selection_role": role, "selection_criterion": criterion})
        rows.append(row)
    return pd.DataFrame(rows)


def render_main(
    probes: pd.DataFrame,
    cv: pd.DataFrame,
    ml: np.ndarray,
    ap: np.ndarray,
    surface: np.ndarray,
    effective: np.ndarray,
    count: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    mask: np.ndarray,
    metrics: dict,
    cases: pd.DataFrame,
    path: Path,
):
    shape = (len(ap), len(ml))
    surface_2d = np.where(mask, surface, np.nan).reshape(shape)
    effective_2d = effective.reshape(shape)
    count_2d = count.reshape(shape)
    half_width_2d = np.where(mask, (ci_high - ci_low) / 2.0, np.nan).reshape(shape)
    values = probes.median_log2_rf_area.to_numpy(float)
    vmin, vmax = np.quantile(values, [0.02, 0.98])
    fig, axes = plt.subplots(2, 2, figsize=(14, 11.5), constrained_layout=True)

    ax = axes[0, 0]
    raw = ax.scatter(
        probes.ccf_ml_mm,
        probes.ccf_ap_mm,
        c=values,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        s=32 + 0.45 * probes.n_units,
        edgecolors="white",
        linewidths=0.8,
    )
    for _, case in cases.iterrows():
        ax.text(case.ccf_ml_mm, case.ccf_ap_mm, str(int(case.ecephys_session_id))[-4:], fontsize=7)
    fig.colorbar(raw, ax=ax, label="Probe median log₂ RF area (deg²)")
    ax.set_title(f"A  Primary observations: {len(probes)} probe tracks / {probes.specimen_id.nunique()} animals")

    ax = axes[0, 1]
    image = ax.pcolormesh(ml, ap, surface_2d, cmap="viridis", vmin=vmin, vmax=vmax, shading="auto")
    if np.nanmax(count_2d) >= 5:
        levels = [level for level in (5, 10, 20, 30) if level <= np.nanmax(count_2d)]
        contours = ax.contour(ml, ap, count_2d, levels=levels, colors="white", linewidths=0.7)
        ax.clabel(contours, fontsize=7, fmt="%d probes")
    ax.scatter(probes.ccf_ml_mm, probes.ccf_ap_mm, s=15, color="black", alpha=0.65)
    fig.colorbar(image, ax=ax, label="Probe-balanced smoothed log₂ RF area (deg²)")
    ax.set_title("B  Probe-balanced Gaussian surface; white contours show local support")

    ax = axes[1, 0]
    uncertainty = ax.pcolormesh(ml, ap, half_width_2d, cmap="magma", shading="auto")
    levels = [level for level in (3, 5, 10, 20) if level <= np.nanmax(effective_2d)]
    if levels:
        contours = ax.contour(ml, ap, effective_2d, levels=levels, colors="white", linewidths=0.7)
        ax.clabel(contours, fontsize=7, fmt="n_eff %.0f")
    ax.scatter(probes.ccf_ml_mm, probes.ccf_ap_mm, s=13, color="#22d3ee", edgecolors="black", linewidths=0.25)
    fig.colorbar(uncertainty, ax=ax, label="Animal-bootstrap 95% CI half-width (log₂ units)")
    ax.set_title("C  Cluster-bootstrap uncertainty; contours show effective probes")

    ax = axes[1, 1]
    valid = cv.dropna(subset=["cv_prediction_log2_rf_area"])
    ax.scatter(
        valid.cv_prediction_log2_rf_area,
        valid.median_log2_rf_area,
        c=valid.ccf_ap_mm,
        cmap="coolwarm",
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
        ax.set_xlabel("Medial–lateral CCF (mm; lateral to left)")
        ax.set_ylabel("Anterior–posterior CCF (mm)")
        ax.invert_xaxis()
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#dddddd", linewidth=0.35, alpha=0.45)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Allen V1 RF size across cortical anatomy — probe/animal-balanced exploratory surface\n"
        "Units determine each probe median; spatial replication and uncertainty are at probe/animal level",
        fontsize=14,
    )
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def render_case_residuals(cv: pd.DataFrame, cases: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8.8, 7.2), constrained_layout=True)
    valid = cv.dropna(subset=["cv_error"])
    limit = float(np.nanmax(np.abs(valid.cv_error)))
    scatter = ax.scatter(
        valid.ccf_ml_mm,
        valid.ccf_ap_mm,
        c=valid.cv_error,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        s=35 + 0.45 * valid.n_units,
        edgecolors="white",
        linewidths=0.8,
    )
    for _, case in cases.iterrows():
        if np.isfinite(case.get("cv_error", np.nan)):
            ax.scatter(case.ccf_ml_mm, case.ccf_ap_mm, facecolors="none", edgecolors="black", s=180, linewidths=1.4)
            ax.text(
                case.ccf_ml_mm,
                case.ccf_ap_mm,
                f" {case.selection_role}\n {int(case.ecephys_session_id)}",
                fontsize=8,
                va="bottom",
            )
    fig.colorbar(scatter, ax=ax, label="Observed − animal-grouped CV prediction (log₂ RF area)")
    ax.set(
        title="Selected concrete cases and animal-grouped cross-validation residuals",
        xlabel="Medial–lateral CCF (mm; lateral to left)",
        ylabel="Anterior–posterior CCF (mm)",
        aspect="equal",
    )
    ax.invert_xaxis()
    ax.grid(color="#dddddd", linewidth=0.4)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    fits_path = args.fits.resolve()
    units_path = args.units.resolve()
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
    probes.to_csv(output / "probe_rf_size_summary.csv", index=False, float_format="%.9g")

    ml, ap, grid_ml, grid_ap = grid_coordinates(probes)
    weights, effective, count = kernel_components(
        probes, grid_ml, grid_ap, args.bandwidth_mm, args.support_radius_mm
    )
    values = probes.median_log2_rf_area.to_numpy(float)
    surface = smooth_surface(weights, values)
    mask = (count >= args.min_probes) & (effective >= args.min_effective_probes)
    ci_low, bootstrap_median, ci_high = cluster_bootstrap(
        probes, weights, args.bootstrap_repeats, args.seed
    )

    loo, loo_metrics = leave_one_animal_out(
        probes,
        args.bandwidth_mm,
        args.support_radius_mm,
        args.min_probes,
        args.min_effective_probes,
    )
    loo.to_csv(output / "leave_one_animal_out_predictions.csv", index=False, float_format="%.9g")
    cv_draws, cv, cv_metrics = grouped_cross_validation(
        probes,
        args.bandwidth_mm,
        args.support_radius_mm,
        args.min_probes,
        args.min_effective_probes,
        repeats=100,
        folds=5,
        seed=args.seed + 1,
    )
    cv_draws.to_csv(output / "grouped_cross_validation_draws.csv.gz", index=False, compression="gzip", float_format="%.9g")
    cv.to_csv(output / "grouped_cross_validation_probe_summary.csv", index=False, float_format="%.9g")
    cases = choose_cases(cv)
    cases.to_csv(output / "case_selection.csv", index=False, float_format="%.9g")

    sensitivity_rows = []
    for sensitivity_index, bandwidth in enumerate((0.25, 0.35, 0.50, 0.70)):
        radius = max(args.support_radius_mm, 1.7 * bandwidth)
        _, _, metrics = grouped_cross_validation(
            probes,
            bandwidth,
            radius,
            args.min_probes,
            args.min_effective_probes,
            repeats=25,
            folds=5,
            seed=args.seed + 100 + sensitivity_index,
        )
        sensitivity_rows.append({"bandwidth_mm": bandwidth, "support_radius_mm": radius, **metrics})
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(output / "bandwidth_sensitivity.csv", index=False, float_format="%.9g")

    grid = pd.DataFrame(
        {
            "ccf_ml_mm": grid_ml.ravel(),
            "ccf_ap_mm": grid_ap.ravel(),
            "smoothed_log2_rf_area": np.where(mask, surface, np.nan),
            "bootstrap_median_log2_rf_area": np.where(mask, bootstrap_median, np.nan),
            "bootstrap_ci_low": np.where(mask, ci_low, np.nan),
            "bootstrap_ci_high": np.where(mask, ci_high, np.nan),
            "support_probe_count": count,
            "effective_probe_count": effective,
            "supported": mask,
        }
    )
    grid.to_csv(output / "surface_grid.csv.gz", index=False, compression="gzip", float_format="%.9g")

    main_figure = output / "Figure_v1_rf_size_cortical_surface_probe_balanced.png"
    case_figure = output / "Figure_v1_rf_size_cortical_surface_selected_cases.png"
    render_main(
        probes, cv, ml, ap, surface, effective, count, ci_low, ci_high, mask, cv_metrics, cases, main_figure
    )
    render_case_residuals(cv, cases, case_figure)

    manifest = {
        "status": "exploratory probe-balanced cortical RF-size surface",
        "inputs": {
            "fits": {"path": str(fits_path), "sha256": sha256(fits_path)},
            "units": {"path": str(units_path), "sha256": sha256(units_path)},
        },
        "metric": "probe median log2 axis-aligned analytic-aperture RF half-maximum ellipse area (deg2)",
        "coordinate": "median unit AP/ML CCF position per V1 probe track; a cortical-column proxy, not a pia intersection",
        "filters": {
            "structure": "VISp",
            "spatial_model": "aperture",
            "axis_censored": False,
            "minimum_axis_edge_distance_deg": args.edge_exclusion_deg,
            "minimum_split_half_spearman_brown_exclusive": args.minimum_split_half_reliability,
        },
        "audit": audit,
        "n_probes": int(len(probes)),
        "n_animals": int(probes.specimen_id.nunique()),
        "surface": {
            "bandwidth_mm": args.bandwidth_mm,
            "support_radius_mm": args.support_radius_mm,
            "min_probes": args.min_probes,
            "min_effective_probes": args.min_effective_probes,
            "supported_grid_fraction": float(mask.mean()),
        },
        "bootstrap": {"cluster": "specimen_id", "repeats": args.bootstrap_repeats, "seed": args.seed},
        "grouped_cross_validation": cv_metrics,
        "leave_one_animal_out_mae_diagnostic": loo_metrics,
        "limitations": [
            "Most animals contribute one V1 probe, so an animal-wide RF-size offset is confounded with cortical position.",
            "The probe median AP/ML coordinate collapses a slanted 3D track to a cortical-column proxy and is not a reconstructed pia entry point.",
            "The smooth surface is descriptive; unsupported grid regions are masked and must not be extrapolated.",
            "RF size is not used here to register sessions or alter measured anatomy.",
        ],
        "outputs": [
            main_figure.name,
            case_figure.name,
            "probe_rf_size_summary.csv",
            "grouped_cross_validation_draws.csv.gz",
            "grouped_cross_validation_probe_summary.csv",
            "leave_one_animal_out_predictions.csv",
            "bandwidth_sensitivity.csv",
            "case_selection.csv",
            "surface_grid.csv.gz",
            "README.md",
        ],
    }
    if reliability_path is not None:
        manifest["inputs"]["reliability"] = {
            "path": str(reliability_path),
            "sha256": sha256(reliability_path),
        }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    lines = [
        "# Probe-balanced Allen V1 RF-size cortical surface",
        "",
        "## What this is",
        "",
        "An exploratory map of absolute V1 RF size over AP/ML CCF anatomy. Eligible units are collapsed to one robust estimate per probe, so the spatial replicate is the probe/animal rather than the cell.",
        "",
        "## Cohort and metric",
        "",
        f"- {audit['eligible_units']} eligible units from {len(probes)} V1 probe tracks in {probes.specimen_id.nunique()} animals.",
        f"- Excluded parameter-bound fits and RF centers within {args.edge_exclusion_deg:g} degrees of a sampled stimulus edge.",
        "- RF size is the probe median of log2 axis-aligned analytic-aperture half-maximum ellipse area in deg2.",
        "- Cortical position is the median AP/ML CCF coordinate of eligible units on that probe; it is a column proxy, not a fitted surface entry point.",
        "",
        "## Primary validation",
        "",
        f"Repeated animal-grouped five-fold predictions were available for {cv_metrics['n_probe_predictions']}/{len(probes)} probes. Median absolute error was {cv_metrics['median_absolute_error']:.3f} log2 units versus {cv_metrics['baseline_median_absolute_error']:.3f} for the training-cohort median; median per-probe improvement was {cv_metrics['median_absolute_error_improvement']:+.3f}, with {100 * cv_metrics['fraction_better_than_baseline']:.1f}% of probes improved. Only {100 * cv_metrics['fraction_folds_with_positive_improvement']:.1f}% of held-out folds had positive median improvement (paired one-sided Wilcoxon p={cv_metrics['paired_wilcoxon_baseline_error_greater_p']:.3g}).",
        "",
        "## Interpretation limits",
        "",
        "- Most animals contribute only one V1 probe. A global animal-to-animal RF-size shift is therefore confounded with cortical position.",
        "- Units within a probe are not treated as independent spatial samples.",
        "- The surface is shown only where at least five probes and three effective probes contribute locally.",
        "- This does not validate RF size as a registration feature; anatomy is fixed and RF size is only mapped onto it.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python -m scripts.build_allen_v1_rf_size_cortical_surface",
        "```",
    ]
    if args.minimum_split_half_reliability is not None:
        lines.insert(
            lines.index("- RF size is the probe median of log2 axis-aligned analytic-aperture half-maximum ellipse area in deg2."),
            f"- Required unit RF split-half Spearman-Brown reliability > {args.minimum_split_half_reliability:g} (strict inequality).",
        )
        lines[-2] += (
            f" --minimum-split-half-reliability {args.minimum_split_half_reliability:g}"
            f" --output-dir {output}"
        )
    (output / "README.md").write_text("\n".join(lines) + "\n")

    print(json.dumps({"audit": audit, "n_probes": len(probes), "n_animals": probes.specimen_id.nunique(), "grouped_cv": cv_metrics, "loo_mae_diagnostic": loo_metrics}, indent=2))
    print(main_figure)


if __name__ == "__main__":
    main()
