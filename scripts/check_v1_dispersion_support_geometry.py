#!/usr/bin/env python3
"""Concrete check of anatomical-support inflation in V1 RF covariance.

The earlier physical-sampling control removed every covariance-trace component
predictable from shank position.  That is too broad: genuine retinotopic
dependence of local RF scatter should itself be anatomically organized.  Here we
instead estimate the narrower nuisance term produced when an RF-space
neighborhood contains cells from separated anatomical positions along the probe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts" / "v1_absolute_size_dispersion_translation_checkpoint"
DEFAULT_INPUT = CHECKPOINT / "uncensored_size_sensitivity" / "v1_unit_descriptors.csv.gz"
DEFAULT_UNITS = ROOT / "data" / "unit_table.csv"
DEFAULT_OUTPUT = CHECKPOINT / "support_geometry_control"
DEFAULT_SESSIONS = (760345702, 719161530, 835479236)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--unit-table", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sessions", type=int, nargs="+", default=DEFAULT_SESSIONS)
    parser.add_argument("--map-bandwidths-um", type=float, nargs="+", default=(120, 250, 400))
    parser.add_argument("--rf-bandwidth-deg", type=float, default=15.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def weighted_covariance(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total = weights.sum(axis=1)
    mean = weights @ points / np.maximum(total[:, None], 1e-12)
    centered = points[None, :, :] - mean[:, None, :]
    covariance = np.einsum("ij,ijk,ijl->ikl", weights, centered, centered)
    covariance /= np.maximum(total[:, None, None], 1e-12)
    return covariance


def loo_mean_map(position: np.ndarray, rf: np.ndarray, bandwidth_um: float) -> tuple[np.ndarray, np.ndarray]:
    distance = position[:, None] - position[None, :]
    weights = np.exp(-0.5 * (distance / bandwidth_um) ** 2)
    np.fill_diagonal(weights, 0.0)
    total = weights.sum(axis=1)
    effective = total**2 / np.maximum((weights**2).sum(axis=1), 1e-12)
    predicted = weights @ rf / np.maximum(total[:, None], 1e-12)
    predicted[effective < 3] = np.nan
    return predicted, effective


def support_decomposition(
    position: np.ndarray,
    rf: np.ndarray,
    predicted_rf: np.ndarray,
    rf_bandwidth_deg: float,
) -> pd.DataFrame:
    delta = rf[None, :, :] - rf[:, None, :]
    weights = np.exp(-0.5 * np.sum(delta**2, axis=2) / rf_bandwidth_deg**2)
    np.fill_diagonal(weights, 0.0)
    finite_prediction = np.isfinite(predicted_rf).all(axis=1)
    weights[:, ~finite_prediction] = 0.0
    total = weights.sum(axis=1)
    effective = total**2 / np.maximum((weights**2).sum(axis=1), 1e-12)

    raw_cov = weighted_covariance(rf, weights)
    sampling_cov = weighted_covariance(np.nan_to_num(predicted_rf), weights)
    residual_rf = rf - predicted_rf
    residual_cov = weighted_covariance(np.nan_to_num(residual_rf), weights)

    mean_position = weights @ position / np.maximum(total, 1e-12)
    physical_variance = (
        weights @ (position**2) / np.maximum(total, 1e-12) - mean_position**2
    )
    raw_trace = np.trace(raw_cov, axis1=1, axis2=2)
    sampling_trace = np.trace(sampling_cov, axis1=1, axis2=2)
    residual_trace = np.trace(residual_cov, axis1=1, axis2=2)
    valid = (effective >= 3) & np.isfinite(predicted_rf).all(axis=1)
    residual_cxx = residual_cov[:, 0, 0].copy()
    residual_cyy = residual_cov[:, 1, 1].copy()
    residual_cxy = residual_cov[:, 0, 1].copy()
    for values in (
        raw_trace,
        sampling_trace,
        residual_trace,
        residual_cxx,
        residual_cyy,
        residual_cxy,
        physical_variance,
    ):
        values[~valid] = np.nan
    return pd.DataFrame(
        {
            "raw_trace_deg2": raw_trace,
            "sampling_trace_deg2": sampling_trace,
            "subtractive_excess_trace_deg2": raw_trace - sampling_trace,
            "residual_trace_deg2": residual_trace,
            "residual_cov_azimuth_deg2": residual_cxx,
            "residual_cov_elevation_deg2": residual_cyy,
            "residual_cov_azimuth_elevation_deg2": residual_cxy,
            "rf_neighborhood_effective_n": effective,
            "rf_neighborhood_physical_sd_um": np.sqrt(np.maximum(physical_variance, 0)),
        }
    )


def safe_rho(x: pd.Series, y: pd.Series) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 5 or x[valid].std() == 0 or y[valid].std() == 0:
        return np.nan
    return float(spearmanr(x[valid], y[valid]).statistic)


def load_population(input_path: Path, unit_path: Path) -> pd.DataFrame:
    population = pd.read_csv(input_path, low_memory=False)
    units = pd.read_csv(
        unit_path,
        usecols=[
            "ecephys_unit_id",
            "probe_vertical_position",
            "anterior_posterior_ccf_coordinate",
            "left_right_ccf_coordinate",
            "dorsal_ventral_ccf_coordinate",
        ],
        low_memory=False,
    )
    return population.merge(units, on="ecephys_unit_id", how="left")


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output}; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)

    population = load_population(args.input.resolve(), args.unit_table.resolve())
    roles = {
        760345702: "previous trace success",
        719161530: "previous typical case",
        835479236: "previous failure / strongest CCF association",
    }
    rows = []
    audits = []
    for session_id in args.sessions:
        local = population.loc[population["ecephys_session_id"].eq(session_id)].copy()
        local = local.dropna(
            subset=["probe_vertical_position", "rf_azimuth_deg", "rf_elevation_deg"]
        ).sort_values("probe_vertical_position")
        position = local["probe_vertical_position"].to_numpy(float)
        rf = local[["rf_azimuth_deg", "rf_elevation_deg"]].to_numpy(float)
        for bandwidth in args.map_bandwidths_um:
            predicted, map_effective = loo_mean_map(position, rf, bandwidth)
            decomposed = support_decomposition(
                position, rf, predicted, args.rf_bandwidth_deg
            )
            result = local.reset_index(drop=True).copy()
            result["map_bandwidth_um"] = float(bandwidth)
            result["predicted_rf_azimuth_deg"] = predicted[:, 0]
            result["predicted_rf_elevation_deg"] = predicted[:, 1]
            result["mean_map_effective_n"] = map_effective
            result = pd.concat([result, decomposed], axis=1)
            rows.append(result)
            valid = result.dropna(subset=["raw_trace_deg2", "sampling_trace_deg2"])
            audits.append(
                {
                    "ecephys_session_id": session_id,
                    "selection_role": roles.get(session_id, "user-selected"),
                    "v1_units": len(local),
                    "map_bandwidth_um": float(bandwidth),
                    "valid_neighborhoods": len(valid),
                    "raw_vs_sampling_trace_rho": safe_rho(
                        valid["raw_trace_deg2"], valid["sampling_trace_deg2"]
                    ),
                    "raw_vs_physical_spread_rho": safe_rho(
                        valid["raw_trace_deg2"], valid["rf_neighborhood_physical_sd_um"]
                    ),
                    "sampling_vs_physical_spread_rho": safe_rho(
                        valid["sampling_trace_deg2"], valid["rf_neighborhood_physical_sd_um"]
                    ),
                    "median_sampling_fraction": float(
                        np.nanmedian(
                            valid["sampling_trace_deg2"]
                            / np.maximum(valid["raw_trace_deg2"], 1e-12)
                        )
                    ),
                    "median_raw_trace_deg2": float(np.nanmedian(valid["raw_trace_deg2"])),
                    "median_sampling_trace_deg2": float(
                        np.nanmedian(valid["sampling_trace_deg2"])
                    ),
                    "median_residual_trace_deg2": float(
                        np.nanmedian(valid["residual_trace_deg2"])
                    ),
                    "negative_subtractive_excess_fraction": float(
                        np.mean(valid["subtractive_excess_trace_deg2"] < 0)
                    ),
                }
            )

    results = pd.concat(rows, ignore_index=True)
    audit = pd.DataFrame(audits)
    results.to_csv(output / "unit_neighborhood_support_decomposition.csv.gz", index=False, compression="gzip")
    audit.to_csv(output / "session_bandwidth_audit.csv", index=False)

    display_bandwidth = 250.0
    fig, axes = plt.subplots(len(args.sessions), 4, figsize=(17, 4.2 * len(args.sessions)))
    if len(args.sessions) == 1:
        axes = axes[None, :]
    for row_index, session_id in enumerate(args.sessions):
        local = results.loc[
            results["ecephys_session_id"].eq(session_id)
            & results["map_bandwidth_um"].eq(display_bandwidth)
        ].sort_values("probe_vertical_position")
        ax = axes[row_index, 0]
        ax.scatter(local["probe_vertical_position"], local["rf_azimuth_deg"], s=13, alpha=.55, label="observed az")
        ax.scatter(local["probe_vertical_position"], local["rf_elevation_deg"], s=13, alpha=.55, label="observed el")
        ax.plot(local["probe_vertical_position"], local["predicted_rf_azimuth_deg"], lw=2, label="smooth az")
        ax.plot(local["probe_vertical_position"], local["predicted_rf_elevation_deg"], lw=2, label="smooth el")
        ax.set(xlabel="probe vertical position (µm)", ylabel="RF coordinate (deg)", title="Cross-fitted mean map")
        if row_index == 0:
            ax.legend(frameon=False, ncol=2, fontsize=8)

        ax = axes[row_index, 1]
        scatter = ax.scatter(
            local["sampling_trace_deg2"], local["raw_trace_deg2"],
            c=local["rf_neighborhood_physical_sd_um"], cmap="viridis", s=28, alpha=.8,
        )
        finite = np.isfinite(local["sampling_trace_deg2"]) & np.isfinite(local["raw_trace_deg2"])
        upper = float(np.nanmax(local.loc[finite, ["sampling_trace_deg2", "raw_trace_deg2"]].to_numpy()))
        ax.plot([0, upper], [0, upper], color="0.35", ls="--", lw=1)
        rho = safe_rho(local["raw_trace_deg2"], local["sampling_trace_deg2"])
        ax.set(xlabel="expected from sampled anatomy (deg²)", ylabel="observed RF covariance trace (deg²)", title=f"Support contribution: ρ={rho:.2f}")
        fig.colorbar(scatter, ax=ax, label="physical SD in RF neighborhood (µm)")

        ax = axes[row_index, 2]
        scatter = ax.scatter(
            local["rf_azimuth_deg"], local["rf_elevation_deg"],
            c=local["sampling_trace_deg2"], cmap="magma", s=34,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set(xlabel="RF azimuth (deg)", ylabel="RF elevation (deg)", title="Sampling-only covariance")
        fig.colorbar(scatter, ax=ax, label="trace (deg²)")

        ax = axes[row_index, 3]
        scatter = ax.scatter(
            local["rf_azimuth_deg"], local["rf_elevation_deg"],
            c=np.log2(np.maximum(local["residual_trace_deg2"], 1e-6)), cmap="cividis", s=34,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set(xlabel="RF azimuth (deg)", ylabel="RF elevation (deg)", title="Conditional residual scatter")
        fig.colorbar(scatter, ax=ax, label="log₂ residual trace")
        axes[row_index, 0].text(
            .01, 1.08, f"{session_id}: {roles.get(session_id, 'user-selected')}",
            transform=axes[row_index, 0].transAxes, fontsize=11, fontweight="bold",
        )

    fig.suptitle(
        "Narrow sampling control: subtract mean-map variation within each exact RF neighborhood\n"
        "250-µm LOO anatomical mean map; 15° RF-space neighborhoods",
        y=.995,
    )
    fig.tight_layout(rect=(0, 0, 1, .97))
    figure_path = output / "Figure_v1_support_geometry_control_cases.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    sensitivity_figure = output / "Figure_v1_support_geometry_bandwidth_sensitivity.png"
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for session_id in args.sessions:
        local = audit.loc[audit["ecephys_session_id"].eq(session_id)].sort_values("map_bandwidth_um")
        label = f"{session_id}: {roles.get(session_id, 'user-selected')}"
        axes[0].plot(local["map_bandwidth_um"], 100 * local["median_sampling_fraction"], marker="o", label=label)
        axes[1].plot(local["map_bandwidth_um"], local["raw_vs_sampling_trace_rho"], marker="o")
        axes[2].plot(local["map_bandwidth_um"], local["median_residual_trace_deg2"], marker="o")
    axes[0].set(xlabel="mean-map bandwidth (µm)", ylabel="median sampling / raw trace (%)", title="Estimated nuisance magnitude")
    axes[1].set(xlabel="mean-map bandwidth (µm)", ylabel="Spearman ρ", title="Raw vs sampling-only covariance")
    axes[2].set(xlabel="mean-map bandwidth (µm)", ylabel="median residual trace (deg²)", title="Conditional RF scatter")
    axes[1].axhline(0, color="0.5", lw=1)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Mean-map smoothness is an explicit unresolved scale choice")
    fig.tight_layout()
    fig.savefig(sensitivity_figure, dpi=180, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "question": "Does discontinuous/wide anatomical support inflate RF-space covariance beyond the smooth CCF-to-RF map?",
        "mean_map": "leave-one-cell-out Gaussian smoothing of azimuth/elevation over probe vertical position",
        "nuisance": "weighted covariance of predicted mean RFs at the exact cells contributing to each RF-space neighborhood",
        "target": "weighted covariance of RF residual vectors after subtracting the smooth mean map",
        "rf_neighborhood_bandwidth_deg": args.rf_bandwidth_deg,
        "map_bandwidths_um": list(args.map_bandwidths_um),
        "sessions": list(args.sessions),
        "outputs": [figure_path.name, sensitivity_figure.name, "session_bandwidth_audit.csv", "unit_neighborhood_support_decomposition.csv.gz"],
    }
    (output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(figure_path)
    print(sensitivity_figure)
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
