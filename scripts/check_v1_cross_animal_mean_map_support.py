#!/usr/bin/env python3
"""Leave-one-animal-out V1 mean-map sampling correction in concrete cases.

The common CCF->RF geometry is estimated from robust blocks along all other
animals' V1 probe trajectories. Session fixed effects remove unknown visual-field
translations. The held-out animal's exact unit CCF positions are then passed
through that independently learned map to estimate covariance caused only by its
anatomical sampling support.
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

from scripts.check_v1_dispersion_support_geometry import support_decomposition


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts" / "v1_absolute_size_dispersion_translation_checkpoint"
DEFAULT_INPUT = CHECKPOINT / "uncensored_size_sensitivity" / "v1_unit_descriptors.csv.gz"
DEFAULT_UNITS = ROOT / "data" / "unit_table.csv"
DEFAULT_OUTPUT = CHECKPOINT / "cross_animal_mean_map_support_control"
DEFAULT_SESSIONS = (760345702, 719161530, 835479236)
MODELS = ("affine", "quadratic")
CCF_COLUMNS = ("anterior_posterior_ccf_coordinate", "left_right_ccf_coordinate")
RF_COLUMNS = ("rf_azimuth_deg", "rf_elevation_deg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--unit-table", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sessions", type=int, nargs="+", default=DEFAULT_SESSIONS)
    parser.add_argument("--physical-blocks", type=int, default=6)
    parser.add_argument("--rf-bandwidth-deg", type=float, default=15.0)
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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
    population = population.merge(units, on="ecephys_unit_id", how="left")
    return population.dropna(subset=[*CCF_COLUMNS, *RF_COLUMNS, "probe_vertical_position"]).copy()


def physical_blocks(values: pd.Series, count: int) -> np.ndarray:
    ranks = values.rank(method="dense").to_numpy(float) - 1
    unique_count = max(int(np.nanmax(ranks)) + 1, 1)
    return np.minimum((ranks * count / unique_count).astype(int), count - 1)


def make_block_table(population: pd.DataFrame, block_count: int) -> pd.DataFrame:
    frames = []
    for session_id, local in population.groupby("ecephys_session_id", observed=True):
        local = local.copy()
        local["physical_block"] = physical_blocks(local["probe_vertical_position"], block_count)
        blocks = (
            local.groupby("physical_block", as_index=False)
            .agg(
                specimen_id=("specimen_id", "first"),
                units=("ecephys_unit_id", "size"),
                probe_vertical_position=("probe_vertical_position", "median"),
                anterior_posterior_ccf_coordinate=(CCF_COLUMNS[0], "median"),
                left_right_ccf_coordinate=(CCF_COLUMNS[1], "median"),
                rf_azimuth_deg=(RF_COLUMNS[0], "median"),
                rf_elevation_deg=(RF_COLUMNS[1], "median"),
            )
        )
        blocks["ecephys_session_id"] = int(session_id)
        frames.append(blocks)
    return pd.concat(frames, ignore_index=True)


def feature_matrix(ccf: np.ndarray, origin: np.ndarray, model: str) -> np.ndarray:
    xy = (np.asarray(ccf, float) - origin) / 1000.0
    x, y = xy[:, 0], xy[:, 1]
    if model == "affine":
        return np.column_stack([x, y])
    if model == "quadratic":
        return np.column_stack([x, y, x * x, x * y, y * y])
    raise ValueError(model)


def within_session_center(values: np.ndarray, sessions: np.ndarray) -> np.ndarray:
    centered = values.copy()
    for session_id in np.unique(sessions):
        selected = sessions == session_id
        centered[selected] -= np.mean(centered[selected], axis=0)
    return centered


def fit_fixed_effect_geometry(
    blocks: pd.DataFrame, model: str, ridge: float
) -> dict[str, np.ndarray | float]:
    origin = blocks[list(CCF_COLUMNS)].to_numpy(float).mean(axis=0)
    features = feature_matrix(blocks[list(CCF_COLUMNS)].to_numpy(float), origin, model)
    response = blocks[list(RF_COLUMNS)].to_numpy(float)
    sessions = blocks["ecephys_session_id"].to_numpy(int)
    x = within_session_center(features, sessions)
    y = within_session_center(response, sessions)
    counts = blocks.groupby("ecephys_session_id")["ecephys_session_id"].transform("size").to_numpy(float)
    base_weight = 1.0 / counts
    robust_weight = np.ones(len(blocks))
    coefficient = np.zeros((x.shape[1], 2))
    for _ in range(50):
        weight = base_weight * robust_weight
        gram = x.T @ (weight[:, None] * x) + ridge * np.eye(x.shape[1])
        updated = np.linalg.solve(gram, x.T @ (weight[:, None] * y))
        residual = y - x @ updated
        radius = np.sqrt(np.sum(residual**2, axis=1))
        scale = 1.4826 * np.median(np.abs(radius - np.median(radius))) + 1e-6
        cutoff = 1.5 * scale
        new_robust = np.minimum(1.0, cutoff / np.maximum(radius, 1e-12))
        if np.max(np.abs(updated - coefficient)) < 1e-8:
            coefficient = updated
            robust_weight = new_robust
            break
        coefficient = updated
        robust_weight = new_robust
    residual = y - x @ coefficient
    return {
        "origin": origin,
        "coefficient": coefficient,
        "training_centered_rmse_deg": float(np.sqrt(np.mean(residual**2))),
        "training_sessions": int(blocks["ecephys_session_id"].nunique()),
        "training_blocks": len(blocks),
    }


def predict(ccf: np.ndarray, fit: dict, model: str) -> np.ndarray:
    return feature_matrix(ccf, np.asarray(fit["origin"]), model) @ np.asarray(fit["coefficient"])


def centered_validation(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed_centered = observed - observed.mean(axis=0)
    predicted_centered = predicted - predicted.mean(axis=0)
    residual = observed_centered - predicted_centered
    constant_mse = float(np.mean(observed_centered**2))
    model_mse = float(np.mean(residual**2))
    output = {
        "heldout_gradient_rmse_deg": float(np.sqrt(model_mse)),
        "heldout_constant_rmse_deg": float(np.sqrt(constant_mse)),
        "heldout_gradient_r2_vs_constant": float(1 - model_mse / constant_mse) if constant_mse > 0 else np.nan,
    }
    for index, label in enumerate(("azimuth", "elevation")):
        if np.std(observed_centered[:, index]) > 0 and np.std(predicted_centered[:, index]) > 0:
            output[f"heldout_{label}_rho"] = float(
                spearmanr(observed_centered[:, index], predicted_centered[:, index]).statistic
            )
        else:
            output[f"heldout_{label}_rho"] = np.nan
    return output


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output}; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)

    population = load_population(args.input.resolve(), args.unit_table.resolve())
    blocks = make_block_table(population, args.physical_blocks)
    usable_sessions = blocks.groupby("ecephys_session_id").size()
    usable_sessions = usable_sessions.index[usable_sessions >= 4]
    blocks = blocks.loc[blocks["ecephys_session_id"].isin(usable_sessions)].copy()
    population = population.loc[population["ecephys_session_id"].isin(usable_sessions)].copy()
    roles = {
        760345702: "previous covariance-trace success",
        719161530: "previous typical case",
        835479236: "previous failure / strongest CCF association",
    }

    result_frames = []
    block_frames = []
    audit_rows = []
    coefficient_rows = []
    for session_id in args.sessions:
        target_units = population.loc[population["ecephys_session_id"].eq(session_id)].copy()
        target_blocks = blocks.loc[blocks["ecephys_session_id"].eq(session_id)].copy()
        target_specimen = int(target_units["specimen_id"].iloc[0])
        training = blocks.loc[blocks["specimen_id"].ne(target_specimen)].copy()
        for model in MODELS:
            fit = fit_fixed_effect_geometry(training, model, args.ridge)
            block_predicted = predict(target_blocks[list(CCF_COLUMNS)].to_numpy(float), fit, model)
            validation = centered_validation(target_blocks[list(RF_COLUMNS)].to_numpy(float), block_predicted)
            unit_predicted = predict(target_units[list(CCF_COLUMNS)].to_numpy(float), fit, model)
            decomposition = support_decomposition(
                target_units["probe_vertical_position"].to_numpy(float),
                target_units[list(RF_COLUMNS)].to_numpy(float),
                unit_predicted,
                args.rf_bandwidth_deg,
            )
            unit_result = target_units.reset_index(drop=True).copy()
            unit_result["model"] = model
            unit_result["predicted_rf_azimuth_untranslated_deg"] = unit_predicted[:, 0]
            unit_result["predicted_rf_elevation_untranslated_deg"] = unit_predicted[:, 1]
            unit_result = pd.concat([unit_result, decomposition], axis=1)
            result_frames.append(unit_result)

            block_result = target_blocks.reset_index(drop=True).copy()
            display_translation = (
                block_result[list(RF_COLUMNS)].to_numpy(float) - block_predicted
            ).mean(axis=0)
            block_result["model"] = model
            block_result["predicted_rf_azimuth_untranslated_deg"] = block_predicted[:, 0]
            block_result["predicted_rf_elevation_untranslated_deg"] = block_predicted[:, 1]
            block_result["predicted_rf_azimuth_display_deg"] = block_predicted[:, 0] + display_translation[0]
            block_result["predicted_rf_elevation_display_deg"] = block_predicted[:, 1] + display_translation[1]
            block_frames.append(block_result)

            valid = unit_result.dropna(subset=["raw_trace_deg2", "sampling_trace_deg2"])
            audit_rows.append(
                {
                    "ecephys_session_id": session_id,
                    "specimen_id": target_specimen,
                    "selection_role": roles.get(session_id, "user-selected"),
                    "model": model,
                    "target_units": len(target_units),
                    "target_blocks": len(target_blocks),
                    "training_animals": fit["training_sessions"],
                    "training_blocks": fit["training_blocks"],
                    "training_centered_rmse_deg": fit["training_centered_rmse_deg"],
                    **validation,
                    "median_raw_trace_deg2": float(np.nanmedian(valid["raw_trace_deg2"])),
                    "median_sampling_trace_deg2": float(np.nanmedian(valid["sampling_trace_deg2"])),
                    "median_sampling_fraction": float(
                        np.nanmedian(valid["sampling_trace_deg2"] / np.maximum(valid["raw_trace_deg2"], 1e-12))
                    ),
                    "median_residual_trace_deg2": float(np.nanmedian(valid["residual_trace_deg2"])),
                    "display_translation_azimuth_deg": float(display_translation[0]),
                    "display_translation_elevation_deg": float(display_translation[1]),
                }
            )
            for feature_index in range(np.asarray(fit["coefficient"]).shape[0]):
                coefficient_rows.append(
                    {
                        "heldout_session_id": session_id,
                        "model": model,
                        "feature_index": feature_index,
                        "azimuth_coefficient": np.asarray(fit["coefficient"])[feature_index, 0],
                        "elevation_coefficient": np.asarray(fit["coefficient"])[feature_index, 1],
                    }
                )

    results = pd.concat(result_frames, ignore_index=True)
    block_results = pd.concat(block_frames, ignore_index=True)
    audit = pd.DataFrame(audit_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    results.to_csv(output / "heldout_unit_support_decomposition.csv.gz", index=False, compression="gzip")
    block_results.to_csv(output / "heldout_block_predictions.csv", index=False)
    audit.to_csv(output / "heldout_session_model_audit.csv", index=False)
    coefficients.to_csv(output / "heldout_model_coefficients.csv", index=False)

    fig, axes = plt.subplots(len(args.sessions), 4, figsize=(17, 4.1 * len(args.sessions)))
    if len(args.sessions) == 1:
        axes = axes[None, :]
    primary_model = "quadratic"
    for row_index, session_id in enumerate(args.sessions):
        local_blocks = block_results.loc[
            block_results["ecephys_session_id"].eq(session_id) & block_results["model"].eq(primary_model)
        ].sort_values("probe_vertical_position")
        local_units = results.loc[
            results["ecephys_session_id"].eq(session_id) & results["model"].eq(primary_model)
        ]
        local_audit = audit.loc[
            audit["ecephys_session_id"].eq(session_id) & audit["model"].eq(primary_model)
        ].iloc[0]

        ax = axes[row_index, 0]
        ax.scatter(local_blocks["rf_azimuth_deg"], local_blocks["rf_elevation_deg"], s=52, label="observed block median")
        ax.scatter(local_blocks["predicted_rf_azimuth_display_deg"], local_blocks["predicted_rf_elevation_display_deg"], marker="x", s=60, label="LOAO prediction + display translation")
        for observed, predicted in zip(
            local_blocks[list(RF_COLUMNS)].to_numpy(float),
            local_blocks[["predicted_rf_azimuth_display_deg", "predicted_rf_elevation_display_deg"]].to_numpy(float),
        ):
            ax.plot([observed[0], predicted[0]], [observed[1], predicted[1]], color="0.6", lw=.8)
        combined = np.vstack(
            [
                local_blocks[list(RF_COLUMNS)].to_numpy(float),
                local_blocks[["predicted_rf_azimuth_display_deg", "predicted_rf_elevation_display_deg"]].to_numpy(float),
            ]
        )
        center = (combined.min(axis=0) + combined.max(axis=0)) / 2
        half_span = max(float(np.ptp(combined[:, 0])), float(np.ptp(combined[:, 1])), 6.0) / 2 + 1.0
        ax.set_xlim(center[0] - half_span, center[0] + half_span)
        ax.set_ylim(center[1] - half_span, center[1] + half_span)
        ax.set_aspect("equal", adjustable="box")
        ax.set(
            xlabel="RF azimuth (deg)",
            ylabel="RF elevation (deg)",
            title=(
                f"{session_id}: {roles.get(session_id, 'user-selected')}\n"
                f"Held-out gradient: R²={local_audit.heldout_gradient_r2_vs_constant:.2f}"
            ),
        )
        if row_index == 0:
            ax.legend(frameon=False, fontsize=8)

        ax = axes[row_index, 1]
        ax.scatter(local_units["sampling_trace_deg2"], local_units["raw_trace_deg2"], c=local_units["rf_neighborhood_physical_sd_um"], cmap="viridis", s=28)
        upper = float(np.nanmax(local_units[["sampling_trace_deg2", "raw_trace_deg2"]].to_numpy()))
        ax.plot([0, upper], [0, upper], color="0.4", ls="--", lw=1)
        ax.set(xlabel="LOAO sampling-only trace (deg²)", ylabel="raw covariance trace (deg²)", title=f"Median sampling fraction={100*local_audit.median_sampling_fraction:.1f}%")

        ax = axes[row_index, 2]
        scatter = ax.scatter(local_units["rf_azimuth_deg"], local_units["rf_elevation_deg"], c=local_units["sampling_trace_deg2"], cmap="magma", s=34)
        ax.set_aspect("equal", adjustable="box")
        ax.set(xlabel="RF azimuth (deg)", ylabel="RF elevation (deg)", title="Independent sampling correction")
        fig.colorbar(scatter, ax=ax, label="trace (deg²)")

        ax = axes[row_index, 3]
        scatter = ax.scatter(local_units["rf_azimuth_deg"], local_units["rf_elevation_deg"], c=np.log2(np.maximum(local_units["residual_trace_deg2"], 1e-6)), cmap="cividis", s=34)
        ax.set_aspect("equal", adjustable="box")
        ax.set(xlabel="RF azimuth (deg)", ylabel="RF elevation (deg)", title="Conditional residual scatter")
        fig.colorbar(scatter, ax=ax, label="log₂ residual trace")

    fig.suptitle(
        "Leave-one-animal-out V1 CCF→RF geometry: quadratic fixed-effect map\n"
        "unknown animal translations removed during training; translation added only for display",
        y=.995,
    )
    fig.tight_layout(rect=(0, 0, 1, .97))
    figure_path = output / "Figure_v1_cross_animal_mean_map_support_cases.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    comparison_figure = output / "Figure_v1_cross_animal_model_comparison.png"
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    x = np.arange(len(args.sessions))
    width = .34
    for model_index, model in enumerate(MODELS):
        local = audit.loc[audit["model"].eq(model)].set_index("ecephys_session_id").loc[list(args.sessions)]
        offset = (model_index - .5) * width
        axes[0].bar(x + offset, local["heldout_gradient_r2_vs_constant"], width, label=model)
        axes[1].bar(x + offset, 100 * local["median_sampling_fraction"], width)
        axes[2].bar(x + offset, local["heldout_gradient_rmse_deg"], width)
    labels = [str(value) for value in args.sessions]
    for ax in axes:
        ax.set_xticks(x, labels, rotation=25)
    axes[0].axhline(0, color="0.4", lw=1)
    axes[0].set(ylabel="held-out R² vs session mean", title="Does external geometry predict gradient?")
    axes[1].set(ylabel="median sampling / raw trace (%)", title="Estimated sampling contribution")
    axes[2].set(ylabel="held-out block RMSE (deg)", title="Gradient prediction error")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(comparison_figure, dpi=180, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "status": "concrete-case exploratory checkpoint",
        "training_unit": "six robust physical blocks per animal/session",
        "translation_handling": "session fixed effects removed by within-session centering",
        "heldout_unit": "entire specimen; one specimen per session in this cohort",
        "primary_model": primary_model,
        "sensitivity_model": "affine",
        "ridge": args.ridge,
        "rf_neighborhood_bandwidth_deg": args.rf_bandwidth_deg,
        "sessions": list(args.sessions),
        "outputs": [figure_path.name, comparison_figure.name, "heldout_session_model_audit.csv", "heldout_block_predictions.csv", "heldout_unit_support_decomposition.csv.gz", "heldout_model_coefficients.csv"],
    }
    (output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(figure_path)
    print(comparison_figure)
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
