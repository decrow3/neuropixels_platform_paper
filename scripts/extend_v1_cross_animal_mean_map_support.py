#!/usr/bin/env python3
"""Extend the leave-one-animal-out V1 sampling control across usable sessions."""

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

from scripts.check_v1_cross_animal_mean_map_support import (
    CCF_COLUMNS,
    DEFAULT_INPUT,
    DEFAULT_UNITS,
    MODELS,
    RF_COLUMNS,
    centered_validation,
    fit_fixed_effect_geometry,
    load_population,
    make_block_table,
    predict,
)
from scripts.check_v1_dispersion_support_geometry import support_decomposition


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts" / "v1_absolute_size_dispersion_translation_checkpoint"
DEFAULT_OUTPUT = CHECKPOINT / "cross_animal_mean_map_support_extended"
ORIGINAL_CASES = {
    760345702: "original success",
    719161530: "original typical",
    835479236: "original failure",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--unit-table", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--physical-blocks", type=int, default=6)
    parser.add_argument("--rf-bandwidth-deg", type=float, default=15.0)
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def fit_session(
    session_id: int,
    population: pd.DataFrame,
    blocks: pd.DataFrame,
    model: str,
    ridge: float,
    rf_bandwidth: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    target_units = population.loc[population["ecephys_session_id"].eq(session_id)].copy()
    target_blocks = blocks.loc[blocks["ecephys_session_id"].eq(session_id)].copy()
    specimen_id = int(target_units["specimen_id"].iloc[0])
    training = blocks.loc[blocks["specimen_id"].ne(specimen_id)].copy()
    fit = fit_fixed_effect_geometry(training, model, ridge)

    block_prediction = predict(target_blocks[list(CCF_COLUMNS)].to_numpy(float), fit, model)
    validation = centered_validation(target_blocks[list(RF_COLUMNS)].to_numpy(float), block_prediction)
    unit_prediction = predict(target_units[list(CCF_COLUMNS)].to_numpy(float), fit, model)
    decomposition = support_decomposition(
        target_units["probe_vertical_position"].to_numpy(float),
        target_units[list(RF_COLUMNS)].to_numpy(float),
        unit_prediction,
        rf_bandwidth,
    )

    unit_result = target_units.reset_index(drop=True).copy()
    unit_result["model"] = model
    unit_result["predicted_rf_azimuth_untranslated_deg"] = unit_prediction[:, 0]
    unit_result["predicted_rf_elevation_untranslated_deg"] = unit_prediction[:, 1]
    unit_result = pd.concat([unit_result, decomposition], axis=1)

    block_result = target_blocks.reset_index(drop=True).copy()
    display_translation = (
        block_result[list(RF_COLUMNS)].to_numpy(float) - block_prediction
    ).mean(axis=0)
    block_result["model"] = model
    block_result["predicted_rf_azimuth_display_deg"] = block_prediction[:, 0] + display_translation[0]
    block_result["predicted_rf_elevation_display_deg"] = block_prediction[:, 1] + display_translation[1]

    valid = unit_result.dropna(subset=["raw_trace_deg2", "sampling_trace_deg2"])
    raw = valid["raw_trace_deg2"].to_numpy(float)
    sampling = valid["sampling_trace_deg2"].to_numpy(float)
    physical = valid["rf_neighborhood_physical_sd_um"].to_numpy(float)
    ordered_blocks = target_blocks.sort_values("probe_vertical_position")
    block_ccf = ordered_blocks[list(CCF_COLUMNS)].to_numpy(float)
    ccf_steps = np.sqrt(np.sum(np.diff(block_ccf, axis=0) ** 2, axis=1))
    ccf_span = np.sqrt(np.sum((block_ccf.max(axis=0) - block_ccf.min(axis=0)) ** 2))
    audit = {
        "ecephys_session_id": session_id,
        "specimen_id": specimen_id,
        "model": model,
        "target_units": len(target_units),
        "target_blocks": len(target_blocks),
        "maximum_consecutive_tangential_ccf_step_um": float(np.max(ccf_steps)),
        "tangential_ccf_span_um": float(ccf_span),
        "training_animals": fit["training_sessions"],
        "training_blocks": fit["training_blocks"],
        "training_centered_rmse_deg": fit["training_centered_rmse_deg"],
        **validation,
        "median_raw_trace_deg2": float(np.nanmedian(raw)),
        "median_sampling_trace_deg2": float(np.nanmedian(sampling)),
        "median_sampling_fraction": float(np.nanmedian(sampling / np.maximum(raw, 1e-12))),
        "p90_sampling_fraction": float(np.nanquantile(sampling / np.maximum(raw, 1e-12), .9)),
        "median_residual_trace_deg2": float(np.nanmedian(valid["residual_trace_deg2"])),
        "raw_vs_sampling_trace_rho": float(spearmanr(raw, sampling).statistic),
        "sampling_vs_physical_spread_rho": float(spearmanr(sampling, physical).statistic),
    }
    return unit_result, block_result, audit


def select_cases(audit: pd.DataFrame) -> pd.DataFrame:
    primary = audit.loc[audit["model"].eq("quadratic")].copy()
    wide = audit.pivot(index="ecephys_session_id", columns="model", values="heldout_gradient_r2_vs_constant")
    primary = primary.merge(
        (wide["quadratic"] - wide["affine"]).abs().rename("model_r2_disagreement"),
        on="ecephys_session_id",
    )
    selected: list[dict] = []
    used: set[int] = set()

    def add(role: str, criterion: str, frame: pd.DataFrame, ascending: bool) -> None:
        available = frame.loc[~frame["ecephys_session_id"].isin(used)].sort_values(criterion, ascending=ascending)
        row = available.iloc[0]
        session_id = int(row["ecephys_session_id"])
        used.add(session_id)
        selected.append(
            {
                "ecephys_session_id": session_id,
                "selection_role": role,
                "criterion": criterion,
                "criterion_value": float(row[criterion]),
                "reference_model": "quadratic",
                "selection_method": "algorithmic after complete-cohort fitting",
            }
        )

    add("largest estimated sampling correction", "median_sampling_fraction", primary, False)
    add("strongest held-out gradient", "heldout_gradient_r2_vs_constant", primary, False)
    median_r2 = float(primary["heldout_gradient_r2_vs_constant"].median())
    primary["distance_from_median_r2"] = (primary["heldout_gradient_r2_vs_constant"] - median_r2).abs()
    add("typical held-out gradient", "distance_from_median_r2", primary, True)
    add("worst held-out gradient", "heldout_gradient_r2_vs_constant", primary, True)
    add("largest affine-quadratic disagreement", "model_r2_disagreement", primary, False)
    return pd.DataFrame(selected)


def square_limits(observed: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, float]:
    combined = np.vstack([observed, predicted])
    center = (combined.min(axis=0) + combined.max(axis=0)) / 2
    half = max(float(np.ptp(combined[:, 0])), float(np.ptp(combined[:, 1])), 6.0) / 2 + 1
    return center, half


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output}; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)

    population = load_population(args.input.resolve(), args.unit_table.resolve())
    blocks = make_block_table(population, args.physical_blocks)
    counts = blocks.groupby("ecephys_session_id").size()
    usable = counts.index[counts >= 4].astype(int).tolist()
    excluded = sorted(set(population["ecephys_session_id"].astype(int)) - set(usable))
    population = population.loc[population["ecephys_session_id"].isin(usable)].copy()
    blocks = blocks.loc[blocks["ecephys_session_id"].isin(usable)].copy()

    unit_frames = []
    block_frames = []
    audit_rows = []
    for index, session_id in enumerate(usable, start=1):
        for model in MODELS:
            units, target_blocks, audit = fit_session(
                session_id, population, blocks, model, args.ridge, args.rf_bandwidth_deg
            )
            unit_frames.append(units)
            block_frames.append(target_blocks)
            audit_rows.append(audit)
        if index % 10 == 0:
            print(f"completed {index}/{len(usable)} sessions")

    unit_results = pd.concat(unit_frames, ignore_index=True)
    block_results = pd.concat(block_frames, ignore_index=True)
    audit = pd.DataFrame(audit_rows)
    selection = select_cases(audit)
    unit_results.to_csv(output / "all_session_unit_support_decomposition.csv.gz", index=False, compression="gzip")
    block_results.to_csv(output / "all_session_block_predictions.csv.gz", index=False, compression="gzip")
    audit.to_csv(output / "all_session_model_audit.csv", index=False)
    selection.to_csv(output / "selected_followup_cases.csv", index=False)

    primary = audit.loc[audit["model"].eq("quadratic")].copy()
    paired = audit.pivot(index="ecephys_session_id", columns="model")
    median_r2 = float(primary["heldout_gradient_r2_vs_constant"].median())
    median_sampling = float(primary["median_sampling_fraction"].median())
    positive_fraction = float((primary["heldout_gradient_r2_vs_constant"] > 0).mean())
    sampling_below_10 = float((primary["median_sampling_fraction"] < .1).mean())
    model_rho = float(
        spearmanr(
            paired["heldout_gradient_r2_vs_constant"]["affine"],
            paired["heldout_gradient_r2_vs_constant"]["quadratic"],
        ).statistic
    )

    selected_ids = set(selection["ecephys_session_id"].astype(int))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].hist(primary["heldout_gradient_r2_vs_constant"], bins=16, color="#4477aa", alpha=.85)
    axes[0, 0].axvline(0, color="0.3", lw=1)
    axes[0, 0].axvline(median_r2, color="#bb5566", ls="--", lw=1.5)
    axes[0, 0].set(xlabel="held-out gradient R² vs session mean", ylabel="sessions", title=f"Gradient prediction: median={median_r2:.2f}; {100*positive_fraction:.0f}% > 0")

    axes[0, 1].hist(100 * primary["median_sampling_fraction"], bins=16, color="#228833", alpha=.85)
    axes[0, 1].axvline(100 * median_sampling, color="#bb5566", ls="--", lw=1.5)
    axes[0, 1].set(xlabel="median sampling / raw covariance (%)", ylabel="sessions", title=f"Sampling correction: median={100*median_sampling:.1f}%; {100*sampling_below_10:.0f}% < 10%")

    axes[1, 0].scatter(primary["heldout_gradient_r2_vs_constant"], 100 * primary["median_sampling_fraction"], color="0.65", s=30)
    for row in primary.itertuples():
        if int(row.ecephys_session_id) in selected_ids or int(row.ecephys_session_id) in ORIGINAL_CASES:
            axes[1, 0].annotate(str(int(row.ecephys_session_id)), (row.heldout_gradient_r2_vs_constant, 100 * row.median_sampling_fraction), fontsize=7)
    axes[1, 0].axvline(0, color="0.4", lw=1)
    axes[1, 0].set(xlabel="held-out gradient R²", ylabel="median sampling contribution (%)", title="Map validity versus correction magnitude")

    affine_r2 = paired["heldout_gradient_r2_vs_constant"]["affine"]
    quadratic_r2 = paired["heldout_gradient_r2_vs_constant"]["quadratic"]
    axes[1, 1].scatter(affine_r2, quadratic_r2, color="#aa4499", alpha=.75)
    limits = [float(min(affine_r2.min(), quadratic_r2.min())), float(max(affine_r2.max(), quadratic_r2.max()))]
    axes[1, 1].plot(limits, limits, color="0.4", ls="--", lw=1)
    axes[1, 1].set(xlabel="affine held-out R²", ylabel="quadratic held-out R²", title=f"Model agreement: Spearman ρ={model_rho:.2f}")
    fig.suptitle(f"Leave-one-animal-out V1 mean-map support control across {len(usable)} sessions")
    fig.tight_layout()
    summary_figure = output / "Figure_v1_cross_animal_support_population_summary.png"
    fig.savefig(summary_figure, dpi=190, bbox_inches="tight")
    plt.close(fig)

    selected_primary = selection.merge(primary, on="ecephys_session_id", how="left")
    fig, axes = plt.subplots(len(selection), 3, figsize=(13, 3.6 * len(selection)))
    for row_index, selected in enumerate(selection.itertuples()):
        session_id = int(selected.ecephys_session_id)
        local_blocks = block_results.loc[
            block_results["ecephys_session_id"].eq(session_id) & block_results["model"].eq("quadratic")
        ].sort_values("probe_vertical_position")
        local_units = unit_results.loc[
            unit_results["ecephys_session_id"].eq(session_id) & unit_results["model"].eq("quadratic")
        ]
        local_audit = primary.loc[primary["ecephys_session_id"].eq(session_id)].iloc[0]
        observed = local_blocks[list(RF_COLUMNS)].to_numpy(float)
        predicted = local_blocks[["predicted_rf_azimuth_display_deg", "predicted_rf_elevation_display_deg"]].to_numpy(float)
        ax = axes[row_index, 0]
        ax.scatter(observed[:, 0], observed[:, 1], s=48, label="observed")
        ax.scatter(predicted[:, 0], predicted[:, 1], marker="x", s=55, label="LOAO prediction")
        for first, second in zip(observed, predicted):
            ax.plot([first[0], second[0]], [first[1], second[1]], color="0.65", lw=.8)
        center, half = square_limits(observed, predicted)
        ax.set(xlim=(center[0]-half, center[0]+half), ylim=(center[1]-half, center[1]+half), xlabel="RF azimuth", ylabel="RF elevation", title=f"{session_id}: {selected.selection_role}\nGradient R²={local_audit.heldout_gradient_r2_vs_constant:.2f}")
        ax.set_aspect("equal", adjustable="box")
        if row_index == 0:
            ax.legend(frameon=False, fontsize=8)

        ax = axes[row_index, 1]
        ax.scatter(local_units["sampling_trace_deg2"], local_units["raw_trace_deg2"], c=local_units["rf_neighborhood_physical_sd_um"], cmap="viridis", s=25)
        upper = float(np.nanmax(local_units[["sampling_trace_deg2", "raw_trace_deg2"]].to_numpy()))
        ax.plot([0, upper], [0, upper], color="0.4", ls="--", lw=1)
        ax.set(xlabel="sampling-only trace (deg²)", ylabel="raw trace (deg²)", title=f"Median correction={100*local_audit.median_sampling_fraction:.1f}%")

        ax = axes[row_index, 2]
        scatter = ax.scatter(local_units["rf_azimuth_deg"], local_units["rf_elevation_deg"], c=np.log2(np.maximum(local_units["residual_trace_deg2"], 1e-6)), cmap="cividis", s=30)
        ax.set(xlabel="RF azimuth", ylabel="RF elevation", title="Conditional residual scatter")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(scatter, ax=ax, label="log₂ residual trace")
    fig.suptitle("Algorithmically selected wider-set cases", y=.998)
    fig.tight_layout(rect=(0, 0, 1, .99))
    cases_figure = output / "Figure_v1_cross_animal_support_selected_cases.png"
    fig.savefig(cases_figure, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "usable_sessions": len(usable),
        "excluded_sessions": excluded,
        "primary_model": "quadratic",
        "median_heldout_gradient_r2": median_r2,
        "fraction_gradient_r2_positive": positive_fraction,
        "median_sampling_fraction": median_sampling,
        "fraction_sampling_below_10_percent": sampling_below_10,
        "affine_quadratic_r2_spearman": model_rho,
        "physical_blocks": args.physical_blocks,
        "rf_neighborhood_bandwidth_deg": args.rf_bandwidth_deg,
        "ridge": args.ridge,
        "outputs": [summary_figure.name, cases_figure.name, "all_session_model_audit.csv", "selected_followup_cases.csv", "all_session_block_predictions.csv.gz", "all_session_unit_support_decomposition.csv.gz"],
    }
    (output / "population_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(summary_figure)
    print(cases_figure)
    print(json.dumps(summary, indent=2))
    print(selected_primary[["ecephys_session_id", "selection_role", "criterion", "criterion_value", "heldout_gradient_r2_vs_constant", "median_sampling_fraction"]].to_string(index=False))


if __name__ == "__main__":
    main()
