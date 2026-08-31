#!/usr/bin/env python3
"""Refit the timescale variance-components model under three eligibility rules."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_timescale_inclusion_variability import (  # noqa: E402
    HVA_GROUPS,
    build_allen_base,
    build_mouse_base,
    select_allen,
    select_mouse,
)
from scripts.figure3_full_cell_multilevel_extension import (  # noqa: E402
    aggregate_point_fits,
    bootstrap_timescale_v1,
)
from scripts.figure3_full_cell_multilevel_model import (  # noqa: E402
    COMPONENTS,
    bootstrap_models,
    fit_once,
    ratio_summary,
)


OUTPUT = ROOT / (
    "artifacts/figure3/07_big_picture_concrete_first/"
    "timescale_population_sensitivity"
)
REFERENCE_POINT = ROOT / (
    "artifacts/figure3/07_big_picture_concrete_first/full_cell_model/"
    "metric_extension_cortical_hvas/full_cell_variance_components.csv"
)
REFERENCE_BOOTSTRAP = ROOT / (
    "artifacts/figure3/07_big_picture_concrete_first/full_cell_model/"
    "metric_extension_cortical_hvas/full_cell_session_block_bootstrap.csv"
)
REFERENCE_RATIOS = ROOT / (
    "artifacts/figure3/07_big_picture_concrete_first/full_cell_model/"
    "metric_extension_cortical_hvas/full_cell_variance_ratios.csv"
)
SCENARIOS = ["Primary", "Ten complete draws", "Fit error <10 ms"]
COLORS = {
    "Primary": "#4c78a8",
    "Ten complete draws": "#8b6fb1",
    "Fit error <10 ms": "#d17b48",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-repetitions", type=int, default=100)
    parser.add_argument("--maxiter", type=int, default=3000)
    parser.add_argument(
        "--seed", type=int, default=20260827,
        help="Validated extension seed; the timescale streams are children 3 and 4.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    return parser.parse_args()


def build_scenarios() -> dict[str, dict[str, pd.DataFrame]]:
    _, bridge, _ = build_mouse_base()
    allen = build_allen_base("common_qc")
    primary_hva = select_allen(allen, groups=HVA_GROUPS)
    return {
        "Primary": {
            "Within-V1": select_mouse(bridge, primary_population=True),
            "Post-V1": primary_hva,
        },
        "Ten complete draws": {
            "Within-V1": select_mouse(
                bridge, primary_population=True, complete_neuron_draws=True
            ),
            "Post-V1": primary_hva,
        },
        "Fit error <10 ms": {
            "Within-V1": select_mouse(
                bridge, primary_population=True, error_lt=10
            ),
            "Post-V1": select_allen(allen, groups=HVA_GROUPS, error_lt=10),
        },
    }


def point_fit_scenario(
    scenario: str, units: dict[str, pd.DataFrame], maxiter: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_rows = []
    diagnostic_rows = []
    for dataset in ["Within-V1", "Post-V1"]:
        local = units[dataset]
        draw_ids = sorted(local["subsample"].unique()) if dataset == "Within-V1" else [-1]
        for draw_id in draw_ids:
            draw = local.loc[local["subsample"].eq(draw_id)] if dataset == "Within-V1" else local
            fit, diagnostics = fit_once(
                draw, methods=["powell", "nm", "lbfgs"], maxiter=maxiter
            )
            fit.update({
                "metric": "Response timescale (ms)", "dataset": dataset,
                "draw_id": draw_id, "scenario": scenario,
            })
            diagnostics.insert(0, "draw_id", draw_id)
            diagnostics.insert(0, "dataset", dataset)
            diagnostics.insert(0, "scenario", scenario)
            fit_rows.append(fit)
            diagnostic_rows.append(diagnostics)
            print(f"Point fit: {scenario}, {dataset}, draw={draw_id}", flush=True)
    fits = pd.DataFrame(fit_rows)
    point = aggregate_point_fits(fits)
    point.insert(0, "scenario", scenario)
    return point, pd.concat(diagnostic_rows, ignore_index=True)


def bootstrap_scenario(
    scenario: str,
    units: dict[str, pd.DataFrame],
    point: pd.DataFrame,
    *, repetitions: int, maxiter: int,
    v1_seed: np.random.SeedSequence, hva_seed: np.random.SeedSequence,
    reference_hva: pd.DataFrame | None = None,
) -> pd.DataFrame:
    outputs = []
    for dataset, seed in zip(["Within-V1", "Post-V1"], [v1_seed, hva_seed]):
        if dataset == "Post-V1" and reference_hva is not None:
            result = reference_hva.copy()
            result["scenario"] = scenario
            outputs.append(result)
            continue
        local = units[dataset]
        preferred = str(point.loc[point["dataset"].eq(dataset), "optimizer"].iloc[0])
        rng = np.random.default_rng(seed)
        if dataset == "Within-V1":
            result = bootstrap_timescale_v1(
                local.rename(columns={"subsample": "draw_id"}),
                repetitions=repetitions, preferred_method=preferred,
                maxiter=maxiter, rng=rng,
            )
        else:
            result = bootstrap_models(
                local, repetitions=repetitions, preferred_method=preferred,
                maxiter=maxiter, rng=rng, dataset=dataset,
            )
        result["scenario"] = scenario
        outputs.append(result)
    return pd.concat(outputs, ignore_index=True)


def primary_validation(point: pd.DataFrame) -> pd.DataFrame:
    reference = pd.read_csv(REFERENCE_POINT)
    reference = reference.loc[
        reference["metric"].eq("Response timescale (ms)"),
        ["dataset", *[f"variance_{name}" for name in COMPONENTS], "variance_total_structured"],
    ]
    primary = point.loc[
        point["scenario"].eq("Primary"),
        ["dataset", *[f"variance_{name}" for name in COMPONENTS], "variance_total_structured"],
    ]
    audit = primary.merge(reference, on="dataset", suffixes=("_refit", "_reference"))
    for column in [*[f"variance_{name}" for name in COMPONENTS], "variance_total_structured"]:
        audit[f"absolute_error_{column}"] = (
            audit[f"{column}_refit"] - audit[f"{column}_reference"]
        ).abs()
    errors = audit.filter(like="absolute_error_").to_numpy(float)
    if np.nanmax(errors) > 1e-5:
        raise ValueError(f"Primary point refit does not reproduce reference:\n{audit}")
    return audit


def summarize_ratios(
    points: pd.DataFrame, bootstraps: pd.DataFrame,
) -> pd.DataFrame:
    outputs = []
    for scenario in SCENARIOS:
        local_point = points.loc[points["scenario"].eq(scenario)].copy()
        local_boot = bootstraps.loc[bootstraps["scenario"].eq(scenario)].copy()
        ratios = ratio_summary(local_point, local_boot)
        ratios.insert(0, "scenario", scenario)
        outputs.append(ratios)
    return pd.concat(outputs, ignore_index=True)


def validate_primary_ratios(ratios: pd.DataFrame) -> pd.DataFrame:
    reference = pd.read_csv(REFERENCE_RATIOS)
    reference = reference.loc[
        reference["metric"].eq("Response timescale (ms)")
    ].drop(columns="metric")
    primary = ratios.loc[ratios["scenario"].eq("Primary")].drop(columns="scenario")
    audit = primary.merge(reference, on="contrast", suffixes=("_refit", "_reference"))
    numeric = [
        column for column in primary.columns
        if column != "contrast" and pd.api.types.is_numeric_dtype(primary[column])
    ]
    for column in numeric:
        audit[f"absolute_error_{column}"] = (
            audit[f"{column}_refit"] - audit[f"{column}_reference"]
        ).abs()
    # Some boundary-ratio upper limits are ~1e8, so CSV round-tripping can
    # change their last decimal by ~3e-8 while all underlying draws agree.
    if np.nanmax(audit.filter(like="absolute_error_").to_numpy(float)) > 1e-6:
        raise ValueError("Primary bootstrap ratios do not reproduce the validated reference")
    return audit


def render(points: pd.DataFrame, ratios: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), constrained_layout=True)
    x = np.arange(len(SCENARIOS))
    width = 0.34
    for offset, dataset, label, color in [
        (-width / 2, "Within-V1", "V1 locations", "#7564a8"),
        (width / 2, "Post-V1", "Cortical HVAs", "#555555"),
    ]:
        values = [
            points.loc[
                points["scenario"].eq(scenario) & points["dataset"].eq(dataset),
                "variance_total_structured",
            ].iloc[0]
            for scenario in SCENARIOS
        ]
        axes[0].bar(x + offset, values, width, color=color, alpha=0.86, label=label)
    axes[0].set_xticks(x, ["Primary", "10 complete\ndraws", "Error <10 ms"])
    axes[0].set_ylabel("Total structured variance (ms²)")
    axes[0].set_title("Point variance components", loc="left", fontweight="bold")
    axes[0].legend(frameon=False)

    total = ratios.loc[ratios["contrast"].eq("total_structured")].set_index("scenario")
    differences = total.loc[SCENARIOS, "bootstrap_median_difference"].to_numpy(float)
    diff_low = total.loc[SCENARIOS, "difference_ci_low"].to_numpy(float)
    diff_high = total.loc[SCENARIOS, "difference_ci_high"].to_numpy(float)
    for index, scenario in enumerate(SCENARIOS):
        axes[1].errorbar(
            differences[index], index,
            xerr=[[differences[index] - diff_low[index]], [diff_high[index] - differences[index]]],
            fmt="o", markersize=8, capsize=3, color=COLORS[scenario], linewidth=1.8,
        )
    axes[1].axvline(0, color="#777777", linestyle="--", linewidth=1)
    axes[1].set_yticks(np.arange(len(SCENARIOS)), ["Primary", "10 complete draws", "Error <10 ms"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("HVA − V1 structured variance (ms²)")
    axes[1].set_title("Session-block bootstrap", loc="left", fontweight="bold")

    point_ratios = total.loc[SCENARIOS, "hva_to_v1_variance_ratio"].to_numpy(float)
    for index, (scenario, value) in enumerate(zip(SCENARIOS, point_ratios)):
        axes[2].scatter(value, index, s=65, color=COLORS[scenario], zorder=3)
        axes[2].text(value * 1.06, index, f"{value:.2f}×", va="center", fontsize=9)
    axes[2].axvline(1, color="#777777", linestyle="--", linewidth=1)
    axes[2].set_xscale("log")
    axes[2].set_yticks(np.arange(len(SCENARIOS)), ["Primary", "10 complete draws", "Error <10 ms"])
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Point HVA/V1 structured-variance ratio")
    axes[2].set_title("Point ratio", loc="left", fontweight="bold")

    for ax in axes:
        ax.grid(axis="x" if ax is not axes[0] else "y", color="#dfe3e6", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Response-timescale variance: population and fit-quality sensitivity",
        fontsize=14, fontweight="bold",
    )
    figure.savefig(output, dpi=220, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scenarios = build_scenarios()
    points = []
    diagnostics = []
    scenario_points = {}
    for scenario in SCENARIOS:
        point, diagnostic = point_fit_scenario(scenario, scenarios[scenario], args.maxiter)
        points.append(point)
        diagnostics.append(diagnostic)
        scenario_points[scenario] = point
    point_table = pd.concat(points, ignore_index=True)
    validation = primary_validation(point_table)

    reference_bootstrap = pd.read_csv(REFERENCE_BOOTSTRAP)
    reference_bootstrap = reference_bootstrap.loc[
        reference_bootstrap["metric"].eq("Response timescale (ms)")
    ].drop(columns="metric")
    if args.bootstrap_repetitions != 100:
        raise ValueError("Paired sensitivity currently requires the validated 100 resamples")
    reference_bootstrap["scenario"] = "Primary"
    reference_hva = reference_bootstrap.loc[
        reference_bootstrap["dataset"].eq("Post-V1")
    ].drop(columns="scenario")
    bootstraps = [reference_bootstrap]
    for scenario in SCENARIOS[1:]:
        # Recreate the exact two timescale RNG streams from the validated metric
        # extension. The first two child streams belonged to F1/F0.
        children = np.random.SeedSequence(args.seed).spawn(4)
        print(f"Starting paired bootstrap: {scenario}", flush=True)
        bootstraps.append(bootstrap_scenario(
            scenario, scenarios[scenario], scenario_points[scenario],
            repetitions=args.bootstrap_repetitions, maxiter=args.maxiter,
            v1_seed=children[2], hva_seed=children[3],
            reference_hva=reference_hva if scenario == "Ten complete draws" else None,
        ))
    bootstrap_table = pd.concat(bootstraps, ignore_index=True)
    ratios = summarize_ratios(point_table, bootstrap_table)
    ratio_validation = validate_primary_ratios(ratios)

    input_rows = []
    for scenario in SCENARIOS:
        for dataset, table in scenarios[scenario].items():
            local = table.copy()
            local.insert(0, "scenario", scenario)
            local.insert(1, "dataset", dataset)
            input_rows.append(local)
    pd.concat(input_rows, ignore_index=True).to_csv(
        output / "timescale_population_sensitivity_input.csv", index=False
    )
    point_table.to_csv(output / "timescale_population_sensitivity_point.csv", index=False)
    pd.concat(diagnostics, ignore_index=True).to_csv(
        output / "timescale_population_sensitivity_optimizer_diagnostics.csv", index=False
    )
    bootstrap_table.to_csv(
        output / "timescale_population_sensitivity_bootstrap.csv", index=False
    )
    ratios.to_csv(output / "timescale_population_sensitivity_ratios.csv", index=False)
    validation.to_csv(output / "primary_refit_validation.csv", index=False)
    ratio_validation.to_csv(output / "primary_bootstrap_validation.csv", index=False)
    render(points=point_table, ratios=ratios, output=output / "Figure_timescale_population_sensitivity.png")
    print("\nPOINT\n", point_table.to_string(index=False))
    print("\nRATIOS\n", ratios.to_string(index=False))


if __name__ == "__main__":
    main()
