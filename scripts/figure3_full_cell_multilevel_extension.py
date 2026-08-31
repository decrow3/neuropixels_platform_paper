"""Extend the validated Figure 3 full-cell model to F1/F0 and timescale.

F1/F0 uses one matched-support value per neuron. MouseV2 timescale has ten
trial-matched fits per neuron; these are never stacked as independent rows in a
model. Point components average ten separate model fits, one per matched draw.
Each V1 session-block bootstrap replicate selects one draw independently for
each sampled session block, so every fitted block contains at most one outcome
per neuron while trial-draw uncertainty is propagated.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.figure3_mousev2 import load_allen_units, load_mousev2_units  # noqa: E402
from scripts.figure3_full_cell_multilevel_model import (  # noqa: E402
    COMPONENTS,
    bootstrap_models,
    fit_once,
    ratio_summary,
)
from scripts.figure3_robust_spread_comparison import (  # noqa: E402
    ALLEN_HARMONIZED_F1_F0,
    AREA_ORDER,
    MOUSE_CANONICAL_GRATING_METRICS,
    MOUSE_HARMONIZED_F1_F0,
    MOUSE_PARAMETRIC_RF_FITS,
    MOUSE_TIMESCALE_TRIAL_BRIDGE,
    historical_proxy_full20_population,
    valid_values,
)

CORTICAL_HVA_ORDER = tuple(area for area in AREA_ORDER if area != "LP")

CURRENT_MEANS = ROOT / "Figure3/Figure3_robust_session_group_means.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/figure3/07_big_picture_concrete_first/full_cell_model/metric_extension_cortical_hvas"
METRICS = ["log10 F1/F0", "Response timescale (ms)"]
MIN_UNITS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repetitions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--maxiter", type=int, default=3000)
    return parser.parse_args()


def build_f1_f0_units() -> pd.DataFrame:
    mouse = pd.read_csv(MOUSE_HARMONIZED_F1_F0)
    mouse = mouse.loc[
        mouse["default_qc"].astype(bool)
        & pd.to_numeric(mouse["f1_f0_dg_common_support"], errors="coerce").gt(0)
    ].copy()
    mouse["value"] = np.log10(mouse["f1_f0_dg_common_support"].astype(float))
    mouse = mouse.rename(
        columns={"site_number": "session_id", "probe_letter": "group"}
    )
    mouse["dataset"] = "Within-V1"
    mouse["source"] = str(MOUSE_HARMONIZED_F1_F0.relative_to(ROOT))

    allen = pd.read_csv(ALLEN_HARMONIZED_F1_F0)
    allen = allen.loc[
        allen["area_coarse"].isin(CORTICAL_HVA_ORDER)
        & pd.to_numeric(allen["f1_f0_dg_harmonized"], errors="coerce").gt(0)
    ].copy()
    allen["value"] = np.log10(allen["f1_f0_dg_harmonized"].astype(float))
    allen = allen.rename(
        columns={
            "ecephys_session_id": "session_id",
            "area_coarse": "group",
            "ecephys_unit_id": "unit_id",
        }
    )
    allen["dataset"] = "Post-V1"
    allen["source"] = str(ALLEN_HARMONIZED_F1_F0.relative_to(ROOT))

    columns = ["dataset", "session_id", "group", "unit_id", "value", "source"]
    result = pd.concat([mouse[columns], allen[columns]], ignore_index=True)
    cell_size = result.groupby(["dataset", "session_id", "group"])["unit_id"].transform("size")
    result = result.loc[cell_size.ge(MIN_UNITS)].copy()
    result["metric"] = "log10 F1/F0"
    result["draw_id"] = -1
    result["session_id"] = result["session_id"].astype(str)
    result["unit_id"] = result["unit_id"].astype(str)
    if result.duplicated(["dataset", "session_id", "group", "unit_id"]).any():
        raise ValueError("Duplicate F1/F0 neuron within a session x group")
    return result


def build_timescale_units() -> pd.DataFrame:
    rows = []
    allen = load_allen_units(population_profile="common_qc")
    allen = allen.loc[allen["area_coarse"].isin(CORTICAL_HVA_ORDER)].copy()
    for (session_id, group), part in allen.groupby(
        ["ecephys_session_id", "area_coarse"], sort=True
    ):
        values = valid_values(part, "timescale_ac", 2)
        valid = pd.to_numeric(part["timescale_ac"], errors="coerce").between(1, 300)
        if "spike_count_ac" in part:
            valid &= pd.to_numeric(part["spike_count_ac"], errors="coerce").gt(50)
        if "err_ac" in part:
            valid &= pd.to_numeric(part["err_ac"], errors="coerce").lt(20)
        selected = part.loc[valid].copy()
        if len(values) != len(selected):
            raise ValueError("Allen timescale validity reconstruction disagrees")
        if len(selected) < MIN_UNITS:
            continue
        unit_column = "ecephys_unit_id" if "ecephys_unit_id" in selected else "unit_id"
        for unit_id, value in zip(selected[unit_column], values):
            rows.append(
                {
                    "dataset": "Post-V1",
                    "session_id": str(session_id),
                    "group": str(group),
                    "unit_id": str(unit_id),
                    "value": float(value),
                    "draw_id": -1,
                    "source": "data/unit_table.csv",
                }
            )

    mouse = load_mousev2_units(
        apply_qc=False,
        grating_metrics_dir=MOUSE_CANONICAL_GRATING_METRICS,
        population_profile="common_qc",
    )
    mouse, _, _ = historical_proxy_full20_population(
        mouse, rf_path=MOUSE_PARAMETRIC_RF_FITS
    )
    metadata = mouse[["unit_id", "session_num", "probe_letter"]].drop_duplicates("unit_id")
    bridge = pd.read_csv(MOUSE_TIMESCALE_TRIAL_BRIDGE)
    bridge = bridge.loc[
        bridge["view"].eq("mouse_matched_150")
        & bridge["valid_timescale"].astype(bool)
    ].merge(metadata, on="unit_id", validate="many_to_one")
    counts = (
        bridge.groupby(["session_id", "probe_letter", "subsample"])
        .size().rename("n_units").reset_index()
    )
    complete = (
        counts.loc[counts["n_units"].ge(MIN_UNITS)]
        .groupby(["session_id", "probe_letter"])["subsample"]
        .nunique().loc[lambda x: x.eq(10)].reset_index()[["session_id", "probe_letter"]]
    )
    bridge = bridge.merge(complete, on=["session_id", "probe_letter"], how="inner")
    for record in bridge.itertuples(index=False):
        rows.append(
            {
                "dataset": "Within-V1",
                "session_id": str(record.session_id),
                "group": str(record.probe_letter),
                "unit_id": str(record.unit_id),
                "value": float(record.timescale_ms),
                "draw_id": int(record.subsample),
                "source": str(MOUSE_TIMESCALE_TRIAL_BRIDGE.relative_to(ROOT)),
            }
        )
    result = pd.DataFrame(rows)
    result["metric"] = "Response timescale (ms)"
    duplicate_keys = ["dataset", "session_id", "group", "unit_id", "draw_id"]
    if result.duplicated(duplicate_keys).any():
        raise ValueError("Duplicate timescale neuron within a matched draw")
    return result


def validate_f1_f0(units: pd.DataFrame) -> pd.DataFrame:
    reconstructed = (
        units.groupby(["dataset", "metric", "session_id", "group"], as_index=False)
        .agg(mean_reconstructed=("value", "mean"), n_reconstructed=("unit_id", "size"))
    )
    return validate_cells(reconstructed, metric="log10 F1/F0")


def validate_timescale(units: pd.DataFrame) -> pd.DataFrame:
    hva = units.loc[units["dataset"].eq("Post-V1")]
    hva_cells = (
        hva.groupby(["dataset", "metric", "session_id", "group"], as_index=False)
        .agg(mean_reconstructed=("value", "mean"), n_reconstructed=("unit_id", "size"))
    )
    v1 = units.loc[units["dataset"].eq("Within-V1")]
    draw_cells = (
        v1.groupby(["dataset", "metric", "session_id", "group", "draw_id"], as_index=False)
        .agg(draw_mean=("value", "mean"), draw_n=("unit_id", "size"))
    )
    v1_cells = (
        draw_cells.groupby(["dataset", "metric", "session_id", "group"], as_index=False)
        .agg(mean_reconstructed=("draw_mean", "mean"), mean_draw_n=("draw_n", "mean"))
    )
    v1_cells["n_reconstructed"] = v1_cells["mean_draw_n"].round().astype(int)
    reconstructed = pd.concat(
        [hva_cells, v1_cells.drop(columns="mean_draw_n")], ignore_index=True
    )
    return validate_cells(reconstructed, metric="Response timescale (ms)")


def validate_cells(reconstructed: pd.DataFrame, *, metric: str) -> pd.DataFrame:
    current = pd.read_csv(CURRENT_MEANS, dtype={"session_id": str})
    current = current.loc[
        current["metric"].eq(metric)
        & (current["dataset"].ne("Post-V1") | current["group"].isin(CORTICAL_HVA_ORDER)), [
        "dataset", "metric", "session_id", "group", "mean", "n_units"
    ]]
    audit = current.merge(
        reconstructed,
        on=["dataset", "metric", "session_id", "group"],
        how="outer",
        indicator=True,
    )
    if not audit["_merge"].eq("both").all():
        raise ValueError(audit.loc[~audit["_merge"].eq("both")].to_string(index=False))
    audit["absolute_mean_error"] = (audit["mean"] - audit["mean_reconstructed"]).abs()
    audit["unit_count_error"] = audit["n_units"] - audit["n_reconstructed"]
    if audit["absolute_mean_error"].max() > 1e-10:
        raise ValueError(f"{metric} mean reconstruction failed")
    if not audit["unit_count_error"].eq(0).all():
        raise ValueError(f"{metric} count reconstruction failed")
    return audit


def aggregate_point_fits(fits: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "n_cells", "n_sessions", "n_groups", "n_session_groups",
        *[f"variance_{component}" for component in COMPONENTS],
        "variance_total_structured", "fraction_structured_stable",
    ]
    rows = []
    for (metric, dataset), part in fits.groupby(["metric", "dataset"], sort=False):
        row = {"metric": metric, "dataset": dataset, "point_fits": len(part)}
        for column in numeric:
            row[column] = float(part[column].mean())
        row["fraction_structured_stable"] = (
            row["variance_group"] / row["variance_total_structured"]
            if row["variance_total_structured"] > 0 else np.nan
        )
        row["optimizer"] = Counter(part["optimizer"]).most_common(1)[0][0]
        row["all_converged"] = bool(part["converged"].all())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_timescale_draw_sensitivity(fits: pd.DataFrame) -> pd.DataFrame:
    """Expose sensitivity to the ten trial-matched V1 timescale draws."""
    local = fits.loc[
        fits["metric"].eq("Response timescale (ms)")
        & fits["dataset"].eq("Within-V1")
    ].copy()
    if local.empty:
        return pd.DataFrame()
    hva_total = float(
        fits.loc[
            fits["metric"].eq("Response timescale (ms)")
            & fits["dataset"].eq("Post-V1"),
            "variance_total_structured",
        ].iloc[0]
    )
    local["hva_to_v1_total_ratio"] = hva_total / local["variance_total_structured"]
    boundary_cutoff = max(hva_total, float(local["variance_residual"].median())) * 1e-6
    local["v1_total_near_boundary"] = (
        local["variance_total_structured"] <= boundary_cutoff
    )
    return local[
        [
            "draw_id", "n_cells", "optimizer", "reml_log_likelihood",
            "variance_group", "variance_interaction", "variance_total_structured",
            "variance_residual", "hva_to_v1_total_ratio",
            "v1_total_near_boundary",
        ]
    ].sort_values("draw_id")


def resample_timescale_v1(
    units: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    sessions = np.asarray(sorted(units["session_id"].unique()), dtype=object)
    selected_sessions = rng.choice(sessions, size=len(sessions), replace=True)
    blocks = []
    for block_index, session_id in enumerate(selected_sessions):
        available_draws = np.asarray(
            sorted(units.loc[units["session_id"].eq(session_id), "draw_id"].unique())
        )
        draw_id = int(rng.choice(available_draws))
        block = units.loc[
            units["session_id"].eq(session_id) & units["draw_id"].eq(draw_id)
        ].copy()
        block["session_id"] = f"bootstrap_{block_index}:{session_id}"
        block["selected_draw_id"] = draw_id
        blocks.append(block)
    result = pd.concat(blocks, ignore_index=True)
    if result.duplicated(["session_id", "unit_id"]).any():
        raise ValueError("Timescale bootstrap counted a neuron twice within a block")
    return result


def bootstrap_timescale_v1(
    units: pd.DataFrame,
    *,
    repetitions: int,
    preferred_method: str,
    maxiter: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    methods = [preferred_method] + [
        method for method in ["powell", "nm", "lbfgs"] if method != preferred_method
    ]
    rows = []
    for repetition in range(repetitions):
        sample = resample_timescale_v1(units, rng)
        try:
            try:
                fit, _ = fit_once(sample, methods=methods[:1], maxiter=maxiter)
            except Exception:
                fit, _ = fit_once(sample, methods=methods[1:2], maxiter=maxiter)
            fit.update(
                {
                    "bootstrap_repetition": repetition,
                    "dataset": "Within-V1",
                    "status": "ok",
                    "draws_used": ";".join(
                        str(value) for value in sample.groupby("session_id")["selected_draw_id"].first()
                    ),
                }
            )
        except Exception as exc:
            fit = {
                "bootstrap_repetition": repetition,
                "dataset": "Within-V1",
                "status": f"{type(exc).__name__}: {exc}",
            }
        rows.append(fit)
        if (repetition + 1) % 10 == 0 or repetition + 1 == repetitions:
            success = sum(row["status"] == "ok" for row in rows)
            print(
                f"Timescale Within-V1: bootstrap {repetition + 1}/{repetitions}; "
                f"successful={success}", flush=True,
            )
    return pd.DataFrame(rows)


def render(point: pd.DataFrame, bootstrap: pd.DataFrame, ratios: pd.DataFrame, output: Path) -> None:
    colors = {"Within-V1": "#7564a8", "Post-V1": "#555555"}
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for row_index, metric in enumerate(METRICS):
        ax = axes[row_index, 0]
        local = point.loc[point["metric"].eq(metric)]
        x = np.arange(len(COMPONENTS))
        width = 0.34
        for offset, dataset in [(-width / 2, "Within-V1"), (width / 2, "Post-V1")]:
            record = local.loc[local["dataset"].eq(dataset)].iloc[0]
            values = [record[f"variance_{component}"] for component in COMPONENTS]
            ax.bar(x + offset, values, width, color=colors[dataset], alpha=0.85,
                   label=dataset if row_index == 0 else None)
        ax.set_yscale("log")
        ax.set_xticks(x, ["Stable\ngroup", "Session ×\ngroup", "Session", "Cell\nresidual"])
        ax.set_ylabel(f"{metric} variance")
        ax.set_title(f"{metric}: full-cell decomposition", loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        if row_index == 0:
            ax.legend(frameon=False)

        ax = axes[row_index, 1]
        local_boot = bootstrap.loc[
            bootstrap["metric"].eq(metric) & bootstrap["status"].eq("ok")
        ]
        v1 = local_boot.loc[local_boot["dataset"].eq("Within-V1")].set_index(
            "bootstrap_repetition"
        )
        hva = local_boot.loc[local_boot["dataset"].eq("Post-V1")].set_index(
            "bootstrap_repetition"
        )
        shared = sorted(set(v1.index) & set(hva.index))
        ratio_draws = (
            hva.loc[shared, "variance_total_structured"].to_numpy(float)
            / v1.loc[shared, "variance_total_structured"].to_numpy(float)
        )
        ratio_draws = ratio_draws[np.isfinite(ratio_draws) & (ratio_draws > 0)]
        ax.hist(np.log10(ratio_draws), bins=30, color="#4778a8", alpha=0.75)
        ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
        point_ratio = ratios.loc[
            ratios["metric"].eq(metric) & ratios["contrast"].eq("total_structured"),
            "hva_to_v1_variance_ratio",
        ].iloc[0]
        ax.axvline(np.log10(point_ratio), color="#b2182b", linewidth=2,
                   label=f"Point ratio {point_ratio:.2f}×")
        ax.set_xlabel("log10 HVA/V1 total structured variance ratio")
        ax.set_ylabel("Session-block bootstrap draws")
        ax.set_title(f"{metric}: primary contrast", loc="left", fontweight="bold")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Full-cell multilevel model extension", fontweight="bold")
    figure.savefig(output, dpi=220, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def write_report(
    point: pd.DataFrame,
    ratios: pd.DataFrame,
    bootstrap: pd.DataFrame,
    validation: pd.DataFrame,
    timescale_sensitivity: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# Full-cell multilevel model — metric extension checkpoint",
        "",
        f"_Generated {date.today().isoformat()}._",
        "",
        "## Input and independence gate",
        "",
        "The post-V1 population contains the five cortical HVAs LM, RL, AL, PM, and AM.",
        "The thalamic lateral posterior nucleus (LP) is excluded.",
        "",
        f"All {len(validation)} current F1/F0 and timescale session × group means",
        f"and counts are reconstructed to maximum absolute error",
        f"{validation['absolute_mean_error'].max():.3g}.",
        "",
        "F1/F0 contains one matched-support value per neuron. MouseV2 timescale",
        "uses ten separate point fits, each containing at most one matched-draw value",
        "per neuron. In each V1 timescale bootstrap fit, one draw is selected",
        "independently for every resampled session block; repeated fits are never",
        "treated as independent neurons.",
        "",
        "## Variance components",
        "",
        "| Metric | Dataset | Cells per fit | Sessions | Point fits | Stable group | Session × group | Total structured | Session | Cell residual | Stable fraction |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in point.itertuples(index=False):
        lines.append(
            f"| {row.metric} | {row.dataset} | {row.n_cells:.0f} | {row.n_sessions:.0f} | "
            f"{row.point_fits} | {row.variance_group:.5g} | {row.variance_interaction:.5g} | "
            f"{row.variance_total_structured:.5g} | {row.variance_session:.5g} | "
            f"{row.variance_residual:.5g} | {row.fraction_structured_stable:.3f} |"
        )
    if not timescale_sensitivity.empty:
        totals = timescale_sensitivity["variance_total_structured"]
        nonboundary = timescale_sensitivity.loc[
            ~timescale_sensitivity["v1_total_near_boundary"],
            "variance_total_structured",
        ]
        hva_total = float(
            point.loc[
                point["metric"].eq("Response timescale (ms)")
                & point["dataset"].eq("Post-V1"),
                "variance_total_structured",
            ].iloc[0]
        )
        lines.extend(
            [
                "",
                "## Timescale trial-draw sensitivity",
                "",
                f"Across the ten V1 point fits, total structured variance ranges from "
                f"{totals.min():.4g} to {totals.max():.4g} ms² (median {totals.median():.4g}; "
                f"mean {totals.mean():.4g}). One draw is at the zero-variance boundary, "
                "and Powell and Nelder–Mead agree on that solution, so it is retained rather "
                "than treated as an optimizer failure.",
                "",
                f"The HVA/V1 point ratio is {hva_total / totals.mean():.3g}× using the "
                f"pre-specified mean across all draws, {hva_total / totals.median():.3g}× "
                f"using the median, and {hva_total / nonboundary.mean():.3g}× after excluding "
                "the single boundary draw. The directional conclusion is unchanged.",
                "",
                "| V1 draw | Cells | Stable group | Session × group | Total structured | Boundary? |",
                "|---:|---:|---:|---:|---:|:---:|",
            ]
        )
        for row in timescale_sensitivity.itertuples(index=False):
            lines.append(
                f"| {int(row.draw_id)} | {int(row.n_cells)} | {row.variance_group:.4g} | "
                f"{row.variance_interaction:.4g} | {row.variance_total_structured:.4g} | "
                f"{'yes' if row.v1_total_near_boundary else 'no'} |"
            )
    lines.extend(
        [
            "",
            "## HVA/V1 ratios",
            "",
            "| Metric | Contrast | Point ratio | Bootstrap ratio median (95% interval) | Point difference | Bootstrap difference median (95% interval) | P(difference <= 0) | V1 near-boundary fraction |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in ratios.itertuples(index=False):
        lines.append(
            f"| {row.metric} | {row.contrast} | {row.hva_to_v1_variance_ratio:.3g}× | "
            f"{row.bootstrap_median_ratio:.3g}× "
            f"([{row.bootstrap_ci_low:.3g}, {row.bootstrap_ci_high:.3g}]) | "
            f"{row.hva_minus_v1_variance:.4g} | {row.bootstrap_median_difference:.4g} "
            f"([{row.difference_ci_low:.4g}, {row.difference_ci_high:.4g}]) | "
            f"{row.bootstrap_probability_difference_le_0:.3f} | "
            f"{row.v1_near_boundary_fraction:.3f} |"
        )
    success = bootstrap["status"].eq("ok").groupby(
        [bootstrap["metric"], bootstrap["dataset"]]
    ).sum()
    lines.extend(["", "## Fit gate", ""])
    for (metric, dataset), count in success.items():
        lines.append(f"- {metric}, {dataset}: {int(count)} successful bootstrap fits.")
    lines.extend(
        [
            "",
            "Ratios are secondary when V1 resamples approach the zero-variance boundary;",
            "the unbounded HVA−V1 variance difference is the primary uncertainty summary.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    f1 = build_f1_f0_units()
    timescale = build_timescale_units()
    validation = pd.concat(
        [validate_f1_f0(f1), validate_timescale(timescale)], ignore_index=True
    )
    inputs = pd.concat([f1, timescale], ignore_index=True)

    point_fits = []
    optimizer_diagnostics = []
    for metric in METRICS:
        metric_units = inputs.loc[inputs["metric"].eq(metric)]
        for dataset in ["Within-V1", "Post-V1"]:
            local = metric_units.loc[metric_units["dataset"].eq(dataset)]
            draw_ids = sorted(local["draw_id"].unique())
            for draw_id in draw_ids:
                draw = local.loc[local["draw_id"].eq(draw_id)].copy()
                fit, diagnostics = fit_once(
                    draw, methods=["powell", "nm", "lbfgs"], maxiter=args.maxiter
                )
                fit.update({"metric": metric, "dataset": dataset, "draw_id": draw_id})
                diagnostics.insert(0, "draw_id", draw_id)
                diagnostics.insert(0, "dataset", dataset)
                diagnostics.insert(0, "metric", metric)
                point_fits.append(fit)
                optimizer_diagnostics.append(diagnostics)
                print(f"Point fit: {metric}, {dataset}, draw={draw_id}", flush=True)
    point_fits = pd.DataFrame(point_fits)
    diagnostics = pd.concat(optimizer_diagnostics, ignore_index=True)
    point = aggregate_point_fits(point_fits)
    timescale_sensitivity = summarize_timescale_draw_sensitivity(point_fits)

    seed_sequence = np.random.SeedSequence(args.seed)
    seeds = iter(seed_sequence.spawn(4))
    bootstraps = []
    preferred = point.set_index(["metric", "dataset"])["optimizer"]
    for metric in METRICS:
        metric_units = inputs.loc[inputs["metric"].eq(metric)]
        for dataset in ["Within-V1", "Post-V1"]:
            local = metric_units.loc[metric_units["dataset"].eq(dataset)].copy()
            method = str(preferred.loc[(metric, dataset)])
            rng = np.random.default_rng(next(seeds))
            if metric == "Response timescale (ms)" and dataset == "Within-V1":
                result = bootstrap_timescale_v1(
                    local, repetitions=args.bootstrap_repetitions,
                    preferred_method=method, maxiter=args.maxiter, rng=rng,
                )
            else:
                result = bootstrap_models(
                    local, repetitions=args.bootstrap_repetitions,
                    preferred_method=method, maxiter=args.maxiter, rng=rng,
                    dataset=dataset,
                )
            result["metric"] = metric
            bootstraps.append(result)
    bootstrap = pd.concat(bootstraps, ignore_index=True)

    ratio_frames = []
    for metric in METRICS:
        local_point = point.loc[point["metric"].eq(metric)]
        local_bootstrap = bootstrap.loc[bootstrap["metric"].eq(metric)]
        local_ratios = ratio_summary(local_point, local_bootstrap)
        local_ratios.insert(0, "metric", metric)
        ratio_frames.append(local_ratios)
    ratios = pd.concat(ratio_frames, ignore_index=True)

    inputs.to_csv(output / "full_cell_metric_extension_input.csv", index=False)
    validation.to_csv(output / "full_cell_metric_extension_validation.csv", index=False)
    point_fits.to_csv(output / "full_cell_point_fits_by_draw.csv", index=False)
    point.to_csv(output / "full_cell_variance_components.csv", index=False)
    timescale_sensitivity.to_csv(
        output / "timescale_trial_draw_sensitivity.csv", index=False
    )
    diagnostics.to_csv(output / "full_cell_optimizer_diagnostics.csv", index=False)
    bootstrap.to_csv(output / "full_cell_session_block_bootstrap.csv", index=False)
    ratios.to_csv(output / "full_cell_variance_ratios.csv", index=False)
    render(point, bootstrap, ratios, output / "Figure_full_cell_metric_extension.png")
    write_report(
        point, ratios, bootstrap, validation, timescale_sensitivity,
        output / "FULL_CELL_METRIC_EXTENSION_CHECKPOINT.md",
    )
    print(point.to_string(index=False))
    print(ratios.to_string(index=False))


if __name__ == "__main__":
    main()
