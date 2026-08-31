"""Full-cell variance-components model for the Figure 3 comparison.

Checkpoint 1 is TTFS-only. Every retained neuron enters the likelihood:

    y_isg = mu + session_s + group_g + session_group_sg + residual_isg

The model is fit separately to V1 locations and post-V1 areas. The primary
structured variance is group variance + session x group variance; stable group
variance is reported separately. Session-block bootstrap resampling retains all
neurons belonging to a sampled recording session and relabels duplicated blocks.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.figure3_robust_spread_comparison import AREA_ORDER  # noqa: E402

CORTICAL_HVA_ORDER = tuple(area for area in AREA_ORDER if area != "LP")

TTFS_BASE = ROOT / "artifacts/figure3/06r_probe_flash_photoartifact_pilot/ttfs_source_provenance"
MOUSE_TTFS_UNITS = TTFS_BASE / "figure3_response_filtered_preferred_ttfs_units.csv"
ALLEN_TTFS_AUDIT = TTFS_BASE / "figure3_response_filtered_ttfs_all_areas_unit_audit.csv"
CURRENT_MEANS = ROOT / "Figure3/Figure3_robust_session_group_means.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/figure3/07_big_picture_concrete_first/full_cell_model/ttfs_cortical_hvas"
COMPONENTS = ["group", "interaction", "session", "residual"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repetitions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--maxiter", type=int, default=3000)
    return parser.parse_args()


def build_ttfs_unit_table() -> pd.DataFrame:
    mouse = pd.read_csv(MOUSE_TTFS_UNITS)
    mouse = mouse.loc[
        mouse["cohort"].eq("MouseV2")
        & mouse["selected"].astype(bool)
        & pd.to_numeric(mouse["preferred_0_250_ttfs_ms"], errors="coerce").lt(100)
    ].copy()
    mouse = mouse.rename(
        columns={
            "preferred_0_250_ttfs_ms": "value",
            "location": "group",
        }
    )
    mouse["dataset"] = "Within-V1"
    mouse["source"] = str(MOUSE_TTFS_UNITS.relative_to(ROOT))

    allen = pd.read_csv(ALLEN_TTFS_AUDIT)
    allen = allen.loc[
        allen["area_coarse"].isin(CORTICAL_HVA_ORDER)
        & allen["selected_positive_responder_area"].astype(bool)
        & pd.to_numeric(allen["preferred_0_250_ttfs_ms"], errors="coerce").lt(100)
    ].copy()
    cell_size = allen.groupby(["session_id", "area_coarse"])["unit_id"].transform("size")
    allen = allen.loc[cell_size.ge(10)].rename(
        columns={
            "preferred_0_250_ttfs_ms": "value",
            "area_coarse": "group",
        }
    )
    allen["dataset"] = "Post-V1"
    allen["source"] = str(ALLEN_TTFS_AUDIT.relative_to(ROOT))

    columns = ["dataset", "session_id", "group", "unit_id", "value", "source"]
    result = pd.concat([mouse[columns], allen[columns]], ignore_index=True)
    result["metric"] = "TTFS (ms)"
    result["session_id"] = result["session_id"].astype(str)
    result["group"] = result["group"].astype(str)
    result["unit_id"] = result["unit_id"].astype(str)
    if result.duplicated(["dataset", "session_id", "group", "unit_id"]).any():
        raise ValueError("Duplicate TTFS neuron within a session x group population")
    return result


def validate_unit_table(units: pd.DataFrame) -> pd.DataFrame:
    reconstructed = (
        units.groupby(["dataset", "metric", "session_id", "group"], as_index=False)
        .agg(mean_reconstructed=("value", "mean"), n_units_reconstructed=("unit_id", "size"))
    )
    current = pd.read_csv(CURRENT_MEANS, dtype={"session_id": str})
    current = current.loc[
        current["metric"].eq("TTFS (ms)")
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
    audit["unit_count_error"] = audit["n_units"] - audit["n_units_reconstructed"]
    if audit["absolute_mean_error"].max() > 1e-10 or not audit["unit_count_error"].eq(0).all():
        raise ValueError("Full-cell TTFS input does not reproduce current Figure 3 cells")
    return audit


def prepare_model_table(units: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    local = units.copy()
    scale = float(local["value"].std(ddof=1))
    local["z"] = (local["value"] - local["value"].mean()) / scale
    local["session"] = local["session_id"].astype(str)
    local["session_group"] = local["session"] + ":" + local["group"]
    local["all_rows"] = "all"
    return local, scale


def fit_once(
    units: pd.DataFrame,
    *,
    methods: list[str],
    maxiter: int,
) -> tuple[dict[str, float | str | bool], pd.DataFrame]:
    table, outcome_scale = prepare_model_table(units)
    model = smf.mixedlm(
        "z ~ 1",
        table,
        groups=table["all_rows"],
        re_formula="0",
        vc_formula={
            "session": "0 + C(session)",
            "group": "0 + C(group)",
            "interaction": "0 + C(session_group)",
        },
    )
    candidates: list[tuple[object, str, list[str]]] = []
    diagnostics: list[dict[str, object]] = []
    for method in methods:
        caught: list[str] = []
        try:
            with warnings.catch_warnings(record=True) as records:
                warnings.simplefilter("always")
                result = model.fit(
                    reml=True, method=method, maxiter=maxiter, disp=False
                )
            caught = [str(record.message) for record in records]
            diagnostics.append(
                {
                    "optimizer": method,
                    "completed": True,
                    "converged": bool(result.converged),
                    "reml_log_likelihood": float(result.llf),
                    "warnings": " | ".join(caught),
                }
            )
            if result.converged and np.isfinite(result.llf):
                candidates.append((result, method, caught))
        except Exception as exc:
            diagnostics.append(
                {
                    "optimizer": method,
                    "completed": False,
                    "converged": False,
                    "reml_log_likelihood": np.nan,
                    "warnings": f"{type(exc).__name__}: {exc}",
                }
            )
    if not candidates:
        raise RuntimeError("No optimizer produced a converged finite fit")
    result, selected_method, selected_warnings = max(candidates, key=lambda item: item[0].llf)
    variances = {
        name: float(value * outcome_scale**2)
        for name, value in zip(model.exog_vc.names, result.vcomp)
    }
    variances["residual"] = float(result.scale * outcome_scale**2)
    row: dict[str, float | str | bool] = {
        "optimizer": selected_method,
        "converged": bool(result.converged),
        "reml_log_likelihood": float(result.llf),
        "outcome_scale": outcome_scale,
        "n_cells": int(len(table)),
        "n_sessions": int(table["session"].nunique()),
        "n_groups": int(table["group"].nunique()),
        "n_session_groups": int(table["session_group"].nunique()),
        **{f"variance_{name}": variances[name] for name in COMPONENTS},
    }
    row["variance_total_structured"] = variances["group"] + variances["interaction"]
    row["fraction_structured_stable"] = (
        variances["group"] / row["variance_total_structured"]
        if row["variance_total_structured"] > 0 else np.nan
    )
    row["selected_warnings"] = " | ".join(selected_warnings)
    return row, pd.DataFrame(diagnostics)


def resample_sessions(units: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    session_ids = np.asarray(sorted(units["session_id"].unique()), dtype=object)
    selected = rng.choice(session_ids, size=len(session_ids), replace=True)
    blocks = []
    for block_index, session_id in enumerate(selected):
        block = units.loc[units["session_id"].eq(session_id)].copy()
        block["session_id"] = f"bootstrap_{block_index}:{session_id}"
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def bootstrap_models(
    units: pd.DataFrame,
    *,
    repetitions: int,
    preferred_method: str,
    maxiter: int,
    rng: np.random.Generator,
    dataset: str,
) -> pd.DataFrame:
    rows = []
    methods = [preferred_method] + [
        method for method in ["powell", "nm", "lbfgs"] if method != preferred_method
    ]
    for repetition in range(repetitions):
        resampled = resample_sessions(units, rng)
        try:
            if preferred_method == "lbfgs":
                # L-BFGS can converge to a high-variance, non-positive-definite
                # stationary point in sparse HVA resamples. Compare all three
                # optimizers and retain the highest REML likelihood, as for the
                # full-data point fit.
                fit, _ = fit_once(resampled, methods=methods, maxiter=maxiter)
            else:
                try:
                    fit, _ = fit_once(resampled, methods=methods[:1], maxiter=maxiter)
                except Exception:
                    fit, _ = fit_once(resampled, methods=methods[1:2], maxiter=maxiter)
            fit["bootstrap_repetition"] = repetition
            fit["dataset"] = dataset
            fit["status"] = "ok"
            rows.append(fit)
        except Exception as exc:
            rows.append(
                {
                    "bootstrap_repetition": repetition,
                    "dataset": dataset,
                    "status": f"{type(exc).__name__}: {exc}",
                }
            )
        if (repetition + 1) % 10 == 0 or repetition + 1 == repetitions:
            completed = sum(row["status"] == "ok" for row in rows)
            print(
                f"{dataset}: bootstrap {repetition + 1}/{repetitions}; "
                f"successful={completed}",
                flush=True,
            )
    return pd.DataFrame(rows)


def ratio_summary(point: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    point_index = point.set_index("dataset")
    good = bootstrap.loc[bootstrap["status"].eq("ok")].copy()
    v1 = good.loc[good["dataset"].eq("Within-V1")].sort_values("bootstrap_repetition")
    hva = good.loc[good["dataset"].eq("Post-V1")].sort_values("bootstrap_repetition")
    shared = sorted(set(v1["bootstrap_repetition"]) & set(hva["bootstrap_repetition"]))
    v1 = v1.set_index("bootstrap_repetition").loc[shared]
    hva = hva.set_index("bootstrap_repetition").loc[shared]
    rows = []
    for label, column in [
        ("total_structured", "variance_total_structured"),
        ("stable_identity", "variance_group"),
        ("session_specific", "variance_interaction"),
    ]:
        point_ratio = float(
            point_index.loc["Post-V1", column]
            / point_index.loc["Within-V1", column]
        )
        point_difference = float(
            point_index.loc["Post-V1", column]
            - point_index.loc["Within-V1", column]
        )
        draws = hva[column].to_numpy(float) / v1[column].to_numpy(float)
        difference_draws = hva[column].to_numpy(float) - v1[column].to_numpy(float)
        finite = draws[np.isfinite(draws)]
        boundary_threshold = max(
            1e-10, 0.01 * float(point_index.loc["Within-V1", column])
        )
        rows.append(
            {
                "contrast": label,
                "hva_to_v1_variance_ratio": point_ratio,
                "bootstrap_median_ratio": float(np.median(finite)),
                "bootstrap_ci_low": float(np.quantile(finite, 0.025)),
                "bootstrap_ci_high": float(np.quantile(finite, 0.975)),
                "bootstrap_probability_ratio_le_1": float(np.mean(finite <= 1)),
                "hva_minus_v1_variance": point_difference,
                "bootstrap_median_difference": float(np.median(difference_draws)),
                "difference_ci_low": float(np.quantile(difference_draws, 0.025)),
                "difference_ci_high": float(np.quantile(difference_draws, 0.975)),
                "bootstrap_probability_difference_le_0": float(
                    np.mean(difference_draws <= 0)
                ),
                "v1_near_boundary_fraction": float(
                    np.mean(v1[column].to_numpy(float) < boundary_threshold)
                ),
                "v1_boundary_threshold": boundary_threshold,
                "successful_paired_bootstraps": int(len(finite)),
            }
        )
    return pd.DataFrame(rows)


def render(point: pd.DataFrame, bootstrap: pd.DataFrame, ratios: pd.DataFrame, output: Path) -> None:
    colors = {"Within-V1": "#7564a8", "Post-V1": "#555555"}
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    x = np.arange(len(COMPONENTS))
    width = 0.34
    for offset, dataset in [(-width / 2, "Within-V1"), (width / 2, "Post-V1")]:
        row = point.loc[point["dataset"].eq(dataset)].iloc[0]
        values = [row[f"variance_{component}"] for component in COMPONENTS]
        axes[0].bar(x + offset, values, width, color=colors[dataset], alpha=0.85, label=dataset)
    axes[0].set_xticks(x, ["Stable\ngroup", "Session ×\ngroup", "Session", "Cell\nresidual"])
    axes[0].set_ylabel("TTFS variance (ms²)")
    axes[0].set_yscale("log")
    axes[0].set_title("Full-cell variance decomposition", loc="left", fontweight="bold")
    axes[0].legend(frameon=False)
    axes[0].spines[["top", "right"]].set_visible(False)

    good = bootstrap.loc[bootstrap["status"].eq("ok")]
    v1 = good.loc[good["dataset"].eq("Within-V1")].set_index("bootstrap_repetition")
    hva = good.loc[good["dataset"].eq("Post-V1")].set_index("bootstrap_repetition")
    shared = sorted(set(v1.index) & set(hva.index))
    total_ratio = (
        hva.loc[shared, "variance_total_structured"].to_numpy(float)
        / v1.loc[shared, "variance_total_structured"].to_numpy(float)
    )
    finite_ratio = total_ratio[np.isfinite(total_ratio) & (total_ratio > 0)]
    axes[1].hist(np.log10(finite_ratio), bins=30, color="#4778a8", alpha=0.75)
    axes[1].axvline(0, color="#777777", linestyle="--", linewidth=1)
    point_ratio = ratios.set_index("contrast").loc[
        "total_structured", "hva_to_v1_variance_ratio"
    ]
    axes[1].axvline(np.log10(point_ratio), color="#b2182b", linewidth=2, label=f"Point ratio {point_ratio:.2f}×")
    axes[1].set_xlabel("log10 HVA/V1 total structured variance ratio")
    axes[1].set_ylabel("Session-block bootstrap draws")
    axes[1].set_title("Primary contrast", loc="left", fontweight="bold")
    axes[1].legend(frameon=False)
    axes[1].spines[["top", "right"]].set_visible(False)
    figure.suptitle("TTFS full-cell multilevel model pilot", fontweight="bold")
    figure.savefig(output, dpi=220, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def write_report(
    point: pd.DataFrame,
    diagnostics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    ratios: pd.DataFrame,
    validation: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# Full-cell multilevel model — TTFS pilot checkpoint",
        "",
        f"_Generated {date.today().isoformat()}._",
        "",
        "## Input gate",
        "",
        "The post-V1 population contains the five cortical HVAs LM, RL, AL, PM, and AM.",
        "The thalamic lateral posterior nucleus (LP) is excluded.",
        "",
        f"All {len(validation)} current TTFS session × group populations are reconstructed",
        f"from {int(point['n_cells'].sum())} neuron rows. Maximum group-mean error is",
        f"{validation['absolute_mean_error'].max():.3g}; every retained count matches.",
        "",
        "## Model",
        "",
        "Each retained neuron enters a Gaussian mixed model with crossed random",
        "components for session, stable group identity, and session × group, plus",
        "cell residual variance. V1 and HVA are fit separately on their original",
        "TTFS scale. The primary structured component is stable group plus",
        "session × group variance.",
        "",
        "## Point estimates",
        "",
        "| Dataset | Cells | Sessions | Groups | Optimizer | Stable group | Session × group | Total structured | Session | Cell residual | Stable fraction |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in point.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.n_cells} | {row.n_sessions} | {row.n_groups} | "
            f"{row.optimizer} | {row.variance_group:.4g} | {row.variance_interaction:.4g} | "
            f"{row.variance_total_structured:.4g} | {row.variance_session:.4g} | "
            f"{row.variance_residual:.4g} | {row.fraction_structured_stable:.3f} |"
        )
    lines.extend(
        [
            "",
            "## HVA/V1 variance ratios",
            "",
            "| Contrast | Point ratio | Bootstrap ratio median (95% interval) | Point difference | Bootstrap difference median (95% interval) | P(difference <= 0) | V1 near-boundary fraction |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in ratios.itertuples(index=False):
        lines.append(
            f"| {row.contrast} | {row.hva_to_v1_variance_ratio:.3g}× | "
            f"{row.bootstrap_median_ratio:.3g}× "
            f"([{row.bootstrap_ci_low:.3g}, {row.bootstrap_ci_high:.3g}]) | "
            f"{row.hva_minus_v1_variance:.3g} | "
            f"{row.bootstrap_median_difference:.3g} "
            f"([{row.difference_ci_low:.3g}, {row.difference_ci_high:.3g}]) | "
            f"{row.bootstrap_probability_difference_le_0:.3f} | "
            f"{row.v1_near_boundary_fraction:.3f} |"
        )
    successful = bootstrap["status"].eq("ok").groupby(bootstrap["dataset"]).sum()
    lines.extend(
        [
            "",
            "## Convergence and interpretation gate",
            "",
            f"Successful session-block fits: Within-V1 {int(successful.get('Within-V1', 0))}, "
            f"Post-V1 {int(successful.get('Post-V1', 0))}.",
            "Point estimates compare Powell, Nelder–Mead, and L-BFGS and retain the",
            "converged solution with the highest REML likelihood. Bootstrap fits use",
            "the selected optimizer with one fallback; when L-BFGS is selected, all",
            "three optimizers are compared to reject its known high-variance stationary point.",
            "Ratio intervals become unstable when a resampled eight-session V1 fit",
            "places structured variance near zero. The near-boundary diagnostic uses",
            "1% of the full-data V1 component as its threshold; the unbounded variance",
            "difference is the primary uncertainty summary in that case.",
            "",
            "This pilot conditions on the response-selected TTFS population and uses",
            "a Gaussian cell residual. Inspect optimizer agreement and the bootstrap",
            "distribution before extending the specification to the other metrics.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    units = build_ttfs_unit_table()
    validation = validate_unit_table(units)
    point_rows = []
    diagnostic_rows = []
    selected_methods: dict[str, str] = {}
    for dataset in ["Within-V1", "Post-V1"]:
        local = units.loc[units["dataset"].eq(dataset)].copy()
        fit, diagnostics = fit_once(
            local, methods=["powell", "nm", "lbfgs"], maxiter=args.maxiter
        )
        fit["dataset"] = dataset
        point_rows.append(fit)
        diagnostics.insert(0, "dataset", dataset)
        diagnostic_rows.append(diagnostics)
        selected_methods[dataset] = str(fit["optimizer"])
    point = pd.DataFrame(point_rows)
    diagnostics = pd.concat(diagnostic_rows, ignore_index=True)

    seed_sequence = np.random.SeedSequence(args.seed)
    seeds = seed_sequence.spawn(2)
    bootstraps = []
    for dataset, seed in zip(["Within-V1", "Post-V1"], seeds):
        local = units.loc[units["dataset"].eq(dataset)].copy()
        bootstraps.append(
            bootstrap_models(
                local,
                repetitions=args.bootstrap_repetitions,
                preferred_method=selected_methods[dataset],
                maxiter=args.maxiter,
                rng=np.random.default_rng(seed),
                dataset=dataset,
            )
        )
    bootstrap = pd.concat(bootstraps, ignore_index=True)
    ratios = ratio_summary(point, bootstrap)

    units.to_csv(output / "ttfs_full_cell_input.csv", index=False)
    validation.to_csv(output / "ttfs_full_cell_input_validation.csv", index=False)
    point.to_csv(output / "ttfs_variance_components.csv", index=False)
    diagnostics.to_csv(output / "ttfs_optimizer_diagnostics.csv", index=False)
    bootstrap.to_csv(output / "ttfs_session_block_bootstrap.csv", index=False)
    ratios.to_csv(output / "ttfs_variance_ratios.csv", index=False)
    render(point, bootstrap, ratios, output / "Figure_ttfs_full_cell_model.png")
    write_report(
        point, diagnostics, bootstrap, ratios, validation,
        output / "TTFS_FULL_CELL_MODEL_CHECKPOINT.md",
    )
    print(point.to_string(index=False))
    print(ratios.to_string(index=False))


if __name__ == "__main__":
    main()
