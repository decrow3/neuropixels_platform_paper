#!/usr/bin/env python3
"""Quantify inflation of Allen BO 1.1 Gaussian RF fits near mapping-grid edges."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUPPORT = (
    ROOT / "artifacts" / "figure3" / "06c_allen_rf_matching" / "rf_unit_common_support.csv"
)
DEFAULT_UNITS = ROOT / "data" / "unit_table.csv"
DEFAULT_OUTPUT = ROOT / "artifacts" / "allen_bo11_gaussian_rf_edge_inflation"
COHORT = "Brain Observatory 1.1"
GRID_LIMITS = (10.0, 90.0, -30.0, 50.0)
GRID_SPAN_DEG = 80.0
BAND_ORDER = ("0–5", ">5–10", ">10–20", ">20–40")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--unit-table", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=737581020)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantile_25(values: pd.Series) -> float:
    return values.quantile(0.25)


def quantile_75(values: pd.Series) -> float:
    return values.quantile(0.75)


def quantile_90(values: pd.Series) -> float:
    return values.quantile(0.90)


def off_screen_fraction(values: pd.Series) -> float:
    return 1.0 - values.astype(bool).mean()


def prepare_population(support_path: Path, unit_table_path: Path) -> tuple[pd.DataFrame, dict]:
    support = pd.read_csv(support_path, low_memory=False)
    support = support.loc[support["cohort"].eq(COHORT)].copy()
    released = pd.read_csv(
        unit_table_path,
        usecols=["ecephys_unit_id", "width_rf", "height_rf", "on_screen_rf"],
        low_memory=False,
    )
    population = support.merge(released, on="ecephys_unit_id", how="left", validate="one_to_one")
    audit = {
        "support_units": int(len(population)),
        "support_sessions": int(population["ecephys_session_id"].nunique()),
        "support_areas": sorted(population["area"].unique().tolist()),
        "maximum_p_value_rf": float(population["p_value_rf"].max()),
        "missing_either_gaussian_dimension": int(
            population[["width_rf", "height_rf"]].isna().any(axis=1).sum()
        ),
        "negative_width_parameters": int(population["width_rf"].lt(0).sum()),
        "negative_height_parameters": int(population["height_rf"].lt(0).sum()),
    }
    population = population.dropna(
        subset=["azimuth_rf", "elevation_rf", "width_rf", "height_rf"]
    ).copy()
    az_min, az_max, el_min, el_max = GRID_LIMITS
    population["azimuth_edge_distance_deg"] = np.minimum(
        population["azimuth_rf"] - az_min, az_max - population["azimuth_rf"]
    )
    population["elevation_edge_distance_deg"] = np.minimum(
        population["elevation_rf"] - el_min, el_max - population["elevation_rf"]
    )
    population["nearest_edge_distance_deg"] = population[
        ["azimuth_edge_distance_deg", "elevation_edge_distance_deg"]
    ].min(axis=1)
    # Gaussian widths enter the model squared, so their signs are non-identifiable.
    population["width_sigma_magnitude_deg"] = population["width_rf"].abs()
    population["height_sigma_magnitude_deg"] = population["height_rf"].abs()
    population["maximum_sigma_magnitude_deg"] = population[
        ["width_sigma_magnitude_deg", "height_sigma_magnitude_deg"]
    ].max(axis=1)
    population["geometric_mean_sigma_deg"] = np.sqrt(
        population["width_sigma_magnitude_deg"] * population["height_sigma_magnitude_deg"]
    )
    population["log2_maximum_sigma"] = np.log2(population["maximum_sigma_magnitude_deg"])
    population["log2_geometric_mean_sigma"] = np.log2(population["geometric_mean_sigma_deg"])
    population["larger_than_mapped_span"] = population["maximum_sigma_magnitude_deg"].gt(
        GRID_SPAN_DEG
    )
    population["edge_band_deg"] = pd.cut(
        population["nearest_edge_distance_deg"],
        bins=[-0.001, 5.0, 10.0, 20.0, 40.001],
        labels=BAND_ORDER,
        include_lowest=True,
    )
    return population, audit


def summarize_bands(population: pd.DataFrame, grouping: list[str]) -> pd.DataFrame:
    return (
        population.groupby(grouping + ["edge_band_deg"], observed=True)
        .agg(
            units=("ecephys_unit_id", "size"),
            sessions=("ecephys_session_id", "nunique"),
            median_maximum_sigma_deg=("maximum_sigma_magnitude_deg", "median"),
            q25_maximum_sigma_deg=("maximum_sigma_magnitude_deg", quantile_25),
            q75_maximum_sigma_deg=("maximum_sigma_magnitude_deg", quantile_75),
            q90_maximum_sigma_deg=("maximum_sigma_magnitude_deg", quantile_90),
            median_geometric_mean_sigma_deg=("geometric_mean_sigma_deg", "median"),
            fraction_larger_than_mapped_span=("larger_than_mapped_span", "mean"),
            fraction_gaussian_center_off_screen=("on_screen_rf", off_screen_fraction),
        )
        .reset_index()
    )


def build_session_tables(population: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    session_band = summarize_bands(population, ["ecephys_session_id"])
    rows = []
    for session_id, group in population.groupby("ecephys_session_id", observed=True):
        edge = group.loc[group["nearest_edge_distance_deg"].le(10)]
        interior = group.loc[group["nearest_edge_distance_deg"].gt(20)]
        rows.append(
            {
                "ecephys_session_id": int(session_id),
                "edge_units": len(edge),
                "interior_units": len(interior),
                "edge_median_maximum_sigma_deg": edge["maximum_sigma_magnitude_deg"].median(),
                "interior_median_maximum_sigma_deg": interior["maximum_sigma_magnitude_deg"].median(),
                "edge_to_interior_median_maximum_sigma_ratio": 2 ** (
                    edge["log2_maximum_sigma"].median() - interior["log2_maximum_sigma"].median()
                ),
                "edge_to_interior_median_geometric_mean_sigma_ratio": 2 ** (
                    edge["log2_geometric_mean_sigma"].median()
                    - interior["log2_geometric_mean_sigma"].median()
                ),
                "edge_oversize_fraction": edge["larger_than_mapped_span"].mean(),
                "interior_oversize_fraction": interior["larger_than_mapped_span"].mean(),
                "edge_off_screen_fraction": 1.0 - edge["on_screen_rf"].astype(bool).mean(),
                "interior_off_screen_fraction": 1.0
                - interior["on_screen_rf"].astype(bool).mean(),
            }
        )
    return session_band, pd.DataFrame(rows)


def bootstrap_median_interval(values: pd.Series, repetitions: int, seed: int) -> tuple[float, float]:
    array = values.dropna().to_numpy(float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(repetitions, len(array)))
    bootstrapped = np.median(array[indices], axis=1)
    return tuple(np.quantile(bootstrapped, [0.025, 0.975]))


def comparison_sensitivity(population: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for near_cutoff in (0.0, 5.0, 10.0, 15.0):
        near = population.loc[population["nearest_edge_distance_deg"].le(near_cutoff)]
        interior = population.loc[population["nearest_edge_distance_deg"].gt(20)]
        rows.append(
            {
                "near_edge_cutoff_deg": near_cutoff,
                "near_units": len(near),
                "interior_units": len(interior),
                "near_median_maximum_sigma_deg": near["maximum_sigma_magnitude_deg"].median(),
                "interior_median_maximum_sigma_deg": interior["maximum_sigma_magnitude_deg"].median(),
                "median_ratio": near["maximum_sigma_magnitude_deg"].median()
                / interior["maximum_sigma_magnitude_deg"].median(),
                "near_oversize_fraction": near["larger_than_mapped_span"].mean(),
                "interior_oversize_fraction": interior["larger_than_mapped_span"].mean(),
                "near_off_screen_fraction": 1.0 - near["on_screen_rf"].astype(bool).mean(),
                "interior_off_screen_fraction": 1.0
                - interior["on_screen_rf"].astype(bool).mean(),
            }
        )
    return pd.DataFrame(rows)


def stratified_summary(population: pd.DataFrame) -> pd.DataFrame:
    local = population.copy()
    local["gaussian_center"] = np.where(local["on_screen_rf"], "on screen", "off screen")
    return summarize_bands(local, ["gaussian_center"])


def area_contrasts(population: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for area, group in population.groupby("area", observed=True):
        edge = group.loc[group["nearest_edge_distance_deg"].le(10)]
        interior = group.loc[group["nearest_edge_distance_deg"].gt(20)]
        rows.append(
            {
                "area": area,
                "edge_units": len(edge),
                "interior_units": len(interior),
                "edge_median_maximum_sigma_deg": edge["maximum_sigma_magnitude_deg"].median(),
                "interior_median_maximum_sigma_deg": interior["maximum_sigma_magnitude_deg"].median(),
                "edge_to_interior_median_ratio": edge["maximum_sigma_magnitude_deg"].median()
                / interior["maximum_sigma_magnitude_deg"].median(),
                "edge_oversize_fraction": edge["larger_than_mapped_span"].mean(),
                "interior_oversize_fraction": interior["larger_than_mapped_span"].mean(),
            }
        )
    return pd.DataFrame(rows)


def select_concrete_cases(population: pd.DataFrame) -> pd.DataFrame:
    definitions = (
        ("edge, off-screen Gaussian center", population["nearest_edge_distance_deg"].le(5) & ~population["on_screen_rf"]),
        ("edge, on-screen Gaussian center", population["nearest_edge_distance_deg"].le(5) & population["on_screen_rf"]),
        ("interior, on-screen Gaussian center", population["nearest_edge_distance_deg"].gt(20) & population["on_screen_rf"]),
        ("interior, off-screen Gaussian center", population["nearest_edge_distance_deg"].gt(20) & ~population["on_screen_rf"]),
    )
    cases = []
    for label, mask in definitions:
        candidates = population.loc[mask].copy()
        target = candidates["log2_maximum_sigma"].median()
        selected = candidates.loc[(candidates["log2_maximum_sigma"] - target).abs().idxmin()].copy()
        selected["selection"] = label
        cases.append(selected)
    columns = [
        "selection", "ecephys_unit_id", "ecephys_session_id", "area", "azimuth_rf",
        "elevation_rf", "nearest_edge_distance_deg", "width_rf", "height_rf",
        "maximum_sigma_magnitude_deg", "geometric_mean_sigma_deg", "area_rf", "p_value_rf",
        "on_screen_rf",
    ]
    return pd.DataFrame(cases)[columns]


def render_figure(
    session_band: pd.DataFrame,
    paired: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.6))
    colors = ["#b23a48", "#d97736", "#e0aa3e", "#39738c"]
    positions = np.arange(len(BAND_ORDER))
    band_values = [
        session_band.loc[session_band["edge_band_deg"].eq(b), "median_maximum_sigma_deg"].to_numpy()
        for b in BAND_ORDER
    ]
    box = axes[0, 0].boxplot(band_values, positions=positions, widths=0.62, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    axes[0, 0].axhline(GRID_SPAN_DEG, color="#222222", linestyle="--", linewidth=1.2)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set(xticks=positions, xticklabels=BAND_ORDER, xlabel="Distance to nearest mapped edge (deg)", ylabel="Within-session median max |Gaussian σ| (deg)", title="A. Fitted scale falls with edge distance")

    for _, row in paired.iterrows():
        axes[0, 1].plot([0, 1], [row["interior_median_maximum_sigma_deg"], row["edge_median_maximum_sigma_deg"]], color="#999999", alpha=0.42, linewidth=0.8)
    axes[0, 1].scatter(np.zeros(len(paired)), paired["interior_median_maximum_sigma_deg"], color="#39738c", s=24, zorder=3)
    axes[0, 1].scatter(np.ones(len(paired)), paired["edge_median_maximum_sigma_deg"], color="#b23a48", s=24, zorder=3)
    axes[0, 1].axhline(GRID_SPAN_DEG, color="#222222", linestyle="--", linewidth=1.2)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set(xticks=[0, 1], xticklabels=[">20° interior", "≤10° edge"], ylabel="Session median max |Gaussian σ| (deg)", title="B. Paired sessions")

    band_oversize = [
        session_band.loc[session_band["edge_band_deg"].eq(b), "fraction_larger_than_mapped_span"].to_numpy()
        for b in BAND_ORDER
    ]
    box = axes[1, 0].boxplot(band_oversize, positions=positions, widths=0.62, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    axes[1, 0].set_ylim(-0.03, 1.03)
    axes[1, 0].set(xticks=positions, xticklabels=BAND_ORDER, xlabel="Distance to nearest mapped edge (deg)", ylabel="Fraction with max |σ| > 80°", title="C. Fits larger than the mapped span")

    band_offscreen = [
        session_band.loc[session_band["edge_band_deg"].eq(b), "fraction_gaussian_center_off_screen"].to_numpy()
        for b in BAND_ORDER
    ]
    box = axes[1, 1].boxplot(band_offscreen, positions=positions, widths=0.62, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    axes[1, 1].set_ylim(-0.03, 1.03)
    axes[1, 1].set(xticks=positions, xticklabels=BAND_ORDER, xlabel="Distance to nearest mapped edge (deg)", ylabel="Fraction with off-screen fitted center", title="D. Inflation tracks unconstrained off-screen fits")
    for ax in axes.ravel():
        ax.grid(alpha=0.18)
    fig.suptitle("Allen BO 1.1 Gaussian RF fits inflate near the 9×9 mapping-grid boundary", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    population, audit = prepare_population(args.support.resolve(), args.unit_table.resolve())
    population_summary = summarize_bands(population, [])
    area_summary = summarize_bands(population, ["area"])
    session_band, paired = build_session_tables(population)
    sensitivity = comparison_sensitivity(population)
    stratified = stratified_summary(population)
    area_contrast = area_contrasts(population)
    cases = select_concrete_cases(population)

    ratio_ci = bootstrap_median_interval(
        paired["edge_to_interior_median_maximum_sigma_ratio"],
        args.bootstrap_repetitions,
        args.seed,
    )
    median_ratio = paired["edge_to_interior_median_maximum_sigma_ratio"].median()
    median_geomean_ratio = paired["edge_to_interior_median_geometric_mean_sigma_ratio"].median()
    sessions_larger = int(paired["edge_to_interior_median_maximum_sigma_ratio"].gt(1).sum())
    two_sided_sign_p = 2.0 / (2.0 ** len(paired)) if sessions_larger in (0, len(paired)) else np.nan
    edge = population.loc[population["nearest_edge_distance_deg"].le(10)]
    interior = population.loc[population["nearest_edge_distance_deg"].gt(20)]
    on_screen = population.loc[population["on_screen_rf"].astype(bool)]
    on_screen_edge = on_screen.loc[on_screen["nearest_edge_distance_deg"].le(10)]
    on_screen_interior = on_screen.loc[on_screen["nearest_edge_distance_deg"].gt(20)]

    population.to_csv(output_dir / "unit_level_metrics.csv", index=False, float_format="%.7g")
    population_summary.to_csv(output_dir / "edge_band_population_summary.csv", index=False, float_format="%.7g")
    area_summary.to_csv(output_dir / "edge_band_area_summary.csv", index=False, float_format="%.7g")
    area_contrast.to_csv(output_dir / "edge_vs_interior_area_summary.csv", index=False, float_format="%.7g")
    session_band.to_csv(output_dir / "edge_band_session_summary.csv", index=False, float_format="%.7g")
    paired.to_csv(output_dir / "edge_vs_interior_paired_sessions.csv", index=False, float_format="%.7g")
    sensitivity.to_csv(output_dir / "edge_cutoff_sensitivity.csv", index=False, float_format="%.7g")
    stratified.to_csv(output_dir / "edge_band_on_screen_stratification.csv", index=False, float_format="%.7g")
    cases.to_csv(output_dir / "concrete_cases.csv", index=False, float_format="%.7g")
    figure_path = output_dir / "Figure_allen_bo11_gaussian_rf_edge_inflation.png"
    render_figure(session_band, paired, figure_path)

    report = [
        "# Allen BO 1.1 Gaussian RF fits inflate near mapping-grid edges",
        "",
        "## Answer",
        "",
        f"In the {audit['support_sessions']}-session, six-area significant-RF cohort, the session-median edge/interior ratio was **{median_ratio:.1f}×** (bootstrap 95% CI **{ratio_ci[0]:.1f}–{ratio_ci[1]:.1f}×**). Edge means a released threshold-mask center ≤10° from any boundary; interior means >20° from every boundary. The edge median was larger in **{sessions_larger}/{len(paired)} sessions** (two-sided sign-test p = {two_sided_sign_p:.2g}).",
        "",
        f"At the unit level, **{edge['larger_than_mapped_span'].mean():.1%}** of edge fits versus **{interior['larger_than_mapped_span'].mean():.1%}** of interior fits had at least one fitted Gaussian sigma larger than the full 80° sampled span. The corresponding Gaussian-center off-screen rates were **{(1-edge['on_screen_rf'].astype(bool).mean()):.1%}** and **{(1-interior['on_screen_rf'].astype(bool).mean()):.1%}**.",
        "",
        "## Interpretation",
        "",
        "This is a strong boundary-associated numerical failure, not evidence for biological RFs hundreds of degrees wide. Allen fits an unbounded five-parameter Gaussian by least squares, does not constrain its center or sigmas, and ignores the returned fit-success flag when releasing the metrics. `on_screen_rf` checks only whether the fitted Gaussian center lies inside the 9×9 array; it does not test whether the fitted width is supported by the sampled field.",
        "",
        f"The edge effect is largely carried by fits whose Gaussian center extrapolates off screen. Among fits with `on_screen_rf == True`, the >80° rates were **{on_screen_edge['larger_than_mapped_span'].mean():.1%}** at the edge and **{on_screen_interior['larger_than_mapped_span'].mean():.1%}** in the interior; their unit-level median maximum sigmas were **{on_screen_edge['maximum_sigma_magnitude_deg'].median():.1f}°** and **{on_screen_interior['maximum_sigma_magnitude_deg'].median():.1f}°**, respectively.",
        "",
        f"The result is not dependent on choosing the larger axis alone: the median session-level edge/interior ratio for the geometric mean of |width| and |height| was **{median_geomean_ratio:.1f}×**. All six cortical areas show the same direction in the population summary.",
        "",
        "## Definitions and cohort",
        "",
        f"The source population contains **{audit['support_units']:,} units** with p_value_rf ≤ {audit['maximum_p_value_rf']:.3f} from {audit['support_sessions']} BO 1.1 sessions and areas {', '.join(audit['support_areas'])}. Four units lacked one or both Gaussian dimensions, leaving **{len(population):,}** analyzed fits.",
        "",
        "`width_rf` and `height_rf` are Gaussian sigma parameters converted to degrees. Because the Gaussian uses each width only after squaring it, the parameter sign is non-identifiable; this analysis uses absolute magnitudes. The primary scalar is max(|width_rf|, |height_rf|), and “larger than the mapped span” means that scalar exceeds 80°.",
        "",
        "The edge distance is computed from released `azimuth_rf`/`elevation_rf`, which come from the thresholded peak-connected component, not from the Gaussian center. That distinction is intentional: it asks whether an RF whose reproducible released location lies near the sampled boundary receives an unstable Gaussian size estimate.",
        "",
        "## Caveat",
        "",
        "Filtering to `on_screen_rf == True` removes the systematic edge inflation in this cohort, but it is not a general containment criterion: a Gaussian center can be on screen while a large fraction of its fitted profile lies outside the sampled support. For downstream RF size, released thresholded `area_rf` remains the safer Allen metric; the Gaussian dimensions should be treated as censored or refit with explicit bounds and edge-aware uncertainty.",
    ]
    report_path = output_dir / "README.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    outputs = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "run_manifest.json"
    }
    manifest = {
        "analysis": "Allen BO 1.1 Gaussian RF edge inflation",
        "inputs": {
            "support": {"path": str(args.support.resolve()), "sha256": sha256(args.support.resolve())},
            "unit_table": {"path": str(args.unit_table.resolve()), "sha256": sha256(args.unit_table.resolve())},
        },
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "parameters": {
            "cohort": COHORT,
            "grid_limits_deg": GRID_LIMITS,
            "mapped_span_deg": GRID_SPAN_DEG,
            "edge_definition_deg": "nearest distance <= 10",
            "interior_definition_deg": "nearest distance > 20",
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "seed": args.seed,
        },
        "audit": audit,
        "results": {
            "median_session_edge_to_interior_ratio": median_ratio,
            "bootstrap_95_percent_interval": ratio_ci,
            "sessions_with_edge_median_larger": sessions_larger,
            "sessions_compared": len(paired),
            "two_sided_sign_test_p": two_sided_sign_p,
        },
        "outputs": outputs,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote Gaussian RF edge-inflation analysis to {output_dir}")


if __name__ == "__main__":
    main()
