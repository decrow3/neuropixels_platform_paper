#!/usr/bin/env python3
"""Summarize RF-size distributions in the selected trusted-interior Allen cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "artifacts" / "figure3" / "06c_allen_rf_matching"
DEFAULT_UNITS = ROOT / "data" / "unit_table.csv"
DEFAULT_SUPPORT = AUDIT / "rf_unit_common_support.csv"
DEFAULT_SESSIONS = (
    AUDIT
    / "v1_rf_size_translation_fixed_penalty_bound_30"
    / "selected_v1_rf_size_translations.csv"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "allen_trusted_interior_rf_distributions"
AREA_MAP = {
    "VISp": "V1",
    "VISl": "LM",
    "VISrl": "RL",
    "VISal": "AL",
    "VISpm": "PM",
    "VISam": "AM",
}
AREA_ORDER = ("V1", "LM", "RL", "AL", "PM", "AM")
GROUP_ORDER = ("V1", "HVA pooled", "LM", "RL", "AL", "PM", "AM")
GRID_LIMITS = (10.0, 90.0, -30.0, 50.0)
EDGE_EXCLUSION_DEG = 20.0
AREA_CAP_DEG2 = 2500.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-table", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--support", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--selected-sessions", type=Path, default=DEFAULT_SESSIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_geometry(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    az_min, az_max, el_min, el_max = GRID_LIMITS
    result["distance_to_nearest_grid_edge_deg"] = np.minimum.reduce(
        [
            result["azimuth_rf"] - az_min,
            az_max - result["azimuth_rf"],
            result["elevation_rf"] - el_min,
            el_max - result["elevation_rf"],
        ]
    )
    result["equivalent_circle_diameter_deg"] = 2.0 * np.sqrt(result["area_rf"] / np.pi)
    result["width_sigma_magnitude_deg"] = result["width_rf"].abs()
    result["height_sigma_magnitude_deg"] = result["height_rf"].abs()
    result["major_sigma_magnitude_deg"] = result[
        ["width_sigma_magnitude_deg", "height_sigma_magnitude_deg"]
    ].max(axis=1)
    result["minor_sigma_magnitude_deg"] = result[
        ["width_sigma_magnitude_deg", "height_sigma_magnitude_deg"]
    ].min(axis=1)
    result["major_fwhm_deg"] = 2.354820045 * result["major_sigma_magnitude_deg"]
    result["minor_fwhm_deg"] = 2.354820045 * result["minor_sigma_magnitude_deg"]
    result["gaussian_fwhm_area_deg2"] = (
        2.0
        * np.pi
        * np.log(2.0)
        * result["width_sigma_magnitude_deg"]
        * result["height_sigma_magnitude_deg"]
    )
    return result


def prepare_populations(
    units_path: Path, support_path: Path, sessions_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    sessions = set(
        pd.read_csv(sessions_path, usecols=["ecephys_session_id"])["ecephys_session_id"]
        .astype(int)
        .tolist()
    )
    units = pd.read_csv(units_path, low_memory=False)
    units["area"] = units["ecephys_structure_acronym"].map(AREA_MAP)
    numeric = (
        "azimuth_rf", "elevation_rf", "area_rf", "p_value_rf", "snr",
        "firing_rate_dg", "width_rf", "height_rf",
    )
    for column in numeric:
        units[column] = pd.to_numeric(units[column], errors="coerce")
    base = (
        units["session_type"].eq("brain_observatory_1.1")
        & units["ecephys_session_id"].isin(sessions)
        & units["area"].isin(AREA_ORDER)
        & units[["azimuth_rf", "elevation_rf", "area_rf"]].notna().all(axis=1)
        & units["p_value_rf"].lt(0.01)
        & units["snr"].gt(1.0)
        & units["firing_rate_dg"].gt(0.1)
    )
    uncapped = add_geometry(units.loc[base].copy())
    uncapped = uncapped.loc[
        uncapped["distance_to_nearest_grid_edge_deg"].ge(EDGE_EXCLUSION_DEG)
    ].copy()
    trusted = uncapped.loc[uncapped["area_rf"].lt(AREA_CAP_DEG2)].copy()

    support = pd.read_csv(support_path, low_memory=False)
    support = support.loc[
        support["cohort"].eq("Brain Observatory 1.1")
        & support["ecephys_session_id"].isin(sessions)
    ].copy()
    support = add_geometry(
        support.merge(
            units[["ecephys_unit_id", "width_rf", "height_rf", "on_screen_rf"]],
            on="ecephys_unit_id", how="left", validate="one_to_one",
        )
    )
    support = support.loc[
        support["distance_to_nearest_grid_edge_deg"].ge(EDGE_EXCLUSION_DEG)
    ].copy()
    expected_ids = set(support["ecephys_unit_id"].astype(int))
    reconstructed_ids = set(trusted["ecephys_unit_id"].astype(int))
    if expected_ids != reconstructed_ids:
        raise ValueError(
            "Reconstructed trusted-interior cohort differs from alignment support: "
            f"missing={len(expected_ids - reconstructed_ids)}, "
            f"extra={len(reconstructed_ids - expected_ids)}"
        )
    trusted = support
    for frame in (trusted, uncapped):
        frame["pooled_group"] = np.where(frame["area"].eq("V1"), "V1", "HVA pooled")
    audit = {
        "selected_sessions": len(sessions),
        "trusted_units": len(trusted),
        "uncapped_published_like_units": len(uncapped),
        "area_cap_excluded_units": len(uncapped) - len(trusted),
        "trusted_id_reconstruction_exact": True,
    }
    return trusted, uncapped, audit


def group_views(table: pd.DataFrame):
    yield "V1", table.loc[table["area"].eq("V1")]
    yield "HVA pooled", table.loc[table["area"].ne("V1")]
    for area in AREA_ORDER[1:]:
        yield area, table.loc[table["area"].eq(area)]


def distribution_summary(
    table: pd.DataFrame, metrics: dict[str, str], *, population: str
) -> pd.DataFrame:
    rows = []
    for group, selected in group_views(table):
        for metric, unit in metrics.items():
            values = pd.to_numeric(selected[metric], errors="coerce")
            values = values[np.isfinite(values)]
            rows.append(
                {
                    "population": population,
                    "group": group,
                    "metric": metric,
                    "unit": unit,
                    "units": len(values),
                    "sessions": selected.loc[values.index, "ecephys_session_id"].nunique(),
                    "mean": values.mean(),
                    "std": values.std(),
                    "minimum": values.min(),
                    "q25": values.quantile(0.25),
                    "median": values.median(),
                    "q75": values.quantile(0.75),
                    "q90": values.quantile(0.90),
                    "q95": values.quantile(0.95),
                    "q97_5": values.quantile(0.975),
                    "q99": values.quantile(0.99),
                    "maximum": values.max(),
                }
            )
    return pd.DataFrame(rows)


def ecdf(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(pd.to_numeric(values, errors="coerce").dropna().to_numpy(float))
    return x, np.arange(1, len(x) + 1) / len(x)


def render_figure(
    trusted: pd.DataFrame,
    uncapped: pd.DataFrame,
    gaussian: pd.DataFrame,
    path: Path,
) -> None:
    colors = {"V1": "#39738c", "HVA pooled": "#d97736"}
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.6))
    for label in ("V1", "HVA pooled"):
        selected = trusted.loc[trusted["pooled_group"].eq(label), "area_rf"]
        x, y = ecdf(selected)
        axes[0, 0].step(x, y, where="post", color=colors[label], linewidth=2, label=f"{label} (n={len(x):,})")
    axes[0, 0].set(
        xlim=(0, 2500), ylim=(0, 1.01), xlabel="Released threshold area (deg²)",
        ylabel="Cumulative fraction", title="Trusted-interior area distribution",
    )
    axes[0, 0].legend(frameon=False)

    for label in ("V1", "HVA pooled"):
        selected = uncapped.loc[uncapped["pooled_group"].eq(label), "area_rf"]
        x, y = ecdf(selected)
        axes[0, 1].step(x, y, where="post", color=colors[label], linewidth=2, label=f"{label} (n={len(x):,})")
    axes[0, 1].axvline(AREA_CAP_DEG2, color="#333333", linestyle="--", linewidth=1.2, label="Existing 2500 deg² cap")
    axes[0, 1].set(
        xlim=(0, 6000), ylim=(0, 1.01), xlabel="Released threshold area (deg²)",
        ylabel="Cumulative fraction", title="Identical QC and interior gate, area cap removed",
    )
    axes[0, 1].legend(frameon=False)

    for label in ("V1", "HVA pooled"):
        selected = gaussian.loc[gaussian["pooled_group"].eq(label), "major_sigma_magnitude_deg"]
        x, y = ecdf(selected)
        axes[1, 0].step(x, y, where="post", color=colors[label], linewidth=2, label=f"{label} (n={len(x):,})")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set(
        xlim=(3, 350), ylim=(0, 1.01), xlabel="Released max(|width_rf|, |height_rf|) σ (deg)",
        ylabel="Cumulative fraction", title="Gaussian scale among on-screen trusted-interior fits",
    )
    axes[1, 0].legend(frameon=False)

    area_values = [
        gaussian.loc[gaussian["area"].eq(area), "major_sigma_magnitude_deg"].dropna().to_numpy()
        for area in AREA_ORDER
    ]
    box = axes[1, 1].boxplot(area_values, labels=AREA_ORDER, showfliers=False, patch_artist=True)
    for index, patch in enumerate(box["boxes"]):
        patch.set_facecolor("#39738c" if index == 0 else "#d97736")
        patch.set_alpha(0.68)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set(
        xlabel="Visual area", ylabel="Released major Gaussian σ (deg)",
        title="Within-area Gaussian scale (outliers omitted from boxes)",
    )
    for ax in axes.ravel():
        ax.grid(alpha=0.18)
    fig.suptitle(
        "Allen BO 1.1 RF-size distributions in the selected 20° trusted-interior cohort",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def lookup(summary: pd.DataFrame, population: str, group: str, metric: str) -> pd.Series:
    return summary.loc[
        summary["population"].eq(population)
        & summary["group"].eq(group)
        & summary["metric"].eq(metric)
    ].iloc[0]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    trusted, uncapped, audit = prepare_populations(
        args.unit_table.resolve(), args.support.resolve(), args.selected_sessions.resolve()
    )
    area_metrics = {
        "area_rf": "deg2",
        "equivalent_circle_diameter_deg": "deg",
    }
    area_summary = pd.concat(
        [
            distribution_summary(trusted, area_metrics, population="trusted alignment support"),
            distribution_summary(uncapped, area_metrics, population="published-like except area cap"),
        ],
        ignore_index=True,
    )
    gaussian = trusted.loc[
        trusted["on_screen_rf"].fillna(False).astype(bool)
        & trusted[["width_rf", "height_rf"]].notna().all(axis=1)
    ].copy()
    gaussian_metrics = {
        "width_sigma_magnitude_deg": "deg sigma",
        "height_sigma_magnitude_deg": "deg sigma",
        "major_sigma_magnitude_deg": "deg sigma",
        "minor_sigma_magnitude_deg": "deg sigma",
        "major_fwhm_deg": "deg FWHM",
        "minor_fwhm_deg": "deg FWHM",
        "gaussian_fwhm_area_deg2": "deg2 at half maximum",
    }
    gaussian_summary = distribution_summary(
        gaussian, gaussian_metrics, population="trusted + Gaussian center on screen"
    )

    trusted.to_csv(output_dir / "trusted_interior_units.csv", index=False, float_format="%.7g")
    uncapped.to_csv(output_dir / "interior_units_without_area_cap.csv", index=False, float_format="%.7g")
    area_summary.to_csv(output_dir / "area_rf_distribution_summary.csv", index=False, float_format="%.7g")
    gaussian_summary.to_csv(output_dir / "gaussian_dimension_distribution_summary.csv", index=False, float_format="%.7g")
    counts = (
        trusted.groupby(["pooled_group", "area_rf"], observed=True)
        .size().rename("units").reset_index()
    )
    counts.to_csv(output_dir / "trusted_area_rf_counts.csv", index=False)
    figure_path = output_dir / "Figure_trusted_interior_rf_distributions.png"
    render_figure(trusted, uncapped, gaussian, figure_path)

    v1_area = lookup(area_summary, "trusted alignment support", "V1", "area_rf")
    hva_area = lookup(area_summary, "trusted alignment support", "HVA pooled", "area_rf")
    v1_uncapped = lookup(area_summary, "published-like except area cap", "V1", "area_rf")
    hva_uncapped = lookup(area_summary, "published-like except area cap", "HVA pooled", "area_rf")
    v1_sigma = lookup(
        gaussian_summary, "trusted + Gaussian center on screen", "V1", "major_sigma_magnitude_deg"
    )
    hva_sigma = lookup(
        gaussian_summary, "trusted + Gaussian center on screen", "HVA pooled", "major_sigma_magnitude_deg"
    )
    cap_exclusions = uncapped.loc[uncapped["area_rf"].ge(AREA_CAP_DEG2)]
    cap_v1 = int(cap_exclusions["area"].eq("V1").sum())
    cap_hva = int(cap_exclusions["area"].ne("V1").sum())
    report = [
        "# Trusted-interior Allen RF-size distributions",
        "",
        "## Cohort",
        "",
        f"This exactly reconstructs the **{audit['trusted_units']:,}-unit** cohort used by the selected V1 `area_rf` alignment: the same {audit['selected_sessions']} BO 1.1 sessions, published-like RF/QC support, and RF centers at least {EDGE_EXCLUSION_DEG:g}° from every 9×9 mapping-grid boundary. Unit identifiers match the saved alignment support exactly.",
        "",
        "## Released threshold area",
        "",
        f"V1 contains **{int(v1_area.units):,} units** with median `area_rf` **{v1_area['median']:.0f} deg²**, 90th percentile **{v1_area.q90:.0f} deg²**, and 95th percentile **{v1_area.q95:.0f} deg²**. Pooled HVAs contain **{int(hva_area.units):,} units** with median **{hva_area['median']:.0f} deg²**, 90th percentile **{hva_area.q90:.0f} deg²**, and 95th percentile **{hva_area.q95:.0f} deg²**.",
        "",
        f"The trusted source was already conditioned on `area_rf < {AREA_CAP_DEG2:.0f} deg²`. Under identical session, significance, SNR, firing-rate, area, and interior-center criteria but with that one cap removed, V1's 95th/99th percentiles are **{v1_uncapped.q95:.0f}/{v1_uncapped.q99:.0f} deg²** and HVA's are **{hva_uncapped.q95:.0f}/{hva_uncapped.q99:.0f} deg²**. The cap excludes **{cap_v1} V1** and **{cap_hva} HVA** interior units.",
        "",
        "## Released Gaussian dimensions within the trusted cohort",
        "",
        f"After additionally requiring Allen's fitted Gaussian center to be on screen and both dimensions finite, **{len(gaussian):,}/{len(trusted):,} fits** remain. The major-sigma median/90th/95th percentiles are **{v1_sigma['median']:.1f}/{v1_sigma.q90:.1f}/{v1_sigma.q95:.1f}°** in V1 and **{hva_sigma['median']:.1f}/{hva_sigma.q90:.1f}/{hva_sigma.q95:.1f}°** in pooled HVAs.",
        "",
        "The upper Gaussian tail remains implausibly broad even in this interior/on-screen subset, consistent with the previously identified no-baseline failure. These quantiles are descriptive evidence for choosing and testing a V1/HVA-specific regularizer; they should not by themselves be interpreted as biological upper limits.",
        "",
        "## Use for the proposed DC ring",
        "",
        "The central threshold-area distributions support a larger prior scale for HVAs than V1. The exact DC-ring radius still requires a declared size convention (sigma, FWHM diameter, equivalent diameter, or contour area). The held-out-repeat comparison should test candidate radii around these empirical central quantiles rather than fitting the contaminated Gaussian tail.",
    ]
    (output_dir / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    outputs = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "run_manifest.json"
    }
    manifest = {
        "analysis": "trusted-interior Allen RF-size distributions",
        "inputs": {
            "unit_table": {"path": str(args.unit_table.resolve()), "sha256": sha256(args.unit_table.resolve())},
            "alignment_support": {"path": str(args.support.resolve()), "sha256": sha256(args.support.resolve())},
            "selected_sessions": {"path": str(args.selected_sessions.resolve()), "sha256": sha256(args.selected_sessions.resolve())},
        },
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "parameters": {
            "edge_exclusion_deg": EDGE_EXCLUSION_DEG,
            "area_cap_deg2": AREA_CAP_DEG2,
            "published_like_except_area_cap": "p_value_rf < .01; snr > 1; firing_rate_dg > .1; target visual areas",
            "gaussian_gate": "trusted cohort; on_screen_rf true; finite width_rf and height_rf",
        },
        "audit": audit,
        "outputs": outputs,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote trusted-interior RF distributions to {output_dir}")


if __name__ == "__main__":
    main()
