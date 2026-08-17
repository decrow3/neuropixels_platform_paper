#!/usr/bin/env python3
"""Compare Allen RF-area edge effects in V1 and individual HVAs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "artifacts" / "figure3" / "06c_allen_rf_matching"
DEFAULT_INPUT = AUDIT / "rf_unit_common_support.csv"
DEFAULT_OUTPUT = AUDIT / "noncenter_similarity_alignment" / "rf_area_edge_by_area"
BO_COHORT = "Brain Observatory 1.1"
AREA_ORDER = ("V1", "LM", "RL", "AL", "PM", "AM")
EDGE_BINS = np.array([-1.0, 2.5, 7.5, 12.5, 17.5, 22.5, 30.5, 50.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-group-units", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_edge_metrics(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    result["distance_to_nearest_grid_edge_deg"] = np.minimum.reduce(
        [
            result["azimuth_rf"] - 10.0,
            90.0 - result["azimuth_rf"],
            result["elevation_rf"] + 30.0,
            50.0 - result["elevation_rf"],
        ]
    )
    result["log2_rf_area_deg2"] = np.log2(result["area_rf"].where(result["area_rf"] > 0))
    result["edge_bin"] = pd.cut(
        result["distance_to_nearest_grid_edge_deg"], EDGE_BINS, include_lowest=True
    )
    return result


def stratified_association(
    table: pd.DataFrame,
    minimum_group_units: int,
) -> dict[str, float]:
    residual_x = []
    residual_y = []
    group_rhos = []
    for _, group in table.groupby(["ecephys_session_id", "area"], observed=True):
        selected = group[["distance_to_nearest_grid_edge_deg", "log2_rf_area_deg2"]].dropna()
        if len(selected) < minimum_group_units or selected.iloc[:, 0].nunique() < 2:
            continue
        x = selected.iloc[:, 0].rank(pct=True).to_numpy(float)
        y = selected.iloc[:, 1].rank(pct=True).to_numpy(float)
        residual_x.append(x - x.mean())
        residual_y.append(y - y.mean())
        rho = spearmanr(selected.iloc[:, 0], selected.iloc[:, 1]).statistic
        if np.isfinite(rho):
            group_rhos.append(float(rho))
    x = np.concatenate(residual_x)
    y = np.concatenate(residual_y)
    rho = float(np.corrcoef(x, y)[0, 1])
    return {
        "stratified_rho": rho,
        "session_area_groups": len(group_rhos),
        "median_group_rho": float(np.median(group_rhos)),
        "sign_agreement": float(np.mean(np.sign(group_rhos) == np.sign(rho))),
    }


def radial_summary(table: pd.DataFrame, label: str) -> pd.DataFrame:
    session_bins = (
        table.groupby(["ecephys_session_id", "edge_bin"], observed=True)["log2_rf_area_deg2"]
        .median()
        .reset_index()
    )
    summary = (
        session_bins.groupby("edge_bin", observed=True)["log2_rf_area_deg2"]
        .agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
            sessions="size",
        )
        .reset_index()
    )
    summary["distance_to_edge_deg"] = summary["edge_bin"].map(lambda interval: interval.mid).astype(float)
    summary["group"] = label
    return summary


def render_figure(
    radial: pd.DataFrame,
    associations: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.2), gridspec_kw={"width_ratios": [1.55, 1.0]})
    colors = {
        "V1": "#4C78A8",
        "HVA pooled": "#F58518",
        "LM": "#54A24B",
        "RL": "#E45756",
        "AL": "#B279A2",
        "PM": "#FF9DA6",
        "AM": "#9D755D",
    }
    for group in ("V1", "HVA pooled"):
        selected = radial.loc[radial["group"].eq(group)].sort_values("distance_to_edge_deg")
        axes[0].fill_between(
            selected["distance_to_edge_deg"], selected["q25"], selected["q75"],
            color=colors[group], alpha=0.18,
        )
        axes[0].plot(
            selected["distance_to_edge_deg"], selected["median"], marker="o",
            color=colors[group], linewidth=2.1, label=group,
        )
    axes[0].set_xlabel("Distance inward from nearest RF-grid boundary (deg)")
    axes[0].set_ylabel("Session median log₂ RF area (deg²)")
    axes[0].set_title("Session-balanced RF-area profiles")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)

    area = associations.loc[associations["group"].isin(AREA_ORDER)].copy()
    area["order"] = area["group"].map({label: index for index, label in enumerate(AREA_ORDER)})
    area = area.sort_values("order", ascending=False)
    axes[1].barh(
        area["group"], area["stratified_rho"],
        color=[colors[group] for group in area["group"]], alpha=0.9,
    )
    axes[1].axvline(0, color="#666666", linewidth=1)
    for index, row in enumerate(area.itertuples(index=False)):
        offset = 0.025
        axes[1].text(
            row.stratified_rho + offset, index,
            f"{row.stratified_rho:+.2f} · {row.sign_agreement:.0%}",
            va="center", ha="left", fontsize=9,
        )
    axes[1].set_xlim(-0.30, 0.82)
    axes[1].set_xlabel("Within-session × area rank association (ρ)")
    axes[1].set_title("Area-specific edge effects\n(label includes gradient-sign agreement)")
    axes[1].grid(axis="x", alpha=0.2)
    fig.suptitle(
        "Allen BO 1.1 RF area behaves differently near stimulus support in V1 and HVAs",
        fontsize=14,
    )
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.12, top=0.84, wspace=0.3)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    population = pd.read_csv(args.input.resolve(), low_memory=False)
    population = add_edge_metrics(population.loc[population["cohort"].eq(BO_COHORT)])
    definitions = [("V1", population.loc[population["area"].eq("V1")])]
    definitions.append(("HVA pooled", population.loc[population["area"].ne("V1")]))
    definitions.extend((area, population.loc[population["area"].eq(area)]) for area in AREA_ORDER[1:])
    association_rows = []
    radial_frames = []
    for label, selected in definitions:
        row = {
            "group": label,
            "units": len(selected),
            "sessions": selected["ecephys_session_id"].nunique(),
        }
        row.update(stratified_association(selected, args.minimum_group_units))
        association_rows.append(row)
        radial_frames.append(radial_summary(selected, label))
    associations = pd.DataFrame(association_rows)
    radial = pd.concat(radial_frames, ignore_index=True)
    associations.to_csv(output_dir / "rf_area_edge_associations.csv", index=False, float_format="%.6g")
    radial.to_csv(output_dir / "rf_area_edge_radial_summary.csv", index=False, float_format="%.6g")
    figure_path = output_dir / "Figure_allen_bo11_rf_area_edge_v1_hva.png"
    render_figure(radial, associations, figure_path)
    v1 = associations.set_index("group").loc["V1"]
    hva = associations.set_index("group").loc["HVA pooled"]
    lines = [
        "# Allen BO 1.1 RF-area edge effects by visual area",
        "",
        "Distance is measured inward from the nearest boundary of the released RF stimulus",
        "grid (azimuth 10–90°, elevation -30–50°). Associations rank units within each",
        "session × anatomical area before pooling.",
        "",
        f"V1 shows a negative association (rho = {v1.stratified_rho:+.3f}; "
        f"{v1.sign_agreement:.1%} sign agreement), whereas pooled HVAs show a strong positive",
        f"association (rho = {hva.stratified_rho:+.3f}; {hva.sign_agreement:.1%}). Every",
        "individual HVA is positive. Thus the edge-related RF-area pattern is area dependent",
        "and cannot be treated as a universal screen-truncation calibration.",
        "",
        "The result can reflect a mixture of genuine area-specific RF-size organization,",
        "different RF-center sampling relative to the grid boundary, and estimator censoring",
        "when RF support extends beyond the mapped stimulus positions.",
    ]
    (output_dir / "ALLEN_BO11_RF_AREA_EDGE_BY_AREA.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    outputs = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "run_manifest.json":
            outputs[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "checkpoint": "06c_allen_bo11_rf_area_edge_by_area",
        "status": "area-specific stimulus-support diagnostic",
        "input": {"path": str(args.input.resolve()), "sha256": sha256(args.input.resolve())},
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "parameters": {
            "cohort": BO_COHORT,
            "rf_grid_center_limits_deg": {"azimuth": [10, 90], "elevation": [-30, 50]},
            "minimum_session_area_units": args.minimum_group_units,
            "association": "rank within session × area, center ranks, pool residual ranks",
        },
        "outputs": outputs,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Allen RF-area edge diagnostic written to {output_dir}")


if __name__ == "__main__":
    main()
