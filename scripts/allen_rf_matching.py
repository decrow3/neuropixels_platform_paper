#!/usr/bin/env python3
"""Audit achieved Allen V1/HVA receptive-field matching.

This is the first executable Iteration 6C checkpoint.  It does not alter the
released unit table or fit response-property models.  It separates intended
retinotopic targeting from achieved unit RF centers and writes the summaries
needed to specify the subsequent RF-adjusted analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.figure3_mousev2 import FINE_TO_COARSE, load_config  # noqa: E402


DEFAULT_OUTPUT = ROOT / "artifacts" / "figure3" / "06c_allen_rf_matching"
AREA_ORDER = ("V1", "LM", "RL", "AL", "PM", "AM")
HVA_ORDER = AREA_ORDER[1:]
AREA_COLORS = {
    "V1": "#4D4D4D",
    "LM": "#1B9E77",
    "RL": "#D95F02",
    "AL": "#7570B3",
    "PM": "#E7298A",
    "AM": "#66A61E",
}
COHORT_LABELS = {
    "brain_observatory_1.1": "Brain Observatory 1.1",
    "functional_connectivity": "Functional Connectivity",
}
REQUIRED_COLUMNS = {
    "ecephys_unit_id",
    "ecephys_session_id",
    "ecephys_probe_id",
    "specimen_id",
    "session_type",
    "ecephys_structure_acronym",
    "azimuth_rf",
    "elevation_rf",
    "area_rf",
    "p_value_rf",
    "snr",
    "firing_rate_dg",
    "amplitude_cutoff",
    "presence_ratio",
    "isi_violations",
}
POPULATIONS = ("rf_only", "published_like", "intersection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--population",
        choices=POPULATIONS,
        default="published_like",
        help="Population used for center and support summaries.",
    )
    parser.add_argument(
        "--support-quantile",
        type=float,
        default=0.025,
        help="Tail probability removed on each axis for the robust V1 box.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_columns(table: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(table.columns))
    if missing:
        raise ValueError(f"Allen unit table lacks required RF audit columns: {missing}")
    if table["ecephys_unit_id"].duplicated().any():
        raise ValueError("Allen unit table has duplicate ecephys_unit_id values")


def add_population_flags(table: pd.DataFrame) -> pd.DataFrame:
    """Add explicit nested RF/QC population flags without dropping rows."""
    result = table.copy()
    result["area"] = result["ecephys_structure_acronym"].map(FINE_TO_COARSE)
    result["cohort"] = result["session_type"].map(COHORT_LABELS)
    numeric = (
        "azimuth_rf",
        "elevation_rf",
        "area_rf",
        "p_value_rf",
        "snr",
        "firing_rate_dg",
        "amplitude_cutoff",
        "presence_ratio",
        "isi_violations",
    )
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result["target_area"] = result["area"].isin(AREA_ORDER)
    result["finite_rf_center"] = result[["azimuth_rf", "elevation_rf"]].notna().all(axis=1)
    result["significant_rf"] = result["p_value_rf"].lt(0.01)
    result["bounded_rf_area"] = result["area_rf"].lt(2500)
    result["rf_only"] = (
        result["target_area"]
        & result["finite_rf_center"]
        & result["significant_rf"]
        & result["bounded_rf_area"]
    )
    result["published_like"] = (
        result["rf_only"]
        & result["snr"].gt(1)
        & result["firing_rate_dg"].gt(0.1)
    )
    result["common_qc"] = (
        result["amplitude_cutoff"].lt(0.1)
        & result["presence_ratio"].gt(0.8)
        & result["isi_violations"].lt(0.5)
    )
    result["intersection"] = result["published_like"] & result["common_qc"]
    return result


def population_flow(table: pd.DataFrame) -> pd.DataFrame:
    stages = (
        ("target_area", table["target_area"]),
        ("finite_rf_center", table["target_area"] & table["finite_rf_center"]),
        (
            "significant_rf",
            table["target_area"] & table["finite_rf_center"] & table["significant_rf"],
        ),
        ("rf_only", table["rf_only"]),
        ("published_like", table["published_like"]),
        ("intersection", table["intersection"]),
    )
    rows: list[dict[str, object]] = []
    for cohort in COHORT_LABELS.values():
        for area in AREA_ORDER:
            base = table["cohort"].eq(cohort) & table["area"].eq(area)
            for stage, mask in stages:
                rows.append(
                    {
                        "cohort": cohort,
                        "area": area,
                        "stage": stage,
                        "units": int((base & mask).sum()),
                    }
                )
    return pd.DataFrame(rows)


def _dispersion(values: pd.DataFrame) -> pd.Series:
    az = values["azimuth_rf"].to_numpy(dtype=float)
    el = values["elevation_rf"].to_numpy(dtype=float)
    az_median = float(np.median(az))
    el_median = float(np.median(el))
    radius = np.hypot(az - az_median, el - el_median)
    covariance = np.cov(np.column_stack([az, el]), rowvar=False) if len(values) > 1 else np.full((2, 2), np.nan)
    return pd.Series(
        {
            "n_units": len(values),
            "azimuth_median_deg": az_median,
            "azimuth_q25_deg": float(np.quantile(az, 0.25)),
            "azimuth_q75_deg": float(np.quantile(az, 0.75)),
            "elevation_median_deg": el_median,
            "elevation_q25_deg": float(np.quantile(el, 0.25)),
            "elevation_q75_deg": float(np.quantile(el, 0.75)),
            "median_radial_dispersion_deg": float(np.median(radius)),
            "cov_azimuth_deg2": float(covariance[0, 0]),
            "cov_elevation_deg2": float(covariance[1, 1]),
            "cov_azimuth_elevation_deg2": float(covariance[0, 1]),
            "rf_area_median_deg2": float(np.median(values["area_rf"])),
        }
    )


def summarize_groups(units: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    summary = units.groupby(keys, observed=True, sort=True).apply(
        _dispersion, include_groups=False
    )
    return summary.reset_index()


def paired_hva_v1_offsets(session_area: pd.DataFrame) -> pd.DataFrame:
    """Pair robust HVA and V1 RF centers within the same Allen session."""
    identity = ["ecephys_session_id", "specimen_id", "cohort"]
    v1 = session_area.loc[session_area["area"].eq("V1")].copy()
    v1 = v1[identity + ["n_units", "azimuth_median_deg", "elevation_median_deg"]]
    v1 = v1.rename(
        columns={
            "n_units": "v1_n_units",
            "azimuth_median_deg": "v1_azimuth_median_deg",
            "elevation_median_deg": "v1_elevation_median_deg",
        }
    )
    hva = session_area.loc[session_area["area"].isin(HVA_ORDER)].copy()
    paired = hva.merge(v1, on=identity, how="inner", validate="many_to_one")
    paired["delta_azimuth_deg"] = (
        paired["azimuth_median_deg"] - paired["v1_azimuth_median_deg"]
    )
    paired["delta_elevation_deg"] = (
        paired["elevation_median_deg"] - paired["v1_elevation_median_deg"]
    )
    paired["distance_from_v1_deg"] = np.hypot(
        paired["delta_azimuth_deg"], paired["delta_elevation_deg"]
    )
    paired["targeting_rule"] = np.where(
        paired["area"].eq("RL"), "RL geometric-center accommodation", "shared retinotopic rule"
    )
    return paired.sort_values(["cohort", "ecephys_session_id", "area"]).reset_index(drop=True)


def add_session_relative_coordinates(
    units: pd.DataFrame, session_area: pd.DataFrame
) -> pd.DataFrame:
    identity = ["ecephys_session_id", "specimen_id", "cohort"]
    v1 = session_area.loc[session_area["area"].eq("V1"), identity + [
        "azimuth_median_deg", "elevation_median_deg"
    ]].rename(
        columns={
            "azimuth_median_deg": "v1_azimuth_median_deg",
            "elevation_median_deg": "v1_elevation_median_deg",
        }
    )
    result = units.merge(v1, on=identity, how="inner", validate="many_to_one")
    result["relative_azimuth_deg"] = result["azimuth_rf"] - result["v1_azimuth_median_deg"]
    result["relative_elevation_deg"] = result["elevation_rf"] - result["v1_elevation_median_deg"]
    return result


def points_in_convex_hull(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Classify query points against a 2-D reference hull, including edges."""
    reference = np.asarray(reference, dtype=float)
    query = np.asarray(query, dtype=float)
    finite_reference = reference[np.isfinite(reference).all(axis=1)]
    finite_query = np.isfinite(query).all(axis=1)
    inside = np.zeros(len(query), dtype=bool)
    unique = np.unique(finite_reference, axis=0)
    if len(unique) < 3:
        return inside
    try:
        hull = ConvexHull(unique)
    except QhullError:
        return inside
    equations = hull.equations
    inside[finite_query] = np.all(
        query[finite_query] @ equations[:, :-1].T + equations[:, -1] <= 1e-9,
        axis=1,
    )
    return inside


def classify_common_support(
    relative_units: pd.DataFrame, *, tail_probability: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 <= tail_probability < 0.5:
        raise ValueError("support quantile must be in [0, 0.5)")
    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    coordinate_columns = ["relative_azimuth_deg", "relative_elevation_deg"]
    for cohort, cohort_units in relative_units.groupby("cohort", observed=True):
        v1_points = cohort_units.loc[cohort_units["area"].eq("V1"), coordinate_columns].to_numpy(dtype=float)
        finite_v1 = v1_points[np.isfinite(v1_points).all(axis=1)]
        if not len(finite_v1):
            continue
        low = np.quantile(finite_v1, tail_probability, axis=0)
        high = np.quantile(finite_v1, 1 - tail_probability, axis=0)
        for area in AREA_ORDER:
            group = cohort_units.loc[cohort_units["area"].eq(area)].copy()
            points = group[coordinate_columns].to_numpy(dtype=float)
            group["inside_v1_convex_hull"] = points_in_convex_hull(finite_v1, points)
            group["inside_v1_robust_box"] = np.all(
                (points >= low) & (points <= high), axis=1
            )
            frames.append(group)
            summaries.append(
                {
                    "cohort": cohort,
                    "area": area,
                    "n_units": len(group),
                    "inside_v1_convex_hull_fraction": float(group["inside_v1_convex_hull"].mean()) if len(group) else np.nan,
                    "inside_v1_robust_box_fraction": float(group["inside_v1_robust_box"].mean()) if len(group) else np.nan,
                    "v1_azimuth_low_deg": low[0],
                    "v1_azimuth_high_deg": high[0],
                    "v1_elevation_low_deg": low[1],
                    "v1_elevation_high_deg": high[1],
                }
            )
    classified = pd.concat(frames, ignore_index=True) if frames else relative_units.iloc[0:0].copy()
    return classified, pd.DataFrame(summaries)


def paired_offset_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (cohort, area), group in paired.groupby(["cohort", "area"], observed=True):
        rows.append(
            {
                "cohort": cohort,
                "area": area,
                "session_pairs": len(group),
                "delta_azimuth_median_deg": group["delta_azimuth_deg"].median(),
                "delta_elevation_median_deg": group["delta_elevation_deg"].median(),
                "distance_median_deg": group["distance_from_v1_deg"].median(),
                "distance_q25_deg": group["distance_from_v1_deg"].quantile(0.25),
                "distance_q75_deg": group["distance_from_v1_deg"].quantile(0.75),
            }
        )
    return pd.DataFrame(rows)


def render_figure(
    paired: pd.DataFrame,
    offset_summary: pd.DataFrame,
    support_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))

    for area in HVA_ORDER:
        group = paired.loc[paired["area"].eq(area)]
        axes[0].scatter(
            group["delta_azimuth_deg"],
            group["delta_elevation_deg"],
            s=24,
            alpha=0.65,
            color=AREA_COLORS[area],
            label=area,
        )
    axes[0].axhline(0, color="#888888", lw=0.8)
    axes[0].axvline(0, color="#888888", lw=0.8)
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].set_xlabel("HVA − V1 azimuth (deg)")
    axes[0].set_ylabel("HVA − V1 elevation (deg)")
    axes[0].set_title("Paired session RF-center offsets")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)

    values = [paired.loc[paired["area"].eq(area), "distance_from_v1_deg"].dropna() for area in HVA_ORDER]
    boxes = axes[1].boxplot(values, labels=HVA_ORDER, patch_artist=True, showfliers=False)
    for patch, area in zip(boxes["boxes"], HVA_ORDER):
        patch.set_facecolor(AREA_COLORS[area])
        patch.set_alpha(0.65)
    axes[1].set_ylabel("session-center distance from V1 (deg)")
    axes[1].set_title("Achieved offset, not RF size")

    cohorts = list(COHORT_LABELS.values())
    width = 0.36
    x = np.arange(len(HVA_ORDER))
    for index, cohort in enumerate(cohorts):
        lookup = support_summary.loc[support_summary["cohort"].eq(cohort)].set_index("area")
        fractions = [lookup.loc[area, "inside_v1_robust_box_fraction"] if area in lookup.index else np.nan for area in HVA_ORDER]
        axes[2].bar(x + (index - 0.5) * width, fractions, width=width, label=cohort)
    axes[2].set_xticks(x, HVA_ORDER)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("unit fraction inside V1 robust box")
    axes[2].set_title("RF common support by stimulus set")
    axes[2].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Allen achieved receptive-field matching audit", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(
    *,
    population: str,
    units: pd.DataFrame,
    paired: pd.DataFrame,
    offsets: pd.DataFrame,
    support: pd.DataFrame,
    common_support_units: int,
    output_path: Path,
) -> None:
    pooled_offsets = paired.groupby("area", observed=True)["distance_from_v1_deg"].agg(
        session_pairs="size", median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)
    ).reindex(HVA_ORDER)
    lines = [
        "# Iteration 6C — Allen achieved RF matching audit",
        "",
        "## Status: targeting audit implemented; response adjustment pending",
        "",
        "Allen used ISI-derived retinotopic maps to target a common V1-aligned region",
        "in V1, LM, AL, AM, and PM. RL received a documented geometric-center",
        "accommodation because its retinotopic center often lies near the RL–S1 boundary.",
        "This audit tests achieved unit RF centers; it does not reinterpret intended",
        "target coordinates as neural measurements.",
        "",
        f"Primary audit population: `{population}` ({len(units):,} units with finite RF centers).",
        f"Common-support subset: {common_support_units:,} units from sessions containing a valid V1 center.",
        "RF-center dispersion and individual RF area are reported separately.",
        "",
        "## Paired HVA–V1 session-center distances",
        "",
        "| Area | Session pairs | Median (deg) | IQR (deg) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for area, row in pooled_offsets.iterrows():
        lines.append(
            f"| {area} | {int(row.session_pairs)} | {row['median']:.1f} | {row.q25:.1f}–{row.q75:.1f} |"
        )
    lines.extend(
        [
            "",
            "These distances summarize robust session × area centers in screen coordinates",
            "relative to simultaneously recorded V1. They are achieved sampling offsets, not",
            "errors in the ISI map and not estimates of individual RF size.",
            "",
            "## Common-support interpretation",
            "",
            "`rf_common_support_summary.csv` reports two diagnostics after centering every",
            "session on its V1 median: inclusion in the full V1 convex hull and in a robust",
            "axis-aligned V1 box. The robust-box result is deliberately conservative and is",
            "the figure's displayed diagnostic; neither rule is yet a matching estimator.",
            "",
            "## Claim gate",
            "",
            "The audit outputs are sufficient to specify the RF-adjusted response model and",
            "matching/weighting strategy. They do not yet establish that the hierarchy metrics",
            "survive RF adjustment. The next implementation must fit the predeclared",
            "session-aware models and report balance and discarded support.",
            "",
            "## Outputs",
            "",
            "- `rf_population_flow.csv`: nested population counts by cohort and area.",
            "- `rf_probe_summary.csv`: achieved centers and dispersion per probe.",
            "- `rf_session_area_summary.csv`: robust combined session × area summaries.",
            "- `rf_paired_hva_v1_offsets.csv`: paired signed offsets and distances.",
            "- `rf_paired_offset_summary.csv`: cohort × area offset summaries.",
            "- `rf_population_sensitivity.csv`: pooled paired offsets across all declared populations.",
            "- `rf_unit_common_support.csv`: session-centered unit coordinates and support flags.",
            "- `rf_common_support_summary.csv`: cohort × area support fractions.",
            "- `Figure_allen_rf_matching.png`: targeting-audit diagnostic.",
            "- `run_manifest.json`: input checksum and audit parameters.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite non-empty checkpoint {output_dir}; use --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    unit_path = Path(config["allen_unit_table"])
    if not unit_path.is_absolute():
        unit_path = ROOT / unit_path
    raw = pd.read_csv(unit_path, low_memory=False)
    validate_columns(raw)
    flagged = add_population_flags(raw)
    flow = population_flow(flagged)
    selected = flagged.loc[flagged[args.population]].copy()
    keys = ["ecephys_session_id", "specimen_id", "cohort", "area"]

    sensitivity_frames: list[pd.DataFrame] = []
    for population in POPULATIONS:
        population_units = flagged.loc[flagged[population]].copy()
        population_session_area = summarize_groups(population_units, keys)
        population_paired = paired_hva_v1_offsets(population_session_area)
        population_summary = paired_offset_summary(population_paired)
        population_summary.insert(0, "population", population)
        sensitivity_frames.append(population_summary)
    population_sensitivity = pd.concat(sensitivity_frames, ignore_index=True)

    probe_keys = keys + ["ecephys_probe_id"]
    probe_summary = summarize_groups(selected, probe_keys)
    session_area = summarize_groups(selected, keys)
    paired = paired_hva_v1_offsets(session_area)
    offsets = paired_offset_summary(paired)
    relative = add_session_relative_coordinates(selected, session_area)
    classified, support = classify_common_support(
        relative, tail_probability=args.support_quantile
    )

    flow.to_csv(output_dir / "rf_population_flow.csv", index=False)
    probe_summary.to_csv(output_dir / "rf_probe_summary.csv", index=False)
    session_area.to_csv(output_dir / "rf_session_area_summary.csv", index=False)
    paired.to_csv(output_dir / "rf_paired_hva_v1_offsets.csv", index=False)
    offsets.to_csv(output_dir / "rf_paired_offset_summary.csv", index=False)
    population_sensitivity.to_csv(
        output_dir / "rf_population_sensitivity.csv", index=False
    )
    unit_support_columns = [
        "ecephys_unit_id",
        "ecephys_session_id",
        "ecephys_probe_id",
        "specimen_id",
        "cohort",
        "area",
        "azimuth_rf",
        "elevation_rf",
        "area_rf",
        "p_value_rf",
        "v1_azimuth_median_deg",
        "v1_elevation_median_deg",
        "relative_azimuth_deg",
        "relative_elevation_deg",
        "inside_v1_convex_hull",
        "inside_v1_robust_box",
    ]
    classified[unit_support_columns].to_csv(
        output_dir / "rf_unit_common_support.csv", index=False
    )
    support.to_csv(output_dir / "rf_common_support_summary.csv", index=False)
    render_figure(paired, offsets, support, output_dir / "Figure_allen_rf_matching.png")
    write_report(
        population=args.population,
        units=selected,
        paired=paired,
        offsets=offsets,
        support=support,
        common_support_units=len(classified),
        output_path=output_dir / "ALLEN_RF_MATCHING.md",
    )
    output_records = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "run_manifest.json":
            output_records[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    manifest = {
        "checkpoint": "06c_allen_rf_matching",
        "status": "targeting audit implemented; response adjustment pending",
        "input": {"path": str(unit_path), "sha256": sha256(unit_path), "rows": len(raw)},
        "code": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "parameters": {
            "population": args.population,
            "support_tail_probability": args.support_quantile,
            "areas": list(AREA_ORDER),
            "rl_targeting_exception": True,
        },
        "outputs": output_records,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Allen RF matching audit written to {output_dir}")


if __name__ == "__main__":
    main()
