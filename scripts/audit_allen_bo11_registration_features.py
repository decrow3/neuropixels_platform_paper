#!/usr/bin/env python3
"""Audit non-center Allen BO 1.1 features that could inform session registration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "artifacts" / "figure3" / "06c_allen_rf_matching"
DEFAULT_SUPPORT = AUDIT / "rf_unit_common_support.csv"
DEFAULT_UNITS = ROOT / "data" / "unit_table.csv"
DEFAULT_OUTPUT = AUDIT / "noncenter_registration_feature_audit"
BO_COHORT = "Brain Observatory 1.1"

FEATURES = {
    "area_rf": ("RF area", "RF mapping"),
    "time_to_peak_rf": ("RF response time-to-peak", "RF mapping"),
    "firing_rate_rf": ("RF response rate", "RF mapping"),
    "fano_rf": ("RF Fano factor", "RF mapping"),
    "lifetime_sparseness_rf": ("RF lifetime sparseness", "RF mapping"),
    "time_to_first_spike_fl": ("Flash first-spike latency", "Flash / latency"),
    "time_to_peak_fl": ("Flash time-to-peak", "Flash / latency"),
    "sustained_idx_fl": ("Flash sustained index", "Flash / latency"),
    "firing_rate_fl": ("Flash response rate", "Flash / latency"),
    "fano_fl": ("Flash Fano factor", "Flash / latency"),
    "lifetime_sparseness_fl": ("Flash lifetime sparseness", "Flash / latency"),
    "mod_idx_dg": ("Drifting-grating modulation index", "Grating response"),
    "f1_f0_dg": ("Drifting-grating F1/F0", "Grating response"),
    "g_osi_dg": ("Drifting-grating orientation selectivity", "Grating response"),
    "g_dsi_dg": ("Drifting-grating direction selectivity", "Grating response"),
    "time_to_peak_sg": ("Static-grating time-to-peak", "Grating response"),
    "firing_rate_dg": ("Drifting-grating response rate", "Grating response"),
    "fano_dg": ("Drifting-grating Fano factor", "Grating response"),
    "lifetime_sparseness_dg": ("Drifting-grating lifetime sparseness", "Grating response"),
    "c50_dg": ("Drifting-grating C50", "Grating response"),
    "dorsal_ventral_ccf_coordinate": ("Dorsal–ventral CCF coordinate", "Anatomy / probe"),
    "anterior_posterior_ccf_coordinate": ("Anterior–posterior CCF coordinate", "Anatomy / probe"),
    "left_right_ccf_coordinate": ("Left–right CCF coordinate", "Anatomy / probe"),
    "cortical_depth": ("Cortical depth", "Anatomy / probe"),
    "probe_horizontal_position": ("Probe horizontal position", "Anatomy / probe"),
    "probe_vertical_position": ("Probe vertical position", "Anatomy / probe"),
    "waveform_duration": ("Waveform duration", "Cell physiology"),
    "timescale_ac": ("Autocorrelation timescale", "Cell physiology"),
    "timescale_it": ("Intrinsic timescale", "Cell physiology"),
}
COORDINATES = {
    "azimuth_rf": "RF azimuth",
    "elevation_rf": "RF elevation",
    "rf_eccentricity": "RF eccentricity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--unit-table", type=Path, default=DEFAULT_UNITS)
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


def stratified_rank_association(
    table: pd.DataFrame,
    feature: str,
    coordinate: str,
    *,
    minimum_group_units: int,
) -> tuple[float, int, int, float, float]:
    """Pooled within-session × area rank association and group stability."""
    residual_x = []
    residual_y = []
    group_rhos = []
    for _, group in table.groupby(["ecephys_session_id", "area"], observed=True):
        selected = group[[feature, coordinate]].dropna()
        if len(selected) < minimum_group_units:
            continue
        if selected[feature].nunique() < 2 or selected[coordinate].nunique() < 2:
            continue
        x_rank = selected[feature].rank(method="average", pct=True).to_numpy(float)
        y_rank = selected[coordinate].rank(method="average", pct=True).to_numpy(float)
        residual_x.append(x_rank - x_rank.mean())
        residual_y.append(y_rank - y_rank.mean())
        rho = spearmanr(selected[feature], selected[coordinate]).statistic
        if np.isfinite(rho):
            group_rhos.append(float(rho))
    if not residual_x:
        return np.nan, 0, 0, np.nan, np.nan
    x = np.concatenate(residual_x)
    y = np.concatenate(residual_y)
    rho = float(np.corrcoef(x, y)[0, 1])
    group_rhos_array = np.asarray(group_rhos)
    sign_agreement = float(np.mean(np.sign(group_rhos_array) == np.sign(rho))) if rho else 0.5
    return rho, len(x), len(group_rhos), float(np.median(group_rhos_array)), sign_agreement


def audit_features(table: pd.DataFrame, minimum_group_units: int) -> pd.DataFrame:
    rows = []
    for feature, (label, family) in FEATURES.items():
        feature_values = pd.to_numeric(table[feature], errors="coerce")
        row = {
            "feature": feature,
            "label": label,
            "family": family,
            "coverage_fraction": float(feature_values.notna().mean()),
            "finite_units": int(feature_values.notna().sum()),
            "sessions_with_data": int(table.loc[feature_values.notna(), "ecephys_session_id"].nunique()),
        }
        for coordinate in COORDINATES:
            rho, units, groups, median_group_rho, sign_agreement = stratified_rank_association(
                table,
                feature,
                coordinate,
                minimum_group_units=minimum_group_units,
            )
            row[f"rho_{coordinate}"] = rho
            row[f"units_{coordinate}"] = units
            row[f"groups_{coordinate}"] = groups
            row[f"median_group_rho_{coordinate}"] = median_group_rho
            row[f"sign_agreement_{coordinate}"] = sign_agreement
        rho_columns = [f"rho_{coordinate}" for coordinate in COORDINATES]
        absolute = np.abs([row[column] for column in rho_columns])
        if np.isfinite(absolute).any():
            index = int(np.nanargmax(absolute))
            strongest_coordinate = list(COORDINATES)[index]
            row["strongest_coordinate"] = strongest_coordinate
            row["strongest_rho"] = row[f"rho_{strongest_coordinate}"]
            row["strongest_abs_rho"] = abs(row["strongest_rho"])
            row["strongest_sign_agreement"] = row[f"sign_agreement_{strongest_coordinate}"]
            row["strongest_median_group_rho"] = row[f"median_group_rho_{strongest_coordinate}"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("strongest_abs_rho", ascending=False).reset_index(drop=True)


def render_audit(summary: pd.DataFrame, output_path: Path, top_n: int = 22) -> None:
    selected = summary.head(top_n).iloc[::-1].copy()
    correlation_columns = [f"rho_{coordinate}" for coordinate in COORDINATES]
    correlations = selected[correlation_columns].to_numpy(float)
    figure_height = max(8.0, 0.36 * len(selected) + 2.0)
    fig, axes = plt.subplots(1, 3, figsize=(14.8, figure_height), gridspec_kw={"width_ratios": [2.5, 0.72, 0.72]})
    limit = max(0.2, float(np.nanmax(np.abs(correlations))))
    artist = axes[0].imshow(correlations, aspect="auto", cmap="coolwarm", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
    axes[0].set_yticks(np.arange(len(selected)), selected["label"])
    axes[0].set_xticks(np.arange(len(COORDINATES)), [label.replace("RF ", "") for label in COORDINATES.values()], rotation=25, ha="right")
    axes[0].set_title("Within-session × area rank association")
    for row in range(len(selected)):
        for column in range(len(COORDINATES)):
            axes[0].text(column, row, f"{correlations[row, column]:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(artist, ax=axes[0], fraction=0.035, pad=0.03, label="stratified Spearman-like ρ")
    y = np.arange(len(selected))
    axes[1].barh(y, selected["coverage_fraction"], color="#4C78A8")
    axes[1].set_xlim(0, 1.03)
    axes[1].invert_yaxis()
    axes[1].set_yticks([])
    axes[1].set_xlabel("fraction")
    axes[1].set_title("Coverage")
    axes[1].grid(axis="x", alpha=0.2)
    axes[2].barh(y, selected["strongest_sign_agreement"], color="#F58518")
    axes[2].axvline(0.5, color="#777777", linestyle="--", linewidth=1)
    axes[2].set_xlim(0, 1.03)
    axes[2].invert_yaxis()
    axes[2].set_yticks([])
    axes[2].set_xlabel("fraction")
    axes[2].set_title("Gradient-sign agreement")
    axes[2].grid(axis="x", alpha=0.2)
    fig.suptitle(
        "Allen BO 1.1 candidate non-center registration features\n"
        "RF-supported visual-cortical units; area and session effects removed",
        fontsize=14,
    )
    fig.subplots_adjust(left=0.28, right=0.97, bottom=0.1, top=0.9, wspace=0.38)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, population: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Allen BO 1.1 non-center registration feature audit",
        "",
        f"Population: {len(population):,} RF-supported visual-cortical units across "
        f"{population.ecephys_session_id.nunique()} simultaneous V1/HVA sessions.",
        "Associations rank units within session × area before pooling, so gross session and",
        "area differences do not create the reported gradients. These are screening statistics,",
        "not evidence that any feature can identify a session transform.",
        "",
        "| Candidate | Family | Coverage | Strongest coordinate | ρ | Group sign agreement |",
        "| --- | --- | ---: | --- | ---: | ---: |",
    ]
    for row in summary.head(15).itertuples(index=False):
        lines.append(
            f"| {row.label} | {row.family} | {row.coverage_fraction:.1%} | "
            f"{COORDINATES.get(row.strongest_coordinate, row.strongest_coordinate)} | "
            f"{row.strongest_rho:+.3f} | {row.strongest_sign_agreement:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- RF area is the valid released RF-size measure. `width_rf` and `height_rf` are",
            "  not used because their released values show implausible scales and do not form a",
            "  reliable size/shape decomposition.",
            "- RF-mapping response features can be used as non-center scalar fields, although",
            "  they are not independent of the RF stimulus block.",
            "- Flash latency is stimulus-independent of RF and grating tuning and is therefore",
            "  attractive if its spatial gradient is sufficiently reproducible.",
            "- Drifting-grating modulation/F1–F0 can inform an exploratory transform, but using",
            "  them to fit a transform and then claiming improved TF agreement would be circular.",
            "  They require SF evaluation or explicit cross-fitting.",
            "- CCF/probe coordinates have the strongest available gradients but may encode probe",
            "  trajectory and cortical depth rather than eye/screen displacement. They should be",
            "  combined with physiological fields, not used alone.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    support = pd.read_csv(args.support.resolve(), low_memory=False)
    support = support.loc[support["cohort"].eq(BO_COHORT)].copy()
    columns = ["ecephys_unit_id", *[feature for feature in FEATURES if feature not in support.columns]]
    metrics = pd.read_csv(args.unit_table.resolve(), usecols=columns, low_memory=False)
    population = support.merge(metrics, on="ecephys_unit_id", how="left", validate="one_to_one")
    population["rf_eccentricity"] = np.hypot(
        population["azimuth_rf"].to_numpy(float),
        population["elevation_rf"].to_numpy(float),
    )
    summary = audit_features(population, args.minimum_group_units)
    summary.to_csv(output_dir / "allen_bo11_registration_feature_associations.csv", index=False, float_format="%.6g")
    render_audit(summary, output_dir / "Figure_allen_bo11_registration_feature_audit.png")
    write_report(summary, population, output_dir / "ALLEN_BO11_REGISTRATION_FEATURE_AUDIT.md")
    outputs = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "run_manifest.json":
            outputs[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "checkpoint": "06c_allen_bo11_noncenter_registration_feature_audit",
        "status": "screening audit; no transform fitted",
        "inputs": {
            "support": {"path": str(args.support.resolve()), "sha256": sha256(args.support.resolve())},
            "unit_table": {"path": str(args.unit_table.resolve()), "sha256": sha256(args.unit_table.resolve())},
        },
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "parameters": {
            "cohort": BO_COHORT,
            "features": list(FEATURES),
            "coordinates": list(COORDINATES),
            "minimum_session_area_units": args.minimum_group_units,
            "association": "rank within session × area, center ranks, pool residual ranks",
            "eccentricity_center_deg": [0.0, 0.0],
        },
        "outputs": outputs,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Allen registration-feature audit written to {output_dir}")


if __name__ == "__main__":
    main()
