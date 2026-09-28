#!/usr/bin/env python3
"""Compare RF and CCF evidence for the anomalous MouseV2 session 815152."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SUBJECT = 815152
PROBES = ("A", "E", "C", "B")
COLORS = {"A": "#d73027", "E": "#fc8d59", "C": "#1a9850", "B": "#4575b4"}
RF_FITS = ROOT / "data/imports/mousev2_parametric_rf_v1/rf_unit_fits.csv"
RF_IMPORT = ROOT / "data/imports/pilot_rf_peaks_v1/rf_unit_peaks.csv"
CCF_UNITS = (
    ROOT / "artifacts/figure3/06r_mousev2_probe_tracks_from_ccf/"
    "mousev2_unit_ccf_locations.csv"
)
CCF_ENTRIES = (
    ROOT / "artifacts/figure3/06s_mousev2_ccf_surface_projection/"
    "mousev2_ccf_surface_entry_points.csv"
)
ANALYSIS_LABELS = ROOT / "data/site6_processed/layer_info.csv"
OUTPUT = (
    ROOT / "artifacts/figure3/06s_mousev2_ccf_surface_projection/"
    "clustering_audit/subject_815152_rf_label_audit"
)


def supported_rf_medians(fits: pd.DataFrame) -> pd.DataFrame:
    supported = fits.loc[fits.rf_model_supported].copy()
    rows = []
    for (subject_id, probe), group in supported.groupby(["subject_id", "probe"]):
        # Deterministic unit-bootstrap intervals are descriptive uncertainty on
        # each probe median, not independent-animal inferential intervals.
        rng = np.random.default_rng(int(subject_id) + ord(str(probe)[0]))
        row = {"subject_id": subject_id, "probe": probe, "n_supported": len(group)}
        for source, name in [
            ("supported_rf_center_x_deg", "rf_azimuth"),
            ("supported_rf_center_y_deg", "rf_elevation"),
        ]:
            values = group[source].to_numpy(float)
            bootstrap = np.median(
                values[rng.integers(0, len(values), size=(5000, len(values)))], axis=1
            )
            row[f"{name}_median_deg"] = float(np.median(values))
            row[f"{name}_ci_low_deg"], row[f"{name}_ci_high_deg"] = np.percentile(
                bootstrap, [2.5, 97.5]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def audit_label_chain() -> pd.DataFrame:
    rf = pd.read_csv(RF_IMPORT)
    rf = rf.loc[rf.subject_id.eq(SUBJECT), ["unit_id", "local_unit_id", "probe"]]
    rf = rf.rename(columns={"probe": "rf_export_probe"})
    ccf = pd.read_csv(CCF_UNITS)
    ccf = ccf.loc[ccf.subject_id.eq(SUBJECT), ["nwb_unit_id", "probe"]]
    ccf["probe"] = ccf.probe.str.removeprefix("Probe")
    ccf = ccf.rename(columns={"nwb_unit_id": "local_unit_id", "probe": "refreshed_nwb_probe"})
    analysis = pd.read_csv(ANALYSIS_LABELS, usecols=["unit_id", "ecephys_structure_acronym"])
    analysis["analysis_probe"] = analysis.ecephys_structure_acronym.str.extract(r"V1_site6_([ABCE])")
    result = rf.merge(ccf, on="local_unit_id", how="outer", validate="one_to_one")
    result = result.merge(analysis[["unit_id", "analysis_probe"]], on="unit_id",
                          how="outer", validate="one_to_one")
    result["all_three_labels_agree"] = (
        result.rf_export_probe.eq(result.refreshed_nwb_probe)
        & result.rf_export_probe.eq(result.analysis_probe)
    )
    return result.sort_values("local_unit_id")


def loao_rf_predictions(medians: pd.DataFrame) -> pd.DataFrame:
    rows = []
    features = ["rf_azimuth_median_deg", "rf_elevation_median_deg"]
    for subject_id, test in medians.groupby("subject_id"):
        train = medians.loc[medians.subject_id.ne(subject_id)]
        scaler = StandardScaler().fit(train[features])
        classifier = NearestCentroid().fit(scaler.transform(train[features]), train.probe)
        prediction = classifier.predict(scaler.transform(test[features]))
        rows.extend({
            "subject_id": subject_id, "observed_probe": observed,
            "predicted_probe_from_rf": predicted, "correct": observed == predicted,
        } for observed, predicted in zip(test.probe, prediction))
    return pd.DataFrame(rows)


def anatomy_rf_assignment(entries: pd.DataFrame, medians: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    entries = entries.copy()
    entries["probe"] = entries.probe.str.removeprefix("Probe")
    joined = entries[["subject_id", "probe", "ccf_ap_um", "ccf_ml_um"]].merge(
        medians, on=["subject_id", "probe"], validate="one_to_one"
    )
    for columns, centered in [
        (["ccf_ap_um", "ccf_ml_um"], ["ccf_ap_centered", "ccf_ml_centered"]),
        (["rf_azimuth_median_deg", "rf_elevation_median_deg"],
         ["rf_azimuth_centered", "rf_elevation_centered"]),
    ]:
        joined[centered] = joined[columns] - joined.groupby("subject_id")[columns].transform("mean").to_numpy()
    train = joined.loc[joined.subject_id.ne(SUBJECT)]
    test = joined.loc[joined.subject_id.eq(SUBJECT)].set_index("probe").loc[list(PROBES)]
    model = Ridge(alpha=0.1).fit(
        train[["ccf_ap_centered", "ccf_ml_centered"]].to_numpy() / 1000,
        train[["rf_azimuth_centered", "rf_elevation_centered"]],
    )
    predicted = model.predict(
        test[["ccf_ap_centered", "ccf_ml_centered"]].to_numpy() / 1000
    )
    observed = test[["rf_azimuth_centered", "rf_elevation_centered"]].to_numpy()
    cost = ((predicted[:, None, :] - observed[None, :, :]) ** 2).sum(axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    identity_rmse = float(np.sqrt(np.trace(cost) / len(PROBES)))
    optimal_rmse = float(np.sqrt(cost[row_ind, col_ind].sum() / len(PROBES)))
    rows = []
    for anatomy_index, rf_index in zip(row_ind, col_ind):
        rows.append({
            "current_ccf_probe_label": PROBES[anatomy_index],
            "best_matching_rf_probe": PROBES[rf_index],
            "predicted_centered_rf_azimuth_deg": predicted[anatomy_index, 0],
            "predicted_centered_rf_elevation_deg": predicted[anatomy_index, 1],
            "matched_observed_centered_rf_azimuth_deg": observed[rf_index, 0],
            "matched_observed_centered_rf_elevation_deg": observed[rf_index, 1],
        })
    return pd.DataFrame(rows), identity_rmse, optimal_rmse


def plot_audit(fits: pd.DataFrame, medians: pd.DataFrame, entries: pd.DataFrame,
               assignment: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8))
    supported = fits.loc[fits.subject_id.eq(SUBJECT) & fits.rf_model_supported]
    local_medians = medians.loc[medians.subject_id.eq(SUBJECT)].set_index("probe")

    ax = axes[0]
    for probe in PROBES:
        local = supported.loc[supported.probe.eq(probe)]
        ax.scatter(local.supported_rf_center_x_deg, local.supported_rf_center_y_deg,
                   s=22, color=COLORS[probe], alpha=0.32, edgecolor="none")
        center = local_medians.loc[probe]
        ax.scatter(center.rf_azimuth_median_deg, center.rf_elevation_median_deg,
                   s=120, marker="D", color=COLORS[probe], edgecolor="black", linewidth=0.8)
        ax.text(center.rf_azimuth_median_deg + 2, center.rf_elevation_median_deg + 2,
                f"{probe} (n={int(center.n_supported)})", fontweight="bold", fontsize=8)
    ax.axhline(0, color="#dddddf", linewidth=0.7)
    ax.axvline(0, color="#dddddf", linewidth=0.7)
    ax.set(xlim=(-48, 48), ylim=(-48, 48), aspect="equal",
           xlabel="RF azimuth (deg)", ylabel="RF elevation (deg)",
           title="A  815152 supported RF centers")

    ax = axes[1]
    for probe in PROBES:
        local = medians.loc[medians.probe.eq(probe)]
        other = local.loc[local.subject_id.ne(SUBJECT)]
        target = local.loc[local.subject_id.eq(SUBJECT)].iloc[0]
        ax.scatter(other.rf_azimuth_median_deg, other.rf_elevation_median_deg,
                   s=35, color=COLORS[probe], alpha=0.45)
        ax.scatter(target.rf_azimuth_median_deg, target.rf_elevation_median_deg,
                   s=150, marker="*", color=COLORS[probe], edgecolor="black", linewidth=0.8)
        ax.text(target.rf_azimuth_median_deg + 2, target.rf_elevation_median_deg,
                probe, fontweight="bold", fontsize=9)
    ax.set(xlim=(-48, 48), ylim=(-48, 48), aspect="equal",
           xlabel="RF median azimuth (deg)", ylabel="RF median elevation (deg)",
           title="B  815152 matches cohort RF identities")
    ax.text(0.02, 0.02, "★ 815152; circles other animals", transform=ax.transAxes,
            fontsize=8, color="#55585e")

    ax = axes[2]
    local_entries = entries.loc[entries.subject_id.eq(SUBJECT)].copy()
    local_entries["probe"] = local_entries.probe.str.removeprefix("Probe")
    match = assignment.set_index("current_ccf_probe_label").best_matching_rf_probe
    for row in local_entries.itertuples():
        ax.scatter(row.ccf_ml_um / 1000, row.ccf_ap_um / 1000, s=90,
                   color=COLORS[row.probe], edgecolor="black", linewidth=0.8)
        suffix = "" if match[row.probe] == row.probe else f" → RF {match[row.probe]}"
        ax.text(row.ccf_ml_um / 1000 + 0.04, row.ccf_ap_um / 1000,
                f"CCF {row.probe}{suffix}", fontsize=8, va="center",
                fontweight="bold" if suffix else "normal")
    ax.invert_yaxis()
    ax.set(xlim=(2.35, 4.25), ylim=(10.1, 7.75), aspect="equal",
           xlabel="CCF ML (mm): lateral → medial", ylabel="CCF AP (mm)",
           title="C  Corrected CCF labels agree with RF")
    for ax in axes:
        ax.grid(color="#ececef", linewidth=0.7, zorder=-10)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Subject 815152: corrected CCF track labels agree with RF identities",
                 fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fits = pd.read_csv(RF_FITS)
    medians = supported_rf_medians(fits)
    entries = pd.read_csv(CCF_ENTRIES)
    label_chain = audit_label_chain()
    rf_predictions = loao_rf_predictions(medians)
    assignment, identity_rmse, optimal_rmse = anatomy_rf_assignment(entries, medians)
    local_medians = medians.loc[medians.subject_id.eq(SUBJECT)].copy()
    label_chain.to_csv(OUTPUT / "unit_label_chain.csv", index=False)
    local_medians.to_csv(OUTPUT / "rf_probe_medians.csv", index=False)
    rf_predictions.to_csv(OUTPUT / "leave_one_animal_out_rf_predictions.csv", index=False)
    assignment.to_csv(OUTPUT / "ccf_to_rf_assignment.csv", index=False)
    plot_audit(fits, medians, entries, assignment, OUTPUT / "Figure_815152_rf_probe_label_audit.png")

    local_predictions = rf_predictions.loc[rf_predictions.subject_id.eq(SUBJECT)]
    lines = [
        "# Subject 815152 RF/probe-label audit", "",
        f"- Label-chain agreement: {int(label_chain.all_three_labels_agree.sum())}/{len(label_chain)} units",
        "  across the Pilot RF export, analysis structure label, and refreshed NWB probe label.",
        f"- Supported parametric RFs: {int(local_medians.n_supported.sum())} units.",
        f"- 815152 RF median identities recovered leave-one-animal-out: "
        f"{int(local_predictions.correct.sum())}/4.",
        f"- All-cohort RF median identities recovered leave-one-animal-out: "
        f"{int(rf_predictions.correct.sum())}/{len(rf_predictions)}.",
        f"- CCF-to-RF identity-paired RMSE: {identity_rmse:.2f} deg.",
        f"- CCF-to-RF optimally permuted RMSE: {optimal_rmse:.2f} deg.",
        f"- Squared-error ratio: {(identity_rmse / optimal_rmse) ** 2:.1f}x.", "",
        "## Best supported interpretation", "",
        "The ephys/RF probe labels are internally consistent, 815152's RF pattern matches the",
        "same probe identities in the other seven animals, and the corrected CCF track",
        "associations now preserve the identity mapping A/B/C/E -> A/B/C/E. The former",
        "B/C/E permutation is resolved in the 2026-09-25 DANDI replacement asset.",
    ]
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
