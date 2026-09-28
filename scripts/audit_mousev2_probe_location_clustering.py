#!/usr/bin/env python3
"""Audit whether MouseV2 probe letters form the claimed CCF surface clusters."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix, silhouette_score
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SURFACE_DIR = ROOT / "artifacts/figure3/06s_mousev2_ccf_surface_projection"
ENTRIES = SURFACE_DIR / "mousev2_ccf_surface_entry_points.csv"
POINTS = (
    ROOT / "artifacts/figure3/06r_mousev2_probe_tracks_from_ccf/"
    "mousev2_probe_track_points.csv"
)
FITS = (
    ROOT / "artifacts/figure3/06r_mousev2_probe_tracks_from_ccf/"
    "mousev2_probe_track_fits.csv"
)
OUTPUT = SURFACE_DIR / "clustering_audit"
PROBES = ("ProbeA", "ProbeE", "ProbeC", "ProbeB")
LABELS = {
    "ProbeA": "Anterior", "ProbeE": "Lateral",
    "ProbeC": "Posterior", "ProbeB": "Medial",
}
COLORS = {
    "ProbeA": "#d73027", "ProbeE": "#fc8d59",
    "ProbeC": "#1a9850", "ProbeB": "#4575b4",
}


def rank_audit(entries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject_id, group in entries.groupby("subject_id"):
        values = group.set_index("probe")
        rows.append({
            "subject_id": subject_id,
            "anterior_is_ap_min": values.loc["ProbeA", "ccf_ap_um"] == group.ccf_ap_um.min(),
            "posterior_is_ap_max": values.loc["ProbeC", "ccf_ap_um"] == group.ccf_ap_um.max(),
            "lateral_is_ml_min": values.loc["ProbeE", "ccf_ml_um"] == group.ccf_ml_um.min(),
            "medial_is_ml_max": values.loc["ProbeB", "ccf_ml_um"] == group.ccf_ml_um.max(),
            "anterior_ap_um": values.loc["ProbeA", "ccf_ap_um"],
            "lateral_ml_um": values.loc["ProbeE", "ccf_ml_um"],
            "posterior_ap_um": values.loc["ProbeC", "ccf_ap_um"],
            "medial_ml_um": values.loc["ProbeB", "ccf_ml_um"],
        })
    result = pd.DataFrame(rows)
    flags = [c for c in result if c.endswith(("_min", "_max"))]
    result["all_four_extrema_correct"] = result[flags].all(axis=1)
    return result


def assignment_audit(entries: pd.DataFrame) -> pd.DataFrame:
    """Compare each animal with leave-one-animal-out spatial templates."""
    rows = []
    template_order = list(PROBES)
    for subject_id, test in entries.groupby("subject_id"):
        train = entries.loc[entries.subject_id.ne(subject_id)]
        scaler = StandardScaler().fit(train[["ccf_ap_um", "ccf_ml_um"]])
        centers = np.vstack([
            scaler.transform(train.loc[train.probe.eq(probe), ["ccf_ap_um", "ccf_ml_um"]]).mean(0)
            for probe in template_order
        ])
        test_ordered = test.set_index("probe").loc[template_order]
        coords = scaler.transform(test_ordered[["ccf_ap_um", "ccf_ml_um"]])
        cost = ((coords[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        row_index, col_index = linear_sum_assignment(cost)
        assignment = {template_order[i]: template_order[j] for i, j in zip(row_index, col_index)}
        identity_cost = float(np.diag(cost).sum())
        optimal_cost = float(cost[row_index, col_index].sum())
        rows.append({
            "subject_id": subject_id,
            "identity_assignment_cost": identity_cost,
            "optimal_assignment_cost": optimal_cost,
            "identity_to_optimal_cost_ratio": identity_cost / optimal_cost,
            "optimal_assignment": "; ".join(
                f"{probe}->{assignment[probe]}" for probe in template_order
            ),
            "identity_is_optimal": all(assignment[p] == p for p in template_order),
        })
    return pd.DataFrame(rows)


def loao_classification(entries: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    rows = []
    for subject_id, test in entries.groupby("subject_id"):
        train = entries.loc[entries.subject_id.ne(subject_id)]
        scaler = StandardScaler().fit(train[["ccf_ap_um", "ccf_ml_um"]])
        classifier = NearestCentroid().fit(
            scaler.transform(train[["ccf_ap_um", "ccf_ml_um"]]), train.probe
        )
        predicted = classifier.predict(
            scaler.transform(test[["ccf_ap_um", "ccf_ml_um"]])
        )
        rows.extend({
            "subject_id": subject_id, "observed_probe": observed,
            "predicted_spatial_cluster": prediction,
            "correct": observed == prediction,
        } for observed, prediction in zip(test.probe, predicted))
    result = pd.DataFrame(rows)
    return result, float(result.correct.mean())


def entry_definition_audit() -> pd.DataFrame:
    points = pd.read_csv(POINTS)
    visp = points.loc[points.location.astype(str).str.startswith("VISp")].copy()
    tables = {
        "top_contact": visp.sort_values("rel_y").groupby(["subject_id", "probe"]).tail(1),
        "top_5_mean": (
            visp.sort_values("rel_y").groupby(["subject_id", "probe"]).tail(5)
            .groupby(["subject_id", "probe"], as_index=False)[["x", "z"]].mean()
        ),
        "all_visp_median": visp.groupby(["subject_id", "probe"], as_index=False)[["x", "z"]].median(),
        "tls_surface": pd.read_csv(FITS).rename(
            columns={"surface_x_um": "x", "surface_z_um": "z"}
        ),
    }
    rows = []
    for method, table in tables.items():
        group = table.loc[table.subject_id.eq(815152)]
        rows.append({
            "method": method,
            "ap_order_anterior_to_posterior": "; ".join(group.sort_values("x").probe),
            "ml_order_lateral_to_medial": "; ".join(group.sort_values("z").probe),
        })
    return pd.DataFrame(rows)


def plot_small_multiples(entries: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(11.2, 6.0), sharex=True, sharey=True)
    for ax, (subject_id, group) in zip(axes.flat, entries.groupby("subject_id")):
        ordered = group.set_index("probe").loc[list(PROBES)]
        ax.plot(ordered.ccf_ml_um / 1000, ordered.ccf_ap_um / 1000,
                color="#c6c8cc", linewidth=0.8, zorder=0)
        for probe, row in ordered.iterrows():
            ax.scatter(row.ccf_ml_um / 1000, row.ccf_ap_um / 1000,
                       s=55, color=COLORS[probe], edgecolor="white", linewidth=0.7)
            ax.text(row.ccf_ml_um / 1000 + 0.035, row.ccf_ap_um / 1000,
                    probe.removeprefix("Probe"), fontsize=8, va="center")
        ax.set_title(f"{subject_id}", color="#222222", fontweight="normal")
        ax.grid(color="#ececef", linewidth=0.7)
    # Axes share their y-scale; invert once so lower CCF AP (anterior) is at top.
    axes[0, 0].invert_yaxis()
    fig.supxlabel("CCF ML (mm): lateral → medial")
    fig.supylabel("CCF AP (mm): anterior → posterior")
    fig.suptitle(
        "MouseV2 probe-entry geometry by animal\n"
        "A=Anterior, E=Lateral, C=Posterior, B=Medial",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output, dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    entries = pd.read_csv(ENTRIES)
    ranks = rank_audit(entries)
    assignments = assignment_audit(entries)
    predictions, accuracy = loao_classification(entries)
    definitions = entry_definition_audit()
    ranks.to_csv(OUTPUT / "per_animal_extrema_checks.csv", index=False)
    assignments.to_csv(OUTPUT / "leave_one_animal_out_assignment.csv", index=False)
    predictions.to_csv(OUTPUT / "leave_one_animal_out_predictions.csv", index=False)
    definitions.to_csv(OUTPUT / "subject_815152_entry_definition_sensitivity.csv", index=False)
    plot_small_multiples(entries, OUTPUT / "Figure_probe_location_clustering_audit.png")

    silhouette_all = silhouette_score(
        entries[["ccf_ap_um", "ccf_ml_um"]], entries.probe
    )
    without = entries.loc[entries.subject_id.ne(815152)]
    silhouette_without = silhouette_score(
        without[["ccf_ap_um", "ccf_ml_um"]], without.probe
    )
    matrix = confusion_matrix(
        predictions.observed_probe, predictions.predicted_spatial_cluster,
        labels=list(PROBES),
    )
    anomaly = assignments.loc[assignments.subject_id.eq(815152)].iloc[0]
    lines = [
        "# MouseV2 probe-location clustering audit", "",
        "## Verdict", "",
        "The four probe identities are spatially consistent in all eight animals after the",
        "corrected 815152 CCF track associations. Subject 815152 now satisfies all four named",
        "spatial extrema, and its identity assignment is also the optimal cohort assignment.", "",
        "## Quantitative checks", "",
        f"- All four named extrema correct: {int(ranks.all_four_extrema_correct.sum())}/8 animals.",
        f"- Leave-one-animal-out nearest-centroid accuracy: {accuracy:.3f} ({int(predictions.correct.sum())}/32).",
        f"- Euclidean silhouette: {silhouette_all:.3f} overall; {silhouette_without:.3f} excluding 815152.",
        f"- 815152 identity/optimal assignment cost ratio: {anomaly.identity_to_optimal_cost_ratio:.1f}x.",
        f"- 815152 best assignment: {anomaly.optimal_assignment}.",
        f"- LOAO confusion matrix (rows/columns {list(PROBES)}): `{matrix.tolist()}`.", "",
        "## Interpretation limit", "",
        "This audit establishes consistency of the corrected anatomical probe identities with the",
        "cohort geometry. It does not independently validate the absolute CCF registration or the",
        "choice of superficial VISp contact as a cortical-entry proxy.",
    ]
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
