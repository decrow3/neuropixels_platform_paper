#!/usr/bin/env python3
"""Audit whether Allen Neuropixels LGd sampling can support a population RF map."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "v1_absolute_size_dispersion_translation_checkpoint" / "lgd_map_sufficiency"
UNIT_TABLE = ROOT / "data" / "unit_table.csv"
OFFSET_AUDIT = ROOT / "artifacts" / "v1_absolute_size_dispersion_translation_checkpoint" / "thalamic_offset_corroboration" / "p01_onscreen_LGd_offsets.csv"
CCF = ["anterior_posterior_ccf_coordinate", "left_right_ccf_coordinate", "dorsal_ventral_ccf_coordinate"]
RF = ["azimuth_rf", "elevation_rf"]
CASES = {
    755434585: "strong held-out map progression",
    760345702: "typical anatomical span",
    754829445: "broad multi-probe coverage but failed held-out progression",
}


def nearest_other_animal(table, radii=(100, 150, 200, 250, 300)):
    points = table[CCF].to_numpy(float)
    specimens = table.specimen_id.to_numpy(int)
    counts = np.zeros((len(table), len(radii)), int)
    nearest = np.full(len(table), np.inf)
    for i, point in enumerate(points):
        distance = np.linalg.norm(points-point, axis=1)
        distance[specimens == specimens[i]] = np.inf
        nearest[i] = distance.min()
        for j, radius in enumerate(radii):
            counts[i, j] = np.unique(specimens[distance <= radius]).size
    return nearest, counts


def session_metrics(table, offset_audit):
    rows = []
    for sid, local in table.groupby("ecephys_session_id", observed=True):
        points = local[CCF].to_numpy(float)
        span = np.ptp(points, axis=0)
        probes = local.ecephys_probe_id.nunique()
        pca1 = np.nan
        if len(local) >= 2:
            singular = np.linalg.svd(points-points.mean(axis=0), compute_uv=False)
            pca1 = singular[0]**2 / np.sum(singular**2)
        rows.append({
            "ecephys_session_id": int(sid), "units": len(local), "probes": probes,
            "ap_span_um": span[0], "ml_span_um": span[1], "dv_span_um": span[2],
            "ccf_diagonal_span_um": np.linalg.norm(span), "pca1_variance_fraction": pca1,
            "azimuth_span_deg": np.ptp(local.azimuth_rf), "elevation_span_deg": np.ptp(local.elevation_rf),
        })
    result = pd.DataFrame(rows)
    if offset_audit.exists():
        result = result.merge(
            pd.read_csv(offset_audit)[["ecephys_session_id", "heldout_centered_r2", "split_distance_deg"]],
            on="ecephys_session_id", how="left",
        )
    return result


def trajectory_directions(table):
    rows = []
    for (sid, probe), local in table.groupby(["ecephys_session_id", "ecephys_probe_id"], observed=True):
        if len(local) < 3:
            continue
        points = local[CCF].to_numpy(float); points -= points.mean(axis=0)
        _, _, vectors = np.linalg.svd(points, full_matrices=False)
        direction = vectors[0]
        if direction[2] < 0: direction *= -1
        rows.append({"ecephys_session_id": sid, "ecephys_probe_id": probe,
                     "direction_ap": direction[0], "direction_ml": direction[1], "direction_dv": direction[2],
                     "units": len(local)})
    return pd.DataFrame(rows)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    units = pd.read_csv(UNIT_TABLE, low_memory=False)
    lgd = units.loc[units.ecephys_structure_acronym.eq("LGd") & units[CCF+RF].notna().all(axis=1)].copy()
    significant = lgd.loc[lgd.p_value_rf.lt(.01)].copy()
    clean = significant.loc[significant.on_screen_rf.fillna(False)].copy()
    metrics = session_metrics(clean, OFFSET_AUDIT)
    directions = trajectory_directions(clean)
    metrics.to_csv(OUTPUT / "session_sampling_metrics.csv", index=False)
    directions.to_csv(OUTPUT / "probe_trajectory_directions.csv", index=False)

    nearest, support = nearest_other_animal(clean)
    radii = np.array([100, 150, 200, 250, 300])
    support_rows = []
    for j, radius in enumerate(radii):
        support_rows.append({
            "radius_um": radius, "fraction_with_other_animal": np.mean(support[:, j] >= 1),
            "fraction_with_three_other_animals": np.mean(support[:, j] >= 3),
            "median_other_animals": np.median(support[:, j]),
        })
    support_table = pd.DataFrame(support_rows)
    support_table.to_csv(OUTPUT / "cross_animal_spatial_support.csv", index=False)

    selection = []
    for sid, role in CASES.items():
        row = metrics.loc[metrics.ecephys_session_id.eq(sid)].iloc[0]
        selection.append({"ecephys_session_id": sid, "selection_role": role,
                          "units": int(row.units), "probes": int(row.probes),
                          "ccf_diagonal_span_um": row.ccf_diagonal_span_um,
                          "heldout_centered_r2": row.heldout_centered_r2})
    pd.DataFrame(selection).to_csv(OUTPUT / "concrete_case_selection.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    projections = [(0, 1, "AP", "ML"), (1, 2, "ML", "DV")]
    for col, (a, b, alabel, blabel) in enumerate(projections):
        for row, (value, title, cmap, limits) in enumerate([
            ("azimuth_rf", "RF azimuth", "turbo", (10, 90)),
            ("elevation_rf", "RF elevation", "coolwarm", (-30, 50)),
        ]):
            ax = axes[row, col]
            scatter = ax.scatter(clean[CCF[a]]/1000, clean[CCF[b]]/1000, c=clean[value],
                                 s=18, alpha=.75, cmap=cmap, vmin=limits[0], vmax=limits[1], linewidth=0)
            ax.set_xlabel(f"CCF {alabel} (mm)"); ax.set_ylabel(f"CCF {blabel} (mm)")
            ax.set_title(f"{title}: {alabel}-{blabel}")
            ax.set_aspect("equal"); fig.colorbar(scatter, ax=ax, shrink=.75, label="degrees")
    ax = axes[0, 2]
    ax.plot(support_table.radius_um, support_table.fraction_with_other_animal, "o-", label=">=1 other animal")
    ax.plot(support_table.radius_um, support_table.fraction_with_three_other_animals, "s-", label=">=3 other animals")
    ax.set(xlabel="CCF radius (um)", ylabel="fraction of clean LGd cells", ylim=(0, 1.03), title="Cross-animal spatial support")
    ax.legend(frameon=False)
    ax = axes[1, 2]
    evaluable = metrics.heldout_centered_r2.notna()
    positive = metrics.heldout_centered_r2 > 0
    ax.scatter(metrics.ccf_diagonal_span_um, metrics.heldout_centered_r2,
               c=np.where(metrics.probes >= 2, "#d62728", "#1f77b4"), s=35+metrics.units*2, alpha=.8)
    ax.axhline(0, color=".7", lw=1)
    ax.set(xlabel="within-session CCF diagonal span (um)", ylabel="LOAO centered map R2",
           title="Released-map recovery by session")
    ax.text(.98, .03, f"positive: {positive.sum()}/{evaluable.sum()} evaluable sessions\nred: >=2 LGd probes",
            transform=ax.transAxes, ha="right", va="bottom")
    fig.tight_layout(); fig.savefig(OUTPUT / "Figure_LGd_population_sampling_audit.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(len(CASES), 2, figsize=(10, 12))
    for row, (sid, role) in enumerate(CASES.items()):
        local = clean.loc[clean.ecephys_session_id.eq(sid)]
        for col, (value, title, cmap, limits) in enumerate([
            ("azimuth_rf", "azimuth", "turbo", (10, 90)),
            ("elevation_rf", "elevation", "coolwarm", (-30, 50)),
        ]):
            ax = axes[row, col]
            sc = ax.scatter(local[CCF[1]]/1000, local[CCF[2]]/1000, c=local[value], s=55,
                            cmap=cmap, vmin=limits[0], vmax=limits[1], edgecolor="k", linewidth=.25)
            for probe, probe_data in local.groupby("ecephys_probe_id"):
                ordered = probe_data.sort_values("probe_vertical_position")
                ax.plot(ordered[CCF[1]]/1000, ordered[CCF[2]]/1000, color=".4", lw=.7, zorder=0)
            ax.set_aspect("equal"); ax.set_xlabel("CCF ML (mm)"); ax.set_ylabel("CCF DV (mm)")
            ax.set_title(f"{sid}: {title}\n{role}", fontsize=10)
            fig.colorbar(sc, ax=ax, shrink=.7, label="degrees")
    fig.tight_layout(); fig.savefig(OUTPUT / "Figure_LGd_concrete_sessions.png", dpi=180); plt.close(fig)

    centered = clean[CCF].to_numpy(float)
    session_ids = clean.ecephys_session_id.to_numpy(int)
    for sid in np.unique(session_ids): centered[session_ids == sid] -= centered[session_ids == sid].mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    direction_matrix = directions[["direction_ap", "direction_ml", "direction_dv"]].to_numpy(float)
    direction_eigenvalues = np.linalg.eigvalsh(direction_matrix.T @ direction_matrix / len(direction_matrix))
    summary = {
        "all_lgd_units": int(len(units.loc[units.ecephys_structure_acronym.eq('LGd')])),
        "lgd_with_released_rf_and_ccf": int(len(lgd)),
        "p01_released_rf_units": int(len(significant)),
        "p01_on_screen_units": int(len(clean)),
        "p01_on_screen_sessions": int(clean.ecephys_session_id.nunique()),
        "median_clean_units_per_session": float(clean.groupby('ecephys_session_id').size().median()),
        "multi_probe_sessions": int((clean.groupby('ecephys_session_id').ecephys_probe_id.nunique() >= 2).sum()),
        "median_nearest_other_animal_um": float(np.median(nearest)),
        "within_session_design_variance_fractions": (singular**2/np.sum(singular**2)).tolist(),
        "probe_direction_second_moment_eigenvalues": direction_eigenvalues.tolist(),
        "positive_heldout_map_sessions": int((metrics.heldout_centered_r2 > 0).sum()),
        "heldout_map_sessions": int(metrics.heldout_centered_r2.notna().sum()),
        "median_heldout_centered_r2": float(metrics.heldout_centered_r2.median()),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2)); print("\n", support_table.to_string(index=False))


if __name__ == "__main__":
    main()
