#!/usr/bin/env python3
"""Estimate Allen probe insertion angle directly from per-unit CCF coordinates.

Unlike MouseV2 (no CCF/area labels at all, see `compute_mousev2_csd_insertion_angle.py` and
`compare_rf_depth_span_mousev2_vs_allen.py` for indirect CSD- and RF-span-based angle proxies),
every Allen unit ships with a real 3D common-coordinate-framework position
(anterior_posterior/dorsal_ventral/left_right_ccf_coordinate, in um). Units recorded on the same
probe sit at different points along that probe's physical shank, so their CCF positions should
trace out a straight line through the brain -- the probe's own insertion trajectory. This fits
that line directly (total least squares / PCA) per probe and reports:
  - angle_from_vertical_deg: angle between the fitted 3D line and the CCF dorsal-ventral axis
    (the standard "how far off perpendicular-to-cortical-surface" insertion angle).
  - azimuth_from_anterior_deg: compass direction (0=anterior, +90=lateral/right, 180=posterior,
    -90=medial/left) of the horizontal lean from probe tip toward the surface entry point.
  - r2_colinearity: fraction of positional variance explained by the fitted line (PC1 of the
    3-D SVD) -- near 1.0 for a genuinely straight probe track; low values flag registration noise
    or a probe whose units don't form a clean line and should not be trusted.
  - ccf_3d_path_length_um vs. depth_range_um (electrode-position span along the probe): these
    should closely track each other for any real straight rigid probe, independent of the fitted
    angle, and serve as a built-in sanity check on the whole approach.

Allen's Neuropixels visual coding rig inserts a fixed 6-probe montage (probeA-probeF) at
consistent nominal angles/azimuths across sessions, so per-probe-letter consistency across many
independent sessions is itself a validation of the method (not something assumed by it).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UNITS = ROOT / "data" / "unit_table.csv"
OUTPUT = ROOT / "artifacts" / "figure3" / "06k_allen_probe_insertion_angle_from_ccf"

CCF_COLUMNS = (
    "anterior_posterior_ccf_coordinate",
    "dorsal_ventral_ccf_coordinate",
    "left_right_ccf_coordinate",
)
MIN_UNITS_PER_PROBE = 20
PROBE_LETTER_COLORS = {
    "probeA": "#d73027", "probeB": "#fc8d59", "probeC": "#66bd63",
    "probeD": "#1a9850", "probeE": "#4575b4", "probeF": "#8073ac",
}
PROBE_LETTER_ORDER = ("probeA", "probeB", "probeC", "probeD", "probeE", "probeF")


def fit_probe_line(points_um: np.ndarray, depth_um: np.ndarray) -> dict:
    """Total-least-squares line fit through 3D CCF points, oriented tip->surface via depth."""
    centroid = points_um.mean(axis=0)
    centered = points_um - centroid
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    t = centered @ direction
    # orient so increasing t follows increasing probe_vertical_position (tip -> surface, larger
    # probe_vertical_position = closer to the surface, per register_mousev2_units_along_probe_shank.py)
    if np.corrcoef(t, depth_um)[0, 1] < 0:
        direction = -direction
        t = -t
    r2_colinearity = float(singular_values[0] ** 2 / np.sum(singular_values ** 2))

    ap, dv, lr = direction
    angle_from_vertical_deg = float(np.degrees(np.arccos(np.clip(abs(dv), -1.0, 1.0))))
    azimuth_from_anterior_deg = float(np.degrees(np.arctan2(lr, -ap)))

    return {
        "direction": direction,
        "centroid": centroid,
        "r2_colinearity": r2_colinearity,
        "angle_from_vertical_deg": angle_from_vertical_deg,
        "azimuth_from_anterior_deg": azimuth_from_anterior_deg,
        "ccf_3d_path_length_um": float(t.max() - t.min()),
        "depth_range_um": float(depth_um.max() - depth_um.min()),
    }


def build_table(units: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, dict]]:
    rows = []
    fits_by_probe: dict[int, dict] = {}
    for probe_id, group in units.groupby("ecephys_probe_id", observed=True):
        if len(group) < MIN_UNITS_PER_PROBE:
            continue
        points_um = group[list(CCF_COLUMNS)].to_numpy(dtype=float)
        depth_um = group["probe_vertical_position"].to_numpy(dtype=float)
        fit = fit_probe_line(points_um, depth_um)
        primary_structure = group["ecephys_structure_acronym"].mode().iat[0]
        probe_letter = group["name"].iat[0]
        session_id = int(group["ecephys_session_id"].iat[0])
        rows.append(
            {
                "ecephys_probe_id": int(probe_id),
                "ecephys_session_id": session_id,
                "session_type": group["session_type"].iat[0],
                "probe_letter": probe_letter,
                "n_units": len(group),
                "primary_structure": primary_structure,
                "n_structures_traversed": int(group["ecephys_structure_acronym"].nunique()),
                "angle_from_vertical_deg": fit["angle_from_vertical_deg"],
                "azimuth_from_anterior_deg": fit["azimuth_from_anterior_deg"],
                "r2_colinearity": fit["r2_colinearity"],
                "ccf_3d_path_length_um": fit["ccf_3d_path_length_um"],
                "depth_range_um": fit["depth_range_um"],
            }
        )
        fits_by_probe[int(probe_id)] = {
            "direction": fit["direction"],
            "centroid": fit["centroid"],
            "points_um": points_um,
            "probe_letter": probe_letter,
            "session_id": session_id,
            "path_length_um": fit["ccf_3d_path_length_um"],
        }
    return pd.DataFrame(rows), fits_by_probe


def make_figure(table: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    rng = np.random.default_rng(0)

    # A: angle from vertical by probe letter
    ax = axes[0, 0]
    for index, letter in enumerate(PROBE_LETTER_ORDER):
        values = table.loc[table.probe_letter.eq(letter), "angle_from_vertical_deg"].to_numpy()
        if len(values) == 0:
            continue
        jitter = rng.uniform(-0.18, 0.18, len(values))
        ax.scatter(index + jitter, values, s=22, alpha=0.65,
                   color=PROBE_LETTER_COLORS[letter], edgecolor="none")
        ax.hlines(np.median(values), index - 0.28, index + 0.28, color="black", lw=2, zorder=3)
    ax.set_xticks(range(len(PROBE_LETTER_ORDER)))
    ax.set_xticklabels([letter.replace("probe", "") for letter in PROBE_LETTER_ORDER])
    ax.set(xlabel="probe", ylabel="angle from vertical (deg)",
           title="Insertion angle by probe letter\n(fitted from per-unit CCF coordinates)")
    ax.grid(axis="y", alpha=0.2)

    # B: azimuth polar scatter by probe letter
    ax = axes[0, 1]
    ax.remove()
    ax = fig.add_subplot(2, 2, 2, projection="polar")
    for letter in PROBE_LETTER_ORDER:
        sub = table.loc[table.probe_letter.eq(letter)]
        if len(sub) == 0:
            continue
        theta = np.radians(sub["azimuth_from_anterior_deg"].to_numpy())
        r = sub["angle_from_vertical_deg"].to_numpy()
        ax.scatter(theta, r, s=22, alpha=0.65, color=PROBE_LETTER_COLORS[letter],
                   edgecolor="none", label=letter.replace("probe", ""))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("Tilt direction (radius = angle from vertical)\n0°=anterior, 90°=lateral/right", pad=20)
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.05), frameon=False, fontsize=8, title="probe")

    # C: histogram of angle from vertical, split by session_type
    ax = axes[1, 0]
    session_types = sorted(table.session_type.unique())
    colors = {"brain_observatory_1.1": "#6F63A6", "functional_connectivity": "#B07AA1"}
    bins = np.linspace(0, table.angle_from_vertical_deg.max() * 1.05, 24)
    for session_type in session_types:
        values = table.loc[table.session_type.eq(session_type), "angle_from_vertical_deg"]
        ax.hist(values, bins=bins, alpha=0.55, label=f"{session_type} (n={len(values)})",
                color=colors.get(session_type, "gray"))
    ax.set(xlabel="angle from vertical (deg)", ylabel="probes",
           title="Angle distribution across all probes")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    # D: sanity check -- CCF 3D path length should track electrode depth range
    ax = axes[1, 1]
    ax.scatter(table.depth_range_um / 1000.0, table.ccf_3d_path_length_um / 1000.0,
               s=18, alpha=0.55, c=table.r2_colinearity, cmap="viridis", vmin=0.9, vmax=1.0)
    lim = max(table.depth_range_um.max(), table.ccf_3d_path_length_um.max()) / 1000.0 * 1.05
    ax.plot([0, lim], [0, lim], color="black", lw=1, linestyle="--", zorder=1)
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="electrode depth range (mm)",
           ylabel="fitted CCF 3D path length (mm)",
           title="Sanity check: fitted line length vs. electrode span\n(color = fit r²)")
    fig.colorbar(ax.collections[0], ax=ax, fraction=0.046, label="r² colinearity")
    ax.grid(alpha=0.2)

    fig.suptitle(
        f"Allen probe insertion angle from per-unit CCF trace (n={len(table)} probes, "
        f"≥{MIN_UNITS_PER_PROBE} units each)",
        fontsize=14, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _set_equal_3d_aspect(ax, points_mm: np.ndarray) -> None:
    spans = points_mm.max(axis=0) - points_mm.min(axis=0)
    spans = np.where(spans > 1e-6, spans, 1e-6)
    ax.set_box_aspect(tuple(spans))


def make_3d_figure(
    table: pd.DataFrame, fits_by_probe: dict[int, dict], output_path: Path
) -> None:
    fig = plt.figure(figsize=(14, 7))

    # left: one example session's actual probe tracks in real CCF space
    session_probe_counts = table.groupby("ecephys_session_id")["probe_letter"].nunique()
    full_sessions = session_probe_counts.loc[session_probe_counts.eq(6)].index
    session_min_r2 = (
        table.loc[table.ecephys_session_id.isin(full_sessions)]
        .groupby("ecephys_session_id")["r2_colinearity"]
        .min()
    )
    rep_session_id = int(session_min_r2.idxmax())
    rep_probes = table.loc[table.ecephys_session_id.eq(rep_session_id)]

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    all_points_mm = []
    for _, row in rep_probes.iterrows():
        fit = fits_by_probe[int(row["ecephys_probe_id"])]
        color = PROBE_LETTER_COLORS[row["probe_letter"]]
        points_mm = fit["points_um"] / 1000.0
        all_points_mm.append(points_mm)
        ax.scatter(points_mm[:, 0], points_mm[:, 2], points_mm[:, 1],
                   s=8, alpha=0.45, color=color, edgecolor="none")
        centroid_mm = fit["centroid"] / 1000.0
        direction = fit["direction"]
        t = (fit["points_um"] - fit["centroid"]) @ direction / 1000.0
        p0_mm = centroid_mm + t.min() * direction
        p1_mm = centroid_mm + t.max() * direction
        ax.plot([p0_mm[0], p1_mm[0]], [p0_mm[2], p1_mm[2]], [p0_mm[1], p1_mm[1]],
                color=color, linewidth=2.5, label=row["probe_letter"].replace("probe", ""))
        ax.scatter([p1_mm[0]], [p1_mm[2]], [p1_mm[1]], color=color, marker="^", s=50, zorder=5)
    _set_equal_3d_aspect(ax, np.concatenate(all_points_mm, axis=0)[:, [0, 2, 1]])
    ax.invert_zaxis()
    ax.set(xlabel="AP (mm, + posterior)", ylabel="LR (mm, + right)", zlabel="DV (mm, + ventral)")
    ax.set_title(f"Example session {rep_session_id}\nprobe tracks in real CCF space\n(▲ = surface entry)", fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.05, 1.0), frameon=False, fontsize=8, title="probe")

    # right: every probe's insertion direction as a vector from a common origin
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    vector_length_mm = 3.0
    for probe_id, fit in fits_by_probe.items():
        color = PROBE_LETTER_COLORS[fit["probe_letter"]]
        tip = fit["direction"] * vector_length_mm
        ax.plot([0, tip[0]], [0, tip[2]], [0, tip[1]], color=color, alpha=0.3, linewidth=1.0)
    ax.plot([0, 0], [0, 0], [0, -vector_length_mm], color="black", linewidth=1.5,
            linestyle="--", label="vertical (DV axis)")
    ax.set_box_aspect((1, 1, 1))
    ax.invert_zaxis()
    ax.set(xlabel="AP (mm)", ylabel="LR (mm)", zlabel="DV (mm)",
           xlim=(-vector_length_mm, vector_length_mm), ylim=(-vector_length_mm, vector_length_mm),
           zlim=(-vector_length_mm, vector_length_mm))
    ax.set_title(f"All {len(fits_by_probe)} probes: insertion direction\nfrom a common origin (tip → surface)", fontsize=11)
    handles = [
        plt.Line2D([0], [0], color=PROBE_LETTER_COLORS[letter], lw=2, label=letter.replace("probe", ""))
        for letter in PROBE_LETTER_ORDER
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.05, 1.0), frameon=False, fontsize=8, title="probe")

    fig.suptitle("Allen probe insertion trajectories in 3D CCF space", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(table: pd.DataFrame, output_path: Path) -> None:
    overall = table.angle_from_vertical_deg
    by_letter = (
        table.groupby("probe_letter")["angle_from_vertical_deg"]
        .agg(["median", "std", "count"])
        .reindex(PROBE_LETTER_ORDER)
    )
    low_r2 = table.loc[table.r2_colinearity < 0.95]
    lines = [
        "# Allen probe insertion angle from per-unit CCF trace",
        "",
        f"Fitted {len(table)} probes (>= {MIN_UNITS_PER_PROBE} CCF-complete good-quality units each) "
        "by total-least-squares line through each probe's own unit CCF coordinates.",
        "",
        f"- Overall median angle from vertical: {overall.median():.1f} deg "
        f"(IQR {overall.quantile(0.25):.1f}-{overall.quantile(0.75):.1f}).",
        f"- Median fit quality (r2 colinearity): {table.r2_colinearity.median():.4f}; "
        f"{len(low_r2)}/{len(table)} probes fell below 0.95 and should be treated cautiously.",
        "",
        "## Median angle from vertical by probe letter",
        "",
        by_letter.to_string(),
        "",
        "## Outputs",
        "",
        "- `allen_probe_insertion_angle_from_ccf.csv`: per-probe angle, azimuth, fit quality, and sanity-check columns.",
        "- `Figure_allen_probe_insertion_angle.png`: summary figure (angle by probe letter, azimuth rose plot, "
        "session-type histogram, and fit sanity check).",
        "- `Figure_allen_probe_insertion_angle_3d.png`: 3D view -- one example session's real probe tracks in "
        "CCF space, and every probe's insertion direction plotted as a vector from a common origin.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    columns = [
        "ecephys_unit_id", "ecephys_probe_id", "ecephys_session_id", "session_type", "name",
        "probe_vertical_position", "ecephys_structure_acronym", *CCF_COLUMNS,
    ]
    units = pd.read_csv(UNITS, usecols=columns, low_memory=False)
    units = units.dropna(subset=list(CCF_COLUMNS))

    table, fits_by_probe = build_table(units)
    table = table.sort_values(["probe_letter", "ecephys_session_id"]).reset_index(drop=True)
    table.to_csv(OUTPUT / "allen_probe_insertion_angle_from_ccf.csv", index=False, float_format="%.4f")

    make_figure(table, OUTPUT / "Figure_allen_probe_insertion_angle.png")
    make_3d_figure(table, fits_by_probe, OUTPUT / "Figure_allen_probe_insertion_angle_3d.png")
    write_report(table, OUTPUT / "ALLEN_PROBE_INSERTION_ANGLE.md")

    print(f"fitted {len(table)} probes")
    print(table.groupby("probe_letter")["angle_from_vertical_deg"].median().reindex(PROBE_LETTER_ORDER))
    print(f"wrote outputs to {OUTPUT}")


if __name__ == "__main__":
    main()
