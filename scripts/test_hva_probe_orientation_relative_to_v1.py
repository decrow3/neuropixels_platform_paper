#!/usr/bin/env python3
"""Do probes near V1 point INTO V1, or fan OUTWARD toward the higher visual areas (HVAs)?

Reuses the per-probe 3D CCF line fit from `compute_allen_probe_insertion_angle_from_ccf.py`.
For each probe, the fitted line's two extremes give a shallow (surface-entry) point and a deep
(tip) point. Separately, for that same session, take the horizontal (AP-LR) centroid of all
VISp-labeled units recorded on the OTHER probes in that session (leave-one-probe-out, so a
probe is never tested against its own units -- this matters most for VISp-targeting probes,
which would trivially look "centered on V1" if judged against their own points).

delta_um = horizontal_distance(tip, V1_centroid) - horizontal_distance(entry, V1_centroid)

  delta_um > 0: the deep end of the probe is FARTHER from V1 than the shallow end -- the probe
                leans OUTWARD, away from V1, as it descends (consistent with a HVA-targeting
                probe diverging from V1 toward its own more peripheral target).
  delta_um < 0: the deep end is CLOSER to V1 than the shallow end -- the probe converges TOWARD
                V1 as it descends.

Restricted to sessions with >= MIN_VISP_UNITS_EXCL_SELF VISp units on other probes, so the V1
reference centroid itself is reasonably stable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_allen_probe_insertion_angle_from_ccf import (  # noqa: E402
    CCF_COLUMNS,
    MIN_UNITS_PER_PROBE,
    PROBE_LETTER_COLORS,
    fit_probe_line,
)

ROOT = Path(__file__).resolve().parents[1]
UNITS = ROOT / "data" / "unit_table.csv"
OUTPUT = ROOT / "artifacts" / "figure3" / "06m_hva_probe_orientation_relative_to_v1"

HVA5 = ("VISal", "VISrl", "VISam", "VISl", "VISpm")
AREA_ORDER = ("VISp", *HVA5)
MIN_VISP_UNITS_EXCL_SELF = 20


def build_table(units: pd.DataFrame) -> pd.DataFrame:
    visp = units.loc[units.ecephys_structure_acronym.eq("VISp")]

    rows = []
    for session_id, session_units in units.groupby("ecephys_session_id", observed=True):
        session_visp = visp.loc[visp.ecephys_session_id.eq(session_id)]
        for probe_id, group in session_units.groupby("ecephys_probe_id", observed=True):
            if len(group) < MIN_UNITS_PER_PROBE:
                continue
            v1_other = session_visp.loc[~session_visp.ecephys_probe_id.eq(probe_id)]
            if len(v1_other) < MIN_VISP_UNITS_EXCL_SELF:
                continue
            v1_centroid_horizontal = v1_other[["anterior_posterior_ccf_coordinate",
                                                "left_right_ccf_coordinate"]].to_numpy(float).mean(axis=0)

            points_um = group[list(CCF_COLUMNS)].to_numpy(dtype=float)
            depth_um = group["probe_vertical_position"].to_numpy(dtype=float)
            fit = fit_probe_line(points_um, depth_um)
            direction, centroid = fit["direction"], fit["centroid"]
            t = (points_um - centroid) @ direction
            entry_point = centroid + t.max() * direction  # shallow / surface
            tip_point = centroid + t.min() * direction  # deep / tip

            def horizontal_dist(point):
                return float(np.linalg.norm(point[[0, 2]] - v1_centroid_horizontal))

            dist_entry = horizontal_dist(entry_point)
            dist_tip = horizontal_dist(tip_point)

            rows.append(
                {
                    "ecephys_probe_id": int(probe_id),
                    "ecephys_session_id": int(session_id),
                    "probe_letter": group["name"].iat[0],
                    "primary_structure": group["ecephys_structure_acronym"].mode().iat[0],
                    "n_units": len(group),
                    "n_visp_units_other_probes": len(v1_other),
                    "dist_entry_to_v1_um": dist_entry,
                    "dist_tip_to_v1_um": dist_tip,
                    "delta_um": dist_tip - dist_entry,
                }
            )
    return pd.DataFrame(rows)


def make_figure(table: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    rng = np.random.default_rng(0)
    areas = [area for area in AREA_ORDER if area in table.primary_structure.values]
    for index, area in enumerate(areas):
        values = table.loc[table.primary_structure.eq(area), "delta_um"].to_numpy() / 1000.0
        jitter = rng.uniform(-0.18, 0.18, len(values))
        colors = np.where(values > 0, "#d73027", "#4575b4")
        ax.scatter(index + jitter, values, s=24, alpha=0.65, c=colors, edgecolor="none")
        ax.hlines(np.median(values), index - 0.28, index + 0.28, color="black", lw=2, zorder=3)
    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax.set_xticks(range(len(areas)))
    ax.set_xticklabels(areas)
    ax.set_ylabel("Δ horizontal distance to V1 centroid, tip − entry (mm)")
    ax.set_title(
        "Do probes point toward V1 or away, toward the HVAs?\n"
        "red/above 0 = leans away from V1 (outward) with depth; blue/below 0 = converges on V1",
        fontsize=11,
    )
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(table: pd.DataFrame, output_path: Path) -> None:
    hva = table.loc[table.primary_structure.isin(HVA5)]
    by_area = (
        table.groupby("primary_structure")["delta_um"]
        .agg(median="median", mean="mean", n="count",
             frac_away_from_v1=lambda s: float((s > 0).mean()))
        .reindex([a for a in AREA_ORDER if a in table.primary_structure.values])
    )
    lines = [
        "# Do HVA-targeting probes point toward V1 or away from it?",
        "",
        f"{len(table)} probes fit (>= {MIN_UNITS_PER_PROBE} units, session V1 reference from "
        f">= {MIN_VISP_UNITS_EXCL_SELF} VISp units on OTHER probes in the same session).",
        "",
        f"- Across all {len(hva)} HVA-primary probes, median delta = {hva.delta_um.median():.1f} um; "
        f"{(hva.delta_um > 0).mean():.0%} lean away from V1 with depth (positive delta).",
        "",
        "## By primary structure (positive = tip farther from V1 than entry = points away from V1)",
        "",
        by_area.to_string(),
        "",
        "## Interpretation",
        "",
        "A consistently positive delta across HVA-primary probes means these probes fan OUTWARD from "
        "V1 as they descend -- their surface entry sits relatively closer to V1's own footprint, and "
        "the electrode tip lands farther out in HVA/associated territory. A negative delta would mean "
        "the opposite: probes converge toward V1 at depth despite entering farther away at the surface.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    columns = [
        "ecephys_unit_id", "ecephys_probe_id", "ecephys_session_id", "name",
        "probe_vertical_position", "ecephys_structure_acronym", *CCF_COLUMNS,
    ]
    units = pd.read_csv(UNITS, usecols=columns, low_memory=False)
    units = units.dropna(subset=list(CCF_COLUMNS))

    table = build_table(units)
    table = table.sort_values(["primary_structure", "ecephys_session_id"]).reset_index(drop=True)
    table.to_csv(OUTPUT / "hva_probe_orientation_relative_to_v1.csv", index=False, float_format="%.4f")

    make_figure(table, OUTPUT / "Figure_hva_probe_orientation_relative_to_v1.png")
    write_report(table, OUTPUT / "HVA_PROBE_ORIENTATION.md")

    print(table.groupby("primary_structure")["delta_um"].agg(["median", "mean", "count"]))
    print(f"wrote outputs to {OUTPUT}")


if __name__ == "__main__":
    main()
