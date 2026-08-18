#!/usr/bin/env python3
"""Estimate visual cortex thickness along the DV axis from per-unit Allen CCF coordinates.

Key geometric point: unlike along-probe electrode distance (which requires an insertion-angle
correction to become a true anatomical thickness, see `compute_allen_probe_insertion_angle_from_ccf.py`
and the MouseV2 CSD/RF-span proxies), the dorsal_ventral_ccf_coordinate itself is already an
absolute anatomical position. If the local pia/white-matter boundary is roughly a horizontal
(constant-DV) sheet, the DV coordinate SPAN of units labeled as visual cortex (VIS*) already IS
the perpendicular cortical thickness, regardless of the recording probe's own insertion angle --
no angle correction is needed. This approximation degrades for areas where the cortical surface
tilts away from horizontal (more lateral/anterior visual areas, away from the dorsal vertex), so
this is reported as a DV-axis estimate, not a claim of true pia-normal thickness everywhere.

Two complementary estimates are reported:
  - per_probe: robust (5th-95th percentile) DV span of VIS*-labeled units on each individual
    probe, then averaged across probes. This is the more physically direct estimate -- each probe
    crosses one real local slab of cortex in one animal.
  - pooled_by_area: robust DV span of ALL VIS*-labeled units sharing a structure acronym, pooled
    across every probe/session. This mixes cross-animal/cross-session CCF registration variability
    into the span, so it is expected to run wider than the per-probe estimate; the gap between the
    two is itself informative about registration variability.
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
OUTPUT = ROOT / "artifacts" / "figure3" / "06l_allen_cortical_dv_thickness_from_ccf"

MIN_UNITS_PER_PROBE = 10
DV_PERCENTILES = (5, 95)
AREA_ORDER = ("VISp", "VISl", "VISal", "VISrl", "VISam", "VISpm", "VISli", "VISmma", "VISmmp")


def robust_span(values: np.ndarray) -> float:
    lo, hi = np.percentile(values, DV_PERCENTILES)
    return float(hi - lo)


def per_probe_table(vis: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for probe_id, group in vis.groupby("ecephys_probe_id", observed=True):
        if len(group) < MIN_UNITS_PER_PROBE:
            continue
        dv = group["dorsal_ventral_ccf_coordinate"].to_numpy(dtype=float)
        rows.append(
            {
                "ecephys_probe_id": int(probe_id),
                "ecephys_session_id": int(group["ecephys_session_id"].iat[0]),
                "probe_letter": group["name"].iat[0],
                "primary_structure": group["ecephys_structure_acronym"].mode().iat[0],
                "n_vis_units": len(group),
                "dv_thickness_um": robust_span(dv),
                "dv_range_minmax_um": float(dv.max() - dv.min()),
            }
        )
    return pd.DataFrame(rows)


def pooled_by_area_table(vis: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for area, group in vis.groupby("ecephys_structure_acronym", observed=True):
        if len(group) < MIN_UNITS_PER_PROBE:
            continue
        dv = group["dorsal_ventral_ccf_coordinate"].to_numpy(dtype=float)
        rows.append(
            {
                "area": area,
                "n_units": len(group),
                "n_probes": group["ecephys_probe_id"].nunique(),
                "dv_thickness_um": robust_span(dv),
            }
        )
    return pd.DataFrame(rows).sort_values("dv_thickness_um")


def make_figure(per_probe: pd.DataFrame, pooled: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    ax = axes[0]
    values = per_probe["dv_thickness_um"] / 1000.0
    ax.hist(values, bins=24, color="#4575b4", alpha=0.8, edgecolor="white")
    ax.axvline(values.mean(), color="black", linewidth=2,
               label=f"mean = {values.mean():.3f} mm")
    ax.axvline(values.median(), color="black", linewidth=1.5, linestyle="--",
               label=f"median = {values.median():.3f} mm")
    ax.set(xlabel="cortical DV thickness per probe (mm)", ylabel="probes",
           title=f"Per-probe robust (P5-P95) DV span\n(n={len(per_probe)} probes, "
                 f">={MIN_UNITS_PER_PROBE} VIS units each)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1]
    rng = np.random.default_rng(0)
    areas = [area for area in AREA_ORDER if area in per_probe["primary_structure"].values]
    for index, area in enumerate(areas):
        vals = per_probe.loc[per_probe.primary_structure.eq(area), "dv_thickness_um"].to_numpy() / 1000.0
        jitter = rng.uniform(-0.16, 0.16, len(vals))
        ax.scatter(index + jitter, vals, s=20, alpha=0.6, color="#4575b4", edgecolor="none")
        if len(vals):
            ax.hlines(np.median(vals), index - 0.26, index + 0.26, color="black", lw=2, zorder=3)
    ax.set_xticks(range(len(areas)))
    ax.set_xticklabels(areas, rotation=30, ha="right")
    ax.set(ylabel="cortical DV thickness per probe (mm)",
           title="Per-probe DV thickness by primary visual area\n(probe's modal structure label)")
    ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Allen visual cortex thickness along the DV axis, from per-unit CCF coordinates", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(per_probe: pd.DataFrame, pooled: pd.DataFrame, output_path: Path) -> None:
    values = per_probe["dv_thickness_um"]
    by_area = (
        per_probe.groupby("primary_structure")["dv_thickness_um"]
        .agg(["median", "mean", "std", "count"])
        .reindex([a for a in AREA_ORDER if a in per_probe.primary_structure.values])
    )
    lines = [
        "# Allen visual cortex DV thickness, from per-unit CCF coordinates",
        "",
        f"Per-probe robust (P5-P95) DV-coordinate span of VIS*-labeled units, "
        f"{len(per_probe)} probes with >= {MIN_UNITS_PER_PROBE} such units each.",
        "",
        f"- Mean per-probe DV thickness: {values.mean():.1f} um ({values.mean()/1000:.3f} mm).",
        f"- Median per-probe DV thickness: {values.median():.1f} um "
        f"(IQR {values.quantile(0.25):.1f}-{values.quantile(0.75):.1f} um).",
        f"- SD across probes: {values.std():.1f} um.",
        "",
        "## By primary visual area (probe's modal structure label)",
        "",
        by_area.to_string(),
        "",
        "## Pooled-by-area cross-check (mixes cross-session registration variability -- expect wider)",
        "",
        pooled.to_string(index=False),
        "",
        "## Caveat",
        "",
        "This treats the local pia/white-matter boundary as a horizontal (constant-DV) sheet, so the "
        "raw DV-coordinate span of cortically-labeled units equals perpendicular thickness without "
        "needing the probe's own insertion angle. That approximation is best near the dorsal vertex and "
        "degrades for areas where the true cortical surface normal tilts away from the DV axis.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    columns = [
        "ecephys_unit_id", "ecephys_probe_id", "ecephys_session_id", "name",
        "ecephys_structure_acronym", "dorsal_ventral_ccf_coordinate",
    ]
    units = pd.read_csv(UNITS, usecols=columns, low_memory=False)
    vis = units.loc[
        units.ecephys_structure_acronym.fillna("").str.startswith("VIS")
        & units.dorsal_ventral_ccf_coordinate.notna()
    ]

    per_probe = per_probe_table(vis)
    pooled = pooled_by_area_table(vis)
    per_probe.to_csv(OUTPUT / "per_probe_dv_thickness.csv", index=False, float_format="%.4f")
    pooled.to_csv(OUTPUT / "pooled_by_area_dv_thickness.csv", index=False, float_format="%.4f")

    make_figure(per_probe, pooled, OUTPUT / "Figure_allen_cortical_dv_thickness.png")
    write_report(per_probe, pooled, OUTPUT / "ALLEN_CORTICAL_DV_THICKNESS.md")

    values = per_probe["dv_thickness_um"]
    print(f"n probes: {len(per_probe)}")
    print(f"mean DV thickness: {values.mean():.1f} um ({values.mean()/1000:.3f} mm)")
    print(f"median DV thickness: {values.median():.1f} um")
    print(f"wrote outputs to {OUTPUT}")


if __name__ == "__main__":
    main()
