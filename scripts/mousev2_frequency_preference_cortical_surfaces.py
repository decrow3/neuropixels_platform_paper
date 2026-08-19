#!/usr/bin/env python3
"""Map supported parametric MouseV2 SF/TF preferences over CORTICAL (Zhuang common-map) V1
position, as the cortical-space counterpart to the retinotopic-space maps in
`mousev2_frequency_preference_surfaces.py` (06d).

Two coordinate frames are shown together rather than merged into one:

- Primary continuous surface: per-unit ANATOMY-ANCHORED position from
  `direction_search_unit_positions.csv` (`render_mousev2_direction_search_depth_spread.py`,
  written into the 06j directory) -- entry point FIXED to the independent anatomy-registered
  position (06j, never consults RF value), shank length from an independently-derived depth-span
  estimate, and only ONE free parameter (shank angle theta) fit from each probe's own RF-vs-depth
  trend. Revised 2026-08-18: an earlier version of this script used
  `register_mousev2_units_along_probe_shank.py` (06g), a free 2-endpoint line fit from RF values
  alone with no anatomical anchor -- `render_allen_vs_mousev2_units_on_map_comparison.py` found
  06g's free-fit entry endpoint sits a median 95px from the true anatomical entry point (about
  half of V1's own diameter), so it was replaced here.
  PUTATIVE, NOT FULLY RESOLVED: the single angle parameter is weakly identified -- without an
  anatomical prior it pointed away from V1's center for 59% of probes, so the search is hard-
  restricted to a +/-90deg "toward V1 center" cone. As of the 2026-08-18 rerun, 26/26 probes fit
  within that cone but the median cosine to the inward direction is only +0.15 (near-orthogonal)
  and the median Huber fit loss is ~42deg. Read every plotted probe DIRECTION as a plausible,
  anatomically-constrained guess, not a resolved measurement; only the anchored entry point (06j)
  is trusted.
- Overlay: per-probe ANATOMY-based entry point from `register_mousev2_area_borders_to_zhuang.py`
  (06j) -- the same anchor the primary surface's positions are built from, so this overlay is no
  longer a fully independent cross-check; it mainly shows how far the putative fitted DIRECTION
  carries each probe's units from their own anchor, not two independent position estimates.

The two are shown as separate layers so a reader can see this relationship directly, rather than
have it collapsed into one number.

Allen V1 row (added 2026-08-19): a third row shows Allen's own Brain Observatory 1.1 V1 units at
their real CCF-registered position (`data/unit_table.csv`, projected into Zhuang pixel space via
`render_allen_vs_mousev2_units_on_map_comparison.py`'s `allen_ccf_to_zhuang_px`), colored by the
same released `pref_sf_sg`/`pref_tf_dg` preference used throughout this project's Allen-side
scripts (inclusive gate, no lifetime-sparseness/firing-rate/uniqueness threshold, matching
`allen_frequency_preference_surfaces.py`'s default checkpoint). Unlike either MouseV2 row, Allen's
per-unit position is genuinely independent of RF/SF/TF value -- true histology, not a fit. SF and
TF panels share one color scale between the MouseV2 surface and the Allen row (per user direction),
which is the more informative choice for comparing spatial PATTERN but means one population's range
can look more saturated: the released MouseV2-vs-Allen preference offset is descriptively large
(median 1.35x for SF, 1.07x for TF; see 06d), and Allen used a different, more restricted grating
stimulus set (fixed SF=0.04 cyc/deg for TF blocks, separate static-grating block for SF) -- so this
remains a descriptive spatial-pattern comparison, not a matched test, consistent with every other
Allen-vs-MouseV2 comparison in this project. 98.6% of the projected Allen V1 units (2,365/2,398)
land inside the VISp mask; the remainder (33 units, entirely from one session, 754829445) is a
single-session registration outlier, within the underlying Allen-map-to-Zhuang geometry's own
documented ~18deg median vector error (`translation_rotation_fit_manifest.json`), not excluded.

Checked directly (2026-08-19): on the shared-scale figure Allen's row looks nearly flat. Its own
within-surface range is real but smaller than MouseV2's (SF: 0.245 vs 0.312 octaves, 78%; TF: 0.148
vs 0.264 octaves, 56%), and its absolute values sit well below MouseV2's (SF 0.057-0.067 vs
0.072-0.089 cyc/deg -- consistent with the released ~1.3x offset), so on a color scale calibrated to
the POOLED range, Allen's true variation is compressed into a narrow strip. A second figure,
`Figure_mousev2_v1_frequency_preference_cortical_surfaces_own_scale.png` (`render_cortical_figure(...,
shared_color_scale=False)`), normalizes each row to its own 2-98% range so each population's internal
spatial pattern is visible on its own terms -- not a replacement for the shared-scale figure (which
is what makes MAGNITUDE comparable), a complement to it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.allen_frequency_preference_surfaces import (  # noqa: E402
    DEFAULT_AUDIT as ALLEN_AUDIT_DIR,
    PREFERENCES as ALLEN_PREFERENCES,
    load_preference_units,
    session_balanced_gaussian_surface,
)
from scripts.mousev2_frequency_preference_surfaces import PREFERENCES, load_mousev2_units  # noqa: E402
from scripts.register_allen_session_to_zhuang import build_template  # noqa: E402
from scripts.render_allen_vs_mousev2_units_on_map_comparison import allen_ccf_to_zhuang_px  # noqa: E402


DEFAULT_RF = ROOT / "data" / "imports" / "mousev2_parametric_rf_v1" / "rf_unit_fits.csv"
DEFAULT_GRATINGS = ROOT / "data" / "imports" / "mousev2_grating_metrics_v1"
DEFAULT_TUNING = (
    ROOT / "data" / "imports" / "mousev2_frequency_tuning_v1" / "frequency_tuning_support.csv"
)
DEFAULT_UNIT_POSITIONS = (
    ROOT / "artifacts" / "figure3" / "06j_mousev2_area_borders_registered_to_zhuang"
    / "direction_search_unit_positions.csv"
)
DEFAULT_ANATOMY_POSITIONS = (
    ROOT / "artifacts" / "figure3" / "06j_mousev2_area_borders_registered_to_zhuang"
    / "probe_anatomical_position.csv"
)
ZHUANG_TEMPLATE = ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/retinotopy_contour_grid.npz"
ALLEN_UNIT_TABLE = ROOT / "data" / "unit_table.csv"
ALLEN_GEOMETRY_MANIFEST = (
    ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases/translation_rotation_fit_manifest.json"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "figure3" / "06p_mousev2_frequency_preference_cortical_surfaces"

POOLED_AREA = "Multi-probe V1"
PROBE_ORDER = ("A", "B", "C", "E")
PROBE_COLORS = {"A": "#d73027", "B": "#4575b4", "C": "#1a9850", "E": "#8073ac"}

# Physically anchored bandwidths (px, at 104.6 px/mm -- see register_mousev2_area_borders_to_zhuang.py
# ZHUANG_PX_PER_MM): ~250/375/500 um, a comparable coverage fraction of V1's ~2 mm span to the
# 8/12/16 deg bandwidths used over the ~80 deg retinotopic range in 06d.
ZHUANG_PX_PER_MM = 104.6
BANDWIDTHS_PX = (0.25 * ZHUANG_PX_PER_MM, 0.375 * ZHUANG_PX_PER_MM, 0.5 * ZHUANG_PX_PER_MM)
PRIMARY_BANDWIDTH_PX = 0.375 * ZHUANG_PX_PER_MM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rf-table", type=Path, default=DEFAULT_RF)
    parser.add_argument("--grating-dir", type=Path, default=DEFAULT_GRATINGS)
    parser.add_argument("--tuning-support", type=Path, default=DEFAULT_TUNING)
    parser.add_argument("--unit-positions", type=Path, default=DEFAULT_UNIT_POSITIONS)
    parser.add_argument("--anatomy-positions", type=Path, default=DEFAULT_ANATOMY_POSITIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qc-profile", choices=("pilot_qc", "default_qc"), default="pilot_qc")
    parser.add_argument("--grid-size", type=int, default=60)
    parser.add_argument("--minimum-effective-sessions", type=float, default=3.0)
    parser.add_argument("--minimum-local-units", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_visp_grid(visp_mask: np.ndarray, grid_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regular row/col grid over VISp's own bounding box, masked to the VISp footprint."""
    rows, cols = np.nonzero(visp_mask)
    row_grid = np.linspace(rows.min(), rows.max(), grid_size)
    col_grid = np.linspace(cols.min(), cols.max(), grid_size)
    row_mesh, col_mesh = np.meshgrid(row_grid, col_grid, indexing="ij")
    row_idx = np.clip(np.round(row_mesh).astype(int), 0, visp_mask.shape[0] - 1)
    col_idx = np.clip(np.round(col_mesh).astype(int), 0, visp_mask.shape[1] - 1)
    in_visp = visp_mask[row_idx, col_idx]
    grid_points = np.column_stack([row_mesh.ravel(), col_mesh.ravel()])
    return row_grid, col_grid, grid_points, in_visp.ravel()


def estimate_cortical_surfaces(
    units: pd.DataFrame,
    grid_points: np.ndarray,
    in_visp: np.ndarray,
    bandwidths: tuple[float, ...],
    *,
    minimum_effective_sessions: float,
    minimum_local_units: int,
) -> pd.DataFrame:
    frames = []
    group_definitions = ((POOLED_AREA, units["probe"].isin(PROBE_ORDER)),) + tuple(
        (f"Probe {probe}", units["probe"].eq(probe)) for probe in PROBE_ORDER
    )
    for preference, specification in PREFERENCES.items():
        for group_label, group_mask in group_definitions:
            group = units.loc[group_mask & units[f"tuning_eligible_{preference}"]].copy()
            points = group[["inferred_row", "inferred_col"]].to_numpy(dtype=float)
            values = np.log2(group[specification["column"]].to_numpy(dtype=float))
            sessions = group["site"].to_numpy()
            for bandwidth in bandwidths:
                result = session_balanced_gaussian_surface(
                    points,
                    values,
                    sessions,
                    grid_points,
                    bandwidth_deg=bandwidth,
                    minimum_effective_sessions=minimum_effective_sessions,
                    minimum_local_units=minimum_local_units,
                )
                supported = result["supported"] & in_visp
                frames.append(
                    pd.DataFrame(
                        {
                            "preference": preference,
                            "area": group_label,
                            "bandwidth_px": bandwidth,
                            "row": grid_points[:, 0],
                            "col": grid_points[:, 1],
                            "in_visp": in_visp,
                            "estimate_log2": np.where(supported, result["estimate_log2"], np.nan),
                            "estimate_preference": np.exp2(np.where(supported, result["estimate_log2"], np.nan)),
                            "effective_sessions": result["effective_sessions"],
                            "local_units": result["local_units"],
                            "supported": supported,
                            "source_units": len(group),
                            "source_sessions": group["site"].nunique(),
                        }
                    )
                )
    return pd.concat(frames, ignore_index=True)


def estimate_allen_cortical_surface(
    allen_v1: pd.DataFrame,
    grid_points: np.ndarray,
    in_visp: np.ndarray,
    bandwidths: tuple[float, ...],
    *,
    minimum_effective_sessions: float,
    minimum_local_units: int,
) -> pd.DataFrame:
    """Same session-balanced Gaussian kernel surface as `estimate_cortical_surfaces`, fit to Allen's
    V1 units at their real CCF-registered Zhuang position, on the SAME grid -- so it is directly
    comparable to, not just visually adjacent to, the MouseV2 surface."""
    frames = []
    for preference, specification in PREFERENCES.items():
        allen_column = ALLEN_PREFERENCES[preference]["column"]
        group = allen_v1.loc[allen_v1[f"tuning_eligible_{preference}"] & allen_v1[allen_column].notna()]
        points = group[["zhuang_row", "zhuang_col"]].to_numpy(dtype=float)
        values = np.log2(group[allen_column].to_numpy(dtype=float))
        sessions = group["ecephys_session_id"].to_numpy()
        for bandwidth in bandwidths:
            result = session_balanced_gaussian_surface(
                points,
                values,
                sessions,
                grid_points,
                bandwidth_deg=bandwidth,
                minimum_effective_sessions=minimum_effective_sessions,
                minimum_local_units=minimum_local_units,
            )
            supported = result["supported"] & in_visp
            frames.append(
                pd.DataFrame(
                    {
                        "preference": preference,
                        "area": "Allen V1",
                        "bandwidth_px": bandwidth,
                        "row": grid_points[:, 0],
                        "col": grid_points[:, 1],
                        "in_visp": in_visp,
                        "estimate_log2": np.where(supported, result["estimate_log2"], np.nan),
                        "estimate_preference": np.exp2(np.where(supported, result["estimate_log2"], np.nan)),
                        "effective_sessions": result["effective_sessions"],
                        "local_units": result["local_units"],
                        "supported": supported,
                        "source_units": len(group),
                        "source_sessions": group["ecephys_session_id"].nunique(),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def cortical_difference_grid(
    mousev2_surfaces: pd.DataFrame, allen_surfaces: pd.DataFrame, *, bandwidth_px: float
) -> pd.DataFrame:
    """MouseV2-minus-Allen difference on the shared cortical grid, mirroring
    `render_mousev2_allen_bo11_polar_comparison.py`'s retinotopic-space difference_grid."""
    frames = []
    for preference in PREFERENCES:
        mousev2_surface = mousev2_surfaces.loc[
            mousev2_surfaces["preference"].eq(preference)
            & mousev2_surfaces["area"].eq(POOLED_AREA)
            & np.isclose(mousev2_surfaces["bandwidth_px"], bandwidth_px)
        ][["row", "col", "estimate_log2", "supported"]].rename(
            columns={"estimate_log2": "mousev2_estimate_log2", "supported": "mousev2_supported"}
        )
        allen_surface = allen_surfaces.loc[
            allen_surfaces["preference"].eq(preference) & np.isclose(allen_surfaces["bandwidth_px"], bandwidth_px)
        ][["row", "col", "estimate_log2", "supported"]].rename(
            columns={"estimate_log2": "allen_estimate_log2", "supported": "allen_supported"}
        )
        merged = mousev2_surface.merge(allen_surface, on=["row", "col"], validate="one_to_one")
        merged["preference"] = preference
        merged["shared_supported"] = (
            merged["mousev2_supported"] & merged["allen_supported"]
            & merged[["mousev2_estimate_log2", "allen_estimate_log2"]].notna().all(axis=1)
        )
        merged["mousev2_minus_allen_log2"] = (
            merged["mousev2_estimate_log2"] - merged["allen_estimate_log2"]
        ).where(merged["shared_supported"])
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def summarize_surfaces(surfaces: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in surfaces.groupby(["preference", "area", "bandwidth_px"], observed=True):
        supported = group.loc[group["supported"] & group["estimate_log2"].notna()]
        rows.append(
            {
                "preference": keys[0],
                "area": keys[1],
                "bandwidth_px": keys[2],
                "source_units": int(group["source_units"].iloc[0]),
                "source_sessions": int(group["source_sessions"].iloc[0]),
                "supported_visp_fraction": float(supported.shape[0] / max(1, int(group["in_visp"].sum()))),
                "surface_median_preference": float(np.exp2(supported["estimate_log2"].median())) if len(supported) else np.nan,
                "surface_p10_preference": float(np.exp2(supported["estimate_log2"].quantile(0.1))) if len(supported) else np.nan,
                "surface_p90_preference": float(np.exp2(supported["estimate_log2"].quantile(0.9))) if len(supported) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def anatomy_probe_medians(units: pd.DataFrame, anatomy: pd.DataFrame) -> pd.DataFrame:
    """Per-probe median SF/TF preference (log2) at the independent anatomy-registered position."""
    rows = []
    for preference, specification in PREFERENCES.items():
        eligible = units.loc[units[f"tuning_eligible_{preference}"]]
        for (site, probe), group in eligible.groupby(["site", "probe"]):
            match = anatomy.loc[(anatomy["site"] == site) & (anatomy["probe"] == probe)]
            if match.empty or len(group) < 3:
                continue
            rows.append(
                {
                    "preference": preference,
                    "site": site,
                    "probe": probe,
                    "n_units": len(group),
                    "median_log2_preference": float(np.log2(group[specification["column"]]).median()),
                    "zhuang_row": float(match["zhuang_row"].iloc[0]),
                    "zhuang_col": float(match["zhuang_col"].iloc[0]),
                }
            )
    return pd.DataFrame(rows)


def load_allen_v1_units() -> pd.DataFrame:
    """Allen Brain Observatory V1 units at their real CCF-registered Zhuang position -- genuinely
    independent of RF/SF/TF value (true histology, not a fit). Inclusive gate (no lifetime-
    sparseness/firing-rate/uniqueness threshold), matching allen_frequency_preference_surfaces.py's
    default checkpoint."""
    units, _, _ = load_preference_units(ALLEN_AUDIT_DIR.resolve(), None)
    v1 = units.loc[units["area"].eq("V1")].copy()
    ccf = pd.read_csv(
        ALLEN_UNIT_TABLE,
        usecols=["ecephys_unit_id", "left_right_ccf_coordinate", "anterior_posterior_ccf_coordinate"],
        low_memory=False,
    )
    v1 = v1.merge(ccf, on="ecephys_unit_id", how="left", validate="one_to_one")
    n_before_ccf = len(v1)
    v1 = v1.dropna(subset=["left_right_ccf_coordinate", "anterior_posterior_ccf_coordinate"]).copy()
    print(f"Allen V1 support units: {n_before_ccf:,}; with a valid CCF position: {len(v1):,} "
          f"({len(v1) / n_before_ccf:.1%})")
    geometry = json.loads(ALLEN_GEOMETRY_MANIFEST.read_text())
    v1["zhuang_col"], v1["zhuang_row"] = allen_ccf_to_zhuang_px(
        v1.left_right_ccf_coordinate.to_numpy(float) / 1000.0,
        v1.anterior_posterior_ccf_coordinate.to_numpy(float) / 1000.0,
        geometry,
    )
    return v1


def render_cortical_figure(
    surfaces: pd.DataFrame,
    anatomy_medians: pd.DataFrame,
    allen_surfaces: pd.DataFrame,
    template: dict,
    primary_bandwidth: float,
    output_path: Path,
    *,
    shared_color_scale: bool = True,
) -> None:
    """Both rows are now actual FITTED SURFACES (same session-balanced Gaussian kernel, same grid),
    not a surface next to raw scatter -- a like-for-like surface comparison, not just a shared map.

    shared_color_scale=True (default): one pooled 2-98% norm per preference across both rows, so
    absolute MAGNITUDE is comparable but a population with a genuinely narrower range (Allen's own
    within-surface range is real but smaller than MouseV2's -- see module docstring) looks flatter.
    shared_color_scale=False: each row normalized to its OWN 2-98% range, so each surface's internal
    spatial PATTERN (not magnitude) is visible on its own terms -- not comparable to the other row's
    colorbar, complementary to the shared-scale version, not a replacement for it."""
    boundary = template["boundary"].astype(float)
    height, width = template["domain"].shape
    color_maps = {"sf": "cividis", "tf": "magma"}
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 13.6))
    for col_idx, (preference, specification) in enumerate(PREFERENCES.items()):
        selected = surfaces.loc[
            surfaces["preference"].eq(preference)
            & surfaces["area"].eq(POOLED_AREA)
            & np.isclose(surfaces["bandwidth_px"], primary_bandwidth)
            & surfaces["supported"]
        ]
        allen_selected = allen_surfaces.loc[
            allen_surfaces["preference"].eq(preference)
            & np.isclose(allen_surfaces["bandwidth_px"], primary_bandwidth)
            & allen_surfaces["supported"]
        ]
        if shared_color_scale:
            pooled_values = np.concatenate(
                [selected["estimate_log2"].to_numpy(float), allen_selected["estimate_log2"].to_numpy(float)]
            ) if len(selected) or len(allen_selected) else np.array([0.0, 1.0])
            top_limits = bottom_limits = np.quantile(pooled_values, [0.02, 0.98])
        else:
            top_limits = np.quantile(selected["estimate_log2"], [0.02, 0.98]) if len(selected) else np.array([0.0, 1.0])
            bottom_limits = np.quantile(allen_selected["estimate_log2"], [0.02, 0.98]) if len(allen_selected) else np.array([0.0, 1.0])
        norm = Normalize(vmin=top_limits[0], vmax=top_limits[1])
        allen_norm = Normalize(vmin=bottom_limits[0], vmax=bottom_limits[1])
        ticks = np.linspace(top_limits[0], top_limits[1], 5)
        allen_ticks = np.linspace(bottom_limits[0], bottom_limits[1], 5)
        fmt = "{:.3f}" if preference == "sf" else "{:.2f}"

        ax = axes[0, col_idx]
        ax.contour(boundary, levels=[0.5], colors="#333333", linewidths=0.6)
        if len(selected):
            scatter = ax.scatter(
                selected["col"], selected["row"], c=selected["estimate_log2"], cmap=color_maps[preference],
                norm=norm, s=42, marker="s", alpha=0.85,
            )
            colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03, extend="both")
            colorbar.set_ticks(ticks)
            colorbar.set_ticklabels([fmt.format(np.exp2(value)) for value in ticks])
            colorbar.set_label(specification["label"] if "label" in specification else f"preferred {preference.upper()}")
        medians = anatomy_medians.loc[anatomy_medians["preference"].eq(preference)]
        if len(medians):
            ax.scatter(
                medians["zhuang_col"], medians["zhuang_row"], c=medians["median_log2_preference"],
                cmap=color_maps[preference], norm=norm,
                s=110, marker="o", edgecolors="white", linewidths=1.3, zorder=5,
            )
        ax.set(
            title=f"MouseV2: preferred {preference.upper()} over cortical V1 position\n"
                  f"(squares = anatomy-anchored surface, angle PUTATIVE; circles = 06j entry point)",
            xlabel="Zhuang common-map x (px)", ylabel="Zhuang common-map y (px; down+)", aspect="equal",
        )
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)

        ax = axes[1, col_idx]
        ax.contour(boundary, levels=[0.5], colors="#333333", linewidths=0.6)
        if len(allen_selected):
            source_units = int(allen_selected["source_units"].iloc[0])
            source_sessions = int(allen_selected["source_sessions"].iloc[0])
            scatter = ax.scatter(
                allen_selected["col"], allen_selected["row"], c=allen_selected["estimate_log2"], cmap=color_maps[preference],
                norm=allen_norm, s=42, marker="s", alpha=0.85,
            )
            colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03, extend="both")
            colorbar.set_ticks(allen_ticks)
            colorbar.set_ticklabels([fmt.format(np.exp2(value)) for value in allen_ticks])
            colorbar.set_label(specification["label"] if "label" in specification else f"preferred {preference.upper()}")
        else:
            source_units = source_sessions = 0
        ax.set(
            title=f"Allen V1: preferred {preference.upper()} over cortical position\n"
                  f"(same session-balanced surface fit; {source_units:,} units, {source_sessions} sessions, real CCF position)",
            xlabel="Zhuang common-map x (px)", ylabel="Zhuang common-map y (px; down+)", aspect="equal",
        )
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
    scale_note = (
        "SAME color scale per column, not a matched test"
        if shared_color_scale else
        "each row on its OWN color scale -- pattern only, magnitudes not comparable across rows"
    )
    fig.suptitle(
        f"SF/TF preference over cortical (Zhuang) V1 position: MouseV2 vs. Allen (bandwidth "
        f"{primary_bandwidth:.0f} px ≈ {primary_bandwidth / ZHUANG_PX_PER_MM * 1000:.0f} µm)\n"
        f"top row: MouseV2, anatomy-anchored (angle PUTATIVE) -- bottom row: Allen, independent CCF "
        f"position, SAME surface-fitting method -- {scale_note}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def render_cortical_difference_figure(
    differences: pd.DataFrame,
    template: dict,
    primary_bandwidth: float,
    output_path: Path,
) -> list[dict[str, object]]:
    """MouseV2 minus Allen, on the shared supported cortical grid -- cortical-space counterpart to
    `render_mousev2_allen_bo11_polar_comparison.py`'s retinotopic-space difference figure, including
    the same surface_correlation diagnostic (Pearson r between the two fitted surfaces, at shared
    supported grid points)."""
    boundary = template["boundary"].astype(float)
    height, width = template["domain"].shape
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 7.2))
    summaries = []
    for ax, preference in zip(axes, PREFERENCES):
        selected = differences.loc[differences["preference"].eq(preference)]
        shared = selected.loc[selected["shared_supported"]]
        ax.contour(boundary, levels=[0.5], colors="#333333", linewidths=0.6)
        if len(shared):
            limit = max(float(np.quantile(np.abs(shared["mousev2_minus_allen_log2"]), 0.98)), 0.05)
            scatter = ax.scatter(
                shared["col"], shared["row"], c=shared["mousev2_minus_allen_log2"], cmap="coolwarm",
                norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit), marker="s", s=42,
            )
            colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03, extend="both")
            colorbar.set_label("MouseV2 - Allen V1 preference (octaves)")
            median = float(shared["mousev2_minus_allen_log2"].median())
            p10, p90 = map(float, shared["mousev2_minus_allen_log2"].quantile([0.1, 0.9]))
            ratio = float(np.exp2(median))
            shared_fraction = float(len(shared) / max(1, len(selected)))
            correlation = float(np.corrcoef(shared["mousev2_estimate_log2"], shared["allen_estimate_log2"])[0, 1])
            ax.set_title(
                f"Preferred {preference.upper()}\n"
                f"median {median:+.3f} oct ({ratio:.2f}×); shared grid {shared_fraction:.1%}; "
                f"surface r={correlation:.2f}",
                fontsize=11,
            )
            summaries.append({
                "preference": preference, "shared_grid_fraction": shared_fraction,
                "median_difference_octaves": median, "p10_difference_octaves": p10, "p90_difference_octaves": p90,
                "median_mousev2_over_allen_ratio": ratio, "surface_correlation": correlation,
            })
        ax.set(
            xlabel="Zhuang common-map x (px)", ylabel="Zhuang common-map y (px; down+)", aspect="equal",
        )
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
    fig.suptitle(
        f"MouseV2 minus Allen V1 SF/TF preference, shared supported cortical grid "
        f"(bandwidth {primary_bandwidth:.0f} px ≈ {primary_bandwidth / ZHUANG_PX_PER_MM * 1000:.0f} µm)\n"
        f"Descriptive contrast, not a matched test -- MouseV2 position is anatomy-anchored but PUTATIVE in "
        f"direction; Allen position is independent CCF histology (see module docstring)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return summaries


def write_report(
    units: pd.DataFrame,
    matched_units: pd.DataFrame,
    summary: pd.DataFrame,
    anatomy_medians: pd.DataFrame,
    allen_v1: pd.DataFrame,
    allen_summary: pd.DataFrame,
    difference_summary: list[dict[str, object]],
    *,
    primary_bandwidth: float,
    output_path: Path,
) -> None:
    primary = summary.loc[
        np.isclose(summary["bandwidth_px"], primary_bandwidth) & summary["area"].eq(POOLED_AREA)
    ]
    lines = [
        "# MouseV2 multi-probe V1 SF/TF preference over cortical (Zhuang) position",
        "",
        "## Status: cortical-space counterpart to 06d, anatomy-anchored surface + independent entry-point overlay",
        "",
        f"RF/tuning-eligible units carried in from `load_mousev2_units` (06d gating): {units['analysis_eligible'].sum():,}.",
        f"Of those, units with an anatomy-anchored cortical position from "
        f"`direction_search_unit_positions.csv` (`render_mousev2_direction_search_depth_spread.py`): "
        f"{len(matched_units):,}. The gap is probes/sites that did not reach that fit's own "
        "minimum-units-per-probe threshold, so they have no position here even though they may be "
        "RF/tuning-eligible.",
        f"Per-probe anatomy-registered entry points (independent photo-anatomy position from 06j): "
        f"{anatomy_medians[['site', 'probe']].drop_duplicates().shape[0]:,} (site, probe) pairs across both preferences.",
        f"Allen Brain Observatory V1 units at real CCF-registered position (inclusive gate; this base "
        f"population is itself gated to have both SF and TF values, hence equal counts): "
        f"{allen_v1['tuning_eligible_sf'].sum():,} SF-eligible, {allen_v1['tuning_eligible_tf'].sum():,} TF-eligible, "
        f"across {allen_v1['ecephys_session_id'].nunique()} sessions. ~25% of the underlying V1 support "
        "population (3,186 units) lacks a valid CCF position in the released unit table and is excluded here.",
        "",
        "| Preference | Units (anatomy-anchored surface) | Sessions | Supported VISp-grid fraction | Surface median | Surface 10–90% |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in primary.sort_values("preference").iterrows():
        lines.append(
            f"| {row.preference.upper()} | {int(row.source_units):,} | {int(row.source_sessions)} | "
            f"{row.supported_visp_fraction:.1%} | {row.surface_median_preference:.3g} | "
            f"{row.surface_p10_preference:.3g}–{row.surface_p90_preference:.3g} |"
        )
    allen_primary = allen_summary.loc[np.isclose(allen_summary["bandwidth_px"], primary_bandwidth)]
    lines.extend(
        [
            "",
            "## Allen V1 surface (same fitting method, real CCF position)",
            "",
            "| Preference | Units | Sessions | Supported VISp-grid fraction | Surface median | Surface 10–90% |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in allen_primary.sort_values("preference").iterrows():
        lines.append(
            f"| {row.preference.upper()} | {int(row.source_units):,} | {int(row.source_sessions)} | "
            f"{row.supported_visp_fraction:.1%} | {row.surface_median_preference:.3g} | "
            f"{row.surface_p10_preference:.3g}–{row.surface_p90_preference:.3g} |"
        )
    lines.extend(
        [
            "",
            "## MouseV2 minus Allen, shared supported cortical grid",
            "",
            "| Preference | Shared grid | Median difference (oct) | Median ratio | Surface correlation (r) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in difference_summary:
        lines.append(
            f"| {row['preference'].upper()} | {row['shared_grid_fraction']:.1%} | "
            f"{row['median_difference_octaves']:+.3f} | {row['median_mousev2_over_allen_ratio']:.2f}× | "
            f"{row['surface_correlation']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This is not a single calibrated map. The continuous surface (squares) fixes each unit's ENTRY point",
            "to the independent anatomy-registered position (06j, never consults RF value), sets shank length from",
            "an independently-derived depth-span estimate, and fits only ONE free parameter -- shank angle theta --",
            "against that probe's own RF-vs-depth trend, hard-restricted to a +/-90deg toward-V1-center cone.",
            "PUTATIVE, NOT FULLY RESOLVED: as of the 2026-08-18 rerun, 26/26 probes fit within that cone, but the",
            "median cosine to the inward direction is only +0.15 (near-orthogonal) and the median Huber fit loss is",
            "~42deg -- treat every plotted probe DIRECTION as a plausible, anatomically-constrained guess, not a",
            "resolved measurement. Only the anchored entry point is independent of RF value; units within a probe",
            "no longer form an independently RF-matched scattered cloud (the artifact this checkpoint originally",
            "had, and that 06f also had until the same 2026-08-18 fix), but the line's own angle could still be",
            "wrong. The overlaid entry points (circles) come from `register_mousev2_area_borders_to_zhuang.py`",
            "(06j) -- the SAME anchor the surface above is built from, so this overlay is no longer a fully",
            "independent cross-check; it mainly shows how far the putative fitted DIRECTION carries each probe's",
            "median-depth position from its own anchor.",
            "Same SF/TF gating and stimulus/estimator caveats as 06d apply unchanged (joint Poisson",
            "log-Gaussian(SF) × log-Gaussian(TF) × von-Mises(orientation) fits, dataset-wide FDR + split-half",
            "reliability + pseudo-R² support contract, extrapolated peaks flagged upstream).",
            "",
            "The bottom row (Allen Brain Observatory 1.1 V1) is now an actual FITTED SURFACE -- the same",
            "session-balanced Gaussian kernel method, same grid, same support gates -- computed on Allen's",
            "single units at their real CCF-registered position, genuinely independent of RF/SF/TF value unlike",
            "either MouseV2 row above. SF and TF panels share ONE color scale between the two fitted surfaces",
            "(2-98% of the pooled range), chosen so spatial PATTERN is comparable across rows -- at the cost of",
            "one population's own range looking more saturated, since the released MouseV2-vs-Allen preference",
            "offset is descriptively large (median 1.35x for SF, 1.07x for TF; see 06d) and Allen used a",
            "different, more restricted grating stimulus set.",
            "",
            "`surface_correlation` (table above, and the difference figure) is the Pearson correlation between",
            "the two fitted log2 surfaces at grid points BOTH consider supported -- a genuine surface-vs-surface",
            "comparison, not just a shared color scale. Still descriptive, not a matched test: the two surfaces",
            "differ in stimulus set, estimator, and (for MouseV2) a putative-direction position axis, so a low",
            "correlation could reflect any of those, not necessarily the absence of a shared cortical gradient.",
            "",
            "## Outputs",
            "",
            "- `mousev2_frequency_preference_cortical_surface_grid.csv`: pooled and per-probe VISp-grid surface estimates.",
            "- `mousev2_frequency_preference_cortical_surface_summary.csv`: bandwidth and coverage summary.",
            "- `mousev2_probe_anatomy_preference_medians.csv`: per-probe median SF/TF preference at the independent anatomy-registered position.",
            "- `allen_v1_units_ccf_zhuang_position.csv`: Allen V1 units with released SF/TF preference and Zhuang-projected CCF position.",
            "- `allen_v1_cortical_surface_grid.csv` / `_summary.csv`: Allen's fitted surface, same method and grid as MouseV2's.",
            "- `mousev2_minus_allen_cortical_difference_grid.csv`: per-grid-point difference, shared-support flag, and both surfaces' values.",
            "- `Figure_mousev2_v1_frequency_preference_cortical_surfaces.png`: MouseV2 (top row) vs. Allen (bottom row) fitted cortical-space SF/TF surfaces, ONE pooled color scale per column (magnitude comparable, one row's true range can look compressed).",
            "- `Figure_mousev2_v1_frequency_preference_cortical_surfaces_own_scale.png`: same panels, each row on its OWN 2-98% color scale (spatial pattern visible per population, magnitudes NOT comparable across rows).",
            "- `Figure_mousev2_minus_allen_cortical_difference.png`: difference map + surface correlation on the shared supported grid.",
            "- `run_manifest.json`: input, code, parameters, and output checksums.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    units, metric_paths = load_mousev2_units(
        args.rf_table.resolve(),
        args.grating_dir.resolve(),
        qc_profile=args.qc_profile,
        require_unique_preference=True,
        tuning_support_path=args.tuning_support.resolve(),
    )

    unit_positions = pd.read_csv(args.unit_positions.resolve())[["unit_id", "inferred_row", "inferred_col"]]
    if unit_positions["unit_id"].duplicated().any():
        raise ValueError(f"{args.unit_positions} contains duplicate unit IDs")
    matched_units = units.merge(unit_positions, on="unit_id", how="inner", validate="one_to_one")

    anatomy = pd.read_csv(args.anatomy_positions.resolve())[["site", "probe", "zhuang_row", "zhuang_col"]]
    allen_v1 = load_allen_v1_units()
    print(f"Allen V1 units at real CCF position: {len(allen_v1):,} "
          f"({int(allen_v1.tuning_eligible_sf.sum()):,} SF-eligible, {int(allen_v1.tuning_eligible_tf.sum()):,} TF-eligible)")

    template = build_template(ZHUANG_TEMPLATE)
    visp_mask = template["area_masks"]["VISp"]
    row_grid, col_grid, grid_points, in_visp = build_visp_grid(visp_mask, args.grid_size)

    surfaces = estimate_cortical_surfaces(
        matched_units,
        grid_points,
        in_visp,
        BANDWIDTHS_PX,
        minimum_effective_sessions=args.minimum_effective_sessions,
        minimum_local_units=args.minimum_local_units,
    )
    summary = summarize_surfaces(surfaces)
    anatomy_medians = anatomy_probe_medians(matched_units, anatomy)

    allen_surfaces = estimate_allen_cortical_surface(
        allen_v1,
        grid_points,
        in_visp,
        BANDWIDTHS_PX,
        minimum_effective_sessions=args.minimum_effective_sessions,
        minimum_local_units=args.minimum_local_units,
    )
    allen_summary = summarize_surfaces(allen_surfaces)
    differences = cortical_difference_grid(surfaces, allen_surfaces, bandwidth_px=PRIMARY_BANDWIDTH_PX)

    surfaces.to_csv(output_dir / "mousev2_frequency_preference_cortical_surface_grid.csv", index=False, float_format="%.6g")
    summary.to_csv(output_dir / "mousev2_frequency_preference_cortical_surface_summary.csv", index=False, float_format="%.6g")
    anatomy_medians.to_csv(output_dir / "mousev2_probe_anatomy_preference_medians.csv", index=False, float_format="%.6g")
    allen_v1[[
        "ecephys_unit_id", "ecephys_session_id", "ecephys_probe_id", "area",
        ALLEN_PREFERENCES["sf"]["column"], ALLEN_PREFERENCES["tf"]["column"],
        "tuning_eligible_sf", "tuning_eligible_tf", "zhuang_row", "zhuang_col",
    ]].to_csv(output_dir / "allen_v1_units_ccf_zhuang_position.csv", index=False, float_format="%.6g")
    allen_surfaces.to_csv(output_dir / "allen_v1_cortical_surface_grid.csv", index=False, float_format="%.6g")
    allen_summary.to_csv(output_dir / "allen_v1_cortical_surface_summary.csv", index=False, float_format="%.6g")
    differences.to_csv(output_dir / "mousev2_minus_allen_cortical_difference_grid.csv", index=False, float_format="%.6g")

    render_cortical_figure(
        surfaces, anatomy_medians, allen_surfaces, template, PRIMARY_BANDWIDTH_PX,
        output_dir / "Figure_mousev2_v1_frequency_preference_cortical_surfaces.png",
    )
    render_cortical_figure(
        surfaces, anatomy_medians, allen_surfaces, template, PRIMARY_BANDWIDTH_PX,
        output_dir / "Figure_mousev2_v1_frequency_preference_cortical_surfaces_own_scale.png",
        shared_color_scale=False,
    )
    difference_summary = render_cortical_difference_figure(
        differences, template, PRIMARY_BANDWIDTH_PX,
        output_dir / "Figure_mousev2_minus_allen_cortical_difference.png",
    )
    write_report(
        units, matched_units, summary, anatomy_medians, allen_v1, allen_summary, difference_summary,
        primary_bandwidth=PRIMARY_BANDWIDTH_PX,
        output_path=output_dir / "MOUSEV2_FREQUENCY_PREFERENCE_CORTICAL_SURFACES.md",
    )

    outputs = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "run_manifest.json":
            outputs[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "checkpoint": "06p_mousev2_frequency_preference_cortical_surfaces",
        "status": "cortical-space SF/TF preference maps implemented, anatomy-anchored surface + independent entry-point overlay",
        "inputs": {
            "rf_table": {"path": str(args.rf_table.resolve()), "sha256": sha256(args.rf_table.resolve())},
            "grating_tables": [{"path": str(path), "sha256": sha256(path)} for path in metric_paths],
            "tuning_support": {"path": str(args.tuning_support.resolve()), "sha256": sha256(args.tuning_support.resolve())},
            "unit_positions": {"path": str(args.unit_positions.resolve()), "sha256": sha256(args.unit_positions.resolve())},
            "anatomy_positions": {"path": str(args.anatomy_positions.resolve()), "sha256": sha256(args.anatomy_positions.resolve())},
            "allen_unit_table": {"path": str(ALLEN_UNIT_TABLE.resolve()), "sha256": sha256(ALLEN_UNIT_TABLE.resolve())},
            "allen_audit_dir": str(ALLEN_AUDIT_DIR.resolve()),
            "allen_geometry_manifest": {
                "path": str(ALLEN_GEOMETRY_MANIFEST.resolve()), "sha256": sha256(ALLEN_GEOMETRY_MANIFEST.resolve()),
            },
        },
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "parameters": {
            "qc_profile": args.qc_profile,
            "bandwidths_px": list(BANDWIDTHS_PX),
            "primary_bandwidth_px": PRIMARY_BANDWIDTH_PX,
            "zhuang_px_per_mm": ZHUANG_PX_PER_MM,
            "grid_size": args.grid_size,
            "minimum_effective_sessions": args.minimum_effective_sessions,
            "minimum_local_units": args.minimum_local_units,
            "primary_position": "anatomy-anchored per-unit position (direction_search_unit_positions.csv; "
                                 "entry fixed to 06j, shank angle is the only free parameter and is PUTATIVE, "
                                 "not fully resolved -- see module docstring)",
            "overlay_position": "anatomy-registered per-probe entry point (06j, photo area-border matching, "
                                 "RF-independent, and the SAME anchor the primary position is built from)",
            "allen_row": "Allen Brain Observatory 1.1 V1 units at real CCF-registered position (independent "
                         "histology), inclusive gate, fit with the SAME session-balanced Gaussian kernel "
                         "surface as MouseV2 on the SAME grid, SAME 2-98% pooled color scale per preference "
                         "-- descriptive spatial-pattern comparison, not a matched test",
        },
        "difference_summary": difference_summary,
        "outputs": outputs,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"cortical SF/TF surfaces: {len(matched_units):,} units with an anatomy-anchored position, "
        f"{anatomy_medians[['site', 'probe']].drop_duplicates().shape[0]} anatomy-overlay probes"
    )
    for row in difference_summary:
        print(f"  {row['preference'].upper()}: median diff {row['median_difference_octaves']:+.3f} oct "
              f"({row['median_mousev2_over_allen_ratio']:.2f}x), shared grid {row['shared_grid_fraction']:.1%}, "
              f"surface r={row['surface_correlation']:.2f}")
    print(f"MouseV2 cortical frequency-preference surfaces written to {output_dir}")


if __name__ == "__main__":
    main()
