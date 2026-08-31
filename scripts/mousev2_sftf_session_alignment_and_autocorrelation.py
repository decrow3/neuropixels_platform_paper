#!/usr/bin/env python3
"""Test two specific hypotheses for why the MouseV2/Allen cortical SF/TF surfaces (06p) might not
show a shared map even if one exists: (1) no per-animal SF/TF offset correction is applied, unlike
retinotopy's per-session RF-value delta (`register_mousev2_rf_to_zhuang_v1.py`); (2) the 375um
smoothing bandwidth was chosen by analogy to the retinotopic-space bandwidth, not from the data's
own spatial autocorrelation structure.

Part 1 -- per-session offset. For each MouseV2 session, find that session's own eligible units'
nearest ALLEN cortical-surface grid point (06p's `allen_v1_cortical_surface_grid.csv`, within 1.5x
bandwidth) and fit a single Huber-location log2 offset between the unit's own preference and that
local Allen reference. This mirrors 06e's session_delta in spirit (a session-level nuisance
correction against an external reference) but corrects the RESPONSE variable, not position. MouseV2's
pooled surface is then refit on offset-corrected values and re-compared to Allen (06p's
`cortical_difference_grid`/`render_cortical_difference_figure`, reused unchanged).

CAVEATS, read before trusting the "improvement": (a) this is an IN-SAMPLE correction -- with only 8
MouseV2 sessions, no leave-one-session-out cross-validation is attempted, so any correlation gain is
a diagnostic, not a validated claim; (b) only MouseV2 is corrected -- Allen's own reference surface
has no analogous per-session correction anywhere in this project, so residual Allen miscalibration
would cap how much improvement is achievable even if the MouseV2 correction is exactly right;
(c) a single session-level SCALAR offset cannot itself manufacture spatial correlation with Allen's
pattern -- it only removes a session-wide additive nuisance, so an improvement in spatial
`surface_correlation` (not just in overall median offset) is the informative signal here.

Part 2 -- empirical variogram. Semivariance (0.5*mean((v_i-v_j)^2) in log2 units) vs. pairwise
cortical distance, computed separately for Allen V1 and session-offset-corrected MouseV2 V1, per
preference. The distance at which semivariance plateaus estimates the map's own spatial
autocorrelation range, independent of any assumption borrowed from the retinotopic-space analysis --
compared against the 39px (~375um) bandwidth actually used. CAVEAT: Allen's per-session V1 sampling
is a single Neuropixels track (near-degenerate in row/col, only depth varies), so Allen's x,y spatial
spread comes entirely from BETWEEN-session variation -- if Allen also carries uncorrected per-session
SF/TF offsets (plausible, not fixed here), that inflates its own semivariance at ALL distances,
including short ones, and would bias the estimated range upward. MouseV2's own spatial support (26
probes clustered within one V1) is similarly sparse. Treat the estimated ranges as order-of-magnitude
diagnostics, not calibrated numbers.
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
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fit_multistructure_fixed_effect_translation import huber_location  # noqa: E402
from scripts.mousev2_frequency_preference_cortical_surfaces import (  # noqa: E402
    BANDWIDTHS_PX,
    DEFAULT_GRATINGS,
    DEFAULT_RF,
    DEFAULT_TUNING,
    DEFAULT_UNIT_POSITIONS,
    POOLED_AREA,
    PRIMARY_BANDWIDTH_PX,
    ZHUANG_PX_PER_MM,
    ZHUANG_TEMPLATE,
    build_visp_grid,
    cortical_difference_grid,
    estimate_cortical_surfaces,
    render_cortical_difference_figure,
    summarize_surfaces,
)
from scripts.mousev2_frequency_preference_surfaces import PREFERENCES, load_mousev2_units  # noqa: E402
from scripts.register_allen_session_to_zhuang import build_template  # noqa: E402

DEFAULT_ALLEN_SURFACE = (
    ROOT / "artifacts" / "figure3" / "06p_mousev2_frequency_preference_cortical_surfaces"
    / "allen_v1_cortical_surface_grid.csv"
)
DEFAULT_MOUSEV2_SURFACE = (
    ROOT / "artifacts" / "figure3" / "06p_mousev2_frequency_preference_cortical_surfaces"
    / "mousev2_frequency_preference_cortical_surface_grid.csv"
)
DEFAULT_ALLEN_UNITS = (
    ROOT / "artifacts" / "figure3" / "06p_mousev2_frequency_preference_cortical_surfaces"
    / "allen_v1_units_ccf_zhuang_position.csv"
)
ALLEN_PREFERENCE_COLUMN = {"sf": "pref_sf_sg", "tf": "pref_tf_dg"}
DEFAULT_OUTPUT = ROOT / "artifacts" / "figure3" / "06q_mousev2_sftf_session_alignment_and_autocorrelation"

MIN_UNITS_FOR_OFFSET = 10
MAX_REFERENCE_DISTANCE_FACTOR = 1.5
VARIOGRAM_BIN_EDGES_PX = np.arange(0.0, 260.0, 20.0)
MIN_PAIRS_PER_BIN = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rf-table", type=Path, default=DEFAULT_RF)
    parser.add_argument("--grating-dir", type=Path, default=DEFAULT_GRATINGS)
    parser.add_argument("--tuning-support", type=Path, default=DEFAULT_TUNING)
    parser.add_argument("--unit-positions", type=Path, default=DEFAULT_UNIT_POSITIONS)
    parser.add_argument("--allen-surface", type=Path, default=DEFAULT_ALLEN_SURFACE)
    parser.add_argument("--mousev2-surface", type=Path, default=DEFAULT_MOUSEV2_SURFACE)
    parser.add_argument("--allen-units", type=Path, default=DEFAULT_ALLEN_UNITS)
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


def fit_session_offsets(
    matched_units: pd.DataFrame,
    allen_surfaces: pd.DataFrame,
    preference: str,
    column: str,
    *,
    bandwidth_px: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Per-MouseV2-session Huber-location log2 offset against the nearest supported Allen surface
    grid point, within MAX_REFERENCE_DISTANCE_FACTOR x bandwidth."""
    allen_selected = allen_surfaces.loc[
        allen_surfaces["preference"].eq(preference)
        & np.isclose(allen_surfaces["bandwidth_px"], bandwidth_px)
        & allen_surfaces["supported"]
    ]
    allen_points = allen_selected[["row", "col"]].to_numpy(float)
    allen_values = allen_selected["estimate_log2"].to_numpy(float)

    eligible = matched_units.loc[matched_units[f"tuning_eligible_{preference}"]].copy()
    positions = eligible[["inferred_row", "inferred_col"]].to_numpy(float)
    distance = np.sqrt(((positions[:, None, :] - allen_points[None, :, :]) ** 2).sum(axis=2))
    nearest_index = distance.argmin(axis=1)
    nearest_distance = distance[np.arange(len(eligible)), nearest_index]
    has_reference = nearest_distance <= MAX_REFERENCE_DISTANCE_FACTOR * bandwidth_px
    eligible = eligible.loc[has_reference].copy()
    eligible["allen_local_log2"] = allen_values[nearest_index[has_reference]]
    eligible["own_log2"] = np.log2(eligible[column].to_numpy(float))
    eligible["residual_log2"] = eligible["own_log2"] - eligible["allen_local_log2"]

    rows = []
    offsets: dict[str, float] = {}
    for site, group in eligible.groupby("site"):
        if len(group) < MIN_UNITS_FOR_OFFSET:
            rows.append({"site": site, "n_units_with_allen_reference": len(group), "offset_log2": np.nan, "offset_ratio": np.nan})
            continue
        offset = float(huber_location(group["residual_log2"].to_numpy(float)[:, None])[0])
        offsets[site] = offset
        rows.append({"site": site, "n_units_with_allen_reference": len(group), "offset_log2": offset, "offset_ratio": float(np.exp2(offset))})
    return pd.DataFrame(rows), offsets


def apply_offsets(units: pd.DataFrame, column: str, offsets: dict[str, float]) -> pd.DataFrame:
    corrected = units.copy()
    for site, offset in offsets.items():
        mask = corrected["site"].eq(site)
        corrected.loc[mask, column] = np.exp2(np.log2(corrected.loc[mask, column].to_numpy(float)) - offset)
    return corrected


def render_offset_figure(offset_tables: dict[str, pd.DataFrame], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    for ax, preference in zip(axes, PREFERENCES):
        table = offset_tables[preference].dropna(subset=["offset_log2"]).sort_values("offset_log2")
        colors = ["#d73027" if value > 0 else "#4575b4" for value in table["offset_log2"]]
        ax.barh(table["site"], table["offset_log2"], color=colors)
        ax.axvline(0.0, color="#333333", linewidth=0.8)
        median = float(table["offset_log2"].median()) if len(table) else float("nan")
        spread = float(table["offset_log2"].std()) if len(table) > 1 else float("nan")
        ax.set(
            title=f"Preferred {preference.upper()}: per-session offset vs. Allen reference\n"
                  f"median {median:+.3f} oct; std across sessions {spread:.3f} oct",
            xlabel="session log2 offset (own unit value - local Allen reference)",
        )
    fig.suptitle(
        "MouseV2 per-session SF/TF offset relative to the nearest Allen cortical-surface reference\n"
        "(Huber location of residuals; sessions below the min-unit-with-reference threshold omitted)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def empirical_variogram(points: np.ndarray, values: np.ndarray, *, bin_edges: np.ndarray) -> pd.DataFrame:
    distance = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))
    value_diff_sq = (values[:, None] - values[None, :]) ** 2
    upper = np.triu_indices(len(points), k=1)
    distance_flat = distance[upper]
    value_diff_sq_flat = value_diff_sq[upper]
    bin_index = np.digitize(distance_flat, bin_edges) - 1
    rows = []
    for b in range(len(bin_edges) - 1):
        mask = bin_index == b
        if mask.sum() < MIN_PAIRS_PER_BIN:
            continue
        rows.append(
            {
                "distance_lo_px": bin_edges[b],
                "distance_hi_px": bin_edges[b + 1],
                "distance_mid_px": 0.5 * (bin_edges[b] + bin_edges[b + 1]),
                "n_pairs": int(mask.sum()),
                "semivariance": float(0.5 * np.mean(value_diff_sq_flat[mask])),
            }
        )
    return pd.DataFrame(rows)


MIN_RANGE_TREND_CORRELATION = 0.3
MIN_RANGE_RELATIVE_RISE = 0.15


def estimate_range(variogram: pd.DataFrame) -> tuple[float, dict[str, float]]:
    """A 'range' (distance where semivariance plateaus) is only meaningful if semivariance actually
    RISES with distance. Per-unit SF/TF measurement noise can be large enough that semivariance is
    ~flat (noise-dominated at every distance, a pure 'nugget' with no detectable 'sill'); reporting a
    number from a flat curve would be a spurious precise-looking answer to a question the data can't
    answer. Requires both a real rising trend (Pearson r between distance and semivariance) and a
    relative rise of at least MIN_RANGE_RELATIVE_RISE before trusting the 90%-of-plateau distance."""
    if len(variogram) < 4:
        return float("nan"), {"trend_correlation": float("nan"), "relative_rise": float("nan")}
    near = float(variogram["semivariance"].head(2).mean())
    far = float(variogram["semivariance"].tail(3).mean())
    trend_correlation = float(np.corrcoef(variogram["distance_mid_px"], variogram["semivariance"])[0, 1])
    relative_rise = (far - near) / near if near > 0 else float("nan")
    diagnostics = {"trend_correlation": trend_correlation, "relative_rise": relative_rise, "near_semivariance": near, "far_semivariance": far}
    if not (trend_correlation >= MIN_RANGE_TREND_CORRELATION and relative_rise >= MIN_RANGE_RELATIVE_RISE):
        return float("nan"), diagnostics
    reached = variogram.loc[variogram["semivariance"] >= near + 0.9 * (far - near)]
    return (float(reached["distance_mid_px"].iloc[0]) if len(reached) else float("nan")), diagnostics


def render_variogram_figure(
    variograms: dict[tuple[str, str], pd.DataFrame], ranges: dict[tuple[str, str], float], output_path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    colors = {"MouseV2 (offset-corrected)": "#d73027", "Allen V1": "#4575b4"}
    for ax, preference in zip(axes, PREFERENCES):
        for population in ("MouseV2 (offset-corrected)", "Allen V1"):
            table = variograms[(preference, population)]
            if not len(table):
                continue
            ax.plot(table["distance_mid_px"], table["semivariance"], marker="o", ms=4, color=colors[population], label=population)
            range_px = ranges[(preference, population)]
            if np.isfinite(range_px):
                ax.axvline(range_px, color=colors[population], linestyle="--", linewidth=1.0, alpha=0.7)
        ax.axvline(PRIMARY_BANDWIDTH_PX, color="#333333", linestyle=":", linewidth=1.3, label=f"bandwidth used ({PRIMARY_BANDWIDTH_PX:.0f}px)")
        ax.set(
            title=f"Preferred {preference.upper()}: empirical semivariogram",
            xlabel="pairwise cortical distance (px)", ylabel="semivariance (log2 units²)",
        )
        ax.legend(fontsize=8)
    fig.suptitle(
        "Empirical spatial autocorrelation: dashed = estimated range (90% of plateau); "
        "dotted = bandwidth actually used in 06p",
        fontsize=12,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.86)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    units, _ = load_mousev2_units(
        args.rf_table.resolve(), args.grating_dir.resolve(),
        qc_profile=args.qc_profile, require_unique_preference=True,
        tuning_support_path=args.tuning_support.resolve(),
    )
    unit_positions = pd.read_csv(args.unit_positions.resolve())[["unit_id", "inferred_row", "inferred_col"]]
    matched_units = units.merge(unit_positions, on="unit_id", how="inner", validate="one_to_one")
    print(f"MouseV2 units with anatomy-anchored position: {len(matched_units):,}")

    allen_surfaces = pd.read_csv(args.allen_surface.resolve())
    allen_units = pd.read_csv(args.allen_units.resolve())
    original_mousev2_surfaces = pd.read_csv(args.mousev2_surface.resolve())

    template = build_template(ZHUANG_TEMPLATE)
    visp_mask = template["area_masks"]["VISp"]
    _, _, grid_points, in_visp = build_visp_grid(visp_mask, args.grid_size)

    # -- Part 1: per-session offset against the Allen reference surface --
    offset_tables = {}
    corrected_units = matched_units.copy()
    for preference, specification in PREFERENCES.items():
        offset_table, offsets = fit_session_offsets(
            matched_units, allen_surfaces, preference, specification["column"], bandwidth_px=PRIMARY_BANDWIDTH_PX,
        )
        offset_tables[preference] = offset_table
        offset_table.to_csv(output_dir / f"mousev2_session_offset_vs_allen_{preference}.csv", index=False, float_format="%.6g")
        n_fit = offset_table["offset_log2"].notna().sum()
        print(f"{preference.upper()} session offsets: fit for {n_fit}/{len(offset_table)} sessions "
              f"(median {offset_table['offset_log2'].median():+.3f} oct, "
              f"std across sessions {offset_table['offset_log2'].std():.3f} oct)")
        corrected_units = apply_offsets(corrected_units, specification["column"], offsets)
    render_offset_figure(offset_tables, output_dir / "Figure_mousev2_session_offset_vs_allen.png")

    corrected_surfaces = estimate_cortical_surfaces(
        corrected_units, grid_points, in_visp, BANDWIDTHS_PX,
        minimum_effective_sessions=args.minimum_effective_sessions, minimum_local_units=args.minimum_local_units,
    )
    corrected_summary = summarize_surfaces(corrected_surfaces)
    corrected_surfaces.to_csv(output_dir / "mousev2_session_corrected_cortical_surface_grid.csv", index=False, float_format="%.6g")
    corrected_summary.to_csv(output_dir / "mousev2_session_corrected_cortical_surface_summary.csv", index=False, float_format="%.6g")

    original_differences = cortical_difference_grid(original_mousev2_surfaces, allen_surfaces, bandwidth_px=PRIMARY_BANDWIDTH_PX)
    corrected_differences = cortical_difference_grid(corrected_surfaces, allen_surfaces, bandwidth_px=PRIMARY_BANDWIDTH_PX)
    corrected_differences.to_csv(output_dir / "mousev2_session_corrected_minus_allen_difference_grid.csv", index=False, float_format="%.6g")

    def difference_stats(differences: pd.DataFrame, preference: str) -> dict[str, float]:
        selected = differences.loc[differences["preference"].eq(preference)]
        shared = selected.loc[selected["shared_supported"]]
        if not len(shared):
            return {"shared_grid_fraction": 0.0, "median_difference_octaves": np.nan, "surface_correlation": np.nan}
        return {
            "shared_grid_fraction": float(len(shared) / len(selected)),
            "median_difference_octaves": float(shared["mousev2_minus_allen_log2"].median()),
            "surface_correlation": float(np.corrcoef(shared["mousev2_estimate_log2"], shared["allen_estimate_log2"])[0, 1]),
        }

    comparison_rows = []
    for preference in PREFERENCES:
        before = difference_stats(original_differences, preference)
        after = difference_stats(corrected_differences, preference)
        comparison_rows.append({"preference": preference, "stage": "before_correction", **before})
        comparison_rows.append({"preference": preference, "stage": "after_session_correction", **after})
        print(f"{preference.upper()} surface_correlation: {before['surface_correlation']:.2f} -> {after['surface_correlation']:.2f}")
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_dir / "before_after_session_correction_comparison.csv", index=False, float_format="%.6g")

    render_cortical_difference_figure(
        corrected_differences, template, PRIMARY_BANDWIDTH_PX,
        output_dir / "Figure_mousev2_minus_allen_cortical_difference_session_corrected.png",
    )

    # -- Part 2: empirical variogram (spatial autocorrelation length) --
    variograms: dict[tuple[str, str], pd.DataFrame] = {}
    ranges: dict[tuple[str, str], float] = {}
    range_diagnostics: dict[tuple[str, str], dict[str, float]] = {}
    variogram_rows = []
    for preference, specification in PREFERENCES.items():
        mousev2_eligible = corrected_units.loc[corrected_units[f"tuning_eligible_{preference}"]]
        mousev2_points = mousev2_eligible[["inferred_row", "inferred_col"]].to_numpy(float)
        mousev2_values = np.log2(mousev2_eligible[specification["column"]].to_numpy(float))
        table = empirical_variogram(mousev2_points, mousev2_values, bin_edges=VARIOGRAM_BIN_EDGES_PX)
        variograms[(preference, "MouseV2 (offset-corrected)")] = table
        ranges[(preference, "MouseV2 (offset-corrected)")], range_diagnostics[(preference, "MouseV2 (offset-corrected)")] = estimate_range(table)

        allen_eligible = allen_units.loc[allen_units[f"tuning_eligible_{preference}"]]
        allen_points = allen_eligible[["zhuang_row", "zhuang_col"]].to_numpy(float)
        allen_values = np.log2(allen_eligible[ALLEN_PREFERENCE_COLUMN[preference]].to_numpy(float))
        allen_table = empirical_variogram(allen_points, allen_values, bin_edges=VARIOGRAM_BIN_EDGES_PX)
        variograms[(preference, "Allen V1")] = allen_table
        ranges[(preference, "Allen V1")], range_diagnostics[(preference, "Allen V1")] = estimate_range(allen_table)

        for population, tbl in ((("MouseV2 (offset-corrected)"), table), ("Allen V1", allen_table)):
            tbl = tbl.copy()
            tbl["preference"] = preference
            tbl["population"] = population
            variogram_rows.append(tbl)

        for population in ("MouseV2 (offset-corrected)", "Allen V1"):
            range_px = ranges[(preference, population)]
            diag = range_diagnostics[(preference, population)]
            range_text = f"{range_px:.0f}px" if np.isfinite(range_px) else "NO DETECTABLE RANGE (flat/noise-dominated)"
            print(f"{preference.upper()} {population}: {range_text} "
                  f"(trend r={diag['trend_correlation']:+.2f}, near-to-far rise={diag['relative_rise']:+.1%})")

    pd.concat(variogram_rows, ignore_index=True).to_csv(output_dir / "empirical_variograms.csv", index=False, float_format="%.6g")
    range_summary = pd.DataFrame([
        {"preference": pref, "population": pop, "estimated_range_px": ranges[(pref, pop)], **range_diagnostics[(pref, pop)]}
        for pref, pop in ranges
    ])
    range_summary.to_csv(output_dir / "empirical_range_summary.csv", index=False, float_format="%.6g")
    render_variogram_figure(variograms, ranges, output_dir / "Figure_empirical_variogram.png")

    manifest = {
        "checkpoint": "06q_mousev2_sftf_session_alignment_and_autocorrelation",
        "status": "per-session SF/TF offset correction (vs. Allen reference) and empirical spatial "
                  "autocorrelation (variogram) diagnostics implemented",
        "inputs": {
            "rf_table": {"path": str(args.rf_table.resolve()), "sha256": sha256(args.rf_table.resolve())},
            "unit_positions": {"path": str(args.unit_positions.resolve()), "sha256": sha256(args.unit_positions.resolve())},
            "allen_surface": {"path": str(args.allen_surface.resolve()), "sha256": sha256(args.allen_surface.resolve())},
            "mousev2_surface": {"path": str(args.mousev2_surface.resolve()), "sha256": sha256(args.mousev2_surface.resolve())},
            "allen_units": {"path": str(args.allen_units.resolve()), "sha256": sha256(args.allen_units.resolve())},
        },
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "parameters": {
            "min_units_for_offset": MIN_UNITS_FOR_OFFSET,
            "max_reference_distance_factor": MAX_REFERENCE_DISTANCE_FACTOR,
            "variogram_bin_edges_px": list(VARIOGRAM_BIN_EDGES_PX),
            "min_pairs_per_bin": MIN_PAIRS_PER_BIN,
            "primary_bandwidth_px": PRIMARY_BANDWIDTH_PX,
            "caveat": "in-sample offset correction, no leave-one-session-out CV; Allen reference surface "
                      "not itself session-corrected; both populations' spatial sampling is sparse -- see "
                      "module docstring",
        },
        "before_after_comparison": comparison_rows,
        "estimated_ranges_px": {f"{k[0]}_{k[1]}": v for k, v in ranges.items()},
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Session alignment and autocorrelation diagnostics written to {output_dir}")


if __name__ == "__main__":
    main()
