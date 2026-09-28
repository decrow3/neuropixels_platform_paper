#!/usr/bin/env python3
"""Select and render concrete MouseV2 RF completion cases before support analysis."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.figure3_mousev2 import load_config
from common.parametric_models import (
    _numeric_design, aggregate_presentations, elliptical_gaussian_rate,
    presentation_counts,
)

FITS = ROOT / "data/imports/mousev2_parametric_rf_figure4_v1/rf_unit_fits.csv"
PEAKS = ROOT / "data/imports/pilot_rf_peaks_v1/rf_unit_peaks.csv"
INVENTORY = ROOT / "artifacts/figure4_rerun/v1/mousev2_rf_refit_inventory.csv"
OUT = ROOT / "artifacts/figure4_rerun/v1/rf_completion"


def select_cases(table: pd.DataFrame) -> pd.DataFrame:
    selected: list[dict[str, object]] = []
    used: set[str] = set()

    def choose(role: str, candidates: pd.DataFrame, criterion: str, mode: str, detail: str) -> None:
        candidates = candidates.loc[~candidates.unit_id.astype(str).isin(used)].copy()
        if candidates.empty:
            return
        index = candidates[criterion].idxmax() if mode == "max" else candidates[criterion].idxmin()
        row = candidates.loc[index]
        used.add(str(row.unit_id))
        selected.append({
            "unit_id": str(row.unit_id), "site": row.site, "probe": row.probe,
            "selection_role": role, "criterion": criterion,
            "criterion_value": float(row[criterion]), "selection_detail": detail,
            "provenance": "algorithmic selection before viewing raw RF maps",
        })

    new = ~table.legacy_fit_available
    for probe, label in [("E", "lateral"), ("B", "medial")]:
        accepted = table.loc[new & table.probe.eq(probe) & table.rf_model_supported].copy()
        choose(f"{label}_new_strong_accepted", accepted, "rf_pseudo_r2", "max", "largest pseudo-R2 among newly fitted supported cells")
        if len(accepted):
            accepted["typical_distance"] = (accepted.rf_pseudo_r2 - accepted.rf_pseudo_r2.median()).abs()
            choose(f"{label}_new_typical_accepted", accepted, "typical_distance", "min", "pseudo-R2 nearest the newly accepted probe median")

        core = (
            new & table.probe.eq(probe) & table.rf_fit_success & table.rf_center_on_screen
            & table.rf_lrt_q.le(.05) & table.rf_reliability_q.le(.05)
        )
        base = core & table.rf_sigma_major_deg.between(3.05, 79.5) & table.rf_sigma_minor_deg.between(3.05, 79.5)
        reliability = table.loc[base & table.rf_pseudo_r2.ge(.1) & table.rf_split_half_spearman_brown.lt(.3)].copy()
        choose(f"{label}_reliability_boundary_reject", reliability, "rf_split_half_spearman_brown", "max", "closest failure below the fixed reliability threshold")
        model = table.loc[base & table.rf_split_half_spearman_brown.ge(.3) & table.rf_pseudo_r2.lt(.1)].copy()
        choose(f"{label}_pseudo_r2_boundary_reject", model, "rf_pseudo_r2", "max", "closest failure below the fixed pseudo-R2 threshold")

        offscreen = table.loc[new & table.probe.eq(probe) & table.rf_fit_success & ~table.rf_center_on_screen].copy()
        offscreen["outside_screen_distance_deg"] = np.hypot(
            np.maximum(np.abs(offscreen.rf_center_x_deg) - 40, 0),
            np.maximum(np.abs(offscreen.rf_center_y_deg) - 40, 0),
        )
        choose(f"{label}_offscreen_boundary_reject", offscreen, "outside_screen_distance_deg", "min", "fitted center nearest the sampled screen boundary from outside")

        width = table.loc[
            core & table.rf_split_half_spearman_brown.ge(.3) & table.rf_pseudo_r2.ge(.1)
            & ~(table.rf_sigma_major_deg.between(3.05, 79.5) & table.rf_sigma_minor_deg.between(3.05, 79.5))
        ].copy()
        width["width_boundary_distance_deg"] = np.minimum.reduce([
            (width.rf_sigma_major_deg - 3.05).abs(), (width.rf_sigma_major_deg - 79.5).abs(),
            (width.rf_sigma_minor_deg - 3.05).abs(), (width.rf_sigma_minor_deg - 79.5).abs(),
        ])
        choose(f"{label}_width_boundary_reject", width, "width_boundary_distance_deg", "min", "width nearest an excluded fit bound")

        disagreement = table.loc[new & table.probe.eq(probe) & table.rf_model_supported].copy()
        disagreement["peak_fit_distance_deg"] = np.hypot(
            disagreement.rf_center_x_deg - disagreement.rf_center_x_deg_peak,
            disagreement.rf_center_y_deg - disagreement.rf_center_y_deg_peak,
        )
        choose(f"{label}_largest_peak_fit_disagreement", disagreement, "peak_fit_distance_deg", "max", "largest raw-argmax versus supported fitted-center distance")
    return pd.DataFrame(selected)


def raw_maps(cases: pd.DataFrame, fits: pd.DataFrame) -> pd.DataFrame:
    from generate_retinotopic_csvs import read_nwb_tables

    config = load_config()
    by_site = {item["site"]: item for item in config["sessions"]}
    rows = []
    for site, local_cases in cases.groupby("site", sort=True):
        session = by_site[site]
        nwb = Path(config["nwb_input"]["default_root"]) / session["nwb_relative_path"]
        extracted = read_nwb_tables(str(nwb))
        presentations = extracted.intervals_tables["receptive_field_block_presentations"]
        design, codes, cells = _numeric_design(presentations, ("x_position", "y_position"))
        global_ids = local_cases.unit_id.astype(int).to_numpy()
        local_ids = global_ids - int(session["id_offset"])
        responses = presentation_counts(
            local_ids, extracted.spikes_by_unit,
            design.start_time.to_numpy(), design.stop_time.to_numpy(),
        )
        totals, trials = aggregate_presentations(responses, codes, len(cells))
        for row_index, global_id in enumerate(global_ids):
            fit = fits.loc[fits.unit_id.astype(int).eq(global_id)].iloc[0]
            parameters = np.array([
                fit.rf_baseline_spikes, fit.rf_amplitude_spikes,
                fit.rf_center_x_deg, fit.rf_center_y_deg,
                fit.rf_sigma_major_deg, fit.rf_sigma_minor_deg,
                np.deg2rad(fit.rf_theta_deg),
            ])
            predicted = elliptical_gaussian_rate(
                parameters, cells.x_position.to_numpy(), cells.y_position.to_numpy(),
            )
            for cell_index, cell in cells.iterrows():
                rows.append({
                    "unit_id": str(global_id), "x_position": cell.x_position,
                    "y_position": cell.y_position,
                    "observed_mean_spikes": totals[row_index, cell_index] / trials[cell_index],
                    "fitted_mean_spikes": predicted[cell_index],
                    "trials": trials[cell_index],
                })
    return pd.DataFrame(rows)


def render(cases: pd.DataFrame, maps: pd.DataFrame, fits: pd.DataFrame, peaks: pd.DataFrame) -> None:
    n = len(cases)
    fig, axes = plt.subplots(n, 2, figsize=(8.5, max(2.25 * n, 5)), squeeze=False)
    joined = fits.merge(peaks[["unit_id", "rf_center_x_deg_peak", "rf_center_y_deg_peak"]], on="unit_id", validate="one_to_one").set_index("unit_id")
    for row_index, case in enumerate(cases.itertuples(index=False)):
        part = maps.loc[maps.unit_id.eq(str(case.unit_id))]
        fit = joined.loc[str(case.unit_id)]
        xs, ys = sorted(part.x_position.unique()), sorted(part.y_position.unique())
        for column, field in enumerate(["observed_mean_spikes", "fitted_mean_spikes"]):
            ax = axes[row_index, column]
            grid = part.pivot(index="y_position", columns="x_position", values=field).reindex(index=ys, columns=xs)
            image = ax.imshow(grid.to_numpy(), origin="lower", aspect="auto", extent=[min(xs), max(xs), min(ys), max(ys)], cmap="viridis")
            ax.scatter(fit.rf_center_x_deg_peak, fit.rf_center_y_deg_peak, marker="x", color="white", s=28, linewidth=1.4, label="raw peak")
            ax.scatter(fit.rf_center_x_deg, fit.rf_center_y_deg, facecolors="none", edgecolors="red", s=40, linewidth=1.3, label="fit")
            fig.colorbar(image, ax=ax, fraction=.04, pad=.02)
            ax.set_xlabel("x position (deg)")
            ax.set_ylabel("y position (deg)")
            if column == 0:
                ax.set_title(f"{case.selection_role}\nunit {case.unit_id}, {case.site}/{case.probe}\nobserved mean spikes", fontsize=8.5)
            else:
                ax.set_title(f"fitted Gaussian\nR²={fit.rf_pseudo_r2:.3f}, reliability={fit.rf_split_half_spearman_brown:.3f}\nsupported={bool(fit.rf_model_supported)}", fontsize=8.5)
                ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "selected_rf_cases.png", dpi=180)
    fig.savefig(OUT / "selected_rf_cases.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fits = pd.read_csv(FITS, dtype={"unit_id": str})
    peaks = pd.read_csv(PEAKS, dtype={"unit_id": str}).rename(columns={
        "rf_center_x_deg": "rf_center_x_deg_peak", "rf_center_y_deg": "rf_center_y_deg_peak",
    })
    inventory = pd.read_csv(INVENTORY, dtype={"unit_id": str})[["unit_id", "legacy_fit_available"]]
    table = fits.merge(peaks[["unit_id", "rf_center_x_deg_peak", "rf_center_y_deg_peak"]], on="unit_id", validate="one_to_one")
    table = table.merge(inventory, on="unit_id", validate="one_to_one")
    if args.render_only:
        cases = pd.read_csv(OUT / "selected_cases.csv", dtype={"unit_id": str})
        maps = pd.read_csv(OUT / "selected_case_raw_maps.csv", dtype={"unit_id": str})
    else:
        cases = select_cases(table)
        if len(cases) < 12:
            raise ValueError(f"Only {len(cases)} prespecified case roles were available")
        maps = raw_maps(cases, fits)
        cases = cases.merge(table, on=["unit_id", "site", "probe"], validate="one_to_one")
        cases.to_csv(OUT / "selected_cases.csv", index=False)
        maps.to_csv(OUT / "selected_case_raw_maps.csv", index=False)
    render(cases, maps, fits, peaks)
    print(cases[["unit_id", "site", "probe", "selection_role", "criterion_value", "rf_model_supported"]].to_string(index=False))
    print(OUT / "selected_rf_cases.png")


if __name__ == "__main__":
    main()
