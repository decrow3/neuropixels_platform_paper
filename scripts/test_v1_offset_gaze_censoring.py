#!/usr/bin/env python3
"""Concrete test of two candidate anchors for the V1/screen translation.

The CCF->RF geometry is learned leave-one-animal-out from improved V1 fits.
Its otherwise arbitrary session translation is compared with (1) the absolute
mean gaze reported by the Allen eye model and (2) held-out RF on-screen labels
from V1 units not used to estimate that translation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr

from scripts.check_v1_cross_animal_mean_map_support import (
    CCF_COLUMNS,
    RF_COLUMNS,
    fit_fixed_effect_geometry,
    load_population,
    make_block_table,
    predict,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts" / "v1_absolute_size_dispersion_translation_checkpoint"
DEFAULT_INPUT = CHECKPOINT / "uncensored_size_sensitivity" / "v1_unit_descriptors.csv.gz"
DEFAULT_UNITS = ROOT / "data" / "unit_table.csv"
DEFAULT_INVENTORY = Path(
    "/media/huklaban5/Data/MouseV2/allen_visual_coding_neuropixels_sessions/session_inventory.json"
)
DEFAULT_OUTPUT = CHECKPOINT / "gaze_censor_anchor_checkpoint"
DEFAULT_CASES = (760345702, 798911424, 781842082)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--unit-table", type=Path, default=DEFAULT_UNITS)
    p.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--sessions", nargs="+", type=int, default=DEFAULT_CASES)
    p.add_argument("--all-gaze", action="store_true")
    p.add_argument("--overwrite-gaze", action="store_true")
    return p.parse_args()


def robust_center(x):
    return np.nanmedian(np.asarray(x, float), axis=0)


def gaze_summary(nwb_path: Path, session_id: int) -> dict:
    # Read the two required DynamicTable/TimeSeries arrays directly. This is
    # equivalent to get_screen_gaze_data for these columns and avoids loading
    # the multi-gigabyte units/spikes tables merely to compute a block median.
    with h5py.File(nwb_path, "r") as nwb:
        presentations = nwb["intervals/gabors_presentations"]
        start = float(np.min(presentations["start_time"][()]))
        stop = float(np.max(presentations["stop_time"][()]))
        series = nwb["processing/filtered_gaze_mapping/screen_coordinates_spherical"]
        t = series["timestamps"][()]
        data = series["data"][()]
    selected = (t >= start) & (t <= stop)
    # NWB stores [vertical, horizontal]; AllenSDK exposes these as y and x.
    x = np.asarray(data[selected, 1], float)
    y = np.asarray(data[selected, 0], float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    return {
        "ecephys_session_id": session_id,
        "gaze_x_median_deg": float(np.median(x)),
        "gaze_y_median_deg": float(np.median(y)),
        "gaze_x_mean_deg": float(np.mean(x)),
        "gaze_y_mean_deg": float(np.mean(y)),
        "gaze_x_iqr_deg": float(np.quantile(x, .75) - np.quantile(x, .25)),
        "gaze_y_iqr_deg": float(np.quantile(y, .75) - np.quantile(y, .25)),
        "gaze_valid_samples": int(valid.sum()),
        "gaze_valid_fraction": float(valid.mean()),
    }


def geometry_offsets(population: pd.DataFrame, blocks: pd.DataFrame):
    rows, fits = [], {}
    for sid, local in population.groupby("ecephys_session_id", observed=True):
        sid = int(sid)
        specimen = int(local.specimen_id.iloc[0])
        training = blocks.loc[~blocks.specimen_id.eq(specimen)]
        fit = fit_fixed_effect_geometry(training, "quadratic", .05)
        pred = predict(local[list(CCF_COLUMNS)].to_numpy(float), fit, "quadratic")
        residual = local[list(RF_COLUMNS)].to_numpy(float) - pred
        center = robust_center(residual)
        rng = np.random.default_rng(20260816 + sid)
        order = rng.permutation(len(local))
        half_a, half_b = order[0::2], order[1::2]
        a, b = robust_center(residual[half_a]), robust_center(residual[half_b])
        rows.append({
            "ecephys_session_id": sid,
            "specimen_id": specimen,
            "session_type": str(local.session_type.iloc[0]),
            "improved_units": len(local),
            "offset_az_raw_deg": center[0],
            "offset_el_raw_deg": center[1],
            "half_a_az_raw_deg": a[0],
            "half_a_el_raw_deg": a[1],
            "half_b_az_raw_deg": b[0],
            "half_b_el_raw_deg": b[1],
            "split_distance_deg": float(np.linalg.norm(a - b)),
            "split_az_difference_deg": float(a[0] - b[0]),
            "split_el_difference_deg": float(a[1] - b[1]),
        })
        fits[sid] = fit
    result = pd.DataFrame(rows)
    for axis in ("az", "el"):
        raw = f"offset_{axis}_raw_deg"
        result[f"offset_{axis}_relative_deg"] = result[raw] - result[raw].median()
        for half in ("half_a", "half_b"):
            raw = f"{half}_{axis}_raw_deg"
            result[f"{half}_{axis}_relative_deg"] = result[raw] - result[raw].median()
    return result, fits


def heldout_units(unit_path: Path, improved_ids: set[int]) -> pd.DataFrame:
    use = [
        "ecephys_unit_id", "ecephys_session_id", "specimen_id",
        "ecephys_structure_acronym", "anterior_posterior_ccf_coordinate",
        "left_right_ccf_coordinate", "on_screen_rf", "p_value_rf", "snr",
        "firing_rate_dg", "area_rf",
    ]
    table = pd.read_csv(unit_path, usecols=use, low_memory=False)
    keep = (
        table.ecephys_structure_acronym.eq("VISp")
        & ~table.ecephys_unit_id.isin(improved_ids)
        & table[list(CCF_COLUMNS)].notna().all(axis=1)
        & table.on_screen_rf.notna()
    )
    return table.loc[keep].copy()


def detection_features(points: np.ndarray) -> np.ndarray:
    # Flexible but deliberately low-dimensional empirical screen-support field.
    x = (points[:, 0] - 50.0) / 40.0
    y = (points[:, 1] - 10.0) / 40.0
    return np.column_stack([x, y, x*x, y*y, x*y])


def fit_logistic(x: np.ndarray, y: np.ndarray, ridge: float = .5) -> np.ndarray:
    x = np.column_stack([np.ones(len(x)), x])
    y = np.asarray(y, float)
    # Balance outcomes so a session's on-screen fraction cannot dominate shape.
    counts = np.bincount(y.astype(int), minlength=2)
    weights = np.where(y > 0, .5 / max(counts[1], 1), .5 / max(counts[0], 1)) * len(y)
    def objective(beta):
        eta = x @ beta
        loss = np.sum(weights * (np.logaddexp(0, eta) - y * eta))
        return loss + .5 * ridge * np.sum(beta[1:]**2)
    return minimize(objective, np.zeros(x.shape[1]), method="BFGS").x


def logistic_probability(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return expit(np.column_stack([np.ones(len(x)), x]) @ beta)


def censor_losses(target_id, heldout, offsets, fits, shifts):
    training_parts = []
    for sid, local in heldout.groupby("ecephys_session_id", observed=True):
        sid = int(sid)
        if sid == target_id or sid not in fits:
            continue
        offset = offsets.loc[offsets.ecephys_session_id.eq(sid), [
            "offset_az_raw_deg", "offset_el_raw_deg"
        ]].to_numpy(float)[0]
        pred = predict(local[list(CCF_COLUMNS)].to_numpy(float), fits[sid], "quadratic") + offset
        part = pd.DataFrame(detection_features(pred))
        part["outcome"] = local.on_screen_rf.astype(int).to_numpy()
        part["specimen_id"] = local.specimen_id.to_numpy()
        training_parts.append(part)
    training = pd.concat(training_parts, ignore_index=True)
    model = fit_logistic(training.iloc[:, :5].to_numpy(float), training.outcome.to_numpy(int))
    target = heldout.loc[heldout.ecephys_session_id.eq(target_id)]
    base = predict(target[list(CCF_COLUMNS)].to_numpy(float), fits[target_id], "quadratic")
    outcome = target.on_screen_rf.astype(int).to_numpy()
    losses = []
    for shift in shifts:
        probability = logistic_probability(model, detection_features(base + shift))
        probability = np.clip(probability, 1e-5, 1 - 1e-5)
        loss = -np.mean(outcome*np.log(probability) + (1-outcome)*np.log(1-probability))
        losses.append(loss)
    return np.asarray(losses), len(target), model


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    population = load_population(args.input, args.unit_table)
    blocks = make_block_table(population, 6)
    offsets, fits = geometry_offsets(population, blocks)
    offsets.to_csv(args.output_dir / "all_session_anatomy_offsets.csv", index=False)

    inventory = {int(r["ecephys_session_id"]): r for r in json.loads(args.inventory.read_text())}
    sessions = sorted(offsets.ecephys_session_id.astype(int).unique()) if args.all_gaze else args.sessions
    gaze_path = args.output_dir / "gaze_summary.csv"
    existing = pd.read_csv(gaze_path) if gaze_path.exists() and not args.overwrite_gaze else pd.DataFrame()
    done = set(existing.ecephys_session_id.astype(int)) if len(existing) else set()
    gaze_rows = existing.to_dict("records")
    for sid in sessions:
        if sid in done:
            continue
        record = inventory.get(int(sid))
        if not record or not record.get("validation", {}).get("has_raw_gaze_mapping"):
            continue
        try:
            gaze_rows.append(gaze_summary(Path(record["nwb_path"]), int(sid)))
        except Exception as exc:
            gaze_rows.append({"ecephys_session_id": int(sid), "error": repr(exc)})
        pd.DataFrame(gaze_rows).to_csv(gaze_path, index=False)
    gaze = pd.DataFrame(gaze_rows)

    improved_ids = set(population.ecephys_unit_id.astype(int))
    heldout = heldout_units(args.unit_table, improved_ids)
    axis = np.arange(-30., 30.01, 2.)
    raw_median = offsets[["offset_az_raw_deg", "offset_el_raw_deg"]].median().to_numpy(float)
    relative_shifts = np.array([(x, y) for y in axis for x in axis])
    shifts = relative_shifts + raw_median
    case_rows = []
    losses_by_case = {}
    for sid in args.sessions:
        if sid not in fits:
            continue
        losses, n, _ = censor_losses(sid, heldout, offsets, fits, shifts)
        best_raw = shifts[int(np.argmin(losses))]
        best = best_raw - raw_median
        anatomy = offsets.loc[offsets.ecephys_session_id.eq(sid), [
            "offset_az_relative_deg", "offset_el_relative_deg"
        ]].to_numpy(float)[0]
        anatomy_raw = anatomy + raw_median
        nearest = int(np.argmin(np.sum((shifts - anatomy_raw)**2, axis=1)))
        zero_raw = int(np.argmin(np.sum((shifts - raw_median)**2, axis=1)))
        case_rows.append({
            "ecephys_session_id": sid,
            "heldout_censor_units": n,
            "anatomy_relative_az_deg": anatomy[0],
            "anatomy_relative_el_deg": anatomy[1],
            "censor_best_relative_az_deg": best[0],
            "censor_best_relative_el_deg": best[1],
            "censor_gain_at_anatomy_vs_cohort_center_nats": float(losses[zero_raw]-losses[nearest]),
            "censor_best_gain_vs_cohort_center_nats": float(losses[zero_raw]-losses.min()),
        })
        losses_by_case[sid] = losses.reshape(len(axis), len(axis))
    cases = pd.DataFrame(case_rows)
    cases.to_csv(args.output_dir / "concrete_case_censor_results.csv", index=False)

    merged = offsets.merge(gaze, on="ecephys_session_id", how="inner")
    merged.to_csv(args.output_dir / "geometry_gaze_comparison.csv", index=False)
    correlation_rows = []
    for group_name, local in [("all", merged), *merged.groupby("session_type", observed=True)]:
        for gaze_col, offset_col, relation in [
            ("gaze_x_median_deg", "offset_az_relative_deg", "x_to_az"),
            ("gaze_y_median_deg", "offset_el_relative_deg", "y_to_el"),
            ("gaze_x_median_deg", "offset_el_relative_deg", "x_to_el_cross"),
            ("gaze_y_median_deg", "offset_az_relative_deg", "y_to_az_cross"),
        ]:
            valid = local[[gaze_col, offset_col]].dropna()
            if len(valid) < 5:
                continue
            pr = pearsonr(valid[gaze_col], valid[offset_col])
            sr = spearmanr(valid[gaze_col], valid[offset_col])
            correlation_rows.append({
                "group": group_name, "relation": relation, "sessions": len(valid),
                "pearson_r": pr.statistic, "pearson_p": pr.pvalue,
                "spearman_rho": sr.statistic, "spearman_p": sr.pvalue,
            })
    correlations = pd.DataFrame(correlation_rows)
    correlations.to_csv(args.output_dir / "gaze_offset_correlations.csv", index=False)

    prediction_rows = []
    for group_name, local in [("all", merged), *merged.groupby("session_type", observed=True)]:
        for gaze_col, offset_col, relation in [
            ("gaze_x_median_deg", "offset_az_relative_deg", "x_to_az"),
            ("gaze_y_median_deg", "offset_el_relative_deg", "y_to_el"),
        ]:
            valid = local[[gaze_col, offset_col]].dropna().reset_index(drop=True)
            if len(valid) < 6:
                continue
            observed, predicted, baseline = [], [], []
            for i in range(len(valid)):
                train = valid.drop(index=i)
                design = np.column_stack([np.ones(len(train)), train[gaze_col]])
                beta = np.linalg.lstsq(design, train[offset_col], rcond=None)[0]
                observed.append(valid.loc[i, offset_col])
                predicted.append(beta[0] + beta[1] * valid.loc[i, gaze_col])
                baseline.append(train[offset_col].median())
            observed, predicted, baseline = map(np.asarray, (observed, predicted, baseline))
            prediction_rows.append({
                "group": group_name, "relation": relation, "sessions": len(valid),
                "loo_gaze_mae_deg": np.mean(np.abs(observed-predicted)),
                "loo_constant_mae_deg": np.mean(np.abs(observed-baseline)),
                "loo_mae_gain_deg": np.mean(np.abs(observed-baseline))-np.mean(np.abs(observed-predicted)),
                "loo_gaze_r2": 1-np.sum((observed-predicted)**2)/np.sum((observed-observed.mean())**2),
            })
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(args.output_dir / "gaze_offset_loo_prediction.csv", index=False)

    ncols = 1 + len(args.sessions)
    fig, axes = plt.subplots(2, ncols, figsize=(5*ncols, 8))
    for col, axis_name in enumerate(("az", "el")):
        ax = axes[col, 0]
        if len(merged):
            gx = "gaze_x_median_deg" if axis_name == "az" else "gaze_y_median_deg"
            ax.scatter(merged[gx], merged[f"offset_{axis_name}_relative_deg"], s=55)
            for _, r in merged.iterrows():
                ax.annotate(str(int(r.ecephys_session_id))[-3:], (r[gx], r[f"offset_{axis_name}_relative_deg"]), fontsize=8)
            ax.set_xlabel(gx.replace("_", " "))
            ax.set_ylabel(f"anatomy residual {axis_name} (deg)")
        ax.axhline(0, color=".7", lw=1)
    for col, sid in enumerate(args.sessions, start=1):
        if col >= axes.shape[1] or sid not in losses_by_case:
            continue
        ax = axes[0, col]
        im = ax.imshow(losses_by_case[sid], origin="lower", extent=[axis.min(),axis.max(),axis.min(),axis.max()], aspect="equal", cmap="viridis_r")
        ax.set_title(f"{sid} held-out censor loss")
        ax.set_xlabel("relative az offset (deg)"); ax.set_ylabel("relative el offset (deg)")
        fig.colorbar(im, ax=ax, shrink=.75)
        row = cases.loc[cases.ecephys_session_id.eq(sid)].iloc[0]
        axes[1, col].axis("off")
        axes[1, col].text(.02, .95, "\n".join([
            f"held-out cells: {int(row.heldout_censor_units)}",
            f"anatomy relative: ({row.anatomy_relative_az_deg:+.1f}, {row.anatomy_relative_el_deg:+.1f})°",
            f"censor best relative: ({row.censor_best_relative_az_deg:+.1f}, {row.censor_best_relative_el_deg:+.1f})°",
            f"gain at anatomy: {row.censor_gain_at_anatomy_vs_cohort_center_nats:+.4f} nats/cell",
        ]), va="top", family="monospace")
    fig.tight_layout()
    fig.savefig(args.output_dir / "Figure_gaze_and_censor_anchor_checkpoint.png", dpi=180)
    plt.close(fig)
    manifest = {
        "sessions": args.sessions,
        "population_sessions": int(population.ecephys_session_id.nunique()),
        "population_units": len(population),
        "heldout_non_improved_v1_units": len(heldout),
        "notes": [
            "Anatomy offsets are LOAO robust medians of improved RF center minus predicted CCF geometry.",
            "Gaze is the absolute Allen filtered spherical screen estimate during the Gabor block.",
            "Censor outcomes use V1 units excluded from the improved-fit geometry offset.",
            "Allen explicitly provides no absolute accuracy guarantee for screen-gaze estimates.",
        ],
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    split_summary = {}
    for axis_name in ("az", "el"):
        a = offsets[f"half_a_{axis_name}_relative_deg"]
        b = offsets[f"half_b_{axis_name}_relative_deg"]
        split_summary[axis_name] = {
            "pearson_r": float(pearsonr(a, b).statistic),
            "spearman_rho": float(spearmanr(a, b).statistic),
            "median_absolute_half_difference_deg": float(np.median(np.abs(a-b))),
        }
    split_summary["two_dimensional"] = {
        "median_half_distance_deg": float(offsets.split_distance_deg.median()),
        "sessions_within_10_deg": int((offsets.split_distance_deg <= 10).sum()),
        "sessions": len(offsets),
    }
    (args.output_dir / "anatomy_offset_split_half_summary.json").write_text(
        json.dumps(split_summary, indent=2)
    )
    print(cases.to_string(index=False))
    print("\nGaze correlations\n", correlations.to_string(index=False))
    print("\nGaze leave-one-out prediction\n", predictions.to_string(index=False))
    print("\nAnatomy offset split-half\n", json.dumps(split_summary, indent=2))
    print(merged[[c for c in merged if c in ["ecephys_session_id", "offset_az_relative_deg", "offset_el_relative_deg", "gaze_x_median_deg", "gaze_y_median_deg"]]].to_string(index=False))


if __name__ == "__main__":
    main()
