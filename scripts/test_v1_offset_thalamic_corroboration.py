#!/usr/bin/env python3
"""Test whether simultaneous LGd/LP RFs corroborate V1/screen offsets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts" / "v1_absolute_size_dispersion_translation_checkpoint"
DEFAULT_V1 = CHECKPOINT / "gaze_censor_anchor_checkpoint" / "all_session_anatomy_offsets.csv"
DEFAULT_UNITS = ROOT / "data" / "unit_table.csv"
DEFAULT_OUTPUT = CHECKPOINT / "thalamic_offset_corroboration"
CCF = ("anterior_posterior_ccf_coordinate", "left_right_ccf_coordinate", "dorsal_ventral_ccf_coordinate")
RF = ("azimuth_rf", "elevation_rf")
AREAS = ("LGd", "LP")
CASES = (760345702, 798911424, 771990200)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--unit-table", type=Path, default=DEFAULT_UNITS)
    p.add_argument("--v1-offsets", type=Path, default=DEFAULT_V1)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def features(points, origin, model):
    xyz = (np.asarray(points, float) - origin) / 1000.0
    x, y, z = xyz.T
    if model == "affine":
        return np.column_stack([x, y, z])
    return np.column_stack([x, y, z, x*x, y*y, z*z, x*y, x*z, y*z])


def center_by_session(values, sessions):
    out = np.asarray(values, float).copy()
    for sid in np.unique(sessions):
        selected = sessions == sid
        out[selected] -= np.mean(out[selected], axis=0)
    return out


def fit_geometry(blocks, model="affine", ridge=.1):
    origin = blocks[list(CCF)].to_numpy(float).mean(axis=0)
    sessions = blocks.ecephys_session_id.to_numpy(int)
    x = center_by_session(features(blocks[list(CCF)], origin, model), sessions)
    y = center_by_session(blocks[list(RF)].to_numpy(float), sessions)
    counts = blocks.groupby("ecephys_session_id").ecephys_session_id.transform("size").to_numpy(float)
    base_weight = 1 / counts
    robust = np.ones(len(blocks))
    beta = np.zeros((x.shape[1], 2))
    for _ in range(50):
        weight = base_weight * robust
        updated = np.linalg.solve(
            x.T @ (weight[:, None] * x) + ridge*np.eye(x.shape[1]),
            x.T @ (weight[:, None] * y),
        )
        residual = y - x @ updated
        radius = np.linalg.norm(residual, axis=1)
        scale = 1.4826*np.median(np.abs(radius-np.median(radius))) + 1e-6
        robust_new = np.minimum(1, 1.5*scale/np.maximum(radius, 1e-12))
        if np.max(np.abs(updated-beta)) < 1e-8:
            beta = updated
            break
        beta, robust = updated, robust_new
    return {"origin": origin, "beta": beta, "model": model}


def predict(points, fit):
    return features(points, fit["origin"], fit["model"]) @ fit["beta"]


def physical_blocks(units, count=5):
    frames = []
    for (sid, probe), local in units.groupby(["ecephys_session_id", "ecephys_probe_id"], observed=True):
        local = local.sort_values("probe_vertical_position").copy()
        ranks = local.probe_vertical_position.rank(method="dense").to_numpy()-1
        bins = np.minimum((ranks*count/max(ranks.max()+1, 1)).astype(int), count-1)
        local["physical_block"] = bins
        block = local.groupby("physical_block", as_index=False).agg(
            specimen_id=("specimen_id", "first"), units=("ecephys_unit_id", "size"),
            **{c: (c, "median") for c in (*CCF, *RF)},
        )
        block["ecephys_session_id"] = int(sid)
        block["ecephys_probe_id"] = int(probe)
        frames.append(block)
    return pd.concat(frames, ignore_index=True)


def centered_r2(observed, predicted):
    observed = observed - observed.mean(axis=0)
    predicted = predicted - predicted.mean(axis=0)
    denominator = np.sum(observed**2)
    return np.nan if denominator <= 0 else 1 - np.sum((observed-predicted)**2)/denominator


def estimate_offsets(units, model):
    blocks = physical_blocks(units)
    rows = []
    for sid, local in units.groupby("ecephys_session_id", observed=True):
        sid = int(sid)
        specimen = int(local.specimen_id.iloc[0])
        training = blocks.loc[~blocks.specimen_id.eq(specimen)]
        if training.ecephys_session_id.nunique() < 8:
            continue
        fit = fit_geometry(training, model=model)
        pred = predict(local[list(CCF)], fit)
        residual = local[list(RF)].to_numpy(float) - pred
        offset = np.median(residual, axis=0)
        rng = np.random.default_rng(20260816 + sid)
        order = rng.permutation(len(local))
        a_index, b_index = order[0::2], order[1::2]
        a = np.median(residual[a_index], axis=0) if len(a_index) else np.full(2, np.nan)
        b = np.median(residual[b_index], axis=0) if len(b_index) else np.full(2, np.nan)
        target_blocks = blocks.loc[blocks.ecephys_session_id.eq(sid)]
        block_prediction = predict(target_blocks[list(CCF)], fit)
        rows.append({
            "ecephys_session_id": sid, "specimen_id": specimen, "units": len(local),
            "blocks": len(target_blocks), "offset_az_raw_deg": offset[0],
            "offset_el_raw_deg": offset[1], "half_a_az_raw_deg": a[0],
            "half_a_el_raw_deg": a[1], "half_b_az_raw_deg": b[0],
            "half_b_el_raw_deg": b[1],
            "split_distance_deg": float(np.linalg.norm(a-b)),
            "heldout_centered_r2": centered_r2(target_blocks[list(RF)].to_numpy(float), block_prediction),
        })
    result = pd.DataFrame(rows)
    for axis in ("az", "el"):
        for stem in ("offset", "half_a", "half_b"):
            col = f"{stem}_{axis}_raw_deg"
            result[f"{stem}_{axis}_relative_deg"] = result[col] - result[col].median()
    return result


def paired_stats(v1, thal, label):
    merged = v1.merge(thal, on="ecephys_session_id", suffixes=("_v1", "_thal"))
    rows = []
    for axis in ("az", "el"):
        x = merged[f"offset_{axis}_relative_deg_v1"].to_numpy(float)
        y = merged[f"offset_{axis}_relative_deg_thal"].to_numpy(float)
        pr, sr = pearsonr(x, y), spearmanr(x, y)
        predictions, baselines = [], []
        for i in range(len(x)):
            keep = np.arange(len(x)) != i
            design = np.column_stack([np.ones(keep.sum()), y[keep]])
            beta = np.linalg.lstsq(design, x[keep], rcond=None)[0]
            predictions.append(beta[0] + beta[1]*y[i])
            baselines.append(np.median(x[keep]))
        predictions, baselines = np.asarray(predictions), np.asarray(baselines)
        rows.append({
            "analysis": label, "axis": axis, "sessions": len(x),
            "pearson_r": pr.statistic, "pearson_p": pr.pvalue,
            "spearman_rho": sr.statistic, "spearman_p": sr.pvalue,
            "loo_thalamus_mae_deg": np.mean(np.abs(x-predictions)),
            "loo_constant_mae_deg": np.mean(np.abs(x-baselines)),
            "loo_mae_gain_deg": np.mean(np.abs(x-baselines))-np.mean(np.abs(x-predictions)),
            "loo_r2": 1-np.sum((x-predictions)**2)/np.sum((x-x.mean())**2),
        })
    return merged, rows


def combined_offsets(tables):
    stacked = pd.concat([
        t.assign(area=area) for area, t in tables.items()
    ], ignore_index=True)
    rows = []
    for sid, local in stacked.groupby("ecephys_session_id", observed=True):
        weights = np.sqrt(local.units.to_numpy(float))
        row = {"ecephys_session_id": int(sid), "units": int(local.units.sum()), "nuclei": len(local)}
        for axis in ("az", "el"):
            row[f"offset_{axis}_relative_deg"] = np.average(local[f"offset_{axis}_relative_deg"], weights=weights)
        row["heldout_centered_r2"] = np.average(local.heldout_centered_r2, weights=weights)
        rows.append(row)
    result = pd.DataFrame(rows)
    for axis in ("az", "el"):
        result[f"offset_{axis}_relative_deg"] -= result[f"offset_{axis}_relative_deg"].median()
    return result


def main():
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    usecols = ["ecephys_unit_id", "ecephys_session_id", "specimen_id", "ecephys_probe_id",
               "ecephys_structure_acronym", "probe_vertical_position", "on_screen_rf", "p_value_rf", *CCF, *RF]
    all_units = pd.read_csv(args.unit_table, usecols=usecols, low_memory=False)
    base = (all_units.ecephys_structure_acronym.isin(AREAS) & all_units.on_screen_rf.fillna(False)
            & all_units[[*CCF, *RF, "probe_vertical_position"]].notna().all(axis=1))
    v1 = pd.read_csv(args.v1_offsets)
    all_stats, merged_tables, split_rows = [], {}, []
    model_audit = []
    for selection, selected in [("onscreen", base), ("p01_onscreen", base & all_units.p_value_rf.lt(.01))]:
        chosen_tables = {}
        for area in AREAS:
            local = all_units.loc[selected & all_units.ecephys_structure_acronym.eq(area)].copy()
            candidates = {}
            for model in ("affine", "quadratic"):
                result = estimate_offsets(local, model)
                candidates[model] = result
                model_audit.append({
                    "selection": selection, "area": area, "model": model,
                    "sessions": len(result), "units": len(local),
                    "median_heldout_centered_r2": result.heldout_centered_r2.median(),
                    "positive_r2_sessions": int((result.heldout_centered_r2 > 0).sum()),
                    "median_split_distance_deg": result.split_distance_deg.median(),
                })
            # Choose architecture by median LOAO progression, never by V1 agreement.
            chosen_model = max(candidates, key=lambda m: candidates[m].heldout_centered_r2.median())
            chosen = candidates[chosen_model].assign(model=chosen_model, selection=selection, area=area)
            chosen.to_csv(args.output_dir / f"{selection}_{area}_offsets.csv", index=False)
            chosen_tables[area] = chosen
            for axis in ("az", "el"):
                a = chosen[f"half_a_{axis}_relative_deg"]
                b = chosen[f"half_b_{axis}_relative_deg"]
                valid = a.notna() & b.notna()
                split_rows.append({
                    "selection": selection, "area": area, "axis": axis,
                    "sessions": int(valid.sum()),
                    "pearson_r": pearsonr(a[valid], b[valid]).statistic if valid.sum() >= 3 else np.nan,
                    "spearman_rho": spearmanr(a[valid], b[valid]).statistic if valid.sum() >= 3 else np.nan,
                    "median_absolute_half_difference_deg": np.median(np.abs(a[valid]-b[valid])) if valid.any() else np.nan,
                })
            merged, rows = paired_stats(v1, chosen, f"{selection}_{area}")
            merged.to_csv(args.output_dir / f"{selection}_{area}_v1_comparison.csv", index=False)
            all_stats.extend(rows)
        combined = combined_offsets(chosen_tables)
        combined.to_csv(args.output_dir / f"{selection}_combined_offsets.csv", index=False)
        merged, rows = paired_stats(v1, combined, f"{selection}_combined")
        merged.to_csv(args.output_dir / f"{selection}_combined_v1_comparison.csv", index=False)
        all_stats.extend(rows); merged_tables[selection] = merged
    audit = pd.DataFrame(model_audit); audit.to_csv(args.output_dir / "thalamic_geometry_model_audit.csv", index=False)
    stats = pd.DataFrame(all_stats); stats.to_csv(args.output_dir / "v1_thalamic_corroboration_stats.csv", index=False)
    split = pd.DataFrame(split_rows); split.to_csv(args.output_dir / "thalamic_offset_split_half.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9), sharex="row", sharey="row")
    for row, selection in enumerate(("onscreen", "p01_onscreen")):
        for col, area in enumerate((*AREAS, "combined")):
            path = args.output_dir / f"{selection}_{area}_v1_comparison.csv"
            data = pd.read_csv(path)
            ax = axes[row, col]
            ax.axhline(0, color=".8", lw=1); ax.axvline(0, color=".8", lw=1)
            ax.scatter(data.offset_az_relative_deg_thal, data.offset_az_relative_deg_v1,
                       label="azimuth", alpha=.8)
            ax.scatter(data.offset_el_relative_deg_thal, data.offset_el_relative_deg_v1,
                       marker="s", label="elevation", alpha=.8)
            lim = 32; ax.plot([-lim, lim], [-lim, lim], ls="--", color=".6", lw=1)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
            ax.set_title(f"{selection}: {area} (n={len(data)})")
            ax.set_xlabel("thalamic relative offset (deg)"); ax.set_ylabel("V1 relative offset (deg)")
            if row == 0 and col == 0: ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(args.output_dir / "Figure_v1_thalamic_offset_corroboration.png", dpi=180); plt.close(fig)

    case_rows = []
    for selection, data in merged_tables.items():
        for sid in CASES:
            local = data.loc[data.ecephys_session_id.eq(sid)]
            if len(local):
                case_rows.append({"selection": selection, **local.iloc[0].to_dict()})
    pd.DataFrame(case_rows).to_csv(args.output_dir / "concrete_cases.csv", index=False)
    concrete = pd.DataFrame(case_rows)
    concrete = concrete.loc[concrete.selection.eq("p01_onscreen")]
    fig, axes = plt.subplots(1, len(CASES), figsize=(12, 4))
    for ax, sid in zip(axes, CASES):
        local = concrete.loc[concrete.ecephys_session_id.eq(sid)]
        ax.axhline(0, color=".85", lw=1); ax.axvline(0, color=".85", lw=1)
        if len(local):
            r = local.iloc[0]
            ax.arrow(0, 0, r.offset_az_relative_deg_v1, r.offset_el_relative_deg_v1,
                     width=.25, length_includes_head=True, color="#1f77b4", label="V1")
            ax.arrow(0, 0, r.offset_az_relative_deg_thal, r.offset_el_relative_deg_thal,
                     width=.25, length_includes_head=True, color="#ff7f0e", label="LGd/LP")
            ax.text(.03, .97, f"units={int(r.units)}, nuclei={int(r.nuclei)}\nheld-out R2={r.heldout_centered_r2:+.2f}",
                    transform=ax.transAxes, va="top", fontsize=9)
        ax.set_title(str(sid)); ax.set_xlim(-25, 25); ax.set_ylim(-25, 25); ax.set_aspect("equal")
        ax.set_xlabel("relative azimuth offset (deg)"); ax.set_ylabel("relative elevation offset (deg)")
    axes[0].legend(frameon=False); fig.tight_layout()
    fig.savefig(args.output_dir / "Figure_concrete_thalamic_cases.png", dpi=180); plt.close(fig)
    manifest = {
        "primary_selection": "on_screen_rf == True",
        "sensitivity_selection": "on_screen_rf == True and p_value_rf < 0.01",
        "model_selection": "highest median leave-one-animal-out centered block R2, separately by nucleus and selection",
        "v1_offsets_fixed_before_thalamic_analysis": True,
        "interpretation_limit": "Released thalamic and cortical RFs share stimulus and RF pipeline, but use independent cells and anatomical maps.",
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("MODEL AUDIT\n", audit.to_string(index=False))
    print("\nCORROBORATION\n", stats.to_string(index=False))
    print("\nSPLIT HALF\n", split.to_string(index=False))


if __name__ == "__main__":
    main()
