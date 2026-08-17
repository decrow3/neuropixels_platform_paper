#!/usr/bin/env python3
"""Screen: which functional/tuning properties (NOT RF size, NOT RF dispersion -- both already
tried and rejected as session-translation signals, `reports/V1_HVA_LGD_RF_DISPERSION_REGISTRATION.md`
secs 2-4) show even a weak, reproducible relationship with RF location in V1?

Motivation: per-session translation is still underconstrained by every signal tried so far
(atlas-anchored per-session offset failed CV; RF size failed corroboration; RF dispersion only
worked for 1/5 discovery-set sessions). The user's ask: find some OTHER variable that varies
even weakly with RF location, so it can serve the same role RF size/dispersion were meant to --
an independent axis along which a session-specific surface could be compared to a population
template to identify a translation.

Candidates (literature-motivated where noted):
  - pref_sf_sg, pref_tf_dg, pref_speed_dm: spatial/temporal frequency and speed tuning. Mouse
    upper vs. lower visual field is known to differ ecologically (sky vs. ground), so these are
    the strongest a priori candidates for tracking ELEVATION specifically.
  - g_osi_dg, g_dsi_dg: orientation/direction selectivity strength.
  - c50_dg: contrast sensitivity (half-saturation contrast).
  - time_to_peak_rf, time_to_peak_sg: response latency.
  - run_mod_rf, run_mod_dg: running modulation (weak prior; included as a cheap negative-control-
    ish check since it's not obviously retinotopic).
  - f1_f0_dg: simple/complex classification.
  - lifetime_sparseness_dg, lifetime_sparseness_ns: response sparseness.

Method: fast, honest first-pass screen -- leave-one-SESSION-out linear regression of each
property on (RF azimuth, RF elevation) within V1, evaluated by held-out R^2. Linear-only is a
deliberately low bar (even weak monotonic/linear structure would show up); a null result here
doesn't rule out nonlinear structure, but a positive one is worth pursuing further with the
fuller nested-corroboration design RF size/dispersion received.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def fit_linear(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    design = np.column_stack([X, np.ones(len(X))])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coef


def predict_linear(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    design = np.column_stack([X, np.ones(len(X))])
    return design @ coef

ROOT = Path(__file__).resolve().parents[1]
V1_DESCRIPTORS = ROOT / "artifacts/v1_absolute_size_dispersion_translation_checkpoint/v1_unit_descriptors.csv.gz"
UNIT_TABLE = ROOT / "data/unit_table.csv"
OUTPUT = ROOT / "artifacts/v1_absolute_size_dispersion_translation_checkpoint/functional_property_vs_rf_location_screen"

CANDIDATES = [
    "pref_sf_sg", "pref_tf_dg", "pref_speed_dm", "g_osi_dg", "g_dsi_dg", "c50_dg",
    "time_to_peak_rf", "time_to_peak_sg", "run_mod_rf", "run_mod_dg", "f1_f0_dg",
    "lifetime_sparseness_dg", "lifetime_sparseness_ns",
]
MIN_TRAIN = 100
MIN_HELD = 10
N_SHUFFLES = 500
SHUFFLE_TARGETS = ["f1_f0_dg", "run_mod_rf"]


def held_out_stats(X_all: np.ndarray, y_all: np.ndarray, session_ids: np.ndarray, rng: np.random.Generator | None = None) -> tuple[float, float]:
    if rng is not None:
        y_all = y_all.copy()
        for sid in np.unique(session_ids):
            mask = session_ids == sid
            y_all[mask] = rng.permutation(y_all[mask])
    fold_r2 = []
    for sid in np.unique(session_ids):
        train = session_ids != sid
        held = session_ids == sid
        if train.sum() < MIN_TRAIN or held.sum() < MIN_HELD:
            continue
        coef = fit_linear(X_all[train], y_all[train])
        pred = predict_linear(coef, X_all[held])
        residual = y_all[held] - pred
        baseline_residual = y_all[held] - y_all[train].mean()
        ss_res = float(np.sum(residual ** 2))
        ss_tot = float(np.sum(baseline_residual ** 2))
        if ss_tot > 0:
            fold_r2.append(1.0 - ss_res / ss_tot)
    fold_r2 = np.array(fold_r2)
    return float(np.median(fold_r2)), float((fold_r2 > 0).mean())


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    v1 = pd.read_csv(V1_DESCRIPTORS, usecols=["ecephys_unit_id", "ecephys_session_id", "rf_azimuth_deg", "rf_elevation_deg"])
    units = pd.read_csv(UNIT_TABLE, usecols=["ecephys_unit_id", "quality"] + CANDIDATES, low_memory=False)
    merged = v1.merge(units, on="ecephys_unit_id", how="inner", validate="one_to_one")
    merged = merged.loc[merged.quality.eq("good")] if "quality" in merged else merged
    print(f"V1 units with RF location: {len(v1)}; after merge with unit_table properties: {len(merged)}")

    rows = []
    for prop in CANDIDATES:
        sub = merged.loc[merged[prop].notna() & merged.rf_azimuth_deg.notna() & merged.rf_elevation_deg.notna()].copy()
        if len(sub) < 200:
            rows.append({"property": prop, "n_cells": len(sub), "n_sessions": 0, "median_held_r2": np.nan,
                         "frac_folds_positive": np.nan, "note": "too few cells"})
            continue
        X_all = sub[["rf_azimuth_deg", "rf_elevation_deg"]].to_numpy(float)
        y_all = sub[prop].to_numpy(float)
        session_ids = sub.ecephys_session_id.to_numpy()

        fold_r2 = []
        for sid in np.unique(session_ids):
            train = session_ids != sid
            held = session_ids == sid
            if train.sum() < MIN_TRAIN or held.sum() < MIN_HELD:
                continue
            coef = fit_linear(X_all[train], y_all[train])
            pred = predict_linear(coef, X_all[held])
            residual = y_all[held] - pred
            baseline_residual = y_all[held] - y_all[train].mean()
            ss_res = float(np.sum(residual ** 2))
            ss_tot = float(np.sum(baseline_residual ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            if np.isfinite(r2):
                fold_r2.append(r2)

        fold_r2 = np.array(fold_r2)
        rows.append({
            "property": prop, "n_cells": len(sub), "n_sessions": len(fold_r2),
            "median_held_r2": float(np.median(fold_r2)) if len(fold_r2) else np.nan,
            "frac_folds_positive": float((fold_r2 > 0).mean()) if len(fold_r2) else np.nan,
            "note": "",
        })

    table = pd.DataFrame(rows).sort_values("median_held_r2", ascending=False)
    table.to_csv(OUTPUT / "functional_property_screen_summary.csv", index=False)
    print(table.to_string(index=False))
    print(f"\nwrote {OUTPUT / 'functional_property_screen_summary.csv'}")

    print(f"\n=== shuffle null ({N_SHUFFLES} shuffles, within-session) for top candidates ===")
    rng = np.random.default_rng(20260817)
    null_rows = []
    for prop in SHUFFLE_TARGETS:
        sub = merged.loc[merged[prop].notna() & merged.rf_azimuth_deg.notna() & merged.rf_elevation_deg.notna()].copy()
        X_all = sub[["rf_azimuth_deg", "rf_elevation_deg"]].to_numpy(float)
        y_all = sub[prop].to_numpy(float)
        session_ids = sub.ecephys_session_id.to_numpy()

        observed_r2, observed_frac = held_out_stats(X_all, y_all, session_ids)
        null_r2 = np.empty(N_SHUFFLES)
        null_frac = np.empty(N_SHUFFLES)
        for i in range(N_SHUFFLES):
            null_r2[i], null_frac[i] = held_out_stats(X_all, y_all, session_ids, rng=rng)
        p_r2 = float((null_r2 >= observed_r2).mean())
        p_frac = float((null_frac >= observed_frac).mean())
        print(f"{prop}: observed median_r2={observed_r2:.5f} (null mean={null_r2.mean():.5f}, p={p_r2:.3f}), "
              f"observed frac_positive={observed_frac:.3f} (null mean={null_frac.mean():.3f}, p={p_frac:.3f})")
        null_rows.append({"property": prop, "observed_median_r2": observed_r2, "null_mean_r2": float(null_r2.mean()),
                           "p_r2": p_r2, "observed_frac_positive": observed_frac,
                           "null_mean_frac_positive": float(null_frac.mean()), "p_frac_positive": p_frac})
    pd.DataFrame(null_rows).to_csv(OUTPUT / "shuffle_null_top_candidates.csv", index=False)
    print(f"\nwrote {OUTPUT / 'shuffle_null_top_candidates.csv'}")


if __name__ == "__main__":
    main()
