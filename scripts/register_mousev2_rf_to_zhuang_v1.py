#!/usr/bin/env python3
"""Register MouseV2 probes to the Zhuang V1 compartment using RF VALUES, not anatomy.

MouseV2 NWBs have no CCF/anatomical coordinates (confirmed: `electrodes/location`
is uniformly "unknown" in all 8 sessions), but the recordings are experimentally
known to be within V1. This inverts the direction used everywhere else in this
project: instead of anatomy -> predicted RF value, this finds, for each probe,
the position within Zhuang's V1 compartment whose predicted (azimuth, elevation)
best matches that probe's own observed RF centers -- restricting the search to
V1 removes almost all of the inversion ambiguity a full multi-area inversion
would have (no area-boundary confusion, no folded/reversed gradients).

"Joint translation of RFs on sets recorded in the same session": one shared 2D
RF-value offset per session (the same delta_s role used throughout this whole
project) is fit jointly with the 4 probes' positions via alternation:
  1. given delta, find each probe's best-matching V1 position (nearest predicted
     value to that probe's own robust RF center, offset-corrected);
  2. given positions, refit delta as the Huber location of
     (predicted_at_position - probe_center) pooled across the session's probes;
  3. repeat to convergence.

Validation: the experiment's declared relative probe order (B>C>A>E, roughly
lateral->medial, from `pilot_rf_peaks_v1/rf_probe_ordering.csv`) was NEVER used
in fitting -- checking whether inferred positions respect it is a genuine,
independent sanity check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_multistructure_fixed_effect_translation import huber_location  # noqa: E402
from register_allen_session_to_zhuang import AREA_SEEDS_XY, build_template  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ZHUANG_TEMPLATE = ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/retinotopy_contour_grid.npz"
ZHUANG_SPAN_MATCHED = (
    ROOT / "artifacts/retinotopy_template/zhuang2017_figure9/interpolation_field_sign_qa"
    / "interpolated_fields_and_field_sign_domain_patched_span_matched.npz"
)
GEOMETRY_MANIFEST = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases/translation_rotation_fit_manifest.json"
RF_FITS = ROOT / "data/imports/mousev2_parametric_rf_v1/rf_unit_fits.csv"
PROBE_ORDERING = ROOT / "data/imports/pilot_rf_peaks_v1/rf_probe_ordering.csv"
OUTPUT = ROOT / "artifacts/figure3/06e_mousev2_rf_registered_to_zhuang_v1"

# NOTE: the +50/+10 constants used by `mousev2_frequency_preference_surfaces.py` were chosen
# to match Allen's DECLARED grid range (10..90 azimuth, -30..50 elevation), not validated
# against true retinotopic correspondence with the Zhuang V1 map. Cross-checking against V1's
# own atlas median: azimuth's borrowed +50 empirically checks out (implied offset ~+47.8, close
# to +50), but elevation's +10 does not (implied offset ~-1.5, an ~11.5 deg discrepancy) --
# `common/parametric_models.py::fit_parametric_rf_models` passes raw NWB x_position/y_position
# into the fit with no sign flip, so this is a harmonization-constant issue, not a sign bug in
# the RF extraction itself. Calibrated empirically below by matching medians against V1's own
# atlas distribution (the population-level anchor a dense per-session nearest-match search
# cannot discover on its own -- position and delta trade off almost perfectly against each
# other when candidates are this dense, so delta stays near its initial guess regardless of the
# true miscalibration).
MIN_UNITS_PER_PROBE = 15
MIN_SUPPORTED_PROBES_PER_SESSION = 2
MAX_ITER = 20
PROBE_COLORS = {"A": "#d73027", "B": "#4575b4", "C": "#1a9850", "E": "#8073ac"}
PROBE_ORDER_LETTERS = ["B", "C", "A", "E"]  # declared canonical order


def load_mousev2_units(azimuth_offset: float, elevation_offset: float) -> pd.DataFrame:
    rf = pd.read_csv(RF_FITS, low_memory=False)
    supported = rf.loc[rf.pilot_qc & rf.rf_model_supported].copy()
    supported["azimuth_deg"] = supported.supported_rf_center_x_deg + azimuth_offset
    supported["elevation_deg"] = supported.supported_rf_center_y_deg + elevation_offset
    return supported


def calibrate_harmonization_offsets(visp_azimuth: np.ndarray, visp_elevation: np.ndarray) -> tuple[float, float]:
    """Empirically calibrate the MouseV2 x/y -> azimuth/elevation offsets by matching pooled
    medians against V1's own atlas distribution, rather than trusting a borrowed constant.

    2026-08-18: a multiplicative GAIN on top of this offset was tried and reverted. It was
    calibrated by matching pooled IQR of MouseV2's raw RF values against the IQR of the Zhuang
    V1 MASK's per-pixel value spread (every VISp pixel's azimuth/elevation), and found a large
    azimuth gain (~1.31). That directly contradicts `rescale_zhuang_field_to_naive_span.py`'s
    prior, more careful diagnosis of this exact question -- comparing the Zhuang field against an
    INDEPENDENT densely-pooled multi-session empirical RF map ("naive"), not MouseV2 data -- which
    found azimuth's IQR already matched almost exactly (~0.99) and left it deliberately unscaled;
    only elevation had a real, deliberate gain baked into `..._span_matched.npz` already. The
    gain attempt's comparison was apples-to-oranges: MouseV2's raw spread reflects wherever its
    ~27 probes happened to land (a scattered SAMPLE of V1), not V1's full anatomical retinotopic
    RANGE (the atlas mask's own pixel spread) -- a sampling-coverage gap, not a calibration gap,
    and would appear even with perfect calibration. Trusting the prior, better-controlled
    validation for both axes: no gain, offset only, as below."""
    rf = pd.read_csv(RF_FITS, low_memory=False)
    supported = rf.loc[rf.pilot_qc & rf.rf_model_supported]
    raw_x_median = supported.supported_rf_center_x_deg.median()
    raw_y_median = supported.supported_rf_center_y_deg.median()
    atlas_azimuth_median = float(np.median(visp_azimuth))
    atlas_elevation_median = float(np.median(visp_elevation))
    azimuth_offset = atlas_azimuth_median - raw_x_median
    elevation_offset = atlas_elevation_median - raw_y_median
    print(f"calibration: raw x median={raw_x_median:.2f}, V1 atlas azimuth median={atlas_azimuth_median:.2f} "
          f"-> azimuth_offset={azimuth_offset:+.2f} (borrowed constant was +50.0)")
    print(f"calibration: raw y median={raw_y_median:.2f}, V1 atlas elevation median={atlas_elevation_median:.2f} "
          f"-> elevation_offset={elevation_offset:+.2f} (borrowed constant was +10.0)")
    return azimuth_offset, elevation_offset


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    template = build_template(ZHUANG_TEMPLATE)
    visp_mask = template["area_masks"]["VISp"]
    smoothed = {k: v for k, v in np.load(ZHUANG_SPAN_MATCHED).items()}
    az_field = smoothed["azimuth_span_matched_deg"]
    el_field = smoothed["elevation_span_matched_deg"]

    candidate_rows, candidate_cols = np.nonzero(visp_mask & np.isfinite(az_field) & np.isfinite(el_field))
    candidates = np.column_stack([az_field[candidate_rows, candidate_cols], el_field[candidate_rows, candidate_cols]])
    print(f"V1 candidate positions: {len(candidates)}")

    azimuth_offset, elevation_offset = calibrate_harmonization_offsets(candidates[:, 0], candidates[:, 1])
    units = load_mousev2_units(azimuth_offset, elevation_offset)
    print(f"MouseV2 supported units (pilot_qc & rf_model_supported): {len(units)}")

    ordering = pd.read_csv(PROBE_ORDERING)
    declared_rank = {letter: idx for idx, letter in enumerate(PROBE_ORDER_LETTERS)}

    probe_rows = []
    session_rows = []
    for site, session_units in units.groupby("site"):
        probes = {}
        for probe, group in session_units.groupby("probe"):
            if len(group) < MIN_UNITS_PER_PROBE:
                continue
            rf_center = huber_location(group[["azimuth_deg", "elevation_deg"]].to_numpy(float))
            probes[probe] = {"n_units": len(group), "rf_center": rf_center}
        if len(probes) < MIN_SUPPORTED_PROBES_PER_SESSION:
            print(f"{site}: only {len(probes)} probes with >= {MIN_UNITS_PER_PROBE} units, skipping")
            continue

        delta = np.zeros(2)
        assigned_idx = {probe: None for probe in probes}
        for iteration in range(MAX_ITER):
            new_assigned = {}
            for probe, info in probes.items():
                target = info["rf_center"] + delta  # candidate should predict rf_center + delta
                distances = np.sum((candidates - target) ** 2, axis=1)
                new_assigned[probe] = int(np.argmin(distances))
            converged = new_assigned == assigned_idx
            assigned_idx = new_assigned
            deltas_this_session = np.array([
                candidates[assigned_idx[probe]] - probes[probe]["rf_center"] for probe in probes
            ])
            new_delta = huber_location(deltas_this_session) if len(deltas_this_session) >= 2 else deltas_this_session[0]
            if np.allclose(new_delta, delta, atol=1e-6) and converged:
                delta = new_delta
                break
            delta = new_delta
        else:
            iteration = MAX_ITER

        for probe, info in probes.items():
            idx = assigned_idx[probe]
            predicted = candidates[idx]
            residual = predicted - delta - info["rf_center"]
            probe_rows.append({
                "site": site, "probe": probe, "n_units": info["n_units"],
                "observed_rf_azimuth_deg": info["rf_center"][0], "observed_rf_elevation_deg": info["rf_center"][1],
                "inferred_row": int(candidate_rows[idx]), "inferred_col": int(candidate_cols[idx]),
                "predicted_azimuth_deg": predicted[0], "predicted_elevation_deg": predicted[1],
                "residual_deg": float(np.linalg.norm(residual)),
                "declared_rank": declared_rank.get(probe, np.nan),
            })
        session_rows.append({
            "site": site, "n_probes_supported": len(probes), "iterations": iteration + 1,
            "delta_azimuth_deg": float(delta[0]), "delta_elevation_deg": float(delta[1]),
        })

    probe_table = pd.DataFrame(probe_rows)
    session_table = pd.DataFrame(session_rows)

    # order-agreement diagnostic: correlate declared probe rank against inferred position's
    # projection onto that session's own PC1 axis (never used in fitting). PC1's SIGN is
    # arbitrary (SVD does not canonicalize it), so |rho| is the honest per-session statistic;
    # the SIGN across sessions is reported separately since consistency there is itself
    # informative (random under a true null, not an artifact).
    order_rows = []
    for site, group in probe_table.groupby("site"):
        if len(group) < 3:
            continue
        positions = group[["inferred_row", "inferred_col"]].to_numpy(float)
        centered = positions - positions.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        pc1 = centered @ vt[0]
        rho, p = spearmanr(group.declared_rank, pc1)
        order_rows.append({"site": site, "n_probes": len(group), "spearman_rho": rho,
                            "abs_spearman_rho": abs(rho), "p_value": p})
    order_table = pd.DataFrame(order_rows)

    # stronger, sign-unambiguous validation: does probe LETTER (categorical, never used in
    # fitting beyond grouping) explain more of the pooled inferred-position variance than
    # chance -- omega-squared, matching this project's own established convention
    # (Figure3_stats.md), plus a label-permutation null since n is modest (~8 sessions).
    def omega_squared(values: np.ndarray, groups: np.ndarray) -> float:
        overall_mean = values.mean()
        unique_groups = np.unique(groups)
        k = len(unique_groups)
        ss_total = np.sum((values - overall_mean) ** 2)
        ss_between = sum(
            len(values[groups == g]) * (values[groups == g].mean() - overall_mean) ** 2
            for g in unique_groups
        )
        ss_within = ss_total - ss_between
        n = len(values)
        ms_within = ss_within / (n - k) if n > k else np.nan
        return float((ss_between - (k - 1) * ms_within) / (ss_total + ms_within)) if ss_total + ms_within > 0 else np.nan

    rng = np.random.default_rng(20260817)
    n_shuffles = 2000
    clustering_rows = []
    for axis_name, col in (("row", "inferred_row"), ("col", "inferred_col")):
        values = probe_table[col].to_numpy(float)
        groups = probe_table["probe"].to_numpy()
        observed = omega_squared(values, groups)
        null = np.array([omega_squared(values, rng.permutation(groups)) for _ in range(n_shuffles)])
        p_value = float((null >= observed).mean())
        clustering_rows.append({"axis": axis_name, "omega_squared": observed,
                                 "null_mean_omega_squared": float(np.nanmean(null)), "p_value": p_value})
    clustering_table = pd.DataFrame(clustering_rows)

    probe_table.to_csv(OUTPUT / "mousev2_probe_inferred_v1_position.csv", index=False)
    session_table.to_csv(OUTPUT / "mousev2_session_delta.csv", index=False)
    order_table.to_csv(OUTPUT / "declared_order_agreement.csv", index=False)
    clustering_table.to_csv(OUTPUT / "probe_letter_clustering_omega_squared.csv", index=False)

    print(f"\nsessions registered: {len(session_table)}/{units.site.nunique()}")
    print(f"CAVEAT: median residual (deg) = {probe_table.residual_deg.median():.3f} -- NOT diagnostic of correctness. "
          f"With {len(candidates)} dense V1 candidate positions spanning the full value range, almost any single "
          f"target point finds a near-exact match somewhere in V1 regardless of whether it is the right one.")
    print(f"median |delta| (deg): {np.hypot(session_table.delta_azimuth_deg, session_table.delta_elevation_deg).median():.2f}")
    print("\ndeclared-order agreement (PC1 sign is arbitrary from SVD -- |rho| is the honest per-session stat; "
          "never used in fitting):")
    print(order_table.to_string(index=False))
    print(f"pooled median |rho|: {order_table.abs_spearman_rho.median():.3f}; "
          f"{int((order_table.spearman_rho < 0).sum())}/{len(order_table)} sessions share the same sign "
          f"(consistency itself is informative -- random under a true null)")
    print("\nprobe-LETTER clustering (sign-unambiguous; never used in fitting beyond grouping) "
          f"-- omega-squared with {n_shuffles}-shuffle label-permutation null:")
    print(clustering_table.to_string(index=False))

    manifest = {
        "calibrated_azimuth_offset_deg": azimuth_offset, "calibrated_elevation_offset_deg": elevation_offset,
        "gain_note": "A multiplicative gain on top of this offset was tried (2026-08-18) and reverted -- see "
            "calibrate_harmonization_offsets docstring. Trusting rescale_zhuang_field_to_naive_span.py's prior, "
            "better-controlled validation against an independent empirical RF map: azimuth needs no gain "
            "(~0.99 IQR match), elevation's gain is already baked into azimuth_span_matched_deg/"
            "elevation_span_matched_deg upstream. Offset only, both axes.",
        "borrowed_azimuth_offset_deg_for_reference": 50.0, "borrowed_elevation_offset_deg_for_reference": 10.0,
        "n_candidates": len(candidates), "min_units_per_probe": MIN_UNITS_PER_PROBE,
        "min_supported_probes_per_session": MIN_SUPPORTED_PROBES_PER_SESSION,
        "n_sessions_registered": len(session_table), "n_sessions_total": int(units.site.nunique()),
        "median_residual_deg_NOT_diagnostic": float(probe_table.residual_deg.median()),
        "median_delta_magnitude_deg": float(np.hypot(session_table.delta_azimuth_deg, session_table.delta_elevation_deg).median()),
        "pooled_median_abs_declared_order_spearman_rho": float(order_table.abs_spearman_rho.median()) if len(order_table) else None,
        "declared_order_sign_consistency": f"{int((order_table.spearman_rho < 0).sum())}/{len(order_table)}",
        "probe_letter_clustering_omega_squared": clustering_table.to_dict(orient="records"),
    }
    (OUTPUT / "registration_manifest.json").write_text(json.dumps(manifest, indent=2))

    # figure: inferred positions over the Zhuang V1 compartment
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    ax = axes[0]
    boundary = template["boundary"].astype(float)
    ax.contour(boundary, levels=[0.5], colors="#333333", linewidths=0.55)
    visp_rows, visp_cols = np.nonzero(visp_mask)
    ax.scatter(visp_cols, visp_rows, s=0.3, color="#dddddd", alpha=0.3, rasterized=True, zorder=0)
    for probe, group in probe_table.groupby("probe"):
        ax.scatter(group.inferred_col, group.inferred_row, marker="o", s=70,
                   color=PROBE_COLORS.get(probe, "black"), edgecolors="white", linewidths=0.8,
                   label=f"{probe} (n={len(group)} sessions)", zorder=3)
    for site, group in probe_table.groupby("site"):
        order = group.set_index("probe").reindex([p for p in PROBE_ORDER_LETTERS if p in group.probe.values])
        ax.plot(order.inferred_col, order.inferred_row, color="#999999", linewidth=0.7, alpha=0.6, zorder=2)
    x, y = AREA_SEEDS_XY["VISp"]
    ax.text(x, y, "V1", ha="center", va="center", fontsize=11, color="#555555")
    ax.set(title="MouseV2 probes: RF-inferred position within Zhuang V1\n(lines connect same-session probes in declared B>C>A>E order)",
           xlabel="Zhuang common-map x (px)", ylabel="Zhuang common-map y (px; down+)", aspect="equal")
    ax.legend(fontsize=8)
    height, width = template["domain"].shape
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)

    ax = axes[1]
    ax.hist(order_table.spearman_rho, bins=np.linspace(-1, 1, 11), color="#2864a8", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(order_table.spearman_rho.median(), color="#b33f62", linewidth=1.5, linestyle="--",
               label=f"median={order_table.spearman_rho.median():+.2f}")
    ax.set(title="Declared probe order (B>C>A>E) vs.\ninferred-position PC1 -- never used in fitting",
           xlabel="Spearman rho (per session)", ylabel="sessions")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "Figure_mousev2_rf_registered_to_zhuang_v1.png", dpi=170)
    plt.close(fig)
    print(f"\n{OUTPUT / 'Figure_mousev2_rf_registered_to_zhuang_v1.png'}")


if __name__ == "__main__":
    main()
