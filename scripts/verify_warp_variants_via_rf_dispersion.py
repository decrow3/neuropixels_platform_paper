#!/usr/bin/env python3
"""Verify the warp variants (rotation+offset, anisotropic scale+offset, rotation+scale+offset)
using RF DISPERSION instead of leave-one-probe-out cross-validation.

Motivation: sessions have only 3-6 probes, so leave-one-probe-out CV (used in
`fit_per_session_incremental_warp_cv.py` / `fit_per_session_anisotropic_scale_cv.py`) throws
away a large fraction of a session's already-scarce independent anatomical landmarks just to
validate 1-3 extra parameters. RF dispersion gives an alternative check that costs NO held-out
data, because it is a genuinely different statistic than what the in-sample fit optimizes.

Per `reports/V1_HVA_LGD_RF_DISPERSION_REGISTRATION.md`'s decomposition, for RF r_i = mu(x_i) +
eta_i over a sampled neighborhood:

    Var(r) = Var[mu(x)] + E[Var(r|x)]

-- variance explained by moving along the retinotopic map, plus genuine local scatter. A probe
is a natural "sampled neighborhood": its cells sit at nearly the same CCF position. The in-
sample fits (Huber location for offset, median residual norm for choosing dtheta/dscale) target
the MEAN residual, dominated by between-probe differences (tens of degrees). They say nothing
about whether a candidate's implied local gradient correctly explains WITHIN-probe RF scatter
(a few degrees) -- that is a near-orthogonal piece of information, so checking it doesn't need a
held-out split: every probe gets scored under every candidate, and the comparison is paired.

Metric: for each probe (>=5 cells), residual_i = atlas_predicted_i(candidate params) - offset -
naive_rf_i, then within-probe dispersion = trace of the sample covariance of residual_i around
the PROBE'S OWN mean (i.e. explicitly excludes the probe-level offset/shift -- that part is
already captured by the mean-matching objective; this isolates the leftover scatter). Lower is
better: it means less of the within-probe RF spread is left unexplained by the candidate's local
map.
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
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_multistructure_fixed_effect_translation import huber_location  # noqa: E402
from render_warp_variant_overlays import (  # noqa: E402
    DSCALE_AXIS_GRID, DTHETA_GRID_DEG, build_interpolators, fit_variant, sample,
)

ROOT = Path(__file__).resolve().parents[1]
NAIVE_CELLS = (
    ROOT / "artifacts/v1_absolute_size_dispersion_translation_checkpoint"
    / "naive_v1_centered_cross_session_map" / "naive_v1_centered_cells.csv.gz"
)
GEOMETRY_MANIFEST = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases/translation_rotation_fit_manifest.json"
OUTPUT = ROOT / "artifacts/retinotopy_template/naive_map_registered_to_atlases/per_session_warp_cv"

MIN_CELLS_PER_PROBE = 5


def per_cell_predicted(cells, fits, geometry, az_interp, el_interp):
    """For each session, sample the atlas at that session's own chosen (dtheta, dscale_ap,
    dscale_ml) for all its cells, and attach the fitted offset -- returns arrays aligned to
    `cells`."""
    merged = cells.merge(fits, on="ecephys_session_id", how="left")
    predicted_az = np.full(len(merged), np.nan)
    predicted_el = np.full(len(merged), np.nan)
    for sid, idx in merged.groupby("ecephys_session_id").groups.items():
        group = merged.loc[idx]
        ccf = group[["ccf_ap_mm", "ccf_ml_mm"]].to_numpy(float)
        dtheta = np.radians(group.dtheta_deg.iloc[0])
        dscale_ap = group.dscale_ap.iloc[0]
        dscale_ml = group.dscale_ml.iloc[0]
        predicted = sample(ccf, geometry, dtheta, dscale_ap, dscale_ml, az_interp, el_interp)[0]
        pos = merged.index.get_indexer(idx)
        predicted_az[pos] = predicted[:, 0]
        predicted_el[pos] = predicted[:, 1]
    merged["predicted_azimuth_deg"] = predicted_az
    merged["predicted_elevation_deg"] = predicted_el
    merged["offset_az_session"] = merged["offset_az"]
    merged["offset_el_session"] = merged["offset_el"]
    merged["residual_az"] = merged.predicted_azimuth_deg - merged.offset_az_session - merged.normalized_rf_x
    merged["residual_el"] = merged.predicted_elevation_deg - merged.offset_el_session - merged.normalized_rf_y
    return merged


def within_probe_dispersion(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sid, probe), group in merged.groupby(["ecephys_session_id", "ecephys_probe_id"]):
        residual = group[["residual_az", "residual_el"]].to_numpy(float)
        valid = np.isfinite(residual).all(axis=1)
        residual = residual[valid]
        if len(residual) < MIN_CELLS_PER_PROBE:
            continue
        centered = residual - residual.mean(axis=0)
        dispersion = float(np.mean(np.sum(centered**2, axis=1)))  # trace of sample covariance
        rows.append({"ecephys_session_id": int(sid), "ecephys_probe_id": int(probe),
                     "n_cells": int(valid.sum()), "dispersion_deg2": dispersion})
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    geometry = json.loads(GEOMETRY_MANIFEST.read_text())
    smoothed, az_interp, el_interp = build_interpolators()

    cells = pd.read_csv(NAIVE_CELLS)
    cells["ccf_ap_mm"] = cells.anterior_posterior_ccf_coordinate / 1000.0
    cells["ccf_ml_mm"] = cells.left_right_ccf_coordinate / 1000.0

    variant_defs = (
        ("offset_only", False, False),
        ("rotation_offset", True, False),
        ("scale_offset", False, True),
        ("rotation_scale_offset", True, True),
    )

    dispersion_by_variant = {}
    for key, allow_rotation, allow_scale in variant_defs:
        csv_path = OUTPUT / f"in_sample_fit_{key}.csv"
        if csv_path.exists():
            fits = pd.read_csv(csv_path)
        else:
            fits = fit_variant(cells, geometry, az_interp, el_interp, allow_rotation, allow_scale)
            fits.to_csv(csv_path, index=False)
        merged = per_cell_predicted(cells, fits, geometry, az_interp, el_interp)
        table = within_probe_dispersion(merged)
        table["variant"] = key
        dispersion_by_variant[key] = table
        print(f"{key:22s} probes={len(table):4d}  median_dispersion={table.dispersion_deg2.median():7.2f} deg^2  "
              f"(sqrt={np.sqrt(table.dispersion_deg2.median()):.2f} deg)")

    # paired comparison: same probes appear in every variant, so join on (session, probe)
    baseline = dispersion_by_variant["offset_only"][["ecephys_session_id", "ecephys_probe_id", "dispersion_deg2"]]
    baseline = baseline.rename(columns={"dispersion_deg2": "baseline_dispersion_deg2"})

    print("\npaired comparison vs. offset_only (positive improvement_deg2 = candidate has LESS dispersion, i.e. better):")
    decisions = {}
    for key in ("rotation_offset", "scale_offset", "rotation_scale_offset"):
        candidate = dispersion_by_variant[key][["ecephys_session_id", "ecephys_probe_id", "dispersion_deg2"]]
        candidate = candidate.rename(columns={"dispersion_deg2": "candidate_dispersion_deg2"})
        paired = baseline.merge(candidate, on=["ecephys_session_id", "ecephys_probe_id"])
        paired["improvement_deg2"] = paired.baseline_dispersion_deg2 - paired.candidate_dispersion_deg2
        median_improvement = float(paired.improvement_deg2.median())
        frac_improved = float((paired.improvement_deg2 > 0).mean())
        try:
            _, p_value = wilcoxon(paired.improvement_deg2)
        except ValueError:
            p_value = 1.0
        print(f"  {key:22s} n_probes={len(paired):4d}  median_improvement={median_improvement:+8.2f} deg^2  "
              f"frac_improved={frac_improved:.1%}  wilcoxon_p={p_value:.4g}")
        decisions[key] = {"n_probes": len(paired), "median_improvement_deg2": median_improvement,
                           "fraction_probes_improved": frac_improved, "wilcoxon_p": float(p_value)}
        paired.to_csv(OUTPUT / f"rf_dispersion_paired_{key}_vs_offset_only.csv", index=False)

    summary = {"metric": "within-probe residual dispersion (trace of covariance around probe's own mean), deg^2, lower=better",
               "min_cells_per_probe": MIN_CELLS_PER_PROBE,
               "medians_deg2": {k: float(v.dispersion_deg2.median()) for k, v in dispersion_by_variant.items()},
               "paired_vs_offset_only": decisions}
    (OUTPUT / "rf_dispersion_verification_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUTPUT / 'rf_dispersion_verification_summary.json'}")

    # figure: distribution of within-probe dispersion per variant + paired improvement histograms
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    order = ["offset_only", "rotation_offset", "scale_offset", "rotation_scale_offset"]
    data = [np.sqrt(dispersion_by_variant[k].dispersion_deg2.to_numpy()) for k in order]
    axes[0].boxplot(data, labels=[k.replace("_", "\n") for k in order], showfliers=False)
    axes[0].set(title="Within-probe RF dispersion by variant", ylabel="sqrt(dispersion) (deg)")

    for key, color in zip(("rotation_offset", "scale_offset", "rotation_scale_offset"),
                           ("#2864a8", "#b33f62", "#5f8f3e")):
        paired = pd.read_csv(OUTPUT / f"rf_dispersion_paired_{key}_vs_offset_only.csv")
        axes[1].hist(paired.improvement_deg2, bins=30, histtype="step", linewidth=1.6, color=color,
                     label=f"{key} (median={paired.improvement_deg2.median():+.1f})")
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set(title="Paired improvement vs. offset_only\n(positive = candidate has less dispersion)",
                xlabel="baseline_dispersion - candidate_dispersion (deg^2)", ylabel="probes")
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "Figure_rf_dispersion_verification.png", dpi=170)
    plt.close(fig)
    print(OUTPUT / "Figure_rf_dispersion_verification.png")


if __name__ == "__main__":
    main()
