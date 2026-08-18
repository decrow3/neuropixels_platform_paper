#!/usr/bin/env python3
"""Estimate MouseV2 probe insertion angle from the along-probe DEPTH SPAN of units with a
significant receptive field -- reuses already-computed unit-level RF fits (no raw LFP/CSD signal
processing needed), and crucially gives a real multi-session Allen reference (24 V1 probes with
>=15 significant-RF units) instead of the single locally-cached LFP probe the CSD approach was
stuck with.

Like the CSD responsive-band-thickness approach, this is a RELATIVE (span) measurement, so any
fixed per-probe/per-dataset zero-point offset in probe_vertical_position cancels out -- it does
not require MouseV2 and Allen to share an absolute depth-reference convention, only that a "span
of significant-RF units" measures the same underlying thing (along-probe distance through the
visually responsive part of cortex) in both datasets.

Allen: V1 (VISp), quality=='good', published-like RF significance (p_value_rf<0.01,
on_screen_rf<0.01, area_rf<2500, snr>1, firing_rate_dg>0.1) -- same filter already established
and used earlier in this project (register_allen_session_to_zhuang.py / get_all_RFs.py).
MouseV2: pilot_qc & rf_model_supported (the dataset-wide BH-FDR-significant, split-half-reliable
parametric RF fit criterion already established in mousev2_parametric_rf_v1) PLUS an extra
Allen-analogous reasonableness gate (see REASONABLENESS_GATE below). This matters specifically
for MouseV2 because, unlike Allen, MouseV2 has no CCF/area labels at all (`electrodes/location`
is uniformly "unknown"), so `rf_model_supported` alone cannot exclude a unit for being outside
V1 -- a probe's full ALL-quality-unit depth range (median 2751 um across 32 probes) is ~3.6x
Allen's anatomically-restricted VISp range (median 760 um), because Allen's structure_acronym
filter is a real histological area boundary and MouseV2's is not. `rf_model_supported`'s
significance/reliability filter narrows this gap a lot (to ~1.75x on the RF-significant subset)
but the base thresholds (BH-FDR q<=0.05, pseudo-R2>=0.1, reliability>=0.3, no explicit RF-area
cap) are looser than Allen's published-like gate and were never tuned to specifically exclude the
kind of borderline/oversized RF fit that would spuriously extend a probe's apparent span. The
extra gate below tightens q-value, pseudo-R2, reliability, and adds Allen's explicit
area_rf_deg2 < 2500 cap, to reduce (not eliminate -- MouseV2 still has no anatomical ground
truth) that asymmetry before trusting the resulting span ratio as an insertion-angle estimate.

Span metric: rather than the raw 5th-95th percentile of ALL significant-RF units on a probe
(which lets a few scattered, far-flung units -- plausibly outside true V1, or false-positive RF
fits -- drag the span wide even after the reasonableness gate above), find the LARGEST
CONTIGUOUS run of units along depth (no consecutive gap larger than MAX_CONTIGUOUS_GAP_UM) and
use that run's own span. This is applied identically to Allen and MouseV2 for fairness.
MAX_CONTIGUOUS_GAP_UM is calibrated from Allen's own within-probe consecutive-unit gap
distribution (median 40um, 95th pct 140um across 559 gaps) -- 150um sits just above that 95th
percentile, so it tolerates normal sparse sampling but breaks on the kind of large jump that
signals leaving the responsive column. Picking a run instead of a fixed-size window also lets
probes with fewer TOTAL significant units still qualify, as long as the units they do have are
depth-contiguous rather than scattered -- so MIN_UNITS_PER_PROBE can drop from 15 (needed for a
noisy percentile-based span) to MIN_CONTIGUOUS_UNITS (10), applied to the run, not the raw count.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ALLEN_UNITS = ROOT / "data/unit_table.csv"
RF_FITS = ROOT / "data/imports/mousev2_parametric_rf_v1/rf_unit_fits.csv"
DATA_DIR = ROOT / "data"
OUTPUT = ROOT / "artifacts/figure3/06h_mousev2_csd_insertion_angle"

MIN_UNITS_PER_PROBE = 15
MIN_CONTIGUOUS_UNITS = 10
MAX_CONTIGUOUS_GAP_UM = 150.0
DEPTH_PERCENTILES = (5, 95)

# Allen reference for "ratio=1" (perpendicular insertion): the population MEDIAN span, the
# standard central-tendency estimate of Allen's own (near-vertical) probes.
ALLEN_REFERENCE_PERCENTILE = 50.0

# Angle formula: span = T / cos(angle) for a probe traversing a fixed perpendicular thickness T
# at angle from vertical, so ratio = span/allen_reference = 1/cos(angle) and is >=1 for any real
# angle -- a ratio<1 is not "angle=0", it means the OBSERVED along-probe span was shorter than a
# perpendicular reference, which happens when a probe passes near V1's own lateral/areal boundary
# at a steep angle and exits the responsive column before completing the full vertical thickness,
# not when it is more vertical than Allen's probes. So deviation from ratio=1 in EITHER direction
# implies a nonzero angle -- use effective_ratio = max(ratio, 1/ratio) >= 1 symmetrically, rather
# than clamping ratio<1 to angle=0. A probe with a short span still gets a real (often large)
# angle estimate this way, but its resulting TANGENTIAL length (depth_range * sin(angle), used
# downstream for shank-length regularization) stays small relative to a long-span/large-angle
# probe, because that formula also scales with the probe's own (short) observed depth range.

# Allen-analogous reasonableness gate, applied on top of pilot_qc & rf_model_supported -- see
# module docstring. rf_model_supported already requires q<=0.05/reliability>=0.3/pseudo_r2>=0.1
# and sigma in [3.05, 79.5] deg; this tightens those and adds Allen's explicit area cap.
REASONABLENESS_FDR_Q = 0.01
REASONABLENESS_MIN_RELIABILITY = 0.5
REASONABLENESS_MIN_PSEUDO_R2 = 0.2
REASONABLENESS_MAX_AREA_DEG2 = 2500.0


def robust_span(depths: np.ndarray) -> float:
    lo, hi = np.percentile(depths, DEPTH_PERCENTILES)
    return float(hi - lo)


def largest_contiguous_run(depths: np.ndarray, max_gap: float = MAX_CONTIGUOUS_GAP_UM) -> np.ndarray:
    d = np.sort(depths)
    if len(d) == 0:
        return d
    gaps = np.diff(d)
    breaks = np.nonzero(gaps > max_gap)[0]
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(d) - 1]])
    run_lengths = ends - starts + 1
    best = int(np.argmax(run_lengths))
    return d[starts[best]:ends[best] + 1]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # -- Allen: many V1 probes --
    units = pd.read_csv(ALLEN_UNITS, low_memory=False,
                         usecols=["ecephys_unit_id", "ecephys_session_id", "ecephys_probe_id",
                                  "ecephys_structure_acronym", "probe_vertical_position", "p_value_rf",
                                  "area_rf", "on_screen_rf", "snr", "firing_rate_dg", "quality"])
    v1 = units.loc[units.ecephys_structure_acronym.eq("VISp") & units.quality.eq("good")]
    sig = v1.loc[v1.p_value_rf.notna() & (v1.p_value_rf < 0.01) & (v1.on_screen_rf < 0.01)
                 & (v1.area_rf < 2500) & (v1.snr > 1) & (v1.firing_rate_dg > 0.1)]
    allen_rows = []
    for (session, probe), group in sig.groupby(["ecephys_session_id", "ecephys_probe_id"]):
        depths = group.probe_vertical_position.dropna().to_numpy(float)
        run = largest_contiguous_run(depths)
        if len(run) < MIN_CONTIGUOUS_UNITS:
            continue
        allen_rows.append({"session": session, "probe": probe, "n_units": len(depths),
                            "n_units_in_run": len(run), "depth_span_um": robust_span(run),
                            "median_depth_um": float(np.median(run))})
    allen_table = pd.DataFrame(allen_rows)
    allen_table.to_csv(OUTPUT / "allen_rf_depth_span.csv", index=False)
    print(f"Allen: {len(allen_table)} V1 probes with >= {MIN_CONTIGUOUS_UNITS} units in largest contiguous run "
          f"(gap<={MAX_CONTIGUOUS_GAP_UM:.0f}um)")
    print(allen_table.depth_span_um.describe())
    allen_reference = float(np.percentile(allen_table.depth_span_um, ALLEN_REFERENCE_PERCENTILE))
    print(f"Allen reference ({ALLEN_REFERENCE_PERCENTILE:.0f}th percentile span, ratio=1 baseline): "
          f"{allen_reference:.0f} um (median was {allen_table.depth_span_um.median():.0f}, "
          f"IQR {allen_table.depth_span_um.quantile(.25):.0f}-{allen_table.depth_span_um.quantile(.75):.0f}, "
          f"n={len(allen_table)} probes)")

    # -- MouseV2: all probes --
    rf = pd.read_csv(RF_FITS, low_memory=False)
    base_supported = rf.loc[rf.pilot_qc & rf.rf_model_supported].copy()
    base_supported["area_rf_deg2"] = np.pi * base_supported.rf_sigma_major_deg * base_supported.rf_sigma_minor_deg
    reasonable = (
        base_supported.rf_lrt_q.le(REASONABLENESS_FDR_Q)
        & base_supported.rf_reliability_q.le(REASONABLENESS_FDR_Q)
        & base_supported.rf_split_half_spearman_brown.ge(REASONABLENESS_MIN_RELIABILITY)
        & base_supported.rf_pseudo_r2.ge(REASONABLENESS_MIN_PSEUDO_R2)
        & base_supported.area_rf_deg2.lt(REASONABLENESS_MAX_AREA_DEG2)
    )
    print(f"MouseV2 reasonableness gate: {reasonable.sum()}/{len(base_supported)} pilot_qc & "
          f"rf_model_supported units pass (q<={REASONABLENESS_FDR_Q}, reliability>="
          f"{REASONABLENESS_MIN_RELIABILITY}, pseudo_r2>={REASONABLENESS_MIN_PSEUDO_R2}, "
          f"area<{REASONABLENESS_MAX_AREA_DEG2} deg2)")
    supported = base_supported.loc[reasonable].copy()
    depth_frames = [pd.read_csv(p, usecols=["unit_id", "cortical_depth"])
                     for p in sorted(DATA_DIR.glob("site*_processed/layer_info.csv"))]
    depth_table = pd.concat(depth_frames, ignore_index=True).drop_duplicates("unit_id")
    supported = supported.merge(depth_table, on="unit_id", how="inner")
    print(f"MouseV2: {len(supported)} units with depth after reasonableness gate")

    mousev2_rows = []
    for (site, probe), group in supported.groupby(["site", "probe"]):
        depths = group.cortical_depth.dropna().to_numpy(float)
        run = largest_contiguous_run(depths)
        if len(run) < MIN_CONTIGUOUS_UNITS:
            continue
        span = robust_span(run)
        ratio = span / allen_reference
        effective_ratio = max(ratio, 1.0 / ratio)
        angle_deg = float(np.degrees(np.arccos(np.clip(1.0 / effective_ratio, -1.0, 1.0))))
        mousev2_rows.append({"site": site, "probe": probe, "n_units": len(depths), "n_units_in_run": len(run),
                             "depth_span_um": span, "median_depth_um": float(np.median(run)),
                             "ratio_to_allen": ratio, "estimated_angle_from_vertical_deg": angle_deg})
    mousev2_table = pd.DataFrame(mousev2_rows)
    mousev2_table.to_csv(OUTPUT / "mousev2_rf_depth_span.csv", index=False)
    print(mousev2_table.to_string(index=False))
    print(f"\nMouseV2 median span: {mousev2_table.depth_span_um.median():.0f} um "
          f"(IQR {mousev2_table.depth_span_um.quantile(.25):.0f}-{mousev2_table.depth_span_um.quantile(.75):.0f}, "
          f"n={len(mousev2_table)} probes)")
    print(f"median estimated angle from vertical: {mousev2_table.estimated_angle_from_vertical_deg.median():.1f} deg "
          f"(IQR {mousev2_table.estimated_angle_from_vertical_deg.quantile(.25):.1f}-"
          f"{mousev2_table.estimated_angle_from_vertical_deg.quantile(.75):.1f})")

    # significance check: is MouseV2's span distribution actually different from Allen's?
    from scipy.stats import mannwhitneyu
    stat, p = mannwhitneyu(mousev2_table.depth_span_um, allen_table.depth_span_um, alternative="two-sided")
    print(f"Mann-Whitney U test (MouseV2 vs Allen span): p={p:.4g}")

    manifest = {
        "allen_n_probes": len(allen_table),
        "allen_reference_percentile": ALLEN_REFERENCE_PERCENTILE, "allen_reference_span_um": allen_reference,
        "allen_median_span_um": float(allen_table.depth_span_um.median()),
        "allen_iqr_span_um": [float(allen_table.depth_span_um.quantile(.25)), float(allen_table.depth_span_um.quantile(.75))],
        "mousev2_n_probes": len(mousev2_table), "mousev2_median_span_um": float(mousev2_table.depth_span_um.median()),
        "mousev2_median_angle_deg": float(mousev2_table.estimated_angle_from_vertical_deg.median()),
        "mousev2_n_probes_ratio_below_one": int((mousev2_table.ratio_to_allen < 1.0).sum()),
        "mann_whitney_p": float(p),
        "depth_percentiles_used_for_span": DEPTH_PERCENTILES,
        "min_contiguous_units_per_probe": MIN_CONTIGUOUS_UNITS,
        "max_contiguous_gap_um": MAX_CONTIGUOUS_GAP_UM,
    }
    (OUTPUT / "rf_depth_span_manifest.json").write_text(json.dumps(manifest, indent=2))

    # -- figure --
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    bins = np.linspace(0, max(allen_table.depth_span_um.max(), mousev2_table.depth_span_um.max()), 30)
    ax.hist(allen_table.depth_span_um, bins=bins, alpha=0.5, density=True, label=f"Allen V1 (n={len(allen_table)} probes)", color="#4575b4")
    ax.hist(mousev2_table.depth_span_um, bins=bins, alpha=0.5, density=True, label=f"MouseV2 (n={len(mousev2_table)} probes)", color="#d73027")
    ax.axvline(allen_reference, color="#4575b4", linestyle="--", linewidth=1.2)
    ax.axvline(mousev2_table.depth_span_um.median(), color="#d73027", linestyle="--", linewidth=1.2)
    ax.set(title=f"Significant-RF depth span per probe\nMann-Whitney p={p:.3g}",
           xlabel=f"depth span, {DEPTH_PERCENTILES[0]}-{DEPTH_PERCENTILES[1]} pct (um)", ylabel="density")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.hist(mousev2_table.estimated_angle_from_vertical_deg, bins=15, color="#2864a8", alpha=0.85)
    ax.axvline(mousev2_table.estimated_angle_from_vertical_deg.median(), color="#b33f62", linestyle="--",
               label=f"median={mousev2_table.estimated_angle_from_vertical_deg.median():.1f} deg")
    ax.set(title="Estimated MouseV2 insertion angle from vertical\n(symmetric: ratio<1 also implies a nonzero angle)",
           xlabel="angle (deg)", ylabel="probes")
    ax.legend(fontsize=9)
    fig.suptitle("RF-significant-unit depth span: MouseV2 vs. Allen V1", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT / "Figure_rf_depth_span_mousev2_vs_allen.png", dpi=160)
    plt.close(fig)
    print(f"\n{OUTPUT / 'Figure_rf_depth_span_mousev2_vs_allen.png'}")


if __name__ == "__main__":
    main()
